"""
Unit tests for GET /v1/models and GET /v1/models/{model_id} endpoints.

Covers VAL-MODELS-001 through VAL-MODELS-018:
- OpenAI list envelope shape
- whisper-1 alias presence and owned_by
- All MLX canonical model names present
- large-v3-turbo present
- No distil-* models
- Each entry has id/object/owned_by with correct types
- Each entry has object == "model"
- Works without faster_whisper installed
- No faster_whisper fallback names
- GET /v1/models/{id} returns matching object for valid ids
- GET /v1/models/{id} returns 404 OpenAI error for unknown ids
- GET /v1/models/{id} returns 404 for faster-whisper-only ids
- Model ids are unique
- Dotted ids (tiny.en) resolve correctly
- whisper-1 owned_by "openai"
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Expected model names (from MLX_MODEL_MAP in pipeline.py)
# ---------------------------------------------------------------------------
EXPECTED_MLX_MODELS = [
    "tiny", "tiny.en", "base", "base.en",
    "small", "small.en", "medium", "medium.en",
    "large", "large-v1", "large-v2", "large-v3",
    "large-v3-turbo", "turbo",
]

FASTER_WHISPER_ONLY_MODELS = [
    "distil-large-v2", "distil-medium.en",
    "distil-small.en", "distil-large-v3",
    "distil-large-v3.5",
]


@pytest.fixture()
def client():
    """
    Create a TestClient with minimal mocking.

    The /v1/models endpoint does not invoke the pipeline or load models,
    so we only need to patch enough for app import to succeed.
    """
    with (
        patch("app.openai_compat.whispermlx") as mock_wmlx,
        patch("app.openai_compat.run_in_queue"),
        patch("app.pipeline.whispermlx"),
        patch("app.main.whispermlx") as mock_main_wmlx,
    ):
        mock_main_wmlx.load_audio.return_value = np.zeros(16000, dtype=np.float32)
        mock_wmlx.load_audio.return_value = np.zeros(16000, dtype=np.float32)

        from app.main import app

        with TestClient(app) as c:
            yield c


# ===================================================================
# VAL-MODELS-001: GET /v1/models returns an OpenAI list envelope
# ===================================================================
class TestModelsListEnvelope:
    def test_returns_200(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200

    def test_object_is_list(self, client):
        resp = client.get("/v1/models")
        body = resp.json()
        assert body["object"] == "list"

    def test_data_is_array(self, client):
        resp = client.get("/v1/models")
        body = resp.json()
        assert isinstance(body["data"], list)

    def test_data_non_empty(self, client):
        resp = client.get("/v1/models")
        body = resp.json()
        assert len(body["data"]) >= 1


# ===================================================================
# VAL-MODELS-002: Model list includes the whisper-1 alias
# ===================================================================
class TestWhisper1Alias:
    def test_whisper_1_present(self, client):
        resp = client.get("/v1/models")
        ids = [m["id"] for m in resp.json()["data"]]
        assert "whisper-1" in ids

    def test_whisper_1_exactly_once(self, client):
        resp = client.get("/v1/models")
        ids = [m["id"] for m in resp.json()["data"]]
        assert ids.count("whisper-1") == 1


# ===================================================================
# VAL-MODELS-003: Model list includes all canonical MLX model names
# ===================================================================
class TestAllMLXModelsPresent:
    @pytest.mark.parametrize("model_name", EXPECTED_MLX_MODELS)
    def test_mlx_model_present(self, client, model_name):
        resp = client.get("/v1/models")
        ids = [m["id"] for m in resp.json()["data"]]
        assert model_name in ids, f"{model_name} not found in model list"


# ===================================================================
# VAL-MODELS-004: large-v3-turbo is present
# ===================================================================
class TestLargeV3TurboPresent:
    def test_large_v3_turbo_in_list(self, client):
        resp = client.get("/v1/models")
        ids = [m["id"] for m in resp.json()["data"]]
        assert "large-v3-turbo" in ids


# ===================================================================
# VAL-MODELS-005: distil-* models are NOT present
# ===================================================================
class TestNoDistilModels:
    @pytest.mark.parametrize("distil_name", FASTER_WHISPER_ONLY_MODELS)
    def test_distil_model_absent(self, client, distil_name):
        resp = client.get("/v1/models")
        ids = [m["id"] for m in resp.json()["data"]]
        assert distil_name not in ids, f"{distil_name} should not be in model list"

    def test_no_id_starts_with_distil(self, client):
        resp = client.get("/v1/models")
        ids = [m["id"] for m in resp.json()["data"]]
        distil_ids = [i for i in ids if i.startswith("distil-")]
        assert distil_ids == [], f"Found distil-* ids: {distil_ids}"


# ===================================================================
# VAL-MODELS-006: Each model entry has id, object, and owned_by fields
# ===================================================================
class TestEntryFields:
    def test_every_entry_has_id_object_owned_by(self, client):
        resp = client.get("/v1/models")
        for entry in resp.json()["data"]:
            assert "id" in entry, f"Missing 'id' in {entry}"
            assert "object" in entry, f"Missing 'object' in {entry}"
            assert "owned_by" in entry, f"Missing 'owned_by' in {entry}"

    def test_fields_are_non_empty_strings(self, client):
        resp = client.get("/v1/models")
        for entry in resp.json()["data"]:
            assert isinstance(entry["id"], str) and entry["id"], f"id not a non-empty string: {entry}"
            assert isinstance(entry["object"], str) and entry["object"], f"object not a non-empty string: {entry}"
            assert isinstance(entry["owned_by"], str) and entry["owned_by"], f"owned_by not a non-empty string: {entry}"


# ===================================================================
# VAL-MODELS-007: Each model entry has object == "model"
# ===================================================================
class TestObjectFieldIsModel:
    def test_every_entry_object_is_model(self, client):
        resp = client.get("/v1/models")
        for entry in resp.json()["data"]:
            assert entry["object"] == "model", f"Expected object='model', got '{entry['object']}' for id={entry['id']}"


# ===================================================================
# VAL-MODELS-008: Endpoint works despite faster-whisper being uninstalled
# ===================================================================
class TestNoFasterWhisperDependency:
    def test_no_faster_whisper_import_in_pipeline(self):
        """Verify pipeline.py does not import faster_whisper."""
        import inspect
        from app.pipeline import get_canonical_models

        source = inspect.getsource(get_canonical_models)
        assert "faster_whisper" not in source
        assert "faster-whisper" not in source

    def test_no_faster_whisper_import_in_openai_compat(self):
        """Verify openai_compat.py does not import faster_whisper."""
        import inspect
        from app.openai_compat import _build_available_models

        source = inspect.getsource(_build_available_models)
        assert "faster_whisper" not in source
        assert "faster-whisper" not in source

    def test_endpoint_returns_populated_list(self, client):
        """Verify the endpoint returns a populated list (not empty/errored)."""
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()["data"]
        # 14 MLX names + whisper-1 = 15
        assert len(data) >= 15


# ===================================================================
# VAL-MODELS-009: No faster_whisper fallback names
# ===================================================================
class TestNoFasterWhisperNames:
    @pytest.mark.parametrize("fw_name", FASTER_WHISPER_ONLY_MODELS)
    def test_faster_whisper_only_name_absent(self, client, fw_name):
        resp = client.get("/v1/models")
        ids = [m["id"] for m in resp.json()["data"]]
        assert fw_name not in ids


# ===================================================================
# VAL-MODELS-010: GET /v1/models/large-v3 returns matching object
# ===================================================================
class TestGetModelLargeV3:
    def test_returns_200(self, client):
        resp = client.get("/v1/models/large-v3")
        assert resp.status_code == 200

    def test_id_matches(self, client):
        resp = client.get("/v1/models/large-v3")
        body = resp.json()
        assert body["id"] == "large-v3"

    def test_object_is_model(self, client):
        resp = client.get("/v1/models/large-v3")
        assert resp.json()["object"] == "model"

    def test_has_owned_by(self, client):
        resp = client.get("/v1/models/large-v3")
        assert "owned_by" in resp.json()


# ===================================================================
# VAL-MODELS-011: GET /v1/models/whisper-1 returns matching object
# ===================================================================
class TestGetModelWhisper1:
    def test_returns_200(self, client):
        resp = client.get("/v1/models/whisper-1")
        assert resp.status_code == 200

    def test_id_is_whisper_1(self, client):
        resp = client.get("/v1/models/whisper-1")
        assert resp.json()["id"] == "whisper-1"

    def test_object_is_model(self, client):
        resp = client.get("/v1/models/whisper-1")
        assert resp.json()["object"] == "model"


# ===================================================================
# VAL-MODELS-012: GET /v1/models/large-v3-turbo resolves
# ===================================================================
class TestGetModelLargeV3Turbo:
    def test_returns_200(self, client):
        resp = client.get("/v1/models/large-v3-turbo")
        assert resp.status_code == 200

    def test_id_matches(self, client):
        resp = client.get("/v1/models/large-v3-turbo")
        assert resp.json()["id"] == "large-v3-turbo"


# ===================================================================
# VAL-MODELS-013: GET /v1/models/{id} returns 404 for unknown id
# ===================================================================
class TestGetModelUnknown404:
    def test_unknown_model_returns_404(self, client):
        resp = client.get("/v1/models/does-not-exist")
        assert resp.status_code == 404

    def test_gibberish_model_returns_404(self, client):
        resp = client.get("/v1/models/not-a-real-model-at-all")
        assert resp.status_code == 404


# ===================================================================
# VAL-MODELS-014: 404 body is an OpenAI-shaped error
# ===================================================================
class TestGetModel404OpenAIError:
    def test_error_envelope_shape(self, client):
        resp = client.get("/v1/models/does-not-exist")
        body = resp.json()
        assert "error" in body
        error = body["error"]
        assert "message" in error
        assert "type" in error
        assert isinstance(error["message"], str)

    def test_error_code_is_model_not_found(self, client):
        resp = client.get("/v1/models/does-not-exist")
        body = resp.json()
        assert body["error"]["code"] == "model_not_found"

    def test_error_type_is_invalid_request_error(self, client):
        resp = client.get("/v1/models/does-not-exist")
        body = resp.json()
        assert body["error"]["type"] == "invalid_request_error"

    def test_error_has_param_field(self, client):
        resp = client.get("/v1/models/does-not-exist")
        body = resp.json()
        # param may be None but the key must exist
        assert "param" in body["error"]


# ===================================================================
# VAL-MODELS-015: distil-large-v3 returns 404
# ===================================================================
class TestDistilModelReturns404:
    def test_distil_large_v3_404(self, client):
        resp = client.get("/v1/models/distil-large-v3")
        assert resp.status_code == 404

    def test_distil_large_v3_error_code(self, client):
        resp = client.get("/v1/models/distil-large-v3")
        body = resp.json()
        assert body["error"]["code"] == "model_not_found"

    @pytest.mark.parametrize("distil_name", FASTER_WHISPER_ONLY_MODELS)
    def test_distil_model_404(self, client, distil_name):
        resp = client.get(f"/v1/models/{distil_name}")
        assert resp.status_code == 404


# ===================================================================
# VAL-MODELS-016: Model ids are unique (no duplicates)
# ===================================================================
class TestModelIdsUnique:
    def test_all_ids_unique(self, client):
        resp = client.get("/v1/models")
        ids = [m["id"] for m in resp.json()["data"]]
        assert len(set(ids)) == len(ids), f"Duplicate ids found: {[i for i in ids if ids.count(i) > 1]}"


# ===================================================================
# VAL-MODELS-017: Dotted ids (e.g. tiny.en) resolve
# ===================================================================
class TestDottedIds:
    def test_tiny_en_returns_200(self, client):
        resp = client.get("/v1/models/tiny.en")
        assert resp.status_code == 200

    def test_tiny_en_id_matches(self, client):
        resp = client.get("/v1/models/tiny.en")
        assert resp.json()["id"] == "tiny.en"

    def test_tiny_en_object_is_model(self, client):
        resp = client.get("/v1/models/tiny.en")
        assert resp.json()["object"] == "model"

    @pytest.mark.parametrize("dotted_name", ["tiny.en", "base.en", "small.en", "medium.en"])
    def test_all_dotted_en_ids_resolve(self, client, dotted_name):
        resp = client.get(f"/v1/models/{dotted_name}")
        assert resp.status_code == 200
        assert resp.json()["id"] == dotted_name


# ===================================================================
# VAL-MODELS-018: whisper-1 entry is owned_by "openai"
# ===================================================================
class TestWhisper1OwnedByOpenAI:
    def test_whisper_1_owned_by_openai(self, client):
        resp = client.get("/v1/models")
        for entry in resp.json()["data"]:
            if entry["id"] == "whisper-1":
                assert entry["owned_by"] == "openai"
                return
        pytest.fail("whisper-1 not found in model list")

    def test_mlx_models_not_owned_by_openai(self, client):
        """Canonical MLX model entries should not be owned by openai."""
        resp = client.get("/v1/models")
        for entry in resp.json()["data"]:
            if entry["id"] != "whisper-1":
                assert entry["owned_by"] != "openai", f"Non-whisper-1 model {entry['id']} is owned_by 'openai'"

    def test_mlx_models_owned_by_whispermlx(self, client):
        """Canonical MLX model entries should be owned_by 'whispermlx'."""
        resp = client.get("/v1/models")
        for entry in resp.json()["data"]:
            if entry["id"] != "whisper-1":
                assert entry["owned_by"] == "whispermlx", f"Expected owned_by='whispermlx', got '{entry['owned_by']}' for {entry['id']}"
