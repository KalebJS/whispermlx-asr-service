"""Unit tests for VAL-OAI-025: FastAPI RequestValidationError on /v1/ paths
returns the OpenAI error envelope instead of the bare {"detail": [...]} shape.

CRITICAL: the native /asr endpoint must KEEP its bare FastAPI 422 detail shape
(VAL-ASR-011, VAL-OPS-014).

Tests cover:
- missing model on /v1/audio/transcriptions -> OpenAI envelope (400)
- missing file on /v1/audio/transcriptions -> OpenAI envelope (400)
- missing model on /v1/audio/translations -> OpenAI envelope (400)
- missing file on /v1/audio/translations -> OpenAI envelope (400)
- missing audio_file on /asr -> still bare 422 {"detail": [...]}
- OpenAI envelope has message, type, param, code fields
"""

from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_AUDIO = b"RIFF" + b"\x00" * 100  # minimal WAV-like bytes

CANONICAL_MODELS = [
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large",
    "large-v1",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
    "turbo",
]


def _make_pipeline_result(
    segments=None,
    language="en",
    word_segments=None,
):
    """Build a realistic pipeline result dict for mocking."""
    if segments is None:
        segments = [
            {
                "start": 0.0,
                "end": 2.5,
                "text": "Hello world",
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 1.2},
                    {"word": "world", "start": 1.3, "end": 2.5},
                ],
            },
        ]
    if word_segments is None:
        word_segments = []
        for seg in segments:
            word_segments.extend(seg.get("words", []))
    return {
        "segments": segments,
        "language": language,
        "word_segments": word_segments,
    }


@pytest.fixture()
def client():
    """
    Create a TestClient with pipeline/queue/whispermlx mocked.
    Returns the TestClient instance.
    """
    with (
        patch("app.openai_compat.run_in_queue") as mock_queue,
        patch("app.openai_compat.whispermlx") as mock_wmlx,
        patch("app.openai_compat.resolve_model_name") as mock_resolve,
        patch("app.openai_compat.get_canonical_models") as mock_canonical,
        patch("app.openai_compat.pipeline_transcribe") as mock_transcribe,
        patch("app.openai_compat.pipeline_align") as mock_align,
        patch("app.openai_compat.DEFAULT_MODEL", "large-v3"),
        patch("app.main.run_in_queue") as mock_main_queue,
        patch("app.main.whispermlx") as mock_main_wmlx,
        patch("app.main.load_whisper_model"),
        patch("app.main.resolve_model_name") as mock_main_resolve,
        patch("app.main.get_canonical_models") as mock_main_canonical,
    ):
        # resolve_model_name: aliases -> canonical, canonical stays, unknown stays
        def _resolve(m):
            if not m:
                return "large-v3"
            if m in CANONICAL_MODELS:
                return m
            aliases = {
                "whisper-1": "large-v3",
                "whisper-large-v3": "large-v3",
                "whisper-large-v2": "large-v2",
                "whisper-medium": "medium",
                "whisper-small": "small",
                "whisper-base": "base",
                "whisper-tiny": "tiny",
            }
            if m in aliases:
                return aliases[m]
            if m.startswith("whisper-"):
                stripped = m[len("whisper-") :]
                if stripped in CANONICAL_MODELS:
                    return stripped
            return m

        mock_resolve.side_effect = _resolve
        mock_canonical.return_value = CANONICAL_MODELS
        mock_main_resolve.side_effect = _resolve
        mock_main_canonical.return_value = CANONICAL_MODELS

        # Mock whispermlx.load_audio to return a numpy array (1 second at 16kHz)
        mock_wmlx.load_audio.return_value = np.zeros(16000, dtype=np.float32)
        mock_main_wmlx.load_audio.return_value = np.zeros(16000, dtype=np.float32)

        # Mock pipeline functions
        result = _make_pipeline_result()
        mock_transcribe.return_value = result
        mock_align.return_value = result

        # Mock run_in_queue to just call the function synchronously
        async def _fake_queue(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        mock_queue.side_effect = _fake_queue
        mock_main_queue.side_effect = _fake_queue

        from app.main import app

        with TestClient(app) as c:
            yield c


# ===================================================================
# VAL-OAI-025: /v1/ validation errors use OpenAI error envelope
# ===================================================================


class TestV1MissingModelTranscriptions:
    """Missing model on /v1/audio/transcriptions returns OpenAI envelope."""

    def test_missing_model_returns_400(self, client):
        """Missing required model field returns 400 (not 422)."""
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 400, f"Expected 400, got {resp.status_code}"

    def test_missing_model_returns_openai_envelope(self, client):
        """Response body uses {error: {message, type, param, code}} shape."""
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        body = resp.json()
        assert "error" in body, f"Missing 'error' key in response: {body}"
        error = body["error"]
        assert "message" in error and isinstance(error["message"], str)
        assert "type" in error and isinstance(error["type"], str)
        assert "param" in error
        assert "code" in error

    def test_missing_model_error_type_is_invalid_request(self, client):
        """error.type is 'invalid_request_error'."""
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        body = resp.json()
        assert body["error"]["type"] == "invalid_request_error"

    def test_missing_model_param_is_model(self, client):
        """error.param points to the missing field name."""
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        body = resp.json()
        assert body["error"]["param"] == "model"

    def test_missing_model_no_fastapi_detail_shape(self, client):
        """Response must NOT use bare FastAPI {"detail": [...]} shape."""
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        body = resp.json()
        assert "detail" not in body, f"Found bare 'detail' in /v1/ response: {body}"

    def test_missing_model_message_mentions_model(self, client):
        """Error message references the missing model field."""
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        body = resp.json()
        assert "model" in body["error"]["message"].lower()


class TestV1MissingFileTranscriptions:
    """Missing file on /v1/audio/transcriptions returns OpenAI envelope."""

    def test_missing_file_returns_400(self, client):
        """Missing required file field returns 400 (not 422)."""
        resp = client.post(
            "/v1/audio/transcriptions",
            data={"model": "whisper-1"},
        )
        assert resp.status_code == 400, f"Expected 400, got {resp.status_code}"

    def test_missing_file_returns_openai_envelope(self, client):
        """Response body uses {error: {message, type, param, code}} shape."""
        resp = client.post(
            "/v1/audio/transcriptions",
            data={"model": "whisper-1"},
        )
        body = resp.json()
        assert "error" in body, f"Missing 'error' key in response: {body}"
        error = body["error"]
        assert "message" in error and isinstance(error["message"], str)
        assert "type" in error and isinstance(error["type"], str)
        assert "param" in error
        assert "code" in error

    def test_missing_file_error_type_is_invalid_request(self, client):
        resp = client.post(
            "/v1/audio/transcriptions",
            data={"model": "whisper-1"},
        )
        body = resp.json()
        assert body["error"]["type"] == "invalid_request_error"

    def test_missing_file_param_is_file(self, client):
        """error.param points to 'file'."""
        resp = client.post(
            "/v1/audio/transcriptions",
            data={"model": "whisper-1"},
        )
        body = resp.json()
        assert body["error"]["param"] == "file"

    def test_missing_file_no_fastapi_detail_shape(self, client):
        resp = client.post(
            "/v1/audio/transcriptions",
            data={"model": "whisper-1"},
        )
        body = resp.json()
        assert "detail" not in body


class TestV1MissingModelTranslations:
    """Missing model on /v1/audio/translations returns OpenAI envelope."""

    def test_missing_model_returns_400(self, client):
        resp = client.post(
            "/v1/audio/translations",
            files={"file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 400, f"Expected 400, got {resp.status_code}"

    def test_missing_model_returns_openai_envelope(self, client):
        resp = client.post(
            "/v1/audio/translations",
            files={"file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        body = resp.json()
        assert "error" in body
        error = body["error"]
        assert "message" in error and isinstance(error["message"], str)
        assert "type" in error and isinstance(error["type"], str)
        assert "param" in error
        assert "code" in error

    def test_missing_model_param_is_model(self, client):
        resp = client.post(
            "/v1/audio/translations",
            files={"file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        body = resp.json()
        assert body["error"]["param"] == "model"

    def test_missing_model_no_fastapi_detail_shape(self, client):
        resp = client.post(
            "/v1/audio/translations",
            files={"file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        body = resp.json()
        assert "detail" not in body


class TestV1MissingFileTranslations:
    """Missing file on /v1/audio/translations returns OpenAI envelope."""

    def test_missing_file_returns_400(self, client):
        resp = client.post(
            "/v1/audio/translations",
            data={"model": "whisper-1"},
        )
        assert resp.status_code == 400, f"Expected 400, got {resp.status_code}"

    def test_missing_file_returns_openai_envelope(self, client):
        resp = client.post(
            "/v1/audio/translations",
            data={"model": "whisper-1"},
        )
        body = resp.json()
        assert "error" in body
        error = body["error"]
        assert "message" in error and isinstance(error["message"], str)
        assert "type" in error and isinstance(error["type"], str)
        assert "param" in error
        assert "code" in error

    def test_missing_file_param_is_file(self, client):
        resp = client.post(
            "/v1/audio/translations",
            data={"model": "whisper-1"},
        )
        body = resp.json()
        assert body["error"]["param"] == "file"

    def test_missing_file_no_fastapi_detail_shape(self, client):
        resp = client.post(
            "/v1/audio/translations",
            data={"model": "whisper-1"},
        )
        body = resp.json()
        assert "detail" not in body


# ===================================================================
# VAL-ASR-011 / VAL-OPS-014: /asr still uses bare FastAPI 422 detail
# ===================================================================


class TestAsrMissingAudioFileStillBare422:
    """Missing audio_file on /asr still returns bare FastAPI 422 {"detail": [...]}."""

    def test_missing_audio_file_returns_422(self, client):
        """Status code is 422 (not 400)."""
        resp = client.post("/asr")
        assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"

    def test_missing_audio_file_has_detail_key(self, client):
        """Response body has 'detail' key (bare FastAPI shape)."""
        resp = client.post("/asr")
        body = resp.json()
        assert "detail" in body, f"Missing 'detail' in /asr 422 response: {body}"

    def test_missing_audio_file_detail_is_list(self, client):
        """detail value is a list of error dicts."""
        resp = client.post("/asr")
        body = resp.json()
        assert isinstance(body["detail"], list), f"'detail' is not a list: {body}"
        assert len(body["detail"]) > 0

    def test_missing_audio_file_no_openai_envelope(self, client):
        """Response must NOT use the OpenAI {error: ...} envelope."""
        resp = client.post("/asr")
        body = resp.json()
        assert "error" not in body, f"Found OpenAI 'error' envelope in /asr response: {body}"

    def test_missing_audio_file_detail_mentions_audio_file(self, client):
        """At least one detail entry references 'audio_file'."""
        resp = client.post("/asr")
        body = resp.json()
        # The detail entries should mention audio_file in the location
        detail_str = str(body["detail"])
        assert "audio_file" in detail_str.lower(), f"Detail does not mention audio_file: {body['detail']}"


# ===================================================================
# Existing application-level 400 OpenAI errors remain unchanged
# ===================================================================


class TestExistingAppLevelErrorsUnchanged:
    """Verify that existing 400 errors from process_audio still use OpenAI envelope."""

    def test_invalid_model_still_returns_400_openai_envelope(self, client):
        """An invalid model still returns 400 with OpenAI envelope (not affected by the handler)."""
        resp = client.post(
            "/v1/audio/transcriptions",
            data={"model": "gpt-4o"},
            files={"file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 400
        body = resp.json()
        assert "error" in body
        assert body["error"]["param"] == "model"
        assert "gpt-4o" in body["error"]["message"]

    def test_temperature_out_of_range_still_returns_400_openai_envelope(self, client):
        """Temperature validation error still uses OpenAI envelope."""
        resp = client.post(
            "/v1/audio/transcriptions",
            data={"model": "whisper-1", "temperature": "2.0"},
            files={"file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 400
        body = resp.json()
        assert "error" in body
        assert body["error"]["param"] == "temperature"
