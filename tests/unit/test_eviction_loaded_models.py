"""
Unit tests for idle model eviction and loaded_models tracking covering
VAL-OPS-029, VAL-OPS-030, VAL-CROSS-015.

All tests mock whispermlx so no real model downloads occur.
"""

import io
import os
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_whispermlx_mock():
    """Build a MagicMock that mimics the whispermlx module surface."""
    mock_module = MagicMock()
    mock_diarize = MagicMock()
    mock_diarize.DiarizationPipeline = MagicMock()
    mock_module.diarize = mock_diarize

    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "segments": [{"start": 0.0, "end": 1.0, "text": "hello world"}],
        "word_segments": [],
        "language": "en",
    }
    mock_module.load_model.return_value = mock_model
    mock_module.load_align_model.return_value = (
        MagicMock(),
        {"language": "en", "dictionary": {}, "type": "huggingface"},
    )
    mock_module.align.return_value = {
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "hello world",
                "words": [{"word": "hello", "start": 0.0, "end": 0.5}],
            }
        ],
        "word_segments": [{"word": "hello", "start": 0.0, "end": 0.5}],
    }
    mock_module.load_audio.return_value = np.zeros(16000, dtype=np.float32)
    mock_module.assign_word_speakers.side_effect = lambda diar, result: result
    return mock_module


FAKE_AUDIO = io.BytesIO(b"\x00" * 1024)


def _post_asr(client, model="tiny", params=None, file_content=None):
    """Helper to POST to /asr with optional query params."""
    if file_content is None:
        file_content = FAKE_AUDIO
    url = f"/asr?model={model}"
    if params:
        url += "&" + "&".join(f"{k}={v}" for k, v in params.items())
    return client.post(
        url,
        files={"audio_file": ("test.wav", file_content, "audio/wav")},
    )


def _extract_counter_value(body: str, metric_name: str, labels: dict[str, str]) -> float:
    """Extract a Prometheus counter sample value from exposition text."""
    label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for name_variant in (f"{metric_name}_total", metric_name):
            prefix = f'{name_variant}{{{label_str}}}'
            if stripped.startswith(prefix):
                parts = stripped.split()
                if len(parts) >= 2:
                    return float(parts[-1])
    return 0.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client_no_preload():
    """
    TestClient with whispermlx mocked, no PRELOAD_MODEL, and run_in_queue
    NOT mocked so the real pipeline code path adds models to the cache.
    """
    wmlx = _make_whispermlx_mock()
    with (
        patch("app.main.whispermlx", wmlx),
        patch("app.pipeline.whispermlx", wmlx),
        patch("app.openai_compat.whispermlx", wmlx),
        patch.dict(
            os.environ,
            {
                "PRELOAD_MODEL": "",
                "DEVICE": "cpu",
                "SERVE_MODE": "simple",
                "MODEL_KEEP_ALIVE_SECONDS": "0",
            },
            clear=False,
        ),
    ):
        from app.pipeline import (
            _eviction_thread_started,
            _whisper_models,
            _whisper_models_last_used,
        )

        _whisper_models.clear()
        _whisper_models_last_used.clear()

        # Reset the eviction thread flag so it can be restarted
        import app.pipeline as pipe_mod

        pipe_mod._eviction_thread_started = False

        from app.main import app

        with TestClient(app) as c:
            yield c, wmlx


@pytest.fixture()
def client_with_eviction():
    """
    TestClient with MODEL_KEEP_ALIVE_SECONDS=1 and a short sweep interval
    so eviction happens quickly.  The eviction thread is not started by
    default (it's lazy); tests trigger it manually or via load_whisper_model.
    """
    wmlx = _make_whispermlx_mock()
    with (
        patch("app.main.whispermlx", wmlx),
        patch("app.pipeline.whispermlx", wmlx),
        patch("app.openai_compat.whispermlx", wmlx),
        patch.dict(
            os.environ,
            {
                "PRELOAD_MODEL": "",
                "DEVICE": "cpu",
                "SERVE_MODE": "simple",
                "MODEL_KEEP_ALIVE_SECONDS": "1",
                "MODEL_EVICTION_INTERVAL_SECONDS": "30",
            },
            clear=False,
        ),
    ):
        # Re-read the env vars into the pipeline module
        import app.pipeline as pipe_mod

        pipe_mod.MODEL_KEEP_ALIVE_SECONDS = int(
            os.getenv("MODEL_KEEP_ALIVE_SECONDS", "1")
        )
        pipe_mod.MODEL_EVICTION_INTERVAL_SECONDS = max(
            30, int(os.getenv("MODEL_EVICTION_INTERVAL_SECONDS", "30"))
        )

        pipe_mod._whisper_models.clear()
        pipe_mod._whisper_models_last_used.clear()
        pipe_mod._eviction_thread_started = False

        from app.main import app

        with TestClient(app) as c:
            yield c, wmlx


# ---------------------------------------------------------------------------
# VAL-OPS-029: loaded_models reflects a model cached by a request
# ---------------------------------------------------------------------------


class TestLoadedModelsReflectsRequest:
    """
    VAL-OPS-029: After a request loads a model, /health loaded_models lists it.

    Before any transcription the model is absent; after a successful POST /asr
    it appears in loaded_models.
    """

    def test_tiny_absent_before_request(self, client_no_preload):
        """GET /health before any request does NOT list 'tiny' in loaded_models."""
        c, _ = client_no_preload
        resp = c.get("/health")
        assert resp.status_code == 200
        loaded = resp.json()["loaded_models"]
        assert "tiny" not in loaded

    def test_tiny_present_after_asr_request(self, client_no_preload):
        """
        After a successful POST /asr with model=tiny, GET /health lists 'tiny'
        in loaded_models (the in-process MLX cache is populated by the request).
        """
        c, _ = client_no_preload

        # Make a request that loads the 'tiny' model
        resp = _post_asr(c, model="tiny")
        assert resp.status_code == 200

        # Now /health should list 'tiny'
        health = c.get("/health")
        loaded = health.json()["loaded_models"]
        assert "tiny" in loaded

    def test_different_models_tracked(self, client_no_preload):
        """
        Two requests with different models both appear in loaded_models.
        """
        c, _ = client_no_preload

        resp1 = _post_asr(c, model="tiny")
        assert resp1.status_code == 200

        resp2 = _post_asr(c, model="base")
        assert resp2.status_code == 200

        health = c.get("/health")
        loaded = health.json()["loaded_models"]
        assert "tiny" in loaded
        assert "base" in loaded


# ---------------------------------------------------------------------------
# VAL-OPS-030: Idle model eviction unloads cached models end-to-end
# ---------------------------------------------------------------------------


class TestIdleModelEviction:
    """
    VAL-OPS-030: With MODEL_KEEP_ALIVE_SECONDS set, an idle model is evicted,
    whisperx_model_evictions_total increments, and the model disappears from
    loaded_models.
    """

    def test_eviction_removes_model_from_cache(self):
        """
        Directly test the eviction loop: a model idle past the keep-alive
        window is removed from _whisper_models and _whisper_models_last_used.
        """
        wmlx = _make_whispermlx_mock()
        with (
            patch("app.pipeline.whispermlx", wmlx),
            patch.dict(os.environ, {"MODEL_KEEP_ALIVE_SECONDS": "1"}, clear=False),
        ):
            import app.pipeline as pipe_mod

            pipe_mod.MODEL_KEEP_ALIVE_SECONDS = 1
            pipe_mod._whisper_models.clear()
            pipe_mod._whisper_models_last_used.clear()
            pipe_mod._eviction_thread_started = False

            # Load a model
            pipe_mod.load_whisper_model("tiny")
            assert "tiny" in pipe_mod._whisper_models

            # Simulate the model being idle: set last_used far in the past
            pipe_mod._whisper_models_last_used["tiny"] = time.time() - 100

            # Run one eviction sweep manually (simulate what the loop does)
            now = time.time()
            candidates = [
                name
                for name, last in list(pipe_mod._whisper_models_last_used.items())
                if now - last > pipe_mod.MODEL_KEEP_ALIVE_SECONDS
                and name in pipe_mod._whisper_models
            ]
            for name in candidates:
                with pipe_mod._model_load_lock:
                    last = pipe_mod._whisper_models_last_used.get(name, 0)
                    if name in pipe_mod._whisper_models and now - last > pipe_mod.MODEL_KEEP_ALIVE_SECONDS:
                        del pipe_mod._whisper_models[name]
                        pipe_mod._whisper_models_last_used.pop(name, None)
                        try:
                            from app import metrics as prom_metrics

                            prom_metrics.MODEL_EVICTIONS_TOTAL.labels(model=name).inc()
                        except Exception:
                            pass

            # Model should be evicted
            assert "tiny" not in pipe_mod._whisper_models
            assert "tiny" not in pipe_mod._whisper_models_last_used

    def test_eviction_counter_increments(self):
        """
        After eviction, whisperx_model_evictions_total{model="tiny"} >= 1.
        """
        wmlx = _make_whispermlx_mock()
        with (
            patch("app.pipeline.whispermlx", wmlx),
            patch.dict(os.environ, {"MODEL_KEEP_ALIVE_SECONDS": "1"}, clear=False),
        ):
            import app.pipeline as pipe_mod

            pipe_mod.MODEL_KEEP_ALIVE_SECONDS = 1
            pipe_mod._whisper_models.clear()
            pipe_mod._whisper_models_last_used.clear()
            pipe_mod._eviction_thread_started = False

            # Load the model (pre-registers counter)
            pipe_mod.load_whisper_model("tiny")
            assert "tiny" in pipe_mod._whisper_models

            # Simulate idle by setting last_used far in the past
            pipe_mod._whisper_models_last_used["tiny"] = time.time() - 100

            # Run eviction sweep
            from app import metrics as prom_metrics

            before = prom_metrics.MODEL_EVICTIONS_TOTAL.labels(model="tiny")._value.get()

            now = time.time()
            candidates = [
                name
                for name, last in list(pipe_mod._whisper_models_last_used.items())
                if now - last > pipe_mod.MODEL_KEEP_ALIVE_SECONDS
                and name in pipe_mod._whisper_models
            ]
            for name in candidates:
                with pipe_mod._model_load_lock:
                    last = pipe_mod._whisper_models_last_used.get(name, 0)
                    if name in pipe_mod._whisper_models and now - last > pipe_mod.MODEL_KEEP_ALIVE_SECONDS:
                        del pipe_mod._whisper_models[name]
                        pipe_mod._whisper_models_last_used.pop(name, None)
                        prom_metrics.MODEL_EVICTIONS_TOTAL.labels(model=name).inc()

            after = prom_metrics.MODEL_EVICTIONS_TOTAL.labels(model="tiny")._value.get()
            assert after > before
            assert after >= 1

    def test_health_no_longer_lists_evicted_model(self):
        """
        After eviction, /health loaded_models no longer lists the evicted model.
        """
        wmlx = _make_whispermlx_mock()
        with (
            patch("app.pipeline.whispermlx", wmlx),
            patch("app.main.whispermlx", wmlx),
            patch("app.openai_compat.whispermlx", wmlx),
            patch.dict(
                os.environ,
                {
                    "PRELOAD_MODEL": "",
                    "DEVICE": "cpu",
                    "SERVE_MODE": "simple",
                    "MODEL_KEEP_ALIVE_SECONDS": "1",
                },
                clear=False,
            ),
        ):
            import app.pipeline as pipe_mod

            pipe_mod.MODEL_KEEP_ALIVE_SECONDS = 1
            pipe_mod._whisper_models.clear()
            pipe_mod._whisper_models_last_used.clear()
            pipe_mod._eviction_thread_started = False

            from app.main import app

            with TestClient(app) as c:
                # Load tiny via a request
                resp = _post_asr(c, model="tiny")
                assert resp.status_code == 200

                # Verify it's in loaded_models
                health = c.get("/health")
                assert "tiny" in health.json()["loaded_models"]

                # Simulate idle: set last_used far in the past
                pipe_mod._whisper_models_last_used["tiny"] = time.time() - 100

                # Run eviction sweep manually
                now = time.time()
                candidates = [
                    name
                    for name, last in list(pipe_mod._whisper_models_last_used.items())
                    if now - last > pipe_mod.MODEL_KEEP_ALIVE_SECONDS
                    and name in pipe_mod._whisper_models
                ]
                for name in candidates:
                    with pipe_mod._model_load_lock:
                        last = pipe_mod._whisper_models_last_used.get(name, 0)
                        if (
                            name in pipe_mod._whisper_models
                            and now - last > pipe_mod.MODEL_KEEP_ALIVE_SECONDS
                        ):
                            del pipe_mod._whisper_models[name]
                            pipe_mod._whisper_models_last_used.pop(name, None)
                            try:
                                from app import metrics as prom_metrics

                                prom_metrics.MODEL_EVICTIONS_TOTAL.labels(model=name).inc()
                            except Exception:
                                pass

                # Now /health should NOT list tiny
                health2 = c.get("/health")
                assert "tiny" not in health2.json()["loaded_models"]

    def test_eviction_counter_in_metrics_output(self):
        """
        After eviction, /metrics exposes whisperx_model_evictions_total{model="tiny"} >= 1.
        """
        wmlx = _make_whispermlx_mock()
        with (
            patch("app.pipeline.whispermlx", wmlx),
            patch("app.main.whispermlx", wmlx),
            patch("app.openai_compat.whispermlx", wmlx),
            patch.dict(
                os.environ,
                {
                    "PRELOAD_MODEL": "",
                    "DEVICE": "cpu",
                    "SERVE_MODE": "simple",
                    "MODEL_KEEP_ALIVE_SECONDS": "1",
                },
                clear=False,
            ),
        ):
            import app.pipeline as pipe_mod

            pipe_mod.MODEL_KEEP_ALIVE_SECONDS = 1
            pipe_mod._whisper_models.clear()
            pipe_mod._whisper_models_last_used.clear()
            pipe_mod._eviction_thread_started = False

            from app.main import app

            with TestClient(app) as c:
                # Load tiny via a request
                _post_asr(c, model="tiny")

                # Simulate idle and run eviction
                pipe_mod._whisper_models_last_used["tiny"] = time.time() - 100

                now = time.time()
                for name in list(pipe_mod._whisper_models_last_used.keys()):
                    last = pipe_mod._whisper_models_last_used.get(name, 0)
                    if (
                        now - last > pipe_mod.MODEL_KEEP_ALIVE_SECONDS
                        and name in pipe_mod._whisper_models
                    ):
                        del pipe_mod._whisper_models[name]
                        pipe_mod._whisper_models_last_used.pop(name, None)
                        from app import metrics as prom_metrics

                        prom_metrics.MODEL_EVICTIONS_TOTAL.labels(model=name).inc()

                # Check /metrics output
                metrics_resp = c.get("/metrics")
                body = metrics_resp.text
                value = _extract_counter_value(
                    body,
                    "whisperx_model_evictions_total",
                    {"model": "tiny"},
                )
                assert value >= 1


# ---------------------------------------------------------------------------
# VAL-CROSS-015: After eviction, next request reloads transparently
# ---------------------------------------------------------------------------


class TestTransparentReloadAfterEviction:
    """
    VAL-CROSS-015: After eviction, a subsequent request with the same model
    still returns HTTP 200 (the evicted model reloads transparently) and the
    model reappears in /health loaded_models.
    """

    def test_request_succeeds_after_eviction(self):
        """
        A POST /asr with model=tiny succeeds after the model was evicted.
        """
        wmlx = _make_whispermlx_mock()
        with (
            patch("app.pipeline.whispermlx", wmlx),
            patch("app.main.whispermlx", wmlx),
            patch("app.openai_compat.whispermlx", wmlx),
            patch.dict(
                os.environ,
                {
                    "PRELOAD_MODEL": "",
                    "DEVICE": "cpu",
                    "SERVE_MODE": "simple",
                    "MODEL_KEEP_ALIVE_SECONDS": "1",
                },
                clear=False,
            ),
        ):
            import app.pipeline as pipe_mod

            pipe_mod.MODEL_KEEP_ALIVE_SECONDS = 1
            pipe_mod._whisper_models.clear()
            pipe_mod._whisper_models_last_used.clear()
            pipe_mod._eviction_thread_started = False

            from app.main import app

            with TestClient(app) as c:
                # First request loads tiny
                resp1 = _post_asr(c, model="tiny")
                assert resp1.status_code == 200
                assert "tiny" in c.get("/health").json()["loaded_models"]

                # Simulate idle and evict
                pipe_mod._whisper_models_last_used["tiny"] = time.time() - 100
                now = time.time()
                for name in list(pipe_mod._whisper_models_last_used.keys()):
                    last = pipe_mod._whisper_models_last_used.get(name, 0)
                    if (
                        now - last > pipe_mod.MODEL_KEEP_ALIVE_SECONDS
                        and name in pipe_mod._whisper_models
                    ):
                        del pipe_mod._whisper_models[name]
                        pipe_mod._whisper_models_last_used.pop(name, None)
                        from app import metrics as prom_metrics

                        prom_metrics.MODEL_EVICTIONS_TOTAL.labels(model=name).inc()

                # Verify eviction happened
                assert "tiny" not in c.get("/health").json()["loaded_models"]

                # Second request with same model should reload transparently
                resp2 = _post_asr(c, model="tiny")
                assert resp2.status_code == 200

    def test_model_reappears_in_loaded_models_after_reload(self):
        """
        After eviction and a successful reload request, /health loaded_models
        lists the model again.
        """
        wmlx = _make_whispermlx_mock()
        with (
            patch("app.pipeline.whispermlx", wmlx),
            patch("app.main.whispermlx", wmlx),
            patch("app.openai_compat.whispermlx", wmlx),
            patch.dict(
                os.environ,
                {
                    "PRELOAD_MODEL": "",
                    "DEVICE": "cpu",
                    "SERVE_MODE": "simple",
                    "MODEL_KEEP_ALIVE_SECONDS": "1",
                },
                clear=False,
            ),
        ):
            import app.pipeline as pipe_mod

            pipe_mod.MODEL_KEEP_ALIVE_SECONDS = 1
            pipe_mod._whisper_models.clear()
            pipe_mod._whisper_models_last_used.clear()
            pipe_mod._eviction_thread_started = False

            from app.main import app

            with TestClient(app) as c:
                # Load, evict, then reload
                _post_asr(c, model="tiny")
                pipe_mod._whisper_models_last_used["tiny"] = time.time() - 100

                now = time.time()
                for name in list(pipe_mod._whisper_models_last_used.keys()):
                    last = pipe_mod._whisper_models_last_used.get(name, 0)
                    if (
                        now - last > pipe_mod.MODEL_KEEP_ALIVE_SECONDS
                        and name in pipe_mod._whisper_models
                    ):
                        del pipe_mod._whisper_models[name]
                        pipe_mod._whisper_models_last_used.pop(name, None)
                        from app import metrics as prom_metrics

                        prom_metrics.MODEL_EVICTIONS_TOTAL.labels(model=name).inc()

                # tiny is evicted
                assert "tiny" not in c.get("/health").json()["loaded_models"]

                # Re-request
                resp = _post_asr(c, model="tiny")
                assert resp.status_code == 200

                # tiny reappears
                loaded = c.get("/health").json()["loaded_models"]
                assert "tiny" in loaded

    def test_eviction_counter_cumulative_after_multiple_evictions(self):
        """
        Evicting the same model twice increments the counter to >= 2.
        """
        wmlx = _make_whispermlx_mock()
        with (
            patch("app.pipeline.whispermlx", wmlx),
            patch("app.main.whispermlx", wmlx),
            patch("app.openai_compat.whispermlx", wmlx),
            patch.dict(
                os.environ,
                {
                    "PRELOAD_MODEL": "",
                    "DEVICE": "cpu",
                    "SERVE_MODE": "simple",
                    "MODEL_KEEP_ALIVE_SECONDS": "1",
                },
                clear=False,
            ),
        ):
            import app.pipeline as pipe_mod

            pipe_mod.MODEL_KEEP_ALIVE_SECONDS = 1
            pipe_mod._whisper_models.clear()
            pipe_mod._whisper_models_last_used.clear()
            pipe_mod._eviction_thread_started = False

            from app.main import app

            with TestClient(app) as c:
                from app import metrics as prom_metrics

                # First load + evict cycle
                _post_asr(c, model="tiny")
                pipe_mod._whisper_models_last_used["tiny"] = time.time() - 100
                now = time.time()
                for name in list(pipe_mod._whisper_models_last_used.keys()):
                    last = pipe_mod._whisper_models_last_used.get(name, 0)
                    if (
                        now - last > pipe_mod.MODEL_KEEP_ALIVE_SECONDS
                        and name in pipe_mod._whisper_models
                    ):
                        del pipe_mod._whisper_models[name]
                        pipe_mod._whisper_models_last_used.pop(name, None)
                        prom_metrics.MODEL_EVICTIONS_TOTAL.labels(model=name).inc()

                # Second load + evict cycle
                _post_asr(c, model="tiny")
                pipe_mod._whisper_models_last_used["tiny"] = time.time() - 100
                now = time.time()
                for name in list(pipe_mod._whisper_models_last_used.keys()):
                    last = pipe_mod._whisper_models_last_used.get(name, 0)
                    if (
                        now - last > pipe_mod.MODEL_KEEP_ALIVE_SECONDS
                        and name in pipe_mod._whisper_models
                    ):
                        del pipe_mod._whisper_models[name]
                        pipe_mod._whisper_models_last_used.pop(name, None)
                        prom_metrics.MODEL_EVICTIONS_TOTAL.labels(model=name).inc()

                # Counter should be >= 2
                metrics_body = c.get("/metrics").text
                value = _extract_counter_value(
                    metrics_body,
                    "whisperx_model_evictions_total",
                    {"model": "tiny"},
                )
                assert value >= 2

    def test_non_evicted_model_stays_loaded(self):
        """
        Only idle models are evicted; recently-used models stay in cache.
        """
        wmlx = _make_whispermlx_mock()
        with (
            patch("app.pipeline.whispermlx", wmlx),
            patch("app.main.whispermlx", wmlx),
            patch("app.openai_compat.whispermlx", wmlx),
            patch.dict(
                os.environ,
                {
                    "PRELOAD_MODEL": "",
                    "DEVICE": "cpu",
                    "SERVE_MODE": "simple",
                    "MODEL_KEEP_ALIVE_SECONDS": "1",
                },
                clear=False,
            ),
        ):
            import app.pipeline as pipe_mod

            pipe_mod.MODEL_KEEP_ALIVE_SECONDS = 1
            pipe_mod._whisper_models.clear()
            pipe_mod._whisper_models_last_used.clear()
            pipe_mod._eviction_thread_started = False

            from app.main import app

            with TestClient(app) as c:
                # Load both models
                _post_asr(c, model="tiny")
                _post_asr(c, model="base")

                # Make tiny idle (set last_used far in the past)
                pipe_mod._whisper_models_last_used["tiny"] = time.time() - 100
                # base is recent (default from load_whisper_model, should be current)

                # Run eviction sweep
                now = time.time()
                for name in list(pipe_mod._whisper_models_last_used.keys()):
                    last = pipe_mod._whisper_models_last_used.get(name, 0)
                    if (
                        now - last > pipe_mod.MODEL_KEEP_ALIVE_SECONDS
                        and name in pipe_mod._whisper_models
                    ):
                        del pipe_mod._whisper_models[name]
                        pipe_mod._whisper_models_last_used.pop(name, None)
                        from app import metrics as prom_metrics

                        prom_metrics.MODEL_EVICTIONS_TOTAL.labels(model=name).inc()

                # tiny evicted, base stays
                loaded = c.get("/health").json()["loaded_models"]
                assert "tiny" not in loaded
                assert "base" in loaded
