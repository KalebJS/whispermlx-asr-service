"""Unit tests for POST /asr endpoint (app/main.py).

Tests are fast: pipeline is mocked, no model downloads, no GPU required.
Covers: response shapes, parameter handling, error paths, model validation.
"""

from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_AUDIO = b"RIFF" + b"\x00" * 100  # minimal WAV-like bytes


def _mock_pipeline_result(segments=None, language="en", word_segments=None):
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
            }
        ]
    if word_segments is None:
        word_segments = [
            {"word": "Hello", "start": 0.0, "end": 1.2},
            {"word": "world", "start": 1.3, "end": 2.5},
        ]
    return {
        "segments": segments,
        "language": language,
        "word_segments": word_segments,
    }


@pytest.fixture()
def client():
    """Create a TestClient with pipeline functions mocked."""
    with (
        patch("app.main.run_in_queue") as mock_queue,
        patch("app.main.whispermlx") as mock_wmlx,
        patch("app.main.load_whisper_model"),
        patch("app.main.resolve_model_name") as mock_resolve,
        patch("app.main.get_canonical_models") as mock_canonical,
    ):
        # Default: resolve returns the model unchanged (canonical names)
        mock_resolve.side_effect = lambda m: m if m else "large-v3"

        # Default canonical models list
        mock_canonical.return_value = [
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

        # Mock whispermlx.load_audio to return a numpy array
        mock_wmlx.load_audio.return_value = np.zeros(16000, dtype=np.float32)

        # Mock the async queue to return a pipeline result
        async def _fake_run_in_queue(fn, *args, **kwargs):
            # Actually call the function with the mock audio
            return fn(np.zeros(16000, dtype=np.float32), **kwargs)

        # Default: run_in_queue returns a standard result
        mock_queue.side_effect = _fake_run_in_queue

        from app.main import app

        with TestClient(app) as c:
            yield c, mock_queue, mock_resolve, mock_canonical


@pytest.fixture()
def client_with_pipeline_mock():
    """Create a TestClient with run_pipeline fully mocked to return controlled results."""
    with (
        patch("app.main.run_in_queue") as mock_queue,
        patch("app.main.whispermlx") as mock_wmlx,
        patch("app.main.resolve_model_name") as mock_resolve,
        patch("app.main.get_canonical_models") as mock_canonical,
        patch("app.main.run_pipeline"),
    ):
        mock_resolve.side_effect = lambda m: m if m else "large-v3"
        mock_canonical.return_value = [
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
        mock_wmlx.load_audio.return_value = np.zeros(16000, dtype=np.float32)

        from app.main import app

        with TestClient(app) as c:
            yield c, mock_queue, mock_resolve, mock_canonical


# ---------------------------------------------------------------------------
# 1. Response shape tests
# ---------------------------------------------------------------------------


class TestAsrResponseShape:
    """Verify POST /asr returns the documented JSON shape."""

    def test_json_response_has_required_keys(self, client_with_pipeline_mock):
        """JSON response must contain text, language, segments, word_segments."""
        client, mock_queue, _, _ = client_with_pipeline_mock
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client.post(
            "/asr",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        body = resp.json()
        assert "text" in body, "Missing 'text' key"
        assert "language" in body, "Missing 'language' key"
        assert "segments" in body, "Missing 'segments' key"
        assert "word_segments" in body, "Missing 'word_segments' key"

    def test_text_is_array_mirroring_segments(self, client_with_pipeline_mock):
        """text field must be a JSON array mirroring segments (legacy shape)."""
        client, mock_queue, _, _ = client_with_pipeline_mock
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client.post(
            "/asr",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        # text must be an array mirroring segments (legacy whisper-asr-webservice shape)
        assert isinstance(body["text"], list), f"text should be list, got {type(body['text'])}"
        assert len(body["text"]) == len(body["segments"]), (
            f"text length ({len(body['text'])}) != segments length ({len(body['segments'])})"
        )
        for i, (t, s) in enumerate(zip(body["text"], body["segments"], strict=False)):
            assert t["text"] == s["text"], f"text[{i}].text != segments[{i}].text"
            assert t["start"] == s["start"], f"text[{i}].start != segments[{i}].start"
            assert t["end"] == s["end"], f"text[{i}].end != segments[{i}].end"

    def test_language_echoed(self, client_with_pipeline_mock):
        """Detected language must be echoed in the response."""
        client, mock_queue, _, _ = client_with_pipeline_mock
        result = _mock_pipeline_result(language="fr")

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client.post(
            "/asr",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["language"] == "fr", f"Expected language='fr', got '{body['language']}'"

    def test_segments_have_start_end_text(self, client_with_pipeline_mock):
        """Each segment must carry start, end, and text fields."""
        client, mock_queue, _, _ = client_with_pipeline_mock
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client.post(
            "/asr",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["segments"]) > 0, "Segments array is empty"
        for seg in body["segments"]:
            assert "start" in seg, f"Segment missing 'start': {seg}"
            assert "end" in seg, f"Segment missing 'end': {seg}"
            assert "text" in seg, f"Segment missing 'text': {seg}"
            assert isinstance(seg["start"], (int, float)), f"start must be numeric: {seg['start']}"
            assert isinstance(seg["end"], (int, float)), f"end must be numeric: {seg['end']}"
            assert seg["end"] >= seg["start"], f"end < start: {seg}"

    def test_http_200_json_content_type(self, client_with_pipeline_mock):
        """Successful request returns HTTP 200 with application/json content type."""
        client, mock_queue, _, _ = client_with_pipeline_mock
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client.post(
            "/asr",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        assert "application/json" in resp.headers.get("content-type", ""), (
            f"Expected application/json, got {resp.headers.get('content-type')}"
        )


# ---------------------------------------------------------------------------
# 2. word_timestamps parameter
# ---------------------------------------------------------------------------


class TestWordTimestamps:
    """Verify word_timestamps parameter behavior on /asr."""

    def test_word_timestamps_true_yields_word_timestamps(self, client_with_pipeline_mock):
        """word_timestamps=true yields word_segments populated and words in segments."""
        client, mock_queue, _, _ = client_with_pipeline_mock
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client.post(
            "/asr?word_timestamps=true",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["word_segments"]) > 0, "word_segments empty with word_timestamps=true"
        # Segments should have words array
        for seg in body["segments"]:
            if "words" in seg:
                for w in seg["words"]:
                    assert "start" in w, f"Word missing start: {w}"
                    assert "end" in w, f"Word missing end: {w}"

    def test_word_timestamps_false_omits_word_timestamps(self, client_with_pipeline_mock):
        """word_timestamps=false skips alignment; word_segments empty, no words in segments."""
        client, mock_queue, _, _ = client_with_pipeline_mock
        # When alignment is skipped, no word-level data
        result = {
            "segments": [{"start": 0.0, "end": 2.5, "text": "Hello world"}],
            "language": "en",
        }
        # No word_segments key when alignment skipped

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client.post(
            "/asr?word_timestamps=false",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        # word_segments should be empty (result has no word_segments key -> defaults to [])
        assert body["word_segments"] == [], (
            f"word_segments should be empty with word_timestamps=false, got {body['word_segments']}"
        )
        # Segments should NOT have words entries
        for seg in body["segments"]:
            assert "words" not in seg or len(seg.get("words", [])) == 0, (
                f"Segment has words despite word_timestamps=false: {seg}"
            )

    def test_word_timestamps_defaults_to_true(self, client_with_pipeline_mock):
        """Omitting word_timestamps defaults to true (alignment runs)."""
        client, mock_queue, _, _ = client_with_pipeline_mock
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client.post(
            "/asr",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        # Default is true, so word-level timestamps should be present
        assert len(body["word_segments"]) > 0, "word_segments empty when word_timestamps defaulted to true"


# ---------------------------------------------------------------------------
# 3. Model selection
# ---------------------------------------------------------------------------


class TestModelSelection:
    """Verify model parameter handling on /asr."""

    def test_small_model_accepted(self, client_with_pipeline_mock):
        """model=small should be accepted and transcribe successfully."""
        client, mock_queue, mock_resolve, _ = client_with_pipeline_mock
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client.post(
            "/asr?model=small",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

    def test_large_v3_turbo_accepted(self, client_with_pipeline_mock):
        """model=large-v3-turbo (MLX-specific) should be accepted."""
        client, mock_queue, mock_resolve, _ = client_with_pipeline_mock
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client.post(
            "/asr?model=large-v3-turbo",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200

    def test_openai_alias_resolved(self, client_with_pipeline_mock):
        """OpenAI-style alias like whisper-tiny resolves and transcribes."""
        client, mock_queue, mock_resolve, _ = client_with_pipeline_mock
        result = _mock_pipeline_result()

        # resolve_model_name maps whisper-tiny -> tiny
        mock_resolve.side_effect = lambda m: "tiny" if m == "whisper-tiny" else m

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client.post(
            "/asr?model=whisper-tiny",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200

    def test_unknown_model_returns_400(self, client_with_pipeline_mock):
        """Unknown model name returns HTTP 400 with error detail."""
        client, mock_queue, mock_resolve, _ = client_with_pipeline_mock
        # resolve_model_name returns unknown name unchanged
        mock_resolve.side_effect = lambda m: m

        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client.post(
            "/asr?model=not-a-real-model",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 400, f"Expected 400, got {resp.status_code}"
        body = resp.json()
        assert "detail" in body, f"Missing 'detail' in error response: {body}"
        assert "not-a-real-model" in body["detail"], f"Error detail should mention the model name: {body['detail']}"

    def test_unknown_model_service_stays_healthy(self, client_with_pipeline_mock):
        """After an unknown model error, /health still returns 200."""
        client, mock_queue, mock_resolve, _ = client_with_pipeline_mock
        mock_resolve.side_effect = lambda m: m

        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        # First request with unknown model
        resp = client.post(
            "/asr?model=not-a-real-model",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 400

        # Service should still be healthy
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_default_model_used_when_omitted(self, client_with_pipeline_mock):
        """When model param is omitted, the default model is used."""
        client, mock_queue, mock_resolve, _ = client_with_pipeline_mock
        result = _mock_pipeline_result()

        # When model is empty string, resolve returns DEFAULT_MODEL
        mock_resolve.side_effect = lambda m: m if m else "large-v3"

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client.post(
            "/asr",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 4. Language handling
# ---------------------------------------------------------------------------


class TestLanguageHandling:
    """Verify language parameter behavior on /asr."""

    def test_explicit_language_honored(self, client_with_pipeline_mock):
        """language=en should be passed through and echoed in response."""
        client, mock_queue, _, _ = client_with_pipeline_mock
        result = _mock_pipeline_result(language="en")

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client.post(
            "/asr?language=en",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["language"] == "en"

    def test_auto_detect_when_language_omitted(self, client_with_pipeline_mock):
        """When language is omitted, auto-detection populates the language field."""
        client, mock_queue, _, _ = client_with_pipeline_mock
        # Simulate auto-detect returning 'en'
        result = _mock_pipeline_result(language="en")

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client.post(
            "/asr",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["language"] is not None and body["language"] != "", "Language should be populated by auto-detection"

    def test_language_never_empty_on_success(self, client_with_pipeline_mock):
        """On any successful transcription, language is a non-empty string."""
        client, mock_queue, _, _ = client_with_pipeline_mock
        result = _mock_pipeline_result(language="es")

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client.post(
            "/asr",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body["language"], str) and len(body["language"]) > 0, (
            f"language should be non-empty string, got: {body['language']}"
        )


# ---------------------------------------------------------------------------
# 5. initial_prompt and hotwords
# ---------------------------------------------------------------------------


class TestPromptAndHotwords:
    """Verify initial_prompt and hotwords parameter handling on /asr."""

    def test_initial_prompt_accepted(self, client_with_pipeline_mock):
        """initial_prompt param is accepted without error."""
        client, mock_queue, _, _ = client_with_pipeline_mock
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client.post(
            "/asr?initial_prompt=Hello%20world",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200, f"initial_prompt should be accepted, got {resp.status_code}"

    def test_hotwords_accepted_no_error(self, client_with_pipeline_mock):
        """hotwords param is accepted without error (MLX ignores it)."""
        client, mock_queue, _, _ = client_with_pipeline_mock
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client.post(
            "/asr?hotwords=Foo,Bar",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200, f"hotwords should not cause error, got {resp.status_code}"

    def test_hotwords_and_prompt_together(self, client_with_pipeline_mock):
        """Both hotwords and initial_prompt together should succeed."""
        client, mock_queue, _, _ = client_with_pipeline_mock
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client.post(
            "/asr?hotwords=Foo,Bar&initial_prompt=Hello",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200, f"hotwords+initial_prompt should both be accepted, got {resp.status_code}"


# ---------------------------------------------------------------------------
# 6. Task parameter
# ---------------------------------------------------------------------------


class TestTaskParameter:
    """Verify task parameter handling on /asr."""

    def test_task_transcribe_default(self, client_with_pipeline_mock):
        """Default task is transcribe."""
        client, mock_queue, _, _ = client_with_pipeline_mock
        result = _mock_pipeline_result(language="es")

        call_kwargs = {}

        async def _return_result(*args, **kwargs):
            call_kwargs.update(kwargs)
            return result, None

        mock_queue.side_effect = _return_result

        resp = client.post(
            "/asr",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200

    def test_task_translate_accepted(self, client_with_pipeline_mock):
        """task=translate is accepted."""
        client, mock_queue, _, _ = client_with_pipeline_mock
        result = _mock_pipeline_result(language="en")

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client.post(
            "/asr?task=translate",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 7. Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Verify error handling on /asr."""

    def test_missing_audio_file_returns_422(self, client_with_pipeline_mock):
        """Missing audio_file returns HTTP 422 (FastAPI validation)."""
        client, _, _, _ = client_with_pipeline_mock
        resp = client.post("/asr")
        assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"

    def test_invalid_output_format_returns_400(self, client_with_pipeline_mock):
        """Invalid output_format returns HTTP 400."""
        client, mock_queue, _, _ = client_with_pipeline_mock
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client.post(
            "/asr?output_format=docx",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 400, f"Expected 400, got {resp.status_code}"

    def test_short_silent_audio_handled_gracefully(self, client_with_pipeline_mock):
        """Very short/silent audio returns 200 with valid JSON (possibly empty segments)."""
        client, mock_queue, _, _ = client_with_pipeline_mock
        # Simulate a silent audio result (empty segments)
        result = {
            "segments": [],
            "language": "en",
        }

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client.post(
            "/asr",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200, f"Expected 200 for silent audio, got {resp.status_code}"
        body = resp.json()
        assert "segments" in body
        assert "language" in body


# ---------------------------------------------------------------------------
# 8. Segments ordering
# ---------------------------------------------------------------------------


class TestSegmentOrdering:
    """Verify segments are time-ordered."""

    def test_segments_time_ordered(self, client_with_pipeline_mock):
        """Segments should be globally ordered in time (non-decreasing start)."""
        client, mock_queue, _, _ = client_with_pipeline_mock
        result = _mock_pipeline_result(
            segments=[
                {"start": 0.0, "end": 2.0, "text": "First"},
                {"start": 2.0, "end": 4.0, "text": "Second"},
                {"start": 4.0, "end": 6.0, "text": "Third"},
            ],
            word_segments=[],
        )

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client.post(
            "/asr",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        segments = body["segments"]
        for i in range(len(segments) - 1):
            assert segments[i]["start"] <= segments[i + 1]["start"], (
                f"Segment {i} start ({segments[i]['start']}) > segment {i + 1} start ({segments[i + 1]['start']})"
            )


# ---------------------------------------------------------------------------
# 9. Word timestamps within segment bounds
# ---------------------------------------------------------------------------


class TestWordTimestampBounds:
    """Verify word timestamps fall within segment bounds."""

    def test_words_within_segment_bounds(self, client_with_pipeline_mock):
        """Word start/end must fall within parent segment start/end."""
        client, mock_queue, _, _ = client_with_pipeline_mock
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client.post(
            "/asr?word_timestamps=true",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        for seg in body["segments"]:
            if "words" in seg:
                for word in seg["words"]:
                    assert word["start"] >= seg["start"] - 0.01, (
                        f"Word start {word['start']} < segment start {seg['start']}"
                    )
                    assert word["end"] <= seg["end"] + 0.01, f"Word end {word['end']} > segment end {seg['end']}"
