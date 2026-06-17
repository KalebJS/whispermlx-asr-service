"""Unit tests for POST /v1/audio/transcriptions (app/openai_compat.py).

Tests are fast: pipeline is mocked, no model downloads, no GPU required.
Covers: response formats, model resolution, temperature validation,
timestamp_granularities, error envelopes, prompt/hotwords handling.
"""

import io
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_AUDIO = b"RIFF" + b"\x00" * 100  # minimal WAV-like bytes


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
            {
                "start": 2.5,
                "end": 5.0,
                "text": "this is a test",
                "words": [
                    {"word": "this", "start": 2.5, "end": 3.0},
                    {"word": "is", "start": 3.1, "end": 3.4},
                    {"word": "a", "start": 3.5, "end": 3.6},
                    {"word": "test", "start": 3.7, "end": 5.0},
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


# Canonical MLX model list used by resolve_model_name
CANONICAL_MODELS = [
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large", "large-v1", "large-v2",
    "large-v3", "large-v3-turbo", "turbo",
]


@pytest.fixture()
def client():
    """
    Create a TestClient with pipeline/queue/whispermlx mocked.

    Returns (client, mock_queue) so tests can inspect queue call args.
    """
    with (
        patch("app.openai_compat.run_in_queue") as mock_queue,
        patch("app.openai_compat.whispermlx") as mock_wmlx,
        patch("app.openai_compat.resolve_model_name") as mock_resolve,
        patch("app.openai_compat.get_canonical_models") as mock_canonical,
        patch("app.openai_compat.pipeline_transcribe") as mock_transcribe,
        patch("app.openai_compat.pipeline_align") as mock_align,
        patch("app.openai_compat.DEFAULT_MODEL", "large-v3"),
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
                stripped = m[len("whisper-"):]
                if stripped in CANONICAL_MODELS:
                    return stripped
            return m

        mock_resolve.side_effect = _resolve
        mock_canonical.return_value = CANONICAL_MODELS

        # Mock whispermlx.load_audio to return a numpy array (1 second at 16kHz)
        mock_wmlx.load_audio.return_value = np.zeros(16000, dtype=np.float32)

        # Mock pipeline functions
        result = _make_pipeline_result()
        mock_transcribe.return_value = result
        mock_align.return_value = result

        # Mock run_in_queue to just call the function synchronously
        async def _fake_queue(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        mock_queue.side_effect = _fake_queue

        from app.main import app

        with TestClient(app) as c:
            yield c, mock_queue, mock_resolve


def _post_transcriptions(client, data: dict, files=None):
    """Helper to POST to /v1/audio/transcriptions."""
    if files is None:
        files = {"file": ("test.wav", FAKE_AUDIO, "audio/wav")}
    return client.post("/v1/audio/transcriptions", data=data, files=files)


# ===================================================================
# VAL-OAI-001: default response_format returns JSON {text}
# ===================================================================

class TestDefaultJsonFormat:
    """POST /v1/audio/transcriptions with no response_format -> {text}."""

    def test_default_returns_json_with_text_key_only(self, client):
        c, _, _ = client
        resp = _post_transcriptions(c, {"model": "whisper-1"})
        assert resp.status_code == 200
        body = resp.json()
        assert "text" in body
        assert body["text"] == "Hello world this is a test"
        # Only key should be "text" per VAL-OAI-001
        assert list(body.keys()) == ["text"]

    def test_default_content_type_is_application_json(self, client):
        c, _, _ = client
        resp = _post_transcriptions(c, {"model": "whisper-1"})
        assert resp.status_code == 200
        ct = resp.headers.get("content-type", "")
        assert "application/json" in ct


# ===================================================================
# VAL-OAI-002: explicit response_format=json returns {text}
# ===================================================================

class TestExplicitJsonFormat:

    def test_explicit_json_returns_text_only(self, client):
        c, _, _ = client
        resp = _post_transcriptions(c, {"model": "whisper-1", "response_format": "json"})
        assert resp.status_code == 200
        body = resp.json()
        assert "text" in body
        assert body["text"] == "Hello world this is a test"
        # No segments/words/language fields
        assert "segments" not in body
        assert "words" not in body
        assert "language" not in body


# ===================================================================
# VAL-OAI-003: response_format=text returns plain text
# ===================================================================

class TestTextFormat:

    def test_text_format_returns_plain_text(self, client):
        c, _, _ = client
        resp = _post_transcriptions(c, {"model": "whisper-1", "response_format": "text"})
        assert resp.status_code == 200
        assert resp.text == "Hello world this is a test"
        ct = resp.headers.get("content-type", "")
        assert "text/plain" in ct

    def test_text_format_not_json(self, client):
        c, _, _ = client
        resp = _post_transcriptions(c, {"model": "whisper-1", "response_format": "text"})
        # Should NOT be parseable as JSON (no surrounding quotes/braces)
        with pytest.raises(json.JSONDecodeError):
            resp.json()


# ===================================================================
# VAL-OAI-004 + VAL-OAI-035: srt format + Content-Type text/plain
# ===================================================================

class TestSrtFormat:

    def test_srt_returns_valid_cues(self, client):
        c, _, _ = client
        resp = _post_transcriptions(c, {"model": "whisper-1", "response_format": "srt"})
        assert resp.status_code == 200
        body = resp.text
        # SRT cue pattern: index, timecode, text
        assert "1\n" in body
        assert "-->" in body
        # Comma decimal separator in SRT timecodes
        import re
        timecode_pattern = r"\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}"
        assert re.search(timecode_pattern, body), f"No SRT timecode found in: {body!r}"

    def test_srt_content_type_is_text_plain(self, client):
        c, _, _ = client
        resp = _post_transcriptions(c, {"model": "whisper-1", "response_format": "srt"})
        assert resp.status_code == 200
        ct = resp.headers.get("content-type", "")
        assert ct.startswith("text/plain")


# ===================================================================
# VAL-OAI-005: vtt format returns WEBVTT
# ===================================================================

class TestVttFormat:

    def test_vtt_returns_webvtt_header(self, client):
        c, _, _ = client
        resp = _post_transcriptions(c, {"model": "whisper-1", "response_format": "vtt"})
        assert resp.status_code == 200
        body = resp.text
        assert body.startswith("WEBVTT")
        # Period decimal separator in VTT timecodes
        import re
        timecode_pattern = r"\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}"
        assert re.search(timecode_pattern, body), f"No VTT timecode found in: {body!r}"

    def test_vtt_content_type(self, client):
        c, _, _ = client
        resp = _post_transcriptions(c, {"model": "whisper-1", "response_format": "vtt"})
        ct = resp.headers.get("content-type", "")
        assert "text/vtt" in ct


# ===================================================================
# VAL-OAI-006 + VAL-OAI-007 + VAL-OAI-008: verbose_json
# ===================================================================

class TestVerboseJsonFormat:

    def test_verbose_json_returns_full_object(self, client):
        c, _, _ = client
        resp = _post_transcriptions(c, {"model": "whisper-1", "response_format": "verbose_json"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["task"] == "transcribe"
        assert "language" in body
        assert "duration" in body
        assert body["duration"] > 0
        assert "text" in body
        assert "segments" in body
        assert len(body["segments"]) > 0

    def test_verbose_json_segment_fields(self, client):
        """VAL-OAI-007 + VAL-OAI-034: segments have full OpenAI field set."""
        c, _, _ = client
        resp = _post_transcriptions(c, {"model": "whisper-1", "response_format": "verbose_json"})
        assert resp.status_code == 200
        body = resp.json()
        for seg in body["segments"]:
            assert "id" in seg and isinstance(seg["id"], int)
            assert "seek" in seg and isinstance(seg["seek"], int)
            assert "start" in seg and isinstance(seg["start"], (int, float))
            assert "end" in seg and isinstance(seg["end"], (int, float))
            assert seg["end"] >= seg["start"]
            assert "text" in seg and isinstance(seg["text"], str)
            # Full OpenAI segment field set (VAL-OAI-034)
            assert "tokens" in seg and isinstance(seg["tokens"], list)
            assert "temperature" in seg and isinstance(seg["temperature"], (int, float))
            assert "avg_logprob" in seg and isinstance(seg["avg_logprob"], (int, float))
            assert "compression_ratio" in seg and isinstance(seg["compression_ratio"], (int, float))
            assert "no_speech_prob" in seg and isinstance(seg["no_speech_prob"], (int, float))

    def test_verbose_json_default_has_segments_no_words(self, client):
        """VAL-OAI-008: default verbose_json has segments but no words."""
        c, _, _ = client
        resp = _post_transcriptions(c, {"model": "whisper-1", "response_format": "verbose_json"})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["segments"]) > 0
        assert body.get("words") is None


# ===================================================================
# VAL-OAI-009: verbose_json + timestamp_granularities[]=segment
# ===================================================================

class TestTimestampGranularitiesSegment:

    def test_segment_granularity_has_segments(self, client):
        c, _, _ = client
        resp = _post_transcriptions(c, {
            "model": "whisper-1",
            "response_format": "verbose_json",
            "timestamp_granularities[]": "segment",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["segments"]) > 0


# ===================================================================
# VAL-OAI-010: verbose_json + timestamp_granularities[]=word -> words[]
# ===================================================================

class TestTimestampGranularitiesWord:

    def test_word_granularity_includes_words(self, client):
        c, _, _ = client
        resp = _post_transcriptions(c, {
            "model": "whisper-1",
            "response_format": "verbose_json",
            "timestamp_granularities[]": "word",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "words" in body
        assert body["words"] is not None
        assert len(body["words"]) > 0
        for w in body["words"]:
            assert "word" in w and isinstance(w["word"], str)
            assert "start" in w and isinstance(w["start"], (int, float))
            assert "end" in w and isinstance(w["end"], (int, float))
            assert w["end"] >= w["start"]


# ===================================================================
# VAL-OAI-011 + VAL-CROSS-003: both segment and word granularities
# ===================================================================

class TestBothGranularities:

    def test_both_granularities_has_segments_and_words(self, client):
        c, _, _ = client
        resp = _post_transcriptions(c, {
            "model": "whisper-1",
            "response_format": "verbose_json",
            "timestamp_granularities[]": ["segment", "word"],
        })
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["segments"]) > 0
        assert body["words"] is not None
        assert len(body["words"]) > 0


# ===================================================================
# VAL-OAI-012: timestamp_granularities without verbose_json -> 400
# ===================================================================

class TestTimestampGranularitiesWithoutVerboseJson:

    def test_granularities_with_json_format_returns_400(self, client):
        c, _, _ = client
        resp = _post_transcriptions(c, {
            "model": "whisper-1",
            "response_format": "json",
            "timestamp_granularities[]": "word",
        })
        assert resp.status_code == 400
        body = resp.json()
        assert "error" in body
        assert body["error"]["param"] == "timestamp_granularities"
        assert "verbose_json" in body["error"]["message"].lower() or "verbose" in body["error"]["message"].lower()

    def test_granularities_with_text_format_returns_400(self, client):
        c, _, _ = client
        resp = _post_transcriptions(c, {
            "model": "whisper-1",
            "response_format": "text",
            "timestamp_granularities[]": "word",
        })
        assert resp.status_code == 400


# ===================================================================
# VAL-OAI-013 + VAL-OAI-014: temperature out of range -> 400 OpenAI envelope
# ===================================================================

class TestTemperatureValidation:

    def test_temperature_above_range_returns_400_openai_envelope(self, client):
        """VAL-OAI-013: temperature=1.5 -> 400 with OpenAI envelope."""
        c, _, _ = client
        resp = _post_transcriptions(c, {"model": "whisper-1", "temperature": "1.5"})
        assert resp.status_code == 400
        body = resp.json()
        assert "error" in body
        assert body["error"]["param"] == "temperature"
        # Verify it's the OpenAI envelope, not FastAPI 422
        assert "detail" not in body

    def test_temperature_below_range_returns_400_openai_envelope(self, client):
        """VAL-OAI-014: temperature=-0.1 -> 400 with OpenAI envelope."""
        c, _, _ = client
        resp = _post_transcriptions(c, {"model": "whisper-1", "temperature": "-0.1"})
        assert resp.status_code == 400
        body = resp.json()
        assert "error" in body
        assert body["error"]["param"] == "temperature"
        assert "detail" not in body

    def test_temperature_in_range_accepted(self, client):
        """VAL-OAI-015: temperature=0.5 -> 200 (accepted)."""
        c, _, _ = client
        resp = _post_transcriptions(c, {"model": "whisper-1", "temperature": "0.5"})
        assert resp.status_code == 200
        assert "text" in resp.json()

    def test_temperature_zero_accepted(self, client):
        """Edge of range: temperature=0 -> 200."""
        c, _, _ = client
        resp = _post_transcriptions(c, {"model": "whisper-1", "temperature": "0"})
        assert resp.status_code == 200

    def test_temperature_one_accepted(self, client):
        """Edge of range: temperature=1 -> 200."""
        c, _, _ = client
        resp = _post_transcriptions(c, {"model": "whisper-1", "temperature": "1"})
        assert resp.status_code == 200


# ===================================================================
# VAL-OAI-016/017/018: model aliases
# ===================================================================

class TestModelAliases:

    def test_whisper_1_alias_resolves(self, client):
        """VAL-OAI-016: whisper-1 resolves to default model."""
        c, mock_queue, _ = client
        resp = _post_transcriptions(c, {"model": "whisper-1"})
        assert resp.status_code == 200
        assert "text" in resp.json()

    def test_whisper_large_v3_alias_resolves(self, client):
        """VAL-OAI-017: whisper-large-v3 resolves."""
        c, _, _ = client
        resp = _post_transcriptions(c, {"model": "whisper-large-v3"})
        assert resp.status_code == 200

    def test_whisper_tiny_alias_resolves(self, client):
        """VAL-OAI-018: whisper-tiny resolves."""
        c, _, _ = client
        resp = _post_transcriptions(c, {"model": "whisper-tiny"})
        assert resp.status_code == 200


# ===================================================================
# VAL-OAI-019: raw MLX model name accepted
# ===================================================================

class TestRawMLXModelNames:

    def test_tiny_model_accepted(self, client):
        c, _, _ = client
        resp = _post_transcriptions(c, {"model": "tiny"})
        assert resp.status_code == 200

    def test_base_model_accepted(self, client):
        c, _, _ = client
        resp = _post_transcriptions(c, {"model": "base"})
        assert resp.status_code == 200

    def test_small_model_accepted(self, client):
        c, _, _ = client
        resp = _post_transcriptions(c, {"model": "small"})
        assert resp.status_code == 200


# ===================================================================
# VAL-OAI-020: invalid model -> 400 OpenAI error
# ===================================================================

class TestInvalidModel:

    def test_invalid_model_returns_400_openai_error(self, client):
        c, _, _ = client
        resp = _post_transcriptions(c, {"model": "gpt-4o"})
        assert resp.status_code == 400
        body = resp.json()
        assert "error" in body
        assert body["error"]["param"] == "model"
        assert "gpt-4o" in body["error"]["message"]


# ===================================================================
# VAL-OAI-021: missing model -> client error with OpenAI envelope
# ===================================================================

class TestMissingModel:

    def test_missing_model_returns_4xx(self, client):
        c, _, _ = client
        # Don't include model in the form data
        resp = c.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code >= 400 and resp.status_code < 500

    def test_missing_model_returns_openai_envelope(self, client):
        """VAL-OAI-025: missing model returns OpenAI error envelope, not bare FastAPI 422."""
        c, _, _ = client
        resp = c.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 400
        body = resp.json()
        assert "error" in body, f"Expected OpenAI envelope, got: {body}"
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["param"] == "model"
        assert "detail" not in body, "Should not have bare FastAPI 'detail' shape"


# ===================================================================
# VAL-OAI-022: missing file -> client error with OpenAI envelope
# ===================================================================

class TestMissingFile:

    def test_missing_file_returns_4xx(self, client):
        c, _, _ = client
        resp = c.post(
            "/v1/audio/transcriptions",
            data={"model": "whisper-1"},
        )
        assert resp.status_code >= 400 and resp.status_code < 500

    def test_missing_file_returns_openai_envelope(self, client):
        """VAL-OAI-025: missing file returns OpenAI error envelope, not bare FastAPI 422."""
        c, _, _ = client
        resp = c.post(
            "/v1/audio/transcriptions",
            data={"model": "whisper-1"},
        )
        assert resp.status_code == 400
        body = resp.json()
        assert "error" in body, f"Expected OpenAI envelope, got: {body}"
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["param"] == "file"
        assert "detail" not in body, "Should not have bare FastAPI 'detail' shape"


# ===================================================================
# VAL-OAI-023: prompt accepted as initial_prompt
# ===================================================================

class TestPrompt:

    def test_prompt_accepted(self, client):
        """VAL-OAI-023: prompt parameter is accepted (200)."""
        c, mock_queue, _ = client
        resp = _post_transcriptions(c, {"model": "whisper-1", "prompt": "context phrase"})
        assert resp.status_code == 200


# ===================================================================
# VAL-OAI-024: hotwords accepted but no-op
# ===================================================================

class TestHotwords:

    def test_hotwords_accepted_no_error(self, client):
        """VAL-OAI-024: hotwords parameter is accepted (200)."""
        c, _, _ = client
        resp = _post_transcriptions(c, {"model": "whisper-1", "hotwords": "Speaker,CTranslate2"})
        assert resp.status_code == 200
        assert "text" in resp.json()


# ===================================================================
# VAL-OAI-025: OpenAI error envelope shape
# ===================================================================

class TestOpenAIErrorEnvelope:

    def test_error_has_openai_envelope(self, client):
        """Error responses use {error: {message, type, param, code}} shape."""
        c, _, _ = client
        # Trigger a 400 with invalid model
        resp = _post_transcriptions(c, {"model": "gpt-4o"})
        assert resp.status_code == 400
        body = resp.json()
        assert "error" in body
        error = body["error"]
        assert "message" in error and isinstance(error["message"], str)
        assert "type" in error and isinstance(error["type"], str)
        assert "param" in error
        assert "code" in error

    def test_no_fastapi_detail_shape_on_validation_errors(self, client):
        """4xx from our validation should NOT use {"detail": ...} shape."""
        c, _, _ = client
        resp = _post_transcriptions(c, {"model": "gpt-4o"})
        body = resp.json()
        assert "detail" not in body

    def test_fastapi_request_validation_returns_openai_envelope(self, client):
        """VAL-OAI-025: FastAPI RequestValidationError on /v1/ paths returns OpenAI envelope."""
        c, _, _ = client
        # Missing required 'model' triggers FastAPI's RequestValidationError
        resp = c.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 400
        body = resp.json()
        assert "error" in body, f"Expected OpenAI envelope, got: {body}"
        assert "detail" not in body, f"Should not have bare 'detail' shape: {body}"
        error = body["error"]
        assert error["type"] == "invalid_request_error"
        assert error["param"] == "model"


# ===================================================================
# VAL-OAI-033: json/text transcript text matches
# ===================================================================

class TestTextConsistency:

    def test_json_and_text_formats_agree(self, client):
        c, _, _ = client
        # JSON format
        resp_json = _post_transcriptions(c, {"model": "whisper-1", "response_format": "json"})
        assert resp_json.status_code == 200
        json_text = resp_json.json()["text"]

        # Text format
        resp_text = _post_transcriptions(c, {"model": "whisper-1", "response_format": "text"})
        assert resp_text.status_code == 200
        text_text = resp_text.text.strip()

        assert json_text.strip() == text_text

    def test_verbose_json_text_matches_json(self, client):
        c, _, _ = client
        resp_json = _post_transcriptions(c, {"model": "whisper-1", "response_format": "json"})
        json_text = resp_json.json()["text"].strip()

        resp_verbose = _post_transcriptions(c, {
            "model": "whisper-1",
            "response_format": "verbose_json",
            "timestamp_granularities[]": ["segment", "word"],
        })
        verbose_text = resp_verbose.json()["text"].strip()

        assert json_text == verbose_text


# ===================================================================
# VAL-CROSS-004: verbose_json text equals concatenation of segments
# ===================================================================

class TestVerboseJsonTextConsistency:

    def test_text_equals_segment_concatenation(self, client):
        c, _, _ = client
        resp = _post_transcriptions(c, {
            "model": "whisper-1",
            "response_format": "verbose_json",
            "timestamp_granularities[]": ["segment", "word"],
        })
        assert resp.status_code == 200
        body = resp.json()
        top_text = body["text"].strip()
        seg_text = " ".join(s["text"].strip() for s in body["segments"]).strip()
        assert top_text == seg_text


# ===================================================================
# VAL-OAI-037: word timestamps globally time-ordered
# ===================================================================

class TestWordTimestampOrdering:

    def test_word_timestamps_monotonically_non_decreasing(self, client):
        c, _, _ = client
        resp = _post_transcriptions(c, {
            "model": "whisper-1",
            "response_format": "verbose_json",
            "timestamp_granularities[]": "word",
        })
        assert resp.status_code == 200
        body = resp.json()
        words = body["words"]
        assert len(words) > 0
        # Check end >= start for each word
        for w in words:
            assert w["end"] >= w["start"]
        # Check start times are non-decreasing
        for i in range(1, len(words)):
            assert words[i]["start"] >= words[i - 1]["start"], (
                f"Word {i} start {words[i]['start']} < word {i-1} start {words[i-1]['start']}"
            )


# ===================================================================
# VAL-OAI-038: MLX-specific canonical model ids accepted
# ===================================================================

class TestMLXSpecificModels:

    def test_large_v3_turbo_accepted(self, client):
        c, _, _ = client
        resp = _post_transcriptions(c, {"model": "large-v3-turbo"})
        assert resp.status_code == 200

    def test_turbo_accepted(self, client):
        c, _, _ = client
        resp = _post_transcriptions(c, {"model": "turbo"})
        assert resp.status_code == 200

    def test_tiny_en_accepted(self, client):
        c, _, _ = client
        resp = _post_transcriptions(c, {"model": "tiny.en"})
        assert resp.status_code == 200

    def test_large_accepted(self, client):
        c, _, _ = client
        resp = _post_transcriptions(c, {"model": "large"})
        assert resp.status_code == 200

    def test_large_v1_accepted(self, client):
        c, _, _ = client
        resp = _post_transcriptions(c, {"model": "large-v1"})
        assert resp.status_code == 200
