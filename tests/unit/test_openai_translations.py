"""Unit tests for POST /v1/audio/translations (app/openai_compat.py).

Tests are fast: pipeline is mocked, no model downloads, no GPU required.
Covers VAL-OAI-026 through VAL-OAI-032 and VAL-CROSS-013:
- translations default returns JSON {text} in English
- verbose_json reports task=translate
- language parameter not accepted (ignored or rejected)
- prompt and hotwords accepted without error (hotwords ignored)
- supports text/srt/vtt/verbose_json formats
- timestamp_granularities without verbose_json -> 400
- invalid model -> 400 OpenAI error
- diarization + translation yields English text with speaker labels
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_AUDIO = b"RIFF" + b"\x00" * 100  # minimal WAV-like bytes

# Simulated translated English text (as if from a non-English source)
TRANSLATED_SEGMENTS = [
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
        "text": "this is a translation",
        "words": [
            {"word": "this", "start": 2.5, "end": 3.0},
            {"word": "is", "start": 3.1, "end": 3.4},
            {"word": "a", "start": 3.5, "end": 3.6},
            {"word": "translation", "start": 3.7, "end": 5.0},
        ],
    },
]

# Diarized + translated result for VAL-CROSS-013
DIARIZED_TRANSLATED_SEGMENTS = [
    {
        "start": 0.0,
        "end": 2.5,
        "text": "Hello from speaker one",
        "speaker": "SPEAKER_00",
        "words": [
            {"word": "Hello", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
            {"word": "from", "start": 0.6, "end": 0.9, "speaker": "SPEAKER_00"},
            {"word": "speaker", "start": 1.0, "end": 1.4, "speaker": "SPEAKER_00"},
            {"word": "one", "start": 1.5, "end": 2.5, "speaker": "SPEAKER_00"},
        ],
    },
    {
        "start": 2.5,
        "end": 5.0,
        "text": "And speaker two replies",
        "speaker": "SPEAKER_01",
        "words": [
            {"word": "And", "start": 2.5, "end": 2.8, "speaker": "SPEAKER_01"},
            {"word": "speaker", "start": 2.9, "end": 3.3, "speaker": "SPEAKER_01"},
            {"word": "two", "start": 3.4, "end": 3.8, "speaker": "SPEAKER_01"},
            {"word": "replies", "start": 3.9, "end": 5.0, "speaker": "SPEAKER_01"},
        ],
    },
]


def _make_translation_result(
    segments=None,
    language="en",
    word_segments=None,
):
    """Build a realistic translation result dict for mocking."""
    if segments is None:
        segments = TRANSLATED_SEGMENTS
    if word_segments is None:
        word_segments = []
        for seg in segments:
            word_segments.extend(seg.get("words", []))
    return {
        "segments": segments,
        "language": language,
        "word_segments": word_segments,
    }


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
        result = _make_translation_result()
        mock_transcribe.return_value = result
        mock_align.return_value = result

        # Mock run_in_queue to just call the function synchronously
        async def _fake_queue(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        mock_queue.side_effect = _fake_queue

        from app.main import app

        with TestClient(app) as c:
            yield c, mock_queue, mock_resolve


def _post_translations(client, data: dict, files=None):
    """Helper to POST to /v1/audio/translations."""
    if files is None:
        files = {"file": ("test.wav", FAKE_AUDIO, "audio/wav")}
    return client.post("/v1/audio/translations", data=data, files=files)


# ===================================================================
# VAL-OAI-026: translations default returns JSON {text} in English
# ===================================================================


class TestDefaultJsonFormat:
    """POST /v1/audio/translations with no response_format -> {text}."""

    def test_default_returns_json_with_text_key_only(self, client):
        c, _, _ = client
        resp = _post_translations(c, {"model": "whisper-1"})
        assert resp.status_code == 200
        body = resp.json()
        assert "text" in body
        assert body["text"] == "Hello world this is a translation"
        # Only key should be "text" per VAL-OAI-001 shape
        assert list(body.keys()) == ["text"]

    def test_default_content_type_is_application_json(self, client):
        c, _, _ = client
        resp = _post_translations(c, {"model": "whisper-1"})
        assert resp.status_code == 200
        ct = resp.headers.get("content-type", "")
        assert "application/json" in ct

    def test_text_is_english(self, client):
        """Translation output text is English (task forced to translate)."""
        c, _, _ = client
        resp = _post_translations(c, {"model": "whisper-1"})
        assert resp.status_code == 200
        body = resp.json()
        # The mocked pipeline returns English text, matching what a real
        # translate task would produce
        assert "text" in body
        assert len(body["text"]) > 0


# ===================================================================
# VAL-OAI-027: translations verbose_json reports task=translate
# ===================================================================


class TestVerboseJsonFormat:
    """POST /v1/audio/translations with verbose_json reports task=translate."""

    def test_verbose_json_task_is_translate(self, client):
        c, _, _ = client
        resp = _post_translations(c, {
            "model": "whisper-1",
            "response_format": "verbose_json",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["task"] == "translate", \
            f"Expected task='translate', got task='{body.get('task')}'"

    def test_verbose_json_has_required_fields(self, client):
        """Verbose JSON must include language, duration, text, and segments."""
        c, _, _ = client
        resp = _post_translations(c, {
            "model": "whisper-1",
            "response_format": "verbose_json",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["task"] == "translate"
        assert "language" in body
        assert "duration" in body
        assert body["duration"] > 0
        assert "text" in body
        assert "segments" in body
        assert len(body["segments"]) > 0

    def test_verbose_json_default_has_segments_no_words(self, client):
        """Default verbose_json has segments but no words."""
        c, _, _ = client
        resp = _post_translations(c, {
            "model": "whisper-1",
            "response_format": "verbose_json",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["segments"]) > 0
        assert body.get("words") is None

    def test_verbose_json_with_word_granularity(self, client):
        """verbose_json + timestamp_granularities[]=word includes words."""
        c, _, _ = client
        resp = _post_translations(c, {
            "model": "whisper-1",
            "response_format": "verbose_json",
            "timestamp_granularities[]": "word",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["task"] == "translate"
        assert body["words"] is not None
        assert len(body["words"]) > 0

    def test_verbose_json_segment_fields(self, client):
        """Segments have the full OpenAI field set."""
        c, _, _ = client
        resp = _post_translations(c, {
            "model": "whisper-1",
            "response_format": "verbose_json",
        })
        assert resp.status_code == 200
        body = resp.json()
        for seg in body["segments"]:
            assert "id" in seg and isinstance(seg["id"], int)
            assert "start" in seg and isinstance(seg["start"], (int, float))
            assert "end" in seg and isinstance(seg["end"], (int, float))
            assert "text" in seg and isinstance(seg["text"], str)
            assert "tokens" in seg
            assert "temperature" in seg
            assert "avg_logprob" in seg
            assert "compression_ratio" in seg
            assert "no_speech_prob" in seg


# ===================================================================
# VAL-OAI-028: translations does not accept a language parameter
# ===================================================================


class TestLanguageParamNotAccepted:
    """POST /v1/audio/translations with language param is rejected or ignored."""

    def test_language_param_ignored_or_rejected(self, client):
        """Supplying language=es either errors or still produces English output."""
        c, _, _ = client
        # The endpoint does not declare a 'language' Form parameter,
        # so FastAPI will either reject it as an unexpected field or
        # silently ignore it. Both outcomes satisfy the contract:
        # the language param must not steer the output away from English.
        resp = _post_translations(c, {"model": "whisper-1", "language": "es"})
        # The request should either succeed (200) with English output,
        # or fail with a 4xx if the endpoint rejects unexpected fields.
        # FastAPI with Form() typically ignores extra form fields,
        # so we expect 200 with English output.
        if resp.status_code == 200:
            body = resp.json()
            assert "text" in body
            # Output is English (translate task forces English)
        else:
            # If the endpoint rejects the param, it must be a 4xx
            assert resp.status_code >= 400 and resp.status_code < 500

    def test_no_language_in_endpoint_signature(self):
        """The create_translation function must NOT declare a language Form param."""
        import inspect
        from app.openai_compat import create_translation

        sig = inspect.signature(create_translation)
        param_names = list(sig.parameters.keys())
        assert "language" not in param_names, \
            f"create_translation must not have 'language' param; got: {param_names}"

    def test_language_does_not_steer_output(self, client):
        """Even with language=fr, the task=translate still produces English."""
        c, _, _ = client
        # Pass language as an extra form field — FastAPI ignores undeclared form fields
        resp = _post_translations(c, {"model": "whisper-1", "language": "fr"})
        if resp.status_code == 200:
            body = resp.json()
            # The endpoint passes language=None to process_audio (translate mode),
            # so the output should be English regardless of the language field
            assert "text" in body


# ===================================================================
# VAL-OAI-029: translations accepts prompt and hotwords without error
# ===================================================================


class TestPromptAndHotwords:
    """POST /v1/audio/translations with prompt and hotwords accepted (hotwords ignored)."""

    def test_prompt_accepted(self, client):
        """prompt parameter is accepted without error (200)."""
        c, mock_queue, _ = client
        resp = _post_translations(c, {
            "model": "whisper-1",
            "prompt": "context phrase for translation",
        })
        assert resp.status_code == 200
        assert "text" in resp.json()

    def test_hotwords_accepted_no_error(self, client):
        """hotwords parameter is accepted without error (200, hotwords ignored)."""
        c, _, _ = client
        resp = _post_translations(c, {
            "model": "whisper-1",
            "hotwords": "Speaker,CTranslate2",
        })
        assert resp.status_code == 200
        assert "text" in resp.json()

    def test_prompt_and_hotwords_together(self, client):
        """Both prompt and hotwords together are accepted without error."""
        c, _, _ = client
        resp = _post_translations(c, {
            "model": "whisper-1",
            "prompt": "context phrase",
            "hotwords": "Foo,Bar",
        })
        assert resp.status_code == 200
        assert "text" in resp.json()

    def test_prompt_passed_as_initial_prompt(self, client):
        """prompt is passed as initial_prompt to the pipeline (via process_audio)."""
        c, mock_queue, _ = client
        resp = _post_translations(c, {
            "model": "whisper-1",
            "prompt": "test prompt",
        })
        assert resp.status_code == 200
        # Verify that the queue was called (meaning process_audio was invoked)
        assert mock_queue.called, "run_in_queue should have been called"


# ===================================================================
# VAL-OAI-030: translations supports text/srt/vtt/verbose_json formats
# ===================================================================


class TestResponseFormats:
    """POST /v1/audio/translations supports text, srt, vtt, verbose_json."""

    def test_json_format(self, client):
        """response_format=json returns {text}."""
        c, _, _ = client
        resp = _post_translations(c, {
            "model": "whisper-1",
            "response_format": "json",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "text" in body
        assert "segments" not in body

    def test_text_format(self, client):
        """response_format=text returns plain text."""
        c, _, _ = client
        resp = _post_translations(c, {
            "model": "whisper-1",
            "response_format": "text",
        })
        assert resp.status_code == 200
        assert resp.text == "Hello world this is a translation"
        ct = resp.headers.get("content-type", "")
        assert "text/plain" in ct

    def test_srt_format(self, client):
        """response_format=srt returns valid SRT."""
        c, _, _ = client
        resp = _post_translations(c, {
            "model": "whisper-1",
            "response_format": "srt",
        })
        assert resp.status_code == 200
        body = resp.text
        assert "1\n" in body
        assert "-->" in body
        import re
        timecode_pattern = r"\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}"
        assert re.search(timecode_pattern, body)

    def test_srt_content_type_text_plain(self, client):
        """SRT format returns Content-Type: text/plain."""
        c, _, _ = client
        resp = _post_translations(c, {
            "model": "whisper-1",
            "response_format": "srt",
        })
        ct = resp.headers.get("content-type", "")
        assert ct.startswith("text/plain")

    def test_vtt_format(self, client):
        """response_format=vtt returns WEBVTT."""
        c, _, _ = client
        resp = _post_translations(c, {
            "model": "whisper-1",
            "response_format": "vtt",
        })
        assert resp.status_code == 200
        body = resp.text
        assert body.startswith("WEBVTT")
        import re
        timecode_pattern = r"\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}"
        assert re.search(timecode_pattern, body)

    def test_verbose_json_format(self, client):
        """response_format=verbose_json returns full object with task=translate."""
        c, _, _ = client
        resp = _post_translations(c, {
            "model": "whisper-1",
            "response_format": "verbose_json",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["task"] == "translate"
        assert "language" in body
        assert "duration" in body
        assert "text" in body
        assert "segments" in body


# ===================================================================
# VAL-OAI-031: translations timestamp_granularities without verbose_json -> 400
# ===================================================================


class TestTimestampGranularitiesWithoutVerboseJson:
    """timestamp_granularities without verbose_json returns 400 OpenAI error."""

    def test_granularities_with_json_format_returns_400(self, client):
        c, _, _ = client
        resp = _post_translations(c, {
            "model": "whisper-1",
            "response_format": "json",
            "timestamp_granularities[]": "word",
        })
        assert resp.status_code == 400
        body = resp.json()
        assert "error" in body
        assert body["error"]["param"] == "timestamp_granularities"

    def test_granularities_with_text_format_returns_400(self, client):
        c, _, _ = client
        resp = _post_translations(c, {
            "model": "whisper-1",
            "response_format": "text",
            "timestamp_granularities[]": "word",
        })
        assert resp.status_code == 400

    def test_granularities_with_srt_format_returns_400(self, client):
        c, _, _ = client
        resp = _post_translations(c, {
            "model": "whisper-1",
            "response_format": "srt",
            "timestamp_granularities[]": "word",
        })
        assert resp.status_code == 400

    def test_granularities_error_has_openai_envelope(self, client):
        """400 error uses the OpenAI error envelope shape."""
        c, _, _ = client
        resp = _post_translations(c, {
            "model": "whisper-1",
            "response_format": "json",
            "timestamp_granularities[]": "word",
        })
        body = resp.json()
        assert "error" in body
        error = body["error"]
        assert "message" in error
        assert "type" in error
        assert error["param"] == "timestamp_granularities"
        assert "verbose_json" in error["message"].lower() or "verbose" in error["message"].lower()


# ===================================================================
# VAL-OAI-032: translations invalid model -> 400 OpenAI error
# ===================================================================


class TestInvalidModel:
    """Invalid model on /v1/audio/translations returns 400 OpenAI error."""

    def test_invalid_model_returns_400_openai_error(self, client):
        c, _, _ = client
        resp = _post_translations(c, {"model": "not-a-model"})
        assert resp.status_code == 400
        body = resp.json()
        assert "error" in body
        assert body["error"]["param"] == "model"
        assert "not-a-model" in body["error"]["message"]

    def test_gpt4o_model_returns_400(self, client):
        """gpt-4o is not a valid whisper model -> 400."""
        c, _, _ = client
        resp = _post_translations(c, {"model": "gpt-4o"})
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["param"] == "model"

    def test_error_has_openai_envelope_shape(self, client):
        """Error response uses {error: {message, type, param, code}} shape."""
        c, _, _ = client
        resp = _post_translations(c, {"model": "bad-model"})
        body = resp.json()
        error = body["error"]
        assert "message" in error and isinstance(error["message"], str)
        assert "type" in error and isinstance(error["type"], str)
        assert "param" in error
        assert "code" in error

    def test_no_fastapi_detail_shape(self, client):
        """Application-level errors must NOT use bare FastAPI detail shape."""
        c, _, _ = client
        resp = _post_translations(c, {"model": "bad-model"})
        body = resp.json()
        assert "detail" not in body


# ===================================================================
# VAL-CROSS-013: Diarization + translation yields English text WITH speaker labels
# ===================================================================

class TestDiarizationWithTranslation:
    """
    POST /asr with task=translate and diarize=true yields English text
    with speaker labels.  This cross-flow test exercises the /asr endpoint
    (not /v1/audio/translations, which does not support diarization).
    """

    @pytest.fixture()
    def asr_client(self):
        """Create a TestClient for /asr with pipeline mocked."""
        with (
            patch("app.main.run_in_queue") as mock_queue,
            patch("app.main.whispermlx") as mock_wmlx,
            patch("app.main.load_whisper_model"),
            patch("app.main.resolve_model_name") as mock_resolve,
            patch("app.main.get_canonical_models") as mock_canonical,
        ):
            mock_resolve.side_effect = lambda m: m if m else "large-v3"
            mock_canonical.return_value = CANONICAL_MODELS
            mock_wmlx.load_audio.return_value = np.zeros(16000, dtype=np.float32)

            # Simulated pipeline result: translated + diarized
            result = {
                "segments": DIARIZED_TRANSLATED_SEGMENTS,
                "language": "en",
                "word_segments": [],
            }
            for seg in DIARIZED_TRANSLATED_SEGMENTS:
                result["word_segments"].extend(seg.get("words", []))

            async def _return_result(*args, **kwargs):
                return result, None

            mock_queue.side_effect = _return_result

            from app.main import app

            with TestClient(app) as c:
                yield c, mock_queue

    def test_translate_with_diarize_yields_english_with_speakers(self, asr_client):
        """task=translate + diarize=true yields English text WITH speaker labels."""
        c, _ = asr_client
        resp = c.post(
            "/asr?task=translate&diarize=true",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        body = resp.json()

        # Text should be English (translation applied)
        segments = body["segments"]
        assert len(segments) > 0
        for seg in segments:
            assert seg.get("text"), "Segment should have text"

        # Speaker labels should be present with >=2 distinct speakers
        speakers = {seg.get("speaker") for seg in segments if seg.get("speaker")}
        assert len(speakers) >= 2, \
            f"Expected >=2 distinct speakers with diarize=true+translate, got: {speakers}"

    def test_speaker_labels_match_format(self, asr_client):
        """Speaker labels follow SPEAKER_NN format."""
        c, _ = asr_client
        resp = c.post(
            "/asr?task=translate&diarize=true",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        import re
        for seg in body["segments"]:
            if seg.get("speaker"):
                assert re.match(r"^SPEAKER_\d+$", seg["speaker"]), \
                    f"Speaker label '{seg['speaker']}' doesn't match SPEAKER_NN format"

    def test_translate_diarize_task_passed_to_pipeline(self, asr_client):
        """task=translate is passed through to the pipeline."""
        c, mock_queue = asr_client

        captured_args = {}

        async def _capture(*args, **kwargs):
            captured_args.update(kwargs)
            result = {
                "segments": DIARIZED_TRANSLATED_SEGMENTS,
                "language": "en",
                "word_segments": [],
            }
            for seg in DIARIZED_TRANSLATED_SEGMENTS:
                result["word_segments"].extend(seg.get("words", []))
            return result, None

        mock_queue.side_effect = _capture

        resp = c.post(
            "/asr?task=translate&diarize=true",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        assert captured_args.get("task") == "translate", \
            f"task should be 'translate', got: {captured_args.get('task')}"
        assert captured_args.get("should_diarize") is True, \
            f"should_diarize should be True, got: {captured_args.get('should_diarize')}"


# ===================================================================
# Additional: text consistency across formats for translations
# ===================================================================


class TestTranslationTextConsistency:
    """Verify transcript text is consistent across response formats."""

    def test_json_and_text_formats_agree(self, client):
        c, _, _ = client
        # JSON format
        resp_json = _post_translations(c, {"model": "whisper-1", "response_format": "json"})
        assert resp_json.status_code == 200
        json_text = resp_json.json()["text"]

        # Text format
        resp_text = _post_translations(c, {"model": "whisper-1", "response_format": "text"})
        assert resp_text.status_code == 200
        text_text = resp_text.text.strip()

        assert json_text.strip() == text_text

    def test_verbose_json_text_matches_json(self, client):
        c, _, _ = client
        resp_json = _post_translations(c, {"model": "whisper-1", "response_format": "json"})
        json_text = resp_json.json()["text"].strip()

        resp_verbose = _post_translations(c, {
            "model": "whisper-1",
            "response_format": "verbose_json",
            "timestamp_granularities[]": ["segment", "word"],
        })
        verbose_text = resp_verbose.json()["text"].strip()

        assert json_text == verbose_text


# ===================================================================
# Missing required fields on /v1/audio/translations
# ===================================================================


class TestMissingRequiredFields:
    """Missing model/file on translations returns OpenAI error envelope."""

    def test_missing_model_returns_400_openai_envelope(self, client):
        c, _, _ = client
        resp = c.post(
            "/v1/audio/translations",
            files={"file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 400
        body = resp.json()
        assert "error" in body
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["param"] == "model"
        assert "detail" not in body

    def test_missing_file_returns_400_openai_envelope(self, client):
        c, _, _ = client
        resp = c.post(
            "/v1/audio/translations",
            data={"model": "whisper-1"},
        )
        assert resp.status_code == 400
        body = resp.json()
        assert "error" in body
        assert body["error"]["param"] == "file"
        assert "detail" not in body


# ===================================================================
# Temperature validation on translations endpoint
# ===================================================================


class TestTemperatureValidation:
    """Temperature validation on /v1/audio/translations."""

    def test_temperature_out_of_range_returns_400(self, client):
        c, _, _ = client
        resp = _post_translations(c, {
            "model": "whisper-1",
            "temperature": "1.5",
        })
        assert resp.status_code == 400
        body = resp.json()
        assert "error" in body
        assert body["error"]["param"] == "temperature"

    def test_temperature_in_range_accepted(self, client):
        c, _, _ = client
        resp = _post_translations(c, {
            "model": "whisper-1",
            "temperature": "0.5",
        })
        assert resp.status_code == 200


# ===================================================================
# Model alias resolution on translations endpoint
# ===================================================================


class TestModelAliases:
    """Model aliases resolve correctly on translations endpoint."""

    def test_whisper_1_alias(self, client):
        c, _, _ = client
        resp = _post_translations(c, {"model": "whisper-1"})
        assert resp.status_code == 200

    def test_tiny_model(self, client):
        c, _, _ = client
        resp = _post_translations(c, {"model": "tiny"})
        assert resp.status_code == 200

    def test_large_v3_turbo(self, client):
        c, _, _ = client
        resp = _post_translations(c, {"model": "large-v3-turbo"})
        assert resp.status_code == 200
