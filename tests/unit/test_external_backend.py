"""
Unit tests for the optional external ASR backend (ASR_BACKEND=external).

Ported from upstream cfa5eb3 (#22). HTTP providers are mocked via
requests.post; the whispermlx/qwen3 stages in app.pipeline are exercised
through routing assertions. No network calls occur.
"""

from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np

import app.external_backend as external
import app.pipeline as pipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_audio(seconds: float = 1.0) -> np.ndarray:
    return np.zeros(int(seconds * external.SAMPLE_RATE), dtype=np.float32)


# ---------------------------------------------------------------------------
# external_backend internals
# ---------------------------------------------------------------------------


class TestWavBytes:
    def test_produces_valid_wav(self):
        audio = np.zeros(external.SAMPLE_RATE, dtype=np.float32)
        wav = external._wav_bytes(audio)
        # RIFF header
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"

    def test_clipping(self):
        audio = np.array([2.0, -2.0], dtype=np.float32)
        wav = external._wav_bytes(audio)
        # 2 samples x 2 bytes
        assert len(wav) > 44


class TestLanguageCode:
    def test_short_code_passthrough(self):
        assert external._language_code("EN", None) == "en"
        assert external._language_code("zh ", None) == "zh"

    def test_name_resolved_via_languages_table(self):
        if external._language_code("english", "xx") == "english":
            raise AssertionError("name not resolved")
        assert external._language_code("english", "fr") == "en"

    def test_none_falls_back(self):
        assert external._language_code(None, "de") == "de"

    def test_unknown_name_falls_back(self):
        assert external._language_code("klingon", "en") == "en"


class TestRequireConfig:
    def test_missing_config_raises(self):
        with (
            patch.object(external, "BASE_URL", ""),
            patch.object(external, "MODEL", ""),
        ):
            try:
                external._require_config()
                raised = False
            except RuntimeError as e:
                raised = True
                assert "EXTERNAL_ASR_BASE_URL" in str(e)
                assert "EXTERNAL_ASR_MODEL" in str(e)
        assert raised

    def test_complete_config_ok(self):
        with (
            patch.object(external, "BASE_URL", "https://api.example.com/v1"),
            patch.object(external, "MODEL", "whisper-1"),
        ):
            external._require_config()  # should not raise


class TestTranscribeEndpoint:
    def test_timestamped_segments_no_qwen_align(self):
        resp = MagicMock(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: {
                "text": "",
                "language": "english",
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "hello"},
                    {"start": 1.0, "end": 2.0, "text": "there"},
                ],
            },
        )
        with (
            patch.object(external, "BASE_URL", "https://api.example.com/v1"),
            patch.object(external, "MODEL", "whisper-1"),
            patch.object(external.requests, "post", return_value=resp) as post_mock,
        ):
            result = external.transcribe(make_audio(2.0), language="en")
        assert result["_asr_backend"] == "external"
        assert "_qwen_align" not in result
        assert result["language"] == "en"
        assert result["segments"] == [
            {"start": 0.0, "end": 1.0, "text": "hello"},
            {"start": 1.0, "end": 2.0, "text": "there"},
        ]
        assert post_mock.call_args.kwargs["data"]["response_format"] == "verbose_json"

    def test_text_only_response_sets_qwen_align(self):
        resp = MagicMock(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: {"text": "only text", "language": "english"},
        )
        with (
            patch.object(external, "BASE_URL", "https://api.example.com/v1"),
            patch.object(external, "MODEL", "whisper-1"),
            patch.object(external.requests, "post", return_value=resp),
        ):
            result = external.transcribe(make_audio(2.0), language="en")
        assert result["_qwen_align"] is True
        assert result["segments"] == [{"start": 0.0, "end": 2.0, "text": "only text"}]
        assert result["_language_name"] == "English"

    def test_verbose_json_retry_on_400(self):
        """A provider rejecting response_format=verbose_json falls back to json."""
        ok_resp = MagicMock(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: {"text": "plain text", "language": "en"},
        )

        calls = []

        def post_side_effect(*args, **kwargs):
            calls.append(dict(kwargs.get("data") or {}))
            if calls[-1].get("response_format") == "verbose_json":
                return MagicMock(status_code=400, text="Invalid response_format value")
            return ok_resp

        with (
            patch.object(external, "BASE_URL", "https://api.example.com/v1"),
            patch.object(external, "MODEL", "whisper-1"),
            patch.object(external.requests, "post", side_effect=post_side_effect),
        ):
            result = external.transcribe(make_audio(1.0))
        assert len(calls) == 2
        assert calls[0]["response_format"] == "verbose_json"
        assert calls[1]["response_format"] == "json"
        assert result["segments"][0]["text"] == "plain text"

    def test_provider_error_raises(self):
        resp = MagicMock(status_code=503)
        resp.raise_for_status.side_effect = Exception("503")
        with (
            patch.object(external, "BASE_URL", "https://api.example.com/v1"),
            patch.object(external, "MODEL", "whisper-1"),
            patch.object(external.requests, "post", return_value=resp),
        ):
            try:
                external.transcribe(make_audio(1.0))
                raised = False
            except Exception:
                raised = True
        assert raised


class TestTranscribeChat:
    def test_chat_chunks_and_strips_empties(self):
        resp = MagicMock(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": "transcribed chunk"}}]},
        )
        with (
            patch.object(external, "BASE_URL", "https://api.example.com/v1"),
            patch.object(external, "MODEL", "voxtral"),
            patch.object(external, "MODE", "chat"),
            patch.object(external, "CHUNK_SECONDS", 1),
            patch.object(external.requests, "post", return_value=resp) as post_mock,
        ):
            result = external.transcribe(make_audio(2.5), language="en", context="ctx")
        # 2.5s audio with 1s chunks -> 3 chat requests, one per chunk
        assert len(post_mock.call_args_list) == 3
        assert len(result["segments"]) == 3
        assert result["segments"][0] == {"start": 0.0, "end": 1.0, "text": "transcribed chunk"}
        assert result["segments"][2] == {"start": 2.0, "end": 2.5, "text": "transcribed chunk"}
        assert result["_qwen_align"] is True
        body = post_mock.call_args_list[0].kwargs["json"]
        content = body["messages"][0]["content"]
        assert content[0]["type"] == "text"
        assert "Context: ctx" in content[0]["text"]
        assert content[1]["type"] == "input_audio"

    def test_chat_language_and_context_in_instruction(self):
        resp = MagicMock(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": "text"}}]},
        )
        with (
            patch.object(external, "BASE_URL", "https://api.example.com/v1"),
            patch.object(external, "MODEL", "voxtral"),
            patch.object(external, "MODE", "chat"),
            patch.object(external.requests, "post", return_value=resp) as post_mock,
        ):
            external.transcribe(make_audio(0.5), language="fr", context="domain words")
        body = post_mock.call_args.kwargs["json"]
        text = body["messages"][0]["content"][0]["text"]
        assert "The audio language is fr" in text
        assert "Context: domain words" in text

    def test_chat_no_language_no_context_plain_instruction(self):
        resp = MagicMock(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": "text"}}]},
        )
        with (
            patch.object(external, "BASE_URL", "https://api.example.com/v1"),
            patch.object(external, "MODEL", "voxtral"),
            patch.object(external, "MODE", "chat"),
            patch.object(external.requests, "post", return_value=resp) as post_mock,
        ):
            external.transcribe(make_audio(0.5))
        body = post_mock.call_args.kwargs["json"]
        text = body["messages"][0]["content"][0]["text"]
        assert "The audio language is" not in text
        assert "Context:" not in text


# ---------------------------------------------------------------------------
# app.pipeline routing (ASR_BACKEND=external)
# ---------------------------------------------------------------------------


def _env_external():
    return patch.object(pipeline, "ASR_BACKEND", "external")


class TestExternalRouting:
    def test_transcribe_routes_to_external(self):
        mock_transcribe = MagicMock(return_value={"segments": [], "language": "en", "_asr_backend": "external"})
        with (
            _env_external(),
            patch("app.external_backend.transcribe", mock_transcribe),
        ):
            result = pipeline.transcribe(np.zeros(1600, dtype=np.float32), initial_prompt="ctx")
        assert result["language"] == "en"
        assert mock_transcribe.call_args.kwargs.get("context") == "ctx"

    def test_translate_falls_back_to_whisper(self):
        wmlx_model = MagicMock()
        wmlx_model.transcribe.return_value = {"segments": [], "language": "en"}
        q_transcribe = MagicMock()
        with (
            _env_external(),
            patch.object(pipeline, "load_whisper_model", return_value=wmlx_model),
            patch("app.external_backend.transcribe", q_transcribe),
        ):
            result = pipeline.transcribe(np.zeros(1600, dtype=np.float32), task="translate")
        assert result["language"] == "en"
        q_transcribe.assert_not_called()
        wmlx_model.transcribe.assert_called_once()

    def test_timestamped_result_uses_wav2vec2_alignment(self):
        result_in = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
            "language": "en",
            "_asr_backend": "external",
        }
        wmlx = MagicMock()
        wmlx.align.return_value = {**result_in, "segment_aligned": True}
        original_wmlx = pipeline.whispermlx
        pipeline.whispermlx = wmlx
        try:
            with (
                patch.object(pipeline, "load_align_model", return_value=(MagicMock(), {})),
                patch.object(pipeline, "clear_gpu_memory", lambda: None),
                patch("app.qwen3_backend.align", MagicMock()),
            ):
                aligned = pipeline.align(np.zeros(1600, dtype=np.float32), result_in)
        finally:
            pipeline.whispermlx = original_wmlx
        assert aligned.get("segment_aligned") is True
        wmlx.align.assert_called_once()

    def test_text_only_result_uses_qwen_aligner(self):
        result_in = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "only text"}],
            "language": "en",
            "_asr_backend": "external",
            "_qwen_align": True,
        }
        qwen_align = MagicMock(return_value={**result_in, "qwen_aligned": True})
        with (
            patch("app.qwen3_backend.align", qwen_align),
            patch.object(pipeline, "clear_gpu_memory", lambda: None),
        ):
            aligned = pipeline.align(np.zeros(1600, dtype=np.float32), result_in)
        assert aligned.get("qwen_aligned") is True
        qwen_align.assert_called_once()

    def test_run_pipeline_strips_qwen_align_tag(self):
        transcribe_result = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello", "words": []}],
            "language": "en",
            "_asr_backend": "external",
            "_qwen_align": True,
        }
        aligned = {**transcribe_result, "word_segments": []}
        with (
            patch.object(pipeline, "transcribe", return_value=transcribe_result),
            patch.object(pipeline, "align", return_value=aligned),
            patch.object(pipeline, "diarize", return_value=(aligned, None)),
        ):
            result, _ = pipeline.run_pipeline(np.zeros(1600, dtype=np.float32), word_timestamps=True, should_diarize=True)
        assert result.get("_qwen_align") is None
        assert result.get("_asr_backend") is None
        assert result.get("_language_name") is None
