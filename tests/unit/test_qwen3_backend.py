"""
Unit tests for the optional Qwen3-ASR backend (ASR_BACKEND=qwen3).

Ported from upstream 90b093d (v0.4.0). All transformer models are mocked:
these tests exercise the routing and reshaping logic in app.pipeline
(ASR_BACKEND selection, align/diarize hand-off, internal-tag stripping in
run_pipeline) plus the local qwen3_backend chunking/alignment bookkeeping.
No model downloads occur.
"""

from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import torch

import app.pipeline as pipeline
import app.qwen3_backend as qwen3_backend

# ---------------------------------------------------------------------------
# Fake processor inputs
# ---------------------------------------------------------------------------


class _FakeBatch(dict):
    """Minimal BatchFeature stand-in: dict-like for **inputs, .to() chains."""

    def __init__(self, input_ids):
        super().__init__(input_ids=input_ids)
        self._input_ids = input_ids

    def to(self, *args, **kwargs):
        return self


# ---------------------------------------------------------------------------
# qwen3_backend internals
# ---------------------------------------------------------------------------


class TestQwen3LanguageHelpers:
    def test_language_name_none(self):
        assert qwen3_backend._language_name(None) is None
        assert qwen3_backend._language_name("") is None

    def test_language_name_resolves_code(self):
        assert qwen3_backend._language_name("en") == "English"

    def test_language_name_unknown_returns_none(self):
        assert qwen3_backend._language_name("not-a-language") is None

    def test_language_code_resolves_name(self):
        assert qwen3_backend._language_code("English") == "en"

    def test_language_code_none(self):
        assert qwen3_backend._language_code(None) is None


class TestQwen3Chunks:
    def test_splits_audio_into_fixed_chunks(self):
        audio = np.zeros(int(3.5 * qwen3_backend.SAMPLE_RATE), dtype=np.float32)
        with patch.object(qwen3_backend, "CHUNK_SECONDS", 1):
            chunks = qwen3_backend._chunks(audio)
        assert len(chunks) == 4  # 3.5s / 1s
        assert chunks[0][0] == 0.0
        assert isinstance(chunks[0][1], np.ndarray)

    def test_short_audio_single_chunk(self):
        audio = np.zeros(qwen3_backend.SAMPLE_RATE // 2, dtype=np.float32)
        chunks = qwen3_backend._chunks(audio)
        assert len(chunks) == 1
        assert chunks[0][0] == 0.0


class TestQwen3DefaultDevice:
    def test_automatic_choice_without_env(self):
        if torch.cuda.is_available():
            expected = "cuda"
        elif torch.backends.mps.is_available():
            expected = "mps"
        else:
            expected = "cpu"
        with patch.dict("os.environ", {"DEVICE": ""}):
            assert qwen3_backend._default_device() == expected


class TestQwen3Transcribe:
    @staticmethod
    def _mock_asr(decoded_batches):
        processor = MagicMock()
        processor.apply_transcription_request.return_value = _FakeBatch(torch.ones((1, 5), dtype=torch.long))
        processor.decode.side_effect = [dict(d) for d in decoded_batches]
        model = MagicMock()
        model.device = "cpu"
        model.dtype = "float32"
        model.generate.return_value = torch.ones((1, 10), dtype=torch.long)
        return processor, model

    def test_segments_built_per_chunk(self):
        processor, model = self._mock_asr(
            [
                {"transcription": "one", "language": "English"},
                {"transcription": "two", "language": "English"},
                {"transcription": "three", "language": "English"},
            ]
        )
        audio = np.zeros(int(2.5 * qwen3_backend.SAMPLE_RATE), dtype=np.float32)
        with (
            patch.object(qwen3_backend, "_load_asr", return_value=(processor, model)),
            patch.object(qwen3_backend, "CHUNK_SECONDS", 1),
        ):
            result = qwen3_backend.transcribe(audio, language="en")
        assert len(result["segments"]) == 3
        assert result["segments"][0] == {"start": 0.0, "end": 1.0, "text": "one"}
        assert result["segments"][2] == {"start": 2.0, "end": 2.5, "text": "three"}
        assert result["_asr_backend"] == "qwen3"
        assert result["_language_name"] == "English"
        assert result["language"] == "en"

    def test_empty_chunks_dropped(self):
        processor, model = self._mock_asr(
            [
                {"transcription": "", "language": None},
                {"transcription": "spoken", "language": "English"},
            ]
        )
        audio = np.zeros(int(1.5 * qwen3_backend.SAMPLE_RATE), dtype=np.float32)
        with (
            patch.object(qwen3_backend, "_load_asr", return_value=(processor, model)),
            patch.object(qwen3_backend, "CHUNK_SECONDS", 1),
        ):
            result = qwen3_backend.transcribe(audio, language="en")
        assert len(result["segments"]) == 1
        assert result["segments"][0]["text"] == "spoken"

    def test_context_passed_via_system_prompt(self):
        processor, model = self._mock_asr([{"transcription": "hi", "language": "English"}])
        audio = np.zeros(int(0.5 * qwen3_backend.SAMPLE_RATE), dtype=np.float32)
        with (
            patch.object(qwen3_backend, "_load_asr", return_value=(processor, model)),
            patch.object(qwen3_backend, "DEFAULT_CONTEXT", "standing ctx"),
        ):
            qwen3_backend.transcribe(audio, language="en", context="user ctx")
        _, kwargs = processor.apply_transcription_request.call_args
        assert kwargs["prompt"] == "standing ctx\nuser ctx"
        assert kwargs["language"] == "English"

    def test_no_language_leaves_prompt_none(self):
        processor, model = self._mock_asr([{"transcription": "hi", "language": "English"}])
        audio = np.zeros(int(0.5 * qwen3_backend.SAMPLE_RATE), dtype=np.float32)
        with patch.object(qwen3_backend, "_load_asr", return_value=(processor, model)):
            result = qwen3_backend.transcribe(audio, language=None)
        _, kwargs = processor.apply_transcription_request.call_args
        assert kwargs["prompt"] is None
        assert kwargs["language"] is None
        # language falls back to "en" when the parsed language is unknown
        assert result["language"] == "en"


class TestQwen3Align:
    def test_word_timestamps_offset_by_chunk_start(self):
        processor = MagicMock()
        timestamps = [
            {"text": "hello", "start_time": 0.0, "end_time": 0.4},
            {"text": "world.", "start_time": 0.5, "end_time": 0.8},
        ]
        processor.prepare_forced_aligner_inputs.return_value = (MagicMock(), [["hello", "world"]])
        processor.decode_forced_alignment.return_value = [timestamps]
        model = MagicMock()
        model.config.timestamp_token_id = 7

        result = {
            "_asr_backend": "qwen3",
            "_language_name": "English",
            "language": "en",
            "segments": [{"start": 1.0, "end": 2.0, "text": "hello world."}],
        }
        audio = np.zeros(3 * qwen3_backend.SAMPLE_RATE, dtype=np.float32)

        with (
            patch.object(qwen3_backend, "_load_aligner", return_value=(processor, model)),
            patch.object(pipeline, "_words_to_segments", side_effect=lambda words: [{"words": words}]),
        ):
            out = qwen3_backend.align(audio, result)

        assert out["word_segments"] == [
            {"word": "hello", "start": 1.0, "end": 1.4},
            {"word": "world.", "start": 1.5, "end": 1.8},
        ]

    def test_align_failure_keeps_chunk_text(self):
        """A failed chunk keeps its text as one un-timed span, never dropped."""
        processor = MagicMock()
        processor.prepare_forced_aligner_inputs.side_effect = RuntimeError("boom")
        model = MagicMock()

        result = {
            "_asr_backend": "qwen3",
            "_language_name": None,
            "language": "en",
            "segments": [{"start": 0.0, "end": 1.0, "text": "kept text"}],
        }
        audio = np.zeros(qwen3_backend.SAMPLE_RATE, dtype=np.float32)

        with (
            patch.object(qwen3_backend, "_load_aligner", return_value=(processor, model)),
            patch.object(
                pipeline,
                "_words_to_segments",
                side_effect=lambda words: [
                    {
                        "start": words[0]["start"],
                        "end": words[0]["end"],
                        "text": " ".join(w["word"] for w in words),
                        "words": words,
                    }
                ],
            ),
        ):
            out = qwen3_backend.align(audio, result)

        assert len(out["segments"]) == 1
        assert out["segments"][0]["text"] == "kept text"
        assert out["segments"][0]["start"] == 0.0
        assert out["segments"][0]["end"] == 1.0


# ---------------------------------------------------------------------------
# app.pipeline routing (ASR_BACKEND=qwen3)
# ---------------------------------------------------------------------------


class TestTranscribeRouting:
    def test_transcribe_routes_to_qwen3_with_combined_context(self):
        q_transcribe = MagicMock(return_value={"segments": [], "language": "en", "_asr_backend": "qwen3"})
        with (
            patch.object(pipeline, "ASR_BACKEND", "qwen3"),
            patch("app.qwen3_backend.transcribe", q_transcribe),
        ):
            result = pipeline.transcribe(np.zeros(1600, dtype=np.float32), initial_prompt="ctx", hotwords="hw")
        assert result["language"] == "en"
        q_transcribe.assert_called_once()
        assert q_transcribe.call_args.kwargs.get("context") == "ctx hw"

    def test_translate_falls_back_to_whisper(self):
        wmlx_model = MagicMock()
        wmlx_model.transcribe.return_value = {"segments": [], "language": "en"}
        q_transcribe = MagicMock()
        with (
            patch.object(pipeline, "ASR_BACKEND", "qwen3"),
            patch.object(pipeline, "load_whisper_model", return_value=wmlx_model),
            patch("app.qwen3_backend.transcribe", q_transcribe),
        ):
            result = pipeline.transcribe(np.zeros(1600, dtype=np.float32), task="translate")
        assert result["language"] == "en"
        wmlx_model.transcribe.assert_called_once()
        assert wmlx_model.transcribe.call_args.kwargs.get("task") == "translate"
        q_transcribe.assert_not_called()

    def test_default_backend_uses_whispermlx(self):
        wmlx_model = MagicMock()
        wmlx_model.transcribe.return_value = {"segments": [], "language": "en"}
        q_transcribe = MagicMock()
        with (
            patch.object(pipeline, "ASR_BACKEND", "whisper"),
            patch.object(pipeline, "load_whisper_model", return_value=wmlx_model),
            patch("app.qwen3_backend.transcribe", q_transcribe),
        ):
            result = pipeline.transcribe(np.zeros(1600, dtype=np.float32))
        assert result["language"] == "en"
        q_transcribe.assert_not_called()


class TestAlignRouting:
    def test_align_routes_qwen3_results_to_qwen3_backend(self):
        q_align = MagicMock(return_value={"segments": [], "_asr_backend": "qwen3"})
        with (
            patch("app.qwen3_backend.align", q_align),
            patch.object(pipeline, "clear_gpu_memory", lambda: None),
        ):
            result = pipeline.align(np.zeros(1600, dtype=np.float32), {"segments": [], "_asr_backend": "qwen3"})
        assert result == {"segments": [], "_asr_backend": "qwen3"}
        q_align.assert_called_once()

    def test_align_failure_degrades_gracefully(self):
        q_align = MagicMock(side_effect=RuntimeError("boom"))
        with (
            patch("app.qwen3_backend.align", q_align),
            patch.object(pipeline, "clear_gpu_memory", lambda: None),
        ):
            result = pipeline.align(np.zeros(1600, dtype=np.float32), {"segments": [], "_asr_backend": "qwen3"})
        # original (unaligned) result passes through
        assert result == {"segments": [], "_asr_backend": "qwen3"}

    def test_whisper_results_use_wav2vec2_path(self):
        wmlx = MagicMock()
        wmlx.align.return_value = {"segments": [], "language": "en"}
        original_wmlx = pipeline.whispermlx
        pipeline.whispermlx = wmlx
        try:
            with (
                patch.object(pipeline, "load_align_model", return_value=(MagicMock(), {})),
                patch("app.qwen3_backend.align", MagicMock()),
            ):
                result = pipeline.align(np.zeros(1600, dtype=np.float32), {"segments": [], "language": "en"})
        finally:
            pipeline.whispermlx = original_wmlx
        assert result["language"] == "en"
        wmlx.align.assert_called_once()


# ---------------------------------------------------------------------------
# run_pipeline shape handling
# ---------------------------------------------------------------------------


class TestRunPipelineShape:
    def test_qwen3_word_timestamps_disabled_forces_align_then_strips(self):
        transcribe_result = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello",
                    "words": [{"word": "hello", "start": 0.0, "end": 1.0}],
                }
            ],
            "word_segments": [],
            "language": "en",
            "_asr_backend": "qwen3",
            "_language_name": "English",
        }
        aligned = {
            **transcribe_result,
            "word_segments": transcribe_result["segments"][0]["words"],
        }
        with (
            patch.object(pipeline, "transcribe", return_value=transcribe_result),
            patch.object(pipeline, "align", return_value=aligned),
            patch.object(pipeline, "diarize", return_value=(aligned, None)) as diarize_mock,
        ):
            result, _ = pipeline.run_pipeline(
                np.zeros(1600, dtype=np.float32), word_timestamps=False, should_diarize=False
            )
        diarize_mock.assert_not_called()  # should_diarize=False
        assert result.get("_asr_backend") is None
        assert result.get("_language_name") is None
        assert all("words" not in seg for seg in result.get("segments", []))

    def test_qwen3_forces_align_when_diarizing_without_words(self):
        transcribe_result = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
            "language": "en",
            "_asr_backend": "qwen3",
        }
        aligned = {**transcribe_result, "word_segments": []}
        with (
            patch.object(pipeline, "transcribe", return_value=transcribe_result),
            patch.object(pipeline, "align", return_value=aligned) as align_mock,
            patch.object(pipeline, "diarize", return_value=({**transcribe_result}, None)),
        ):
            result, _ = pipeline.run_pipeline(
                np.zeros(1600, dtype=np.float32), word_timestamps=False, should_diarize=True
            )
        align_mock.assert_called_once()
        assert result.get("_asr_backend") is None
        assert result.get("_language_name") is None

    def test_whisper_backend_internal_tags_stripped(self):
        transcribe_result = {"segments": [], "language": "en", "_asr_backend": "whisper", "_language_name": "English"}
        with (
            patch.object(pipeline, "transcribe", return_value=transcribe_result),
            patch.object(pipeline, "align", return_value=transcribe_result),
            patch.object(pipeline, "diarize", return_value=(transcribe_result, None)),
        ):
            result, _ = pipeline.run_pipeline(
                np.zeros(1600, dtype=np.float32), word_timestamps=True, should_diarize=True
            )
        assert result.get("_asr_backend") is None
        assert result.get("_language_name") is None


# ---------------------------------------------------------------------------
# diarize() routing
# ---------------------------------------------------------------------------


class TestDiarizeQwen3Resegmentation:
    @staticmethod
    def _pipeline_factory(df):
        class _FakeDiarizePipeline:
            def __call__(self, audio, **kwargs):
                return df

        return MagicMock(return_value=_FakeDiarizePipeline())

    def _run(self, result_in, resegment_by_speaker=False):
        df = MagicMock(spec=["iterrows"])
        df.iterrows.return_value = iter([])

        wmlx = MagicMock()

        def capture(diarize_df, result, fill_nearest=False):
            return result

        wmlx.assign_word_speakers.side_effect = capture

        original_wmlx, original_token = pipeline.whispermlx, pipeline.HF_TOKEN
        pipeline.whispermlx = wmlx
        pipeline.HF_TOKEN = "fake-token"
        pipeline._diarize_pipeline = None
        try:
            with (
                patch.object(pipeline, "DiarizationPipeline", self._pipeline_factory(df)),
                patch.object(pipeline, "RESEGMENT_BY_SPEAKER", resegment_by_speaker),
                patch.object(pipeline, "DIARIZE_FILL_NEAREST", False),
                patch.object(pipeline, "_resplit_segments_on_diarization_turns", lambda r, d: r),
            ):
                out, _ = pipeline.diarize(np.zeros(1600, dtype=np.float32), result_in)
        finally:
            pipeline.whispermlx, pipeline.HF_TOKEN = original_wmlx, original_token
            pipeline._diarize_pipeline = None
        return out

    def _multi_speaker_result(self):
        """One coarse qwen3 segment spanning two speakers' turns."""
        return {
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "hello there hi",
                    "words": [
                        {"word": "hello", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
                        {"word": "there", "start": 0.5, "end": 1.0, "speaker": "SPEAKER_00"},
                        {"word": "hi", "start": 1.1, "end": 2.0, "speaker": "SPEAKER_01"},
                    ],
                }
            ],
            "language": "en",
        }

    def test_qwen3_result_always_resegmented(self):
        result = self._run(
            {**self._multi_speaker_result(), "_asr_backend": "qwen3"},
            resegment_by_speaker=False,
        )
        # always-on for the qwen3 backend: two speaker turns, tag retained
        assert len(result["segments"]) == 2
        assert result["segments"][0]["speaker"] == "SPEAKER_00"
        assert result["segments"][1]["speaker"] == "SPEAKER_01"
        assert result.get("_asr_backend") == "qwen3"

    def test_whisper_result_not_resegmented_by_default(self):
        result = self._run(self._multi_speaker_result(), resegment_by_speaker=False)
        # opt-in flag off: the coarse segment shape is unchanged
        assert len(result["segments"]) == 1

    def test_whisper_result_resegmented_with_flag(self):
        result = self._run(self._multi_speaker_result(), resegment_by_speaker=True)
        assert len(result["segments"]) == 2
        assert result["segments"][1]["speaker"] == "SPEAKER_01"
