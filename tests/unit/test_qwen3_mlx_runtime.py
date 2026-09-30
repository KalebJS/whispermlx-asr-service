"""
Unit tests for the MLX execution runtime of the Qwen3-ASR backend
(QWEN3_RUNTIME=mlx).

`mlx_qwen3_asr` (runtime) and `mlx` are fully mocked via sys.modules —
no MLX model weights are downloaded or loaded, and no Metal device work
occurs. The tests verify:

- runtime selection logic in app.qwen3_backend (_use_mlx_runtime: auto /
  mlx / torch, package availability, CUDA fallback)
- delegation from app.qwen3_backend transcribe/align to the MLX runtime
- result shape-building in app.mlx_runtime (words -> segments via
  the shared _words_to_segments, language code resolution, internals)
- the standalone-alignment path for text-only results, including the
  chunk-relative timestamp offset and failure fallbacks
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np

import app.qwen3_backend as qwen3_backend
import app.qwen3_mlx_runtime as mlx_runtime

# ---------------------------------------------------------------------------
# Fixtures/helpers
# ---------------------------------------------------------------------------


def make_pkg(transcribe_result=None, align_result=None):
    """Build a fake mlx_qwen3_asr package mock plus a fake mlx.core.

    mlx.core is mocked too: importing the real MLX extension inside
    pytest alongside the torch stack aborts the interpreter on some
    machines. The runtime only reads mx.float16 / mx.Dtype from it.
    """
    pkg = MagicMock()
    session = MagicMock()
    if transcribe_result is not None:
        session.transcribe.return_value = transcribe_result
    pkg.Session.return_value = session

    aligner = MagicMock()
    if align_result is not None:
        aligner.align.return_value = align_result
    pkg.ForcedAligner.return_value = aligner
    mlx_core = MagicMock()
    return (
        {
            "mlx_qwen3_asr": pkg,
            "mlx.core": mlx_core,
            "mlx": MagicMock(
                **{
                    "core": mlx_core,
                }
            ),
        },
        session,
        aligner,
    )


def fake_transcription(text="hello world", language="English", segments=None):
    if segments is None:
        segments = [
            {"text": "hello", "start": 0.0, "end": 0.4},
            {"text": "world", "start": 0.5, "end": 0.8},
        ]
    return SimpleNamespace(
        text=text,
        language=language,
        segments=segments,
        chunks=None,
    )


def make_audio(seconds=1.0):
    return np.zeros(int(seconds * 16000), dtype=np.float32)


def reset_runtime_state():
    mlx_runtime._asr_session = None
    mlx_runtime._aligner = None
    qwen3_backend.RUNTIME = "auto"
    qwen3_backend._mlx_package_available.cache_clear()


# ---------------------------------------------------------------------------
# Runtime selection
# ---------------------------------------------------------------------------


class TestRuntimeSelection:
    def setup_method(self):
        reset_runtime_state()

    def test_explicit_torch_wins_even_with_package(self):
        with (
            patch.dict(sys.modules, {"mlx_qwen3_asr": MagicMock()}),
            patch.object(qwen3_backend, "RUNTIME", "torch"),
        ):
            assert qwen3_backend._use_mlx_runtime() is False

    def test_explicit_mlx_wins_even_without_package(self):
        with patch.dict(sys.modules, {"mlx_qwen3_asr": None}), patch.object(qwen3_backend, "RUNTIME", "mlx"):
            assert qwen3_backend._use_mlx_runtime() is True

    def test_auto_uses_mlx_without_cuda_when_package_present(self):
        import torch

        if torch.cuda.is_available():
            return  # auto prefers torch on CUDA; can't test here
        with (
            patch.dict(sys.modules, {"mlx_qwen3_asr": MagicMock()}),
            patch.object(qwen3_backend, "RUNTIME", "auto"),
        ):
            qwen3_backend._mlx_package_available.cache_clear()
            assert qwen3_backend._use_mlx_runtime() is True

    def test_auto_falls_back_to_torch_without_package(self):
        with patch.dict(sys.modules, {"mlx_qwen3_asr": None}), patch.object(qwen3_backend, "RUNTIME", "auto"):
            qwen3_backend._mlx_package_available.cache_clear()
            assert qwen3_backend._use_mlx_runtime() is False


# ---------------------------------------------------------------------------
# app.mlx_runtime.transcribe
# ---------------------------------------------------------------------------


class TestMlxTranscribe:
    def setup_method(self):
        reset_runtime_state()

    def teardown_method(self):
        reset_runtime_state()

    def test_builds_whisperx_shape_from_words(self):
        result = SimpleNamespace(
            text="hello world",
            language="English",
            segments=[
                {"text": "hello", "start": 0.0, "end": 0.4},
                {"text": "world", "start": 0.5, "end": 0.8},
            ],
        )
        modules, _, _ = make_pkg(transcribe_result=result)
        with (
            patch.dict(sys.modules, modules),
            patch.object(mlx_runtime, "DEFAULT_CONTEXT", ""),
        ):
            out = mlx_runtime.transcribe(make_audio(1.0), language="en")

        # words are grouped into one segment (no speaker/gap boundaries)
        assert len(out["segments"]) == 1
        assert out["segments"][0]["start"] == 0.0
        assert out["segments"][0]["end"] == 0.8
        assert "speaker" not in out["segments"][0]  # no speaker labels pre-diarization
        assert out["word_segments"] == [
            {"word": "hello", "start": 0.0, "end": 0.4},
            {"word": "world", "start": 0.5, "end": 0.8},
        ]
        assert out["segments"][0]["text"] == "hello world"
        assert out["language"] == "en"
        assert out["_asr_backend"] == "qwen3"
        assert out["_asr_runtime"] == "mlx"
        assert out["_language_name"] == "English"

    def test_silence_boundaries_split_words(self):
        """The MLX runtime returns words without speaker labels; grouping
        still splits on silence gaps via the shared logic."""
        result = SimpleNamespace(
            text="first last",
            language="English",
            segments=[
                {"text": "first", "start": 0.0, "end": 0.5},
                {"text": "last", "start": 2.0, "end": 2.5},
            ],
        )
        modules, _, _ = make_pkg(transcribe_result=result)
        with patch.dict(sys.modules, modules):
            out = mlx_runtime.transcribe(make_audio(3.0), language="en")
        assert len(out["segments"]) == 2
        assert out["segments"][0]["text"] == "first"
        assert out["segments"][1]["text"] == "last"

    def test_empty_words_falls_back_to_span(self):
        result = SimpleNamespace(text="only text", language="English", segments=[])
        modules, _, _ = make_pkg(transcribe_result=result)
        with patch.dict(sys.modules, modules):
            out = mlx_runtime.transcribe(make_audio(2.0), language=None)
        assert out["segments"] == [{"start": 0.0, "end": 2.0, "text": "only text"}]
        assert out["language"] == "en"  # fallback when code lookup misses

    def test_language_passed_canonicalized(self):
        result = SimpleNamespace(text="hola", language="Spanish", segments=[])
        modules, session, _ = make_pkg(transcribe_result=result)
        with patch.dict(sys.modules, modules):
            out = mlx_runtime.transcribe(make_audio(0.5), language="es")
        session.transcribe.assert_called_once()
        kwargs = session.transcribe.call_args.kwargs
        assert kwargs["language"] == "Spanish"
        assert out["_language_name"] == "Spanish"

    def test_context_composition(self):
        result = SimpleNamespace(text="x", language="English", segments=[])
        modules, session, _ = make_pkg(transcribe_result=result)
        with (
            patch.dict(sys.modules, modules),
            patch.object(mlx_runtime, "DEFAULT_CONTEXT", "standing"),
        ):
            mlx_runtime.transcribe(make_audio(0.5), context="user ctx")
        assert session.transcribe.call_args.kwargs["context"] == "standing\nuser ctx"

    def test_prompt_empty_without_context(self):
        result = SimpleNamespace(text="x", language="English", segments=[])
        modules, session, _ = make_pkg(transcribe_result=result)
        with (
            patch.dict(sys.modules, modules),
            patch.object(mlx_runtime, "DEFAULT_CONTEXT", ""),
        ):
            mlx_runtime.transcribe(make_audio(0.5))
        assert session.transcribe.call_args.kwargs["context"] == ""


# ---------------------------------------------------------------------------
# app.mlx_runtime.align
# ---------------------------------------------------------------------------


class TestMlxAlign:
    def setup_method(self):
        reset_runtime_state()

    def teardown_method(self):
        reset_runtime_state()

    def test_passthrough_when_words_present(self):
        modules, _, aligner = make_pkg()
        result = {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hello", "words": [{"word": "hello", "start": 0.0, "end": 0.5}]}
            ]
        }
        with patch.dict(sys.modules, modules):
            out = mlx_runtime.align(make_audio(1.0), result)
        assert out is result
        aligner.align.assert_not_called()

    def test_standalone_aligner_with_offsets(self):
        aligned_words = [
            SimpleNamespace(text="only", start_time=0.1, end_time=0.4),
            SimpleNamespace(text="text", start_time=0.5, end_time=0.7),
        ]
        modules, _, aligner = make_pkg(align_result=aligned_words)
        result = {
            "segments": [{"start": 2.0, "end": 3.0, "text": "only text"}],
            "language": "en",
            "_qwen_align": True,
        }
        with (
            patch.dict(sys.modules, modules),
            patch.object(mlx_runtime, "ALIGNER_MODEL_ID", "aligner/id"),
        ):
            out = mlx_runtime.align(make_audio(4.0), result)
        chunk = aligner.align.call_args.args[0]
        assert chunk.shape[0] == int(1.0 * 16000)
        assert aligner.align.call_args.args[1] == "only text"
        assert out["word_segments"] == [
            {"word": "only", "start": 2.1, "end": 2.4},
            {"word": "text", "start": 2.5, "end": 2.7},
        ]

    def test_align_failure_keeps_chunk_text(self):
        """A failing chunk's text rides along un-timed instead of being dropped."""
        modules, _, aligner = make_pkg()
        aligned_words = [SimpleNamespace(text="hi", start_time=0.1, end_time=0.3)]
        aligner.align.side_effect = [aligned_words, RuntimeError("boom")]
        result = {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hi"},
                {"start": 1.0, "end": 2.0, "text": "kept"},
            ],
            "language": "en",
            "_qwen_align": True,
        }
        with (
            patch.dict(sys.modules, modules),
            patch.object(mlx_runtime, "ALIGNER_MODEL_ID", "aligner/id"),
        ):
            out = mlx_runtime.align(make_audio(3.0), result)
        # The failed chunk joins the previous aligned segment's text (shared
        # logic in _words_to_segments); nothing is silently dropped.
        assert len(out["segments"]) == 1
        assert out["segments"][0]["text"] == "hi kept"
        assert out["word_segments"][0] == {"word": "hi", "start": 0.1, "end": 0.3}


# ---------------------------------------------------------------------------
# Routing in app.qwen3_backend
# ---------------------------------------------------------------------------


class TestQwen3BackendRouting:
    def setup_method(self):
        reset_runtime_state()

    def teardown_method(self):
        reset_runtime_state()

    def test_transcribe_delegates_to_mlx(self):
        result = SimpleNamespace(text="hello", language="English", segments=[])
        modules, _, _ = make_pkg(transcribe_result=result)
        with (
            patch.dict(sys.modules, modules),
            patch.object(qwen3_backend, "RUNTIME", "mlx"),
        ):
            out = qwen3_backend.transcribe(make_audio(0.5), language="en")
        assert out["_asr_runtime"] == "mlx"
        assert out["_asr_backend"] == "qwen3"

    def test_transcribe_torch_path_untouched(self):
        modules, session, _ = make_pkg()
        try:
            with (
                patch.dict(sys.modules, modules),
                patch.object(qwen3_backend, "RUNTIME", "torch"),
                patch.object(
                    qwen3_backend,
                    "_load_asr",
                    return_value=(MagicMock(), MagicMock()),
                ),
            ):
                qwen3_backend.transcribe(make_audio(0.5), language="en")
        finally:
            reset_runtime_state()
        # The torch path never touches the MLX session
        session.transcribe.assert_not_called()

    def test_align_delegates_to_mlx(self):
        aligned_words = [SimpleNamespace(text="hi", start_time=0.0, end_time=0.2)]
        modules, _, _ = make_pkg(align_result=aligned_words)
        with (
            patch.dict(sys.modules, modules),
            patch.object(qwen3_backend, "RUNTIME", "mlx"),
            patch.object(mlx_runtime, "ALIGNER_MODEL_ID", "aligner/id"),
        ):
            out = qwen3_backend.align(
                make_audio(2.0),
                {"segments": [{"start": 0.0, "end": 1.0, "text": "hi"}], "language": "en", "_qwen_align": True},
            )
        assert out["word_segments"] == [{"word": "hi", "start": 0.0, "end": 0.2}]
