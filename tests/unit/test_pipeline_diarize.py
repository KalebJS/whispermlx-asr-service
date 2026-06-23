"""Unit tests for app/pipeline.py diarization stage.

Tests are fast: whispermlx is mocked, no model downloads, no GPU required.
Covers: DiarizationPipeline construction, numpy-fed audio, speaker params,
embeddings, exclusive_speaker_diarization, graceful skip, assign_word_speakers.
"""

import logging
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_model():
    """Create a mock MLXWhisperPipeline that mimics whispermlx.load_model()."""
    model = MagicMock()
    model.transcribe.return_value = {
        "segments": [{"start": 0.0, "end": 1.0, "text": "hello world"}],
        "language": "en",
    }
    model.initial_prompt = None
    return model


def _make_diarize_segments():
    """Create a mock diarization output (pyannote Annotation-like)."""
    segments = MagicMock()
    # Simulate the iteration that assign_word_speakers expects
    # by providing a list of speaker turns.
    return segments


def _import_pipeline():
    """Import app.pipeline fresh so module-level patches take effect."""
    import importlib

    import app.pipeline

    importlib.reload(app.pipeline)
    return app.pipeline


# ---------------------------------------------------------------------------
# 1. load_diarize_pipeline — uses token=, not use_auth_token=
# ---------------------------------------------------------------------------


class TestLoadDiarizePipeline:
    """Verify DiarizationPipeline is constructed with token= (not use_auth_token=)."""

    def test_uses_token_not_use_auth_token(self):
        """DiarizationPipeline must be called with token=, not use_auth_token=."""
        pipeline = _import_pipeline()
        import inspect

        source = inspect.getsource(pipeline.load_diarize_pipeline)
        assert "token=" in source, "load_diarize_pipeline must use token= parameter"
        assert "use_auth_token=" not in source, "load_diarize_pipeline must NOT use use_auth_token= (renamed to token=)"

    def test_model_name_is_pyannote_community(self):
        """DiarizationPipeline model must be pyannote/speaker-diarization-community-1."""
        pipeline = _import_pipeline()
        import inspect

        source = inspect.getsource(pipeline.load_diarize_pipeline)
        assert "pyannote/speaker-diarization-community-1" in source, (
            "DiarizationPipeline must use the community-1 model"
        )

    def test_constructs_with_device(self):
        """DiarizationPipeline must be constructed with device=DEVICE."""
        pipeline = _import_pipeline()
        import inspect

        source = inspect.getsource(pipeline.load_diarize_pipeline)
        assert "device=DEVICE" in source, "DiarizationPipeline must pass device=DEVICE"

    def test_actual_construction(self):
        """When loading, the pipeline calls DiarizationPipeline(model_name, token=HF_TOKEN, device=DEVICE)."""
        pipeline = _import_pipeline()
        pipeline._diarize_pipeline = None

        mock_pipeline_instance = MagicMock()
        with patch.object(pipeline, "DiarizationPipeline", return_value=mock_pipeline_instance) as mock_cls:
            result = pipeline.load_diarize_pipeline()
            mock_cls.assert_called_once_with(
                model_name="pyannote/speaker-diarization-community-1",
                token=pipeline.HF_TOKEN,
                device=pipeline.DEVICE,
            )
            assert result is mock_pipeline_instance

    def test_singleton_caching(self):
        """Second call should return the cached pipeline, not construct a new one."""
        pipeline = _import_pipeline()
        pipeline._diarize_pipeline = None

        mock_pipeline_instance = MagicMock()

        with patch.object(pipeline, "DiarizationPipeline", return_value=mock_pipeline_instance) as mock_cls:
            first = pipeline.load_diarize_pipeline()
            second = pipeline.load_diarize_pipeline()
            # Should only construct once
            assert mock_cls.call_count == 1
            assert first is second


# ---------------------------------------------------------------------------
# 2. diarize() — always passes numpy array, never a file path
# ---------------------------------------------------------------------------


class TestDiarizeNumpyFed:
    """Verify diarize() always passes the numpy audio array, never a file path."""

    def test_passes_numpy_array_to_model(self):
        """diarize_model(audio, ...) must receive the numpy array directly."""
        pipeline = _import_pipeline()
        pipeline.HF_TOKEN = "fake-token-for-test"
        pipeline._diarize_pipeline = None

        audio = np.zeros(16000, dtype=np.float32)
        result = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
            "language": "en",
        }

        mock_diarize_model = MagicMock()
        mock_diarize_output = MagicMock()
        mock_diarize_model.return_value = mock_diarize_output

        with (
            patch.object(pipeline, "load_diarize_pipeline", return_value=mock_diarize_model),
            patch.object(pipeline.whispermlx, "assign_word_speakers", return_value=result),
        ):
            pipeline.diarize(audio, result)

        # Verify the diarize model was called with the numpy array
        # as the first positional argument, not a file path string
        call_args = mock_diarize_model.call_args
        first_arg = call_args[0][0]
        assert isinstance(first_arg, np.ndarray), (
            f"First arg to diarize_model must be numpy array, got {type(first_arg)}"
        )
        assert first_arg is audio, "The exact numpy array passed to diarize() must be forwarded to the model"

    def test_no_file_path_in_diarize_call(self):
        """diarize() must never pass a file path string to the model."""
        pipeline = _import_pipeline()
        import inspect

        source = inspect.getsource(pipeline.diarize)
        # Check that no file-path-related logic exists
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"') or stripped.startswith("'"):
                continue
            # The code should not reference temp paths, file paths, or path strings
            assert "audio_path" not in stripped or "audio_path" in stripped and "not " in stripped, (
                f"diarize() should not use audio_path: {stripped}"
            )


# ---------------------------------------------------------------------------
# 3. diarize() — graceful skip when HF_TOKEN missing
# ---------------------------------------------------------------------------


class TestDiarizeGracefulSkip:
    """Verify diarization gracefully skipped when HF_TOKEN missing or diarization fails."""

    def test_skipped_when_hf_token_missing(self):
        """When HF_TOKEN is None, diarize returns result unchanged with no speakers."""
        pipeline = _import_pipeline()
        pipeline.HF_TOKEN = None

        audio = np.zeros(16000, dtype=np.float32)
        result = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
            "language": "en",
        }

        returned_result, embeddings = pipeline.diarize(audio, result)

        # Result should be returned unchanged
        assert returned_result is result, "Result should be returned as-is when HF_TOKEN missing"
        assert embeddings is None, "No embeddings when HF_TOKEN missing"

    def test_skipped_logs_warning_when_hf_token_missing(self, caplog):
        """When HF_TOKEN is None, a warning should be logged."""
        pipeline = _import_pipeline()
        pipeline.HF_TOKEN = None

        audio = np.zeros(16000, dtype=np.float32)
        result = {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}], "language": "en"}

        with caplog.at_level(logging.WARNING):
            pipeline.diarize(audio, result)

        assert any("HF_TOKEN" in record.message and "not set" in record.message.lower() for record in caplog.records), (
            f"Expected HF_TOKEN warning; got: {[r.message for r in caplog.records]}"
        )

    def test_failure_degrades_gracefully(self):
        """When diarization raises an exception, result is preserved without speakers."""
        pipeline = _import_pipeline()
        pipeline.HF_TOKEN = "fake-token"
        pipeline._diarize_pipeline = None

        audio = np.zeros(16000, dtype=np.float32)
        result = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
            "language": "en",
        }

        mock_diarize_model = MagicMock()
        mock_diarize_model.side_effect = RuntimeError("pyannote model error")

        with patch.object(pipeline, "load_diarize_pipeline", return_value=mock_diarize_model):
            returned_result, embeddings = pipeline.diarize(audio, result)

        # Result should still be intact (transcription preserved)
        assert "segments" in returned_result, "Transcription result must be preserved"
        assert returned_result["segments"][0]["text"] == "hello", "Text must be unchanged"
        assert embeddings is None, "No embeddings on diarization failure"

    def test_failure_logs_warning(self, caplog):
        """Diarization failure logs a warning rather than propagating the exception."""
        pipeline = _import_pipeline()
        pipeline.HF_TOKEN = "fake-token"
        pipeline._diarize_pipeline = None

        audio = np.zeros(16000, dtype=np.float32)
        result = {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}], "language": "en"}

        mock_diarize_model = MagicMock()
        mock_diarize_model.side_effect = RuntimeError("pyannote model error")

        with (
            patch.object(pipeline, "load_diarize_pipeline", return_value=mock_diarize_model),
            caplog.at_level(logging.WARNING),
        ):
            pipeline.diarize(audio, result)

        assert any("diarization failed" in record.message.lower() for record in caplog.records), (
            f"Expected diarization failure warning; got: {[r.message for r in caplog.records]}"
        )


# ---------------------------------------------------------------------------
# 4. diarize() — num_speakers, min_speakers, max_speakers parameter passing
# ---------------------------------------------------------------------------


class TestDiarizeSpeakerParams:
    """Verify num/min/max_speakers are passed correctly to the diarization model."""

    def _setup_pipeline(self):
        pipeline = _import_pipeline()
        pipeline.HF_TOKEN = "fake-token"
        pipeline._diarize_pipeline = None
        return pipeline

    def _diarize_result_with_speakers(self, speakers=None):
        """Build a result with speaker labels assigned."""
        if speakers is None:
            speakers = ["SPEAKER_00", "SPEAKER_01"]
        segments = []
        for i, spk in enumerate(speakers):
            segments.append(
                {
                    "start": float(i),
                    "end": float(i + 1),
                    "text": f"Speaker {i}",
                    "speaker": spk,
                }
            )
        return {"segments": segments, "language": "en"}

    def test_num_speakers_passed(self):
        """num_speakers is passed to the diarization model."""
        pipeline = self._setup_pipeline()
        audio = np.zeros(16000, dtype=np.float32)
        result = {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}], "language": "en"}

        mock_diarize_model = MagicMock()
        mock_diarize_model.return_value = MagicMock()

        with (
            patch.object(pipeline, "load_diarize_pipeline", return_value=mock_diarize_model),
            patch.object(pipeline.whispermlx, "assign_word_speakers", return_value=result),
        ):
            pipeline.diarize(audio, result, num_speakers=2)

        call_kwargs = mock_diarize_model.call_args[1]
        assert call_kwargs.get("num_speakers") == 2, f"num_speakers=2 not passed; got kwargs: {call_kwargs}"

    def test_min_max_speakers_passed_when_no_num_speakers(self):
        """min_speakers and max_speakers are passed when num_speakers is not set."""
        pipeline = self._setup_pipeline()
        audio = np.zeros(16000, dtype=np.float32)
        result = {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}], "language": "en"}

        mock_diarize_model = MagicMock()
        mock_diarize_model.return_value = MagicMock()

        with (
            patch.object(pipeline, "load_diarize_pipeline", return_value=mock_diarize_model),
            patch.object(pipeline.whispermlx, "assign_word_speakers", return_value=result),
        ):
            pipeline.diarize(audio, result, min_speakers=2, max_speakers=4)

        call_kwargs = mock_diarize_model.call_args[1]
        assert call_kwargs.get("min_speakers") == 2, f"min_speakers not passed: {call_kwargs}"
        assert call_kwargs.get("max_speakers") == 4, f"max_speakers not passed: {call_kwargs}"
        assert "num_speakers" not in call_kwargs, f"num_speakers should not be passed when not provided: {call_kwargs}"

    def test_num_speakers_overrides_min_max(self):
        """When num_speakers is provided, min/max_speakers are NOT passed."""
        pipeline = self._setup_pipeline()
        audio = np.zeros(16000, dtype=np.float32)
        result = {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}], "language": "en"}

        mock_diarize_model = MagicMock()
        mock_diarize_model.return_value = MagicMock()

        with (
            patch.object(pipeline, "load_diarize_pipeline", return_value=mock_diarize_model),
            patch.object(pipeline.whispermlx, "assign_word_speakers", return_value=result),
        ):
            pipeline.diarize(audio, result, num_speakers=2, min_speakers=4, max_speakers=6)

        call_kwargs = mock_diarize_model.call_args[1]
        assert call_kwargs.get("num_speakers") == 2, f"num_speakers should be 2: {call_kwargs}"
        assert "min_speakers" not in call_kwargs, (
            f"min_speakers should be omitted when num_speakers is set: {call_kwargs}"
        )
        assert "max_speakers" not in call_kwargs, (
            f"max_speakers should be omitted when num_speakers is set: {call_kwargs}"
        )

    def test_no_speaker_params_when_none(self):
        """When no speaker count params are given, none are passed to the model."""
        pipeline = self._setup_pipeline()
        audio = np.zeros(16000, dtype=np.float32)
        result = {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}], "language": "en"}

        mock_diarize_model = MagicMock()
        mock_diarize_model.return_value = MagicMock()

        with (
            patch.object(pipeline, "load_diarize_pipeline", return_value=mock_diarize_model),
            patch.object(pipeline.whispermlx, "assign_word_speakers", return_value=result),
        ):
            pipeline.diarize(audio, result)

        call_kwargs = mock_diarize_model.call_args[1]
        assert "num_speakers" not in call_kwargs, f"num_speakers should not be in kwargs: {call_kwargs}"
        assert "min_speakers" not in call_kwargs, f"min_speakers should not be in kwargs: {call_kwargs}"
        assert "max_speakers" not in call_kwargs, f"max_speakers should not be in kwargs: {call_kwargs}"


# ---------------------------------------------------------------------------
# 5. diarize() — return_embeddings handling
# ---------------------------------------------------------------------------


class TestDiarizeEmbeddings:
    """Verify return_speaker_embeddings parameter handling."""

    def _setup_pipeline(self):
        pipeline = _import_pipeline()
        pipeline.HF_TOKEN = "fake-token"
        pipeline._diarize_pipeline = None
        return pipeline

    def test_return_embeddings_true_adds_to_params(self):
        """return_speaker_embeddings=True adds return_embeddings=True to diarize params."""
        pipeline = self._setup_pipeline()
        audio = np.zeros(16000, dtype=np.float32)
        result = {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}], "language": "en"}

        mock_embeddings = {"SPEAKER_00": [0.1] * 256}
        mock_segments = MagicMock()
        mock_diarize_model = MagicMock(return_value=(mock_segments, mock_embeddings))

        with (
            patch.object(pipeline, "load_diarize_pipeline", return_value=mock_diarize_model),
            patch.object(pipeline.whispermlx, "assign_word_speakers", return_value=result),
        ):
            returned_result, returned_embeddings = pipeline.diarize(audio, result, return_speaker_embeddings=True)

        call_kwargs = mock_diarize_model.call_args[1]
        assert call_kwargs.get("return_embeddings") is True, f"return_embeddings should be True: {call_kwargs}"
        assert returned_embeddings == mock_embeddings, "Embeddings should be returned from the diarize function"

    def test_return_embeddings_false_not_in_params(self):
        """return_speaker_embeddings=False does not add return_embeddings to params."""
        pipeline = self._setup_pipeline()
        audio = np.zeros(16000, dtype=np.float32)
        result = {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}], "language": "en"}

        mock_segments = MagicMock()
        mock_diarize_model = MagicMock(return_value=mock_segments)

        with (
            patch.object(pipeline, "load_diarize_pipeline", return_value=mock_diarize_model),
            patch.object(pipeline.whispermlx, "assign_word_speakers", return_value=result),
        ):
            returned_result, returned_embeddings = pipeline.diarize(audio, result, return_speaker_embeddings=False)

        call_kwargs = mock_diarize_model.call_args[1]
        assert "return_embeddings" not in call_kwargs, (
            f"return_embeddings should not be in kwargs when False: {call_kwargs}"
        )
        assert returned_embeddings is None, "Embeddings should be None when not requested"

    def test_handles_tuple_output_when_embeddings_requested(self):
        """When return_embeddings=True, the output tuple is unpacked correctly."""
        pipeline = self._setup_pipeline()
        audio = np.zeros(16000, dtype=np.float32)
        result = {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}], "language": "en"}

        mock_segments = MagicMock()
        mock_embeddings = {
            "SPEAKER_00": [0.1] * 256,
            "SPEAKER_01": [0.2] * 256,
        }
        mock_diarize_model = MagicMock(return_value=(mock_segments, mock_embeddings))

        with (
            patch.object(pipeline, "load_diarize_pipeline", return_value=mock_diarize_model),
            patch.object(pipeline.whispermlx, "assign_word_speakers", return_value=result),
        ):
            _, returned_embeddings = pipeline.diarize(audio, result, return_speaker_embeddings=True)

        assert returned_embeddings == mock_embeddings, "Should return the exact embeddings dict from the model"

    def test_handles_non_tuple_output_when_no_embeddings(self):
        """When return_embeddings=False, the output is used directly (not unpacked)."""
        pipeline = self._setup_pipeline()
        audio = np.zeros(16000, dtype=np.float32)
        result = {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}], "language": "en"}

        mock_segments = MagicMock()
        mock_diarize_model = MagicMock(return_value=mock_segments)

        with (
            patch.object(pipeline, "load_diarize_pipeline", return_value=mock_diarize_model),
            patch.object(pipeline.whispermlx, "assign_word_speakers", return_value=result),
        ):
            _, returned_embeddings = pipeline.diarize(audio, result, return_speaker_embeddings=False)

        assert returned_embeddings is None, "Embeddings should be None when not requested"


# ---------------------------------------------------------------------------
# 6. diarize() — exclusive_speaker_diarization handling
# ---------------------------------------------------------------------------


class TestExclusiveSpeakerDiarization:
    """Verify exclusive_speaker_diarization attribute is handled."""

    def _setup_pipeline(self):
        pipeline = _import_pipeline()
        pipeline.HF_TOKEN = "fake-token"
        pipeline._diarize_pipeline = None
        return pipeline

    def test_exclusive_speaker_diarization_attribute_used(self):
        """When diarize output has exclusive_speaker_diarization, it is used."""
        pipeline = self._setup_pipeline()
        audio = np.zeros(16000, dtype=np.float32)
        result = {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}], "language": "en"}

        mock_output = MagicMock()
        exclusive = MagicMock()
        mock_output.exclusive_speaker_diarization = exclusive

        mock_diarize_model = MagicMock(return_value=mock_output)

        assign_result = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello", "speaker": "SPEAKER_00"}],
            "language": "en",
        }

        with (
            patch.object(pipeline, "load_diarize_pipeline", return_value=mock_diarize_model),
            patch.object(pipeline.whispermlx, "assign_word_speakers", return_value=assign_result) as mock_assign,
        ):
            pipeline.diarize(audio, result)

        # assign_word_speakers should be called with the exclusive version
        mock_assign.assert_called_once()
        assert mock_assign.call_args[0][0] is exclusive, (
            "assign_word_speakers should receive exclusive_speaker_diarization"
        )

    def test_no_exclusive_attribute_passes_raw_output(self):
        """When diarize output lacks exclusive_speaker_diarization, raw output is passed."""
        pipeline = self._setup_pipeline()
        audio = np.zeros(16000, dtype=np.float32)
        result = {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}], "language": "en"}

        mock_output = MagicMock(spec=[])  # No attributes
        mock_diarize_model = MagicMock(return_value=mock_output)

        assign_result = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello", "speaker": "SPEAKER_00"}],
            "language": "en",
        }

        with (
            patch.object(pipeline, "load_diarize_pipeline", return_value=mock_diarize_model),
            patch.object(pipeline.whispermlx, "assign_word_speakers", return_value=assign_result) as mock_assign,
        ):
            pipeline.diarize(audio, result)

        # assign_word_speakers should be called with the raw output
        mock_assign.assert_called_once()
        assert mock_assign.call_args[0][0] is mock_output, "assign_word_speakers should receive the raw diarize output"


# ---------------------------------------------------------------------------
# 7. diarize() — calls assign_word_speakers
# ---------------------------------------------------------------------------


class TestAssignWordSpeakers:
    """Verify whispermlx.assign_word_speakers is called with correct args."""

    def _setup_pipeline(self):
        pipeline = _import_pipeline()
        pipeline.HF_TOKEN = "fake-token"
        pipeline._diarize_pipeline = None
        return pipeline

    def test_assign_word_speakers_called(self):
        """assign_word_speakers is called with diarize_segments and result."""
        pipeline = self._setup_pipeline()
        audio = np.zeros(16000, dtype=np.float32)
        result = {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}], "language": "en"}

        # Use spec=[] to prevent MagicMock from auto-creating
        # exclusive_speaker_diarization attribute
        mock_diarize_segments = MagicMock(spec=[])
        mock_diarize_model = MagicMock(return_value=mock_diarize_segments)

        with (
            patch.object(pipeline, "load_diarize_pipeline", return_value=mock_diarize_model),
            patch.object(pipeline.whispermlx, "assign_word_speakers", return_value=result) as mock_assign,
        ):
            pipeline.diarize(audio, result)

        mock_assign.assert_called_once_with(mock_diarize_segments, result)

    def test_assign_word_speakers_result_returned(self):
        """The result from assign_word_speakers is returned as the diarized result."""
        pipeline = self._setup_pipeline()
        audio = np.zeros(16000, dtype=np.float32)
        result = {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}], "language": "en"}

        mock_diarize_segments = MagicMock()
        mock_diarize_model = MagicMock(return_value=mock_diarize_segments)

        diarized_result = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello", "speaker": "SPEAKER_00"}],
            "language": "en",
        }

        with (
            patch.object(pipeline, "load_diarize_pipeline", return_value=mock_diarize_model),
            patch.object(pipeline.whispermlx, "assign_word_speakers", return_value=diarized_result),
        ):
            returned_result, _ = pipeline.diarize(audio, result)

        assert returned_result == diarized_result, "Diarized result from assign_word_speakers should be returned"


# ---------------------------------------------------------------------------
# 8. diarize() — uses whispermlx (not whisperx) namespace
# ---------------------------------------------------------------------------


class TestDiarizeNamespace:
    """Verify diarization uses whispermlx namespace, not whisperx."""

    def test_imports_from_whispermlx(self):
        """DiarizationPipeline must be imported from whispermlx.diarize."""
        pipeline = _import_pipeline()
        import inspect

        source = inspect.getsource(pipeline)
        assert "from whispermlx.diarize import DiarizationPipeline" in source, (
            "Must import DiarizationPipeline from whispermlx.diarize"
        )

    def test_no_whisperx_diarize_import(self):
        """Must not import from whisperx.diarize."""
        pipeline = _import_pipeline()
        import inspect

        source = inspect.getsource(pipeline)
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "from whisperx" not in stripped, f"Found whisperx import in pipeline.py: {stripped}"

    def test_assign_word_speakers_from_whispermlx(self):
        """assign_word_speakers must be called via whispermlx, not whisperx."""
        pipeline = _import_pipeline()
        import inspect

        source = inspect.getsource(pipeline.diarize)
        assert "whispermlx.assign_word_speakers" in source, "assign_word_speakers must use whispermlx namespace"


# ---------------------------------------------------------------------------
# 9. run_pipeline — diarization integration
# ---------------------------------------------------------------------------


class TestRunPipelineDiarization:
    """Verify run_pipeline correctly orchestrates diarization."""

    def _setup_pipeline(self):
        pipeline = _import_pipeline()
        pipeline.HF_TOKEN = "fake-token"
        pipeline._diarize_pipeline = None
        pipeline._whisper_models.clear()
        pipeline._whisper_models_last_used.clear()
        pipeline._align_models.clear()
        return pipeline

    def test_diarize_true_runs_diarization(self):
        """run_pipeline(should_diarize=True) calls diarize()."""
        pipeline = self._setup_pipeline()
        audio = np.zeros(16000, dtype=np.float32)

        mock_model = _make_mock_model()
        pipeline._whisper_models["base"] = mock_model

        mock_align_model = MagicMock()
        mock_metadata = {"language": "en", "dictionary": {}, "type": "huggingface"}
        pipeline._align_models["en"] = (mock_align_model, mock_metadata)

        diarized_result = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello", "speaker": "SPEAKER_00"}],
            "language": "en",
        }

        with (
            patch.object(
                pipeline.whispermlx,
                "align",
                return_value={
                    "segments": [{"start": 0.0, "end": 1.0, "text": "hello", "words": []}],
                    "word_segments": [],
                },
            ),
            patch.object(pipeline, "diarize", return_value=(diarized_result, None)) as mock_diarize,
        ):
            result, embeddings = pipeline.run_pipeline(audio, model_name="base", should_diarize=True)

        mock_diarize.assert_called_once()
        assert result["segments"][0].get("speaker") == "SPEAKER_00"

    def test_diarize_false_skips_diarization(self):
        """run_pipeline(should_diarize=False) does NOT call diarize()."""
        pipeline = self._setup_pipeline()
        audio = np.zeros(16000, dtype=np.float32)

        mock_model = _make_mock_model()
        pipeline._whisper_models["base"] = mock_model

        mock_align_model = MagicMock()
        mock_metadata = {"language": "en", "dictionary": {}, "type": "huggingface"}
        pipeline._align_models["en"] = (mock_align_model, mock_metadata)

        with (
            patch.object(
                pipeline.whispermlx,
                "align",
                return_value={
                    "segments": [{"start": 0.0, "end": 1.0, "text": "hello", "words": []}],
                    "word_segments": [],
                },
            ),
            patch.object(pipeline, "diarize") as mock_diarize,
        ):
            result, embeddings = pipeline.run_pipeline(audio, model_name="base", should_diarize=False)

        mock_diarize.assert_not_called()
        assert embeddings is None

    def test_diarize_receives_speaker_params(self):
        """run_pipeline passes num/min/max_speakers and return_speaker_embeddings to diarize."""
        pipeline = self._setup_pipeline()
        audio = np.zeros(16000, dtype=np.float32)

        mock_model = _make_mock_model()
        pipeline._whisper_models["base"] = mock_model

        mock_align_model = MagicMock()
        mock_metadata = {"language": "en", "dictionary": {}, "type": "huggingface"}
        pipeline._align_models["en"] = (mock_align_model, mock_metadata)

        with (
            patch.object(
                pipeline.whispermlx,
                "align",
                return_value={
                    "segments": [{"start": 0.0, "end": 1.0, "text": "hello", "words": []}],
                    "word_segments": [],
                },
            ),
            patch.object(pipeline, "diarize", return_value=({"segments": [], "language": "en"}, None)) as mock_diarize,
        ):
            pipeline.run_pipeline(
                audio,
                model_name="base",
                should_diarize=True,
                num_speakers=2,
                min_speakers=1,
                max_speakers=4,
                return_speaker_embeddings=True,
            )

        call_kwargs = mock_diarize.call_args[1]
        assert call_kwargs["num_speakers"] == 2
        assert call_kwargs["min_speakers"] == 1
        assert call_kwargs["max_speakers"] == 4
        assert call_kwargs["return_speaker_embeddings"] is True

    def test_diarization_runs_after_align(self):
        """In run_pipeline, diarization runs AFTER alignment when both are on."""
        pipeline = self._setup_pipeline()
        audio = np.zeros(16000, dtype=np.float32)

        mock_model = _make_mock_model()
        pipeline._whisper_models["base"] = mock_model

        call_order = []

        original_align = pipeline.align

        def track_align(*args, **kwargs):
            call_order.append("align")
            return original_align(*args, **kwargs)

        original_diarize = pipeline.diarize

        def track_diarize(*args, **kwargs):
            call_order.append("diarize")
            return original_diarize(*args, **kwargs)

        with (
            patch.object(pipeline, "align", side_effect=track_align),
            patch.object(pipeline, "diarize", side_effect=track_diarize) as mock_diarize,
        ):
            mock_diarize.return_value = ({"segments": [], "language": "en"}, None)
            pipeline.run_pipeline(audio, model_name="base", should_diarize=True, word_timestamps=True)

        # Align must come before diarize
        assert call_order == ["align", "diarize"], f"Expected align before diarize; got: {call_order}"

    def test_diarization_runs_even_when_word_timestamps_false(self):
        """Diarization still runs when word_timestamps=false (alignment skipped)."""
        pipeline = self._setup_pipeline()
        audio = np.zeros(16000, dtype=np.float32)

        mock_model = _make_mock_model()
        pipeline._whisper_models["base"] = mock_model

        with (
            patch.object(pipeline, "align") as mock_align,
            patch.object(
                pipeline,
                "diarize",
                return_value=(
                    {
                        "segments": [{"start": 0.0, "end": 1.0, "text": "hello", "speaker": "SPEAKER_00"}],
                        "language": "en",
                    },
                    None,
                ),
            ) as mock_diarize,
        ):
            result, embeddings = pipeline.run_pipeline(
                audio, model_name="base", should_diarize=True, word_timestamps=False
            )

        mock_align.assert_not_called()
        mock_diarize.assert_called_once()
        assert result["segments"][0].get("speaker") == "SPEAKER_00"


# ---------------------------------------------------------------------------
# 10. DiarizationPipeline import — uses token= not use_auth_token=
# ---------------------------------------------------------------------------


class TestDiarizationPipelineImport:
    """Verify the DiarizationPipeline import path is correct."""

    def test_imported_from_whispermlx(self):
        """DiarizationPipeline should be imported from whispermlx.diarize."""
        pipeline = _import_pipeline()
        # Verify the import path
        assert pipeline.DiarizationPipeline is not None

    def test_no_use_auth_token_in_source(self):
        """The entire pipeline.py must not contain use_auth_token."""
        pipeline = _import_pipeline()
        import inspect

        source = inspect.getsource(pipeline)
        assert "use_auth_token" not in source, (
            "use_auth_token must not appear anywhere in pipeline.py (use token= instead)"
        )


# ---------------------------------------------------------------------------
# 11. clear_gpu_memory — no torch.cuda
# ---------------------------------------------------------------------------


class TestDiarizeClearGpuMemory:
    """Verify diarize() uses clear_gpu_memory (MLX/gc, no torch.cuda)."""

    def test_clear_gpu_memory_after_diarization(self):
        """clear_gpu_memory should be called after successful diarization."""
        pipeline = _import_pipeline()
        pipeline.HF_TOKEN = "fake-token"
        pipeline._diarize_pipeline = None

        audio = np.zeros(16000, dtype=np.float32)
        result = {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}], "language": "en"}

        mock_diarize_model = MagicMock(return_value=MagicMock())

        with (
            patch.object(pipeline, "load_diarize_pipeline", return_value=mock_diarize_model),
            patch.object(pipeline.whispermlx, "assign_word_speakers", return_value=result),
            patch.object(pipeline, "clear_gpu_memory") as mock_clear,
        ):
            pipeline.diarize(audio, result)
            mock_clear.assert_called_once()

    def test_clear_gpu_memory_not_called_when_hf_token_missing(self):
        """clear_gpu_memory should NOT be called when HF_TOKEN missing (early return)."""
        pipeline = _import_pipeline()
        pipeline.HF_TOKEN = None

        audio = np.zeros(16000, dtype=np.float32)
        result = {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}], "language": "en"}

        with patch.object(pipeline, "clear_gpu_memory") as mock_clear:
            pipeline.diarize(audio, result)
            mock_clear.assert_not_called()
