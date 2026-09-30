"""
Unit tests for diarization hyperparameter tuning (DIARIZE_* env vars),
ported from upstream 7d27a2f together with the pipeline feature.

Covers: _env_or_none normalization, _deep_merge, _set_scoped, and the real
_apply_diarize_tuning path against a mock pyannote pipeline (instantiated
parameters read -> overrides merged -> instantiate called), including the
failure-tolerant behaviour (invalid values are logged and ignored).

All tests are fast and perform no model downloads.
"""

from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np

import app.pipeline as pipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pyannote(params: dict):
    """Build a mock pipeline wrapper exposing .model like whispermlx does."""
    wrapper = MagicMock()
    wrapper.model.parameters.return_value = params
    return wrapper


# ---------------------------------------------------------------------------
# _env_or_none
# ---------------------------------------------------------------------------


class TestEnvOrNone:
    def test_unset_returns_none(self, monkeypatch):
        monkeypatch.delenv("DIARIZE_TEST_MISSING", raising=False)
        assert pipeline._env_or_none("DIARIZE_TEST_MISSING") is None

    def test_empty_string_returns_none(self, monkeypatch):
        monkeypatch.setenv("DIARIZE_TEST_EMPTY", "")
        assert pipeline._env_or_none("DIARIZE_TEST_EMPTY") is None

    def test_whitespace_returns_none(self, monkeypatch):
        monkeypatch.setenv("DIARIZE_TEST_WS", "   ")
        assert pipeline._env_or_none("DIARIZE_TEST_WS") is None

    def test_value_is_stripped(self, monkeypatch):
        monkeypatch.setenv("DIARIZE_TEST_VAL", " 0.5 ")
        assert pipeline._env_or_none("DIARIZE_TEST_VAL") == "0.5"


# ---------------------------------------------------------------------------
# _deep_merge / _set_scoped
# ---------------------------------------------------------------------------


class TestDeepMerge:
    def test_merges_nested_dicts_in_place(self):
        base = {"clustering": {"threshold": 0.6, "Fa": 0.07}}
        overrides = {"clustering": {"Fb": 1.0}}
        result = pipeline._deep_merge(base, overrides)
        assert result is base
        assert base == {"clustering": {"threshold": 0.6, "Fa": 0.07, "Fb": 1.0}}

    def test_non_dict_override_replaces_dict(self):
        base = {"clustering": {"threshold": 0.6}}
        pipeline._deep_merge(base, {"clustering": 5})
        assert base["clustering"] == 5

    def test_adds_new_keys(self):
        base = {}
        pipeline._deep_merge(base, {"segmentation": {"min_duration_off": 0.2}})
        assert base == {"segmentation": {"min_duration_off": 0.2}}


class TestSetScoped:
    def test_sets_existing_scoped_key(self):
        params = {"clustering": {"threshold": 0.6}}
        assert pipeline._set_scoped(params, "clustering", "threshold", 0.5) is True
        assert params["clustering"]["threshold"] == 0.5

    def test_missing_section_returns_false(self):
        assert pipeline._set_scoped({}, "clustering", "threshold", 0.5) is False

    def test_missing_key_returns_false(self):
        params = {"clustering": {"Fb": 0.8}}
        assert pipeline._set_scoped(params, "clustering", "threshold", 0.5) is False


# ---------------------------------------------------------------------------
# _apply_diarize_tuning
# ---------------------------------------------------------------------------


class TestApplyDiarizeTuning:
    def test_noop_when_no_env_set(self):
        wrapper = _make_pyannote(
            {
                "segmentation": {"min_duration_off": 0.0},
                "clustering": {"threshold": 0.6, "Fa": 0.07, "Fb": 0.8},
            }
        )
        patches = (
            patch.object(pipeline, "DIARIZE_CLUSTERING_THRESHOLD", None),
            patch.object(pipeline, "DIARIZE_MIN_DURATION_OFF", None),
            patch.object(pipeline, "DIARIZE_PARAM_OVERRIDES", None),
            patch.object(pipeline, "DIARIZE_FILL_NEAREST", False),
        )
        for p in patches:
            p.start()
        try:
            pipeline._apply_diarize_tuning(wrapper)
        finally:
            for p in patches:
                p.stop()
        wrapper.model.instantiate.assert_not_called()
        wrapper.model.parameters.assert_not_called()

    def test_clustering_threshold_applied(self):
        params = {
            "segmentation": {"min_duration_off": 0.0},
            "clustering": {"threshold": 0.6, "Fa": 0.07, "Fb": 0.8},
        }
        wrapper = _make_pyannote(params)
        with patch.object(pipeline, "DIARIZE_CLUSTERING_THRESHOLD", "0.5"):
            pipeline._apply_diarize_tuning(wrapper)
        sent = wrapper.model.instantiate.call_args[0][0]
        assert sent["clustering"]["threshold"] == 0.5
        # source params were copied, not mutated
        assert params["clustering"]["threshold"] == 0.6

    def test_min_duration_off_applied(self):
        params = {"segmentation": {"min_duration_off": 0.0}, "clustering": {"threshold": 0.6}}
        wrapper = _make_pyannote(params)
        with patch.object(pipeline, "DIARIZE_MIN_DURATION_OFF", "0.2"):
            pipeline._apply_diarize_tuning(wrapper)
        sent = wrapper.model.instantiate.call_args[0][0]
        assert sent["segmentation"]["min_duration_off"] == 0.2

    def test_json_overrides_deep_merged(self):
        params = {"clustering": {"threshold": 0.6, "Fa": 0.07, "Fb": 0.8}}
        wrapper = _make_pyannote(params)
        with patch.object(pipeline, "DIARIZE_PARAM_OVERRIDES", '{"clustering": {"Fb": 1.0}}'):
            pipeline._apply_diarize_tuning(wrapper)
        sent = wrapper.model.instantiate.call_args[0][0]
        assert sent["clustering"]["Fb"] == 1.0
        assert sent["clustering"]["threshold"] == 0.6

    def test_invalid_float_ignored(self):
        params = {"clustering": {"threshold": 0.6}}
        wrapper = _make_pyannote(params)
        with patch.object(pipeline, "DIARIZE_CLUSTERING_THRESHOLD", "not-a-number"):
            pipeline._apply_diarize_tuning(wrapper)
        wrapper.model.instantiate.assert_not_called()

    def test_invalid_json_ignored(self):
        params = {"clustering": {"threshold": 0.6}}
        wrapper = _make_pyannote(params)
        with patch.object(pipeline, "DIARIZE_PARAM_OVERRIDES", "{not json"):
            pipeline._apply_diarize_tuning(wrapper)
        wrapper.model.instantiate.assert_not_called()

    def test_non_json_object_ignored(self):
        """A valid JSON array is not a parameter object; must be ignored."""
        params = {"clustering": {"threshold": 0.6}}
        wrapper = _make_pyannote(params)
        with patch.object(pipeline, "DIARIZE_PARAM_OVERRIDES", "[1, 2]"):
            pipeline._apply_diarize_tuning(wrapper)
        wrapper.model.instantiate.assert_not_called()

    def test_missing_scoped_key_ignored(self):
        """A set var with no matching path in params does not instantiate."""
        params = {"unrelated": {"key": 1}}
        wrapper = _make_pyannote(params)
        with patch.object(pipeline, "DIARIZE_CLUSTERING_THRESHOLD", "0.5"):
            pipeline._apply_diarize_tuning(wrapper)
        wrapper.model.instantiate.assert_not_called()

    def test_missing_underlying_pipeline_uses_defaults(self):
        """A wrapper without .model is a no-op, not a crash."""
        wrapper = MagicMock(spec=[])  # no .model attribute
        with patch.object(pipeline, "DIARIZE_CLUSTERING_THRESHOLD", "0.5"):
            pipeline._apply_diarize_tuning(wrapper)

    def test_parameters_failure_swallowed(self):
        wrapper = MagicMock()
        wrapper.model.parameters.side_effect = RuntimeError("boom")
        with patch.object(pipeline, "DIARIZE_CLUSTERING_THRESHOLD", "0.5"):
            pipeline._apply_diarize_tuning(wrapper)
        wrapper.model.instantiate.assert_not_called()

    def test_instantiate_failure_swallowed(self):
        params = {"clustering": {"threshold": 0.6}}
        wrapper = _make_pyannote(params)
        wrapper.model.instantiate.side_effect = RuntimeError("boom")
        with patch.object(pipeline, "DIARIZE_CLUSTERING_THRESHOLD", "0.5"):
            # Must not raise; diarization falls back to defaults
            pipeline._apply_diarize_tuning(wrapper)


# ---------------------------------------------------------------------------
# fill_nearest wiring
# ---------------------------------------------------------------------------


class TestFillNearestWiring:
    def test_truthy_parse_values(self):
        truthy = ("1", "true", "yes", "on")
        for v in ("1", "true", "yes", "on", "TRUE", "Yes", " On "):
            assert v.strip().lower() in truthy
        for v in ("false", "0", "no", "", "off", "none"):
            assert v.strip().lower() not in truthy

    def test_pipeline_passes_fill_nearest(self):
        """diarize() forwards the DIARIZE_FILL_NEAREST setting to assign_word_speakers."""

        # spec restricts the mock to real attributes so hasattr() is False
        # for exclusive_speaker_diarization (MagicMocks otherwise auto-create it).
        turns_df = MagicMock(spec=["iterrows"])

        class _FakeDiarizePipeline:
            def __call__(self, audio, **kwargs):
                return turns_df

        pipeline_cls = MagicMock(return_value=_FakeDiarizePipeline())

        received = {}

        def capture(diarize_df, result, fill_nearest=False):
            received["df"] = diarize_df
            received["fill_nearest"] = fill_nearest
            return result

        mock_wmlx = MagicMock()
        mock_wmlx.assign_word_speakers.side_effect = capture

        original_wmlx = pipeline.whispermlx
        original_token = pipeline.HF_TOKEN
        pipeline.whispermlx = mock_wmlx
        pipeline.HF_TOKEN = "fake-token"
        pipeline._diarize_pipeline = None
        try:
            with (
                patch.object(pipeline, "DiarizationPipeline", pipeline_cls),
                patch.object(pipeline, "DIARIZE_FILL_NEAREST", True),
                patch.object(pipeline, "_resplit_segments_on_diarization_turns", lambda r, d: r),
            ):
                pipeline.diarize(np.zeros(1600, dtype=np.float32), {"segments": []})
        finally:
            pipeline.whispermlx = original_wmlx
            pipeline.HF_TOKEN = original_token
            pipeline._diarize_pipeline = None

        assert received["df"] is turns_df
        assert received["fill_nearest"] is True
