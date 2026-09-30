"""
Unit tests for RESEGMENT_BY_SPEAKER (opt-in resegmentation on the aligned path).

Ported from upstream 90b093d (v0.4.0 speaker-turn segmentation) with a local
implementation of the word-based rebuild. The existing word_timestamps=false
resplit (_resplit_segments_on_diarization_turns) is already covered in
test_diarization_segment_resplit.py; these tests cover the word-level path.

Covers: _is_cjk, _join_words, _words_to_segments (split on speaker change,
silence gap, max length, sentence-ending punctuation + small gap; majority
speaker labelling; un-timed words ride along or are appended), and
resegment_by_speaker guards, plus diarize() wiring via the env flag.

All tests are fast and perform no model downloads.
"""

from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np

import app.pipeline as pipeline
from app.pipeline import _is_cjk
from app.pipeline import _join_words
from app.pipeline import _words_to_segments
from app.pipeline import resegment_by_speaker

# ---------------------------------------------------------------------------
# _is_cjk / _join_words
# ---------------------------------------------------------------------------


class TestIsCjk:
    def test_cjk_characters_detected(self):
        assert _is_cjk("中") is True
        assert _is_cjk("你好") is True

    def test_latin_and_punctuation_not_cjk(self):
        assert _is_cjk("hello") is False
        assert _is_cjk("!") is False
        assert _is_cjk("") is False


class TestJoinWords:
    def test_latin_words_spaced(self):
        assert _join_words([{"word": "hello"}, {"word": "world"}]) == "hello world"

    def test_cjk_joined_without_space(self):
        assert _join_words([{"word": "你"}, {"word": "好"}]) == "你好"

    def test_mixed_script(self):
        # CJK adjacent to latin: no space on the CJK side
        assert _join_words([{"word": "hi"}, {"word": "你"}, {"word": "好"}, {"word": "ok"}]) == "hi你好ok"


# ---------------------------------------------------------------------------
# _words_to_segments
# ---------------------------------------------------------------------------


def _w(word, start, end, speaker=None):
    d = {"word": word, "start": start, "end": end}
    if speaker is not None:
        d["speaker"] = speaker
    return d


class TestWordsToSegments:
    def test_empty(self):
        assert _words_to_segments([]) == []

    def test_same_speaker_one_segment(self):
        words = [_w("hello", 0.0, 0.5, "SPEAKER_00"), _w("world", 0.5, 1.0, "SPEAKER_00")]
        segments = _words_to_segments(words)
        assert len(segments) == 1
        assert segments[0]["start"] == 0.0
        assert segments[0]["end"] == 1.0
        assert segments[0]["speaker"] == "SPEAKER_00"
        assert len(segments[0]["words"]) == 2

    def test_speaker_change_splits(self):
        words = [
            _w("hello", 0.0, 0.5, "SPEAKER_00"),
            _w("hi", 0.6, 1.0, "SPEAKER_01"),
        ]
        segments = _words_to_segments(words)
        assert len(segments) == 2
        assert segments[0]["speaker"] == "SPEAKER_00"
        assert segments[1]["speaker"] == "SPEAKER_01"
        assert segments[0]["end"] == 0.5
        assert segments[1]["start"] == 0.6

    def test_silence_gap_splits(self):
        words = [
            _w("first", 0.0, 0.5, "SPEAKER_00"),
            _w("second", 2.0, 2.5, "SPEAKER_00"),  # 1.5s gap > 1.0
        ]
        segments = _words_to_segments(words)
        assert len(segments) == 2

    def test_small_gap_keeps_single_segment(self):
        words = [
            _w("first", 0.0, 0.5, "SPEAKER_00"),
            _w("second", 0.6, 1.0, "SPEAKER_00"),  # 0.1s gap
        ]
        segments = _words_to_segments(words)
        assert len(segments) == 1

    def test_sentence_punctuation_and_gap_splits(self):
        words = [
            _w("Done.", 0.0, 0.5, "SPEAKER_00"),
            _w("Next", 0.8, 1.2, "SPEAKER_00"),  # 0.3s gap after sentence end
        ]
        segments = _words_to_segments(words)
        assert len(segments) == 2

    def test_max_len_splits(self):
        # One 35s-long word run: second word exceeds the 30s max length.
        words = [_w("a", 0.0, 15.0, "SPEAKER_00"), _w("b", 15.0, 40.0, "SPEAKER_00")]
        segments = _words_to_segments(words)
        assert len(segments) == 2

    def test_majority_speaker_wins(self):
        words = [
            _w("a", 0.0, 0.1, "SPEAKER_00"),
            _w("b", 0.1, 0.2, "SPEAKER_00"),
            _w("c", 0.2, 0.3, "SPEAKER_01"),
        ]
        segments = _words_to_segments(words)
        assert segments[0]["speaker"] == "SPEAKER_00"

    def test_untimed_words_ride_along(self):
        # wav2vec2 can skip a token; an un-timed word cannot trigger boundaries.
        words = [
            _w("timed", 0.0, 0.5, "SPEAKER_00"),
            {"word": "skipped", "speaker": "SPEAKER_01"},  # no start/end
        ]
        segments = _words_to_segments(words)
        assert len(segments) == 1
        assert segments[0]["text"] == "timed skipped"

    def test_untimed_only_words_appended_to_previous(self):
        words = [
            _w("timed", 0.0, 0.5, "SPEAKER_00"),
            {"word": "stray", "speaker": "SPEAKER_00"},  # no timestamps
        ]
        segments = _words_to_segments(words)
        assert len(segments) == 1
        assert segments[0]["text"] == "timed stray"

    def test_leading_untimed_words_with_no_timed_following(self):
        """All-untimed words with no previous timed segment are dropped."""
        segments = _words_to_segments([{"word": "lonely", "speaker": "SPEAKER_00"}])
        assert segments == []


# ---------------------------------------------------------------------------
# resegment_by_speaker
# ---------------------------------------------------------------------------


class TestResegmentBySpeaker:
    def test_no_words_returns_unchanged(self):
        result = {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}], "language": "en"}
        out = resegment_by_speaker(result)
        assert out is result
        assert out["segments"] == result["segments"]

    def test_words_without_timestamps_return_unchanged(self):
        result = {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hello", "words": [{"word": "hello", "speaker": "S0"}]}
            ]
        }
        out = resegment_by_speaker(result)
        assert out["segments"][0]["start"] == 0.0
        assert out["segments"][0]["text"] == "hello"

    def test_multi_speaker_segment_rebuilt(self):
        # One coarse segment spanning two speakers' turns.
        result = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "hello there hi",
                    "words": [
                        _w("hello", 0.0, 0.5, "SPEAKER_00"),
                        _w("there", 0.5, 1.0, "SPEAKER_00"),
                        _w("hi", 1.1, 2.0, "SPEAKER_01"),
                    ],
                }
            ]
        }
        out = resegment_by_speaker(result)
        assert len(out["segments"]) == 2
        assert out["segments"][0]["speaker"] == "SPEAKER_00"
        assert out["segments"][1]["speaker"] == "SPEAKER_01"
        assert out["segments"][0]["text"] == "hello there"
        assert out["segments"][1]["text"] == "hi"
        assert out["word_segments"] == [
            _w("hello", 0.0, 0.5, "SPEAKER_00"),
            _w("there", 0.5, 1.0, "SPEAKER_00"),
            _w("hi", 1.1, 2.0, "SPEAKER_01"),
        ]

    def test_single_speaker_stays_one_segment(self):
        words = [_w("one", 0.0, 0.5, "SPEAKER_00"), _w("two", 0.5, 1.0, "SPEAKER_00")]
        result = {"segments": [{"start": 0.0, "end": 1.0, "text": "one two", "words": words}]}
        out = resegment_by_speaker(result)
        assert len(out["segments"]) == 1
        assert out["segments"][0]["text"] == "one two"


# ---------------------------------------------------------------------------
# diarize() wiring via RESEGMENT_BY_SPEAKER
# ---------------------------------------------------------------------------


class TestDiarizeResegmentWiring:
    @staticmethod
    def _pipeline_factory(df):
        class _FakeDiarizePipeline:
            def __call__(self, audio, **kwargs):
                return df

        # MagicMock models the DiarizationPipeline class: construction
        # returns the pipeline instance, which is then called with audio.
        return MagicMock(return_value=_FakeDiarizePipeline())

    def _run_diarize(self, resegment_enabled, assign_implementation):
        """Run the real diarize() with mocks; return the processed result."""
        df = MagicMock(spec=["iterrows"])
        df.iterrows.return_value = iter([])

        mock_wmlx = MagicMock()
        mock_wmlx.assign_word_speakers.side_effect = assign_implementation

        original_wmlx = pipeline.whispermlx
        original_token = pipeline.HF_TOKEN
        pipeline.whispermlx = mock_wmlx
        pipeline.HF_TOKEN = "fake-token"
        pipeline._diarize_pipeline = None
        try:
            with (
                patch.object(pipeline, "DiarizationPipeline", self._pipeline_factory(df)),
                patch.object(pipeline, "RESEGMENT_BY_SPEAKER", resegment_enabled),
                patch.object(pipeline, "DIARIZE_FILL_NEAREST", False),
            ):
                result, _ = pipeline.diarize(
                    np.zeros(1600, dtype=np.float32),
                    {
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
                        ]
                    },
                )
        finally:
            pipeline.whispermlx = original_wmlx
            pipeline.HF_TOKEN = original_token
            pipeline._diarize_pipeline = None
        return result

    def test_enabled_resplits_on_speaker_change(self):
        def assign(diarize_df, result, fill_nearest=False):
            # assign_word_speakers labels words in place and returns the result
            return result

        result = self._run_diarize(True, assign)
        assert len(result["segments"]) == 2
        assert result["segments"][0]["speaker"] == "SPEAKER_00"
        assert result["segments"][1]["speaker"] == "SPEAKER_01"

    def test_disabled_keeps_segment_shape(self):
        def assign(diarize_df, result, fill_nearest=False):
            return result

        result = self._run_diarize(False, assign)
        assert len(result["segments"]) == 1
        assert result["segments"][0]["text"] == "hello there hi"
