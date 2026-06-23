"""Unit tests for _resplit_segments_on_diarization_turns helper.

Tests are fast: whispermlx is mocked, no model downloads, no GPU required.
Covers: guard (skip when words present), multi-speaker re-split, single-speaker
no-change, no-overlapping-turns preservation, text apportionment, no 'words'
key added, and integration with diarize() function.
"""

from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helper to import pipeline fresh
# ---------------------------------------------------------------------------


def _import_pipeline():
    """Import app.pipeline fresh so module-level patches take effect."""
    import importlib

    import app.pipeline

    importlib.reload(app.pipeline)
    return app.pipeline


# ---------------------------------------------------------------------------
# Fixtures: building diarization DataFrames and result dicts
# ---------------------------------------------------------------------------


def _make_diarize_df(turns):
    """Build a DataFrame mimicking DiarizationPipeline output.

    Args:
        turns: list of (start, end, speaker) tuples

    Returns:
        pd.DataFrame with columns: segment, label, speaker, start, end
    """
    rows = []
    for start, end, speaker in turns:
        rows.append(
            {
                "segment": f"[{start}, {end}]",
                "label": speaker,
                "speaker": speaker,
                "start": start,
                "end": end,
            }
        )
    return pd.DataFrame(rows)


def _make_coarse_result_single_segment(text="Hello from speaker one and speaker two replies"):
    """Build a result dict with ONE coarse segment (no words), simulating
    the bug scenario where assign_word_speakers collapses to 1 speaker."""
    return {
        "segments": [
            {
                "start": 0.0,
                "end": 5.0,
                "text": text,
                "speaker": "SPEAKER_00",  # dominant speaker only
            }
        ],
        "language": "en",
    }


def _make_coarse_result_multi_segment():
    """Build a result dict with multiple segments, each with dominant speaker."""
    return {
        "segments": [
            {
                "start": 0.0,
                "end": 2.5,
                "text": "Hello from speaker one",
                "speaker": "SPEAKER_00",
            },
            {
                "start": 2.5,
                "end": 5.0,
                "text": "And speaker two replies",
                "speaker": "SPEAKER_01",
            },
        ],
        "language": "en",
    }


def _make_result_with_words():
    """Build a result with word-level data (should trigger the guard)."""
    return {
        "segments": [
            {
                "start": 0.0,
                "end": 2.5,
                "text": "Hello world",
                "speaker": "SPEAKER_00",
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
                    {"word": "world", "start": 1.1, "end": 2.5, "speaker": "SPEAKER_00"},
                ],
            },
            {
                "start": 2.5,
                "end": 5.0,
                "text": "Speaker two",
                "speaker": "SPEAKER_01",
                "words": [
                    {"word": "Speaker", "start": 2.5, "end": 3.5, "speaker": "SPEAKER_01"},
                    {"word": "two", "start": 3.6, "end": 5.0, "speaker": "SPEAKER_01"},
                ],
            },
        ],
        "language": "en",
    }


# ---------------------------------------------------------------------------
# 1. Guard: helper returns unchanged when words present
# ---------------------------------------------------------------------------


class TestGuardWordsPresent:
    """When any segment has word-level data, the helper must return result unchanged."""

    def test_words_present_no_resplit(self):
        """Result with words[] should not be modified."""
        pipeline = _import_pipeline()
        result = _make_result_with_words()
        df = _make_diarize_df(
            [
                (0.0, 2.5, "SPEAKER_00"),
                (2.5, 5.0, "SPEAKER_01"),
            ]
        )

        returned = pipeline._resplit_segments_on_diarization_turns(result, df)

        # Segments must be identical
        assert returned["segments"] == result["segments"]
        assert len(returned["segments"]) == 2

    def test_single_segment_with_words_no_resplit(self):
        """Even a single coarse segment with words[] should not trigger resplit."""
        pipeline = _import_pipeline()
        result = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": "Full text",
                    "speaker": "SPEAKER_00",
                    "words": [{"word": "Full", "start": 0.0, "end": 1.0}],
                }
            ],
            "language": "en",
        }
        df = _make_diarize_df(
            [
                (0.0, 2.5, "SPEAKER_00"),
                (2.5, 5.0, "SPEAKER_01"),
            ]
        )

        returned = pipeline._resplit_segments_on_diarization_turns(result, df)
        assert len(returned["segments"]) == 1
        assert "words" in returned["segments"][0]

    def test_empty_words_array_triggers_resplit(self):
        """Segment with words=[] (empty) should NOT trigger the guard, since
        there's no populated word-level data."""
        pipeline = _import_pipeline()
        result = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": "Hello from one and two replies",
                    "speaker": "SPEAKER_00",
                    "words": [],  # Empty — no actual word data
                }
            ],
            "language": "en",
        }
        df = _make_diarize_df(
            [
                (0.0, 2.5, "SPEAKER_00"),
                (2.5, 5.0, "SPEAKER_01"),
            ]
        )

        returned = pipeline._resplit_segments_on_diarization_turns(result, df)
        # Should resplit since words is empty (not populated)
        assert len(returned["segments"]) >= 2


# ---------------------------------------------------------------------------
# 2. Multi-speaker re-split: single coarse segment yields >=2 distinct speakers
# ---------------------------------------------------------------------------


class TestMultiSpeakerResplit:
    """When word_timestamps=false on multi-speaker audio, segments must carry
    >=2 distinct speaker labels with no words[] anywhere."""

    def test_single_coarse_segment_resplit_two_speakers(self):
        """One coarse segment with 2 diarization turns → >=2 sub-segments with
        distinct speakers."""
        pipeline = _import_pipeline()
        result = _make_coarse_result_single_segment()
        df = _make_diarize_df(
            [
                (0.0, 2.5, "SPEAKER_00"),
                (2.5, 5.0, "SPEAKER_01"),
            ]
        )

        returned = pipeline._resplit_segments_on_diarization_turns(result, df)
        speakers = {seg["speaker"] for seg in returned["segments"] if seg.get("speaker")}
        assert len(speakers) >= 2, f"Expected >=2 distinct speakers, got: {speakers}"

    def test_no_words_key_in_resplit_segments(self):
        """After resplit, NO segment should contain a 'words' key."""
        pipeline = _import_pipeline()
        result = _make_coarse_result_single_segment()
        df = _make_diarize_df(
            [
                (0.0, 2.5, "SPEAKER_00"),
                (2.5, 5.0, "SPEAKER_01"),
            ]
        )

        returned = pipeline._resplit_segments_on_diarization_turns(result, df)
        for seg in returned["segments"]:
            assert "words" not in seg, f"Segment must not have 'words' key: {seg}"

    def test_resplit_segments_have_start_end_speaker_text(self):
        """Each sub-segment must have start, end, speaker, and text fields."""
        pipeline = _import_pipeline()
        result = _make_coarse_result_single_segment()
        df = _make_diarize_df(
            [
                (0.0, 2.5, "SPEAKER_00"),
                (2.5, 5.0, "SPEAKER_01"),
            ]
        )

        returned = pipeline._resplit_segments_on_diarization_turns(result, df)
        for seg in returned["segments"]:
            assert "start" in seg, f"Missing 'start': {seg}"
            assert "end" in seg, f"Missing 'end': {seg}"
            assert "speaker" in seg, f"Missing 'speaker': {seg}"
            assert "text" in seg, f"Missing 'text': {seg}"
            assert seg["end"] >= seg["start"], f"end < start: {seg}"

    def test_three_speakers_resplit(self):
        """Three diarization turns produce >=3 distinct speaker labels."""
        pipeline = _import_pipeline()
        result = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 6.0,
                    "text": "First speaker then second and third talks",
                    "speaker": "SPEAKER_00",
                }
            ],
            "language": "en",
        }
        df = _make_diarize_df(
            [
                (0.0, 2.0, "SPEAKER_00"),
                (2.0, 4.0, "SPEAKER_01"),
                (4.0, 6.0, "SPEAKER_02"),
            ]
        )

        returned = pipeline._resplit_segments_on_diarization_turns(result, df)
        speakers = {seg["speaker"] for seg in returned["segments"] if seg.get("speaker")}
        assert len(speakers) >= 3, f"Expected >=3 distinct speakers, got: {speakers}"


# ---------------------------------------------------------------------------
# 3. Text apportionment
# ---------------------------------------------------------------------------


class TestTextApportionment:
    """Text should be apportioned across sub-segments by duration."""

    def test_sub_segment_texts_non_empty(self):
        """Each sub-segment text should be non-empty where possible."""
        pipeline = _import_pipeline()
        result = _make_coarse_result_single_segment()
        df = _make_diarize_df(
            [
                (0.0, 2.5, "SPEAKER_00"),
                (2.5, 5.0, "SPEAKER_01"),
            ]
        )

        returned = pipeline._resplit_segments_on_diarization_turns(result, df)
        for seg in returned["segments"]:
            assert seg.get("text", "").strip(), f"Sub-segment text should be non-empty: {seg}"

    def test_total_text_preserved(self):
        """Total text across sub-segments should contain the original words."""
        pipeline = _import_pipeline()
        original_text = "Hello from speaker one and speaker two replies"
        result = _make_coarse_result_single_segment(text=original_text)
        df = _make_diarize_df(
            [
                (0.0, 2.5, "SPEAKER_00"),
                (2.5, 5.0, "SPEAKER_01"),
            ]
        )

        returned = pipeline._resplit_segments_on_diarization_turns(result, df)
        joined_text = " ".join(seg.get("text", "") for seg in returned["segments"]).strip()
        # All original words should appear in the joined text
        for word in original_text.split():
            assert word in joined_text, f"Word '{word}' missing from joined text: {joined_text}"


# ---------------------------------------------------------------------------
# 4. Single-speaker: no unnecessary splitting
# ---------------------------------------------------------------------------


class TestSingleSpeakerNoSplit:
    """Single-speaker audio should yield exactly 1 speaker label (no over-splitting)."""

    def test_single_speaker_one_label(self):
        """When all turns belong to one speaker, segments should have 1 distinct speaker."""
        pipeline = _import_pipeline()
        result = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": "Hello world this is a single speaker",
                    "speaker": "SPEAKER_00",
                }
            ],
            "language": "en",
        }
        df = _make_diarize_df(
            [
                (0.0, 2.0, "SPEAKER_00"),
                (2.0, 5.0, "SPEAKER_00"),
            ]
        )

        returned = pipeline._resplit_segments_on_diarization_turns(result, df)
        speakers = {seg["speaker"] for seg in returned["segments"] if seg.get("speaker")}
        assert len(speakers) == 1, f"Single-speaker should yield 1 distinct label, got: {speakers}"

    def test_single_speaker_segments_still_have_speaker(self):
        """After resplit, all segments still carry a non-empty speaker label."""
        pipeline = _import_pipeline()
        result = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": "Hello world",
                    "speaker": "SPEAKER_00",
                }
            ],
            "language": "en",
        }
        df = _make_diarize_df(
            [
                (0.0, 5.0, "SPEAKER_00"),
            ]
        )

        returned = pipeline._resplit_segments_on_diarization_turns(result, df)
        for seg in returned["segments"]:
            assert seg.get("speaker"), f"Segment should have a non-empty speaker: {seg}"


# ---------------------------------------------------------------------------
# 5. No overlapping turns: segment preserved as-is
# ---------------------------------------------------------------------------


class TestNoOverlappingTurns:
    """A coarse segment with zero overlapping diarization turns is left intact."""

    def test_segment_outside_turn_bounds_preserved(self):
        """A segment that doesn't overlap any turn is left unchanged."""
        pipeline = _import_pipeline()
        result = {
            "segments": [
                {
                    "start": 10.0,
                    "end": 15.0,
                    "text": "Outside any turn",
                    "speaker": "SPEAKER_00",  # dominant speaker from assign_word_speakers
                }
            ],
            "language": "en",
        }
        df = _make_diarize_df(
            [
                (0.0, 2.5, "SPEAKER_00"),
                (2.5, 5.0, "SPEAKER_01"),
            ]
        )

        returned = pipeline._resplit_segments_on_diarization_turns(result, df)
        assert len(returned["segments"]) == 1, "Non-overlapping segment should be preserved as-is"
        assert returned["segments"][0]["speaker"] == "SPEAKER_00"
        assert returned["segments"][0]["text"] == "Outside any turn"

    def test_segment_partially_overlapping_gets_resplit(self):
        """A segment that partially overlaps turns should still be resplit."""
        pipeline = _import_pipeline()
        result = {
            "segments": [
                {
                    "start": 1.0,
                    "end": 4.0,
                    "text": "Partial overlap",
                    "speaker": "SPEAKER_00",
                }
            ],
            "language": "en",
        }
        df = _make_diarize_df(
            [
                (0.0, 2.5, "SPEAKER_00"),
                (2.5, 5.0, "SPEAKER_01"),
            ]
        )

        returned = pipeline._resplit_segments_on_diarization_turns(result, df)
        speakers = {seg["speaker"] for seg in returned["segments"] if seg.get("speaker")}
        assert len(speakers) >= 2, f"Partial overlap should produce >=2 speakers: {speakers}"


# ---------------------------------------------------------------------------
# 6. Consecutive same-speaker merging
# ---------------------------------------------------------------------------


class TestConsecutiveSameSpeakerMerge:
    """Consecutive same-speaker runs should be merged into one sub-segment."""

    def test_consecutive_same_speaker_merged(self):
        """Multiple consecutive same-speaker turns should be merged into one."""
        pipeline = _import_pipeline()
        result = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 6.0,
                    "text": "Speaker one talks for a while and then another speaks",
                    "speaker": "SPEAKER_00",
                }
            ],
            "language": "en",
        }
        # Three consecutive SPEAKER_00 turns → should merge into one
        df = _make_diarize_df(
            [
                (0.0, 2.0, "SPEAKER_00"),
                (2.0, 3.0, "SPEAKER_00"),
                (3.0, 4.0, "SPEAKER_01"),
                (4.0, 6.0, "SPEAKER_01"),
            ]
        )

        returned = pipeline._resplit_segments_on_diarization_turns(result, df)
        # Should produce exactly 2 sub-segments (SPEAKER_00 merged, SPEAKER_01 merged)
        speakers = [seg["speaker"] for seg in returned["segments"]]
        assert len(speakers) == 2, f"Consecutive same-speaker should merge; got {len(speakers)} segments: {speakers}"
        assert speakers[0] == "SPEAKER_00"
        assert speakers[1] == "SPEAKER_01"

    def test_alternating_speakers_not_merged(self):
        """Alternating speakers should NOT be merged."""
        pipeline = _import_pipeline()
        result = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 6.0,
                    "text": "One two one two one two",
                    "speaker": "SPEAKER_00",
                }
            ],
            "language": "en",
        }
        df = _make_diarize_df(
            [
                (0.0, 1.0, "SPEAKER_00"),
                (1.0, 2.0, "SPEAKER_01"),
                (2.0, 3.0, "SPEAKER_00"),
                (3.0, 4.0, "SPEAKER_01"),
                (4.0, 5.0, "SPEAKER_00"),
                (5.0, 6.0, "SPEAKER_01"),
            ]
        )

        returned = pipeline._resplit_segments_on_diarization_turns(result, df)
        # Should produce 6 sub-segments (no merging since speakers alternate)
        speakers = [seg["speaker"] for seg in returned["segments"]]
        assert len(speakers) == 6, f"Alternating speakers should not merge; got {len(speakers)}"


# ---------------------------------------------------------------------------
# 7. Integration with diarize() function
# ---------------------------------------------------------------------------


class TestDiarizeIntegrationWithResplit:
    """Verify _resplit_segments_on_diarization_turns is called inside diarize()
    after assign_word_speakers, with the diarize_segments DataFrame."""

    def _setup_pipeline(self):
        pipeline = _import_pipeline()
        pipeline.HF_TOKEN = "fake-token"
        pipeline._diarize_pipeline = None
        return pipeline

    def test_resplit_called_after_assign_word_speakers(self):
        """diarize() should call _resplit_segments_on_diarization_turns."""
        pipeline = self._setup_pipeline()
        audio = np.zeros(16000, dtype=np.float32)
        result = {
            "segments": [{"start": 0.0, "end": 5.0, "text": "Hello one two", "speaker": "SPEAKER_00"}],
            "language": "en",
        }

        diarize_df = _make_diarize_df(
            [
                (0.0, 2.5, "SPEAKER_00"),
                (2.5, 5.0, "SPEAKER_01"),
            ]
        )

        mock_diarize_model = MagicMock(return_value=diarize_df)

        assign_result = {
            "segments": [{"start": 0.0, "end": 5.0, "text": "Hello one two", "speaker": "SPEAKER_00"}],
            "language": "en",
        }

        with (
            patch.object(pipeline, "load_diarize_pipeline", return_value=mock_diarize_model),
            patch.object(pipeline.whispermlx, "assign_word_speakers", return_value=assign_result),
            patch.object(
                pipeline,
                "_resplit_segments_on_diarization_turns",
                side_effect=pipeline._resplit_segments_on_diarization_turns,
            ) as mock_resplit,
        ):
            returned_result, _ = pipeline.diarize(audio, result)

        mock_resplit.assert_called_once()

    def test_diarize_resplit_produces_multiple_speakers(self):
        """Full diarize() flow with resplit should produce >=2 speakers
        when word_timestamps=false on multi-speaker input."""
        pipeline = self._setup_pipeline()
        audio = np.zeros(16000, dtype=np.float32)
        result = {
            "segments": [{"start": 0.0, "end": 5.0, "text": "Hello one and two replies", "speaker": "SPEAKER_00"}],
            "language": "en",
        }

        diarize_df = _make_diarize_df(
            [
                (0.0, 2.5, "SPEAKER_00"),
                (2.5, 5.0, "SPEAKER_01"),
            ]
        )

        mock_diarize_model = MagicMock(return_value=diarize_df)

        # assign_word_speakers collapses to dominant speaker
        assign_result = {
            "segments": [{"start": 0.0, "end": 5.0, "text": "Hello one and two replies", "speaker": "SPEAKER_00"}],
            "language": "en",
        }

        with (
            patch.object(pipeline, "load_diarize_pipeline", return_value=mock_diarize_model),
            patch.object(pipeline.whispermlx, "assign_word_speakers", return_value=assign_result),
        ):
            returned_result, _ = pipeline.diarize(audio, result)

        speakers = {seg["speaker"] for seg in returned_result["segments"] if seg.get("speaker")}
        assert len(speakers) >= 2, f"diarize() with resplit should yield >=2 distinct speakers, got: {speakers}"

    def test_diarize_with_words_does_not_resplit(self):
        """When word_timestamps=true (words present), diarize() should NOT resplit."""
        pipeline = self._setup_pipeline()
        audio = np.zeros(16000, dtype=np.float32)
        result = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.5,
                    "text": "Hello world",
                    "speaker": "SPEAKER_00",
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
                        {"word": "world", "start": 1.1, "end": 2.5, "speaker": "SPEAKER_00"},
                    ],
                },
                {
                    "start": 2.5,
                    "end": 5.0,
                    "text": "Speaker two",
                    "speaker": "SPEAKER_01",
                    "words": [
                        {"word": "Speaker", "start": 2.5, "end": 3.5, "speaker": "SPEAKER_01"},
                        {"word": "two", "start": 3.6, "end": 5.0, "speaker": "SPEAKER_01"},
                    ],
                },
            ],
            "language": "en",
        }

        diarize_df = _make_diarize_df(
            [
                (0.0, 2.5, "SPEAKER_00"),
                (2.5, 5.0, "SPEAKER_01"),
            ]
        )

        mock_diarize_model = MagicMock(return_value=diarize_df)

        assign_result = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.5,
                    "text": "Hello world",
                    "speaker": "SPEAKER_00",
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
                        {"word": "world", "start": 1.1, "end": 2.5, "speaker": "SPEAKER_00"},
                    ],
                },
                {
                    "start": 2.5,
                    "end": 5.0,
                    "text": "Speaker two",
                    "speaker": "SPEAKER_01",
                    "words": [
                        {"word": "Speaker", "start": 2.5, "end": 3.5, "speaker": "SPEAKER_01"},
                        {"word": "two", "start": 3.6, "end": 5.0, "speaker": "SPEAKER_01"},
                    ],
                },
            ],
            "language": "en",
        }

        with (
            patch.object(pipeline, "load_diarize_pipeline", return_value=mock_diarize_model),
            patch.object(pipeline.whispermlx, "assign_word_speakers", return_value=assign_result),
            patch.object(
                pipeline,
                "_resplit_segments_on_diarization_turns",
                side_effect=pipeline._resplit_segments_on_diarization_turns,
            ),
        ):
            returned_result, _ = pipeline.diarize(audio, result)

        # Guard should have been triggered; resplit should return early
        # Result should be unchanged (2 segments with words)
        assert len(returned_result["segments"]) == 2
        for seg in returned_result["segments"]:
            assert "words" in seg


# ---------------------------------------------------------------------------
# 8. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for the segment re-split helper."""

    def test_empty_diarize_df(self):
        """Empty diarization DataFrame should return result unchanged."""
        pipeline = _import_pipeline()
        result = _make_coarse_result_single_segment()
        df = _make_diarize_df([])

        returned = pipeline._resplit_segments_on_diarization_turns(result, df)
        assert returned["segments"] == result["segments"]

    def test_empty_segments(self):
        """Empty segments list should not crash."""
        pipeline = _import_pipeline()
        result = {"segments": [], "language": "en"}
        df = _make_diarize_df([(0.0, 2.5, "SPEAKER_00")])

        returned = pipeline._resplit_segments_on_diarization_turns(result, df)
        assert returned["segments"] == []

    def test_segment_with_no_text(self):
        """A segment with empty text should not crash during apportionment."""
        pipeline = _import_pipeline()
        result = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": "",
                    "speaker": "SPEAKER_00",
                }
            ],
            "language": "en",
        }
        df = _make_diarize_df(
            [
                (0.0, 2.5, "SPEAKER_00"),
                (2.5, 5.0, "SPEAKER_01"),
            ]
        )

        returned = pipeline._resplit_segments_on_diarization_turns(result, df)
        # Should produce sub-segments without crashing
        assert len(returned["segments"]) >= 1

    def test_non_dataframe_diarize_segments(self):
        """If diarize_segments is not a DataFrame (e.g., Annotation), helper
        should handle gracefully."""
        pipeline = _import_pipeline()
        result = _make_coarse_result_single_segment()

        # Pass a non-DataFrame object
        returned = pipeline._resplit_segments_on_diarization_turns(result, "not_a_dataframe")
        # Should return result unchanged (graceful fallback)
        assert returned["segments"] == result["segments"]

    def test_turns_exactly_at_segment_boundaries(self):
        """Turns that exactly match segment boundaries should be handled correctly."""
        pipeline = _import_pipeline()
        result = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": "Exact boundaries test",
                    "speaker": "SPEAKER_00",
                }
            ],
            "language": "en",
        }
        df = _make_diarize_df(
            [
                (0.0, 2.5, "SPEAKER_00"),
                (2.5, 5.0, "SPEAKER_01"),
            ]
        )

        returned = pipeline._resplit_segments_on_diarization_turns(result, df)
        speakers = {seg["speaker"] for seg in returned["segments"] if seg.get("speaker")}
        assert len(speakers) >= 2

    def test_multiple_segments_each_resplit(self):
        """When there are multiple coarse segments, each should be resplit independently."""
        pipeline = _import_pipeline()
        result = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": "Speaker one and two in first part",
                    "speaker": "SPEAKER_00",
                },
                {
                    "start": 10.0,
                    "end": 15.0,
                    "text": "Speaker three and four in second part",
                    "speaker": "SPEAKER_02",
                },
            ],
            "language": "en",
        }
        df = _make_diarize_df(
            [
                (0.0, 2.5, "SPEAKER_00"),
                (2.5, 5.0, "SPEAKER_01"),
                (10.0, 12.5, "SPEAKER_02"),
                (12.5, 15.0, "SPEAKER_03"),
            ]
        )

        returned = pipeline._resplit_segments_on_diarization_turns(result, df)
        # First coarse segment should produce 2 sub-segments
        # Second coarse segment should produce 2 sub-segments
        # Total >= 4
        assert len(returned["segments"]) >= 4
        all_speakers = {seg["speaker"] for seg in returned["segments"] if seg.get("speaker")}
        assert len(all_speakers) >= 4
