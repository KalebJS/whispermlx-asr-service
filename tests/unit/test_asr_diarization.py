"""Unit tests for POST /asr diarization parameters and response handling.

Tests are fast: pipeline is mocked, no model downloads, no GPU required.
Covers: diarize/enable_diarization toggle, speaker labels in segments,
embeddings handling, srt/vtt/tsv speaker formatting, graceful skip.
"""

from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_AUDIO = b"RIFF" + b"\x00" * 100  # minimal WAV-like bytes


def _mock_pipeline_result_diarized(
    segments=None,
    language="en",
    word_segments=None,
    speaker_embeddings=None,
):
    """Build a realistic pipeline result with speaker labels."""
    if segments is None:
        segments = [
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
    if word_segments is None:
        word_segments = []
        for seg in segments:
            if "words" in seg:
                word_segments.extend(seg["words"])
    result = {
        "segments": segments,
        "language": language,
        "word_segments": word_segments,
    }
    return result, speaker_embeddings


def _mock_pipeline_result_no_diarization(segments=None, language="en"):
    """Build a pipeline result WITHOUT speaker labels."""
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
    return {
        "segments": segments,
        "language": language,
        "word_segments": [
            {"word": "Hello", "start": 0.0, "end": 1.2},
            {"word": "world", "start": 1.3, "end": 2.5},
        ],
    }


@pytest.fixture()
def client():
    """Create a TestClient with pipeline functions mocked."""
    with patch("app.main.run_in_queue") as mock_queue, patch(
        "app.main.whispermlx"
    ) as mock_wmlx, patch("app.main.load_whisper_model"), patch(
        "app.main.resolve_model_name"
    ) as mock_resolve, patch(
        "app.main.get_canonical_models"
    ) as mock_canonical:
        mock_resolve.side_effect = lambda m: m if m else "large-v3"
        mock_canonical.return_value = [
            "tiny", "tiny.en", "base", "base.en", "small", "small.en",
            "medium", "medium.en", "large", "large-v1", "large-v2",
            "large-v3", "large-v3-turbo", "turbo",
        ]
        mock_wmlx.load_audio.return_value = np.zeros(16000, dtype=np.float32)

        from app.main import app

        with TestClient(app) as c:
            yield c, mock_queue, mock_resolve, mock_canonical


# ---------------------------------------------------------------------------
# 1. Diarization toggle logic
# ---------------------------------------------------------------------------


class TestDiarizeToggle:
    """Verify diarize and enable_diarization toggle behavior on /asr."""

    def test_diarize_true_produces_speaker_labels(self, client):
        """diarize=true should produce segments with speaker labels."""
        client_c, mock_queue, _, _ = client
        result, embeddings = _mock_pipeline_result_diarized()

        async def _return_result(*args, **kwargs):
            return result, embeddings

        mock_queue.side_effect = _return_result

        resp = client_c.post(
            "/asr?diarize=true",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        body = resp.json()
        speakers = {seg.get("speaker") for seg in body["segments"] if seg.get("speaker")}
        assert len(speakers) >= 2, \
            f"Expected >=2 distinct speakers with diarize=true, got: {speakers}"

    def test_diarize_false_omits_speakers(self, client):
        """diarize=false should NOT produce speaker labels."""
        client_c, mock_queue, _, _ = client
        result = _mock_pipeline_result_no_diarization()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client_c.post(
            "/asr?diarize=false",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        for seg in body["segments"]:
            assert not seg.get("speaker"), \
                f"Segment should not have speaker label with diarize=false: {seg}"

    def test_diarization_defaults_on(self, client):
        """When neither diarize nor enable_diarization is supplied, diarization runs by default."""
        client_c, mock_queue, _, _ = client
        result, embeddings = _mock_pipeline_result_diarized()

        async def _return_result(*args, **kwargs):
            return result, embeddings

        mock_queue.side_effect = _return_result

        resp = client_c.post(
            "/asr",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        speakers = {seg.get("speaker") for seg in body["segments"] if seg.get("speaker")}
        assert len(speakers) >= 1, \
            "Diarization should default on; at least one speaker label expected"

    def test_enable_diarization_true_same_as_diarize_true(self, client):
        """enable_diarization=true produces same result as diarize=true."""
        client_c, mock_queue, _, _ = client
        result, embeddings = _mock_pipeline_result_diarized()

        async def _return_result(*args, **kwargs):
            return result, embeddings

        mock_queue.side_effect = _return_result

        resp = client_c.post(
            "/asr?enable_diarization=true",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        speakers = {seg.get("speaker") for seg in body["segments"] if seg.get("speaker")}
        assert len(speakers) >= 2, \
            f"enable_diarization=true should produce speaker labels, got: {speakers}"

    def test_enable_diarization_overrides_diarize_false(self, client):
        """enable_diarization=true overrides diarize=false (OR logic)."""
        client_c, mock_queue, _, _ = client

        # Capture what should_diarize value is passed to run_pipeline
        captured_kwargs = {}

        async def _capture_run(*args, **kwargs):
            captured_kwargs.update(kwargs)
            result, embeddings = _mock_pipeline_result_diarized()
            return result, embeddings

        mock_queue.side_effect = _capture_run

        resp = client_c.post(
            "/asr?diarize=false&enable_diarization=true",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        # should_diarize should be True because enable_diarization=true overrides diarize=false
        assert captured_kwargs.get("should_diarize") is True, \
            f"should_diarize should be True with enable_diarization=true, diarize=false; got {captured_kwargs}"

    def test_diarize_false_and_enable_diarization_false(self, client):
        """Both diarize=false and enable_diarization=false should skip diarization."""
        client_c, mock_queue, _, _ = client

        captured_kwargs = {}

        async def _capture_run(*args, **kwargs):
            captured_kwargs.update(kwargs)
            result = _mock_pipeline_result_no_diarization()
            return result, None

        mock_queue.side_effect = _capture_run

        resp = client_c.post(
            "/asr?diarize=false&enable_diarization=false",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        assert captured_kwargs.get("should_diarize") is False, \
            f"should_diarize should be False with both flags false; got {captured_kwargs}"


# ---------------------------------------------------------------------------
# 2. Speaker labels in response
# ---------------------------------------------------------------------------


class TestSpeakerLabelsInResponse:
    """Verify speaker labels appear in the JSON response."""

    def test_segments_have_speaker_field(self, client):
        """Every segment should carry a speaker field when diarized."""
        client_c, mock_queue, _, _ = client
        result, _ = _mock_pipeline_result_diarized()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client_c.post(
            "/asr?diarize=true",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        for seg in body["segments"]:
            assert "speaker" in seg, f"Segment missing 'speaker' field: {seg}"
            assert seg["speaker"], f"Segment has empty speaker: {seg}"

    def test_speaker_label_format(self, client):
        """Speaker labels should follow SPEAKER_NN format."""
        client_c, mock_queue, _, _ = client
        result, _ = _mock_pipeline_result_diarized()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client_c.post(
            "/asr?diarize=true",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        import re
        for seg in body["segments"]:
            if seg.get("speaker"):
                assert re.match(r"^SPEAKER_\d+$", seg["speaker"]), \
                    f"Speaker label '{seg['speaker']}' doesn't match SPEAKER_NN format"

    def test_words_have_speaker_labels(self, client):
        """When word_timestamps=true and diarize=true, words should have speaker labels."""
        client_c, mock_queue, _, _ = client
        result, _ = _mock_pipeline_result_diarized()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client_c.post(
            "/asr?diarize=true&word_timestamps=true",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        # Check that some words have speaker labels
        word_speakers = set()
        for seg in body["segments"]:
            if "words" in seg:
                for word in seg["words"]:
                    if word.get("speaker"):
                        word_speakers.add(word["speaker"])
        assert len(word_speakers) >= 1, \
            f"Expected at least some words with speaker labels, got: {word_speakers}"

    def test_word_speakers_subset_of_segment_speakers(self, client):
        """Word-level speakers must be a subset of segment-level speakers."""
        client_c, mock_queue, _, _ = client
        result, _ = _mock_pipeline_result_diarized()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client_c.post(
            "/asr?diarize=true&word_timestamps=true",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()

        segment_speakers = {seg.get("speaker") for seg in body["segments"] if seg.get("speaker")}
        word_speakers = set()
        for seg in body["segments"]:
            if "words" in seg:
                for word in seg["words"]:
                    if word.get("speaker"):
                        word_speakers.add(word["speaker"])

        assert word_speakers <= segment_speakers, \
            f"Word speakers ({word_speakers}) not a subset of segment speakers ({segment_speakers})"


# ---------------------------------------------------------------------------
# 3. Speaker embeddings handling
# ---------------------------------------------------------------------------


class TestSpeakerEmbeddings:
    """Verify return_speaker_embeddings parameter and response handling."""

    def test_embeddings_included_when_requested(self, client):
        """return_speaker_embeddings=true includes speaker_embeddings in JSON response."""
        client_c, mock_queue, _, _ = client
        embeddings = {
            "SPEAKER_00": [0.1] * 256,
            "SPEAKER_01": [0.2] * 256,
        }
        result, _ = _mock_pipeline_result_diarized(speaker_embeddings=embeddings)

        async def _return_result(*args, **kwargs):
            return result, embeddings

        mock_queue.side_effect = _return_result

        resp = client_c.post(
            "/asr?diarize=true&return_speaker_embeddings=true",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "speaker_embeddings" in body, "Missing speaker_embeddings key"
        assert set(body["speaker_embeddings"].keys()) == {"SPEAKER_00", "SPEAKER_01"}, \
            f"Embedding keys should match speaker labels, got: {list(body['speaker_embeddings'].keys())}"

    def test_embeddings_omitted_by_default(self, client):
        """return_speaker_embeddings not set → no speaker_embeddings key in response."""
        client_c, mock_queue, _, _ = client
        result, _ = _mock_pipeline_result_diarized()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client_c.post(
            "/asr?diarize=true",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "speaker_embeddings" not in body, \
            f"speaker_embeddings should not be in response when not requested: {body.keys()}"

    def test_embeddings_only_in_json_not_other_formats(self, client):
        """speaker_embeddings only appears in JSON output, not srt/vtt/tsv."""
        client_c, mock_queue, _, _ = client
        embeddings = {"SPEAKER_00": [0.1] * 256}
        result, _ = _mock_pipeline_result_diarized(speaker_embeddings=embeddings)

        for fmt in ["srt", "vtt", "tsv"]:
            async def _return_result(*args, **kwargs):
                return result, embeddings

            mock_queue.side_effect = _return_result

            resp = client_c.post(
                f"/asr?diarize=true&return_speaker_embeddings=true&output_format={fmt}",
                files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
            )
            assert resp.status_code == 200, f"Expected 200 for {fmt}, got {resp.status_code}"
            body = resp.json()
            assert "speaker_embeddings" not in body, \
                f"speaker_embeddings should not be in {fmt} response: {body.keys()}"

    def test_embeddings_keys_match_speaker_labels(self, client):
        """Embedding keys must match the distinct speaker labels in segments."""
        client_c, mock_queue, _, _ = client
        embeddings = {
            "SPEAKER_00": [0.1] * 256,
            "SPEAKER_01": [0.2] * 256,
        }
        result, _ = _mock_pipeline_result_diarized(speaker_embeddings=embeddings)

        async def _return_result(*args, **kwargs):
            return result, embeddings

        mock_queue.side_effect = _return_result

        resp = client_c.post(
            "/asr?diarize=true&return_speaker_embeddings=true",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        segment_speakers = {seg["speaker"] for seg in body["segments"]}
        embedding_keys = set(body["speaker_embeddings"].keys())
        assert embedding_keys == segment_speakers, \
            f"Embedding keys ({embedding_keys}) != segment speakers ({segment_speakers})"

    def test_one_embedding_per_speaker(self, client):
        """There should be exactly one embedding per distinct speaker."""
        client_c, mock_queue, _, _ = client
        embeddings = {
            "SPEAKER_00": [0.1] * 256,
            "SPEAKER_01": [0.2] * 256,
        }
        result, _ = _mock_pipeline_result_diarized(speaker_embeddings=embeddings)

        async def _return_result(*args, **kwargs):
            return result, embeddings

        mock_queue.side_effect = _return_result

        resp = client_c.post(
            "/asr?diarize=true&return_speaker_embeddings=true",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        distinct_speakers = {seg["speaker"] for seg in body["segments"]}
        assert len(body["speaker_embeddings"]) == len(distinct_speakers), \
            f"Expected {len(distinct_speakers)} embeddings, got {len(body['speaker_embeddings'])}"


# ---------------------------------------------------------------------------
# 4. Speaker labels in srt/vtt/tsv formats
# ---------------------------------------------------------------------------


class TestSpeakerLabelsInFormats:
    """Verify speaker labels appear in srt, vtt, and tsv output formats."""

    def test_srt_includes_speaker_labels(self, client):
        """SRT output should include [SPEAKER_NN] prefix on text lines."""
        client_c, mock_queue, _, _ = client
        result, _ = _mock_pipeline_result_diarized()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client_c.post(
            "/asr?diarize=true&output_format=srt",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        srt_content = body["srt"]
        assert "[SPEAKER_" in srt_content, \
            f"SRT output should contain [SPEAKER_NN] prefixes: {srt_content[:200]}"

    def test_vtt_includes_speaker_labels(self, client):
        """VTT output should include [SPEAKER_NN] prefix on text lines."""
        client_c, mock_queue, _, _ = client
        result, _ = _mock_pipeline_result_diarized()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client_c.post(
            "/asr?diarize=true&output_format=vtt",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        vtt_content = body["vtt"]
        assert "[SPEAKER_" in vtt_content, \
            f"VTT output should contain [SPEAKER_NN] prefixes: {vtt_content[:200]}"

    def test_tsv_includes_speaker_column(self, client):
        """TSV output should include speaker column with labels."""
        client_c, mock_queue, _, _ = client
        result, _ = _mock_pipeline_result_diarized()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client_c.post(
            "/asr?diarize=true&output_format=tsv",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        tsv_content = body["tsv"]
        lines = tsv_content.strip().split("\n")
        assert lines[0] == "start\tend\ttext\tspeaker", "TSV header mismatch"
        # Data rows should have speaker label in 4th column
        for line in lines[1:]:
            cols = line.split("\t")
            assert len(cols) == 4, f"Expected 4 TSV columns, got {len(cols)}: {line}"
            assert cols[3].startswith("SPEAKER_"), \
                f"Speaker column should have SPEAKER_NN label, got: {cols[3]}"

    def test_no_speaker_labels_in_srt_when_diarize_false(self, client):
        """SRT output should NOT include speaker labels when diarize=false."""
        client_c, mock_queue, _, _ = client
        result = _mock_pipeline_result_no_diarization()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client_c.post(
            "/asr?diarize=false&output_format=srt",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        srt_content = body["srt"]
        assert "[SPEAKER_" not in srt_content, \
            f"SRT should not contain speaker labels when diarize=false: {srt_content[:200]}"


# ---------------------------------------------------------------------------
# 5. num_speakers, min_speakers, max_speakers parameter passing
# ---------------------------------------------------------------------------


class TestSpeakerCountParams:
    """Verify num/min/max_speakers params are passed to the pipeline."""

    def test_num_speakers_passed_to_pipeline(self, client):
        """num_speakers query param should reach the pipeline."""
        client_c, mock_queue, _, _ = client

        captured_kwargs = {}

        async def _capture_run(*args, **kwargs):
            captured_kwargs.update(kwargs)
            result, _ = _mock_pipeline_result_diarized()
            return result, None

        mock_queue.side_effect = _capture_run

        resp = client_c.post(
            "/asr?diarize=true&num_speakers=2",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        assert captured_kwargs.get("num_speakers") == 2, \
            f"num_speakers=2 not passed to pipeline; got: {captured_kwargs}"

    def test_min_max_speakers_passed_to_pipeline(self, client):
        """min_speakers and max_speakers params should reach the pipeline."""
        client_c, mock_queue, _, _ = client

        captured_kwargs = {}

        async def _capture_run(*args, **kwargs):
            captured_kwargs.update(kwargs)
            result, _ = _mock_pipeline_result_diarized()
            return result, None

        mock_queue.side_effect = _capture_run

        resp = client_c.post(
            "/asr?diarize=true&min_speakers=2&max_speakers=4",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        assert captured_kwargs.get("min_speakers") == 2, \
            f"min_speakers=2 not passed: {captured_kwargs}"
        assert captured_kwargs.get("max_speakers") == 4, \
            f"max_speakers=4 not passed: {captured_kwargs}"

    def test_return_speaker_embeddings_passed_to_pipeline(self, client):
        """return_speaker_embeddings param should reach the pipeline."""
        client_c, mock_queue, _, _ = client

        captured_kwargs = {}

        async def _capture_run(*args, **kwargs):
            captured_kwargs.update(kwargs)
            result, embeddings = _mock_pipeline_result_diarized(
                speaker_embeddings={"SPEAKER_00": [0.1] * 256}
            )
            return result, embeddings

        mock_queue.side_effect = _capture_run

        resp = client_c.post(
            "/asr?diarize=true&return_speaker_embeddings=true",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        assert captured_kwargs.get("return_speaker_embeddings") is True, \
            f"return_speaker_embeddings not passed: {captured_kwargs}"

    def test_return_speaker_embeddings_defaults_to_false(self, client):
        """When return_speaker_embeddings is not set, it defaults to False."""
        client_c, mock_queue, _, _ = client

        captured_kwargs = {}

        async def _capture_run(*args, **kwargs):
            captured_kwargs.update(kwargs)
            result, _ = _mock_pipeline_result_diarized()
            return result, None

        mock_queue.side_effect = _capture_run

        resp = client_c.post(
            "/asr?diarize=true",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        assert captured_kwargs.get("return_speaker_embeddings") is False, \
            f"return_speaker_embeddings should default to False: {captured_kwargs}"


# ---------------------------------------------------------------------------
# 6. Graceful skip when HF_TOKEN missing
# ---------------------------------------------------------------------------


class TestGracefulSkipNoToken:
    """Verify diarization gracefully skipped (transcription preserved) when HF_TOKEN missing."""

    def test_returns_200_without_hf_token(self, client):
        """Even without HF_TOKEN, the request returns 200 with transcription."""
        client_c, mock_queue, _, _ = client
        # Simulate pipeline returning result without speakers (HF_TOKEN missing)
        result = _mock_pipeline_result_no_diarization()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client_c.post(
            "/asr?diarize=true",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        body = resp.json()
        assert len(body["segments"]) > 0, "Transcription should be preserved"
        assert body["segments"][0].get("text"), "Segment text should be present"

    def test_no_speakers_when_hf_token_missing(self, client):
        """No speaker labels when HF_TOKEN is missing."""
        client_c, mock_queue, _, _ = client
        result = _mock_pipeline_result_no_diarization()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client_c.post(
            "/asr?diarize=true",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        for seg in body["segments"]:
            assert not seg.get("speaker"), \
                f"Should not have speaker labels without HF_TOKEN: {seg}"

    def test_no_embeddings_when_hf_token_missing(self, client):
        """No speaker_embeddings when HF_TOKEN is missing, even if requested."""
        client_c, mock_queue, _, _ = client
        result = _mock_pipeline_result_no_diarization()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client_c.post(
            "/asr?diarize=true&return_speaker_embeddings=true",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "speaker_embeddings" not in body, \
            "speaker_embeddings should not be present without HF_TOKEN"


# ---------------------------------------------------------------------------
# 7. Diarization failure degrades gracefully
# ---------------------------------------------------------------------------


class TestDiarizationFailureGraceful:
    """Verify diarization failure preserves transcription."""

    def test_failure_preserves_transcription(self, client):
        """When diarization fails internally, transcription text is still returned."""
        client_c, mock_queue, _, _ = client
        # Simulate diarization failure: pipeline returns result without speakers
        result = _mock_pipeline_result_no_diarization()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client_c.post(
            "/asr?diarize=true",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["segments"]) > 0, "Transcription must be preserved"
        assert body["segments"][0].get("text"), "Text must be present"


# ---------------------------------------------------------------------------
# 8. Diarization with word_timestamps off
# ---------------------------------------------------------------------------


class TestDiarizationWithoutAlignment:
    """Verify diarization works when word_timestamps=false."""

    def test_speakers_assigned_when_word_timestamps_false(self, client):
        """Diarization labels segments even when word_timestamps=false."""
        client_c, mock_queue, _, _ = client
        # When word_timestamps=false, result has segments with speakers but no words
        result = {
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "Speaker one", "speaker": "SPEAKER_00"},
                {"start": 2.5, "end": 5.0, "text": "Speaker two", "speaker": "SPEAKER_01"},
            ],
            "language": "en",
        }

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client_c.post(
            "/asr?diarize=true&word_timestamps=false",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        speakers = {seg.get("speaker") for seg in body["segments"] if seg.get("speaker")}
        assert len(speakers) >= 2, \
            f"Expected >=2 speakers with diarize=true, word_timestamps=false; got: {speakers}"

    def test_no_words_when_word_timestamps_false(self, client):
        """When word_timestamps=false, segments should not have words arrays."""
        client_c, mock_queue, _, _ = client
        result = {
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "Speaker one", "speaker": "SPEAKER_00"},
            ],
            "language": "en",
        }

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client_c.post(
            "/asr?diarize=true&word_timestamps=false",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["word_segments"] == [], \
            f"word_segments should be empty when word_timestamps=false: {body['word_segments']}"
        for seg in body["segments"]:
            assert "words" not in seg or len(seg.get("words", [])) == 0, \
                f"Segment should not have words when word_timestamps=false: {seg}"


# ---------------------------------------------------------------------------
# 9. Single-speaker scenario
# ---------------------------------------------------------------------------


class TestSingleSpeakerDiarization:
    """Verify diarization on single-speaker audio."""

    def test_single_speaker_yields_one_label(self, client):
        """Single-speaker clip should yield exactly one speaker label."""
        client_c, mock_queue, _, _ = client
        result = {
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "Hello world", "speaker": "SPEAKER_00",
                 "words": [{"word": "Hello", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
                           {"word": "world", "start": 1.1, "end": 2.0, "speaker": "SPEAKER_00"}]},
            ],
            "language": "en",
            "word_segments": [{"word": "Hello", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
                              {"word": "world", "start": 1.1, "end": 2.0, "speaker": "SPEAKER_00"}],
        }

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = client_c.post(
            "/asr?diarize=true",
            files={"audio_file": ("test.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        speakers = {seg.get("speaker") for seg in body["segments"] if seg.get("speaker")}
        assert len(speakers) == 1, \
            f"Single-speaker clip should yield 1 speaker label, got: {speakers}"
