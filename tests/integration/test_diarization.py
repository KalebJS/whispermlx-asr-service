"""Integration tests: diarization speaker labels through the real whispermlx path.

Exercises a live uvicorn server with a small MLX model and the multi-speaker
audio fixture, asserting that diarization assigns SPEAKER_NN labels to segments
and words. Requires HF_TOKEN to be set in .env.
"""

from __future__ import annotations

import os
import re

import httpx
import pytest

REQUEST_TIMEOUT = 180  # diarization takes longer (pyannote model load)

SPEAKER_PATTERN = re.compile(r"^SPEAKER_\d+$")


def _has_hf_token() -> bool:
    """Check if HF_TOKEN is available (without printing it)."""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env_file = os.path.join(repo_root, ".env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                if line.strip().startswith("HF_TOKEN="):
                    val = line.strip().split("=", 1)[1]
                    if val and val != "your-hf-token-here":
                        return True
    return bool(os.getenv("HF_TOKEN"))


# Skip the entire module if no HF_TOKEN or no multi-speaker fixture
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _has_hf_token(), reason="HF_TOKEN not set; diarization tests require it"),
]


class TestDiarization:
    """Diarization assigns speaker labels through the real MLX + pyannote path."""

    def test_multispeaker_yields_multiple_speakers(self, server_url: str, multispeaker_audio: str):
        """diarize=true on a multi-speaker clip yields >=2 distinct SPEAKER_NN labels."""
        with open(multispeaker_audio, "rb") as f:
            resp = httpx.post(
                f"{server_url}/asr",
                files={"audio_file": (multispeaker_audio, f, "audio/wav")},
                params={"output_format": "json", "diarize": "true"},
                timeout=REQUEST_TIMEOUT,
            )
        assert resp.status_code == 200
        body = resp.json()
        segments = body["segments"]
        assert len(segments) > 0, "No segments returned"

        speakers = {seg.get("speaker") for seg in segments if seg.get("speaker")}
        assert len(speakers) >= 2, f"Expected >=2 distinct speakers, got {speakers}"

    def test_every_segment_has_speaker_label(self, server_url: str, multispeaker_audio: str):
        """Every segment carries a non-empty speaker field when diarization is on."""
        with open(multispeaker_audio, "rb") as f:
            resp = httpx.post(
                f"{server_url}/asr",
                files={"audio_file": (multispeaker_audio, f, "audio/wav")},
                params={"output_format": "json", "diarize": "true"},
                timeout=REQUEST_TIMEOUT,
            )
        assert resp.status_code == 200
        segments = resp.json()["segments"]
        for seg in segments:
            speaker = seg.get("speaker")
            assert speaker, f"Segment missing speaker label: {seg.get('text', '')[:50]}"

    def test_speaker_labels_match_speaker_nn_format(self, server_url: str, multispeaker_audio: str):
        """Speaker labels follow the SPEAKER_NN pattern."""
        with open(multispeaker_audio, "rb") as f:
            resp = httpx.post(
                f"{server_url}/asr",
                files={"audio_file": (multispeaker_audio, f, "audio/wav")},
                params={"output_format": "json", "diarize": "true"},
                timeout=REQUEST_TIMEOUT,
            )
        assert resp.status_code == 200
        segments = resp.json()["segments"]
        for seg in segments:
            speaker = seg.get("speaker")
            if speaker:
                assert SPEAKER_PATTERN.match(speaker), f"Speaker label '{speaker}' does not match SPEAKER_NN"

    def test_word_speaker_labels_subset_of_segment_speakers(self, server_url: str, multispeaker_audio: str):
        """With word_timestamps + diarization, word speakers are a subset of segment speakers."""
        with open(multispeaker_audio, "rb") as f:
            resp = httpx.post(
                f"{server_url}/asr",
                files={"audio_file": (multispeaker_audio, f, "audio/wav")},
                params={"output_format": "json", "diarize": "true", "word_timestamps": "true"},
                timeout=REQUEST_TIMEOUT,
            )
        assert resp.status_code == 200
        body = resp.json()
        segments = body["segments"]

        segment_speakers = {seg.get("speaker") for seg in segments if seg.get("speaker")}
        assert len(segment_speakers) >= 2

        # Collect word-level speakers
        word_speakers = set()
        for seg in segments:
            for w in seg.get("words", []):
                sp = w.get("speaker")
                if sp:
                    word_speakers.add(sp)

        if word_speakers:
            # Word speakers must be a subset of segment speakers
            assert word_speakers.issubset(segment_speakers), (
                f"Word speakers {word_speakers} not subset of segment speakers {segment_speakers}"
            )

    def test_diarize_false_omits_speakers(self, server_url: str, multispeaker_audio: str):
        """diarize=false produces no speaker labels."""
        with open(multispeaker_audio, "rb") as f:
            resp = httpx.post(
                f"{server_url}/asr",
                files={"audio_file": (multispeaker_audio, f, "audio/wav")},
                params={"output_format": "json", "diarize": "false"},
                timeout=REQUEST_TIMEOUT,
            )
        assert resp.status_code == 200
        segments = resp.json()["segments"]
        for seg in segments:
            assert not seg.get("speaker"), f"Speaker label present despite diarize=false: {seg.get('speaker')}"

    def test_diarization_defaults_on(self, server_url: str, multispeaker_audio: str):
        """When no diarize flag is supplied, diarization defaults ON (speaker labels present)."""
        with open(multispeaker_audio, "rb") as f:
            resp = httpx.post(
                f"{server_url}/asr",
                files={"audio_file": (multispeaker_audio, f, "audio/wav")},
                params={"output_format": "json"},
                timeout=REQUEST_TIMEOUT,
            )
        assert resp.status_code == 200
        segments = resp.json()["segments"]
        speakers = {seg.get("speaker") for seg in segments if seg.get("speaker")}
        assert len(speakers) >= 1, "Diarization defaulted off; expected speaker labels"

    def test_single_speaker_yields_one_label(self, server_url: str, sample_audio: str):
        """A single-speaker clip yields exactly 1 distinct speaker label."""
        with open(sample_audio, "rb") as f:
            resp = httpx.post(
                f"{server_url}/asr",
                files={"audio_file": (sample_audio, f, "audio/wav")},
                params={"output_format": "json", "diarize": "true"},
                timeout=REQUEST_TIMEOUT,
            )
        assert resp.status_code == 200
        segments = resp.json()["segments"]
        speakers = {seg.get("speaker") for seg in segments if seg.get("speaker")}
        assert len(speakers) == 1, f"Expected 1 speaker for single-speaker clip, got {speakers}"

    def test_speaker_labels_in_srt_format(self, server_url: str, multispeaker_audio: str):
        """Speaker labels appear in SRT output as [SPEAKER_NN] prefixes."""
        with open(multispeaker_audio, "rb") as f:
            resp = httpx.post(
                f"{server_url}/asr",
                files={"audio_file": (multispeaker_audio, f, "audio/wav")},
                params={"output_format": "srt", "diarize": "true"},
                timeout=REQUEST_TIMEOUT,
            )
        assert resp.status_code == 200
        srt = resp.json().get("srt", "")
        assert "[SPEAKER_" in srt, "No speaker labels found in SRT output"

    def test_num_speakers_constrains_count(self, server_url: str, multispeaker_audio: str):
        """num_speakers=2 constrains to exactly 2 distinct speakers."""
        with open(multispeaker_audio, "rb") as f:
            resp = httpx.post(
                f"{server_url}/asr",
                files={"audio_file": (multispeaker_audio, f, "audio/wav")},
                params={"output_format": "json", "diarize": "true", "num_speakers": "2"},
                timeout=REQUEST_TIMEOUT,
            )
        assert resp.status_code == 200
        segments = resp.json()["segments"]
        speakers = {seg.get("speaker") for seg in segments if seg.get("speaker")}
        assert len(speakers) == 2, f"Expected exactly 2 speakers with num_speakers=2, got {speakers}"

    def test_enable_diarization_alias(self, server_url: str, multispeaker_audio: str):
        """enable_diarization=true behaves identically to diarize=true."""
        with open(multispeaker_audio, "rb") as f:
            resp = httpx.post(
                f"{server_url}/asr",
                files={"audio_file": (multispeaker_audio, f, "audio/wav")},
                params={"output_format": "json", "enable_diarization": "true"},
                timeout=REQUEST_TIMEOUT,
            )
        assert resp.status_code == 200
        segments = resp.json()["segments"]
        speakers = {seg.get("speaker") for seg in segments if seg.get("speaker")}
        assert len(speakers) >= 2, f"enable_diarization=true should produce >=2 speakers, got {speakers}"
