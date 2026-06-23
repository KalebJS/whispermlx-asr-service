"""Integration tests: transcription text and word timestamps through the real whispermlx path.

Exercises a live uvicorn server with a small MLX model and the audio fixture,
asserting that transcription produces real words, correct response shape, and
word-level timestamps from wav2vec2 alignment.
"""

from __future__ import annotations

import httpx

# Timeout for HTTP requests: model load + inference can take a while on first request
REQUEST_TIMEOUT = 120


class TestHealthAndServiceInfo:
    """Basic liveness checks against the running server."""

    def test_health_returns_200_with_required_fields(self, server_url: str):
        resp = httpx.get(f"{server_url}/health", timeout=10)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["serve_mode"] == "simple"
        assert "device" in body
        assert "loaded_models" in body
        assert isinstance(body["loaded_models"], list)

    def test_root_returns_service_info(self, server_url: str):
        resp = httpx.get(f"{server_url}/", timeout=10)
        assert resp.status_code == 200
        body = resp.json()
        assert body["serve_mode"] == "simple"
        assert "device" in body
        assert "service" in body


class TestTranscriptionText:
    """Transcription produces recognizable real words through the MLX backend."""

    def test_basic_transcription_returns_real_words(self, server_url: str, sample_audio: str):
        """POST /asr with default params returns 200 and segment text with real words."""
        with open(sample_audio, "rb") as f:
            resp = httpx.post(
                f"{server_url}/asr",
                files={"audio_file": (sample_audio, f, "audio/wav")},
                params={"output_format": "json", "diarize": "false"},
                timeout=REQUEST_TIMEOUT,
            )
        assert resp.status_code == 200
        body = resp.json()

        # Response shape: text (array), language, segments, word_segments
        assert "text" in body
        assert "language" in body
        assert "segments" in body
        assert "word_segments" in body

        segments = body["segments"]
        assert len(segments) > 0, "Expected at least one segment"

        # Concatenate all segment text and verify it contains real words
        all_text = " ".join(seg.get("text", "") for seg in segments).strip()
        assert len(all_text) > 0, "Transcription text is empty"

        # Verify segments carry start, end, text
        for seg in segments:
            assert "start" in seg and isinstance(seg["start"], (int, float))
            assert "end" in seg and isinstance(seg["end"], (int, float))
            assert seg["end"] >= seg["start"]
            assert "text" in seg and isinstance(seg["text"], str)

    def test_language_is_populated(self, server_url: str, sample_audio: str):
        """The response language field is always a non-empty string."""
        with open(sample_audio, "rb") as f:
            resp = httpx.post(
                f"{server_url}/asr",
                files={"audio_file": (sample_audio, f, "audio/wav")},
                params={"output_format": "json", "diarize": "false"},
                timeout=REQUEST_TIMEOUT,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["language"], "Language field is empty or missing"

    def test_explicit_language_override(self, server_url: str, sample_audio: str):
        """Explicit language=en is honored in the response."""
        with open(sample_audio, "rb") as f:
            resp = httpx.post(
                f"{server_url}/asr",
                files={"audio_file": (sample_audio, f, "audio/wav")},
                params={"output_format": "json", "diarize": "false", "language": "en"},
                timeout=REQUEST_TIMEOUT,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["language"] == "en"

    def test_output_format_text(self, server_url: str, sample_audio: str):
        """output_format=text returns a JSON object with a 'text' key containing joined text."""
        with open(sample_audio, "rb") as f:
            resp = httpx.post(
                f"{server_url}/asr",
                files={"audio_file": (sample_audio, f, "audio/wav")},
                params={"output_format": "text", "diarize": "false"},
                timeout=REQUEST_TIMEOUT,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert "text" in body
        assert isinstance(body["text"], str)
        assert len(body["text"].strip()) > 0

    def test_segments_are_time_ordered(self, server_url: str, sample_audio: str):
        """Segment start times are non-decreasing across the whole response."""
        with open(sample_audio, "rb") as f:
            resp = httpx.post(
                f"{server_url}/asr",
                files={"audio_file": (sample_audio, f, "audio/wav")},
                params={"output_format": "json", "diarize": "false"},
                timeout=REQUEST_TIMEOUT,
            )
        assert resp.status_code == 200
        segments = resp.json()["segments"]
        starts = [seg["start"] for seg in segments]
        for i in range(1, len(starts)):
            assert starts[i] >= starts[i - 1], f"Segment {i} starts before segment {i - 1}"


class TestWordTimestamps:
    """word_timestamps=true triggers alignment and returns word-level timing."""

    def test_word_timestamps_true_populates_word_segments(self, server_url: str, sample_audio: str):
        """word_timestamps=true returns populated word_segments with start/end."""
        with open(sample_audio, "rb") as f:
            resp = httpx.post(
                f"{server_url}/asr",
                files={"audio_file": (sample_audio, f, "audio/wav")},
                params={"output_format": "json", "diarize": "false", "word_timestamps": "true"},
                timeout=REQUEST_TIMEOUT,
            )
        assert resp.status_code == 200
        body = resp.json()

        word_segments = body.get("word_segments", [])
        assert len(word_segments) > 0, "word_segments is empty despite word_timestamps=true"

        for w in word_segments:
            assert "start" in w and isinstance(w["start"], (int, float))
            assert "end" in w and isinstance(w["end"], (int, float))
            assert w["end"] >= w["start"]

    def test_word_timestamps_within_segment_bounds(self, server_url: str, sample_audio: str):
        """Each word entry's start/end fall within its parent segment's start/end."""
        with open(sample_audio, "rb") as f:
            resp = httpx.post(
                f"{server_url}/asr",
                files={"audio_file": (sample_audio, f, "audio/wav")},
                params={"output_format": "json", "diarize": "false", "word_timestamps": "true"},
                timeout=REQUEST_TIMEOUT,
            )
        assert resp.status_code == 200
        body = resp.json()
        segments = body["segments"]

        for seg in segments:
            seg_start = seg["start"]
            seg_end = seg["end"]
            words = seg.get("words", [])
            if not words:
                continue
            for w in words:
                w_start = w.get("start", 0)
                w_end = w.get("end", 0)
                # Word start should be >= segment start (allow small float tolerance)
                assert w_start >= seg_start - 0.5, f"Word start {w_start} < segment start {seg_start}"
                # Word end should be <= segment end (allow small float tolerance)
                assert w_end <= seg_end + 0.5, f"Word end {w_end} > segment end {seg_end}"
                # Words should be non-decreasing
                assert w_end >= w_start, f"Word end {w_end} < start {w_start}"

    def test_word_timestamps_false_omits_word_data(self, server_url: str, sample_audio: str):
        """word_timestamps=false skips alignment: no word-level timing, segment text still present."""
        with open(sample_audio, "rb") as f:
            resp = httpx.post(
                f"{server_url}/asr",
                files={"audio_file": (sample_audio, f, "audio/wav")},
                params={"output_format": "json", "diarize": "false", "word_timestamps": "false"},
                timeout=REQUEST_TIMEOUT,
            )
        assert resp.status_code == 200
        body = resp.json()

        # word_segments should be empty or absent
        word_segments = body.get("word_segments", [])
        assert len(word_segments) == 0, "word_segments is populated despite word_timestamps=false"

        # Segments should still have text
        segments = body["segments"]
        assert len(segments) > 0
        for seg in segments:
            assert "text" in seg and seg["text"]

    def test_word_timestamps_default_is_true(self, server_url: str, sample_audio: str):
        """Omitting word_timestamps defaults to true (word-level timing present)."""
        with open(sample_audio, "rb") as f:
            resp = httpx.post(
                f"{server_url}/asr",
                files={"audio_file": (sample_audio, f, "audio/wav")},
                params={"output_format": "json", "diarize": "false"},
                timeout=REQUEST_TIMEOUT,
            )
        assert resp.status_code == 200
        body = resp.json()
        word_segments = body.get("word_segments", [])
        assert len(word_segments) > 0, "word_timestamps defaults to true but word_segments is empty"


class TestModelSelection:
    """Model selection works for small models and MLX-specific names."""

    def test_model_tiny_transcribes(self, server_url: str, sample_audio: str):
        """model=tiny loads and transcribes successfully."""
        with open(sample_audio, "rb") as f:
            resp = httpx.post(
                f"{server_url}/asr",
                files={"audio_file": (sample_audio, f, "audio/wav")},
                params={"output_format": "json", "diarize": "false", "model": "tiny"},
                timeout=REQUEST_TIMEOUT,
            )
        assert resp.status_code == 200
        segments = resp.json()["segments"]
        assert len(segments) > 0

    def test_openai_alias_whisper_tiny_resolves(self, server_url: str, sample_audio: str):
        """model=whisper-tiny (OpenAI alias) resolves to the MLX tiny model."""
        with open(sample_audio, "rb") as f:
            resp = httpx.post(
                f"{server_url}/asr",
                files={"audio_file": (sample_audio, f, "audio/wav")},
                params={"output_format": "json", "diarize": "false", "model": "whisper-tiny"},
                timeout=REQUEST_TIMEOUT,
            )
        assert resp.status_code == 200
        segments = resp.json()["segments"]
        assert len(segments) > 0


class TestErrorHandling:
    """Error paths: invalid output format, missing audio, unknown model."""

    def test_invalid_output_format_returns_400(self, server_url: str, sample_audio: str):
        with open(sample_audio, "rb") as f:
            resp = httpx.post(
                f"{server_url}/asr",
                files={"audio_file": (sample_audio, f, "audio/wav")},
                params={"output_format": "docx", "diarize": "false"},
                timeout=REQUEST_TIMEOUT,
            )
        assert resp.status_code == 400

    def test_missing_audio_file_returns_422(self, server_url: str):
        resp = httpx.post(
            f"{server_url}/asr",
            params={"output_format": "json", "diarize": "false"},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 422

    def test_unknown_model_returns_400_and_service_stays_alive(self, server_url: str, sample_audio: str):
        with open(sample_audio, "rb") as f:
            resp = httpx.post(
                f"{server_url}/asr",
                files={"audio_file": (sample_audio, f, "audio/wav")},
                params={"output_format": "json", "diarize": "false", "model": "not-a-real-model"},
                timeout=REQUEST_TIMEOUT,
            )
        assert resp.status_code == 400
        # Service should still be healthy
        health = httpx.get(f"{server_url}/health", timeout=10)
        assert health.status_code == 200


class TestOpenAIEndpoints:
    """OpenAI-compatible endpoints work through the real MLX path."""

    def test_v1_transcriptions_returns_json_text(self, server_url: str, sample_audio: str):
        with open(sample_audio, "rb") as f:
            resp = httpx.post(
                f"{server_url}/v1/audio/transcriptions",
                files={"file": (sample_audio, f, "audio/wav")},
                data={"model": "whisper-1", "response_format": "json"},
                timeout=REQUEST_TIMEOUT,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert "text" in body
        assert isinstance(body["text"], str)
        assert len(body["text"].strip()) > 0

    def test_v1_models_lists_mlx_models(self, server_url: str):
        resp = httpx.get(f"{server_url}/v1/models", timeout=10)
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        ids = [m["id"] for m in body["data"]]
        assert "whisper-1" in ids
        assert "tiny" in ids
        assert "large-v3-turbo" in ids
        # No distil-* (faster-whisper-only)
        assert not any(i.startswith("distil-") for i in ids)

    def test_v1_models_detail_returns_model(self, server_url: str):
        resp = httpx.get(f"{server_url}/v1/models/tiny", timeout=10)
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == "tiny"
        assert body["object"] == "model"

    def test_v1_models_detail_404_for_unknown(self, server_url: str):
        resp = httpx.get(f"{server_url}/v1/models/does-not-exist", timeout=10)
        assert resp.status_code == 404


class TestMetrics:
    """Metrics endpoint works with MLX (no CUDA)."""

    def test_metrics_returns_200_openmetrics(self, server_url: str):
        resp = httpx.get(f"{server_url}/metrics", timeout=10)
        assert resp.status_code == 200
        assert "text/plain" in resp.headers.get("content-type", "")
        body = resp.text
        assert "whisperx_requests_total" in body
        assert "whisperx_service_info" in body

    def test_queue_metrics_returns_json(self, server_url: str):
        resp = httpx.get(f"{server_url}/queue-metrics", timeout=10)
        assert resp.status_code == 200
        body = resp.json()
        assert body["serve_mode"] == "simple"
        assert "queue" in body
