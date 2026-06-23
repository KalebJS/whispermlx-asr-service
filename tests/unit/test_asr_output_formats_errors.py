"""Unit tests for /asr output_format handling, legacy alias, JSON-wrapping,
and error paths (400/413/422).

Covers assertions: VAL-ASR-006, VAL-ASR-007, VAL-ASR-008, VAL-ASR-009,
VAL-ASR-010, VAL-ASR-011, VAL-ASR-027, VAL-ASR-028, VAL-ASR-029,
VAL-ASR-033, VAL-OPS-013, VAL-OPS-014.
"""

from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

FAKE_AUDIO = b"RIFF" + b"\x00" * 100


def _mock_pipeline_result(segments=None, language="en", word_segments=None):
    """Build a realistic pipeline result dict for mocking."""
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
            },
            {
                "start": 2.5,
                "end": 5.0,
                "text": "How are you",
                "words": [
                    {"word": "How", "start": 2.5, "end": 3.0},
                    {"word": "are", "start": 3.1, "end": 3.5},
                    {"word": "you", "start": 3.6, "end": 5.0},
                ],
            },
        ]
    if word_segments is None:
        word_segments = [
            {"word": "Hello", "start": 0.0, "end": 1.2},
            {"word": "world", "start": 1.3, "end": 2.5},
            {"word": "How", "start": 2.5, "end": 3.0},
            {"word": "are", "start": 3.1, "end": 3.5},
            {"word": "you", "start": 3.6, "end": 5.0},
        ]
    return {
        "segments": segments,
        "language": language,
        "word_segments": word_segments,
    }


@pytest.fixture()
def client():
    """Create a TestClient with run_pipeline fully mocked to return controlled results."""
    with (
        patch("app.main.run_in_queue") as mock_queue,
        patch("app.main.whispermlx") as mock_wmlx,
        patch("app.main.resolve_model_name") as mock_resolve,
        patch("app.main.get_canonical_models") as mock_canonical,
    ):
        mock_resolve.side_effect = lambda m: m if m else "large-v3"
        mock_canonical.return_value = [
            "tiny",
            "tiny.en",
            "base",
            "base.en",
            "small",
            "small.en",
            "medium",
            "medium.en",
            "large",
            "large-v1",
            "large-v2",
            "large-v3",
            "large-v3-turbo",
            "turbo",
        ]
        mock_wmlx.load_audio.return_value = np.zeros(16000, dtype=np.float32)

        from app.main import app

        with TestClient(app) as c:
            yield c, mock_queue


def _post_asr(client, params=None, file_name="test.wav", file_content=None):
    """Helper to POST to /asr with optional query params."""
    if file_content is None:
        file_content = FAKE_AUDIO
    url = "/asr"
    if params:
        url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
    return client.post(
        url,
        files={"audio_file": (file_name, file_content, "audio/wav")},
    )


# ===================================================================
# VAL-ASR-006: output_format=text returns joined plain text
# ===================================================================


class TestOutputFormatText:
    """output_format=text returns a JSON object with a single text key
    whose value is the segment texts joined into one plain string."""

    def test_text_format_returns_text_key(self, client):
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c, params={"output_format": "text"})
        assert resp.status_code == 200
        body = resp.json()
        assert "text" in body, f"Missing 'text' key in response: {body}"
        # text must be a string (joined segment texts)
        assert isinstance(body["text"], str), f"text should be str, got {type(body['text'])}"

    def test_text_format_joins_segment_texts(self, client):
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c, params={"output_format": "text"})
        assert resp.status_code == 200
        body = resp.json()
        # The joined text must contain the segment texts
        assert "Hello world" in body["text"], f"Expected 'Hello world' in text, got: {body['text']}"
        assert "How are you" in body["text"], f"Expected 'How are you' in text, got: {body['text']}"

    def test_text_format_only_has_text_key(self, client):
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c, params={"output_format": "text"})
        assert resp.status_code == 200
        body = resp.json()
        # The text format returns a JSON object with a single text key
        assert set(body.keys()) == {"text"}, f"Expected only 'text' key, got: {set(body.keys())}"

    def test_text_format_empty_segments(self, client):
        c, mock_queue = client
        result = _mock_pipeline_result(segments=[], word_segments=[])

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c, params={"output_format": "text"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["text"] == "", f"Expected empty string for empty segments, got: {body['text']}"


# ===================================================================
# VAL-ASR-007: output_format=srt returns valid SRT cues
# ===================================================================


class TestOutputFormatSrt:
    """output_format=srt returns a srt field containing valid SubRip cues."""

    def test_srt_format_returns_srt_key(self, client):
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c, params={"output_format": "srt"})
        assert resp.status_code == 200
        body = resp.json()
        assert "srt" in body, f"Missing 'srt' key in response: {body}"

    def test_srt_has_sequential_index_lines(self, client):
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c, params={"output_format": "srt"})
        assert resp.status_code == 200
        srt = resp.json()["srt"]
        # Each cue must have a sequential integer index
        lines = srt.strip().split("\n")
        # First line should be "1"
        assert lines[0] == "1", f"First SRT index should be '1', got: {lines[0]}"
        # Second cue index should be "2" (after the blank line separator)
        # SRT cue blocks are separated by blank lines
        cues = srt.strip().split("\n\n")
        assert len(cues) >= 1, f"Expected at least 1 SRT cue, got: {cues}"
        for i, cue in enumerate(cues, 1):
            cue_lines = cue.strip().split("\n")
            assert cue_lines[0] == str(i), f"SRT cue {i} index should be '{i}', got: {cue_lines[0]}"

    def test_srt_has_arrow_timestamp_lines(self, client):
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c, params={"output_format": "srt"})
        assert resp.status_code == 200
        srt = resp.json()["srt"]
        cues = srt.strip().split("\n\n")
        import re

        ts_pattern = re.compile(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}")
        for cue in cues:
            cue_lines = cue.strip().split("\n")
            assert len(cue_lines) >= 2, f"SRT cue too short: {cue}"
            assert ts_pattern.match(cue_lines[1]), f"SRT timestamp line malformed: {cue_lines[1]}"

    def test_srt_has_text_lines(self, client):
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c, params={"output_format": "srt"})
        assert resp.status_code == 200
        srt = resp.json()["srt"]
        cues = srt.strip().split("\n\n")
        for cue in cues:
            cue_lines = cue.strip().split("\n")
            assert len(cue_lines) >= 3, f"SRT cue missing text line: {cue}"
            # Text line(s) follow index and timestamp
            text_line = "\n".join(cue_lines[2:])
            assert len(text_line.strip()) > 0, f"SRT cue text is empty: {cue}"

    def test_srt_timestamps_use_comma_separator(self, client):
        """SRT uses comma as the millisecond separator (HH:MM:SS,mmm)."""
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c, params={"output_format": "srt"})
        assert resp.status_code == 200
        srt = resp.json()["srt"]
        # Must not contain period-separated timestamps in SRT timecodes
        import re

        # The arrow line must use commas
        for line in srt.split("\n"):
            if "-->" in line:
                assert "," in line, f"SRT timestamp should use comma: {line}"
                # Should NOT have period between seconds and milliseconds
                # Format: HH:MM:SS,mmm --> HH:MM:SS,mmm
                parts = line.split(" --> ")
                for part in parts:
                    # After the last colon, the separator before ms must be a comma
                    time_part = part.strip()
                    assert re.match(r"\d{2}:\d{2}:\d{2},\d{3}", time_part), f"Bad SRT timestamp format: {time_part}"

    def test_srt_includes_speaker_labels(self, client):
        """When segments have speaker labels, SRT cues prefix text with [SPEAKER_NN]."""
        c, mock_queue = client
        result = _mock_pipeline_result()
        result["segments"][0]["speaker"] = "SPEAKER_00"
        result["segments"][1]["speaker"] = "SPEAKER_01"

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c, params={"output_format": "srt"})
        assert resp.status_code == 200
        srt = resp.json()["srt"]
        assert "[SPEAKER_00]" in srt, f"Expected speaker label in SRT output: {srt}"
        assert "[SPEAKER_01]" in srt, f"Expected speaker label in SRT output: {srt}"


# ===================================================================
# VAL-ASR-008: output_format=vtt returns valid WebVTT with header
# ===================================================================


class TestOutputFormatVtt:
    """output_format=vtt returns a vtt field whose content begins with
    WEBVTT header and contains cues with period-separated timestamps."""

    def test_vtt_format_returns_vtt_key(self, client):
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c, params={"output_format": "vtt"})
        assert resp.status_code == 200
        body = resp.json()
        assert "vtt" in body, f"Missing 'vtt' key in response: {body}"

    def test_vtt_starts_with_webvtt_header(self, client):
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c, params={"output_format": "vtt"})
        assert resp.status_code == 200
        vtt = resp.json()["vtt"]
        assert vtt.startswith("WEBVTT"), f"VTT must start with 'WEBVTT', got: {vtt[:50]}"

    def test_vtt_uses_period_decimal_separator(self, client):
        """VTT uses period as the millisecond separator (HH:MM:SS.mmm)."""
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c, params={"output_format": "vtt"})
        assert resp.status_code == 200
        vtt = resp.json()["vtt"]
        import re

        # VTT timecode pattern: HH:MM:SS.mmm
        for line in vtt.split("\n"):
            if "-->" in line:
                assert "." in line, f"VTT timestamp should use period: {line}"
                parts = line.split(" --> ")
                for part in parts:
                    time_part = part.strip()
                    assert re.match(r"\d{2}:\d{2}:\d{2}\.\d{3}", time_part), f"Bad VTT timestamp format: {time_part}"
                # Must NOT contain commas in timecodes
                assert "," not in line, f"VTT should use period, not comma: {line}"

    def test_vtt_has_valid_cues_with_text(self, client):
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c, params={"output_format": "vtt"})
        assert resp.status_code == 200
        vtt = resp.json()["vtt"]
        # Skip WEBVTT header line and blank line
        lines = vtt.split("\n")
        # After header there should be timestamp lines and text
        found_cue = False
        for line in lines:
            if "-->" in line:
                found_cue = True
                break
        assert found_cue, f"VTT must contain at least one cue: {vtt}"

    def test_vtt_includes_speaker_labels(self, client):
        """When segments have speaker labels, VTT cues prefix text with [SPEAKER_NN]."""
        c, mock_queue = client
        result = _mock_pipeline_result()
        result["segments"][0]["speaker"] = "SPEAKER_00"
        result["segments"][1]["speaker"] = "SPEAKER_01"

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c, params={"output_format": "vtt"})
        assert resp.status_code == 200
        vtt = resp.json()["vtt"]
        assert "[SPEAKER_00]" in vtt, "Expected speaker label in VTT output"
        assert "[SPEAKER_01]" in vtt, "Expected speaker label in VTT output"


# ===================================================================
# VAL-ASR-009: output_format=tsv returns tab-separated rows with header
# ===================================================================


class TestOutputFormatTsv:
    """output_format=tsv returns a tsv field whose first line is the
    header 'start\\tend\\ttext\\tspeaker' followed by one row per segment."""

    def test_tsv_format_returns_tsv_key(self, client):
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c, params={"output_format": "tsv"})
        assert resp.status_code == 200
        body = resp.json()
        assert "tsv" in body, f"Missing 'tsv' key in response: {body}"

    def test_tsv_has_correct_header(self, client):
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c, params={"output_format": "tsv"})
        assert resp.status_code == 200
        tsv = resp.json()["tsv"]
        lines = tsv.rstrip("\n").split("\n")
        assert lines[0] == "start\tend\ttext\tspeaker", f"TSV header wrong: {lines[0]}"

    def test_tsv_has_data_rows(self, client):
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c, params={"output_format": "tsv"})
        assert resp.status_code == 200
        tsv = resp.json()["tsv"]
        lines = tsv.rstrip("\n").split("\n")
        # Header + at least 1 data row
        assert len(lines) >= 2, f"TSV should have header + data rows, got {len(lines)} lines"

    def test_tsv_data_rows_have_4_columns(self, client):
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c, params={"output_format": "tsv"})
        assert resp.status_code == 200
        tsv = resp.json()["tsv"]
        # Use rstrip("\n") instead of strip() to preserve trailing tabs
        lines = tsv.rstrip("\n").split("\n")
        # Skip header
        for line in lines[1:]:
            columns = line.split("\t")
            assert len(columns) == 4, f"TSV row should have 4 columns, got {len(columns)}: {line}"

    def test_tsv_start_end_are_numeric(self, client):
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c, params={"output_format": "tsv"})
        assert resp.status_code == 200
        tsv = resp.json()["tsv"]
        lines = tsv.rstrip("\n").split("\n")
        for line in lines[1:]:
            columns = line.split("\t")
            start_val = float(columns[0])
            end_val = float(columns[1])
            assert end_val >= start_val, f"TSV end ({end_val}) < start ({start_val})"

    def test_tsv_includes_speaker_column(self, client):
        """When segments have speaker labels, TSV speaker column is populated."""
        c, mock_queue = client
        result = _mock_pipeline_result()
        result["segments"][0]["speaker"] = "SPEAKER_00"
        result["segments"][1]["speaker"] = "SPEAKER_01"

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c, params={"output_format": "tsv"})
        assert resp.status_code == 200
        tsv = resp.json()["tsv"]
        lines = tsv.rstrip("\n").split("\n")
        # Data rows should have speaker values
        data_with_speakers = [line for line in lines[1:] if line.split("\t")[3] != ""]
        assert len(data_with_speakers) >= 1, f"Expected speaker labels in TSV: {tsv}"


# ===================================================================
# VAL-ASR-010: Invalid output_format is rejected with 400
# ===================================================================


class TestInvalidOutputFormat:
    """Unsupported output_format returns HTTP 400 with error detail."""

    def test_invalid_format_returns_400(self, client):
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c, params={"output_format": "docx"})
        assert resp.status_code == 400, f"Expected 400, got {resp.status_code}"

    def test_invalid_format_error_mentions_format(self, client):
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c, params={"output_format": "docx"})
        body = resp.json()
        assert "detail" in body, f"Missing 'detail' in error: {body}"
        assert "docx" in body["detail"], f"Error detail should mention the bad format: {body['detail']}"

    def test_various_invalid_formats(self, client):
        """Multiple invalid formats all return 400."""
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        for fmt in ["pdf", "mp3", "wav", "csv", "xml", "html", "FORMAT"]:
            resp = _post_asr(c, params={"output_format": fmt})
            assert resp.status_code == 400, f"output_format={fmt} should return 400, got {resp.status_code}"

    def test_invalid_format_rejected_before_pipeline(self, client):
        """Invalid output_format is rejected before the pipeline runs (no wasted compute).

        The queue/pipeline should never be called when the output_format is invalid.
        """
        c, mock_queue = client

        resp = _post_asr(c, params={"output_format": "docx"})
        assert resp.status_code == 400
        # The pipeline queue must NOT have been called
        mock_queue.assert_not_called()

    def test_invalid_format_via_legacy_output_alias_rejected_before_pipeline(self, client):
        """Legacy `output` alias with an invalid value is also rejected before pipeline."""
        c, mock_queue = client

        resp = _post_asr(c, params={"output": "docx"})
        assert resp.status_code == 400
        mock_queue.assert_not_called()


# ===================================================================
# VAL-ASR-011: Missing audio_file is rejected with 422
# ===================================================================


class TestMissingAudioFile:
    """POST /asr without audio_file returns HTTP 422."""

    def test_missing_audio_file_returns_422(self, client):
        c, mock_queue = client
        resp = c.post("/asr")
        assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"

    def test_missing_audio_file_has_validation_detail(self, client):
        c, mock_queue = client
        resp = c.post("/asr")
        body = resp.json()
        assert "detail" in body, f"Missing 'detail' in 422 response: {body}"


# ===================================================================
# VAL-ASR-027: JSON text field preserves the legacy list shape
# ===================================================================


class TestJsonTextIsArray:
    """For the default output_format=json response, the top-level text
    field is a JSON ARRAY mirroring segments (the legacy shape), NOT a
    joined plain string."""

    def test_json_text_is_array_not_string(self, client):
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c)
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body["text"], list), f"text should be an array, got {type(body['text'])}: {body['text']}"

    def test_json_text_mirrors_segments(self, client):
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c)
        assert resp.status_code == 200
        body = resp.json()
        text = body["text"]
        segments = body["segments"]
        # text array must be the same length as segments
        assert len(text) == len(segments), f"text length ({len(text)}) != segments length ({len(segments)})"
        # Each entry in text must match the corresponding segment
        for i, (t, s) in enumerate(zip(text, segments, strict=False)):
            assert t["text"] == s["text"], f"text[{i}].text ({t.get('text')}) != segments[{i}].text ({s.get('text')})"
            assert t["start"] == s["start"], (
                f"text[{i}].start ({t.get('start')}) != segments[{i}].start ({s.get('start')})"
            )
            assert t["end"] == s["end"], f"text[{i}].end ({t.get('end')}) != segments[{i}].end ({s.get('end')})"

    def test_json_text_is_not_string(self, client):
        """Explicitly verify text is NOT a string (the common mistake)."""
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c)
        assert resp.status_code == 200
        body = resp.json()
        assert not isinstance(body["text"], str), (
            f"text must NOT be a string for output_format=json; got: {body['text']}"
        )


# ===================================================================
# VAL-ASR-028: Legacy output query param aliases output_format
# ===================================================================


class TestLegacyOutputAlias:
    """The legacy `output` query param works like output_format."""

    def test_output_text_alias(self, client):
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c, params={"output": "text"})
        assert resp.status_code == 200
        body = resp.json()
        assert "text" in body, "Legacy output=text should return text key"
        assert isinstance(body["text"], str), "text format should return a string"

    def test_output_srt_alias(self, client):
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c, params={"output": "srt"})
        assert resp.status_code == 200
        body = resp.json()
        assert "srt" in body, "Legacy output=srt should return srt key"

    def test_output_vtt_alias(self, client):
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c, params={"output": "vtt"})
        assert resp.status_code == 200
        body = resp.json()
        assert "vtt" in body, "Legacy output=vtt should return vtt key"
        assert body["vtt"].startswith("WEBVTT"), "VTT should start with WEBVTT"

    def test_output_tsv_alias(self, client):
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c, params={"output": "tsv"})
        assert resp.status_code == 200
        body = resp.json()
        assert "tsv" in body, "Legacy output=tsv should return tsv key"

    def test_output_overrides_output_format(self, client):
        """When both output and output_format are given, output takes precedence."""
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c, params={"output": "srt", "output_format": "text"})
        assert resp.status_code == 200
        body = resp.json()
        # Should return srt format (output overrides output_format)
        assert "srt" in body, f"output should override output_format, got: {body}"

    def test_output_json_alias(self, client):
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c, params={"output": "json"})
        assert resp.status_code == 200
        body = resp.json()
        # json format returns text, language, segments, word_segments
        assert "text" in body, "Missing text key"
        assert "language" in body, "Missing language key"
        assert "segments" in body, "Missing segments key"


# ===================================================================
# VAL-ASR-029: Segments are time-ordered across the whole response
# ===================================================================


class TestSegmentsTimeOrdered:
    """Segments array is globally ordered in time (non-decreasing start)."""

    def test_segments_non_decreasing_start(self, client):
        c, mock_queue = client
        result = _mock_pipeline_result(
            segments=[
                {"start": 0.0, "end": 2.0, "text": "First"},
                {"start": 2.0, "end": 4.0, "text": "Second"},
                {"start": 4.0, "end": 6.0, "text": "Third"},
            ],
            word_segments=[],
        )

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c)
        assert resp.status_code == 200
        body = resp.json()
        segments = body["segments"]
        for i in range(len(segments) - 1):
            assert segments[i]["start"] <= segments[i + 1]["start"], (
                f"Segment {i} start ({segments[i]['start']}) > segment {i + 1} start ({segments[i + 1]['start']})"
            )

    def test_each_segment_end_gte_start(self, client):
        c, mock_queue = client
        result = _mock_pipeline_result(
            segments=[
                {"start": 0.0, "end": 2.0, "text": "First"},
                {"start": 2.0, "end": 4.0, "text": "Second"},
            ],
            word_segments=[],
        )

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c)
        assert resp.status_code == 200
        body = resp.json()
        for seg in body["segments"]:
            assert seg["end"] >= seg["start"], f"Segment end ({seg['end']}) < start ({seg['start']})"


# ===================================================================
# VAL-ASR-033: Non-json /asr formats are returned JSON-wrapped
# ===================================================================


class TestNonJsonFormatsWrapped:
    """Unlike OpenAI endpoints, /asr always returns application/json:
    non-json formats produce a JSON object keyed by the format name."""

    def test_text_format_is_json_wrapped(self, client):
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c, params={"output_format": "text"})
        assert resp.status_code == 200
        assert "application/json" in resp.headers.get("content-type", ""), (
            f"text format should be application/json, got: {resp.headers.get('content-type')}"
        )
        body = resp.json()
        assert "text" in body, "JSON-wrapped text format should have 'text' key"

    def test_srt_format_is_json_wrapped(self, client):
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c, params={"output_format": "srt"})
        assert resp.status_code == 200
        assert "application/json" in resp.headers.get("content-type", ""), (
            f"srt format should be application/json, got: {resp.headers.get('content-type')}"
        )
        body = resp.json()
        assert "srt" in body, "JSON-wrapped srt format should have 'srt' key"

    def test_vtt_format_is_json_wrapped(self, client):
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c, params={"output_format": "vtt"})
        assert resp.status_code == 200
        assert "application/json" in resp.headers.get("content-type", ""), (
            f"vtt format should be application/json, got: {resp.headers.get('content-type')}"
        )
        body = resp.json()
        assert "vtt" in body, "JSON-wrapped vtt format should have 'vtt' key"

    def test_tsv_format_is_json_wrapped(self, client):
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c, params={"output_format": "tsv"})
        assert resp.status_code == 200
        assert "application/json" in resp.headers.get("content-type", ""), (
            f"tsv format should be application/json, got: {resp.headers.get('content-type')}"
        )
        body = resp.json()
        assert "tsv" in body, "JSON-wrapped tsv format should have 'tsv' key"

    def test_json_format_is_also_application_json(self, client):
        """Default json format must also return application/json."""
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c)
        assert resp.status_code == 200
        assert "application/json" in resp.headers.get("content-type", ""), (
            f"json format should be application/json, got: {resp.headers.get('content-type')}"
        )


# ===================================================================
# VAL-OPS-013: Oversized upload returns HTTP 413
# ===================================================================


class TestOversizedUpload:
    """Audio exceeding MAX_FILE_SIZE_MB returns HTTP 413."""

    def test_oversized_file_returns_413(self, client):
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        # Create a file larger than MAX_FILE_SIZE_MB
        # Default MAX_FILE_SIZE_MB is 1000, so we need to override it for testing
        with patch("app.main.MAX_FILE_SIZE_MB", 1):  # 1 MB limit for test
            # Create a 2MB file
            big_content = b"\x00" * (2 * 1024 * 1024)
            resp = _post_asr(c, file_content=big_content)

        assert resp.status_code == 413, f"Expected 413, got {resp.status_code}"

    def test_oversized_file_error_mentions_size(self, client):
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        with patch("app.main.MAX_FILE_SIZE_MB", 1):
            big_content = b"\x00" * (2 * 1024 * 1024)
            resp = _post_asr(c, file_content=big_content)

        assert resp.status_code == 413
        body = resp.json()
        assert "detail" in body
        assert "too large" in body["detail"].lower() or "max" in body["detail"].lower(), (
            f"413 detail should mention size: {body['detail']}"
        )


# ===================================================================
# VAL-OPS-014: Errors return appropriate HTTP status codes
# ===================================================================


class TestErrorStatusCodes:
    """Malformed/invalid requests produce meaningful HTTP error codes."""

    def test_unknown_route_returns_404(self, client):
        c, mock_queue = client
        resp = c.get("/nonexistent-route")
        assert resp.status_code == 404, f"Expected 404 for unknown route, got {resp.status_code}"

    def test_bad_output_format_returns_400(self, client):
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c, params={"output_format": "invalid"})
        assert resp.status_code == 400, f"Expected 400, got {resp.status_code}"

    def test_missing_audio_returns_422(self, client):
        c, mock_queue = client
        resp = c.post("/asr")
        assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"

    def test_oversized_returns_413(self, client):
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        with patch("app.main.MAX_FILE_SIZE_MB", 1):
            big_content = b"\x00" * (2 * 1024 * 1024)
            resp = _post_asr(c, file_content=big_content)

        assert resp.status_code == 413, f"Expected 413, got {resp.status_code}"

    def test_normal_request_returns_200(self, client):
        c, mock_queue = client
        result = _mock_pipeline_result()

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result

        resp = _post_asr(c)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
