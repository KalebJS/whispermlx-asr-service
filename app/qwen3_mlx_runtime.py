"""MLX execution runtime for the Qwen3-ASR backend (QWEN3_RUNTIME=mlx).

Runs the same Qwen3-ASR transcription + Qwen3 forced aligner as the torch
path, but natively in MLX on the Metal GPU -- no torch/transformers in the
inference path. Uses the `mlx-qwen3-asr` runtime (Moona3k's ground-up MLX
reimplementation, validated against the official PyTorch model) together
with its own on-the-fly weight conversion of the official checkpoints.

Model IDs: this runtime accepts the official `Qwen/Qwen3-ASR-*` repos
(converts weights on first load) and the pre-converted
`mlx-community/Qwen3-ASR-*` / `moona3k/mlx-qwen3-asr-*` checkpoints, the
quantized ones being fastest:

- fp16  0.6B ~1.2 GB, 1.7B ~3.4 GB
- 8-bit is lossless vs fp16 and ~1.3x faster; 4-bit is ~1.7x faster

Behaviour differences vs the torch path (same Qwen3 model family):
- Word timestamps come straight from session transcription (native MLX
  aligner internally), so the standalone aligner is only needed when a
  caller hands us segments without words (e.g. external text-only mode).
- Transcription chunking is energy-based (at pauses), not fixed 90 s
  chunks, so there is no one-segment-per-chunk output.

All mlx_qwen3_asr imports are lazy so the torch path works without the
package and unit tests can mock it.
"""

import logging
import os
import threading

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


def _env_or_default(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


# Official (non-hf) repos: the mlx runtime converts their weights on first
# load. mlx-community/moona3k quantized checkpoints are also supported.
ASR_MODEL_ID = _env_or_default("QWEN3_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")
ALIGNER_MODEL_ID = _env_or_default("QWEN3_ALIGNER_MODEL", "Qwen/Qwen3-ForcedAligner-0.6B")
DEFAULT_CONTEXT = _env_or_default("QWEN3_DEFAULT_CONTEXT", "")

_load_lock = threading.Lock()
_asr_session = None
_aligner = None


def _get_session():
    """Load (once) the MLX transcription session. Not evicted; the whisper
    model caches are the only evictable stores."""
    global _asr_session
    if _asr_session is None:
        with _load_lock:
            if _asr_session is None:
                import mlx.core as mx
                from mlx_qwen3_asr import Session

                logger.info(f"Loading Qwen3-ASR MLX session: {ASR_MODEL_ID}")
                _asr_session = Session(model=ASR_MODEL_ID, dtype=mx.float16)
                logger.info("Qwen3-ASR MLX session loaded")
    return _asr_session


def _get_aligner():
    global _aligner
    if _aligner is None:
        with _load_lock:
            if _aligner is None:
                from mlx_qwen3_asr import ForcedAligner

                logger.info(f"Loading Qwen3 forced aligner (MLX): {ALIGNER_MODEL_ID}")
                _aligner = ForcedAligner(model_path=ALIGNER_MODEL_ID)
    return _aligner


def transcribe(
    audio,  # np.ndarray at SAMPLE_RATE
    language: str | None = None,
    context: str | None = None,
) -> dict:
    """Transcribe in one MLX pass with word-level timestamps.

    Returns a whisperx-shaped dict whose segments already carry words, so
    the caller's align stage is a no-op. Chunking is energy-based inside
    the runtime, not the torch path's fixed CHUNK_SECONDS windows.
    """
    from app import qwen3_backend

    session = _get_session()

    # Context biasing, same composition as the torch path.
    system_parts = []
    if DEFAULT_CONTEXT:
        system_parts.append(DEFAULT_CONTEXT)
    if context:
        system_parts.append(context)
    prompt = "\n".join(system_parts)
    lang_name = qwen3_backend._language_name(language)

    logger.info("Starting transcription (qwen3 MLX runtime)...")
    result = session.transcribe(audio, language=lang_name, context=prompt, return_timestamps=True)
    logger.info(f"Transcription complete. Detected language: {result.language}")

    words = []
    for w in result.segments or []:
        text = (w.get("text") or "").strip()
        if not text or "start" not in w or "end" not in w:
            continue
        words.append({"word": text, "start": float(w["start"]), "end": float(w["end"])})

    if words:
        # Rebuild segments at pause/speaker/punctuation boundaries from the
        # word timestamps (shared logic with the torch path).
        from app.pipeline import _words_to_segments

        segments = _words_to_segments(words)
    elif (result.text or "").strip():
        # Degenerate: no aligned words but text present -> one chunk-span segment
        segments = [{"start": 0.0, "end": len(audio) / SAMPLE_RATE, "text": result.text.strip()}]
    else:
        segments = []

    return {
        "segments": segments,
        "word_segments": words,
        "language": qwen3_backend._language_code(result.language) or (language or "en"),
        "_asr_backend": "qwen3",
        "_asr_runtime": "mlx",
        "_language_name": result.language,
    }


def align(audio, result: dict) -> dict:
    """Word-align results that lack words (e.g. external text-only segments).

    Results produced by transcribe() already carry word timestamps and are
    returned unchanged.
    """
    segments = result.get("segments", [])
    if any(seg.get("words") for seg in segments):
        return result

    from app.pipeline import _words_to_segments

    aligner = _get_aligner()

    from app import qwen3_backend

    lang_name = result.get("_language_name") or qwen3_backend._language_name(result.get("language")) or "English"

    words = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        chunk = audio[int(seg["start"] * SAMPLE_RATE) : int(seg["end"] * SAMPLE_RATE)]
        try:
            aligned = aligner.align(chunk, text, lang_name)
        except Exception as e:
            # Keep the segment's text as one un-timed span so content is
            # never silently dropped from the transcript.
            logger.warning(
                f"MLX forced alignment failed for chunk "
                f"{seg['start']:.0f}-{seg['end']:.0f}s: {e}; keeping chunk text unaligned"
            )
            words.append({"word": text, "start": seg["start"], "end": seg["end"]})
            continue
        for item in aligned:
            # AlignedWord times are relative to the passed chunk, so offset
            # by the segment start (same semantics as the torch path).
            words.append(
                {
                    "word": item.text,
                    "start": round(float(seg["start"]) + float(item.start_time), 3),
                    "end": round(float(seg["start"]) + float(item.end_time), 3),
                }
            )

    if words:
        result["segments"] = _words_to_segments(words)
        result["word_segments"] = words
    return result
