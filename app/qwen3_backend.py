"""Optional Qwen3-ASR backend (ASR_BACKEND=qwen3).

Transcribes with Qwen/Qwen3-ASR-*-hf and produces word timestamps with
Qwen/Qwen3-ForcedAligner-0.6B-hf, both via stock transformers (>= 5.13,
where the qwen3_asr architecture is natively supported). No extra
packages are required and model weights download lazily into the HF
cache like every other model in the service.

Differences from the default whisper (whispermlx) backend:
- The Whisper model name is ignored; QWEN3_ASR_MODEL selects the model.
- task=translate is not supported (the caller falls back to whisper).
- The forced aligner is language-agnostic across its supported set, so
  code-switched audio (e.g. zh/en) aligns without a per-language model.
- hotwords/initial_prompt are passed as free-text context in the system
  message (Qwen3-ASR context biasing).

Requires transformers >= 5.13; runs on cuda, mps, or cpu (float32 on
mps/cpu).
"""

import logging
import os
import threading
import time
from functools import lru_cache

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


SAMPLE_RATE = 16000
# The encoder handles up to ~5 min per pass, but shorter chunks are far less
# prone to early-EOS truncation and repetition loops under greedy decoding.
CHUNK_SECONDS = int(os.getenv("QWEN3_CHUNK_SECONDS", "90") or "90")
MAX_NEW_TOKENS = 4096
# Cap generation relative to chunk duration so a repetition loop cannot blow
# up a chunk (real speech stays well under 20 tokens/second).
TOKENS_PER_SECOND_CAP = 20

DEVICE = os.getenv("DEVICE") or _default_device()


def _env_or_default(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


ASR_MODEL_ID = _env_or_default("QWEN3_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B-hf")
ALIGNER_MODEL_ID = _env_or_default("QWEN3_ALIGNER_MODEL", "Qwen/Qwen3-ForcedAligner-0.6B-hf")

# Execution runtime selection for this backend:
#   auto (default) -> native MLX (via mlx-qwen3-asr, always Metal) when the
#       package is installed and there is no CUDA device (i.e. Apple
#       Silicon, where MLX's fp16 Metal path beats torch MPS); torch otherwise.
#   mlx  -> require the MLX runtime (error at load if missing).
#   torch-> stock transformers (upstream parity; required for float16 CUDA).
# See app/qwen3_mlx_runtime.py for the MLX implementation.
RUNTIME = _env_or_default("QWEN3_RUNTIME", "auto").lower()


@lru_cache(maxsize=1)
def _mlx_package_available() -> bool:
    try:
        import mlx_qwen3_asr  # noqa: F401

        return True
    except Exception as e:
        logger.info(f"mlx_qwen3_asr not available: {e}")
        return False


def _use_mlx_runtime() -> bool:
    if RUNTIME == "mlx":
        return True
    if RUNTIME == "torch":
        return False
    # auto: torch on CUDA (upstream parity, fp16), MLX otherwise.
    return not torch.cuda.is_available() and _mlx_package_available()


# Standing context prepended to every request's system message. Important for
# code-switched audio: with no language hint and no context, Qwen3-ASR picks
# one language per chunk and TRANSLATES the other language into it. Any hint
# (a language, or a note that the audio is mixed) makes it transcribe
# verbatim. Empty by default to leave single-language behaviour untouched.
DEFAULT_CONTEXT = _env_or_default("QWEN3_DEFAULT_CONTEXT", "")

_load_lock = threading.Lock()
_asr = None  # (processor, model)
_aligner = None  # (processor, model)


def _load_asr():
    global _asr
    if _asr is None:
        with _load_lock:
            if _asr is None:
                from transformers import AutoModelForMultimodalLM
                from transformers import AutoProcessor

                logger.info(f"Loading Qwen3-ASR model: {ASR_MODEL_ID}")
                t0 = time.time()
                processor = AutoProcessor.from_pretrained(ASR_MODEL_ID)
                dtype = torch.float16 if DEVICE == "cuda" else torch.float32
                model = AutoModelForMultimodalLM.from_pretrained(ASR_MODEL_ID, dtype=dtype).to(DEVICE)
                model.eval()
                _asr = (processor, model)
                logger.info(f"Qwen3-ASR loaded in {time.time() - t0:.1f}s")
    return _asr


def _load_aligner():
    global _aligner
    if _aligner is None:
        with _load_lock:
            if _aligner is None:
                from transformers import AutoModelForTokenClassification
                from transformers import AutoProcessor

                logger.info(f"Loading Qwen3 forced aligner: {ALIGNER_MODEL_ID}")
                t0 = time.time()
                processor = AutoProcessor.from_pretrained(ALIGNER_MODEL_ID)
                dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
                model = AutoModelForTokenClassification.from_pretrained(ALIGNER_MODEL_ID, dtype=dtype).to(DEVICE)
                model.eval()
                _aligner = (processor, model)
                logger.info(f"Qwen3 forced aligner loaded in {time.time() - t0:.1f}s")
    return _aligner


def _language_name(language: str | None) -> str | None:
    """Resolve a code or name to the canonical full name, None if unknown."""
    if not language:
        return None
    try:
        from transformers.audio_utils import resolve_language
        from transformers.models.qwen3_asr.processing_qwen3_asr import LANGUAGE_CODE_TO_NAME

        return resolve_language(language, LANGUAGE_CODE_TO_NAME, return_code=False)
    except Exception:
        return None


def _language_code(name: str | None) -> str | None:
    """Map a canonical full language name back to its ISO code."""
    if not name:
        return None
    try:
        from transformers.models.qwen3_asr.processing_qwen3_asr import LANGUAGE_CODE_TO_NAME

        for code, full in LANGUAGE_CODE_TO_NAME.items():
            if full.lower() == name.lower():
                return code
    except Exception:
        pass
    return None


def _chunks(audio: np.ndarray) -> list[tuple[float, np.ndarray]]:
    """Split audio into (start_seconds, samples) chunks of CHUNK_SECONDS."""
    size = CHUNK_SECONDS * SAMPLE_RATE
    return [(i / SAMPLE_RATE, audio[i : i + size]) for i in range(0, len(audio), size)]


def transcribe(
    audio: np.ndarray,
    language: str | None = None,
    context: str | None = None,
) -> dict:
    """Transcribe audio in chunks. Returns a whisperx-shaped result dict
    with one segment per chunk (word timestamps come from align())."""
    if _use_mlx_runtime():
        from app import qwen3_mlx_runtime

        return qwen3_mlx_runtime.transcribe(audio, language=language, context=context)

    processor, model = _load_asr()

    # transformers' apply_transcription_request builds the system message
    # from `prompt` and pre-fills the assistant turn with
    # "language <NAME><asr_text>" when a language is given, so code-switched
    # audio transcribes verbatim instead of being translated.
    system_parts = []
    if DEFAULT_CONTEXT:
        system_parts.append(DEFAULT_CONTEXT)
    if context:
        system_parts.append(context)
    prompt = "\n".join(system_parts) if system_parts else None
    lang_name = _language_name(language)

    segments = []
    detected_name = lang_name
    for start_s, chunk in _chunks(audio):
        chunk_s = len(chunk) / SAMPLE_RATE
        end_s = start_s + chunk_s
        text, chunk_lang = _generate(processor, model, chunk, prompt, lang_name)
        # Degenerate-output guard: a long chunk yielding almost no text
        # usually means the model went off-format (e.g. echoed the context
        # instead of transcribing). Retry once without the system message.
        if prompt is not None and chunk_s > 60 and len(text) < chunk_s * 0.5:
            logger.warning(
                f"Qwen3-ASR chunk {start_s:.0f}-{end_s:.0f}s produced only {len(text)} chars; retrying without context"
            )
            retry_text, retry_lang = _generate(processor, model, chunk, None, lang_name)
            if len(retry_text) > len(text):
                text, chunk_lang = retry_text, retry_lang
        if detected_name is None:
            detected_name = chunk_lang
        if text:
            segments.append({"start": start_s, "end": end_s, "text": text})
        logger.info(f"Qwen3-ASR chunk {start_s:.0f}-{end_s:.0f}s: {len(text)} chars, language={chunk_lang}")

    return {
        "segments": segments,
        "language": _language_code(detected_name) or (language or "en"),
        "_asr_backend": "qwen3",
        "_language_name": detected_name,
    }


def _generate(
    processor,
    model,
    chunk: np.ndarray,
    prompt: str | None,
    lang_name: str | None,
) -> tuple[str, str | None]:
    inputs = processor.apply_transcription_request(audio=chunk, language=lang_name, prompt=prompt).to(
        model.device, model.dtype
    )
    max_new = min(
        MAX_NEW_TOKENS,
        int(len(chunk) / SAMPLE_RATE * TOKENS_PER_SECOND_CAP) + 128,
    )
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    parsed = processor.decode(generated_ids[0], return_format="parsed")
    return (parsed.get("transcription") or "").strip(), parsed.get("language")


_PUNCT = ".,!?;:)\"'。！？，、；：”'"


def align(audio: np.ndarray, result: dict) -> dict:
    """Run the Qwen3 forced aligner over each transcribed chunk and
    rebuild segments from word timestamps.

    Under the MLX runtime, results from transcribe() already carry word
    timestamps and pass through unchanged; text-only results (e.g. from
    the external backend) use the native MLX forced aligner."""
    if _use_mlx_runtime():
        from app import qwen3_mlx_runtime

        return qwen3_mlx_runtime.align(audio, result)

    from app.pipeline import _words_to_segments

    processor, model = _load_aligner()

    lang_name = _language_name(result.get("_language_name")) or "English"
    words: list[dict] = []
    for seg in result.get("segments", []):
        text = seg.get("text", "").strip()
        if not text:
            continue
        chunk = audio[int(seg["start"] * SAMPLE_RATE) : int(seg["end"] * SAMPLE_RATE)]
        try:
            aligner_inputs, word_lists = processor.prepare_forced_aligner_inputs(
                audio=chunk, transcript=text, language=lang_name
            )
            aligner_inputs = aligner_inputs.to(model.device, model.dtype)
            with torch.inference_mode():
                outputs = model(**aligner_inputs)
            timestamps = processor.decode_forced_alignment(
                logits=outputs.logits,
                input_ids=aligner_inputs["input_ids"],
                word_lists=word_lists,
                timestamp_token_id=model.config.timestamp_token_id,
            )[0]
        except Exception as e:
            # Keep the chunk's text as a single un-timed span so content is
            # never silently dropped from the transcript.
            logger.warning(
                f"Qwen3 alignment failed for chunk "
                f"{seg['start']:.0f}-{seg['end']:.0f}s: {e}; keeping chunk text unaligned"
            )
            words.append({"word": text, "start": seg["start"], "end": seg["end"]})
            continue
        pos = 0
        for item in timestamps:
            token = item["text"]
            # Re-attach the punctuation the aligner's tokenizer stripped, by
            # walking the original transcript in order.
            idx = text.find(token, pos)
            if idx != -1:
                end_idx = idx + len(token)
                while end_idx < len(text) and text[end_idx] in _PUNCT:
                    token += text[end_idx]
                    end_idx += 1
                pos = end_idx
            words.append(
                {
                    "word": token,
                    "start": round(seg["start"] + item["start_time"], 3),
                    "end": round(seg["start"] + item["end_time"], 3),
                }
            )

    if words:
        result["segments"] = _words_to_segments(words)
        result["word_segments"] = words
    return result


def resegment_by_speaker(result: dict) -> dict:
    """Shared entry point so the diarize stage can resegment qwen3 results
    with the same logic as the whisper backend."""
    from app.pipeline import resegment_by_speaker

    return resegment_by_speaker(result)
