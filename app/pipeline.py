"""
Shared ASR pipeline stage functions.

Extracts the 3-stage pipeline (transcribe -> align -> diarize) into
reusable functions consumed by the FastAPI endpoints.
Powered by whispermlx (MLX backend on Apple Silicon).
"""

import gc
import logging
import math
import os
import threading
import time
import warnings
from typing import Any

# Suppress pyannote's torchcodec warning -- we decode audio via whispermlx.load_audio (ffmpeg),
# not pyannote's built-in decoder, so the missing torchcodec is irrelevant.
warnings.filterwarnings("ignore", message=".*torchcodec.*")

import numpy as np
import whispermlx
from whispermlx.diarize import DiarizationPipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (read once at import time, same as before)
# ---------------------------------------------------------------------------
DEVICE = os.getenv("DEVICE", "cpu")
# COMPUTE_TYPE and BATCH_SIZE are accepted for API compatibility with the
# original CUDA-based service but are INERT under the MLX backend.  Setting
# them will not error, but they have no effect on inference behaviour.
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))
HF_TOKEN = os.getenv("HF_TOKEN", None)
CACHE_DIR = os.getenv("CACHE_DIR", os.path.expanduser("~/.cache/whisperx-asr"))
DEFAULT_MODEL = os.getenv("PRELOAD_MODEL", "large-v3")

# Idle model eviction. Set MODEL_KEEP_ALIVE_SECONDS > 0 to unload Whisper
# models that have not been used in that many seconds. Floor of 30s on the
# sweep interval to avoid pegging a thread on tight loops.
MODEL_KEEP_ALIVE_SECONDS = int(os.getenv("MODEL_KEEP_ALIVE_SECONDS", "0"))
MODEL_EVICTION_INTERVAL_SECONDS = max(30, int(os.getenv("MODEL_EVICTION_INTERVAL_SECONDS", "60")))

# MLX model map: short names → HuggingFace repo IDs for the MLX backend.
# Sourced from whispermlx.asr.MLX_MODEL_MAP. Duplicated here so that
# get_canonical_models() and resolve_model_name() work without importing
# faster_whisper. Keep in sync with the upstream whispermlx package.
MLX_MODEL_MAP = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "tiny.en": "mlx-community/whisper-tiny.en-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "base.en": "mlx-community/whisper-base.en-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "small.en": "mlx-community/whisper-small.en-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "medium.en": "mlx-community/whisper-medium.en-mlx",
    "large": "mlx-community/whisper-large-mlx",
    "large-v1": "mlx-community/whisper-large-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}


def get_canonical_models() -> list:
    """
    Canonical model names accepted by the whispermlx MLX backend.

    Sourced from the MLX model map keys.
    """
    return list(MLX_MODEL_MAP.keys())


# OpenAI-style aliases → canonical MLX model names. These are kept for
# backwards compatibility on the request path; new clients should use the
# canonical names returned by /v1/models.
_MODEL_ALIASES = {
    "whisper-1": os.getenv("OPENAI_WHISPER1_MODEL", DEFAULT_MODEL),
    "whisper-large-v3": "large-v3",
    "whisper-large-v2": "large-v2",
    "whisper-medium": "medium",
    "whisper-small": "small",
    "whisper-base": "base",
    "whisper-tiny": "tiny",
}


def resolve_model_name(model: str) -> str:
    """
    Resolve a user-supplied model identifier to a canonical MLX model name.

    Accepts canonical names (tiny, large-v3, ...) as-is and maps OpenAI-style
    aliases (whisper-tiny, whisper-large-v3, ...) to their canonical equivalents.
    Unknown values are returned unchanged so the engine can produce its own
    validation error.
    """
    if not model:
        return DEFAULT_MODEL
    canonical = set(get_canonical_models())
    if model in canonical:
        return model
    if model in _MODEL_ALIASES:
        return _MODEL_ALIASES[model]
    if model.startswith("whisper-"):
        stripped = model[len("whisper-") :]
        if stripped in canonical:
            return stripped
    return model


_model_load_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Model caches
# ---------------------------------------------------------------------------
_whisper_models: dict[str, Any] = {}
_whisper_models_last_used: dict[str, float] = {}
_align_models: dict[str, tuple[Any, Any]] = {}
_diarize_pipeline: DiarizationPipeline | None = None

_eviction_thread_lock = threading.Lock()
_eviction_thread_started = False


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------
def clear_gpu_memory():
    """Clear GPU memory cache to prevent VRAM buildup.

    Uses gc.collect() plus a guarded MLX cache clear.
    MLX inference runs on the Metal GPU automatically; this releases
    MLX-allocated buffers that are no longer referenced.
    """
    gc.collect()
    try:
        import mlx.core

        if hasattr(mlx.core, "clear_cache"):
            mlx.core.clear_cache()
    except Exception:
        pass
    logger.debug("GPU memory cache cleared")


# ---------------------------------------------------------------------------
# Stage 0 -- model loading
# ---------------------------------------------------------------------------
def load_whisper_model(model_name: str):
    """Load whispermlx model with caching (thread-safe)."""
    if model_name not in _whisper_models:
        with _model_load_lock:
            if model_name not in _whisper_models:
                logger.info(f"Loading whispermlx model: {model_name}")
                model = whispermlx.load_model(
                    model_name,
                    device=DEVICE,
                )
                _whisper_models[model_name] = model
                logger.info(f"Model {model_name} loaded successfully")
                # Pre-register the eviction counter time series for this model
                # so the row appears in /metrics with value 0 from the moment
                # the model is loaded, instead of only after the first eviction.
                try:
                    from app import metrics as prom_metrics

                    prom_metrics.MODEL_EVICTIONS_TOTAL.labels(model=model_name)
                except Exception:
                    pass
    _whisper_models_last_used[model_name] = time.time()
    _ensure_eviction_thread()
    return _whisper_models[model_name]


def _ensure_eviction_thread():
    """Lazily start the idle-model eviction daemon (no-op if disabled)."""
    global _eviction_thread_started
    if MODEL_KEEP_ALIVE_SECONDS <= 0 or _eviction_thread_started:
        return
    with _eviction_thread_lock:
        if _eviction_thread_started:
            return
        t = threading.Thread(target=_eviction_loop, daemon=True, name="model-evictor")
        t.start()
        _eviction_thread_started = True
        logger.info(
            f"Idle model eviction enabled: unload after "
            f"{MODEL_KEEP_ALIVE_SECONDS}s idle, sweep every "
            f"{MODEL_EVICTION_INTERVAL_SECONDS}s"
        )


def _eviction_loop():
    while True:
        time.sleep(MODEL_EVICTION_INTERVAL_SECONDS)
        if MODEL_KEEP_ALIVE_SECONDS <= 0:
            continue
        now = time.time()
        candidates = [
            name
            for name, last in list(_whisper_models_last_used.items())
            if now - last > MODEL_KEEP_ALIVE_SECONDS and name in _whisper_models
        ]
        evicted_any = False
        for name in candidates:
            with _model_load_lock:
                last = _whisper_models_last_used.get(name, 0)
                if name in _whisper_models and now - last > MODEL_KEEP_ALIVE_SECONDS:
                    logger.info(f"Evicting idle model {name}")
                    del _whisper_models[name]
                    _whisper_models_last_used.pop(name, None)
                    evicted_any = True
                    try:
                        from app import metrics as prom_metrics

                        prom_metrics.MODEL_EVICTIONS_TOTAL.labels(model=name).inc()
                    except Exception:
                        pass
        if evicted_any:
            clear_gpu_memory()


def load_align_model(language_code: str):
    """Load alignment model with per-language caching (thread-safe)."""
    if language_code not in _align_models:
        with _model_load_lock:
            if language_code not in _align_models:
                logger.info(f"Loading alignment model for language: {language_code}")
                model_a, metadata = whispermlx.load_align_model(
                    language_code=language_code,
                    device=DEVICE,
                    model_dir=CACHE_DIR,
                )
                _align_models[language_code] = (model_a, metadata)
                logger.info(f"Alignment model for {language_code} loaded")
    return _align_models[language_code]


def load_diarize_pipeline() -> DiarizationPipeline:
    """Load diarization pipeline (singleton, thread-safe)."""
    global _diarize_pipeline
    if _diarize_pipeline is None:
        with _model_load_lock:
            if _diarize_pipeline is None:
                logger.info("Loading diarization pipeline: pyannote/speaker-diarization-community-1")
                _diarize_pipeline = DiarizationPipeline(
                    model_name="pyannote/speaker-diarization-community-1",
                    token=HF_TOKEN,
                    device=DEVICE,
                )
                logger.info("Diarization pipeline loaded")
    return _diarize_pipeline


# ---------------------------------------------------------------------------
# Stage 1 -- Transcription
# ---------------------------------------------------------------------------
def transcribe(
    audio: np.ndarray,
    model_name: str = DEFAULT_MODEL,
    language: str | None = None,
    task: str = "transcribe",
    initial_prompt: str | None = None,
    hotwords: str | None = None,
) -> dict:
    """Run whispermlx transcription and return raw result dict.

    hotwords: accepted for API compatibility but IGNORED by the MLX backend.
              A warning is logged; no error is raised.
    initial_prompt: set per-request on the shared cached model (reset in finally).
    """
    whisper_model = load_whisper_model(model_name)

    # Hotwords is a no-op: the MLX backend has no hotwords mechanism.
    if hotwords is not None:
        logger.warning(
            "The MLX backend ignores hotwords; the parameter is accepted for "
            "API compatibility but has no effect on transcription."
        )

    # Set per-request initial_prompt on the shared cached model.
    # Must reset in finally to avoid leaking into subsequent requests.
    if initial_prompt is not None:
        whisper_model.initial_prompt = initial_prompt

    logger.info("Starting transcription...")
    try:
        result = whisper_model.transcribe(
            audio,
            language=language,
            task=task,
        )
    finally:
        # Always reset initial_prompt to avoid leaking to next request
        if initial_prompt is not None:
            whisper_model.initial_prompt = None

    detected_language = result.get("language", language or "en")
    logger.info(f"Transcription complete. Detected language: {detected_language}")

    clear_gpu_memory()
    return result


# ---------------------------------------------------------------------------
# Stage 2 -- Alignment
# ---------------------------------------------------------------------------
def align(audio: np.ndarray, result: dict) -> dict:
    """Run Wav2Vec2 alignment to get word-level timestamps."""
    detected_language = result.get("language", "en")
    logger.info("Aligning timestamps...")
    try:
        model_a, metadata = load_align_model(detected_language)
        result = whispermlx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            DEVICE,
            return_char_alignments=False,
        )
        logger.info("Timestamp alignment complete")
        clear_gpu_memory()
    except Exception as e:
        logger.warning(f"Timestamp alignment failed: {e}, continuing without word-level timestamps")
    return result


# ---------------------------------------------------------------------------
# Stage 3 -- Diarization
# ---------------------------------------------------------------------------
def diarize(
    audio: np.ndarray,
    result: dict,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    return_speaker_embeddings: bool = False,
) -> tuple[dict, dict | None]:
    """
    Run pyannote speaker diarization and assign speakers to segments.

    Returns (result_with_speakers, speaker_embeddings_or_None).
    """
    if not HF_TOKEN:
        logger.warning("Speaker diarization requested but HF_TOKEN not set")
        return result, None

    logger.info("Starting speaker diarization...")
    speaker_embeddings = None
    try:
        diarize_model = load_diarize_pipeline()

        diarize_params: dict[str, Any] = {}
        if num_speakers is not None:
            diarize_params["num_speakers"] = num_speakers
            logger.info(f"Diarization with exact speaker count: {num_speakers}")
        else:
            if min_speakers is not None:
                diarize_params["min_speakers"] = min_speakers
            if max_speakers is not None:
                diarize_params["max_speakers"] = max_speakers
            logger.info(f"Diarization with speaker range: {min_speakers}-{max_speakers}")

        if return_speaker_embeddings:
            diarize_params["return_embeddings"] = True
            logger.info("Speaker embeddings will be returned")

        diarize_output = diarize_model(audio, **diarize_params)

        if return_speaker_embeddings and isinstance(diarize_output, tuple):
            diarize_segments, speaker_embeddings = diarize_output
            logger.info(f"Received speaker embeddings for {len(speaker_embeddings)} speakers")
        else:
            diarize_segments = diarize_output

        if hasattr(diarize_segments, "exclusive_speaker_diarization"):
            diarize_segments = diarize_segments.exclusive_speaker_diarization
            logger.info("Using exclusive speaker diarization for better timestamp reconciliation")

        result = whispermlx.assign_word_speakers(diarize_segments, result)
        logger.info("Speaker diarization complete")
        clear_gpu_memory()
    except Exception as e:
        logger.warning(f"Speaker diarization failed: {e}, continuing without diarization")

    return result, speaker_embeddings


# ---------------------------------------------------------------------------
# Output formatting helpers
# ---------------------------------------------------------------------------
def sanitize_float_values(obj):
    """Recursively sanitize float values for JSON compliance (NaN/Inf -> None)."""
    if isinstance(obj, dict):
        return {key: sanitize_float_values(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_float_values(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return sanitize_float_values(obj.tolist())
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.floating, np.integer)):
        value = float(obj)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return obj


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


# ---------------------------------------------------------------------------
# Convenience: full pipeline in one call
# ---------------------------------------------------------------------------
def run_pipeline(
    audio: np.ndarray,
    model_name: str = DEFAULT_MODEL,
    language: str | None = None,
    task: str = "transcribe",
    initial_prompt: str | None = None,
    hotwords: str | None = None,
    word_timestamps: bool = True,
    should_diarize: bool = True,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    return_speaker_embeddings: bool = False,
) -> tuple[dict, dict | None]:
    """
    Run the full 3-stage pipeline: transcribe -> align -> diarize.

    Returns (result, speaker_embeddings_or_None).
    """
    result = transcribe(
        audio,
        model_name=model_name,
        language=language,
        task=task,
        initial_prompt=initial_prompt,
        hotwords=hotwords,
    )

    if word_timestamps:
        result = align(audio, result)

    speaker_embeddings = None
    if should_diarize:
        result, speaker_embeddings = diarize(
            audio,
            result,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            return_speaker_embeddings=return_speaker_embeddings,
        )

    return result, speaker_embeddings
