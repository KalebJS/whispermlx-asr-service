## 0.4.0 (2026-06-23)

### BREAKING CHANGES

- **Backend swap from faster-whisper/CUDA to whispermlx (MLX).** The inference engine is now [whispermlx](https://pypi.org/project/whispermlx/), an MLX fork of WhisperX that runs natively on Apple Silicon. MLX Whisper ASR uses the Metal GPU automatically. NVIDIA CUDA, `faster_whisper`, and `ctranslate2` are no longer dependencies.
- **Docker and Ray Serve removed.** The service runs natively via `uv run uvicorn app.main:app --port 9001`. No Dockerfile, docker-compose, or Ray Serve deployments. `SERVE_MODE` is hardcoded to `simple`.
- **Port changed from 9000 to 9001.** The default port is now 9001 (9000 may be occupied by system services).
- **`distil-*` models removed.** The model list is now sourced from the whispermlx MLX model map. `distil-large-v2`, `distil-medium.en`, etc. are no longer available.
- **`COMPUTE_TYPE` and `BATCH_SIZE` are inert.** These are accepted for API compatibility but have no effect on the MLX backend.

### Features

- **Native Apple-Silicon service.** Runs on macOS with Apple Silicon (M1/M2/M3/M4) using `uv` and Python 3.13. No Docker, no CUDA, no Ray.
- **MLX model map.** `/v1/models` now lists canonical MLX model names: `tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`, `large`, `large-v1`, `large-v2`, `large-v3`, `large-v3-turbo`, `turbo`, plus the `whisper-1` alias. No `faster_whisper` dependency.
- **OpenAI-style model aliases.** `whisper-1`, `whisper-tiny`, `whisper-large-v3`, etc. resolve to the corresponding MLX model on all endpoints.
- **Per-request `initial_prompt`.** The `initial_prompt` parameter is set on the shared cached model per-request and reset afterward (MLXWhisperPipeline has no `.options` attribute; uses `model.initial_prompt` directly).
- **Numpy-fed diarization.** Diarization is always fed the in-memory numpy audio array (never a file path), avoiding a torchcodec/torch 2.8 ABI mismatch on Apple Silicon. `DiarizationPipeline` uses `token=` (renamed from `use_auth_token=`).
- **Graceful diarization degradation.** When `HF_TOKEN` is missing or diarization fails internally, the service returns HTTP 200 with the transcription intact (no speaker labels, no crash).
- **Diarization segment re-splitting.** When `word_timestamps=false` and diarization is on, coarse transcript segments are re-split along diarization turn boundaries so each sub-segment carries the correct speaker label.
- **MLX memory metrics.** The `whisperx_vram_allocated_bytes` gauge reports MLX active memory via `mlx.core.get_active_memory()` (or 0). No `torch.cuda` anywhere.
- **Idle model eviction.** `MODEL_KEEP_ALIVE_SECONDS` unloads idle Whisper models; the next request reloads transparently.
- **OpenAI-compatible translations.** `POST /v1/audio/translations` translates non-English audio to English with full response format support.
- **OpenAI error envelope.** `/v1/` endpoints return `{error: {message, type, param, code}}` on validation errors. `/asr` retains the bare FastAPI 422 detail shape.
- **FastAPI lifespan API.** Migrated from deprecated `@app.on_event('startup')` to the modern lifespan context-manager.
- **Early output_format validation.** `/asr` rejects invalid `output_format` before pipeline execution to avoid wasted compute.

### Behavioral Notes

- **`hotwords` is a no-op.** The `hotwords` parameter is accepted on all endpoints for API compatibility but is silently ignored by the MLX backend. The service logs a warning (`"hotwords is ignored by the MLX backend"`) and proceeds with normal transcription. The parameter never causes an error. Use `initial_prompt` instead to bias transcription toward specific spellings.
- **Diarization requires `HF_TOKEN`.** Speaker diarization uses `pyannote/speaker-diarization-community-1` and requires an `HF_TOKEN` with the model agreement accepted. Without the token, diarization is gracefully skipped and the transcription is returned without speaker labels.
- **`DEVICE` controls torch stages only.** `DEVICE` (default `cpu`) affects VAD, wav2vec2 alignment, and pyannote diarization. MLX Whisper ASR always runs on the Metal GPU regardless of this setting.

### Infrastructure

- **`uv` + `pyproject.toml`** for dependency management. Python pinned to 3.13 (system Python 3.14 is incompatible with whispermlx `>=3.10,<3.14`).
- **`ruff`** for lint and format. **`commitizen`** for conventional commits and version bumping (`tag_format = "v$version"`).
- **`prek`** pre-commit runner with ruff hooks.
- **pytest** unit suite (whispermlx mocked, no model downloads) and integration suite (live app + small model + audio fixture).
- **Deleted:** `Dockerfile`, `docker-compose.yml`, `docker-compose.dev.yml`, `DOCKERHUB.md`, `app/serve_app.py`, `app/serve_deployments.py`. Orphaned CI workflows (`docker-publish.yml`, `dockerhub-description.yml`) removed.
- **`entrypoint.sh` simplified** to a native uvicorn launch (no Ray branch).

---

## 0.3.2 (2026-05-03)

- Pascal/Blackwell Docker image variants (#15)
- Device-aware `BATCH_SIZE` default (#12): 16 on cuda, 2 on cpu
- Idle model eviction (#16): `MODEL_KEEP_ALIVE_SECONDS` env var
- Real Prometheus `/metrics` (#13): OpenMetrics text instead of JSON
- OpenAI-style model aliases on `/asr`
- `/v1/models` sourced from `faster_whisper.available_models()`

## 0.3.0 (2026-02-28)

- Thread-safe model loading with double-checked locking
- Ray Serve mode for high-throughput ASR with cross-request batching
- Two pipeline strategies: `replicate` and `split`
- Multi-GPU support via `NUM_GPU_REPLICAS`
- Async GPU queue with semaphore for simple mode
- `/metrics` endpoint for pipeline monitoring

## 0.2.0 (2025-01-21)

- Add /v1/models and /v1/audio/transcriptions endpoints for OpenAI API compatibility
- Add diarize parameter for broader API compatibility
- Add offline mode support and fix model caching

## 0.1.1alpha (2025-11-23)

- Initial release
- WhisperX integration with API wrapper
- Speaker diarization support
- Docker deployment

## v0.5.0 (2026-06-23)

### Feat

- add console_scripts entry point for uvx support

### Fix

- pin numba>=0.60 to override mlx-whisper's stale numba==0.53.1
- update repo URLs to match renamed whispermlx-asr-service

## v0.4.0 (2026-06-23)

## v0.3.2 (2026-05-03)

## v0.3.1 (2026-03-01)

## v0.3.0 (2026-02-28)

## v0.2.0 (2026-01-21)
