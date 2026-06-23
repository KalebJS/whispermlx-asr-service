# Repository Guidelines

## Project Structure & Module Organization

```
app/                  FastAPI application source
  main.py             App factory, lifespan, /health, /metrics, /queue-metrics
  pipeline.py         Core ASR pipeline (MLX Whisper, alignment, diarization)
  openai_compat.py    OpenAI-compatible endpoints (/v1/audio/*, /v1/models)
  queue.py            Async request queue + GPU concurrency semaphore
  schemas.py          Pydantic request/response models
  metrics.py          Prometheus metric definitions
  version.py          Version string
tests/
  unit/               Fast unit tests (whispermlx mocked, no model downloads)
  integration/        Slow integration tests (live app + real model + audio)
  stress_test.py      Concurrent load testing script
entrypoint.sh         Loads .env and starts uvicorn
test-api.sh           Shell smoke-test script
pyproject.toml        Project config: dependencies, ruff, pytest, commitizen
```

## Build, Test, and Development Commands

```bash
# Set up environment (Python 3.13 required, installed via uv)
uv venv --python 3.13
uv sync

# Run the service locally
./entrypoint.sh
# or
uv run uvicorn app.main:app --host 127.0.0.1 --port 9001

# Run unit tests (fast, no model downloads)
uv run pytest tests/unit -q

# Run integration tests (slow, requires live models + HF_TOKEN)
uv run pytest tests/integration -q -m slow

# Lint and format
uv run ruff check app tests
uv run ruff format app tests
```

## Coding Style & Naming Conventions

- **Python 3.13** target. Line length 120 (enforced by ruff).
- **Ruff** handles linting and formatting. Config in `pyproject.toml` under `[tool.ruff]`.
- Enabled rule sets: `E`, `W`, `F`, `I` (isort), `UP`, `B`, `SIM`.
- Double quotes, space indentation (ruff formatter defaults).
- FastAPI route handlers use `File()` and `Form()` calls in argument defaults (B008 is suppressed for this reason).

## Testing Guidelines

- **Framework:** pytest with `httpx` for ASGI test client.
- **Test markers:** `@pytest.mark.unit` (fast, mocked) and `@pytest.mark.slow` (integration, live models).
- **Naming:** `test_*.py` files, `test_*` functions.
- Unit tests mock `whispermlx` to avoid model downloads and run quickly in CI.
- Integration tests require a running app with cached models and `HF_TOKEN` set.
- Shared fixtures live in `tests/unit/conftest.py` and `tests/integration/conftest.py`.

## Commit & Pull Request Guidelines

- **Commit convention:** Conventional Commits via `commitizen` (e.g., `feat:`, `fix:`, `test:`, `docs:`, `polish:`).
- Tag format: `v$version` (semver). Changelog auto-updates on bump.
- Keep commits focused and atomic. Reference validation IDs (e.g., `VAL-DIAR-021`) when fixing tracked issues.
- PRs should include a clear description of what changed and why, and link any related issues.

## Upstream & Project History

This project maintains API compatibility with the original [whisper-asr-webservice](https://github.com/ahmadothman/whisper-asr-webservice), whose response shape (the `text` field as a JSON array mirroring `segments`) is preserved for drop-in replacement.

**Backend lineage:** OpenAI Whisper -> [WhisperX](https://github.com/m-bain/whisperX) (word-level alignment + diarization) -> [whispermlx](https://pypi.org/project/whispermlx/) (MLX port of WhisperX for Apple Silicon).

**How we got here (the lore):**

1. **v0.1-alpha:** Started as a WhisperX-based ASR service with Docker and CUDA (`faster_whisper` / `ctranslate2` backend). Ran on NVIDIA GPUs with cuDNN.
2. **v0.2.0:** Added OpenAI-compatible endpoints (`/v1/models`, `/v1/audio/transcriptions`).
3. **v0.3.0:** Added Ray Serve multi-GPU support with `replicate` and `split` pipeline strategies. Thread-safe model loading.
4. **v0.3.2:** Docker image variants (Pascal/Blackwell), real Prometheus `/metrics`, idle model eviction, OpenAI-style model aliases.
5. **v0.4.0 (current):** Major pivot. Swapped the inference engine from `faster-whisper`/CUDA to `whispermlx` (MLX) for native Apple Silicon. Removed Docker, Ray Serve, and all CUDA dependencies entirely. Now runs as a single-process uvicorn server with an async queue. Port changed from 9000 to 9001. `COMPUTE_TYPE` and `BATCH_SIZE` became no-ops (kept for API compatibility). `distil-*` models removed; model list sourced from the whispermlx MLX model map.

When working on this codebase, keep in mind that `COMPUTE_TYPE`, `BATCH_SIZE`, and `hotwords` are accepted but inert on the MLX backend. The `DEVICE` env var only controls torch-based stages (VAD, alignment, diarization); MLX Whisper ASR always runs on the Metal GPU.

## Security & Configuration

- No authentication is built in. Do not expose the service directly to the internet without a reverse proxy or firewall.
- `HF_TOKEN` is required for speaker diarization. Store it in `.env` (gitignored), never hardcoded.
- MLX Whisper ASR always runs on the Metal GPU. The `DEVICE` env var only affects torch-based stages (VAD, alignment, diarization).
- `COMPUTE_TYPE` and `BATCH_SIZE` are accepted for API compatibility but have no effect on the MLX backend.
