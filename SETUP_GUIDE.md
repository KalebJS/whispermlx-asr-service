# Whispermlx ASR Service - Setup Guide

This guide walks you through setting up the native Apple-Silicon ASR service powered by whispermlx (MLX).

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running the Service](#running-the-service)
5. [Testing](#testing)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware

- **Apple Silicon Mac** (M1, M2, M3, or M4)
- **16 GB+ RAM** recommended (8 GB may work with `tiny`/`base` models)
- **50 GB+ free disk space** for model caching

### Software

1. **uv** (Python package manager)

   Install uv:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   Verify installation:
   ```bash
   uv --version
   ```

2. **FFmpeg** (for audio decoding)

   ```bash
   brew install ffmpeg
   ```

3. **Hugging Face Account and Token** (for speaker diarization)

   a. **Create Account:**
      - Visit: https://huggingface.co/join
      - Sign up with your email

   b. **Accept Model Agreement (REQUIRED for diarization):**
      - Visit: https://huggingface.co/pyannote/speaker-diarization-community-1
      - Click **"Agree and access repository"**

   c. **Generate Token:**
      - Visit: https://huggingface.co/settings/tokens
      - Click "New token" -> Name it -> Select "Read" permission
      - Copy token (starts with `hf_...`)

   > Without the HF token, diarization is gracefully skipped. Transcription still works, but no speaker labels are assigned.

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/murtaza-nasir/whisperx-asr-service.git
cd whisperx-asr-service
```

### Step 2: Create Virtual Environment and Install Dependencies

```bash
# Create a Python 3.13 virtual environment
# (Python 3.14 is incompatible with whispermlx >=3.10,<3.14)
uv venv --python 3.13

# Install all dependencies
uv sync
```

This installs:
- `whispermlx` (MLX Whisper backend)
- `fastapi`, `uvicorn[standard]`, `python-multipart`, `pydantic`, `prometheus-client`
- Dev tools: `pytest`, `ruff`, `commitizen`, `httpx`

### Step 3: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit and add your Hugging Face token
nano .env
```

Minimal `.env` configuration:

```bash
# Required for speaker diarization
HF_TOKEN=hf_your_token_here

# Device for torch-based stages (VAD, alignment, diarization).
# MLX Whisper ASR always runs on the Metal GPU automatically.
DEVICE=cpu

# Model to preload on startup (optional, reduces first-request latency)
PRELOAD_MODEL=large-v3

# Service port
PORT=9001
```

---

## Configuration

### Environment Variables

Edit `.env` to customize the service:

```bash
# ---------------------------------------------------------------------------
# Hugging Face Token (REQUIRED for speaker diarization)
# ---------------------------------------------------------------------------
# Without this, diarization is skipped and no speaker labels are produced.
# Accept the model agreement at:
#   https://huggingface.co/pyannote/speaker-diarization-community-1
HF_TOKEN=hf_your_token_here

# ---------------------------------------------------------------------------
# Device Configuration (MLX device semantics)
# ---------------------------------------------------------------------------
# DEVICE controls where VAD, wav2vec2 alignment, and pyannote diarization
# (torch-based stages) run. MLX Whisper ASR always runs on the Metal GPU
# automatically regardless of this setting.
#
# Options:
#   cpu   - Safe default on Apple Silicon; torch stages on CPU, ASR on Metal GPU
#   mps   - Use Apple MPS for torch stages (experimental)
DEVICE=cpu

# ---------------------------------------------------------------------------
# Compute Type & Batch Size (accepted but INERT under MLX)
# ---------------------------------------------------------------------------
# These are read for API compatibility but have NO EFFECT on the MLX backend.
#COMPUTE_TYPE=float16
#BATCH_SIZE=16

# ---------------------------------------------------------------------------
# Model Cache Directories
# ---------------------------------------------------------------------------
CACHE_DIR=~/.cache/whisperx-asr
HF_HOME=~/.cache/whisperx-asr

# Offline mode (optional): set to 1 to prevent network requests after caching
#HF_HUB_OFFLINE=1

# ---------------------------------------------------------------------------
# Model Preloading (OPTIONAL)
# ---------------------------------------------------------------------------
# Preload a model on startup to reduce first-request latency.
# Options: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en,
#          large, large-v1, large-v2, large-v3, large-v3-turbo, turbo
PRELOAD_MODEL=large-v3

# ---------------------------------------------------------------------------
# Service Port
# ---------------------------------------------------------------------------
PORT=9001

# ---------------------------------------------------------------------------
# Upload Limits
# ---------------------------------------------------------------------------
#MAX_FILE_SIZE_MB=1000

# ---------------------------------------------------------------------------
# GPU Concurrency & Queue
# ---------------------------------------------------------------------------
#GPU_CONCURRENCY=1
#MAX_QUEUE_SIZE=32

# ---------------------------------------------------------------------------
# Idle Model Eviction (OPTIONAL)
# ---------------------------------------------------------------------------
#MODEL_KEEP_ALIVE_SECONDS=3600
#MODEL_EVICTION_INTERVAL_SECONDS=60
```

### Model Selection

Available MLX Whisper models:

| Model | Speed | Quality | RAM |
|-------|-------|---------|-----|
| `tiny`, `tiny.en` | Fastest | Lowest | ~2 GB |
| `base`, `base.en` | Very Fast | Low | ~2 GB |
| `small`, `small.en` | Fast | Medium | ~2.3 GB |
| `medium`, `medium.en` | Moderate | Good | ~5 GB |
| `large`, `large-v1` | Slow | Excellent | ~10+ GB |
| `large-v2` | Slow | Excellent | ~10+ GB |
| `large-v3` | Slow | Best | ~10+ GB |
| `large-v3-turbo`, `turbo` | Fast | High | ~5 GB |

OpenAI-style aliases (`whisper-1`, `whisper-tiny`, `whisper-large-v3`) are also accepted and resolve to the corresponding MLX model.

Models are downloaded on first use and cached in `CACHE_DIR` (default `~/.cache/whisperx-asr`).

### Idle Model Eviction

To reclaim memory between bursts of activity:

```bash
MODEL_KEEP_ALIVE_SECONDS=3600          # unload models idle for 1 hour
MODEL_EVICTION_INTERVAL_SECONDS=60     # sweep cadence (floor 30 seconds)
```

Default is `0` (disabled; models stay loaded). The next request that needs an evicted model reloads it transparently.

---

## Running the Service

### Start the Service

**Option A: Using entrypoint.sh (recommended)**

```bash
./entrypoint.sh
```

This loads your `.env` file and starts uvicorn on port 9001.

**Option B: Direct uvicorn**

```bash
# Source .env first, or set environment variables inline
uv run uvicorn app.main:app --host 127.0.0.1 --port 9001
```

**Option C: With specific environment variables**

```bash
DEVICE=cpu PRELOAD_MODEL=base uv run uvicorn app.main:app --host 127.0.0.1 --port 9001
```

Look for:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:9001
```

### Stop the Service

Press `Ctrl+C` in the terminal where the service is running, or kill the process:

```bash
lsof -ti tcp:9001 | xargs kill
```

### Run in the Background

```bash
nohup ./entrypoint.sh &> service.log &
tail -f service.log
```

---

## Testing

### Step 1: Health Check

```bash
curl http://localhost:9001/health
```

Expected response:
```json
{
  "status": "healthy",
  "device": "cpu",
  "loaded_models": ["large-v3"],
  "serve_mode": "simple"
}
```

### Step 2: Prometheus Metrics

```bash
curl http://localhost:9001/metrics | head -20
```

Expected: OpenMetrics text starting with `# HELP whisperx_...` lines.

### Step 3: Test Transcription

```bash
curl -X POST http://localhost:9001/asr \
  -F "audio_file=@test.mp3" \
  -F "language=en" \
  -F "model=small" \
  -F "output_format=json" \
  -F "diarize=false"
```

### Step 4: Test with Diarization

```bash
curl -X POST http://localhost:9001/asr \
  -F "audio_file=@meeting.mp3" \
  -F "language=en" \
  -F "model=small" \
  -F "output_format=json" \
  -F "enable_diarization=true" \
  -F "min_speakers=2" \
  -F "max_speakers=4"
```

### Step 5: Smoke Test Script

```bash
./test-api.sh localhost 9001 path/to/audio.wav
```

### Step 6: Unit and Integration Tests

```bash
# Fast unit tests (whispermlx mocked, no model downloads)
uv run pytest tests/unit -q

# Slow integration tests (live app + small model + audio fixture + HF token)
uv run pytest tests/integration -q -m slow
```

---

## Integration with Speakr

To use this service with [Speakr](https://github.com/murtaza-nasir/speakr):

Update Speakr's `.env` file:

```bash
# Enable ASR endpoint
USE_ASR_ENDPOINT=true

# Point to the whispermlx ASR service
ASR_BASE_URL=http://localhost:9001
```

If the service is on a different machine, replace `localhost` with the machine's IP address and ensure port 9001 is accessible through your firewall.

---

## Troubleshooting

### Issue: Service Won't Start

**Check logs for errors.** Common causes:
- Port 9001 already in use: Change `PORT` in `.env` or stop the conflicting process
- Python version mismatch: Ensure you are using Python 3.13 via `uv` (not system Python 3.14)
- Missing dependencies: Run `uv sync` to install all dependencies

### Issue: Speaker Diarization Fails or No Labels

**Check:**
1. `HF_TOKEN` is set correctly in `.env`
2. You accepted the model agreement at [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
3. Check logs for diarization errors

**Note:** Without `HF_TOKEN`, diarization is gracefully skipped. Transcription still succeeds (HTTP 200) but no speaker labels are assigned. This is by design.

### Issue: Slow Processing

**Solutions:**
- Use a smaller model (e.g., `small` or `base` instead of `large-v3`)
- Use `large-v3-turbo` for a good speed/quality balance
- Disable diarization if not needed: `diarize=false`

### Issue: Out of Memory

**Solutions:**
1. Use a smaller model (`small` or `base`)
2. Reduce `MAX_FILE_SIZE_MB` in `.env`
3. Disable diarization: `diarize=false`
4. Split large audio files into smaller chunks

### Issue: Model Download Fails

**Solutions:**
- Check internet access on first run (models are cached after first download)
- Verify `CACHE_DIR` and `HF_HOME` are set to writable paths
- For air-gapped environments, see the README's [Offline Use](README.md#offline-use) section

### Issue: API Returns 500 Error

**Check logs for error details.** Common causes:
- Invalid audio format (use FFmpeg to convert to MP3 or WAV)
- Model loading failed (check disk space, internet access)
- Incorrect parameters (check API docs at `/docs`)

---

## Security Notes

**This service has NO authentication or security features.**

Basic protection:
```bash
# Restrict access to specific IP
sudo ufw allow from YOUR_IP to any port 9001
sudo ufw deny 9001/tcp
```

Store `HF_TOKEN` in the `.env` file (gitignored), never hardcoded.

---

## Support

If you encounter issues:

1. Check this troubleshooting guide
2. Check the [README](README.md) for full API documentation
3. Run health check: `curl http://localhost:9001/health`
4. Create an issue with logs and configuration details
