#!/bin/bash
set -e

VERSION=$(uv run python -c "from app.version import __version__; print(__version__)")
echo "Whispermlx ASR Service v${VERSION}"
echo "Starting in simple mode (uvicorn)..."
exec uv run uvicorn app.main:app --host 0.0.0.0 --port 9001
