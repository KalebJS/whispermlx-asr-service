"""Shared fixtures for integration tests (live app, small model, audio fixture, HF token).

These tests start a real uvicorn server with a small whispermlx model and exercise
the HTTP API end-to-end. They are marked ``slow`` and excluded from the default
scrutiny gate (``pytest tests/unit``).

Prerequisites:
  - Audio fixtures in tests/testfiles/ (sample.wav, multispeaker.wav)
  - HF_TOKEN in .env (for diarization tests)
  - uv-managed Python 3.13 venv with whispermlx installed
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

# Repo root (tests/integration/conftest.py → ../../)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Audio fixture paths
SAMPLE_WAV = REPO_ROOT / "tests" / "testfiles" / "sample.wav"
MULTISPEAKER_WAV = REPO_ROOT / "tests" / "testfiles" / "multispeaker.wav"

# Small model for fast integration tests
INTEGRATION_MODEL = os.getenv("INTEGRATION_MODEL", "tiny")


def pytest_collection_modifyitems(config, items):
    """Auto-mark every test collected under tests/integration with the ``slow`` marker.

    Only marks tests whose file path is under tests/integration so that unit
    tests collected in the same session are not accidentally marked slow.
    """
    slow_marker = pytest.mark.slow
    integration_dir = str(Path(__file__).resolve().parent)
    for item in items:
        if integration_dir in str(item.fspath):
            item.add_marker(slow_marker)


def _find_free_port() -> int:
    """Find a free port in the reserved 9001-9010 range."""
    for port in range(9001, 9011):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    pytest.fail("No free port in the 9001-9010 range for integration test server")


def _load_env() -> dict[str, str]:
    """Load .env into the environment dict (HF_TOKEN, DEVICE, etc.)."""
    env = os.environ.copy()
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env.setdefault(key.strip(), value.strip())
    return env


@pytest.fixture(scope="session")
def server_url():
    """Start a live uvicorn server with a small model and return its base URL.

    The server is started on a free port in 9001-9010 with PRELOAD_MODEL=tiny
    and DEVICE=cpu. It is terminated when the test session ends.
    """
    port = _find_free_port()
    env = _load_env()
    env["PRELOAD_MODEL"] = INTEGRATION_MODEL
    env["DEVICE"] = "cpu"

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        env=env,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    base_url = f"http://127.0.0.1:{port}"
    try:
        # Wait up to 120s for the model to load and the server to come up
        for _ in range(60):
            if proc.poll() is not None:
                # Process exited early
                out = proc.stdout.read().decode() if proc.stdout else ""
                pytest.fail(f"Server process exited early (code {proc.returncode}). Output:\n{out}")
            try:
                resp = httpx.get(f"{base_url}/health", timeout=2)
                if resp.status_code == 200:
                    break
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout):
                pass
            time.sleep(2)
        else:
            proc.terminate()
            try:
                out, _ = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, _ = proc.communicate()
            pytest.fail(f"Server did not become healthy within 120s. Output:\n{out.decode() if out else ''}")

        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


@pytest.fixture()
def sample_audio():
    """Path to the single-speaker audio fixture, skipping if absent."""
    if not SAMPLE_WAV.exists():
        pytest.skip(f"Audio fixture not found: {SAMPLE_WAV}")
    return str(SAMPLE_WAV)


@pytest.fixture()
def multispeaker_audio():
    """Path to the multi-speaker audio fixture, skipping if absent."""
    if not MULTISPEAKER_WAV.exists():
        pytest.skip(f"Multi-speaker fixture not found: {MULTISPEAKER_WAV}")
    return str(MULTISPEAKER_WAV)
