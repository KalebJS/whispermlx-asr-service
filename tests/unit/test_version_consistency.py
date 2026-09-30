"""
Guard tests keeping app.version in sync with the package version.

The service version appears in /health, the log banner, and OpenAPI
metadata via app.version.__version__, but commitizen bumps only update
pyproject.toml. These tests prevent the drift that left version.py stuck
at 0.4.0 while the package was 0.5.x.
"""

import tomllib
from pathlib import Path

from app.version import __version__

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def test_version_py_matches_pyproject():
    with _PYPROJECT.open("rb") as f:
        data = tomllib.load(f)
    assert data["project"]["version"] == __version__
