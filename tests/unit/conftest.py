"""Shared fixtures for unit tests (whispermlx mocked, fast, no model downloads)."""

from unittest.mock import MagicMock, patch
import pytest


@pytest.fixture(autouse=True)
def _reset_pipeline_caches():
    """Reset pipeline module-level caches between tests to avoid cross-contamination."""
    # We reload the module for each test by not caching imports at module level.
    # Individual tests handle their own patching.
    yield


@pytest.fixture()
def mock_whispermlx():
    """
    Patch the whispermlx import used by app.pipeline so no real model is loaded.

    Returns the mock module dict so tests can configure return values.
    """
    mock_module = MagicMock()
    mock_module.load_model.return_value = MagicMock()
    mock_module.load_align_model.return_value = (MagicMock(), {"language": "en", "dictionary": {}, "type": "huggingface"})
    mock_module.align.return_value = {
        "segments": [{"start": 0.0, "end": 1.0, "text": "hello", "words": [{"word": "hello", "start": 0.0, "end": 1.0}]}],
        "word_segments": [{"word": "hello", "start": 0.0, "end": 1.0}],
    }
    mock_module.load_audio.return_value = MagicMock()
    with patch.dict("sys.modules", {"whispermlx": mock_module}):
        yield mock_module
