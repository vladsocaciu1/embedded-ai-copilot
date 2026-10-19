import importlib
import os
from unittest.mock import MagicMock

import pytest

from app.config import DEFAULT_EXCLUDE_DIRS, DEFAULT_MAX_PROMPT_TOKENS, Settings
from app.index_cache import IndexCache


@pytest.fixture
def ai_service(monkeypatch, tmp_path):
    """Import ai_service with Gemini configured, then replace the live client.

    app.ai_service calls load_settings() and genai.Client() at import time.
    The env vars have to be set before that import. Client() does not make a
    network call.     The module stays in sys.modules; each test rebinds client,
    settings, and index_cache so later tests still see a MagicMock,
    this tmp_path, and a cold cache.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-DO-NOT-LOG")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setenv("PROJECT_PATH", str(tmp_path))
    monkeypatch.setenv("EXCLUDE_DIRS", ",".join(DEFAULT_EXCLUDE_DIRS))
    monkeypatch.setenv("MAX_PROMPT_TOKENS", str(DEFAULT_MAX_PROMPT_TOKENS))
    (tmp_path / "main.c").write_text("int answer = 42;\n", encoding="utf-8")

    # load_settings() calls load_dotenv on first import. Names already set
    # above are not overridden and monkeypatch removes them after the test.
    # Any other name dotenv inserts is not tracked, so drop it here.
    env_before_import = set(os.environ)
    module = importlib.import_module("app.ai_service")
    for name in set(os.environ) - env_before_import:
        os.environ.pop(name, None)

    client = MagicMock()
    monkeypatch.setattr(module, "client", client)
    client.models.count_tokens.return_value.total_tokens = 100
    monkeypatch.setattr(
        module,
        "settings",
        Settings(
            project_path=tmp_path,
            gemini_api_key="test-key-DO-NOT-LOG",
            gemini_model="test-model",
        ),
    )
    monkeypatch.setattr(module, "index_cache", IndexCache())

    from app import gemini_retry

    monkeypatch.setattr(gemini_retry, "sleep", lambda _delay: None)
    monkeypatch.setattr(gemini_retry, "rand", lambda: 0.5)
    return module, client


@pytest.fixture
def sleeps(ai_service, monkeypatch):
    from app import gemini_retry

    recorded: list[float] = []

    def record(delay: float) -> None:
        recorded.append(delay)

    monkeypatch.setattr(gemini_retry, "sleep", record)
    return recorded
