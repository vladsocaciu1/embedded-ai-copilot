import importlib
import os
from unittest.mock import MagicMock

import pytest

from app.config import DEFAULT_EXCLUDE_DIRS, Settings


@pytest.fixture
def ai_service(monkeypatch, tmp_path):
    """Import ai_service with Gemini configured, then replace the live client.

    app.ai_service calls load_settings() and genai.Client() at import time.
    The env vars have to be set before that import. Client() does not make a
    network call. The module stays in sys.modules; each test rebinds client
    and settings so later tests still see a MagicMock and this tmp_path.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-DO-NOT-LOG")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setenv("PROJECT_PATH", str(tmp_path))
    monkeypatch.setenv("EXCLUDE_DIRS", ",".join(DEFAULT_EXCLUDE_DIRS))
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
    monkeypatch.setattr(
        module,
        "settings",
        Settings(
            project_path=tmp_path,
            gemini_api_key="test-key-DO-NOT-LOG",
            gemini_model="test-model",
        ),
    )
    return module, client
