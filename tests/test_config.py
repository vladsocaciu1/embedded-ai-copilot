import importlib
import os
from dataclasses import FrozenInstanceError

import pytest

import app.config as config
from app.config import DEFAULT_PROJECT_PATH, load_settings

@pytest.fixture(autouse=True)
def stub_dotenv_and_clear_env(monkeypatch, request):
    if request.node.name == "test_load_settings_honors_dotenv_file":
        return

    monkeypatch.setattr("app.config.load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

def test_load_settings_returns_settings_for_valid_dir_and_env(tmp_path, monkeypatch):

    # Arrange
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")

    # Act
    settings = load_settings(tmp_path)

    # Assert
    assert settings.project_path == tmp_path
    assert settings.gemini_api_key == "test-key"
    assert settings.gemini_model == "test-model"

def test_settings_is_frozen(tmp_path, monkeypatch):

    # Arrange
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    settings = load_settings(tmp_path)

    # Act / Assert
    with pytest.raises(FrozenInstanceError):
        settings.project_path = tmp_path / "other"

def test_load_settings_missing_path_raises_file_not_found(tmp_path, monkeypatch):

    # Arrange
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    missing_path = tmp_path / "does_not_exist"

    # Act / Assert
    with pytest.raises(FileNotFoundError):
        load_settings(missing_path)

def test_load_settings_missing_gemini_api_key_raises_runtime_error(tmp_path, monkeypatch):

    # Arrange
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_MODEL", "test-model")

    # Act / Assert
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        load_settings(tmp_path)

def test_load_settings_empty_gemini_api_key_raises_runtime_error(tmp_path, monkeypatch):

    # Arrange
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")

    # Act / Assert
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        load_settings(tmp_path)

def test_load_settings_missing_gemini_model_raises_runtime_error(tmp_path, monkeypatch):

    # Arrange
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    # Act / Assert
    with pytest.raises(RuntimeError, match="GEMINI_MODEL"):
        load_settings(tmp_path)

def test_load_settings_empty_gemini_model_raises_runtime_error(tmp_path, monkeypatch):

    # Arrange
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "")

    # Act / Assert
    with pytest.raises(RuntimeError, match="GEMINI_MODEL"):
        load_settings(tmp_path)

def test_importing_config_has_no_side_effects():

    # Arrange
    # GEMINI_* deleted and load_dotenv stubbed by the autouse fixture

    # Act
    importlib.reload(config)

    # Assert
    assert os.getenv("GEMINI_API_KEY") is None
    assert os.getenv("GEMINI_MODEL") is None

def test_load_settings_defaults_to_default_project_path(monkeypatch):

    # Arrange
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")

    # Act
    settings = load_settings()

    # Assert
    assert settings.project_path == DEFAULT_PROJECT_PATH

def test_load_settings_honors_dotenv_file(tmp_path, monkeypatch):

    # Arrange
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (tmp_path / ".env").write_text(
        "GEMINI_API_KEY=from-dotenv-key\nGEMINI_MODEL=from-dotenv-model\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    # setenv then delenv so monkeypatch records the names. delenv on an
    # absent var records nothing, and load_dotenv writes them into os.environ.
    monkeypatch.setenv("GEMINI_API_KEY", "placeholder")
    monkeypatch.setenv("GEMINI_MODEL", "placeholder")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    # Act
    settings = load_settings(project_dir)

    # Assert
    assert settings.gemini_api_key == "from-dotenv-key"
    assert settings.gemini_model == "from-dotenv-model"
