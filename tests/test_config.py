import importlib
import os
from dataclasses import FrozenInstanceError

import pytest

import app.config as config
from app.config import DEFAULT_PROJECT_PATH, load_settings

_DOTENV_TESTS = {
    "test_load_settings_honors_dotenv_file",
    "test_load_settings_honors_project_path_in_dotenv_file",
}


@pytest.fixture(autouse=True)
def stub_dotenv_and_clear_env(monkeypatch, request):
    if request.node.name in _DOTENV_TESTS:
        return

    monkeypatch.setattr("app.config.load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("PROJECT_PATH", raising=False)

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

def test_load_settings_empty_project_path_uses_default(monkeypatch):

    # Arrange
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setenv("PROJECT_PATH", "")

    # Act
    settings = load_settings()

    # Assert
    assert settings.project_path == DEFAULT_PROJECT_PATH

def test_load_settings_whitespace_project_path_uses_default(monkeypatch):

    # Arrange
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setenv("PROJECT_PATH", "   ")

    # Act
    settings = load_settings()

    # Assert
    assert settings.project_path == DEFAULT_PROJECT_PATH

def test_load_settings_absolute_project_path_uses_resolved_dir(tmp_path, monkeypatch):

    # Arrange
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setenv("PROJECT_PATH", str(project))

    # Act
    settings = load_settings()

    # Assert
    assert settings.project_path == project.resolve()

def test_load_settings_relative_project_path_resolves_against_base_dir(tmp_path, monkeypatch):

    # Arrange
    root = tmp_path / "root"
    cwd = tmp_path / "cwd"
    (root / "proj").mkdir(parents=True)
    (cwd / "proj").mkdir(parents=True)
    monkeypatch.setattr(config, "BASE_DIR", root)
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setenv("PROJECT_PATH", "proj")

    # Act
    settings = load_settings()

    # Assert
    assert settings.project_path == (root / "proj").resolve()

def test_load_settings_missing_project_path_env_raises_file_not_found(tmp_path, monkeypatch):

    # Arrange
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setenv("PROJECT_PATH", str(tmp_path / "does_not_exist"))

    # Act / Assert
    with pytest.raises(FileNotFoundError):
        load_settings()

def test_load_settings_file_project_path_env_raises_not_a_directory(tmp_path, monkeypatch):

    # Arrange
    project_file = tmp_path / "not_a_dir.c"
    project_file.write_text("int x;\n", encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setenv("PROJECT_PATH", str(project_file))

    # Act / Assert
    with pytest.raises(NotADirectoryError):
        load_settings()

def test_load_settings_explicit_path_overrides_project_path_env(tmp_path, monkeypatch):

    # Arrange
    explicit = tmp_path / "explicit"
    from_env = tmp_path / "from_env"
    explicit.mkdir()
    from_env.mkdir()
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setenv("PROJECT_PATH", str(from_env))

    # Act
    settings = load_settings(explicit)

    # Assert
    assert settings.project_path == explicit.resolve()

def test_load_settings_explicit_file_path_raises_not_a_directory(tmp_path, monkeypatch):

    # Arrange
    project_file = tmp_path / "not_a_dir.c"
    project_file.write_text("int x;\n", encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setenv("PROJECT_PATH", str(tmp_path))

    # Act / Assert
    with pytest.raises(NotADirectoryError):
        load_settings(project_file)

def test_load_settings_expands_home_in_project_path(tmp_path, monkeypatch):

    # Arrange
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setenv("PROJECT_PATH", "~/proj")

    # Act
    settings = load_settings()

    # Assert
    assert settings.project_path == project.resolve()

def test_load_settings_honors_project_path_in_dotenv_file(tmp_path, monkeypatch):

    # Arrange
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (tmp_path / ".env").write_text(
        "GEMINI_API_KEY=from-dotenv-key\n"
        "GEMINI_MODEL=from-dotenv-model\n"
        f"PROJECT_PATH={project_dir}\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    # setenv then delenv so monkeypatch records the names. delenv on an
    # absent var records nothing, and load_dotenv writes them into os.environ.
    monkeypatch.setenv("GEMINI_API_KEY", "placeholder")
    monkeypatch.setenv("GEMINI_MODEL", "placeholder")
    monkeypatch.setenv("PROJECT_PATH", "placeholder")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("PROJECT_PATH", raising=False)

    # Act
    settings = load_settings()

    # Assert
    assert settings.project_path == project_dir.resolve()
    assert settings.gemini_api_key == "from-dotenv-key"
    assert settings.gemini_model == "from-dotenv-model"
