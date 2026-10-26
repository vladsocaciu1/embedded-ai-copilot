import importlib
import logging
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import app.config as config
from app.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EXCLUDE_DIRS,
    DEFAULT_MAX_PROMPT_TOKENS,
    DEFAULT_PROJECT_PATH,
    load_embedding_settings,
    load_settings,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

_DOTENV_TESTS = {
    "test_load_settings_honors_dotenv_file",
    "test_load_settings_honors_project_path_in_dotenv_file",
    "test_load_settings_honors_exclude_dirs_in_dotenv_file",
    "test_load_settings_honors_max_prompt_tokens_in_dotenv_file",
    "test_load_embedding_settings_honors_dotenv_file",
}


@pytest.fixture(autouse=True)
def stub_dotenv_and_clear_env(monkeypatch, request):
    if request.node.name in _DOTENV_TESTS:
        return

    monkeypatch.setattr("app.config.load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("PROJECT_PATH", raising=False)
    monkeypatch.delenv("EXCLUDE_DIRS", raising=False)
    monkeypatch.delenv("MAX_PROMPT_TOKENS", raising=False)
    monkeypatch.delenv("SEMANTIC_SEARCH", raising=False)
    monkeypatch.delenv("LEARNING_MODE", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_CACHE_DIR", raising=False)

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

def test_load_settings_unset_exclude_dirs_uses_defaults(tmp_path, monkeypatch):

    # Arrange
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.delenv("EXCLUDE_DIRS", raising=False)

    # Act
    settings = load_settings(tmp_path)

    # Assert
    assert settings.exclude_dirs == DEFAULT_EXCLUDE_DIRS

def test_load_settings_whitespace_exclude_dirs_uses_defaults(tmp_path, monkeypatch):

    # Arrange
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setenv("EXCLUDE_DIRS", "   ")

    # Act
    settings = load_settings(tmp_path)

    # Assert
    assert settings.exclude_dirs == DEFAULT_EXCLUDE_DIRS

def test_load_settings_parses_exclude_dirs(tmp_path, monkeypatch):

    # Arrange
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setenv("EXCLUDE_DIRS", " vendor , Generated ,, ")

    # Act
    settings = load_settings(tmp_path)

    # Assert
    assert settings.exclude_dirs == ("vendor", "Generated")

def test_load_settings_comma_exclude_dirs_includes_all(tmp_path, monkeypatch):

    # Arrange
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setenv("EXCLUDE_DIRS", ",")

    # Act
    settings = load_settings(tmp_path)

    # Assert
    assert settings.exclude_dirs == ()

def test_load_settings_exclude_dirs_path_separator_raises_value_error(tmp_path, monkeypatch):

    # Arrange
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setenv("EXCLUDE_DIRS", "src/vendor")

    # Act / Assert
    with pytest.raises(ValueError):
        load_settings(tmp_path)

    monkeypatch.setenv("EXCLUDE_DIRS", "src\\vendor")
    with pytest.raises(ValueError):
        load_settings(tmp_path)

def test_load_settings_honors_exclude_dirs_in_dotenv_file(tmp_path, monkeypatch):

    # Arrange
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (tmp_path / ".env").write_text(
        "GEMINI_API_KEY=from-dotenv-key\n"
        "GEMINI_MODEL=from-dotenv-model\n"
        "EXCLUDE_DIRS=vendor, Generated\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    # setenv then delenv so monkeypatch records the names. delenv on an
    # absent var records nothing, and load_dotenv writes them into os.environ.
    monkeypatch.setenv("GEMINI_API_KEY", "placeholder")
    monkeypatch.setenv("GEMINI_MODEL", "placeholder")
    monkeypatch.setenv("EXCLUDE_DIRS", "placeholder")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("EXCLUDE_DIRS", raising=False)

    # Act
    settings = load_settings(project_dir)

    # Assert
    assert settings.exclude_dirs == ("vendor", "Generated")
    assert settings.gemini_api_key == "from-dotenv-key"
    assert settings.gemini_model == "from-dotenv-model"

def test_load_settings_unset_max_prompt_tokens_uses_default(tmp_path, monkeypatch):

    # Arrange
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.delenv("MAX_PROMPT_TOKENS", raising=False)

    # Act
    settings = load_settings(tmp_path)

    # Assert
    assert settings.max_prompt_tokens == 200_000

def test_load_settings_whitespace_max_prompt_tokens_uses_default(tmp_path, monkeypatch):

    # Arrange
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setenv("MAX_PROMPT_TOKENS", "   ")

    # Act
    settings = load_settings(tmp_path)

    # Assert
    assert settings.max_prompt_tokens == DEFAULT_MAX_PROMPT_TOKENS

def test_load_settings_strips_max_prompt_tokens(tmp_path, monkeypatch):

    # Arrange
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setenv("MAX_PROMPT_TOKENS", " 150000 ")

    # Act
    settings = load_settings(tmp_path)

    # Assert
    assert settings.max_prompt_tokens == 150_000

@pytest.mark.parametrize("raw", ["0", "-5", "abc", "1.5", "200_000", "+5"])
def test_load_settings_invalid_max_prompt_tokens_raises_value_error(tmp_path, monkeypatch, raw):

    # Arrange
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setenv("MAX_PROMPT_TOKENS", raw)

    # Act / Assert
    with pytest.raises(ValueError, match="MAX_PROMPT_TOKENS must be a positive integer."):
        load_settings(tmp_path)

def test_load_settings_honors_max_prompt_tokens_in_dotenv_file(tmp_path, monkeypatch):

    # Arrange
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (tmp_path / ".env").write_text(
        "GEMINI_API_KEY=from-dotenv-key\n"
        "GEMINI_MODEL=from-dotenv-model\n"
        "MAX_PROMPT_TOKENS=150000\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    # setenv then delenv so monkeypatch records the names. delenv on an
    # absent var records nothing, and load_dotenv writes them into os.environ.
    monkeypatch.setenv("GEMINI_API_KEY", "placeholder")
    monkeypatch.setenv("GEMINI_MODEL", "placeholder")
    monkeypatch.setenv("MAX_PROMPT_TOKENS", "placeholder")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("MAX_PROMPT_TOKENS", raising=False)

    # Act
    settings = load_settings(project_dir)

    # Assert
    assert settings.max_prompt_tokens == 150_000
    assert settings.gemini_api_key == "from-dotenv-key"
    assert settings.gemini_model == "from-dotenv-model"

def test_g1_embedding_settings_defaults(tmp_path, monkeypatch):

    # Arrange
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")

    # Act
    # Use the module attribute. test_importing_config_has_no_side_effects reloads
    # app.config, so names imported at the top of this file are the previous class.
    embedding = config.load_embedding_settings()
    settings = config.load_settings(tmp_path)
    by_keyword = config.Settings(
        project_path=tmp_path,
        gemini_api_key="test-key",
        gemini_model="test-model",
    )

    # Assert
    assert embedding == config.EmbeddingSettings()
    assert embedding.semantic_search == "local"
    assert embedding.model == config.DEFAULT_EMBEDDING_MODEL
    assert embedding.cache_dir == config.DEFAULT_EMBEDDING_CACHE_DIR
    assert settings.embedding == config.EmbeddingSettings()
    assert by_keyword.embedding == config.EmbeddingSettings()

def test_g2_semantic_search_parses(tmp_path, monkeypatch, caplog):

    # Arrange
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    caplog.set_level(logging.INFO)

    # Act / Assert
    monkeypatch.setenv("SEMANTIC_SEARCH", "local")
    caplog.clear()
    settings = load_settings(tmp_path)
    assert settings.embedding.semantic_search == "local"
    logged = [
        record
        for record in caplog.records
        if record.message
        == f"Semantic search local (model {DEFAULT_EMBEDDING_MODEL}); used for partial context."
    ]
    assert len(logged) == 1
    assert logged[0].levelno == logging.INFO

    monkeypatch.setenv("SEMANTIC_SEARCH", " LOCAL ")
    assert load_embedding_settings().semantic_search == "local"

    monkeypatch.setenv("SEMANTIC_SEARCH", "Off")
    caplog.clear()
    settings = load_settings(tmp_path)
    assert settings.embedding.semantic_search == "off"
    assert not any("Semantic search" in record.message for record in caplog.records)

    monkeypatch.setenv("SEMANTIC_SEARCH", "   ")
    assert load_embedding_settings().semantic_search == "local"

    monkeypatch.delenv("SEMANTIC_SEARCH", raising=False)
    assert load_embedding_settings().semantic_search == "local"

@pytest.mark.parametrize("raw", ["gemini", "on", "1"])
def test_g2_invalid_semantic_search_raises_value_error(monkeypatch, caplog, raw):

    # Arrange
    monkeypatch.setenv("SEMANTIC_SEARCH", raw)
    caplog.set_level(logging.ERROR)

    # Act / Assert
    with pytest.raises(ValueError, match="SEMANTIC_SEARCH must be one of: off, local."):
        load_embedding_settings()
    assert any(record.levelno == logging.ERROR for record in caplog.records)

def test_g3_embedding_model_blank_and_stripped(monkeypatch):

    # Arrange / Act / Assert
    monkeypatch.setenv("EMBEDDING_MODEL", "   ")
    assert load_embedding_settings().model == DEFAULT_EMBEDDING_MODEL

    monkeypatch.setenv("EMBEDDING_MODEL", "  jinaai/jina-embeddings-v2-base-code ")
    assert load_embedding_settings().model == "jinaai/jina-embeddings-v2-base-code"

def test_g4_embedding_cache_dir(tmp_path, monkeypatch, caplog):

    # Arrange
    absolute = tmp_path / "absolute-cache"
    monkeypatch.setenv("EMBEDDING_CACHE_DIR", str(absolute))

    # Act
    settings = load_embedding_settings()

    # Assert
    assert settings.cache_dir == absolute.resolve()
    assert not settings.cache_dir.exists()

    root = tmp_path / "root"
    cwd = tmp_path / "cwd"
    root.mkdir()
    cwd.mkdir()
    monkeypatch.setattr(config, "BASE_DIR", root)
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("EMBEDDING_CACHE_DIR", "rel-cache")
    resolved = load_embedding_settings().cache_dir
    assert resolved == (root / "rel-cache").resolve()
    assert not resolved.exists()

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EMBEDDING_CACHE_DIR", "~/home-cache")
    assert load_embedding_settings().cache_dir == (tmp_path / "home-cache").resolve()
    assert not (tmp_path / "home-cache").exists()

    cache_file = tmp_path / "not_a_dir"
    cache_file.write_text("x", encoding="utf-8")
    monkeypatch.setenv("EMBEDDING_CACHE_DIR", str(cache_file))
    caplog.set_level(logging.ERROR)
    caplog.clear()
    with pytest.raises(NotADirectoryError):
        load_embedding_settings()
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    missing = tmp_path / "missing-cache"
    monkeypatch.setenv("EMBEDDING_CACHE_DIR", str(missing))
    assert load_embedding_settings().cache_dir == missing.resolve()
    assert not missing.exists()

def test_g6_import_does_not_import_fastembed():

    # Arrange
    code = (
        "import sys\n"
        "import app.config\n"
        "import app.embedder\n"
        "assert 'fastembed' not in sys.modules\n"
    )

    # Act
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    # Assert
    assert result.returncode == 0, result.stderr
    assert "fastembed" not in result.stdout

def test_load_embedding_settings_honors_dotenv_file(tmp_path, monkeypatch):

    # Arrange
    cache = tmp_path / "embed-cache"
    (tmp_path / ".env").write_text(
        "SEMANTIC_SEARCH=local\n"
        "EMBEDDING_MODEL=jinaai/jina-embeddings-v2-base-code\n"
        f"EMBEDDING_CACHE_DIR={cache}\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    # setenv then delenv so monkeypatch records the names. delenv on an
    # absent var records nothing, and load_dotenv writes them into os.environ.
    for name in (
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "SEMANTIC_SEARCH",
        "EMBEDDING_MODEL",
        "EMBEDDING_CACHE_DIR",
    ):
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name, raising=False)

    # Act
    settings = load_embedding_settings()

    # Assert
    assert os.getenv("GEMINI_API_KEY") is None
    assert os.getenv("GEMINI_MODEL") is None
    assert settings.semantic_search == "local"
    assert settings.model == "jinaai/jina-embeddings-v2-base-code"
    assert settings.cache_dir == cache.resolve()
    assert not cache.exists()
