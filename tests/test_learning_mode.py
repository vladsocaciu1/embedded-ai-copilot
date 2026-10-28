import logging
from pathlib import Path

import pytest

from app.config import LEARNING_MODES, load_settings
from app.models import ProjectFile
from app.prompts import (
    assistant_prompt,
    build_assistant_prompt,
    build_partial_prompt,
    learning_instructions,
    learning_prompt,
)


@pytest.fixture
def config_env(monkeypatch):
    monkeypatch.setattr("app.config.load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    for name in ("PROJECT_PATH", "EXCLUDE_DIRS", "MAX_PROMPT_TOKENS", "LEARNING_MODE"):
        monkeypatch.delenv(name, raising=False)


def test_l1_learning_mode_parses_off_by_default(tmp_path, monkeypatch, config_env):
    assert LEARNING_MODES == ("off", "on")
    assert load_settings(tmp_path).learning_mode is False
    for raw, expected in (("", False), ("  ", False), ("off", False), ("Off", False),
                          ("on", True), (" ON ", True)):
        monkeypatch.setenv("LEARNING_MODE", raw)
        assert load_settings(tmp_path).learning_mode is expected, raw


@pytest.mark.parametrize("raw", ["yes", "1", "true", "learning"])
def test_l2_learning_mode_invalid_value_raises(tmp_path, monkeypatch, config_env, caplog, raw):
    monkeypatch.setenv("LEARNING_MODE", raw)
    with pytest.raises(ValueError, match="LEARNING_MODE must be one of: off, on."):
        load_settings(tmp_path)
    assert any(
        record.levelno == logging.ERROR
        and record.message == "LEARNING_MODE must be one of: off, on."
        for record in caplog.records
    )


def test_l3_startup_logs_learning_mode_once_only_when_on(tmp_path, monkeypatch, config_env, caplog):
    caplog.set_level(logging.INFO)
    load_settings(tmp_path)
    assert not any("Learning mode" in record.message for record in caplog.records)
    monkeypatch.setenv("LEARNING_MODE", "on")
    caplog.clear()
    load_settings(tmp_path)
    lines = [r for r in caplog.records if r.message.startswith("Learning mode on")]
    assert len(lines) == 1
    assert lines[0].levelno == logging.INFO


def test_l4_learning_prompt_defines_the_five_part_shape():
    assert learning_instructions == f"{assistant_prompt}\n\n{learning_prompt}"
    for part in ("Short answer", "Where it lives", "How it works", "Documentation", "Read next"):
        assert part in learning_prompt
    assert "No documentation page for this was included." in learning_prompt
    assert "Never invent a page, a path or a link." in learning_prompt
    assert "Never list a file or symbol that is not in the prompt." in learning_prompt
    assert "path:line or path:start-end" in learning_prompt
    assert "Do not list every included file." in learning_prompt
    assert "STM32" not in learning_prompt
    assert "files_included" not in learning_prompt


def test_l5_instructions_keyword_replaces_only_the_first_part():
    files = [ProjectFile(path=Path("a.c"), content="int a;\n")]
    default = build_assistant_prompt(files, "Q?", project_map="MAP")
    learned = build_assistant_prompt(files, "Q?", project_map="MAP", instructions=learning_instructions)
    assert default.startswith(assistant_prompt + "\n\nMAP")
    assert learned == default.replace(assistant_prompt, learning_instructions, 1)
    partial = build_partial_prompt(files, [Path("a.c")], "Q?", "MAP", 1000)
    partial_learned = build_partial_prompt(
        files, [Path("a.c")], "Q?", "MAP", 1000, learning_instructions
    )
    assert partial.startswith(assistant_prompt + "\n\nMAP")
    assert partial_learned == partial.replace(assistant_prompt, learning_instructions, 1)


def test_l12_learning_prompt_asks_for_full_paths_from_the_project_root():
    assert (
        "Always write the full path from the project root, exactly as shown in the "
        "prompt, never a bare file name (for example kernel/mutex.c:120, not mutex.c:120)."
    ) in learning_prompt
    assert "Give the full path there too." in learning_prompt
    assert learning_prompt.count("full path") == 2
