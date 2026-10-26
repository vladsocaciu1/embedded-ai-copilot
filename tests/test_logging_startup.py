import logging
import os
import subprocess
import sys
from pathlib import Path

from app.config import DEFAULT_EXCLUDE_DIRS

REPO_ROOT = Path(__file__).resolve().parent.parent


def _import_main(cwd: Path, project_path: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GEMINI_API_KEY"] = "dummy-key"
    env["GEMINI_MODEL"] = "dummy-model"
    env["EXCLUDE_DIRS"] = ",".join(DEFAULT_EXCLUDE_DIRS)
    env["PROJECT_PATH"] = str(project_path)
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-c", "import app.main"],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_importing_main_logs_existing_project_path(tmp_path):

    # Arrange
    # tmp_path is an existing directory

    # Act
    result = _import_main(tmp_path, tmp_path)

    # Assert
    assert result.returncode == 0
    assert any(
        "app.config" in line and "Project path" in line
        for line in result.stderr.splitlines()
    )


def test_importing_main_logs_missing_project_path(tmp_path):

    # Arrange
    missing = tmp_path / "missing"

    # Act
    result = _import_main(tmp_path, missing)

    # Assert
    assert result.returncode != 0
    assert any(
        "app.config" in line and "ERROR" in line and "does not exist" in line
        for line in result.stderr.splitlines()
    )


def _run_script(cwd: Path, project_path: Path, script: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GEMINI_API_KEY"] = "dummy-key"
    env["GEMINI_MODEL"] = "dummy-model"
    env["EXCLUDE_DIRS"] = ",".join(DEFAULT_EXCLUDE_DIRS)
    env["PROJECT_PATH"] = str(project_path)
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_setup_logging_keeps_console_info_and_app_file_debug(tmp_path):

    # Arrange
    script = """
import logging
import logging.handlers
import app.main

root = logging.getLogger()
consoles = [
    handler.level
    for handler in root.handlers
    if type(handler) is logging.StreamHandler
]
files = [
    (handler.level, handler.baseFilename)
    for handler in root.handlers
    if isinstance(handler, logging.handlers.RotatingFileHandler)
]
print("CONSOLE", ",".join(str(level) for level in consoles))
print("FILE", "|".join(f"{level}\\t{name}" for level, name in files))
print("APP", logging.getLogger("app").level)
print("ROOT", root.level)
print("HTTPX", logging.getLogger("httpx").getEffectiveLevel())
"""

    # Act
    result = _run_script(tmp_path, tmp_path, script)

    # Assert
    assert result.returncode == 0, result.stderr
    lines = dict(line.split(" ", 1) for line in result.stdout.splitlines() if " " in line)
    assert lines["CONSOLE"] == str(logging.INFO)
    level_text, filename = lines["FILE"].split("\t", 1)
    assert int(level_text) == logging.DEBUG
    assert filename.endswith(f"{os.sep}logs{os.sep}app.log")
    assert lines["APP"] == str(logging.DEBUG)
    assert lines["ROOT"] == str(logging.INFO)
    assert lines["HTTPX"] == str(logging.INFO)


def test_cp1252_summary_on_stderr_and_per_file_line_in_log(tmp_path):

    # Arrange
    project = tmp_path / "proj"
    project.mkdir()
    (project / "l9_marker_cp1252.c").write_bytes(b"// it\x92s\nint x;\n")
    log_path = REPO_ROOT / "logs" / "app.log"
    before = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    per_file = "Decoded l9_marker_cp1252.c as cp1252 (not valid UTF-8)."
    summary = (
        "Decoded 1 file as cp1252 (not valid UTF-8); e.g. l9_marker_cp1252.c. "
        "Per-file list: DEBUG, logs/app.log."
    )
    script = """
import os
from pathlib import Path

import app.main
from app.project_loader import load_project_files

load_project_files(Path(os.environ["PROJECT_PATH"]))
"""

    # Act
    result = _run_script(tmp_path, project, script)

    # Assert
    assert result.returncode == 0, result.stderr
    assert summary in result.stderr
    assert per_file not in result.stderr
    after = log_path.read_text(encoding="utf-8")
    assert after.startswith(before)
    assert per_file in after[len(before):]
