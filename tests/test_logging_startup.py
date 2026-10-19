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
