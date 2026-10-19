import argparse
import logging
import re
import resource
import sys
import time
from pathlib import Path

# This file lives in scripts/, so that directory is sys.path[0]. The repo root
# has to be importable for `uv run python scripts/graph_update_check.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.code_graph import build_graph
from app.config import DEFAULT_EXCLUDE_DIRS
from app.index_cache import IndexCache
from app.models import ProjectFile
from app.project_loader import load_project_files

_PROBE = (
    '\n#include "main.h"\n'
    "static int zz_graph_update_probe(void) { return HAL_GetTick(); }\n"
)
_UPDATED_MS = re.compile(r"Graph updated: .* in (\d+) ms\.")


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _select_file(files: list[ProjectFile], rel_path: str | None) -> ProjectFile:
    if rel_path is not None:
        wanted = Path(rel_path)
        for project_file in files:
            if project_file.path == wanted:
                return project_file
        raise SystemExit(f"not in the loaded project: {rel_path}")
    under_projects = [
        project_file
        for project_file in files
        if project_file.path.name == "main.c" and "Projects" in project_file.path.parts
    ]
    if under_projects:
        return under_projects[0]
    for project_file in files:
        if project_file.path.suffix.lower() == ".c":
            return project_file
    raise SystemExit("no .c file in the project")


def _peak_rss_mb() -> int:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_path")
    parser.add_argument("--file", dest="rel_path")
    args = parser.parse_args()
    root = Path(args.project_path)
    files = load_project_files(root, DEFAULT_EXCLUDE_DIRS)
    chosen = _select_file(files, args.rel_path)

    cache = IndexCache()
    started = time.perf_counter()
    cache.get(root, files)
    cold_s = time.perf_counter() - started

    edited = [
        ProjectFile(
            project_file.path,
            project_file.content + _PROBE
            if project_file.path == chosen.path
            else project_file.content,
        )
        for project_file in files
    ]
    handler = _Capture()
    loggers = (
        logging.getLogger("app.code_graph"),
        logging.getLogger("app.index_cache"),
    )
    for logger in loggers:
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
    started = time.perf_counter()
    index, updated = cache.get(root, edited)
    edit_s = time.perf_counter() - started
    for logger in loggers:
        logger.removeHandler(handler)

    saw_updated = any("graph updated" in message for message in handler.messages)
    update_ms = None
    for message in handler.messages:
        match = _UPDATED_MS.search(message)
        if match is not None:
            update_ms = int(match.group(1))
    started = time.perf_counter()
    full = build_graph(index)
    full_s = time.perf_counter() - started
    equal = updated.snapshot() == full.snapshot()

    print(f"files {len(files)}")
    print(f"cold get {cold_s:.3f} s")
    print(f"edit get {edit_s:.3f} s")
    print(f"update_graph {update_ms if update_ms is not None else '?'} ms")
    print(f"full build {full_s:.3f} s")
    print(f"snapshot equal: {equal}")
    print(f"peak RSS {_peak_rss_mb()} MB")
    if not saw_updated or not equal:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
