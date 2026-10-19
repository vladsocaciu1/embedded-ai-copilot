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
_UNIQUE = (
    '#include "main.h"\n'
    "static int zz_graph_probe_unique(void) { return 0; }\n"
)
_MAIN_H = "void zz_graph_probe_main_h(void);\n"
_UPDATED_MS = re.compile(r"Graph updated: .* in (\d+) ms\.")
_LOGGERS = (
    logging.getLogger("app.code_graph"),
    logging.getLogger("app.index_cache"),
)


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


def _logged_get(cache: IndexCache, root: Path, files: list[ProjectFile]):
    handler = _Capture()
    for logger in _LOGGERS:
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
    started = time.perf_counter()
    index, graph = cache.get(root, files)
    elapsed = time.perf_counter() - started
    for logger in _LOGGERS:
        logger.removeHandler(handler)
    update_ms = None
    for message in handler.messages:
        match = _UPDATED_MS.search(message)
        if match is not None:
            update_ms = int(match.group(1))
    saw_updated = any("graph updated" in message for message in handler.messages)
    return index, graph, elapsed, update_ms, saw_updated


def _renamed(files: list[ProjectFile], old: Path, new: Path) -> list[ProjectFile]:
    return [
        ProjectFile(new if item.path == old else item.path, item.content)
        for item in files
    ]


def _include_names(index) -> set[str]:
    names: set[str] = set()
    for file in index.files:
        for include in file.includes:
            part = include.target.replace("\\", "/").rsplit("/", 1)[-1].lower()
            if part and part not in {".", ".."}:
                names.add(part)
    return names


def _unique_c(files: list[ProjectFile], index) -> ProjectFile:
    counts: dict[str, int] = {}
    for item in files:
        counts[item.path.name] = counts.get(item.path.name, 0) + 1
    named = _include_names(index)
    for item in files:
        if item.path.suffix.lower() != ".c":
            continue
        if counts[item.path.name] != 1:
            continue
        if item.path.name.lower() in named:
            continue
        return item
    raise SystemExit("no unique .c file to rename")


def _first_main_h(files: list[ProjectFile]) -> ProjectFile:
    for item in files:
        if item.path.name == "main.h":
            return item
    raise SystemExit("no main.h to rename")


def _path_step(
    cache: IndexCache, root: Path, files: list[ProjectFile], label: str
) -> tuple[object, bool]:
    index, updated, elapsed, update_ms, saw_updated = _logged_get(cache, root, files)
    equal = updated.snapshot() == build_graph(index).snapshot()
    shown = update_ms if update_ms is not None else "?"
    print(
        f"path {label} get {elapsed:.3f} s, update_graph {shown} ms, "
        f"snapshot equal: {equal}"
    )
    return index, equal and saw_updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_path")
    parser.add_argument("--file", dest="rel_path")
    parser.add_argument("--no-paths", action="store_true")
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
    index, updated, edit_s, update_ms, saw_updated = _logged_get(cache, root, edited)
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
    ok = saw_updated and equal
    if not args.no_paths:
        current = edited + [
            ProjectFile(chosen.path.parent / "zz_graph_probe_unique.c", _UNIQUE)
        ]
        _index, step_ok = _path_step(cache, root, current, "add-unique")
        ok = ok and step_ok
        current = [
            item
            for item in current
            if item.path.name != "zz_graph_probe_unique.c"
        ]
        index, step_ok = _path_step(cache, root, current, "remove-unique")
        ok = ok and step_ok

        unique = _unique_c(current, index)
        renamed = unique.path.with_name("zz_" + unique.path.name)
        current = _renamed(current, unique.path, renamed)
        _index, step_ok = _path_step(cache, root, current, "rename-unique")
        ok = ok and step_ok

        main_h = chosen.path.parent / "main.h"
        current = current + [ProjectFile(main_h, _MAIN_H)]
        _index, step_ok = _path_step(cache, root, current, "add-main-h")
        ok = ok and step_ok
        current = [item for item in current if item.path != main_h]
        _index, step_ok = _path_step(cache, root, current, "remove-main-h")
        ok = ok and step_ok

        existing = _first_main_h(current)
        destination = existing.path.parent.parent / "Include" / "main.h"
        current = _renamed(current, existing.path, destination)
        _index, step_ok = _path_step(cache, root, current, "rename-main-h")
        ok = ok and step_ok
    print(f"peak RSS {_peak_rss_mb()} MB")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
