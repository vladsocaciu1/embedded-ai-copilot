import hashlib
import logging
import threading
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

from app.code_graph import CodeGraph, GraphUpdateTooLarge, build_graph, update_graph
from app.code_index import build_index
from app.models import FileSymbols, ProjectFile, ProjectIndex

logger = logging.getLogger(__name__)

INCREMENTAL_MIN_FILES = 200
INCREMENTAL_SHARE_PCT = 2
INCREMENTAL_MAX_INCLUDES = 5000


class IndexCache:
    def __init__(self) -> None:
        self._project: Path | None = None
        self._files: dict[Path, tuple[bytes, FileSymbols]] = {}
        self._fingerprint: tuple[tuple[Path, bytes], ...] | None = None
        self._index: ProjectIndex | None = None
        self._graph: CodeGraph | None = None
        self._lock = threading.Lock()

    def get(
        self, project_path: Path, files: Sequence[ProjectFile]
    ) -> tuple[ProjectIndex, CodeGraph]:
        with self._lock:
            started = perf_counter()
            fingerprint = tuple(
                (
                    project_file.path,
                    hashlib.sha256(project_file.content.encode("utf-8")).digest(),
                )
                for project_file in files
            )
            n = len(files)
            if project_path != self._project:
                self._project = project_path
                self._files = {}
                self._fingerprint = None
                self._index = None
                self._graph = None

            if fingerprint == self._fingerprint:
                reused = n
                parsed = 0
                dropped = 0
                index = self._index
                graph = self._graph
                graph_state = "reused"
            else:
                changed = [
                    project_file
                    for project_file, (path, digest) in zip(files, fingerprint, strict=True)
                    if path not in self._files or self._files[path][0] != digest
                ]
                fresh = {
                    file_symbols.path: file_symbols
                    for file_symbols in (build_index(changed).files if changed else ())
                }
                new_files: dict[Path, tuple[bytes, FileSymbols]] = {}
                ordered: list[FileSymbols] = []
                reused = 0
                parsed = 0
                for project_file, (path, digest) in zip(files, fingerprint, strict=True):
                    cached = self._files.get(path)
                    if cached is not None and cached[0] == digest:
                        symbols = cached[1]
                        reused += 1
                    else:
                        symbols = fresh[path]
                        parsed += 1
                    new_files[path] = (digest, symbols)
                    ordered.append(symbols)
                dropped_paths = [
                    path for path in self._files if path not in new_files
                ]
                dropped = len(dropped_paths)
                self._files = new_files
                index = ProjectIndex(files=tuple(ordered))
                limit = max(
                    INCREMENTAL_MIN_FILES,
                    n * INCREMENTAL_SHARE_PCT // 100,
                )
                if (
                    self._graph is not None
                    and self._index is not None
                    and 0 < parsed + dropped <= limit
                ):
                    changed_paths = [project_file.path for project_file in changed]
                    changed_paths.extend(dropped_paths)
                    try:
                        graph = update_graph(
                            self._graph,
                            self._index,
                            index,
                            changed_paths,
                            max_includes=INCREMENTAL_MAX_INCLUDES,
                        )
                        graph_state = "updated"
                    except GraphUpdateTooLarge as error:
                        logger.info(
                            f"Graph update skipped: {error.includes} includes "
                            f"to re-resolve (limit {INCREMENTAL_MAX_INCLUDES}); "
                            "full build."
                        )
                        graph = build_graph(index)
                        graph_state = "rebuilt"
                else:
                    graph = build_graph(index)
                    graph_state = "rebuilt"
                self._index = index
                self._graph = graph
                self._fingerprint = fingerprint

            elapsed_ms = round((perf_counter() - started) * 1000)
            logger.info(
                f"Index cache: {n} files, {reused} reused, {parsed} parsed, "
                f"{dropped} dropped, graph {graph_state} in {elapsed_ms} ms."
            )
            assert index is not None and graph is not None
            return index, graph
