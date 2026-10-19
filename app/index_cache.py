import hashlib
import logging
import threading
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

from app.code_graph import CodeGraph, build_graph
from app.code_index import build_index
from app.models import FileSymbols, ProjectFile, ProjectIndex

logger = logging.getLogger(__name__)


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
                hit = True
                index = self._index
                graph = self._graph
            else:
                hit = False
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
                dropped = sum(path not in new_files for path in self._files)
                self._files = new_files
                index = ProjectIndex(files=tuple(ordered))
                graph = build_graph(index)
                self._index = index
                self._graph = graph
                self._fingerprint = fingerprint

            elapsed_ms = round((perf_counter() - started) * 1000)
            logger.info(
                f"Index cache: {n} files, {reused} reused, {parsed} parsed, "
                f"{dropped} dropped, graph {'reused' if hit else 'rebuilt'} in {elapsed_ms} ms."
            )
            assert index is not None and graph is not None
            return index, graph
