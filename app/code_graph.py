import logging
import posixpath
from pathlib import Path
from time import perf_counter

from app.models import (
    CallRef,
    Include,
    ProjectIndex,
    ResolvedInclude,
    SymbolKind,
    SymbolRef,
)

logger = logging.getLogger(__name__)

_SOURCE_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cxx"})
_HEADER_SUFFIXES = frozenset({".h", ".hh", ".hpp", ".hxx"})
_STATIC_HEADER_SUFFIXES = frozenset({".h", ".hh", ".hpp", ".hxx", ".inl"})
_SKIP_KINDS = frozenset({SymbolKind.PROTOTYPE, SymbolKind.NAMESPACE})


def build_graph(index: ProjectIndex) -> CodeGraph:
    started = perf_counter()
    graph = _assemble(index)
    file_count = len(index.files)
    call_count = sum(len(file.calls) for file in index.files)
    resolved = ambiguous = unresolved = 0
    for file in index.files:
        for item in graph.includes(file.path):
            count = len(item.candidates)
            if count == 0:
                unresolved += 1
            elif count == 1:
                resolved += 1
            else:
                ambiguous += 1
    paired = sum(bool(graph.pair(file.path)) for file in index.files)
    elapsed_ms = round((perf_counter() - started) * 1000)
    logger.info(
        f"Graph built: {file_count} files, {call_count} calls, "
        f"includes {resolved} resolved / {ambiguous} ambiguous / {unresolved} unresolved, "
        f"{paired} files paired in {elapsed_ms} ms."
    )
    return graph


class CodeGraph:
    def __init__(
        self,
        definitions: dict[str, tuple[SymbolRef, ...]],
        calls_by_callee: dict[str, tuple[CallRef, ...]],
        calls_by_caller: dict[str, tuple[CallRef, ...]],
        includes: dict[Path, tuple[ResolvedInclude, ...]],
        included_by: dict[Path, tuple[Path, ...]],
        by_basename: dict[str, tuple[Path, ...]],
        by_stem: dict[str, tuple[Path, ...]],
        pairs: dict[Path, tuple[Path, ...]],
        paths: frozenset[Path],
    ) -> None:
        self._definitions = definitions
        self._calls_by_callee = calls_by_callee
        self._calls_by_caller = calls_by_caller
        self._includes = includes
        self._included_by = included_by
        self._by_basename = by_basename
        self._by_stem = by_stem
        self._pairs = pairs
        self._paths = paths

    def definitions(
        self, name: str, from_path: Path | None = None
    ) -> tuple[SymbolRef, ...]:
        if from_path is not None and from_path not in self._paths:
            return ()
        refs = self._definitions.get(_simple(name), ())
        if "::" in name:
            refs = tuple(
                ref for ref in refs if _name_matches(ref.symbol.name, name)
            )
        if from_path is None:
            return refs
        return tuple(ref for ref in refs if _visible_from(ref, from_path))

    def callers(self, name: str) -> tuple[CallRef, ...]:
        return self._calls_by_callee.get(_simple(name), ())

    def callees(self, function: str) -> tuple[CallRef, ...]:
        refs = self._calls_by_caller.get(_simple(function), ())
        if "::" not in function:
            return refs
        return tuple(ref for ref in refs if _name_matches(ref.call.caller, function))

    def includes(self, path: Path) -> tuple[ResolvedInclude, ...]:
        return self._includes.get(path, ())

    def included_by(self, path: Path) -> tuple[Path, ...]:
        return self._included_by.get(path, ())

    def pair(self, path: Path) -> tuple[Path, ...]:
        return self._pairs.get(path, ())


def _assemble(index: ProjectIndex) -> CodeGraph:
    by_basename: dict[str, list[Path]] = {}
    by_stem: dict[str, list[Path]] = {}
    by_parts: dict[tuple[str, ...], list[Path]] = {}
    by_stem_group: dict[tuple[str, str], list[Path]] = {}
    lower_parts: dict[Path, tuple[str, ...]] = {}
    parent_parts: dict[Path, tuple[str, ...]] = {}
    for file in index.files:
        path = file.path
        parts = tuple(part.lower() for part in path.parts)
        group = _pair_group(path)
        by_basename.setdefault(path.name.lower(), []).append(path)
        by_stem.setdefault(path.stem.lower(), []).append(path)
        by_parts.setdefault(parts, []).append(path)
        lower_parts[path] = parts
        parent_parts[path] = parts[:-1]
        if group is not None:
            by_stem_group.setdefault((path.stem.lower(), group), []).append(path)
    basename_table = _freeze_paths(by_basename)
    stem_table = _freeze_paths(by_stem)
    part_table = _freeze_paths(by_parts)
    stem_group_table = _freeze_paths(by_stem_group)
    includes, included_by = _include_tables(
        index, part_table, basename_table, lower_parts, parent_parts
    )
    return CodeGraph(
        definitions=_definition_table(index),
        calls_by_callee=_call_table(index, by_callee=True),
        calls_by_caller=_call_table(index, by_callee=False),
        includes=includes,
        included_by=included_by,
        by_basename=basename_table,
        by_stem=stem_table,
        pairs=_pair_table(index, stem_group_table, parent_parts),
        paths=frozenset(file.path for file in index.files),
    )


def _definition_table(index: ProjectIndex) -> dict[str, tuple[SymbolRef, ...]]:
    grouped: dict[str, list[SymbolRef]] = {}
    for file in index.files:
        for symbol in file.symbols:
            if symbol.kind in _SKIP_KINDS:
                continue
            grouped.setdefault(_simple(symbol.name), []).append(
                SymbolRef(path=file.path, symbol=symbol)
            )
    return {
        name: tuple(
            sorted(refs, key=lambda ref: (ref.path.parts, ref.symbol.start_line))
        )
        for name, refs in grouped.items()
    }


def _call_table(
    index: ProjectIndex, by_callee: bool
) -> dict[str, tuple[CallRef, ...]]:
    grouped: dict[str, list[CallRef]] = {}
    for file in index.files:
        for call in file.calls:
            key = _simple(call.callee if by_callee else call.caller)
            grouped.setdefault(key, []).append(CallRef(path=file.path, call=call))
    return {
        name: tuple(sorted(refs, key=lambda ref: (ref.path.parts, ref.call.line)))
        for name, refs in grouped.items()
    }


def _include_tables(
    index: ProjectIndex,
    by_parts: dict[tuple[str, ...], tuple[Path, ...]],
    by_basename: dict[str, tuple[Path, ...]],
    lower_parts: dict[Path, tuple[str, ...]],
    parent_parts: dict[Path, tuple[str, ...]],
) -> tuple[
    dict[Path, tuple[ResolvedInclude, ...]],
    dict[Path, tuple[Path, ...]],
]:
    includes: dict[Path, tuple[ResolvedInclude, ...]] = {}
    included_by: dict[Path, list[Path]] = {}
    for file in index.files:
        resolved = tuple(
            ResolvedInclude(
                include=include,
                candidates=_resolve_include(
                    include,
                    file.path,
                    by_parts,
                    by_basename,
                    lower_parts,
                    parent_parts,
                ),
            )
            for include in file.includes
        )
        includes[file.path] = resolved
        for item in resolved:
            for candidate in item.candidates:
                included_by.setdefault(candidate, []).append(file.path)
    return includes, {
        path: tuple(sorted(set(includers), key=lambda item: item.parts))
        for path, includers in included_by.items()
    }


def _pair_table(
    index: ProjectIndex,
    by_stem_group: dict[tuple[str, str], tuple[Path, ...]],
    parent_parts: dict[Path, tuple[str, ...]],
) -> dict[Path, tuple[Path, ...]]:
    return {
        file.path: _pair_paths(file.path, by_stem_group, parent_parts)
        for file in index.files
    }


def _resolve_include(
    include: Include,
    including: Path,
    by_parts: dict[tuple[str, ...], tuple[Path, ...]],
    by_basename: dict[str, tuple[Path, ...]],
    lower_parts: dict[Path, tuple[str, ...]],
    parent_parts: dict[Path, tuple[str, ...]],
) -> tuple[Path, ...]:
    parts = _target_parts(include.target)
    if not include.is_system:
        relative = _relative_key(including, include.target)
        if relative is not None and relative in by_parts:
            return by_parts[relative]
    tail = tuple(part.lower() for part in parts if part not in {".", ".."})
    if not tail:
        return ()
    matches = [
        path
        for path in by_basename.get(tail[-1], ())
        if _ends_with(lower_parts[path], tail)
    ]
    return _nearest(including, matches, parent_parts)


def _pair_paths(
    path: Path,
    by_stem_group: dict[tuple[str, str], tuple[Path, ...]],
    parent_parts: dict[Path, tuple[str, ...]],
) -> tuple[Path, ...]:
    group = _pair_group(path)
    if group is None:
        return ()
    wanted = "header" if group == "source" else "source"
    return _nearest(
        path,
        by_stem_group.get((path.stem.lower(), wanted), ()),
        parent_parts,
    )


def _nearest(
    origin: Path,
    candidates: list[Path] | tuple[Path, ...],
    parent_parts: dict[Path, tuple[str, ...]],
) -> tuple[Path, ...]:
    if len(candidates) <= 1:
        return tuple(sorted(candidates, key=lambda path: path.parts))
    origin_dir = parent_parts[origin]
    best = 0
    kept: list[Path] = []
    for candidate in candidates:
        score = _prefix_len(origin_dir, parent_parts[candidate])
        if score > best:
            best = score
            kept = [candidate]
        elif score == best:
            kept.append(candidate)
    return tuple(sorted(kept, key=lambda path: path.parts))


def _prefix_len(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    length = 0
    for first, second in zip(left, right):
        if first != second:
            break
        length += 1
    return length


def _target_parts(target: str) -> tuple[str, ...]:
    return tuple(part for part in target.replace("\\", "/").split("/") if part)


def _relative_key(including: Path, target: str) -> tuple[str, ...] | None:
    parent = including.parent.as_posix()
    if parent == ".":
        parent = ""
    joined = posixpath.normpath(
        posixpath.join(parent, target.replace("\\", "/"))
    )
    if (
        joined in {"", "."}
        or joined == ".."
        or joined.startswith("../")
        or posixpath.isabs(joined)
    ):
        return None
    return tuple(part.lower() for part in Path(joined).parts)


def _ends_with(parts: tuple[str, ...], tail: tuple[str, ...]) -> bool:
    if len(parts) < len(tail):
        return False
    offset = len(parts) - len(tail)
    return all(
        parts[offset + index] == tail[index] for index in range(len(tail))
    )


def _freeze_paths(
    grouped: dict[object, list[Path]],
) -> dict[object, tuple[Path, ...]]:
    return {
        key: tuple(sorted(paths, key=lambda path: path.parts))
        for key, paths in grouped.items()
    }


def _pair_group(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in _SOURCE_SUFFIXES:
        return "source"
    if suffix in _HEADER_SUFFIXES:
        return "header"
    return None


def _simple(name: str) -> str:
    return name.rsplit("::", 1)[-1]


def _name_matches(stored: str, query: str) -> bool:
    if "::" in query:
        return stored == query or stored.endswith("::" + query)
    return _simple(stored) == query


def _visible_from(ref: SymbolRef, from_path: Path) -> bool:
    if not ref.symbol.is_static:
        return True
    if ref.path.suffix.lower() in _STATIC_HEADER_SUFFIXES:
        return True
    return ref.path == from_path
