import logging
import posixpath
from collections.abc import Callable, Collection
from pathlib import Path
from time import perf_counter

from app.models import (
    CallRef,
    FileSymbols,
    Include,
    ProjectIndex,
    ResolvedInclude,
    SymbolKind,
    SymbolRef,
)

logger = logging.getLogger(__name__)


class GraphUpdateTooLarge(Exception):
    def __init__(self, *, includes: int) -> None:
        super().__init__(f"{includes} includes to re-resolve")
        self.includes = includes


_SOURCE_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cxx"})
_HEADER_SUFFIXES = frozenset({".h", ".hh", ".hpp", ".hxx"})
_STATIC_HEADER_SUFFIXES = frozenset({".h", ".hh", ".hpp", ".hxx", ".inl"})
_SKIP_KINDS = frozenset({
    SymbolKind.PROTOTYPE,
    SymbolKind.NAMESPACE,
    SymbolKind.SECTION,
    SymbolKind.NODE,
})


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
        pairs: dict[Path, tuple[Path, ...]],
        paths: frozenset[Path],
        *,
        by_parts: dict[tuple[str, ...], tuple[Path, ...]],
        by_basename: dict[str, tuple[Path, ...]],
        lower_parts: dict[Path, tuple[str, ...]],
        parent_parts: dict[Path, tuple[str, ...]],
        by_stem_group: dict[tuple[str, str], tuple[Path, ...]],
        includers_by_key: dict[str, frozenset[Path]],
    ) -> None:
        self._definitions = definitions
        self._calls_by_callee = calls_by_callee
        self._calls_by_caller = calls_by_caller
        self._includes = includes
        self._included_by = included_by
        self._pairs = pairs
        self._paths = paths
        self._by_parts = by_parts
        self._by_basename = by_basename
        self._lower_parts = lower_parts
        self._parent_parts = parent_parts
        self._by_stem_group = by_stem_group
        self._includers_by_key = includers_by_key

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

    def snapshot(self) -> dict[str, object]:
        return {
            "definitions": dict(self._definitions),
            "calls_by_callee": dict(self._calls_by_callee),
            "calls_by_caller": dict(self._calls_by_caller),
            "includes": dict(self._includes),
            "included_by": dict(self._included_by),
            "pairs": dict(self._pairs),
            "includers_by_key": dict(self._includers_by_key),
            "by_parts": dict(self._by_parts),
            "by_basename": dict(self._by_basename),
            "lower_parts": dict(self._lower_parts),
            "parent_parts": dict(self._parent_parts),
            "by_stem_group": dict(self._by_stem_group),
            "paths": tuple(sorted(self._paths, key=lambda path: path.parts)),
        }


def update_graph(
    graph: CodeGraph,
    old_index: ProjectIndex,
    new_index: ProjectIndex,
    changed: Collection[Path],
    *,
    max_includes: int | None = None,
) -> CodeGraph:
    old_paths = {file.path for file in old_index.files}
    new_paths = {file.path for file in new_index.files}
    if old_paths != graph._paths:
        raise ValueError("old index paths must match the graph")
    changed_paths = frozenset(changed)
    if not changed_paths <= old_paths | new_paths:
        raise ValueError("changed paths must belong to the old or new index")
    added = frozenset(new_paths - old_paths)
    removed = frozenset(old_paths - new_paths)
    if not added | removed <= changed_paths:
        raise ValueError("added and removed paths must be in changed")

    started = perf_counter()
    (
        by_parts,
        by_basename,
        lower_parts,
        parent_parts,
        by_stem_group,
        paths,
    ) = _patched_path_tables(graph, added, removed, new_paths)
    partial, include_count = _select_includes(
        graph,
        new_index,
        changed_paths,
        added,
        removed,
        lower_parts,
        parent_parts,
    )
    if max_includes is not None and include_count > max_includes:
        raise GraphUpdateTooLarge(includes=include_count)

    definitions, definition_names = _patch_definitions(
        graph, old_index, new_index, changed_paths
    )
    calls_by_callee, callee_names = _patch_calls(
        graph._calls_by_callee,
        old_index,
        new_index,
        changed_paths,
        by_callee=True,
    )
    calls_by_caller, caller_names = _patch_calls(
        graph._calls_by_caller,
        old_index,
        new_index,
        changed_paths,
        by_callee=False,
    )
    includes, included_by = _apply_includes(
        graph,
        new_index,
        changed_paths,
        removed,
        partial,
        by_parts,
        by_basename,
        lower_parts,
        parent_parts,
    )
    pairs, pair_count = _patched_pairs(
        graph, added, removed, by_stem_group, parent_parts
    )
    updated = CodeGraph(
        definitions=definitions,
        calls_by_callee=calls_by_callee,
        calls_by_caller=calls_by_caller,
        includes=includes,
        included_by=included_by,
        pairs=pairs,
        paths=paths,
        by_parts=by_parts,
        by_basename=by_basename,
        lower_parts=lower_parts,
        parent_parts=parent_parts,
        by_stem_group=by_stem_group,
        includers_by_key=_patched_includers(
            graph, old_index, new_index, changed_paths
        ),
    )
    elapsed_ms = round((perf_counter() - started) * 1000)
    logger.info(
        f"Graph updated: {len(changed_paths)} files "
        f"({len(added)} added, {len(removed)} removed), "
        f"{len(definition_names | callee_names | caller_names)} names, "
        f"{include_count} includes re-resolved, "
        f"{pair_count} pairs recomputed in {elapsed_ms} ms."
    )
    return updated


def _assemble(index: ProjectIndex) -> CodeGraph:
    by_basename: dict[str, list[Path]] = {}
    by_parts: dict[tuple[str, ...], list[Path]] = {}
    by_stem_group: dict[tuple[str, str], list[Path]] = {}
    lower_parts: dict[Path, tuple[str, ...]] = {}
    parent_parts: dict[Path, tuple[str, ...]] = {}
    for file in index.files:
        path = file.path
        parts = tuple(part.lower() for part in path.parts)
        group = _pair_group(path)
        by_basename.setdefault(path.name.lower(), []).append(path)
        by_parts.setdefault(parts, []).append(path)
        lower_parts[path] = parts
        parent_parts[path] = parts[:-1]
        if group is not None:
            by_stem_group.setdefault((path.stem.lower(), group), []).append(path)
    basename_table = _freeze_paths(by_basename)
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
        pairs=_pair_table(index, stem_group_table, parent_parts),
        paths=frozenset(file.path for file in index.files),
        by_parts=part_table,
        by_basename=basename_table,
        lower_parts=lower_parts,
        parent_parts=parent_parts,
        by_stem_group=stem_group_table,
        includers_by_key=_includers_by_key(index),
    )


def _patch_definitions(
    graph: CodeGraph,
    old_index: ProjectIndex,
    new_index: ProjectIndex,
    changed_paths: frozenset[Path],
) -> tuple[dict[str, tuple[SymbolRef, ...]], set[str]]:
    grouped: dict[str, list[SymbolRef]] = {}
    for file in new_index.files:
        if file.path not in changed_paths:
            continue
        for symbol in file.symbols:
            if symbol.kind in _SKIP_KINDS:
                continue
            grouped.setdefault(_simple(symbol.name), []).append(
                SymbolRef(path=file.path, symbol=symbol)
            )
    names = set(grouped) | _definition_names(old_index, changed_paths)
    table = dict(graph._definitions)

    def new_refs(name: str) -> list[SymbolRef]:
        return grouped.get(name, [])

    _merge_refs(
        table,
        names,
        changed_paths,
        new_refs,
        lambda ref: (ref.path.parts, ref.symbol.start_line),
    )
    return table, names


def _patch_calls(
    old_table: dict[str, tuple[CallRef, ...]],
    old_index: ProjectIndex,
    new_index: ProjectIndex,
    changed_paths: frozenset[Path],
    by_callee: bool,
) -> tuple[dict[str, tuple[CallRef, ...]], set[str]]:
    grouped: dict[str, list[CallRef]] = {}
    for file in new_index.files:
        if file.path not in changed_paths:
            continue
        for call in file.calls:
            key = _simple(call.callee if by_callee else call.caller)
            grouped.setdefault(key, []).append(CallRef(path=file.path, call=call))
    names = set(grouped) | _call_names(old_index, changed_paths, by_callee)
    table = dict(old_table)

    def new_refs(name: str) -> list[CallRef]:
        return grouped.get(name, [])

    _merge_refs(
        table,
        names,
        changed_paths,
        new_refs,
        lambda ref: (ref.path.parts, ref.call.line),
    )
    return table, names


def _merge_refs[R: (SymbolRef, CallRef)](
    table: dict[str, tuple[R, ...]],
    names: set[str],
    changed_paths: frozenset[Path],
    new_refs: Callable[[str], list[R]],
    sort_key: Callable[[R], tuple[tuple[str, ...], int]],
) -> None:
    for name in names:
        merged = [
            ref for ref in table.get(name, ()) if ref.path not in changed_paths
        ] + new_refs(name)
        if merged:
            table[name] = tuple(sorted(merged, key=sort_key))
        else:
            table.pop(name, None)


def _definition_names(
    index: ProjectIndex, changed_paths: frozenset[Path]
) -> set[str]:
    names: set[str] = set()
    for file in index.files:
        if file.path not in changed_paths:
            continue
        for symbol in file.symbols:
            if symbol.kind in _SKIP_KINDS:
                continue
            names.add(_simple(symbol.name))
    return names


def _call_names(
    index: ProjectIndex, changed_paths: frozenset[Path], by_callee: bool
) -> set[str]:
    names: set[str] = set()
    for file in index.files:
        if file.path not in changed_paths:
            continue
        for call in file.calls:
            names.add(_simple(call.callee if by_callee else call.caller))
    return names


def _includers_by_key(index: ProjectIndex) -> dict[str, frozenset[Path]]:
    grouped: dict[str, set[Path]] = {}
    for file in index.files:
        for include in file.includes:
            grouped.setdefault(_include_key(include), set()).add(file.path)
    return {key: frozenset(paths) for key, paths in grouped.items()}


def _include_key(include: Include) -> str:
    parts = _target_parts(include.target)
    if not parts or parts[-1] in {".", ".."}:
        return ""
    return parts[-1].lower()


def _patched_path_tables(
    graph: CodeGraph,
    added: frozenset[Path],
    removed: frozenset[Path],
    new_paths: set[Path],
) -> tuple[
    dict[tuple[str, ...], tuple[Path, ...]],
    dict[str, tuple[Path, ...]],
    dict[Path, tuple[str, ...]],
    dict[Path, tuple[str, ...]],
    dict[tuple[str, str], tuple[Path, ...]],
    frozenset[Path],
]:
    if not added and not removed:
        return (
            graph._by_parts,
            graph._by_basename,
            graph._lower_parts,
            graph._parent_parts,
            graph._by_stem_group,
            graph._paths,
        )
    by_parts = dict(graph._by_parts)
    by_basename = dict(graph._by_basename)
    lower_parts = dict(graph._lower_parts)
    parent_parts = dict(graph._parent_parts)
    by_stem_group = dict(graph._by_stem_group)
    for path in removed:
        _remove_indexed_path(
            path, by_parts, by_basename, lower_parts, parent_parts, by_stem_group
        )
    for path in added:
        _add_indexed_path(
            path, by_parts, by_basename, lower_parts, parent_parts, by_stem_group
        )
    return (
        by_parts,
        by_basename,
        lower_parts,
        parent_parts,
        by_stem_group,
        frozenset(new_paths),
    )


def _add_indexed_path(
    path: Path,
    by_parts: dict[tuple[str, ...], tuple[Path, ...]],
    by_basename: dict[str, tuple[Path, ...]],
    lower_parts: dict[Path, tuple[str, ...]],
    parent_parts: dict[Path, tuple[str, ...]],
    by_stem_group: dict[tuple[str, str], tuple[Path, ...]],
) -> None:
    parts = tuple(part.lower() for part in path.parts)
    lower_parts[path] = parts
    parent_parts[path] = parts[:-1]
    _insert_grouped(by_parts, parts, path)
    _insert_grouped(by_basename, path.name.lower(), path)
    group = _pair_group(path)
    if group is not None:
        _insert_grouped(by_stem_group, (path.stem.lower(), group), path)


def _remove_indexed_path(
    path: Path,
    by_parts: dict[tuple[str, ...], tuple[Path, ...]],
    by_basename: dict[str, tuple[Path, ...]],
    lower_parts: dict[Path, tuple[str, ...]],
    parent_parts: dict[Path, tuple[str, ...]],
    by_stem_group: dict[tuple[str, str], tuple[Path, ...]],
) -> None:
    parts = lower_parts.pop(path)
    parent_parts.pop(path)
    _drop_grouped(by_parts, parts, path)
    _drop_grouped(by_basename, path.name.lower(), path)
    group = _pair_group(path)
    if group is not None:
        _drop_grouped(by_stem_group, (path.stem.lower(), group), path)


def _insert_grouped(table: dict[object, tuple[Path, ...]], key: object, path: Path) -> None:
    current = table.get(key, ())
    parts = path.parts
    for index, item in enumerate(current):
        if item.parts > parts:
            table[key] = current[:index] + (path,) + current[index:]
            return
    table[key] = current + (path,)


def _drop_grouped(table: dict[object, tuple[Path, ...]], key: object, path: Path) -> None:
    current = table.get(key)
    if current is None:
        return
    updated = tuple(item for item in current if item != path)
    if updated:
        table[key] = updated
    else:
        del table[key]


def _select_includes(
    graph: CodeGraph,
    new_index: ProjectIndex,
    changed_paths: frozenset[Path],
    added: frozenset[Path],
    removed: frozenset[Path],
    lower_parts: dict[Path, tuple[str, ...]],
    parent_parts: dict[Path, tuple[str, ...]],
) -> tuple[dict[Path, tuple[int, ...]], int]:
    new_by_path = {file.path: file for file in new_index.files}
    count = 0
    for path in changed_paths:
        file = new_by_path.get(path)
        if file is not None:
            count += len(file.includes)
    if not added and not removed:
        return {}, count

    keys = {path.name.lower() for path in added | removed}
    keys.add("")
    dependents: set[Path] = set()
    for key in keys:
        dependents.update(graph._includers_by_key.get(key, ()))
    dependents -= changed_paths
    added_by_key: dict[str, list[Path]] = {}
    for path in added:
        added_by_key.setdefault(path.name.lower(), []).append(path)
    partial: dict[Path, tuple[int, ...]] = {}
    for path in dependents:
        chosen: list[int] = []
        for index, item in enumerate(graph._includes.get(path, ())):
            key = _include_key(item.include)
            if key not in keys:
                continue
            if _include_may_change(
                item,
                path,
                added_by_key.get(key, ()),
                removed,
                graph._by_parts,
                lower_parts,
                parent_parts,
            ):
                chosen.append(index)
        if chosen:
            partial[path] = tuple(chosen)
            count += len(chosen)
    return partial, count


def _include_may_change(
    item: ResolvedInclude,
    including: Path,
    added_with_key: Collection[Path],
    removed: frozenset[Path],
    old_by_parts: dict[tuple[str, ...], tuple[Path, ...]],
    lower_parts: dict[Path, tuple[str, ...]],
    parent_parts: dict[Path, tuple[str, ...]],
) -> bool:
    include = item.include
    if _include_key(include) == "":
        return True
    candidates = item.candidates
    if any(candidate in removed for candidate in candidates):
        return True
    if not added_with_key:
        return False
    relative = (
        None
        if include.is_system
        else _relative_key(including, include.target)
    )
    if any(lower_parts[path] == relative for path in added_with_key):
        return True
    if relative in old_by_parts:
        return False
    parts = _target_parts(include.target)
    tail = tuple(part.lower() for part in parts if part not in {".", ".."})
    best = (
        _prefix_len(parent_parts[including], parent_parts[candidates[0]])
        if candidates
        else -1
    )
    return any(
        _ends_with(lower_parts[path], tail)
        and _prefix_len(parent_parts[including], parent_parts[path]) >= best
        for path in added_with_key
    )


def _apply_includes(
    graph: CodeGraph,
    new_index: ProjectIndex,
    changed_paths: frozenset[Path],
    removed: frozenset[Path],
    partial: dict[Path, tuple[int, ...]],
    by_parts: dict[tuple[str, ...], tuple[Path, ...]],
    by_basename: dict[str, tuple[Path, ...]],
    lower_parts: dict[Path, tuple[str, ...]],
    parent_parts: dict[Path, tuple[str, ...]],
) -> tuple[
    dict[Path, tuple[ResolvedInclude, ...]],
    dict[Path, tuple[Path, ...]],
]:
    if not changed_paths and not partial:
        return graph._includes, graph._included_by

    includes = dict(graph._includes)
    included_by = dict(graph._included_by)
    re_resolved: set[Path] = set()
    touched: set[Path] = set()
    inverted: dict[Path, set[Path]] = {}
    for path in removed:
        for item in includes.pop(path, ()):
            touched.update(item.candidates)
        touched.add(path)

    new_by_path = {file.path: file for file in new_index.files}
    for path in changed_paths:
        file = new_by_path.get(path)
        if file is None:
            continue
        resolved = _resolve_file_includes(
            file, by_parts, by_basename, lower_parts, parent_parts
        )
        for item in graph._includes.get(path, ()):
            touched.update(item.candidates)
        for item in resolved:
            touched.update(item.candidates)
            for candidate in item.candidates:
                inverted.setdefault(candidate, set()).add(path)
        includes[path] = resolved
        re_resolved.add(path)

    for path, indexes in partial.items():
        old_items = graph._includes[path]
        new_items = list(old_items)
        for index in indexes:
            include = old_items[index].include
            new_items[index] = ResolvedInclude(
                include=include,
                candidates=_resolve_include(
                    include,
                    path,
                    by_parts,
                    by_basename,
                    lower_parts,
                    parent_parts,
                ),
            )
        resolved = tuple(new_items)
        if resolved == old_items:
            continue
        for index in indexes:
            touched.update(old_items[index].candidates)
            touched.update(resolved[index].candidates)
        for item in resolved:
            for candidate in item.candidates:
                inverted.setdefault(candidate, set()).add(path)
        includes[path] = resolved
        re_resolved.add(path)

    for target in touched:
        kept = {
            path
            for path in included_by.get(target, ())
            if path not in re_resolved and path not in removed
        }
        kept.update(inverted.get(target, ()))
        if kept:
            included_by[target] = tuple(sorted(kept, key=lambda item: item.parts))
        else:
            included_by.pop(target, None)
    return includes, included_by


def _patched_pairs(
    graph: CodeGraph,
    added: frozenset[Path],
    removed: frozenset[Path],
    by_stem_group: dict[tuple[str, str], tuple[Path, ...]],
    parent_parts: dict[Path, tuple[str, ...]],
) -> tuple[dict[Path, tuple[Path, ...]], int]:
    if not added and not removed:
        return graph._pairs, 0
    pairs = dict(graph._pairs)
    recomputed = 0
    for path in removed:
        pairs.pop(path, None)
    for path in added:
        pairs[path] = _pair_paths(path, by_stem_group, parent_parts)
        recomputed += 1

    affected: set[Path] = set()
    for path in added | removed:
        group = _pair_group(path)
        if group is None:
            continue
        opposite = "header" if group == "source" else "source"
        for other in by_stem_group.get((path.stem.lower(), opposite), ()):
            if other not in added:
                affected.add(other)
    added_grouped: dict[tuple[str, str], list[Path]] = {}
    for path in added:
        group = _pair_group(path)
        if group is None:
            continue
        added_grouped.setdefault((path.stem.lower(), group), []).append(path)
    for path in affected:
        old = graph._pairs.get(path, ())
        if _pair_may_change(path, old, removed, added_grouped, parent_parts):
            pairs[path] = _pair_paths(path, by_stem_group, parent_parts)
            recomputed += 1
    return pairs, recomputed


def _pair_may_change(
    path: Path,
    old: tuple[Path, ...],
    removed: frozenset[Path],
    added_grouped: dict[tuple[str, str], list[Path]],
    parent_parts: dict[Path, tuple[str, ...]],
) -> bool:
    if any(candidate in removed for candidate in old):
        return True
    group = _pair_group(path)
    if group is None:
        return False
    wanted = "header" if group == "source" else "source"
    added_paths = added_grouped.get((path.stem.lower(), wanted), ())
    if not added_paths:
        return False
    best = _prefix_len(parent_parts[path], parent_parts[old[0]]) if old else -1
    return any(
        _prefix_len(parent_parts[path], parent_parts[added]) >= best
        for added in added_paths
    )


def _patched_includers(
    graph: CodeGraph,
    old_index: ProjectIndex,
    new_index: ProjectIndex,
    changed_paths: frozenset[Path],
) -> dict[str, frozenset[Path]]:
    if not changed_paths:
        return graph._includers_by_key
    table = dict(graph._includers_by_key)
    old_files = {file.path: file for file in old_index.files}
    new_files = {file.path: file for file in new_index.files}
    for path in changed_paths:
        old_file = old_files.get(path)
        if old_file is not None:
            for key in {_include_key(include) for include in old_file.includes}:
                _drop_includer(table, key, path)
        new_file = new_files.get(path)
        if new_file is not None:
            for key in {_include_key(include) for include in new_file.includes}:
                _add_includer(table, key, path)
    return table


def _drop_includer(
    table: dict[str, frozenset[Path]], key: str, path: Path
) -> None:
    current = table.get(key)
    if current is None or path not in current:
        return
    updated = current - {path}
    if updated:
        table[key] = updated
    else:
        del table[key]


def _add_includer(
    table: dict[str, frozenset[Path]], key: str, path: Path
) -> None:
    current = table.get(key, frozenset())
    if path not in current:
        table[key] = current | {path}


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
        resolved = _resolve_file_includes(
            file, by_parts, by_basename, lower_parts, parent_parts
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


def _resolve_file_includes(
    file: FileSymbols,
    by_parts: dict[tuple[str, ...], tuple[Path, ...]],
    by_basename: dict[str, tuple[Path, ...]],
    lower_parts: dict[Path, tuple[str, ...]],
    parent_parts: dict[Path, tuple[str, ...]],
) -> tuple[ResolvedInclude, ...]:
    return tuple(
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
