import logging

from pathlib import Path

from app.code_graph import CodeGraph
from app.models import FileSymbols, ProjectFile, ProjectIndex, ProjectMap, Symbol, SymbolKind

logger = logging.getLogger(__name__)

assistant_prompt_path = Path(__file__).resolve().parent / "prompts" / "assistant.md"

with assistant_prompt_path.open('r', encoding='utf-8') as f:
    assistant_prompt = f.read()

_TYPE_KINDS = frozenset({
    SymbolKind.TYPEDEF,
    SymbolKind.STRUCT,
    SymbolKind.UNION,
    SymbolKind.ENUM,
    SymbolKind.CLASS,
})
_DETAIL_TIERS = ("full", "no_macros", "no_statics")
_TIERS = (*_DETAIL_TIERS, "counts")


def build_project_map(index: ProjectIndex, graph: CodeGraph, max_chars: int) -> ProjectMap:
    files = tuple(sorted(index.files, key=lambda file: file.path.parts))
    for tier in _TIERS:
        text = _render_map(files, graph, tier)
        if len(text) <= max_chars:
            return ProjectMap(text=text, tier=tier)
    text = f"Project map omitted: too large for its budget ({len(files)} files)."
    return ProjectMap(text=text, tier="omitted")


def build_assistant_prompt(
    project_files: list[ProjectFile],
    question: str,
    project_map: str | None = None,
) -> str:
    """
    Build a prompt that includes the assistant prompt, context, and the user's question.

    Args:
        project_files (list[ProjectFile]): A list of ProjectFile objects containing the path and content of each file.
        question (str): The user's question.
        project_map: Map text placed after the instructions. None or "" leaves the prompt unchanged.

    Returns:
        str: The complete prompt to send to the AI model.
    """

    header = _format_project_header(project_files)
    project_files_text = _format_project_context(project_files)
    parts = [assistant_prompt]
    if project_map:
        parts.append(project_map)
    parts.append(header)
    if project_files_text:
        parts.append(project_files_text)
    parts.append(f"Question:\n{question}")
    return "\n\n".join(parts)


def _render_map(
    files: tuple[FileSymbols, ...], graph: CodeGraph, tier: str
) -> str:
    n_functions = sum(
        symbol.kind == SymbolKind.FUNCTION
        for file in files
        for symbol in file.symbols
    )
    lines = [
        (
            f"Project map ({tier}): {len(files)} files, {n_functions} functions. "
            "Built by a parser from the files below, which are authoritative. "
            "Line ranges are 1-based and inclusive."
        )
    ]
    if tier in _DETAIL_TIERS:
        external = _external_headers(files, graph)
        if external:
            lines.append(
                "External headers (included, but not among the loaded files): "
                + ", ".join(external)
            )
    for file in files:
        if tier == "counts":
            lines.append(_counts_line(file))
        else:
            lines.append(f"{file.path.as_posix()}{_marker(file)}")
            lines.extend(_field_lines(file, tier))
    return "\n".join(lines)


def _external_headers(files: tuple[FileSymbols, ...], graph: CodeGraph) -> list[str]:
    by_key: dict[str, str] = {}
    for file in files:
        for resolved in graph.includes(file.path):
            if resolved.candidates:
                continue
            include = resolved.include
            shown = f"<{include.target}>" if include.is_system else include.target
            by_key.setdefault(shown.casefold(), shown)
    return sorted(by_key.values(), key=lambda header: (header.casefold(), header))


def _counts_line(file: FileSymbols) -> str:
    functions = sum(symbol.kind == SymbolKind.FUNCTION for symbol in file.symbols)
    types = sum(symbol.kind in _TYPE_KINDS for symbol in file.symbols)
    macros = sum(symbol.kind == SymbolKind.MACRO for symbol in file.symbols)
    return (
        f"{file.path.as_posix()}{_marker(file)}: "
        f"{functions} functions, {types} types, {macros} macros"
    )


def _marker(file: FileSymbols) -> str:
    if file.grammar is None:
        return " (not parsed)"
    if file.has_errors:
        return " (parse errors; may be incomplete)"
    return ""


def _field_lines(file: FileSymbols, tier: str) -> list[str]:
    lines: list[str] = []
    if tier == "full":
        _append_field(lines, "includes", _dedupe(include.target for include in file.includes))
    _append_field(
        lines,
        "types",
        _dedupe(symbol.name for symbol in file.symbols if symbol.kind in _TYPE_KINDS),
    )
    if tier == "full":
        _append_field(
            lines,
            "macros",
            _dedupe(
                symbol.name for symbol in file.symbols if symbol.kind == SymbolKind.MACRO
            ),
        )
    _append_field(
        lines,
        "variables",
        _dedupe(
            symbol.name
            for symbol in file.symbols
            if symbol.kind == SymbolKind.VARIABLE and not symbol.is_static
        ),
    )
    if tier in {"full", "no_macros"}:
        _append_field(
            lines,
            "declares",
            _dedupe(
                symbol.name
                for symbol in file.symbols
                if symbol.kind == SymbolKind.PROTOTYPE
            ),
        )
    _append_field(lines, "functions", _function_entries(file, static=False))
    if tier in {"full", "no_macros"}:
        _append_field(lines, "static", _static_entries(file))
    return lines


def _function_entries(file: FileSymbols, static: bool) -> list[str]:
    return _dedupe(
        _span(symbol)
        for symbol in file.symbols
        if symbol.kind == SymbolKind.FUNCTION and symbol.is_static == static
    )


def _static_entries(file: FileSymbols) -> list[str]:
    entries: list[str] = []
    for symbol in file.symbols:
        if symbol.kind == SymbolKind.FUNCTION and symbol.is_static:
            entries.append(_span(symbol))
        elif symbol.kind == SymbolKind.VARIABLE and symbol.is_static:
            entries.append(symbol.name)
    return _dedupe(entries)


def _span(symbol: Symbol) -> str:
    if symbol.start_line == symbol.end_line:
        return f"{symbol.name} {symbol.start_line}"
    return f"{symbol.name} {symbol.start_line}-{symbol.end_line}"


def _append_field(lines: list[str], label: str, items: list[str]) -> None:
    if items:
        lines.append(f"  {label}: {', '.join(items)}")


def _dedupe(items) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique

def _format_project_header(project_files: list[ProjectFile]) -> str:
    file_count = len(project_files)
    line_count = sum(
        len(project_file.content.splitlines()) for project_file in project_files
    )
    return (
        f"Project context: {file_count} files, {line_count} lines. "
        "Paths are relative to the project root."
    )

def _format_project_context(project_files: list[ProjectFile]) -> str:
    """
    Format the project files into a string that can be included in the prompt.

    Args:
        project_files (list[ProjectFile]): A list of ProjectFile objects.

    Returns:
        str: A formatted string representing the project files contents.
    """
    sections = []
    
    for project_file in project_files:
        line_count = len(project_file.content.splitlines())
        sections.append(f"""========================
File: {project_file.path.as_posix()} ({line_count} lines)
========================
{project_file.content}""")
    
    return "\n".join(sections)
