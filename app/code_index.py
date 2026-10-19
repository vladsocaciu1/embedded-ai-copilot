import logging
import re
from collections.abc import Sequence
from time import perf_counter

import tree_sitter_c
import tree_sitter_cpp
from tree_sitter import Language, Node, Parser, Query, QueryCursor

from app.models import (
    Call,
    FileSymbols,
    Include,
    ProjectFile,
    ProjectIndex,
    Symbol,
    SymbolKind,
)

logger = logging.getLogger(__name__)

C_SUFFIXES = frozenset({".c"})
CPP_SUFFIXES = frozenset({
    ".h",
    ".cpp",
    ".cc",
    ".cxx",
    ".hpp",
    ".hh",
    ".hxx",
    ".inl",
})

_C_LANGUAGE = Language(tree_sitter_c.language())
_CPP_LANGUAGE = Language(tree_sitter_cpp.language())
_CALL_QUERY = "(call_expression function: (_) @function)"
_C_CALL_QUERY = Query(_C_LANGUAGE, _CALL_QUERY)
_CPP_CALL_QUERY = Query(_CPP_LANGUAGE, _CALL_QUERY)

_CONTAINERS = frozenset({
    "translation_unit",
    "preproc_if",
    "preproc_ifdef",
    "preproc_else",
    "preproc_elif",
    "preproc_elifdef",
    "declaration_list",
    "template_declaration",
    "ERROR",
    "field_declaration_list",
})

_TYPE_KINDS = {
    "struct_specifier": SymbolKind.STRUCT,
    "union_specifier": SymbolKind.UNION,
    "enum_specifier": SymbolKind.ENUM,
    "class_specifier": SymbolKind.CLASS,
}

_FUNCTION_NAME_TYPES = frozenset({
    "identifier",
    "field_identifier",
    "qualified_identifier",
    "destructor_name",
    "operator_name",
    "template_function",
})

_NAME_TYPES = _FUNCTION_NAME_TYPES | {"type_identifier"}

_FIELD_STEP = frozenset({
    "pointer_declarator",
    "array_declarator",
    "init_declarator",
    "attributed_declarator",
    "function_declarator",
})

_CHILD_STEP = frozenset({
    "parenthesized_declarator",
    "reference_declarator",
})

_WS_RE = re.compile(r"\s+")
_COLON_RE = re.compile(r"\s*::\s*")

# (node, scope, include-guard name, in_class)
_Frame = tuple[Node, tuple[str, ...], str | None, bool]


def index_file(project_file: ProjectFile) -> FileSymbols:
    grammar = _grammar_for(project_file.path)
    if grammar is None:
        return FileSymbols(
            path=project_file.path,
            grammar=None,
            has_errors=False,
            symbols=(),
            includes=(),
            calls=(),
        )

    language = _C_LANGUAGE if grammar == "c" else _CPP_LANGUAGE
    parser = Parser(language)
    tree = parser.parse(project_file.content.encode("utf-8"))
    root = tree.root_node
    symbols, includes, calls = _collect(root, grammar == "cpp")
    has_errors = root.has_error
    if has_errors:
        logger.debug(f"Parse errors in {project_file.path} ({grammar}).")
    return FileSymbols(
        path=project_file.path,
        grammar=grammar,
        has_errors=has_errors,
        symbols=tuple(symbols),
        includes=tuple(includes),
        calls=tuple(calls),
    )


def build_index(project_files: Sequence[ProjectFile]) -> ProjectIndex:
    started = perf_counter()
    indexed = tuple(index_file(project_file) for project_file in project_files)
    elapsed_ms = round((perf_counter() - started) * 1000)
    c_files = sum(file.grammar == "c" for file in indexed)
    cpp_files = sum(file.grammar == "cpp" for file in indexed)
    skipped = sum(file.grammar is None for file in indexed)
    errored = sum(file.has_errors for file in indexed)
    symbol_count = sum(len(file.symbols) for file in indexed)
    include_count = sum(len(file.includes) for file in indexed)
    call_count = sum(len(file.calls) for file in indexed)
    logger.info(
        f"Indexed {len(indexed)} files "
        f"({c_files} C, {cpp_files} C++, {skipped} not parsed, "
        f"{errored} with parse errors): "
        f"{symbol_count} symbols, {include_count} includes, {call_count} calls "
        f"in {elapsed_ms} ms."
    )
    return ProjectIndex(files=indexed)


def _grammar_for(path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in C_SUFFIXES:
        return "c"
    if suffix in CPP_SUFFIXES:
        return "cpp"
    return None


def _collect(
    root: Node, cpp: bool
) -> tuple[list[Symbol], list[Include], list[Call]]:
    symbols: list[Symbol] = []
    includes: list[Include] = []
    calls: list[Call] = []
    stack: list[_Frame] = [(root, (), None, False)]
    while stack:
        node, scope, guard, in_class = stack.pop()
        node_type = node.type
        if node_type in _CONTAINERS:
            _push_children(node, scope, guard, in_class, stack)
        elif node_type == "linkage_specification":
            _push_body(node, scope, guard, in_class, stack)
        elif node_type == "namespace_definition":
            _handle_namespace(node, scope, guard, in_class, cpp, symbols, stack)
        elif node_type == "function_definition":
            _handle_function(node, scope, cpp, symbols, calls)
        elif node_type == "declaration":
            _handle_declaration(node, scope, guard, in_class, cpp, symbols, stack)
        elif node_type == "type_definition":
            _handle_typedef(node, scope, guard, cpp, symbols, stack)
        elif node_type == "field_declaration":
            _handle_field(node, scope, guard, in_class, cpp, symbols, stack)
        elif node_type in _TYPE_KINDS:
            _handle_type_specifier(node, scope, guard, cpp, symbols, stack)
        elif node_type in {"preproc_def", "preproc_function_def"}:
            _handle_macro(node, guard, symbols)
        elif node_type == "preproc_include":
            _handle_include(node, includes)
    return symbols, includes, calls


def _push_children(
    node: Node,
    scope: tuple[str, ...],
    guard: str | None,
    in_class: bool,
    stack: list[_Frame],
) -> None:
    if node.type == "preproc_ifdef":
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            guard = _node_name(name_node)
    for child in reversed(node.named_children):
        stack.append((child, scope, guard, in_class))


def _push_body(
    node: Node,
    scope: tuple[str, ...],
    guard: str | None,
    in_class: bool,
    stack: list[_Frame],
) -> None:
    body = node.child_by_field_name("body")
    if body is not None:
        stack.append((body, scope, guard, in_class))


def _handle_namespace(
    node: Node,
    scope: tuple[str, ...],
    guard: str | None,
    in_class: bool,
    cpp: bool,
    symbols: list[Symbol],
    stack: list[_Frame],
) -> None:
    name_node = node.child_by_field_name("name")
    new_scope = scope
    if name_node is not None:
        name = _node_name(name_node)
        if name:
            symbols.append(_make_symbol(name, SymbolKind.NAMESPACE, node, scope, cpp))
            new_scope = scope + (name,)
    _push_body(node, new_scope, guard, in_class, stack)


def _handle_function(
    node: Node,
    scope: tuple[str, ...],
    cpp: bool,
    symbols: list[Symbol],
    calls: list[Call],
) -> None:
    declarator = node.child_by_field_name("declarator")
    if declarator is None:
        return
    name = _function_name(declarator)
    if not name:
        return
    symbol = _make_symbol(
        name,
        SymbolKind.FUNCTION,
        node,
        scope,
        cpp,
        is_static=_is_static(node),
    )
    symbols.append(symbol)
    body = node.child_by_field_name("body")
    if body is None:
        return
    calls.extend(_calls_in_body(body, symbol.name, cpp))


def _calls_in_body(body: Node, caller: str, cpp: bool) -> list[Call]:
    query = _CPP_CALL_QUERY if cpp else _C_CALL_QUERY
    captured = QueryCursor(query).captures(body).get("function", [])
    found: dict[tuple[str, int, bool], Call] = {}
    for node in captured:
        extracted = _callee(node)
        if extracted is None:
            continue
        callee, is_member = extracted
        line = _start_line(node)
        key = (callee, line, is_member)
        if key in found:
            continue
        found[key] = Call(
            caller=caller,
            callee=callee,
            line=line,
            is_member=is_member,
        )
    return sorted(found.values(), key=lambda call: (call.line, call.callee))


def _callee(node: Node) -> tuple[str, bool] | None:
    node_type = node.type
    if node_type in {"identifier", "qualified_identifier"}:
        callee = _normalize_name(_text(node))
        if not callee:
            return None
        return callee, False
    if node_type == "template_function":
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return None
        callee = _text(name_node)
        if not callee:
            return None
        return callee, False
    if node_type == "field_expression":
        field_node = node.child_by_field_name("field")
        if field_node is None:
            return None
        callee = _text(field_node)
        if not callee:
            return None
        return callee, True
    return None


def _handle_declaration(
    node: Node,
    scope: tuple[str, ...],
    guard: str | None,
    in_class: bool,
    cpp: bool,
    symbols: list[Symbol],
    stack: list[_Frame],
) -> None:
    _handle_type_field(node, scope, guard, cpp, symbols, stack)
    static = _is_static(node)
    extern = _is_extern(node)
    for declarator in node.children_by_field_name("declarator"):
        name = _function_name(declarator)
        if name:
            symbols.append(
                _make_symbol(
                    name,
                    SymbolKind.PROTOTYPE,
                    node,
                    scope,
                    cpp,
                    is_static=static,
                )
            )
            continue
        if in_class or extern:
            continue
        var_name = _innermost_name(declarator)
        if not var_name:
            continue
        symbols.append(
            _make_symbol(
                var_name,
                SymbolKind.VARIABLE,
                node,
                scope,
                cpp,
                is_static=static,
            )
        )


def _handle_field(
    node: Node,
    scope: tuple[str, ...],
    guard: str | None,
    in_class: bool,
    cpp: bool,
    symbols: list[Symbol],
    stack: list[_Frame],
) -> None:
    _handle_type_field(node, scope, guard, cpp, symbols, stack)
    if not in_class:
        return
    static = _is_static(node)
    for declarator in node.children_by_field_name("declarator"):
        name = _function_name(declarator)
        if not name:
            continue
        symbols.append(
            _make_symbol(
                name,
                SymbolKind.PROTOTYPE,
                node,
                scope,
                cpp,
                is_static=static,
            )
        )


def _handle_typedef(
    node: Node,
    scope: tuple[str, ...],
    guard: str | None,
    cpp: bool,
    symbols: list[Symbol],
    stack: list[_Frame],
) -> None:
    _handle_type_field(node, scope, guard, cpp, symbols, stack)
    for declarator in node.children_by_field_name("declarator"):
        name = _innermost_name(declarator)
        if not name:
            continue
        symbols.append(_make_symbol(name, SymbolKind.TYPEDEF, node, scope, cpp))


def _handle_type_field(
    node: Node,
    scope: tuple[str, ...],
    guard: str | None,
    cpp: bool,
    symbols: list[Symbol],
    stack: list[_Frame],
) -> None:
    type_node = node.child_by_field_name("type")
    if type_node is not None:
        _handle_type_specifier(type_node, scope, guard, cpp, symbols, stack)


def _handle_type_specifier(
    node: Node,
    scope: tuple[str, ...],
    guard: str | None,
    cpp: bool,
    symbols: list[Symbol],
    stack: list[_Frame],
) -> None:
    kind = _TYPE_KINDS.get(node.type)
    if kind is None:
        return
    name_node = node.child_by_field_name("name")
    body = node.child_by_field_name("body")
    if name_node is None or body is None:
        return
    name = _node_name(name_node)
    if not name:
        return
    symbols.append(_make_symbol(name, kind, node, scope, cpp))
    if cpp and node.type in {"class_specifier", "struct_specifier"}:
        stack.append((body, scope + (name,), guard, True))


def _handle_macro(node: Node, guard: str | None, symbols: list[Symbol]) -> None:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = _node_name(name_node)
    if not name:
        return
    if (
        node.type == "preproc_def"
        and node.child_by_field_name("value") is None
        and guard is not None
        and name == guard
    ):
        return
    symbols.append(_make_symbol(name, SymbolKind.MACRO, node, (), False))


def _handle_include(node: Node, includes: list[Include]) -> None:
    path_node = node.child_by_field_name("path")
    if path_node is None:
        return
    if path_node.type == "string_literal":
        is_system = False
    elif path_node.type == "system_lib_string":
        is_system = True
    else:
        return
    text = _text(path_node)
    if len(text) < 2:
        return
    includes.append(
        Include(target=text[1:-1], is_system=is_system, line=_start_line(node))
    )


def _make_symbol(
    name: str,
    kind: SymbolKind,
    node: Node,
    scope: tuple[str, ...],
    cpp: bool,
    is_static: bool = False,
) -> Symbol:
    return Symbol(
        name=_qualify(scope, name, cpp),
        kind=kind,
        start_line=_start_line(node),
        end_line=_end_line(node),
        is_static=is_static,
    )


def _function_name(node: Node | None) -> str | None:
    while node is not None and node.type in _FIELD_STEP | _CHILD_STEP:
        if node.type == "function_declarator":
            inner = node.child_by_field_name("declarator")
            if inner is not None and inner.type in _FUNCTION_NAME_TYPES:
                return _node_name(inner)
        node = _next_declarator(node)
    return None


def _innermost_name(node: Node | None) -> str | None:
    while node is not None and (
        node.type in _FIELD_STEP or node.type in _CHILD_STEP
    ):
        nxt = _next_declarator(node)
        if nxt is None:
            break
        node = nxt
    if node is not None and node.type in _NAME_TYPES:
        return _node_name(node)
    return None


def _next_declarator(node: Node) -> Node | None:
    if node.type in _FIELD_STEP:
        return node.child_by_field_name("declarator")
    if node.type in _CHILD_STEP:
        named = node.named_children
        if not named:
            return None
        return named[-1]
    return None


def _is_static(node: Node) -> bool:
    return "static" in _storage_texts(node)


def _is_extern(node: Node) -> bool:
    return "extern" in _storage_texts(node)


def _storage_texts(node: Node) -> set[str]:
    return {
        _text(child).strip()
        for child in node.named_children
        if child.type == "storage_class_specifier"
    }


def _qualify(scope: tuple[str, ...], name: str, cpp: bool) -> str:
    if cpp and scope:
        return "::".join((*scope, name))
    return name


def _node_name(node: Node) -> str:
    return _normalize_name(_text(node))


def _normalize_name(text: str) -> str:
    collapsed = _WS_RE.sub(" ", text).strip()
    return _COLON_RE.sub("::", collapsed)


def _text(node: Node) -> str:
    raw = node.text
    if not raw:
        return ""
    return raw.decode("utf-8", errors="replace")


def _start_line(node: Node) -> int:
    # Tuple index only. Attribute access on a point segfaults past 256.
    return node.start_point[0] + 1


def _end_line(node: Node) -> int:
    start_row = node.start_point[0]
    end_row = node.end_point[0]
    end_col = node.end_point[1]
    if end_col == 0 and end_row > start_row:
        return end_row
    return end_row + 1
