from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal


@dataclass
class ProjectFile:
    path: Path  # relative to the project root
    content: str


class SymbolKind(StrEnum):
    FUNCTION = "function"    # definition with a body
    PROTOTYPE = "prototype"  # declaration without a body
    MACRO = "macro"
    TYPEDEF = "typedef"
    STRUCT = "struct"
    UNION = "union"
    ENUM = "enum"
    CLASS = "class"          # C++
    NAMESPACE = "namespace"  # C++
    VARIABLE = "variable"    # file/namespace scope only


@dataclass(frozen=True)
class Symbol:
    name: str            # C++: qualified, e.g. "drivers::can::Controller::send"
    kind: SymbolKind
    start_line: int      # 1-based
    end_line: int        # 1-based, inclusive
    is_static: bool = False


@dataclass(frozen=True)
class Include:
    target: str          # without quotes/brackets
    is_system: bool      # True for <...>
    line: int            # 1-based


@dataclass(frozen=True)
class FileSymbols:
    path: Path                          # same value as ProjectFile.path
    grammar: Literal["c", "cpp"] | None # None: not parsed
    has_errors: bool                    # root_node.has_error; False when not parsed
    symbols: tuple[Symbol, ...]         # source order
    includes: tuple[Include, ...]       # source order


@dataclass(frozen=True)
class ProjectIndex:
    files: tuple[FileSymbols, ...]      # input order
