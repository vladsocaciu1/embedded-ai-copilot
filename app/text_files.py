import re
from collections.abc import Iterator
from pathlib import Path

from app.models import Call, ProjectFile, Symbol, SymbolKind

KCONFIG_NAMES_PREFIX = "Kconfig"
PIECE_CAP = 1500
FILE_MAX_PIECES = 8

_KCONFIG_KEYWORD = re.compile(
    r"^(config|menuconfig|menu|endmenu|choice|endchoice|if|endif|source|rsource|"
    r"osource|orsource|comment|mainmenu)\b"
)
_KCONFIG_SYMBOL = re.compile(r"^(?:menuconfig|config)\s+([A-Za-z0-9_]+)\s*$")
_RST_UNDERLINE = re.compile(r"^([=\-#*~^\"'`+:.<>_])\1{2,}\s*$")
_DT_LABEL = re.compile(r"^\s*([A-Za-z_]\w*):\s*[\w@,\-]+\s*\{", re.M)
_DT_ALIAS = re.compile(r"^\s*([A-Za-z_][\w\-]*)\s*=\s*&", re.M)
_CONFIG_REF = re.compile(r"\bCONFIG_[A-Z0-9_]+\b")
_CONF_OPTION = re.compile(r"^\s*(CONFIG_[A-Za-z0-9_]+)\s*=", re.M)
_ASM_SECTION_FUNC = re.compile(r"^\s*SECTION_FUNC\(\s*[^,()]+,\s*(\w+)\s*\)")
_ASM_SUBSEC_FUNC = re.compile(r"^\s*SECTION_SUBSEC_FUNC\(\s*[^,()]+,\s*[^,()]+,\s*(\w+)\s*\)")
_ASM_TYPE_FUNC = re.compile(r"^\s*\.type\s+(\w+)\s*,\s*[%@#]function\b")
_ASM_GLOBAL_DIRECTIVE = re.compile(r"^\s*\.glob(?:al|l)\s+(.+?)\s*(?:[;@].*)?$")
_ASM_GLOBAL_MACRO = re.compile(r"^\s*(?:GTEXT|WTEXT)\(\s*(\w+)\s*\)")
_ASM_LABEL = re.compile(r"^([A-Za-z_]\w*):")
_ASM_MACRO = re.compile(r"^\s*\.macro\s+(\w+)")
_ASM_COMMENT_START = ("/*", "*", "//")
_ASM_DECLARATION = re.compile(r"^\s*(?:GTEXT|WTEXT|GDATA|WDATA)\(|^\s*\.glob(?:al|l)\b")


def text_kind(path: Path) -> str | None:
    name = path.name
    suffix = path.suffix.lower()
    if name == "Kconfig" or name.startswith("Kconfig."):
        return "kconfig"
    if name == "CMakeLists.txt" or suffix == ".cmake":
        return "cmake"
    if suffix in (".overlay", ".dts", ".dtsi"):
        return "devicetree"
    if suffix == ".conf":
        return "conf"
    if suffix in (".yaml", ".yml"):
        return "yaml"
    if suffix == ".rst":
        return "rst"
    if suffix in (".s", ".asm"):
        return "asm"
    return None


def index_text(project_file: ProjectFile, kind: str) -> tuple[Symbol, ...]:
    content = project_file.content
    if kind == "kconfig":
        return tuple(_kconfig_symbols(content))
    if kind == "rst":
        return tuple(_rst_symbols(content))
    if kind == "devicetree":
        names = [m.group(1) for m in _DT_LABEL.finditer(content)]
        names += [m.group(1) for m in _DT_ALIAS.finditer(content)]
        return tuple(Symbol(n, SymbolKind.NODE, 1, 1) for n in dict.fromkeys(names))
    if kind == "asm":
        return tuple(_asm_symbols(content))
    if kind == "conf":
        names = [m.group(1) for m in _CONF_OPTION.finditer(content)]
        return tuple(Symbol(n, SymbolKind.CONFIG, 1, 1) for n in dict.fromkeys(names))
    return ()


def _kconfig_symbols(content: str) -> Iterator[Symbol]:
    lines = content.splitlines()
    name = None
    start = 0
    for number, line in enumerate(lines, 1):
        if not _KCONFIG_KEYWORD.match(line):
            continue
        if name is not None:
            yield Symbol(name, SymbolKind.CONFIG, start, number - 1)
            name = None
        match = _KCONFIG_SYMBOL.match(line)
        if match:
            name, start = "CONFIG_" + match.group(1), number
    if name is not None:
        yield Symbol(name, SymbolKind.CONFIG, start, len(lines))


def _rst_symbols(content: str) -> Iterator[Symbol]:
    lines = content.splitlines()
    heads: list[tuple[str, int]] = []
    i = 0
    while i < len(lines) - 1:
        title, under = lines[i].strip(), lines[i + 1].strip()
        if (
            title
            and not lines[i].startswith((" ", "\t", ".."))
            and not _RST_UNDERLINE.match(title)
            and _RST_UNDERLINE.match(under)
            and len(under) >= len(title)
        ):
            heads.append((title, i + 1))
            i += 2
        else:
            i += 1
    for index, (title, line) in enumerate(heads):
        end = heads[index + 1][1] - 1 if index + 1 < len(heads) else len(lines)
        yield Symbol(title, SymbolKind.SECTION, line, end)


def pieces(text: str, cap: int, limit: int) -> list[str]:
    out: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.splitlines():
        line = line[:cap]
        if size + len(line) + 1 > cap and current:
            out.append("\n".join(current))
            current, size = [], 0
            if len(out) == limit:
                return out
        current.append(line)
        size += len(line) + 1
    if current and len(out) < limit:
        out.append("\n".join(current))
    return out


def _asm_symbols(content: str) -> Iterator[Symbol]:
    lines = content.splitlines()
    declared: set[str] = set()
    for line in lines:
        glob = _ASM_GLOBAL_DIRECTIVE.match(line)
        if glob:
            declared.update(n.strip() for n in glob.group(1).split(",") if n.strip())
        macro = _ASM_GLOBAL_MACRO.match(line)
        if macro:
            declared.add(macro.group(1))
    typed = {m.group(1) for line in lines if (m := _ASM_TYPE_FUNC.match(line))}
    strong: list[tuple[int, str]] = []
    weak: list[tuple[int, str]] = []
    macros: list[Symbol] = []
    for number, line in enumerate(lines, 1):
        found = _ASM_SECTION_FUNC.match(line) or _ASM_SUBSEC_FUNC.match(line)
        if found:
            strong.append((number, found.group(1)))
        else:
            label = _ASM_LABEL.match(line)
            if label and label.group(1) in typed:
                strong.append((number, label.group(1)))
            elif label and label.group(1) in declared:
                weak.append((number, label.group(1)))
        macro_def = _ASM_MACRO.match(line)
        if macro_def:
            macros.append(Symbol(macro_def.group(1), SymbolKind.MACRO, number, number))
    starts = strong or weak
    ends = []
    for i, (number, _) in enumerate(starts):
        if i + 1 < len(starts):
            ends.append(_asm_end(lines, number, starts[i + 1][0]))
        else:
            ends.append(len(lines))
    for (number, name), end in zip(starts, ends):
        yield Symbol(name, SymbolKind.FUNCTION, number, end)
    yield from macros


def _asm_end(lines: list[str], start: int, next_start: int) -> int:
    end = next_start - 1
    while end > start and (not lines[end - 1].strip() or _ASM_DECLARATION.match(lines[end - 1])):
        end -= 1
    comment_end = end
    while comment_end > start and lines[comment_end - 1].strip().startswith(_ASM_COMMENT_START):
        comment_end -= 1
    if comment_end < end and comment_end >= start:
        end = comment_end
        while end > start and not lines[end - 1].strip():
            end -= 1
    return max(end, start)


def config_refs(content: str) -> tuple[Call, ...]:
    if "CONFIG_" not in content:
        return ()
    found: list[Call] = []
    for number, line in enumerate(content.splitlines(), 1):
        if "CONFIG_" not in line:
            continue
        for name in dict.fromkeys(_CONFIG_REF.findall(line)):
            found.append(Call(caller="", callee=name, line=number, is_member=False))
    return tuple(found)
