import hashlib
import math
import textwrap
from pathlib import Path

import pytest

from app.chunker import (
    CARD_SYMBOL_KINDS,
    CHUNK_FORMAT,
    chunk_file,
    chunk_index,
    unique_texts,
)
from app.code_index import build_index, index_file
from app.config import DEFAULT_EXCLUDE_DIRS
from app.models import ProjectFile, SymbolKind
from app.project_loader import load_project_files
from scripts.chunk_stats import main

GOLDEN_PATH = Path("Projects/Board/Examples/GPIO/Src/main.c")
GOLDEN = textwrap.dedent(
    """\
    /**
      ******************************************************************************
      * @file    GPIO/Src/main.c
      * @author  MCD Application Team
      * @brief   This example shows how to toggle the LEDs.
      ******************************************************************************
      * @attention
      * Copyright (c) 2017 STMicroelectronics.
      * All rights reserved.
      ******************************************************************************
      */
    #include "main.h"

    typedef struct { int a; } Pair;

    /**
      * @brief  Toggles LED3 forever.
      * @retval None
      */
    void Led_Toggle(void)
    {
      HAL_GPIO_TogglePin(GPIOD, GPIO_PIN_13);
    }

    static int helper(int x) { return x + 1; }
    """
)
CARD_TEXT = (
    "file Examples/GPIO/Src/main.c\n"
    "@file    GPIO/Src/main.c @brief   This example shows how to toggle the LEDs.\n"
    "symbols: Pair Led_Toggle helper"
)
LED_TEXT = (
    "function Led_Toggle in Examples/GPIO/Src/main.c\n"
    "@brief  Toggles LED3 forever. @retval None\n"
    "void Led_Toggle(void)\n"
    "{\n"
    "  HAL_GPIO_TogglePin(GPIOD, GPIO_PIN_13);\n"
    "}"
)
HELPER_TEXT = (
    "function helper in Examples/GPIO/Src/main.c\n"
    "\n"
    "static int helper(int x) { return x + 1; }"
)


def _chunks(path: str, content: str):
    project_file = ProjectFile(path=Path(path), content=content)
    return chunk_file(index_file(project_file), content)


def _golden():
    return _chunks(str(GOLDEN_PATH), GOLDEN)


def test_c1_golden_card_drops_noise():
    card = _golden()[0]

    assert card.text == CARD_TEXT
    assert "Copyright" not in card.text
    assert "@author" not in card.text
    assert "@attention" not in card.text
    assert "***" not in card.text


def test_c2_golden_function_chunks_and_lines():
    card, led, helper = _golden()

    assert card.kind == "file"
    assert card.name == ""
    assert card.start_line == 1
    assert card.end_line == 25
    assert led.text == LED_TEXT
    assert led.kind == "function"
    assert led.name == "Led_Toggle"
    assert (led.start_line, led.end_line) == (20, 23)
    assert helper.text == HELPER_TEXT
    assert helper.kind == "function"
    assert helper.name == "helper"
    assert (helper.start_line, helper.end_line) == (25, 25)


def test_c3_file_comment_cap_and_absent_comment():
    long_comment = "/* " + ("alpha " * 200) + "*/\nvoid f(void) {}\n"
    card = _chunks("src/long.c", long_comment)[0]
    assert len(card.text.splitlines()[1]) == 600

    include = _chunks("src/a.c", '#include "main.h"\nvoid f(void) {}\n')[0]
    assert include.text.splitlines()[1] == ""

    slashes = _chunks("src/a.c", "// not the card comment\nvoid f(void) {}\n")[0]
    assert slashes.text.splitlines()[1] == ""


def test_c4_card_names_filter_cap_and_empty():
    content = textwrap.dedent(
        """\
        #define BUF 1
        union U { int a; };
        void proto(void);
        int var;
        struct Point { int x; };
        enum Color { RED };
        typedef int MyInt;
        void func(void) {}
        """
    )
    project_file = ProjectFile(path=Path("src/kinds.c"), content=content)
    file_symbols = index_file(project_file)
    present = {symbol.kind for symbol in file_symbols.symbols}
    assert {
        SymbolKind.MACRO,
        SymbolKind.UNION,
        SymbolKind.PROTOTYPE,
        SymbolKind.VARIABLE,
        SymbolKind.STRUCT,
        SymbolKind.ENUM,
        SymbolKind.TYPEDEF,
        SymbolKind.FUNCTION,
    } <= present
    card = chunk_file(file_symbols, content)[0]
    assert card.text.splitlines()[-1] == "symbols: Point Color MyInt func"

    none = textwrap.dedent(
        """\
        #define ONLY 1
        union U { int a; };
        void proto(void);
        int var;
        """
    )
    none_card = _chunks("src/none.c", none)[0]
    assert none_card.text.splitlines()[-1] == "symbols: "

    duplicated = "#ifdef A\nvoid f(void) {}\n#else\nvoid f(void) {}\n#endif\n"
    duplicated_card = _chunks("src/dup.c", duplicated)[0]
    assert duplicated_card.text.splitlines()[-1] == "symbols: f f"

    many = "".join(f"void f{i}(void) {{}}\n" for i in range(45))
    many_card = _chunks("src/many.c", many)[0]
    names = " ".join(f"f{i}" for i in range(40))
    assert many_card.text.splitlines()[-1] == f"symbols: {names}"
    assert "f40" not in many_card.text
    assert set(CARD_SYMBOL_KINDS) == {
        SymbolKind.FUNCTION,
        SymbolKind.TYPEDEF,
        SymbolKind.STRUCT,
        SymbolKind.ENUM,
    }


def test_c5_lead_comment_rules():
    slash = _chunks(
        "src/a.c",
        "// toggles the pin\n// forever\nvoid f(void) {}\n",
    )[1]
    assert slash.text.splitlines()[1] == "toggles the pin forever"

    blank = _chunks("src/a.c", "// hello\n\nvoid f(void) {}\n")[1]
    assert blank.text.splitlines()[1] == ""

    lines = [f"// c{i:02d} text" for i in range(30)]
    lines.append("void f(void) {}")
    nearest = _chunks("src/a.c", "\n".join(lines) + "\n")[1]
    expected = " ".join(f"c{i:02d} text" for i in range(5, 30))
    assert nearest.text.splitlines()[1] == expected

    long_lines = ["// " + ("abcd " * 20) for _ in range(25)]
    long_lines.append("void f(void) {}")
    capped = _chunks("src/a.c", "\n".join(long_lines) + "\n")[1]
    assert len(capped.text.splitlines()[1]) == 400


def test_c6_body_cap_and_inclusive_span():
    content = "void f(void)\n{\n  return 1;\n}\n"
    body = _chunks("src/a.c", content)[1].text.split("\n", 2)[2]
    assert body == "void f(void)\n{\n  return 1;\n}"

    long_body = ["void f(void)", "{"]
    long_body.extend(f"  int x{i} = {i};" for i in range(200))
    long_body.append("}")
    full = "\n".join(long_body)
    capped = _chunks("src/a.c", full + "\n")[1].text.split("\n", 2)[2]
    assert len(capped) == 1500
    assert capped == full[:1500]


def test_c7_tail_keeps_last_four_or_whole_path():
    deep = _chunks("a/b/c/d/e/f.c", "void f(void) {}\n")[0]
    assert deep.text.startswith("file c/d/e/f.c\n")

    short = _chunks("src/a.c", "void f(void) {}\n")[0]
    assert short.text.startswith("file src/a.c\n")


def test_c8_unparsed_empty_and_header_card_only():
    assert _chunks("startup.s", "nop\n") == ()
    assert _chunks("link.ld", "SECTIONS {}\n") == ()

    header = _chunks("inc/a.h", "void g(void);\nint x;\n")
    assert len(header) == 1
    assert header[0].kind == "file"


def test_c9_cpp_qualified_method_name():
    content = textwrap.dedent(
        """\
        namespace ns {
        class C {
        public:
            void m() { return; }
        };
        }
        """
    )
    chunks = _chunks("src/ns.cpp", content)
    method = chunks[1]

    assert method.kind == "function"
    assert method.name == "ns::C::m"
    assert method.text == (
        "function ns::C::m in src/ns.cpp\n"
        "\n"
        "    void m() { return; }"
    )
    assert (method.start_line, method.end_line) == (4, 4)


def test_c10_key_is_format_plus_text_and_tracks_body():
    chunks = _golden()
    for chunk in chunks:
        digest = hashlib.sha256(("chunk-v1\n" + chunk.text).encode()).hexdigest()
        assert chunk.key == digest

    content = "void kept(void) { return 1; }\nvoid edited(void) { return 1; }\n"
    left = _chunks("board/Examples/GPIO/Src/a.c", content)
    right = _chunks("other/Examples/GPIO/Src/a.c", content)
    assert [chunk.key for chunk in left] == [chunk.key for chunk in right]

    changed = content.replace("edited(void) { return 1; }", "edited(void) { return 2; }")
    edited = _chunks("board/Examples/GPIO/Src/a.c", changed)
    assert edited[0].key == left[0].key
    assert edited[1].key == left[1].key
    assert edited[2].key != left[2].key


def test_c11_unique_order_index_order_and_missing_file():
    content = "void a(void) {}\n"
    first = ProjectFile(path=Path("board/Examples/GPIO/Src/a.c"), content=content)
    second = ProjectFile(path=Path("other/Examples/GPIO/Src/a.c"), content=content)
    third = ProjectFile(path=Path("src/c.c"), content="void c(void) {}\n")
    index = build_index([first, second, third])
    chunks = chunk_index(index, [first, second, third])

    assert [chunk.kind for chunk in chunks] == [
        "file",
        "function",
        "file",
        "function",
        "file",
        "function",
    ]
    assert [chunk.path for chunk in chunks] == [
        first.path,
        first.path,
        second.path,
        second.path,
        third.path,
        third.path,
    ]
    assert [chunk.name for chunk in chunks if chunk.kind == "function"] == [
        "a",
        "a",
        "c",
    ]
    texts = unique_texts(chunks)
    assert list(texts) == [
        chunks[0].key,
        chunks[1].key,
        chunks[4].key,
        chunks[5].key,
    ]
    assert texts[chunks[0].key] == chunks[0].text
    assert chunks[0].key == chunks[2].key
    assert chunks[1].key == chunks[3].key

    with pytest.raises(ValueError, match="board/Examples/GPIO/Src/a.c"):
        chunk_index(index, [])


def test_c12_crlf_matches_lf():
    lf = _golden()
    crlf = _chunks(str(GOLDEN_PATH), GOLDEN.replace("\n", "\r\n"))

    assert [chunk.text for chunk in crlf] == [chunk.text for chunk in lf]


def test_c13_deterministic_and_hashable():
    project_file = ProjectFile(path=GOLDEN_PATH, content=GOLDEN)
    file_symbols = index_file(project_file)

    assert chunk_file(file_symbols, GOLDEN) == chunk_file(file_symbols, GOLDEN)
    chunks = chunk_file(file_symbols, GOLDEN)
    assert len(set(chunks)) == len(chunks)


def test_c14_chunk_stats_lines_and_usage(tmp_path, capsys):
    source = tmp_path / "src"
    source.mkdir()
    (source / "a.c").write_text(
        "void f(void) { return; }\n",
        encoding="utf-8",
    )
    (source / "b.h").write_text("void g(void);\n", encoding="utf-8")
    (source / "c.s").write_text("nop\n", encoding="utf-8")

    main([str(tmp_path)])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 6

    files_n, parsed = _pair(lines[0], "files", "parsed")
    chunks_n, file_n, function_n, unique = _chunks_line(lines[1])
    chars, est_tokens = _pair(lines[2], "chars", "est_tokens")
    assert lines[3].startswith("chunk_ms ")
    int(lines[3].split()[1])
    minutes = int(lines[4].removeprefix("local embed at 16 chunks/s: ~").removesuffix(" min"))
    days = int(lines[5].removeprefix("gemini free tier at 1000 texts/day: ~").removesuffix(" days"))

    loaded = load_project_files(tmp_path, DEFAULT_EXCLUDE_DIRS)
    index = build_index(loaded)
    chunks = chunk_index(index, loaded)
    texts = unique_texts(chunks)
    assert files_n == len(loaded)
    assert parsed == sum(item.grammar is not None for item in index.files)
    assert chunks_n == len(chunks)
    assert file_n == sum(chunk.kind == "file" for chunk in chunks)
    assert function_n == sum(chunk.kind == "function" for chunk in chunks)
    assert unique == len(texts)
    assert chars == sum(len(text) for text in texts.values())
    assert est_tokens == chars // 4
    assert minutes == math.ceil(unique / 16 / 60)
    assert days == math.ceil(unique / 1000)

    main([])
    usage = capsys.readouterr().out
    assert usage == "usage: python scripts/chunk_stats.py <project_dir>\n"


def _pair(line: str, left: str, right: str) -> tuple[int, int]:
    parts = line.split()
    assert parts[0] == left
    assert parts[2] == right
    return int(parts[1]), int(parts[3])


def _chunks_line(line: str) -> tuple[int, int, int, int]:
    # chunks C (file F, function G) unique U
    parts = line.replace("(", "").replace(")", "").replace(",", "").split()
    assert parts[0] == "chunks"
    assert parts[2] == "file"
    assert parts[4] == "function"
    assert parts[6] == "unique"
    return int(parts[1]), int(parts[3]), int(parts[5]), int(parts[7])


def _baseline_c_sources() -> list[ProjectFile]:
    """C/C++ inputs of the tests above, built the same way, one file at a time.

    Captured on main 2845fb2 before text-file chunking. H6 compares the key set.
    """
    files: list[ProjectFile] = []

    def add(path: str, content: str) -> None:
        files.append(ProjectFile(path=Path(path), content=content))

    add(str(GOLDEN_PATH), GOLDEN)
    add(str(GOLDEN_PATH), GOLDEN.replace("\n", "\r\n"))

    long_comment = "/* " + ("alpha " * 200) + "*/\nvoid f(void) {}\n"
    add("src/long.c", long_comment)
    add("src/a.c", '#include "main.h"\nvoid f(void) {}\n')
    add("src/a.c", "// not the card comment\nvoid f(void) {}\n")

    kinds = textwrap.dedent(
        """\
        #define BUF 1
        union U { int a; };
        void proto(void);
        int var;
        struct Point { int x; };
        enum Color { RED };
        typedef int MyInt;
        void func(void) {}
        """
    )
    add("src/kinds.c", kinds)
    none = textwrap.dedent(
        """\
        #define ONLY 1
        union U { int a; };
        void proto(void);
        int var;
        """
    )
    add("src/none.c", none)
    add("src/dup.c", "#ifdef A\nvoid f(void) {}\n#else\nvoid f(void) {}\n#endif\n")
    add("src/many.c", "".join(f"void f{i}(void) {{}}\n" for i in range(45)))

    add("src/a.c", "// toggles the pin\n// forever\nvoid f(void) {}\n")
    add("src/a.c", "// hello\n\nvoid f(void) {}\n")
    lines = [f"// c{i:02d} text" for i in range(30)]
    lines.append("void f(void) {}")
    add("src/a.c", "\n".join(lines) + "\n")
    long_lines = ["// " + ("abcd " * 20) for _ in range(25)]
    long_lines.append("void f(void) {}")
    add("src/a.c", "\n".join(long_lines) + "\n")

    add("src/a.c", "void f(void)\n{\n  return 1;\n}\n")
    long_body = ["void f(void)", "{"]
    long_body.extend(f"  int x{i} = {i};" for i in range(200))
    long_body.append("}")
    add("src/a.c", "\n".join(long_body) + "\n")

    add("a/b/c/d/e/f.c", "void f(void) {}\n")
    add("src/a.c", "void f(void) {}\n")
    add("startup.s", "nop\n")
    add("link.ld", "SECTIONS {}\n")
    add("inc/a.h", "void g(void);\nint x;\n")

    cpp = textwrap.dedent(
        """\
        namespace ns {
        class C {
        public:
            void m() { return; }
        };
        }
        """
    )
    add("src/ns.cpp", cpp)

    kept = "void kept(void) { return 1; }\nvoid edited(void) { return 1; }\n"
    add("board/Examples/GPIO/Src/a.c", kept)
    add("other/Examples/GPIO/Src/a.c", kept)
    changed = kept.replace(
        "edited(void) { return 1; }", "edited(void) { return 2; }"
    )
    add("board/Examples/GPIO/Src/a.c", changed)

    add("board/Examples/GPIO/Src/a.c", "void a(void) {}\n")
    add("other/Examples/GPIO/Src/a.c", "void a(void) {}\n")
    add("src/c.c", "void c(void) {}\n")

    add("src/a.c", "void f(void) { return; }\n")
    add("src/b.h", "void g(void);\n")
    add("src/c.s", "nop\n")
    return files


def test_h6_existing_c_chunk_keys_unchanged():
    import json

    expected = json.loads(
        (Path(__file__).parent / "data" / "chunk_keys_main.json").read_text(
            encoding="utf-8"
        )
    )
    keys: list[str] = []
    for project_file in _baseline_c_sources():
        index = build_index([project_file])
        chunks = chunk_index(index, [project_file])
        keys.extend(chunk.key for chunk in chunks)
    assert sorted(set(keys)) == expected


_KCONFIG = (
    'config ALPHA\n'
    '\tbool "a"\n'
    "\n"
    "menuconfig BETA\n"
    '\tbool "b"\n'
    "\n"
    "config $(X)\n"
    '\tbool "skip"\n'
    "\n"
    "config GAMMA\n"
    '\tint "g"\n'
)
_KCONFIG_CARD = (
    "file kernel/Kconfig\n"
    "symbols: CONFIG_ALPHA CONFIG_BETA CONFIG_GAMMA"
)
_KCONFIG_ALPHA = (
    "config CONFIG_ALPHA in kernel/Kconfig\n"
    "config ALPHA\n"
    '\tbool "a"'
)
_KCONFIG_BETA = (
    "config CONFIG_BETA in kernel/Kconfig\n"
    "menuconfig BETA\n"
    '\tbool "b"'
)
_KCONFIG_GAMMA = (
    "config CONFIG_GAMMA in kernel/Kconfig\n"
    "config GAMMA\n"
    '\tint "g"'
)


def test_h1_kconfig_card_and_config_chunks():
    chunks = _chunks("kernel/Kconfig", _KCONFIG)

    assert [chunk.kind for chunk in chunks] == ["file", "config", "config", "config"]
    assert [chunk.name for chunk in chunks] == [
        "",
        "CONFIG_ALPHA",
        "CONFIG_BETA",
        "CONFIG_GAMMA",
    ]
    assert [chunk.text for chunk in chunks] == [
        _KCONFIG_CARD,
        _KCONFIG_ALPHA,
        _KCONFIG_BETA,
        _KCONFIG_GAMMA,
    ]
    assert all("$(X)" not in chunk.text for chunk in chunks)


def test_h2_config_block_is_cut_at_1500_characters():
    payload = "x" * 3988
    full = "config LONG\n" + payload
    assert len(full) == 4000
    chunks = _chunks("subsys/Kconfig", full + "\n")
    body = chunks[1].text.split("\n", 1)[1]

    assert chunks[1].kind == "config"
    assert len(body) == 1500
    assert body == full[:1500]


def test_h3_rst_card_sections_and_headingless_text():
    content = "Intro\n=====\n\nhello\n\nEmpty\n=====\n"
    chunks = _chunks("doc/index.rst", content)

    assert [chunk.kind for chunk in chunks] == ["file", "section", "section"]
    assert chunks[0].text == (
        "file doc/index.rst\n"
        "Intro\n"
        "sections: Intro | Empty"
    )
    assert chunks[1].text == "section Intro in doc/index.rst\nhello"
    assert chunks[1].name == "Intro"
    assert chunks[2].text == "section Empty in doc/index.rst\n"
    assert chunks[2].name == "Empty"
    assert chunks[2].text.split("\n", 1)[1] == ""

    notes = _chunks("doc/notes.rst", "plain notes\nsecond line\n")
    assert len(notes) == 1
    assert notes[0].kind == "text"
    assert notes[0].text == "rst doc/notes.rst\nplain notes\nsecond line"


def test_h4_text_piece_labels_limit_and_line_clip():
    line = "a" * 750
    halves = _chunks("src/CMakeLists.txt", f"{line}\n{line}\n")
    assert [chunk.text for chunk in halves] == [
        f"cmake src/CMakeLists.txt (1/2)\n{line}",
        f"cmake src/CMakeLists.txt (2/2)\n{line}",
    ]

    many = _chunks("src/CMakeLists.txt", ("b" * 1500 + "\n") * 20)
    assert len(many) == 8
    assert many[0].text.startswith("cmake src/CMakeLists.txt (1/8)\n")
    assert many[-1].text.startswith("cmake src/CMakeLists.txt (8/8)\n")

    clipped = _chunks("src/CMakeLists.txt", "c" * 5000 + "\n")
    assert len(clipped) == 1
    assert clipped[0].text == "cmake src/CMakeLists.txt\n" + ("c" * 1500)

    assert _chunks("CMakeLists.txt", "") == ()
    assert _chunks("prj.conf", "") == ()


def test_h5_keys_use_chunk_v1():
    chunks = _chunks("kernel/Kconfig", _KCONFIG)
    chunks += _chunks("src/CMakeLists.txt", "project(app)\n")

    assert CHUNK_FORMAT == "chunk-v1"
    for chunk in chunks:
        digest = hashlib.sha256(("chunk-v1\n" + chunk.text).encode()).hexdigest()
        assert chunk.key == digest


def test_h7_mixed_project_keys_are_unique_and_assembly_is_skipped():
    files = [
        ProjectFile(path=Path("src/main.c"), content="void main(void) { return; }\n"),
        ProjectFile(path=Path("Kconfig"), content="config FOO\n\tbool\n"),
        ProjectFile(path=Path("doc.rst"), content="Title\n=====\n\nbody\n"),
        ProjectFile(path=Path("CMakeLists.txt"), content="project(app)\n"),
        ProjectFile(path=Path("startup.S"), content="nop\n"),
    ]
    chunks = chunk_index(build_index(files), files)

    assert len({chunk.key for chunk in chunks}) == len(chunks)
    assert len(unique_texts(chunks)) == len(chunks)
    assert all(chunk.path.name != "startup.S" for chunk in chunks)
    assert _chunks("startup.S", "nop\n") == ()


def test_s1_other_kinds_line_for_text_chunks_only(tmp_path, capsys):
    source = tmp_path / "src"
    source.mkdir()
    (source / "a.c").write_text("void f(void) { return; }\n", encoding="utf-8")
    (source / "b.h").write_text("void g(void);\n", encoding="utf-8")

    main([str(tmp_path)])
    c_only = capsys.readouterr().out
    assert "other kinds:" not in c_only
    assert len(c_only.splitlines()) == 6

    (tmp_path / "Kconfig").write_text(
        "config FOO\n\tbool\nconfig BAR\n\tbool\n",
        encoding="utf-8",
    )
    (tmp_path / "doc.rst").write_text("Title\n=====\n\nbody\n", encoding="utf-8")
    (tmp_path / "CMakeLists.txt").write_text("project(app)\n", encoding="utf-8")
    main([str(tmp_path)])
    mixed = capsys.readouterr().out.splitlines()
    assert mixed[2] == "other kinds: config 2, section 1, text 1"

    only = tmp_path / "only"
    only.mkdir()
    (only / "Kconfig").write_text("config FOO\n\tbool\n", encoding="utf-8")
    main([str(only)])
    kinds = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("other kinds:")
    ]
    assert kinds == ["other kinds: config 1"]
