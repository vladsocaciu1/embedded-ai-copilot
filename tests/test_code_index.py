import logging
import textwrap
from pathlib import Path

from app.code_index import build_index, index_file
from app.text_files import index_text
from app.models import (
    Call,
    FileSymbols,
    Include,
    ProjectFile,
    ProjectIndex,
    Symbol,
    SymbolKind,
)
from app.project_loader import load_project_files


def _find(file_symbols: FileSymbols, name: str, kind: SymbolKind) -> Symbol:
    found = [
        symbol
        for symbol in file_symbols.symbols
        if symbol.name == name and symbol.kind == kind
    ]
    assert len(found) == 1
    return found[0]


def _index(path: str, content: str) -> FileSymbols:
    return index_file(ProjectFile(path=Path(path), content=textwrap.dedent(content)))


def test_i1_c_function_lines_and_grammar():
    file_symbols = _index(
        "src/x.c",
        """\
        int answer(void)
        {
            return 42;
        }
        """,
    )

    assert file_symbols.grammar == "c"
    symbol = _find(file_symbols, "answer", SymbolKind.FUNCTION)
    assert symbol.start_line == 1
    assert symbol.end_line == 4
    assert symbol.is_static is False


def test_i2_header_prototype_uses_cpp_grammar():
    file_symbols = _index("inc/can.h", "void Can_SendFrame(const Can_Frame *frame);\n")

    assert file_symbols.grammar == "cpp"
    _find(file_symbols, "Can_SendFrame", SymbolKind.PROTOTYPE)


def test_i3_static_function_and_variable():
    file_symbols = _index(
        "src/x.c",
        """\
        static void helper(void)
        {
        }

        static volatile uint32_t tick;
        """,
    )

    assert _find(file_symbols, "helper", SymbolKind.FUNCTION).is_static is True
    assert _find(file_symbols, "tick", SymbolKind.VARIABLE).is_static is True


def test_i4_static_inline_pointer_return_is_function():
    file_symbols = _index("src/x.c", "static inline uint8_t *get_buf(void){}\n")

    _find(file_symbols, "get_buf", SymbolKind.FUNCTION)


def test_i5_weak_function_is_found():
    file_symbols = _index("src/x.c", "__weak void HAL_MspInit(void){}\n")

    _find(file_symbols, "HAL_MspInit", SymbolKind.FUNCTION)


def test_i6_both_ifdef_branches_are_indexed():
    file_symbols = _index(
        "src/x.c",
        """\
        #ifdef FEATURE
        void from_ifdef(void) {}
        #else
        void from_else(void) {}
        #endif
        """,
    )

    _find(file_symbols, "from_ifdef", SymbolKind.FUNCTION)
    _find(file_symbols, "from_else", SymbolKind.FUNCTION)


def test_i7_macros_skip_include_guard_only():
    file_symbols = _index(
        "inc/edge.h",
        """\
        #ifndef EDGE_H
        #define EDGE_H

        #define BUF_SIZE 64
        #define MAX(a,b) ((a) > (b) ? (a) : (b))
        #define FEATURE_X

        #endif
        """,
    )

    names = {
        symbol.name
        for symbol in file_symbols.symbols
        if symbol.kind == SymbolKind.MACRO
    }
    assert names == {"BUF_SIZE", "MAX", "FEATURE_X"}


def test_i8_struct_union_enum_and_typedefs():
    file_symbols = _index(
        "src/types.c",
        """\
        typedef struct
        {
            uint32_t id;
        } Can_Frame;

        typedef struct Node
        {
            int x;
        } Node_t;

        union U {};

        enum Color {};

        typedef void (*handler_t)(int);
        """,
    )

    kinds = {(symbol.name, symbol.kind) for symbol in file_symbols.symbols}
    assert ("Can_Frame", SymbolKind.TYPEDEF) in kinds
    assert ("Can_Frame", SymbolKind.STRUCT) not in kinds
    assert ("Node", SymbolKind.STRUCT) in kinds
    assert ("Node_t", SymbolKind.TYPEDEF) in kinds
    assert ("U", SymbolKind.UNION) in kinds
    assert ("Color", SymbolKind.ENUM) in kinds
    assert ("handler_t", SymbolKind.TYPEDEF) in kinds


def test_i9_file_scope_variables_skip_extern_and_locals():
    file_symbols = _index(
        "src/x.c",
        """\
        uint8_t buf[64], *bufp = buf;
        extern int ext_var;
        static void (*callback)(int);

        void use_local(void)
        {
            int local_value = 1;
        }
        """,
    )

    names = {
        symbol.name
        for symbol in file_symbols.symbols
        if symbol.kind == SymbolKind.VARIABLE
    }
    assert names == {"buf", "bufp", "callback"}


def test_i10_includes():
    file_symbols = _index(
        "src/x.c",
        """\
        #include "hal/can.h"
        #include <stdint.h>
        #include CONFIG_HEADER
        """,
    )

    assert file_symbols.includes == (
        Include(target="hal/can.h", is_system=False, line=1),
        Include(target="stdint.h", is_system=True, line=2),
    )


def test_i11_errors_are_tolerated():
    file_symbols = _index(
        "src/x.c",
        """\
        int broken(int x){ return x + ; }
        int after_broken(void){return 0;}
        """,
    )

    assert file_symbols.has_errors is True
    _find(file_symbols, "after_broken", SymbolKind.FUNCTION)

    garbage = index_file(ProjectFile(Path("src/junk.c"), "}{)(#if\n" * 50))
    assert garbage.grammar == "c"


def test_i12_lines_past_256_do_not_crash():
    c_lines = [f"int v{number};" for number in range(400)]
    c_lines.extend(
        [
            "void past_the_point(void)",
            "{",
            "    return;",
            "}",
        ]
    )
    c_symbols = index_file(
        ProjectFile(Path("src/long.c"), "\n".join(c_lines) + "\n")
    )
    variable = _find(c_symbols, "v300", SymbolKind.VARIABLE)
    assert variable.start_line == 301
    assert variable.end_line == 301
    function = _find(c_symbols, "past_the_point", SymbolKind.FUNCTION)
    assert function.start_line == 401
    assert function.end_line == 404

    h_lines = ["typedef int pad_t;" for _ in range(300)]
    h_lines.append("void late_proto(void);")
    h_symbols = index_file(
        ProjectFile(Path("inc/long.h"), "\n".join(h_lines) + "\n")
    )
    assert h_symbols.grammar == "cpp"
    prototype = _find(h_symbols, "late_proto", SymbolKind.PROTOTYPE)
    assert prototype.start_line == 301
    assert prototype.end_line == 301


def test_i13_deep_ifdef_nesting_finds_the_function():
    depth = 1500
    lines = [f"#ifdef N{index}" for index in range(depth)]
    lines.append("void buried(void) {}")
    lines.extend("#endif" for _ in range(depth))
    file_symbols = index_file(
        ProjectFile(Path("src/deep.c"), "\n".join(lines) + "\n")
    )

    _find(file_symbols, "buried", SymbolKind.FUNCTION)


def test_x1_nested_namespace_class_and_methods():
    file_symbols = _index(
        "inc/controller.hpp",
        """\
        namespace drivers {
        namespace can {
        class Controller {
        public:
            void send();
            int id() { return 1; }
            static Controller &instance();
        };
        }
        }
        """,
    )

    _find(file_symbols, "drivers", SymbolKind.NAMESPACE)
    _find(file_symbols, "drivers::can", SymbolKind.NAMESPACE)
    _find(file_symbols, "drivers::can::Controller", SymbolKind.CLASS)
    _find(file_symbols, "drivers::can::Controller::send", SymbolKind.PROTOTYPE)
    _find(file_symbols, "drivers::can::Controller::id", SymbolKind.FUNCTION)
    instance = _find(
        file_symbols, "drivers::can::Controller::instance", SymbolKind.PROTOTYPE
    )
    assert instance.is_static is True


def test_x2_out_of_line_methods():
    file_symbols = _index(
        "src/controller.cpp",
        """\
        namespace drivers::can {
        Controller::Controller() {}
        Controller::~Controller() {}
        bool Controller::send(int x) { return true; }
        Controller &Controller::instance() { static Controller c; return c; }
        }
        """,
    )

    _find(file_symbols, "drivers::can::Controller::Controller", SymbolKind.FUNCTION)
    _find(file_symbols, "drivers::can::Controller::~Controller", SymbolKind.FUNCTION)
    _find(file_symbols, "drivers::can::Controller::send", SymbolKind.FUNCTION)
    _find(file_symbols, "drivers::can::Controller::instance", SymbolKind.FUNCTION)


def test_x3_function_template_in_namespace():
    file_symbols = _index(
        "inc/clamp.hpp",
        """\
        namespace ns {
        template<typename T>
        T clamp(T value, T low, T high) { return value; }
        }
        """,
    )

    _find(file_symbols, "ns::clamp", SymbolKind.FUNCTION)


def test_x4_extern_c_and_anonymous_namespace_stay_unqualified():
    file_symbols = _index(
        "src/entry.cpp",
        """\
        extern "C" void C_Entry(void) {}

        extern "C" {
        void C_Other(void) {}
        }

        namespace {
        void hidden(void) {}
        }
        """,
    )

    _find(file_symbols, "C_Entry", SymbolKind.FUNCTION)
    _find(file_symbols, "C_Other", SymbolKind.FUNCTION)
    _find(file_symbols, "hidden", SymbolKind.FUNCTION)
    assert not any("::" in symbol.name for symbol in file_symbols.symbols)


def test_x5_c_header_parsed_as_cpp_class():
    file_symbols = _index("inc/foo.h", "class Foo { public: void bar(); };\n")

    assert file_symbols.grammar == "cpp"
    _find(file_symbols, "Foo", SymbolKind.CLASS)
    _find(file_symbols, "Foo::bar", SymbolKind.PROTOTYPE)


def test_g1_grammar_follows_suffix():
    source = "void f(void) {}\n"
    for suffix in (".c", ".C"):
        file_symbols = index_file(ProjectFile(Path(f"file{suffix}"), source))
        assert file_symbols.grammar == "c"
    for suffix in (".h", ".hpp", ".cpp", ".inl"):
        file_symbols = index_file(ProjectFile(Path(f"file{suffix}"), source))
        assert file_symbols.grammar == "cpp"
        _find(file_symbols, "f", SymbolKind.FUNCTION)
    for suffix in (".s", ".S", ".ld"):
        file_symbols = index_file(ProjectFile(Path(f"file{suffix}"), source))
        assert file_symbols.grammar is None
        assert file_symbols.has_errors is False
        assert file_symbols.symbols == ()
        assert file_symbols.includes == ()


def test_p1_build_index_preserves_input_order_and_paths():
    project_files = [
        ProjectFile(Path("b.c"), "void b(void) {}\n"),
        ProjectFile(Path("a.h"), "void a(void);\n"),
        ProjectFile(Path("c.s"), "nop\n"),
    ]

    index = build_index(project_files)

    assert [file_symbols.path for file_symbols in index.files] == [
        project_file.path for project_file in project_files
    ]


def test_p2_one_info_summary_and_debug_for_the_error_file(caplog):
    caplog.set_level(logging.DEBUG)
    project_files = [
        ProjectFile(Path("src/good.c"), "void good(void) {}\n/* MARKER_GOOD_7f3a */\n"),
        ProjectFile(
            Path("src/bad.c"),
            "int broken(int x){ return x + ; }\n/* MARKER_BAD_9c1e */\n",
        ),
    ]

    build_index(project_files)

    info_records = [
        record
        for record in caplog.records
        if record.levelno == logging.INFO and record.name == "app.code_index"
    ]
    assert len(info_records) == 1
    assert "2" in info_records[0].getMessage()
    assert any(
        record.levelno == logging.DEBUG
        and record.name == "app.code_index"
        and "src/bad.c" in record.getMessage()
        for record in caplog.records
    )
    assert "MARKER_GOOD_7f3a" not in caplog.text
    assert "MARKER_BAD_9c1e" not in caplog.text


def test_p3_empty_input():
    assert build_index([]) == ProjectIndex(files=())


def test_p4_sample_project():
    repo = Path(__file__).resolve().parent.parent
    index = build_index(load_project_files(repo / "sample_project"))
    by_path = {
        file_symbols.path.as_posix(): file_symbols for file_symbols in index.files
    }

    _find(by_path["src/can.c"], "Can_SendFrame", SymbolKind.FUNCTION)
    initialized = _find(by_path["src/can.c"], "can_initialized", SymbolKind.VARIABLE)
    assert initialized.is_static is True
    _find(by_path["inc/can.h"], "Can_SendFrame", SymbolKind.PROTOTYPE)
    _find(by_path["inc/can.h"], "Can_Frame", SymbolKind.TYPEDEF)
    _find(by_path["src/main.c"], "main", SymbolKind.FUNCTION)
    assert any(
        include.target == "can.h" and include.is_system is False
        for include in by_path["src/app.c"].includes
    )
    assert all(not file_symbols.has_errors for file_symbols in index.files)


def test_c1_calls_inside_a_function_body():
    file_symbols = _index(
        "src/x.c",
        """\
        void worker(void)
        {
            g(1);
            X = y(z(5));
            if (HAL_GetTick() > 1) {}
        }
        """,
    )

    assert file_symbols.calls == (
        Call("worker", "g", 3, False),
        Call("worker", "y", 4, False),
        Call("worker", "z", 4, False),
        Call("worker", "HAL_GetTick", 5, False),
    )


def test_c2_member_calls_recorded_indirect_calls_skipped():
    file_symbols = _index(
        "src/x.c",
        """\
        void worker(void)
        {
            h->cb(2);
            s.m();
            (*fp)(3);
            tbl[i]();
        }
        """,
    )

    assert file_symbols.calls == (
        Call("worker", "cb", 3, True),
        Call("worker", "m", 4, True),
    )


def test_c3_cpp_qualified_calls_lambda_and_inline_method():
    file_symbols = _index(
        "src/controller.cpp",
        """\
        namespace drivers {
        class Controller {
        public:
            void ping() { helper(); }
        };
        void Controller::send() {
            ns::k();
            t<int>(4);
            auto callback = []() { inner(); };
        }
        }
        """,
    )

    assert file_symbols.calls == (
        Call("drivers::Controller::ping", "helper", 4, False),
        Call("drivers::Controller::send", "ns::k", 7, False),
        Call("drivers::Controller::send", "t", 8, False),
        Call("drivers::Controller::send", "inner", 9, False),
    )


def test_c4_function_like_macro_is_a_call():
    file_symbols = _index(
        "src/x.c",
        """\
        void worker(void)
        {
            __HAL_LOCK(h);
        }
        """,
    )

    assert file_symbols.calls == (Call("worker", "__HAL_LOCK", 3, False),)


def test_c5_file_scope_initializer_and_prototype_have_no_calls():
    file_symbols = _index(
        "src/x.c",
        """\
        int x = f();
        void proto(void);
        """,
    )

    assert file_symbols.calls == ()


def test_c6_weak_function_and_calls_after_a_parse_error():
    file_symbols = _index(
        "src/x.c",
        """\
        __weak void HAL_MspInit(void) { Gpio_Init(); }
        int broken(int x) { return x + ; }
        void next(void) { kept(); }
        """,
    )

    assert file_symbols.calls == (
        Call("HAL_MspInit", "Gpio_Init", 1, False),
        Call("next", "kept", 3, False),
    )


def test_c7_call_line_past_300():
    lines = ["int pad;"] * 310
    lines.extend(
        [
            "void late(void)",
            "{",
            "    far_call();",
            "}",
        ]
    )
    file_symbols = index_file(
        ProjectFile(Path("src/long.c"), "\n".join(lines) + "\n")
    )

    assert file_symbols.calls == (Call("late", "far_call", 313, False),)


def test_i1_text_files_are_unparsed_and_carry_text_symbols():
    samples = {
        "Kconfig": ("kconfig", "config FOO\n\tbool\nconfig BAR\n\tbool\n"),
        "doc.rst": ("rst", "Title\n=====\n\nbody\n"),
        "board.overlay": ("devicetree", "led0: led_0 {\n};\nalias = &led0;\n"),
        "prj.conf": ("conf", "CONFIG_FOO=y\n"),
    }
    for name, (kind, content) in samples.items():
        project_file = ProjectFile(Path(name), content)
        file_symbols = index_file(project_file)
        assert file_symbols.grammar is None
        assert file_symbols.has_errors is False
        assert file_symbols.symbols == index_text(project_file, kind)
        assert file_symbols.includes == ()
        assert file_symbols.calls == ()

    unknown = index_file(ProjectFile(Path("README.md"), "hello\n"))
    assert unknown.grammar is None
    assert unknown.symbols == ()
