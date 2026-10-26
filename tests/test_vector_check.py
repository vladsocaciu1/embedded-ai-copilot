import re
from pathlib import Path

from app.chunker import chunk_index, unique_texts
from app.code_index import build_index
from app.config import DEFAULT_EXCLUDE_DIRS
from app.project_loader import load_project_files
from scripts.embed_check import select_scope
from scripts.vector_check import main
from tests.fake_embedder import FakeEmbedder

_ENSURE = re.compile(
    r"ensure live (\d+) reused (\d+) embedded (\d+) stale (\d+) "
    r"compacted (yes|no) write_failed (yes|no) in (\d+\.\d) s"
)
_SCOPE = re.compile(r"scope (\S+) files (\d+) chunks unique (\d+)")
_INDEX = re.compile(r"index (\d+) rows dims (\d+) dir (.+) size (\d+\.\d) MB")


def _project(root: Path) -> None:
    a = root / "a"
    b = root / "b"
    a.mkdir(parents=True)
    b.mkdir()
    (a / "one.c").write_text(
        "void alpha(void) { int marker = 1; }\n",
        encoding="utf-8",
    )
    (a / "two.c").write_text(
        "void beta(void) { return; }\n",
        encoding="utf-8",
    )
    (b / "three.c").write_text(
        "void gamma(void) { return; }\n",
        encoding="utf-8",
    )


def _unique(root: Path, scopes: list[str]) -> int:
    files = load_project_files(root, DEFAULT_EXCLUDE_DIRS)
    chunks = chunk_index(build_index(files), files)
    return len(unique_texts(select_scope(chunks, scopes)))


def _lines(capsys) -> list[str]:
    return capsys.readouterr().out.splitlines()


# C1
def test_c1_cold_then_warm(tmp_path, capsys):
    root = tmp_path / "proj"
    cache = tmp_path / "cache"
    _project(root)
    unique = _unique(root, ["."])
    fake = FakeEmbedder()

    assert main(
        [str(root), "--scope", ".", "--cache-dir", str(cache), "--model", "fake-bow"],
        embedder=fake,
    ) == 0
    lines = _lines(capsys)
    assert len(lines) == 5
    scope = _SCOPE.fullmatch(lines[0])
    ensure = _ENSURE.fullmatch(lines[1])
    index = _INDEX.fullmatch(lines[2])
    assert scope is not None and ensure is not None and index is not None
    assert scope.group(1) == "."
    assert int(scope.group(3)) == unique
    assert int(ensure.group(1)) == unique
    assert int(ensure.group(3)) == unique
    assert lines[3] == "lookup ok"
    assert re.fullmatch(r"rss (\d+|n/a) MB peak", lines[4])
    assert int(index.group(1)) == unique
    assert int(index.group(2)) == fake.dims

    warm = FakeEmbedder()
    assert main(
        [str(root), "--scope", ".", "--cache-dir", str(cache), "--model", "fake-bow"],
        embedder=warm,
    ) == 0
    warm_lines = _lines(capsys)
    warm_ensure = _ENSURE.fullmatch(warm_lines[1])
    assert warm_ensure is not None
    assert int(warm_ensure.group(3)) == 0
    assert warm_lines[3] == "lookup ok"
    warm_index = _INDEX.fullmatch(warm_lines[2])
    assert warm_index is not None
    assert int(warm_index.group(1)) == unique
    assert int(warm_index.group(2)) == fake.dims


# C2
def test_c2_bad_arguments_do_not_embed(tmp_path, capsys):
    root = tmp_path / "proj"
    cache = tmp_path / "cache"
    _project(root)

    missing_scope = FakeEmbedder()
    assert main([str(root), "--cache-dir", str(cache)], embedder=missing_scope) == 2
    assert missing_scope.document_calls == []
    assert missing_scope.query_calls == []

    unknown = FakeEmbedder()
    assert (
        main(
            [str(root), "--scope", "missing", "--cache-dir", str(cache)],
            embedder=unknown,
        )
        == 2
    )
    assert unknown.document_calls == []
    assert unknown.query_calls == []
    assert "missing" in capsys.readouterr().err

    limited = FakeEmbedder()
    assert (
        main(
            [
                str(root),
                "--scope",
                "a",
                "--cache-dir",
                str(cache),
                "--max-chunks",
                "1",
            ],
            embedder=limited,
        )
        == 2
    )
    assert limited.document_calls == []
    assert limited.query_calls == []
    assert "refusing to embed" in capsys.readouterr().out


# C3
def test_c3_edit_embeds_only_new_chunks(tmp_path, capsys):
    root = tmp_path / "proj"
    cache = tmp_path / "cache"
    _project(root)
    argv = [str(root), "--scope", ".", "--cache-dir", str(cache), "--model", "fake-bow"]
    assert main(argv, embedder=FakeEmbedder()) == 0
    capsys.readouterr()
    assert main(argv, embedder=FakeEmbedder()) == 0
    capsys.readouterr()

    source = root / "a" / "one.c"
    source.write_text(
        source.read_text(encoding="utf-8") + "\nvoid extra(void) { return; }\n",
        encoding="utf-8",
    )
    fake = FakeEmbedder()
    assert main(argv, embedder=fake) == 0
    lines = _lines(capsys)
    scope = _SCOPE.fullmatch(lines[0])
    ensure = _ENSURE.fullmatch(lines[1])
    assert scope is not None and ensure is not None
    unique = int(scope.group(3))
    embedded = int(ensure.group(3))
    assert 0 < embedded < unique
    assert fake.document_calls != []
