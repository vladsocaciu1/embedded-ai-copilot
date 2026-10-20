from pathlib import Path

import pytest

from app.chunker import chunk_index, unique_texts
from app.code_index import build_index
from app.config import DEFAULT_EXCLUDE_DIRS
from app.models import Chunk
from app.project_loader import load_project_files
from scripts.embed_check import main, select_scope
from tests.fake_embedder import FakeEmbedder

MARKER = "UNIQUE_MARKER_zz9"


def _chunk(path: str) -> Chunk:
    return Chunk(
        path=Path(path),
        kind="file",
        name="f",
        start_line=1,
        end_line=1,
        text="t",
        key=path,
    )


def _project(root: Path) -> None:
    a = root / "a"
    b = root / "b"
    a.mkdir()
    b.mkdir()
    (a / "one.c").write_text(
        f"void alpha(void) {{ int {MARKER} = 1; }}\n",
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


def _scoped_unique(root: Path, scopes: list[str]) -> dict[str, str]:
    files = load_project_files(root, DEFAULT_EXCLUDE_DIRS)
    chunks = chunk_index(build_index(files), files)
    return unique_texts(select_scope(chunks, scopes))


def test_k1_scope_embeds_only_that_subdir(tmp_path, capsys, monkeypatch):
    _project(tmp_path)
    monkeypatch.setenv("EMBEDDING_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr("app.config.load_dotenv", lambda *a, **k: False)
    fake = FakeEmbedder()

    main([str(tmp_path), "--scope", "a"], embedder=fake)

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 7
    assert lines[0].startswith("files ")
    assert lines[1].startswith("scope a files ")
    assert "chunks " in lines[1]
    assert "unique " in lines[1]
    assert lines[2].startswith("model ")
    assert lines[2].endswith("before load)")
    assert lines[3].startswith("model load ")
    assert lines[4].startswith("model files ")
    assert lines[5].startswith("embed ")
    assert lines[6].startswith("peak_rss_mb ")
    expected = _scoped_unique(tmp_path, ["a"])
    assert fake.document_calls == [list(expected.values())]
    assert fake.query_calls == ["warm up"]
    assert MARKER not in "\n".join(lines)
    assert "void gamma" not in "\n".join(lines)


def test_k2_scope_and_max_chunks_errors(tmp_path, capsys, monkeypatch):
    _project(tmp_path)
    monkeypatch.setattr("app.config.load_dotenv", lambda *a, **k: False)
    fake = FakeEmbedder()

    with pytest.raises(SystemExit) as missing:
        main([str(tmp_path)], embedder=fake)
    assert missing.value.code == 2

    with pytest.raises(SystemExit) as unknown:
        main([str(tmp_path), "--scope", "missing"], embedder=fake)
    assert unknown.value.code == 2
    captured = capsys.readouterr()
    assert "missing" in captured.err

    fake = FakeEmbedder()
    with pytest.raises(SystemExit) as refused:
        main([str(tmp_path), "--scope", "a", "--max-chunks", "1"], embedder=fake)
    assert refused.value.code == 2
    out = capsys.readouterr().out
    assert "refusing to embed" in out
    assert "narrow --scope" in out
    assert fake.document_calls == []
    assert fake.query_calls == []


def test_k3_select_scope_matches_whole_parts():
    chunks = (
        _chunk("a/x.c"),
        _chunk("ab/x.c"),
        _chunk("b/y.c"),
        _chunk("a/b/z.c"),
    )

    selected = select_scope(chunks, ["a"])
    assert [chunk.path.as_posix() for chunk in selected] == ["a/x.c", "a/b/z.c"]

    both = select_scope(chunks, ["a", "b"])
    assert [chunk.path.as_posix() for chunk in both] == ["a/x.c", "b/y.c", "a/b/z.c"]

    with pytest.raises(ValueError, match="a"):
        select_scope((_chunk("ab/x.c"),), ["a"])
    with pytest.raises(ValueError, match="nope"):
        select_scope(chunks, ["a", "nope"])


def test_k4_no_args_prints_usage(capsys):
    main([])

    out = capsys.readouterr().out
    assert out.startswith("usage: embed_check.py ")
    assert "--scope" in out
    assert MARKER not in out
