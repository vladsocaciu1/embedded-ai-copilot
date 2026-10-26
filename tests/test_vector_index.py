import hashlib
import io
import json
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from app.chunker import CHUNK_FORMAT
from app.embedder import EmbedderUnavailableError
from app.models import Chunk
from app.semantic_search import embed_chunks, semantic_rank
import app.vector_index as vector_index
from app.vector_index import VectorIndex, model_slug, vector_dir
from tests.fake_embedder import FakeEmbedder

# Object-dtype npy fixture, stored as bytes so this test never pickles.
# The loader must refuse it.
_OBJECT_NPY = bytes.fromhex(
    "934e554d5059010076007b276465736372273a20277c4f272c2027666f727472616e5f6f72646572273a2046"
    "616c73652c20277368617065273a2028312c292c207d20202020202020202020202020202020202020202020"
    "20202020202020202020202020202020202020202020202020202020202020202020202020202020200a8004"
    "9595000000000000008c166e756d70792e5f636f72652e6d756c74696172726179948c0c5f7265636f6e7374"
    "727563749493948c056e756d7079948c076e6461727261799493944b0085944301629487945294284b014b01"
    "859468038c0564747970659493948c024f3894898887945294284b038c017c944e4e4e4affffffff4affffffff"
    "4b3f749462895d947d948c046576696c944b0173617494622e"
)


def _chunk(text: str, path: str = "a.c", name: str = "f") -> Chunk:
    key = hashlib.sha256((CHUNK_FORMAT + "\n" + text).encode("utf-8")).hexdigest()
    return Chunk(
        path=Path(path),
        kind="function",
        name=name,
        start_line=1,
        end_line=2,
        text=text,
        key=key,
    )


def _warnings(caplog) -> list:
    return [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING and record.name == "app.vector_index"
    ]


def _files(directory: Path) -> dict[str, bytes]:
    return {
        name: (directory / name).read_bytes()
        for name in ("vectors.npy", "keys.json", "meta.json")
    }


def _mtimes(directory: Path) -> dict[str, int]:
    return {
        name: (directory / name).stat().st_mtime_ns
        for name in ("vectors.npy", "keys.json", "meta.json")
    }


def _load_matrix(directory: Path) -> np.ndarray:
    return np.load(directory / "vectors.npy", allow_pickle=False)


def _rewrite_meta(directory: Path, **updates) -> None:
    path = directory / "meta.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta.update(updates)
    path.write_text(json.dumps(meta), encoding="utf-8")


def _set_vectors(directory: Path, matrix: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, matrix, allow_pickle=False)
    raw = buffer.getvalue()
    (directory / "vectors.npy").write_bytes(raw)
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    meta["vectors_sha256"] = hashlib.sha256(raw).hexdigest()
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def _set_keys(directory: Path, keys: list) -> None:
    raw = json.dumps(keys).encode("utf-8")
    (directory / "keys.json").write_bytes(raw)
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    meta["keys_sha256"] = hashlib.sha256(raw).hexdigest()
    meta["count"] = len(keys)
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


# V1
def test_v1_cold_ensure_writes_three_files(tmp_path):
    cache = tmp_path / "cache"
    model = "fake-bow"
    chunks = [
        _chunk("alpha gpio pin"),
        _chunk("beta uart clock", path="b.c"),
        _chunk("alpha gpio pin", path="c.c"),
    ]
    fake = FakeEmbedder()
    index = VectorIndex(cache, model)

    result = index.ensure(chunks, fake)

    assert fake.document_calls == [[chunks[0].text, chunks[1].text]]
    assert result.live == 2
    assert result.reused == 0
    assert result.embedded == 2
    assert result.stale == 0
    assert result.compacted is False
    assert result.write_failed is False
    directory = vector_dir(cache, model)
    assert index.directory == directory
    assert sorted(path.name for path in directory.iterdir()) == [
        "keys.json",
        "meta.json",
        "vectors.npy",
    ]
    stored = _files(directory)
    meta = json.loads(stored["meta.json"])
    assert meta["format"] == "vectors-v1"
    assert meta["model"] == model
    assert meta["dims"] == 64
    assert meta["chunk_format"] == CHUNK_FORMAT
    assert meta["count"] == 2
    assert meta["keys_sha256"] == hashlib.sha256(stored["keys.json"]).hexdigest()
    assert meta["vectors_sha256"] == hashlib.sha256(stored["vectors.npy"]).hexdigest()
    matrix = np.load(io.BytesIO(stored["vectors.npy"]), allow_pickle=False)
    assert matrix.dtype == np.float32
    assert matrix.shape == (2, 64)


# V2
def test_v2_warm_lookup_matches_embed_chunks(tmp_path):
    cache = tmp_path / "cache"
    model = "fake-bow"
    chunks = [
        _chunk("button debounce handler press", path="a.c"),
        _chunk("uart baud clock register", path="b.c"),
    ]
    VectorIndex(cache, model).ensure(chunks, FakeEmbedder())
    warm = VectorIndex(cache, model)
    fake = FakeEmbedder()

    result = warm.ensure(chunks, fake)
    looked = warm.lookup(chunks)
    fresh = embed_chunks(chunks, FakeEmbedder())

    assert result.reused == result.live
    assert result.embedded == 0
    assert fake.document_calls == []
    assert looked is not None
    assert looked.keys == fresh.keys
    assert looked.chunks == fresh.chunks
    assert np.allclose(looked.matrix, fresh.matrix)
    ranker = FakeEmbedder()
    for question in ("button debounce", "uart baud"):
        assert semantic_rank(question, looked, ranker) == semantic_rank(
            question, fresh, ranker
        )


# V3
def test_v3_incremental_edit_and_revert(tmp_path):
    cache = tmp_path / "cache"
    model = "fake-bow"
    original = _chunk("beta uart clock", path="b.c")
    chunks = [_chunk("alpha gpio pin", path="a.c"), original]
    index = VectorIndex(cache, model)
    index.ensure(chunks, FakeEmbedder())
    changed = _chunk("beta uart clock changed", path="b.c")
    edited = [chunks[0], changed]
    fake = FakeEmbedder()

    edited_result = index.ensure(edited, fake)
    looked = index.lookup(edited)

    assert fake.document_calls == [[changed.text]]
    assert edited_result.stale == 1
    assert looked is not None
    assert looked.matrix.shape[0] == edited_result.live
    assert original.key not in looked.keys

    fake.document_calls.clear()
    reverted = index.ensure(chunks, fake)
    assert fake.document_calls == []
    assert reverted.stale == 1
    assert index.lookup(chunks) is not None


# V4
def test_v4_compaction_boundary_and_empty(tmp_path):
    cache = tmp_path / "cache"
    model = "fake-bow"
    rows = [_chunk(f"row {index} token", path=f"{index}.c") for index in range(5)]
    index = VectorIndex(cache, model)
    index.ensure(rows, FakeEmbedder())
    kept = [rows[1], rows[4], rows[2]]

    compacted = index.ensure(kept, FakeEmbedder())

    assert compacted.compacted is True
    assert compacted.stale == 0
    stored_keys = json.loads((index.directory / "keys.json").read_text(encoding="utf-8"))
    assert stored_keys == [rows[1].key, rows[2].key, rows[4].key]
    assert _load_matrix(index.directory).shape[0] == compacted.live == 3

    boundary_cache = tmp_path / "boundary"
    six = [_chunk(f"keep {index} token", path=f"k{index}.c") for index in range(6)]
    boundary = VectorIndex(boundary_cache, model)
    boundary.ensure(six, FakeEmbedder())
    half = boundary.ensure(six[:4], FakeEmbedder())
    assert half.compacted is False
    assert half.stale == 2
    assert len(json.loads((boundary.directory / "keys.json").read_text(encoding="utf-8"))) == 6
    assert _load_matrix(boundary.directory).shape[0] == 6

    before = _files(index.directory)
    stamps = _mtimes(index.directory)
    quiet = FakeEmbedder()
    emptied = index.ensure([], quiet)
    assert quiet.document_calls == []
    assert emptied.compacted is False
    assert _files(index.directory) == before
    assert _mtimes(index.directory) == stamps


# V5
def test_v5_other_model_uses_another_directory(tmp_path):
    cache = tmp_path / "cache"
    chunks = [_chunk("alpha gpio pin"), _chunk("beta uart clock", path="b.c")]
    first = VectorIndex(cache, "model-a")
    first.ensure(chunks, FakeEmbedder())
    before = _files(first.directory)
    second = VectorIndex(cache, "model-b")
    fake = FakeEmbedder()

    result = second.ensure(chunks, fake)

    assert second.directory != first.directory
    assert result.reused == 0
    assert result.embedded == 2
    assert fake.document_calls == [[chunks[0].text, chunks[1].text]]
    assert _files(first.directory) == before


@pytest.mark.parametrize(
    "change",
    ["model", "format", "chunk_format", "dims", "constructor"],
)
def test_v5_incompatible_meta_rebuilds(tmp_path, caplog, change):
    cache = tmp_path / "cache"
    model = "fake-bow"
    chunks = [_chunk("alpha gpio pin"), _chunk("beta uart clock", path="b.c")]
    VectorIndex(cache, model).ensure(chunks, FakeEmbedder())
    directory = vector_dir(cache, model)
    chunk_format = CHUNK_FORMAT
    if change == "model":
        _rewrite_meta(directory, model="other-model")
    elif change == "format":
        _rewrite_meta(directory, format="vectors-v0")
    elif change == "chunk_format":
        _rewrite_meta(directory, chunk_format="chunk-v0")
    elif change == "dims":
        _rewrite_meta(directory, dims=3)
    else:
        chunk_format = "chunk-v2"

    caplog.set_level(logging.WARNING, logger="app.vector_index")
    index = VectorIndex(cache, model, chunk_format=chunk_format)
    fake = FakeEmbedder()
    result = index.ensure(chunks, fake)

    assert len(_warnings(caplog)) == 1
    message = _warnings(caplog)[0].getMessage()
    assert "ignored (" in message
    assert message.endswith("rebuilding.")
    assert result.reused == 0
    assert result.embedded == 2
    assert len(fake.document_calls) == 1
    caplog.clear()
    fresh = VectorIndex(cache, model, chunk_format=chunk_format)
    assert fresh.missing(chunks) == 0
    assert _warnings(caplog) == []
    matrix = _load_matrix(fresh.directory)
    assert matrix.dtype == np.float32
    assert matrix.shape == (2, 64)


def test_v5_embedder_dims_mismatch_rebuilds(tmp_path, caplog):
    cache = tmp_path / "cache"
    model = "fake-bow"
    chunks = [_chunk("alpha gpio pin"), _chunk("beta uart clock", path="b.c")]
    VectorIndex(cache, model).ensure(chunks, FakeEmbedder())
    caplog.set_level(logging.WARNING, logger="app.vector_index")
    narrow = FakeEmbedder(dims=32)

    result = VectorIndex(cache, model).ensure(chunks, narrow)

    assert len(_warnings(caplog)) == 1
    assert "incompatible dims" in _warnings(caplog)[0].getMessage()
    assert result.reused == 0
    assert result.embedded == 2
    assert _load_matrix(vector_dir(cache, model)).shape == (2, 32)


# V6
def _corrupt(directory: Path, kind: str) -> None:
    if kind == "truncated":
        raw = (directory / "vectors.npy").read_bytes()
        (directory / "vectors.npy").write_bytes(raw[:20])
    elif kind == "garbage-keys":
        (directory / "keys.json").write_text("{not json", encoding="utf-8")
    elif kind == "invalid-meta":
        (directory / "meta.json").write_text("[]", encoding="utf-8")
    elif kind == "flipped-byte":
        raw = bytearray((directory / "vectors.npy").read_bytes())
        raw[-1] ^= 0xFF
        (directory / "vectors.npy").write_bytes(raw)
    elif kind == "count":
        _rewrite_meta(directory, count=99)
    elif kind == "duplicate-keys":
        keys = json.loads((directory / "keys.json").read_text(encoding="utf-8"))
        _set_keys(directory, [keys[0], keys[0]])
    elif kind == "float64":
        matrix = _load_matrix(directory).astype(np.float64)
        _set_vectors(directory, matrix)
    elif kind == "onedim":
        _set_vectors(directory, np.zeros(4, dtype=np.float32))
    elif kind == "object":
        (directory / "vectors.npy").write_bytes(_OBJECT_NPY)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        meta["vectors_sha256"] = hashlib.sha256(_OBJECT_NPY).hexdigest()
        (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    else:
        raise AssertionError(kind)


@pytest.mark.parametrize(
    "kind",
    [
        "truncated",
        "garbage-keys",
        "invalid-meta",
        "flipped-byte",
        "count",
        "duplicate-keys",
        "float64",
        "onedim",
        "object",
    ],
)
def test_v6_corrupt_index_rebuilds(tmp_path, caplog, monkeypatch, kind):
    cache = tmp_path / "cache"
    model = "fake-bow"
    chunks = [_chunk("alpha gpio pin"), _chunk("beta uart clock", path="b.c")]
    VectorIndex(cache, model).ensure(chunks, FakeEmbedder())
    directory = vector_dir(cache, model)
    _corrupt(directory, kind)
    seen: dict[str, object] = {}
    real_load = np.load

    def spy(file, *args, **kwargs):
        seen["allow_pickle"] = kwargs.get("allow_pickle", "missing")
        return real_load(file, *args, **kwargs)

    if kind == "object":
        monkeypatch.setattr("app.vector_index.np.load", spy)

    caplog.set_level(logging.WARNING, logger="app.vector_index")
    index = VectorIndex(cache, model)
    fake = FakeEmbedder()
    result = index.ensure(chunks, fake)

    assert len(_warnings(caplog)) == 1
    message = _warnings(caplog)[0].getMessage()
    assert "ignored (" in message
    assert message.endswith("rebuilding.")
    assert result.embedded == 2
    assert result.reused == 0
    if kind == "object":
        assert seen["allow_pickle"] is False
    caplog.clear()
    fresh = VectorIndex(cache, model)
    assert fresh.missing(chunks) == 0
    assert _warnings(caplog) == []
    matrix = _load_matrix(fresh.directory)
    assert matrix.dtype == np.float32
    assert matrix.shape == (2, 64)


# V7
def test_v7a_failed_temp_write_keeps_previous_files(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    model = "fake-bow"
    first = _chunk("alpha gpio pin")
    second = _chunk("beta uart clock", path="b.c")
    index = VectorIndex(cache, model)
    index.ensure([first], FakeEmbedder())
    before = _files(index.directory)
    calls = {"n": 0}
    real = vector_index._write_temp_file

    def wrapped(path, data):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk full")
        real(path, data)

    monkeypatch.setattr("app.vector_index._write_temp_file", wrapped)
    fake = FakeEmbedder()

    result = index.ensure([first, second], fake)

    assert result.write_failed is True
    assert calls["n"] == 2
    assert _files(index.directory) == before
    assert list(index.directory.glob("*.tmp-*")) == []
    looked = index.lookup([first, second])
    assert looked is not None
    assert looked.matrix.shape == (2, 64)
    assert looked.keys == (first.key, second.key)


def test_v7b_failed_meta_replace_is_corrupt_on_reload(tmp_path, monkeypatch, caplog):
    cache = tmp_path / "cache"
    model = "fake-bow"
    chunks = [_chunk("alpha gpio pin"), _chunk("beta uart clock", path="b.c")]
    seen: list[str] = []
    real = os.replace

    def spy(src, dst, *args, **kwargs):
        seen.append(Path(dst).name)
        if len(seen) == 3:
            raise OSError("replace meta")
        return real(src, dst, *args, **kwargs)

    monkeypatch.setattr("app.vector_index.os.replace", spy)
    index = VectorIndex(cache, model)
    result = index.ensure(chunks, FakeEmbedder())

    assert result.write_failed is True
    assert seen[:3] == ["vectors.npy", "keys.json", "meta.json"]
    caplog.set_level(logging.WARNING, logger="app.vector_index")
    caplog.clear()
    fresh = VectorIndex(cache, model)
    assert fresh.missing(chunks) == 2
    assert fresh.lookup(chunks) is None
    assert len(fresh) == 0
    assert len(_warnings(caplog)) == 1
    fresh.ensure(chunks, FakeEmbedder())


def test_v7c_leftover_temp_is_deleted_on_load(tmp_path, caplog):
    cache = tmp_path / "cache"
    model = "fake-bow"
    chunks = [_chunk("alpha gpio pin")]
    VectorIndex(cache, model).ensure(chunks, FakeEmbedder())
    directory = vector_dir(cache, model)
    leftover = directory / "vectors.npy.tmp-123-0"
    leftover.write_bytes(b"crash")
    caplog.set_level(logging.WARNING, logger="app.vector_index")

    fresh = VectorIndex(cache, model)

    assert fresh.missing(chunks) == 0
    assert not leftover.exists()
    assert _warnings(caplog) == []


# V8
def test_v8_embedder_failure_keeps_first_group(tmp_path, monkeypatch):
    monkeypatch.setattr("app.vector_index.ENSURE_GROUP", 2)
    chunks = [_chunk(f"text {index} word", path=f"{index}.c") for index in range(5)]

    class Boom(FakeEmbedder):
        def embed_documents(self, texts):
            if self.document_calls:
                raise EmbedderUnavailableError("second")
            return super().embed_documents(texts)

    index = VectorIndex(tmp_path / "cache", "fake-bow")
    with pytest.raises(EmbedderUnavailableError, match="second"):
        index.ensure(chunks, Boom())

    matrix = _load_matrix(index.directory)
    assert matrix.shape == (2, 64)
    keys = json.loads((index.directory / "keys.json").read_text(encoding="utf-8"))
    assert keys == [chunks[0].key, chunks[1].key]

    fresh = VectorIndex(tmp_path / "cache", "fake-bow")
    fake = FakeEmbedder()
    result = fresh.ensure(chunks, fake)
    assert result.reused == 2
    assert result.embedded == 3
    assert [text for call in fake.document_calls for text in call] == [
        chunks[2].text,
        chunks[3].text,
        chunks[4].text,
    ]


def test_v8_wrong_group_shape_raises(tmp_path):
    chunks = [_chunk("alpha gpio pin"), _chunk("beta uart clock", path="b.c")]

    class Short(FakeEmbedder):
        def embed_documents(self, texts):
            rows = super().embed_documents(texts)
            return rows[:-1]

    class Wide(FakeEmbedder):
        def embed_documents(self, texts):
            rows = super().embed_documents(texts)
            return np.concatenate(
                [rows, np.zeros((rows.shape[0], 1), dtype=np.float32)], axis=1
            )

    with pytest.raises(EmbedderUnavailableError):
        VectorIndex(tmp_path / "short", "fake-bow").ensure(chunks, Short())
    with pytest.raises(EmbedderUnavailableError):
        VectorIndex(tmp_path / "wide", "fake-bow").ensure(chunks, Wide())


# V9
def test_v9_progress_and_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr("app.vector_index.ENSURE_GROUP", 2)
    monkeypatch.setattr("app.vector_index.CHECKPOINT_EVERY", 2)
    replaced: list[str] = []
    real = os.replace

    def spy(src, dst, *args, **kwargs):
        replaced.append(Path(dst).name)
        return real(src, dst, *args, **kwargs)

    monkeypatch.setattr("app.vector_index.os.replace", spy)

    class Watch(FakeEmbedder):
        def __init__(self):
            super().__init__()
            self.at_call: list[int] = []

        def embed_documents(self, texts):
            self.at_call.append(len(replaced))
            return super().embed_documents(texts)

    chunks = [_chunk(f"text {index} word", path=f"{index}.c") for index in range(5)]
    progress: list[tuple[int, int]] = []
    watch = Watch()
    VectorIndex(tmp_path / "cache", "fake-bow").ensure(
        chunks,
        watch,
        progress=lambda done, total: progress.append((done, total)),
    )

    assert progress == [(2, 5), (4, 5), (5, 5)]
    assert watch.at_call[2] > 3
    assert len(replaced) == watch.at_call[2] + 3
    assert replaced[:3] == ["vectors.npy", "keys.json", "meta.json"]

    progress.clear()
    watch.at_call.clear()
    VectorIndex(tmp_path / "cache", "fake-bow").ensure(
        chunks,
        watch,
        progress=lambda done, total: progress.append((done, total)),
    )
    assert progress == []
    assert watch.at_call == []


# V9b
def test_v9b_embedder_overrides_group_and_checkpoint(tmp_path, monkeypatch):
    replaced: list[int] = []
    real = os.replace

    def spy(src, dst, *args, **kwargs):
        replaced.append(1)
        return real(src, dst, *args, **kwargs)

    monkeypatch.setattr("app.vector_index.os.replace", spy)

    class Sized(FakeEmbedder):
        group_size = 3
        checkpoint_every = 3

        def __init__(self):
            super().__init__()
            self.sizes: list[int] = []
            self.at_call: list[int] = []

        def embed_documents(self, texts):
            items = list(texts)
            self.sizes.append(len(items))
            self.at_call.append(len(replaced))
            return super().embed_documents(items)

    chunks = [_chunk(f"text {index} word", path=f"{index}.c") for index in range(6)]
    sized = Sized()
    VectorIndex(tmp_path / "cache", "fake-bow").ensure(chunks, sized)

    assert sized.sizes == [3, 3]
    assert sized.at_call[1] - sized.at_call[0] == 3
    assert len(replaced) - sized.at_call[1] == 3

    monkeypatch.setattr("app.vector_index.ENSURE_GROUP", 2)
    for override in (0, -1):

        class Fallback(FakeEmbedder):
            group_size = override

            def __init__(self):
                super().__init__()
                self.sizes: list[int] = []

            def embed_documents(self, texts):
                items = list(texts)
                self.sizes.append(len(items))
                return super().embed_documents(items)

        five = [_chunk(f"fallback {override} {index}", path=f"f{index}.c") for index in range(5)]
        fallback = Fallback()
        VectorIndex(tmp_path / f"fallback-{override}", "fake-bow").ensure(five, fallback)
        assert fallback.sizes == [2, 2, 1]


def test_v9b_constants_match_the_spec():
    assert vector_index.VECTORS_FORMAT == "vectors-v1"
    assert vector_index.ENSURE_GROUP == 256
    assert vector_index.CHECKPOINT_EVERY == 2048
    assert vector_index.COMPACT_STALE_RATIO == 0.5


# V10
def test_v10_lookup_and_missing_are_read_only(tmp_path):
    cache = tmp_path / "cache"
    model = "fake-bow"
    chunks = [
        _chunk("alpha gpio pin", path="a.c"),
        _chunk("beta uart clock", path="b.c"),
        _chunk("alpha gpio pin", path="c.c"),
    ]
    index = VectorIndex(cache, model)
    assert index.missing(chunks) == 2
    assert index.lookup(chunks) is None
    empty = index.lookup(())
    assert empty is not None
    assert empty.chunks == ()
    assert empty.keys == ()
    assert empty.matrix.shape == (0, 0)
    assert empty.matrix.dtype == np.float32
    assert not cache.exists()

    fake = FakeEmbedder()
    index.ensure([chunks[0]], fake)
    before = _files(index.directory)
    calls = list(fake.document_calls)
    queries = list(fake.query_calls)

    assert index.lookup(chunks) is None
    assert index.missing(chunks) == 1
    looked = index.lookup([chunks[2], chunks[0]])
    assert looked is not None
    assert looked.keys == (chunks[0].key,)
    assert looked.chunks == (chunks[2], chunks[0])
    assert looked.matrix.shape == (1, 64)
    warmed = index.lookup(())
    assert warmed is not None
    assert warmed.matrix.shape == (0, 64)
    assert fake.document_calls == calls
    assert fake.query_calls == queries
    assert _files(index.directory) == before


# V11
def test_v11_lookup_does_not_wait_for_ensure(tmp_path):
    cache = tmp_path / "cache"
    model = "fake-bow"
    old = [_chunk("alpha gpio pin", path="a.c"), _chunk("beta uart clock", path="b.c")]
    index = VectorIndex(cache, model)
    index.ensure(old, FakeEmbedder())
    added = old + [_chunk("gamma spi bus", path="c.c")]

    class Blocking(FakeEmbedder):
        def __init__(self):
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def embed_documents(self, texts):
            self.entered.set()
            if not self.release.wait(5):
                raise TimeoutError("not released")
            return super().embed_documents(texts)

    blocker = Blocking()
    errors: list[BaseException] = []

    def run():
        try:
            index.ensure(added, blocker)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert blocker.entered.wait(5)
        started = time.perf_counter()
        looked = index.lookup(old)
        missing = index.missing(added)
        elapsed = time.perf_counter() - started
        assert elapsed < 0.5
        assert looked is not None
        assert looked.keys == (old[0].key, old[1].key)
        assert missing == 1
        assert index.lookup(added) is None
    finally:
        blocker.release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert errors == []
    done = index.lookup(added)
    assert done is not None
    assert done.matrix.shape[0] == 3


def test_v11_concurrent_ensure_embeds_once(tmp_path):
    cache = tmp_path / "cache"
    model = "fake-bow"
    chunks = [_chunk("alpha gpio pin"), _chunk("beta uart clock", path="b.c")]
    index = VectorIndex(cache, model)

    class Blocking(FakeEmbedder):
        def __init__(self):
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def embed_documents(self, texts):
            self.entered.set()
            if not self.release.wait(5):
                raise TimeoutError("not released")
            return super().embed_documents(texts)

    blocker = Blocking()
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def run():
        try:
            barrier.wait(5)
            index.ensure(chunks, blocker)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    try:
        assert blocker.entered.wait(5)
    finally:
        blocker.release.set()
        for thread in threads:
            thread.join(5)
    assert errors == []
    assert all(not thread.is_alive() for thread in threads)
    assert len(blocker.document_calls) == 1
    assert blocker.document_calls[0] == [chunks[0].text, chunks[1].text]


# V12
def test_v12_slug_stays_inside_vectors_and_constructor_does_no_io(tmp_path):
    assert model_slug("BAAI/bge-small-en-v1.5") == "BAAI-bge-small-en-v1.5"
    assert model_slug("") == "model"
    assert model_slug("..") == "model"
    cache = tmp_path / "cache"
    index = VectorIndex(cache, "../../evil name")
    assert not cache.exists()
    root = (cache / "vectors").resolve()
    chunks = [_chunk("alpha gpio pin")]
    index.ensure(chunks, FakeEmbedder())
    directory = index.directory.resolve()
    assert directory.is_relative_to(root)
    assert ".." not in directory.parts
    assert directory.parent == root
    for path in directory.rglob("*"):
        assert path.resolve().is_relative_to(directory)
        assert path.resolve().is_relative_to(root)


# V13
def test_v13_one_info_line_per_ensure(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="app.vector_index")
    chunks = [_chunk("alpha gpio pin"), _chunk("beta uart clock", path="b.c")]
    index = VectorIndex(tmp_path / "cache", "fake-model")
    result = index.ensure(chunks, FakeEmbedder())
    infos = [
        record
        for record in caplog.records
        if record.levelno == logging.INFO and record.name == "app.vector_index"
    ]
    assert len(infos) == 1
    message = infos[0].getMessage()
    assert message == (
        "Vector index fake-model: 2 chunks, 0 reused, 2 embedded, 0 stale, "
        f"compacted no, {result.seconds:.1f} s."
    )
    assert chunks[0].text not in message
    caplog.clear()
    index.ensure([], FakeEmbedder())
    assert [
        record
        for record in caplog.records
        if record.name == "app.vector_index"
    ] == []
