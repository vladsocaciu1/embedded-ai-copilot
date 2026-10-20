import logging
import os
import sys
import threading
import time

import numpy as np
import pytest

import app.embedder as embedder_module
from app.embedder import Embedder, EmbedderUnavailableError
from tests.fake_embedder import FakeEmbedder


class FakeTE:
    instances: list["FakeTE"] = []
    delay = 0.0
    rows: list[np.ndarray] | None = None

    def __init__(self, model_name, cache_dir=None, threads=None, **kwargs):
        if FakeTE.delay:
            time.sleep(FakeTE.delay)
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.threads = threads
        self.kwargs = kwargs
        self.embed_calls: list[tuple[list[str], int]] = []
        self.query_calls: list[str] = []
        FakeTE.instances.append(self)

    @classmethod
    def get_embedding_size(cls, model_name):
        return 3

    def embed(self, documents, batch_size=256):
        docs = list(documents)
        self.embed_calls.append((docs, batch_size))
        rows = self.rows if self.rows is not None else [np.array([3.0, 4.0, 0.0])] * len(docs)
        for row in rows:
            yield np.asarray(row, dtype=np.float64)

    def query_embed(self, query):
        self.query_calls.append(query)
        yield np.array([3.0, 4.0, 0.0])


class BoomTE:
    calls = 0

    def __init__(self, model_name, cache_dir=None, threads=None, **kwargs):
        BoomTE.calls += 1
        raise OSError("disk")

    @classmethod
    def get_embedding_size(cls, model_name):
        return 3


class ShortTE:
    def __init__(self, model_name, cache_dir=None, threads=None, **kwargs):
        pass

    @classmethod
    def get_embedding_size(cls, model_name):
        return 3

    def embed(self, documents, batch_size=256):
        yield np.array([1.0, 0.0, 0.0])

    def query_embed(self, query):
        yield np.array([1.0, 0.0, 0.0])


@pytest.fixture
def fake_te(monkeypatch):
    FakeTE.instances = []
    FakeTE.delay = 0.0
    FakeTE.rows = None
    monkeypatch.setattr(embedder_module, "_text_embedding_class", lambda: FakeTE)
    return FakeTE


def test_b1_documents_are_normalized_float32_in_order(fake_te, tmp_path):
    FakeTE.rows = [
        np.array([3.0, 4.0, 0.0]),
        np.array([0.0, 0.0, 5.0]),
    ]
    embedder = Embedder(cache_dir=tmp_path)

    vectors = embedder.embed_documents(["first", "second"])

    assert vectors.shape == (2, 3)
    assert vectors.dtype == np.float32
    assert np.allclose(vectors[0], [0.6, 0.8, 0.0])
    assert np.allclose(vectors[1], [0.0, 0.0, 1.0])
    assert FakeTE.instances[0].embed_calls[0][0] == ["first", "second"]


def test_b2_zero_vector_and_empty_documents(fake_te, tmp_path):
    FakeTE.rows = [np.array([0.0, 0.0, 0.0])]
    embedder = Embedder(cache_dir=tmp_path)

    vectors = embedder.embed_documents(["zero"])

    assert vectors.shape == (1, 3)
    assert vectors.dtype == np.float32
    assert np.all(vectors == 0)
    assert not np.isnan(vectors).any()

    FakeTE.instances = []
    empty = Embedder(cache_dir=tmp_path).embed_documents([])
    assert empty.shape == (0, 3)
    assert empty.dtype == np.float32
    assert FakeTE.instances == []


def test_b3_batch_threads_and_cache_dir(fake_te, monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    embedder = Embedder(cache_dir=cache)
    embedder.embed_documents(["a"])
    assert FakeTE.instances[0].embed_calls[0][1] == 16
    assert FakeTE.instances[0].cache_dir == str(cache / "models")
    assert (cache / "models").is_dir()

    FakeTE.instances = []
    Embedder(cache_dir=cache, batch_size=4).embed_documents(["a"])
    assert FakeTE.instances[0].embed_calls[0][1] == 4

    def threads_for(cpu_count):
        FakeTE.instances = []
        monkeypatch.setattr(os, "cpu_count", lambda: cpu_count)
        Embedder(cache_dir=tmp_path / "t").embed_query("x")
        return FakeTE.instances[-1].threads

    assert threads_for(8) == 7
    assert threads_for(None) == 1
    assert threads_for(1) == 1

    FakeTE.instances = []
    Embedder(cache_dir=cache, threads=3).embed_query("x")
    assert FakeTE.instances[-1].threads == 3

    default_cache = tmp_path / "default-cache"
    monkeypatch.setattr(embedder_module, "DEFAULT_EMBEDDING_CACHE_DIR", default_cache)
    FakeTE.instances = []
    Embedder().embed_query("x")
    assert FakeTE.instances[-1].cache_dir == str(default_cache / "models")
    assert (default_cache / "models").is_dir()


def test_b4_query_uses_query_embed(fake_te, tmp_path):
    embedder = Embedder(cache_dir=tmp_path)

    vector = embedder.embed_query("hello")

    instance = FakeTE.instances[0]
    assert instance.query_calls == ["hello"]
    assert instance.embed_calls == []
    assert vector.shape == (3,)
    assert vector.dtype == np.float32
    assert np.allclose(vector, [0.6, 0.8, 0.0])


def test_b5_loads_once_across_threads(fake_te, tmp_path):
    embedder = Embedder(cache_dir=tmp_path)
    assert FakeTE.instances == []
    assert embedder.loaded is False

    embedder.embed_documents(["a", "b"])
    embedder.embed_documents(["c"])
    embedder.embed_query("d")

    assert len(FakeTE.instances) == 1
    assert embedder.loaded is True

    FakeTE.instances = []
    FakeTE.delay = 0.05
    concurrent = Embedder(cache_dir=tmp_path / "other")
    barrier = threading.Barrier(8)
    errors: list[BaseException] = []

    def worker():
        try:
            barrier.wait()
            concurrent.embed_query("x")
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(FakeTE.instances) == 1
    assert concurrent.loaded is True


def test_b6_constructor_error_is_wrapped_and_retried(monkeypatch, caplog, tmp_path):
    BoomTE.calls = 0
    monkeypatch.setattr(embedder_module, "_text_embedding_class", lambda: BoomTE)
    caplog.set_level(logging.WARNING, logger="app.embedder")
    embedder = Embedder(model="test-model", cache_dir=tmp_path)

    with pytest.raises(
        EmbedderUnavailableError,
        match="Embedding model test-model unavailable: OSError",
    ) as first:
        embedder.embed_query("a")
    assert isinstance(first.value.__cause__, OSError)
    assert any(record.levelno == logging.WARNING for record in caplog.records)
    assert embedder.loaded is False

    with pytest.raises(EmbedderUnavailableError):
        embedder.embed_documents(["b"])
    assert BoomTE.calls == 2


def test_b7_wrong_row_count_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(embedder_module, "_text_embedding_class", lambda: ShortTE)
    embedder = Embedder(cache_dir=tmp_path)

    with pytest.raises(EmbedderUnavailableError):
        embedder.embed_documents(["a", "b"])


def test_b8_guard_blocks_a_real_model(tmp_path):
    embedder = Embedder(cache_dir=tmp_path)

    with pytest.raises((AssertionError, EmbedderUnavailableError)) as caught:
        embedder.embed_query("x")

    error = caught.value
    if isinstance(error, EmbedderUnavailableError):
        assert isinstance(error.__cause__, AssertionError)
        assert "tests must not load a real embedding model" in str(error.__cause__)
    else:
        assert "tests must not load a real embedding model" in str(error)
    assert list(tmp_path.rglob("*.onnx")) == []
    assert "fastembed" not in sys.modules


def test_b9_fake_embedder_is_deterministic():
    first = FakeEmbedder()
    second = FakeEmbedder()
    assert first.loaded is True
    assert first.model == "fake-bow"
    assert first.dims == 64

    text = "Hello, GPIO pin!"
    left = first.embed_query(text)
    right = second.embed_query(text)
    again = first.embed_query("hello gpio pin")
    assert np.array_equal(left, right)
    assert np.array_equal(left, again)
    assert left.dtype == np.float32
    assert left.shape == (64,)
    assert np.isclose(np.linalg.norm(left), 1.0)

    empty = first.embed_query("")
    assert np.all(empty == 0)
    assert not np.isnan(empty).any()

    wide = FakeEmbedder(dims=256)
    shared = wide.embed_query("the gpio pin toggles")
    related = wide.embed_query("gpio pin output")
    unrelated = wide.embed_query("uart baud clock")
    assert float(np.dot(shared, related)) > float(np.dot(shared, unrelated))

    docs = first.embed_documents(["one two", "three"])
    assert docs.shape == (2, 64)
    assert first.document_calls == [["one two", "three"]]
    assert first.query_calls == [text, "hello gpio pin", ""]
