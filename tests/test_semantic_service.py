import logging
import threading
from pathlib import Path

import numpy as np
import pytest

from app.chunker import chunk_index, unique_texts
from app.config import EmbeddingSettings
from app.embedder import Embedder, EmbedderUnavailableError, build_embedder
from app.index_cache import IndexCache
from app.models import ProjectFile
from app.semantic_search import embed_chunks, semantic_rank
from app.semantic_service import (
    RETRY_AFTER_FAILURE_S,
    SemanticResult,
    SemanticService,
)
from app.vector_index import VectorIndex
from tests.fake_embedder import FakeEmbedder

QUESTION = "How does Can_Init work?"


class BlockingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def embed_documents(self, texts):
        items = list(texts)
        self.document_calls.append(items)
        self.entered.set()
        if not self.release.wait(timeout=10):
            raise TimeoutError("semantic build was not released")
        if not items:
            return np.zeros((0, self._dims), dtype=np.float32)
        return np.vstack([self._vector(text) for text in items])


class AlwaysFailEmbedder(FakeEmbedder):
    def embed_documents(self, texts):
        self.document_calls.append(list(texts))
        raise EmbedderUnavailableError("offline")


class MidFailEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.group_size = 1
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        if self.calls == 2:
            raise EmbedderUnavailableError("offline")
        return super().embed_documents(texts)


class EnsureCounter:
    def __init__(self, inner: VectorIndex) -> None:
        self.inner = inner
        self.calls = 0

    def missing(self, chunks):
        return self.inner.missing(chunks)

    def lookup(self, chunks):
        return self.inner.lookup(chunks)

    def ensure(self, chunks, embedder, progress=None):
        self.calls += 1
        return self.inner.ensure(chunks, embedder, progress=progress)


def _source(name: str, body: str) -> ProjectFile:
    return ProjectFile(Path(name), body)


def _can_files() -> list[ProjectFile]:
    return [
        _source(
            "src/can.c",
            "void Can_Init(void)\n{\n    return;\n}\n",
        )
    ]


def _multi_files() -> list[ProjectFile]:
    return [
        _source(
            "src/board.c",
            "void one(void)\n{\n    return;\n}\n"
            "void two(void)\n{\n    return;\n}\n"
            "void three(void)\n{\n    return;\n}\n",
        )
    ]


def _load(files: list[ProjectFile], cache: IndexCache | None = None):
    index, _graph = (cache or IndexCache()).get(Path("/proj"), files)
    return index


def _service(tmp_path, embedder, *, clock=None, counter=False):
    cache_dir = tmp_path / "cache"
    settings = EmbeddingSettings(semantic_search="local", cache_dir=cache_dir)
    vector_index = VectorIndex(cache_dir, settings.model)
    counted = EnsureCounter(vector_index) if counter else None
    kwargs = {}
    if clock is not None:
        kwargs["clock"] = clock
    service = SemanticService(
        settings,
        embedder=embedder,
        vector_index=counted if counted is not None else vector_index,
        **kwargs,
    )
    return service, counted if counted is not None else vector_index


def _build_threads():
    return [thread for thread in threading.enumerate() if thread.name == "semantic-build"]


def _warnings(caplog):
    return [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING and record.name == "app.semantic_service"
    ]


def test_build_embedder_off_raises_and_local_is_unloaded(tmp_path):

    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        build_embedder(EmbeddingSettings(semantic_search="off"))
    embedder = build_embedder(
        EmbeddingSettings(semantic_search="local", cache_dir=tmp_path / "models")
    )
    assert isinstance(embedder, Embedder)
    assert embedder.loaded is False
    assert not (tmp_path / "models").exists()


def test_m1_disabled_builds_nothing(tmp_path, monkeypatch):

    # Arrange
    cache_dir = tmp_path / "cache"
    calls = []

    def spy(settings):
        calls.append(settings)
        raise AssertionError("build_embedder")

    monkeypatch.setattr("app.semantic_service.build_embedder", spy)
    files = _can_files()
    index = _load(files)
    before = list(_build_threads())

    # Act
    service = SemanticService(
        EmbeddingSettings(semantic_search="off", cache_dir=cache_dir)
    )
    result = service.prepare(QUESTION, index, files)
    service.start_build(index, files)

    # Assert
    assert service.enabled is False
    assert result == SemanticResult(None, ())
    assert calls == []
    assert _build_threads() == before
    assert service.progress() is None
    assert service.wait_idle() is True
    assert not cache_dir.exists()


def test_m2_cold_then_ready(tmp_path):

    # Arrange
    embedder = BlockingEmbedder()
    service, counter = _service(tmp_path, embedder, counter=True)
    files = _can_files()
    index = _load(files)
    chunks = chunk_index(index, files)

    # Act
    try:
        first = service.prepare(QUESTION, index, files)
        assert embedder.entered.wait(timeout=5)
        threads = _build_threads()
        second = service.prepare(QUESTION, index, files)
        calls_while_building = counter.calls
        batches_while_building = len(embedder.document_calls)
    finally:
        embedder.release.set()
    assert service.wait_idle(timeout=5) is True
    ready = service.prepare(QUESTION, index, files)

    # Assert
    assert first == SemanticResult("building", ())
    assert second == SemanticResult("building", ())
    assert calls_while_building == 1
    assert batches_while_building == 1
    assert len(threads) == 1
    assert threads[0].daemon is True
    assert threads[0].name == "semantic-build"
    assert ready.status == "ready"
    assert ready.order == semantic_rank(
        QUESTION, embed_chunks(chunks, FakeEmbedder()), FakeEmbedder()
    )
    assert counter.calls == 1
    assert len(embedder.document_calls) == 1
    assert embedder.query_calls == [QUESTION]


def test_m3_edit_embeds_only_changed_chunks(tmp_path):

    # Arrange
    embedder = BlockingEmbedder()
    service, _index = _service(tmp_path, embedder)
    cache = IndexCache()
    original = [
        _source("a.c", "void alpha(void)\n{\n    return;\n}\n"),
        _source("b.c", "void beta(void)\n{\n    return;\n}\n"),
    ]
    changed = [
        original[0],
        _source("b.c", "void beta(void)\n{\n    int edited = 1;\n    return;\n}\n"),
    ]
    index = _load(original, cache)
    try:
        service.prepare(QUESTION, index, original)
        assert embedder.entered.wait(timeout=5)
    finally:
        embedder.release.set()
    assert service.wait_idle(timeout=5) is True
    assert service.prepare(QUESTION, index, original).status == "ready"
    before = len(embedder.document_calls)
    edited = _load(changed, cache)
    assert edited is not index
    embedder.entered.clear()
    embedder.release.clear()

    # Act
    try:
        building = service.prepare(QUESTION, edited, changed)
        assert embedder.entered.wait(timeout=5)
    finally:
        embedder.release.set()
    assert service.wait_idle(timeout=5) is True
    ready = service.prepare(QUESTION, edited, changed)

    # Assert
    old_keys = set(unique_texts(chunk_index(index, original)))
    new_texts = unique_texts(chunk_index(edited, changed))
    changed_texts = {text for key, text in new_texts.items() if key not in old_keys}
    embedded = [
        text
        for batch in embedder.document_calls[before:]
        for text in batch
    ]
    chunks = chunk_index(edited, changed)
    assert building.status == "building"
    assert building.order == ()
    assert set(embedded) == changed_texts
    assert embedded
    assert ready.status == "ready"
    assert ready.order == semantic_rank(
        QUESTION, embed_chunks(chunks, FakeEmbedder()), FakeEmbedder()
    )


def test_m4_failure_backoff_and_reused_rows(tmp_path, caplog):

    # Arrange
    caplog.set_level(logging.WARNING)
    clock = [0.0]
    embedder = AlwaysFailEmbedder()
    service, counter = _service(
        tmp_path, embedder, clock=lambda: clock[0], counter=True
    )
    files = _can_files()
    index = _load(files)

    # Act / Assert: failure, then no second build until 300 s
    assert service.prepare(QUESTION, index, files).status in {"building", "off"}
    assert service.wait_idle(timeout=5) is True
    failed = service.prepare(QUESTION, index, files)
    assert failed == SemanticResult("off", ())
    assert counter.calls == 1
    assert [record.message for record in _warnings(caplog)] == [
        "Semantic index build failed (EmbedderUnavailableError: offline); "
        "using lexical ranking."
    ]
    clock[0] = RETRY_AFTER_FAILURE_S - 1
    assert service.prepare(QUESTION, index, files) == SemanticResult("off", ())
    assert counter.calls == 1
    clock[0] = RETRY_AFTER_FAILURE_S
    again = service.prepare(QUESTION, index, files)
    assert again.status in {"building", "off"}
    assert service.wait_idle(timeout=5) is True
    assert counter.calls == 2
    assert len(_warnings(caplog)) == 2

    # Act / Assert: rows embedded before a mid-run failure are reused
    caplog.clear()
    clock = [0.0]
    flaky = MidFailEmbedder()
    retry, vector_index = _service(
        tmp_path / "retry", flaky, clock=lambda: clock[0]
    )
    many = _multi_files()
    many_index = _load(many)
    assert retry.prepare(QUESTION, many_index, many).status in {"building", "off"}
    assert retry.wait_idle(timeout=5) is True
    assert len(vector_index) == 1
    assert len(_warnings(caplog)) == 1
    stored = flaky.document_calls[0][0]
    clock[0] = RETRY_AFTER_FAILURE_S
    assert retry.prepare(QUESTION, many_index, many).status in {"building", "ready"}
    assert retry.wait_idle(timeout=5) is True
    retried = [text for batch in flaky.document_calls[1:] for text in batch]
    assert stored not in retried
    assert retried
    assert vector_index.missing(chunk_index(many_index, many)) == 0
    ready = retry.prepare(QUESTION, many_index, many)
    assert ready.status == "ready"


def test_m5_query_failure_at_ready(tmp_path, caplog):

    # Arrange
    caplog.set_level(logging.WARNING)
    embedder = FakeEmbedder()
    service, _index = _service(tmp_path, embedder)
    files = _can_files()
    index = _load(files)
    service.start_build(index, files)
    assert service.wait_idle(timeout=5) is True

    def boom(text):
        raise RuntimeError("query-boom")

    embedder.embed_query = boom

    # Act
    result = service.prepare(QUESTION, index, files)

    # Assert
    assert result == SemanticResult("off", ())
    assert [record.message for record in _warnings(caplog)] == [
        "Semantic search failed (RuntimeError); using lexical ranking."
    ]


def test_m6_chunks_are_memoized_by_index_identity(tmp_path, monkeypatch):

    # Arrange
    embedder = FakeEmbedder()
    service, _index = _service(tmp_path, embedder)
    cache = IndexCache()
    files = _can_files()
    index = _load(files, cache)
    calls = []
    real = chunk_index

    def spy(project_index, project_files):
        calls.append(project_index)
        return real(project_index, project_files)

    monkeypatch.setattr("app.semantic_service.chunk_index", spy)

    # Act
    service.prepare(QUESTION, index, files)
    service.wait_idle(timeout=5)
    service.prepare(QUESTION, index, files)
    edited_files = [
        _source("src/can.c", "void Can_Init(void)\n{\n    int x = 1;\n    return;\n}\n")
    ]
    edited = _load(edited_files, cache)
    service.prepare(QUESTION, edited, edited_files)
    assert service.wait_idle(timeout=5) is True

    # Assert
    assert calls == [index, edited]
    assert edited is not index


def test_m7_progress_while_embedding_and_none_when_idle(tmp_path):

    # Arrange
    embedder = BlockingEmbedder()
    service, vector_index = _service(tmp_path, embedder)
    files = _can_files()
    index = _load(files)
    expected = vector_index.missing(chunk_index(index, files))
    assert service.progress() is None

    # Act / Assert
    try:
        assert service.prepare(QUESTION, index, files).status == "building"
        assert embedder.entered.wait(timeout=5)
        progress = service.progress()
        assert progress is not None
        done, total = progress
        assert total == expected
        assert isinstance(done, int)
        assert 0 <= done <= total
    finally:
        embedder.release.set()
    assert service.wait_idle(timeout=5) is True
    assert service.progress() is None


def test_m8_start_build(tmp_path):

    # Arrange / Act / Assert: start_build reaches ready with no second build
    embedder = FakeEmbedder()
    service, counter = _service(tmp_path, embedder, counter=True)
    files = _can_files()
    index = _load(files)
    service.start_build(index, files)
    assert service.wait_idle(timeout=5) is True
    ready = service.prepare(QUESTION, index, files)
    assert ready.status == "ready"
    assert ready.order
    assert counter.calls == 1
    batches = len(embedder.document_calls)
    service.start_build(index, files)
    assert service.wait_idle(timeout=5) is True
    assert counter.calls == 2
    assert len(embedder.document_calls) == batches
    assert service.prepare(QUESTION, index, files).status == "ready"
    assert counter.calls == 2

    # Arrange / Act / Assert: start_build while building is a no-op
    blocked = BlockingEmbedder()
    running, running_counter = _service(tmp_path / "busy", blocked, counter=True)
    try:
        running.start_build(index, files)
        assert blocked.entered.wait(timeout=5)
        running.start_build(index, files)
        assert running_counter.calls == 1
        assert len(_build_threads()) == 1
    finally:
        blocked.release.set()
    assert running.wait_idle(timeout=5) is True

    # Arrange / Act / Assert: no chunks means no thread and status off
    empty_files = [_source("startup.s", "nop\n")]
    empty_index = _load(empty_files)
    assert chunk_index(empty_index, empty_files) == ()
    before = list(_build_threads())
    quiet, _counter = _service(tmp_path / "empty", FakeEmbedder())
    quiet.start_build(empty_index, empty_files)
    result = quiet.prepare(QUESTION, empty_index, empty_files)
    assert _build_threads() == before
    assert result == SemanticResult("off", ())


def test_m9_never_raises(tmp_path, monkeypatch, caplog):

    # Arrange
    caplog.set_level(logging.WARNING)
    files = _can_files()
    index = _load(files)

    def boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    # Act / Assert: missing
    service, vector_index = _service(tmp_path, FakeEmbedder())
    monkeypatch.setattr(vector_index, "missing", boom)
    assert service.prepare(QUESTION, index, files) == SemanticResult("off", ())
    assert _warnings(caplog)[-1].message == (
        "Semantic search failed (RuntimeError); using lexical ranking."
    )

    # Act / Assert: lookup
    caplog.clear()
    lookup_service, lookup_index = _service(tmp_path / "lookup", FakeEmbedder())
    monkeypatch.setattr(lookup_index, "missing", lambda _chunks: 0)
    monkeypatch.setattr(lookup_index, "lookup", boom)
    assert lookup_service.prepare(QUESTION, index, files) == SemanticResult("off", ())
    assert _warnings(caplog)[-1].message == (
        "Semantic search failed (RuntimeError); using lexical ranking."
    )

    # Act / Assert: chunk_index
    caplog.clear()
    chunk_service, _chunk_index = _service(tmp_path / "chunks", FakeEmbedder())
    monkeypatch.setattr("app.semantic_service.chunk_index", boom)
    assert chunk_service.prepare(QUESTION, index, files) == SemanticResult("off", ())
    assert _warnings(caplog)[-1].message == (
        "Semantic search failed (RuntimeError); using lexical ranking."
    )
    assert chunk_service.wait_idle() is True
