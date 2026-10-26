import logging
import shutil
import threading
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.chunker import chunk_index
from app.config import EmbeddingSettings, Settings
from app.context_selector import fuse_rankings, rank_files_detailed
from app.embedder import EmbedderUnavailableError
from app.project_loader import load_project_files
from app.semantic_search import embed_chunks, semantic_rank
from app.semantic_service import SemanticService
from tests.fake_embedder import FakeEmbedder
from tests.test_ai_service import _write_partial_project

QUESTION = "What does Can_Init do?"
ASK_KEYS = {"answer", "context"}
ASK_CONTEXT_KEYS = {
    "mode",
    "files_total",
    "files_included",
    "prompt_tokens",
    "map_tier",
}


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


def _settings(module, limit: int) -> Settings:
    current = module.settings
    return Settings(
        project_path=current.project_path,
        gemini_api_key=current.gemini_api_key,
        gemini_model=current.gemini_model,
        exclude_dirs=current.exclude_dirs,
        max_prompt_tokens=limit,
        embedding=current.embedding,
    )


def _service(tmp_path, embedder, clock=None) -> SemanticService:
    kwargs = {}
    if clock is not None:
        kwargs["clock"] = clock
    return SemanticService(
        EmbeddingSettings(semantic_search="local", cache_dir=tmp_path / "cache"),
        embedder=embedder,
        **kwargs,
    )


def _partial(module, monkeypatch, tmp_path, limit: int = 10_000):
    monkeypatch.setattr(module, "settings", _settings(module, limit))
    _write_partial_project(tmp_path)
    default = tmp_path / "main.c"
    if default.exists():
        default.unlink()


def _build_threads():
    return [thread for thread in threading.enumerate() if thread.name == "semantic-build"]


def _warnings(caplog):
    return [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and record.name == "app.semantic_service"
        and record.message.startswith("Semantic index build failed")
    ]


def test_a1_off_is_identical(ai_service, monkeypatch, tmp_path):

    # Arrange
    module, _client = ai_service
    _partial(module, monkeypatch, tmp_path)
    assert module.semantic_service.enabled is False

    # Act
    off = module.dry_run(QUESTION)
    project_files = load_project_files(
        module.settings.project_path, module.settings.exclude_dirs
    )
    index, graph = module.index_cache.get(module.settings.project_path, project_files)
    embedder = BlockingEmbedder()
    service = _service(tmp_path, embedder)
    monkeypatch.setattr(module, "semantic_service", service)
    try:
        building = module.dry_run(QUESTION)
        assert embedder.entered.wait(timeout=5)
    finally:
        embedder.release.set()
    assert service.wait_idle(timeout=5) is True

    # Assert
    assert off.semantic is None
    assert off.ranked == rank_files_detailed(QUESTION, index, graph)
    assert building.semantic == "building"
    assert building.prompt == off.prompt
    assert building.ranked == off.ranked
    assert embedder.query_calls == []
    assert _build_threads() == []


def test_a2_building_is_lexical(ai_service, monkeypatch, tmp_path, caplog):

    # Arrange
    module, client = ai_service
    _partial(module, monkeypatch, tmp_path)
    client.models.generate_content.return_value.text = "stub answer"
    caplog.set_level(logging.INFO)
    off = module.generate_answer(QUESTION)
    off_contents = client.models.generate_content.call_args.kwargs["contents"]
    off_counts = client.models.count_tokens.call_count
    off_generates = client.models.generate_content.call_count
    client.models.count_tokens.reset_mock()
    client.models.generate_content.reset_mock()
    client.models.generate_content.return_value.text = "stub answer"
    embedder = BlockingEmbedder()
    service = _service(tmp_path, embedder)
    monkeypatch.setattr(module, "semantic_service", service)
    caplog.clear()

    # Act
    try:
        answer = module.generate_answer(QUESTION)
        assert embedder.entered.wait(timeout=5)
        prepared = module.dry_run(QUESTION)
    finally:
        embedder.release.set()
    assert service.wait_idle(timeout=5) is True

    # Assert
    assert prepared.semantic == "building"
    assert prepared.prompt == off_contents
    assert client.models.generate_content.call_args.kwargs["contents"] == off_contents
    assert client.models.count_tokens.call_count == off_counts
    assert client.models.generate_content.call_count == off_generates
    assert answer.text == off.text
    assert answer.context == off.context
    assert any(
        record.levelno == logging.INFO
        and record.message.startswith("Semantic search: building (")
        and record.message.endswith(" chunks); using lexical ranking.")
        for record in caplog.records
    )


def test_a3_ready_fuses(ai_service, monkeypatch, tmp_path, caplog):

    # Arrange
    module, _client = ai_service
    monkeypatch.setattr(module, "settings", _settings(module, 10_000))
    default = tmp_path / "main.c"
    if default.exists():
        default.unlink()
    question = "Explain widget calibrate flux capacitor"
    (tmp_path / "alpha.c").write_text(
        "void widget_marker(void)\n{\n"
        '    const char *note = "widget calibrate flux capacitor";\n}\n',
        encoding="utf-8",
    )
    (tmp_path / "beta.c").write_text(
        "void unused(void)\n{\n"
        '    const char *note = "widget calibrate flux capacitor";\n}\n',
        encoding="utf-8",
    )
    (tmp_path / "noise_both.c").write_text(
        "void widget_calibrate(void)\n{\n    return;\n}\n",
        encoding="utf-8",
    )
    (tmp_path / "noise_flux.c").write_text(
        "void flux_setup(void)\n{\n    return;\n}\n",
        encoding="utf-8",
    )
    pad = "/* pad */\n" * 12_000
    (tmp_path / "zz_pad.c").write_text(pad, encoding="utf-8")
    embedder = FakeEmbedder()
    service = _service(tmp_path, embedder)
    monkeypatch.setattr(module, "semantic_service", service)
    caplog.set_level(logging.INFO)
    project_files = load_project_files(
        module.settings.project_path, module.settings.exclude_dirs
    )
    index, _graph = module.index_cache.get(module.settings.project_path, project_files)

    # Act
    service.start_build(index, project_files)
    assert service.wait_idle(timeout=5) is True
    caplog.clear()
    prepared = module.dry_run(question)

    # Assert
    project_files = load_project_files(
        module.settings.project_path, module.settings.exclude_dirs
    )
    index, graph = module.index_cache.get(module.settings.project_path, project_files)
    chunks = chunk_index(index, project_files)
    order = semantic_rank(question, embed_chunks(chunks, FakeEmbedder()), FakeEmbedder())
    lexical = rank_files_detailed(question, index, graph)
    expected = fuse_rankings(lexical, order)
    lexical_paths = [item.path for item in lexical]
    fused_paths = [item.path for item in prepared.ranked]
    beta = next(item for item in prepared.ranked if item.path == Path("beta.c"))
    assert prepared.semantic == "ready"
    assert prepared.ranked == expected
    assert fused_paths.index(Path("alpha.c")) < lexical_paths.index(Path("alpha.c"))
    assert beta.reason == "semantic"
    assert isinstance(beta.semantic, int) and beta.semantic >= 1
    assert Path("beta.c") in prepared.included
    assert any(
        record.levelno == logging.INFO
        and record.message == f"Semantic search: ready ({len(order)} files ranked)."
        for record in caplog.records
    )


def test_a4_full_mode_does_not_build(ai_service, monkeypatch, tmp_path):

    # Arrange
    module, _client = ai_service
    question = "Where is the answer initialized?"

    # Act
    disabled = module.dry_run(question)
    embedder = FakeEmbedder()
    service = _service(tmp_path, embedder)
    monkeypatch.setattr(module, "semantic_service", service)
    enabled = module.dry_run(question)

    # Assert
    assert disabled.mode == "full"
    assert disabled.semantic is None
    assert enabled.mode == "full"
    assert enabled.semantic == "off"
    assert embedder.document_calls == []
    assert embedder.query_calls == []
    assert _build_threads() == []


def test_a5_warm_up_and_token_partial(ai_service, monkeypatch, tmp_path, caplog):

    # Arrange / Act / Assert: under the pre-check, warm_up does not build
    module, client = ai_service
    caplog.set_level(logging.INFO)
    under = _service(tmp_path, FakeEmbedder())
    monkeypatch.setattr(module, "semantic_service", under)
    caplog.clear()
    module.warm_up()
    assert under.wait_idle(timeout=5) is True
    assert under.progress() is None
    assert not any(thread.name == "semantic-build" for thread in _build_threads())
    assert any(
        record.levelno == logging.INFO
        and record.message.startswith("Index warm-up done: ")
        and " files in " in record.message
        and record.message.endswith(" ms.")
        for record in caplog.records
    )
    assert not any(
        "Semantic index: building in the background." == record.message
        for record in caplog.records
    )

    # Arrange / Act / Assert: over the pre-check, warm_up returns before embed finishes
    monkeypatch.setattr(module, "settings", _settings(module, 10_000))
    _write_partial_project(tmp_path)
    blocked = BlockingEmbedder()
    over = _service(tmp_path, blocked)
    monkeypatch.setattr(module, "semantic_service", over)
    caplog.clear()
    try:
        module.warm_up()
        assert blocked.entered.wait(timeout=5)
        assert blocked.release.is_set() is False
        assert over.progress() is not None
    finally:
        blocked.release.set()
    assert over.wait_idle(timeout=5) is True
    assert any(
        record.message.startswith("Index warm-up done: ")
        and record.message.endswith(" ms.")
        for record in caplog.records
    )
    assert any(
        record.levelno == logging.INFO
        and record.message == "Semantic index: building in the background."
        for record in caplog.records
    )

    # Arrange / Act / Assert: partial only by token count starts on the first request
    for path in tmp_path.iterdir():
        if path.name == "main.c":
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    main = tmp_path / "main.c"
    if not main.exists():
        main.write_text("void Can_Init(void)\n{\n    return;\n}\n", encoding="utf-8")
    monkeypatch.setattr(module, "settings", _settings(module, 3_000))
    counted = FakeEmbedder()
    token_service = _service(tmp_path, counted)
    monkeypatch.setattr(module, "semantic_service", token_service)

    def count_tokens(*, model, contents):
        result = MagicMock()
        result.total_tokens = 10 if "Partial context:" in contents else 3_001
        return result

    client.models.count_tokens.side_effect = count_tokens
    module.warm_up()
    assert counted.document_calls == []
    module.dry_run(QUESTION, exact=True)
    assert token_service.wait_idle(timeout=5) is True
    assert counted.document_calls


def test_a6_failure_keeps_the_lexical_answer(ai_service, monkeypatch, tmp_path, caplog):

    # Arrange
    module, client = ai_service
    _partial(module, monkeypatch, tmp_path)
    client.models.generate_content.return_value.text = "stub answer"
    lexical = module.generate_answer(QUESTION)
    lexical_prompt = module.dry_run(QUESTION).prompt
    clock = [0.0]
    embedder = AlwaysFailEmbedder()
    service = _service(tmp_path, embedder, clock=lambda: clock[0])
    monkeypatch.setattr(module, "semantic_service", service)
    caplog.set_level(logging.WARNING)
    caplog.clear()

    # Act / Assert
    first = module.generate_answer(QUESTION)
    assert service.wait_idle(timeout=5) is True
    assert first.text == lexical.text
    assert first.context == lexical.context
    assert len(_warnings(caplog)) == 1
    second = module.generate_answer(QUESTION)
    assert service.wait_idle(timeout=5) is True
    assert second.text == lexical.text
    assert len(_warnings(caplog)) == 1
    dry = module.dry_run(QUESTION)
    assert dry.mode == "partial"
    assert dry.semantic == "off"
    assert dry.prompt == lexical_prompt
    clock[0] = 300
    third = module.generate_answer(QUESTION)
    assert service.wait_idle(timeout=5) is True
    assert third.text == lexical.text
    assert len(_warnings(caplog)) == 2


def test_a7_ask_shape_unchanged(ai_service, monkeypatch, tmp_path):

    # Arrange
    from app.main import app

    module, client = ai_service
    _partial(module, monkeypatch, tmp_path)
    client.models.generate_content.return_value.text = "stub answer"
    monkeypatch.setattr("app.main.ai_service.warm_up", lambda: None)

    def keys(response):
        body = response.json()
        assert response.status_code == 200
        return set(body), set(body["context"])

    with TestClient(app, raise_server_exceptions=False) as api:
        off_answer = module.generate_answer(QUESTION)
        off_keys = keys(api.post("/ask", json={"question": QUESTION}))

        blocked = BlockingEmbedder()
        building = _service(tmp_path, blocked)
        monkeypatch.setattr(module, "semantic_service", building)
        try:
            building_answer = module.generate_answer(QUESTION)
            assert blocked.entered.wait(timeout=5)
            building_keys = keys(api.post("/ask", json={"question": QUESTION}))
        finally:
            blocked.release.set()
        assert building.wait_idle(timeout=5) is True

        ready_service = _service(tmp_path / "ready", FakeEmbedder())
        monkeypatch.setattr(module, "semantic_service", ready_service)
        module.dry_run(QUESTION)
        assert ready_service.wait_idle(timeout=5) is True
        ready_answer = module.generate_answer(QUESTION)
        ready_keys = keys(api.post("/ask", json={"question": QUESTION}))

    # Assert
    assert off_keys == building_keys == ready_keys == (ASK_KEYS, ASK_CONTEXT_KEYS)
    assert off_answer.context == building_answer.context
    for answer in (off_answer, building_answer, ready_answer):
        assert set(answer.context.__dataclass_fields__) == ASK_CONTEXT_KEYS
