import threading

import pytest
from fastapi.testclient import TestClient

from app.config import EmbeddingSettings
from app.embedder import EmbedderUnavailableError
from app.semantic_service import SemanticService
from tests.fake_embedder import FakeEmbedder

QUESTION = "What does Can_Init do?"
CONTEXT_KEYS = {
    "mode",
    "limit",
    "prompt_tokens",
    "token_source",
    "prompt_chars",
    "files_total",
    "files_included",
    "map_tier",
    "map_files",
    "ranked_total",
    "ranked",
}
ITEM_KEYS = {"path", "reason", "included", "focus", "tier", "hits", "lexical"}


class BlockingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def embed_documents(self, texts):
        import numpy as np

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


@pytest.fixture
def api_client(ai_service, monkeypatch):
    from app.main import app

    monkeypatch.setattr("app.main.ai_service.warm_up", lambda: None)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def _partial(ai_service, monkeypatch, tmp_path):
    from tests.test_ai_service import _settings_with_limit, _write_partial_project

    module, _client = ai_service
    monkeypatch.setattr(module, "settings", _settings_with_limit(module, 10_000))
    _write_partial_project(tmp_path)
    default = tmp_path / "main.c"
    if default.exists():
        default.unlink()
    return module


def _service(tmp_path, embedder) -> SemanticService:
    return SemanticService(
        EmbeddingSettings(semantic_search="local", cache_dir=tmp_path / "cache"),
        embedder=embedder,
    )


def test_n1_disabled_context_has_no_semantic_keys(api_client, ai_service, monkeypatch, tmp_path):

    # Arrange
    module, client = ai_service
    _partial(ai_service, monkeypatch, tmp_path)
    assert module.semantic_service.enabled is False

    # Act
    response = api_client.post("/context", json={"question": QUESTION})

    # Assert
    assert response.status_code == 200
    body = response.json()
    assert set(body) == CONTEXT_KEYS
    assert "semantic" not in body
    assert body["ranked"]
    for item in body["ranked"]:
        assert set(item) == ITEM_KEYS
        assert "semantic" not in item
    client.models.generate_content.assert_not_called()
    client.models.count_tokens.assert_not_called()


def test_n2_enabled_context_reports_semantic_state(
    api_client, ai_service, monkeypatch, tmp_path
):

    # Arrange
    module, _client = ai_service
    _partial(ai_service, monkeypatch, tmp_path)

    def ranked_semantic(body):
        assert set(body) == CONTEXT_KEYS | {"semantic"}
        assert body["ranked"]
        for item in body["ranked"]:
            assert set(item) == ITEM_KEYS | {"semantic"}
        return body["semantic"], [item["semantic"] for item in body["ranked"]]

    blocked = BlockingEmbedder()
    building = _service(tmp_path, blocked)
    monkeypatch.setattr(module, "semantic_service", building)

    # Act / Assert: building
    try:
        response = api_client.post("/context", json={"question": QUESTION})
        assert blocked.entered.wait(timeout=5)
        status, ranks = ranked_semantic(response.json())
        assert status == "building"
        assert ranks
        assert all(rank is None for rank in ranks)
    finally:
        blocked.release.set()
    assert building.wait_idle(timeout=5) is True

    # Act / Assert: ready
    ready = _service(tmp_path / "ready", FakeEmbedder())
    monkeypatch.setattr(module, "semantic_service", ready)
    module.dry_run(QUESTION)
    assert ready.wait_idle(timeout=5) is True
    response = api_client.post("/context", json={"question": QUESTION})
    status, ranks = ranked_semantic(response.json())
    assert status == "ready"
    assert all(rank is None or isinstance(rank, int) for rank in ranks)
    assert any(isinstance(rank, int) for rank in ranks)

    # Act / Assert: off after a failed build
    failed = _service(tmp_path / "failed", AlwaysFailEmbedder())
    monkeypatch.setattr(module, "semantic_service", failed)
    module.dry_run(QUESTION)
    assert failed.wait_idle(timeout=5) is True
    response = api_client.post("/context", json={"question": QUESTION})
    status, ranks = ranked_semantic(response.json())
    assert status == "off"
    assert all(rank is None for rank in ranks)
