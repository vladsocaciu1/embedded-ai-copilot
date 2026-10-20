import logging
import os
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from app.config import DEFAULT_EMBEDDING_CACHE_DIR, DEFAULT_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 16


class EmbedderUnavailableError(RuntimeError):
    """The embedding model can't be loaded."""


def _text_embedding_class():
    # Imported here so `import app.embedder` does not import fastembed.
    from fastembed import TextEmbedding

    return TextEmbedding


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # A zero row stays zero. Dividing by a 0 norm would be NaN.
    norms = np.where(norms == 0, np.float32(1), norms).astype(np.float32)
    return (vectors / norms).astype(np.float32)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return (vector / np.float32(norm)).astype(np.float32)


class Embedder:
    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        cache_dir: Path | None = None,
        threads: int | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self._model_name = model
        if cache_dir is None:
            cache_dir = DEFAULT_EMBEDDING_CACHE_DIR
        self._cache_dir = cache_dir
        self._threads = threads
        self._batch_size = batch_size
        self._loaded = False
        self._instance = None
        self._lock = threading.Lock()

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def dims(self) -> int:
        return _text_embedding_class().get_embedding_size(self._model_name)

    @property
    def loaded(self) -> bool:
        return self._loaded

    def _thread_count(self) -> int:
        if self._threads is None:
            return max(1, (os.cpu_count() or 2) - 1)
        return self._threads

    def _load(self):
        if self._instance is not None:
            return self._instance
        with self._lock:
            if self._instance is not None:
                return self._instance
            started = time.perf_counter()
            try:
                embedding_cls = _text_embedding_class()
                model_cache = self._cache_dir / "models"
                model_cache.mkdir(parents=True, exist_ok=True)
                instance = embedding_cls(
                    model_name=self._model_name,
                    cache_dir=str(model_cache),
                    threads=self._thread_count(),
                )
            except Exception as exc:
                logger.warning(
                    f"Embedding model {self._model_name} unavailable: {type(exc).__name__}"
                )
                raise EmbedderUnavailableError(
                    f"Embedding model {self._model_name} unavailable: {type(exc).__name__}"
                ) from exc
            logger.info(
                f"Embedding model {self._model_name} loaded in {time.perf_counter() - started:.1f} s."
            )
            self._instance = instance
            self._loaded = True
            return instance

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.dims), dtype=np.float32)
        model = self._load()
        rows = [
            np.asarray(vector, dtype=np.float32).reshape(-1)
            for vector in model.embed(list(texts), batch_size=self._batch_size)
        ]
        if len(rows) != len(texts):
            raise EmbedderUnavailableError(
                f"Embedding model {self._model_name} returned {len(rows)} rows for {len(texts)} texts."
            )
        return _normalize_rows(np.vstack(rows))

    def embed_query(self, text: str) -> np.ndarray:
        model = self._load()
        rows = [
            np.asarray(vector, dtype=np.float32).reshape(-1)
            for vector in model.query_embed(text)
        ]
        if len(rows) != 1:
            raise EmbedderUnavailableError(
                f"Embedding model {self._model_name} returned {len(rows)} rows for one query."
            )
        return _normalize_vector(rows[0])
