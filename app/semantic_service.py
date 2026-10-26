import logging
import threading
import time
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.chunker import chunk_index
from app.config import EmbeddingSettings
from app.embedder import EmbedderUnavailableError, TextEmbedder, build_embedder
from app.models import Chunk, ProjectFile, ProjectIndex
from app.semantic_search import semantic_rank
from app.vector_index import VectorIndex

logger = logging.getLogger(__name__)

RETRY_AFTER_FAILURE_S = 300


@dataclass(frozen=True)
class SemanticResult:
    status: Literal["off", "building", "ready"] | None
    order: tuple[Path, ...] = ()


class SemanticService:
    def __init__(
        self,
        settings: EmbeddingSettings,
        *,
        embedder: TextEmbedder | None = None,
        vector_index: VectorIndex | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._clock = clock
        self._enabled = settings.semantic_search != "off"
        self._embedder = None
        self._vector_index = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._progress: tuple[int, int] | None = None
        self._failed_at: float | None = None
        self._lock = threading.Lock()
        self._chunk_memo: weakref.WeakKeyDictionary[ProjectIndex, tuple[Chunk, ...]] = (
            weakref.WeakKeyDictionary()
        )
        if not self._enabled:
            return
        self._embedder = embedder if embedder is not None else build_embedder(settings)
        self._vector_index = (
            vector_index
            if vector_index is not None
            else VectorIndex(settings.cache_dir, settings.model)
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def prepare(
        self,
        question: str,
        index: ProjectIndex,
        files: Sequence[ProjectFile],
    ) -> SemanticResult:
        try:
            if not self._enabled:
                return SemanticResult(None)
            chunks = self._chunks_for(index, files)
            if not chunks:
                return SemanticResult("off")
            vector_index = self._vector_index
            embedder = self._embedder
            if vector_index is None or embedder is None:
                return SemanticResult("off")
            missing = vector_index.missing(chunks)
            if missing > 0:
                self._ensure_build(chunks)
                with self._lock:
                    running = self._running
                    backoff = self._in_backoff()
                if running:
                    return SemanticResult("building")
                if backoff:
                    return SemanticResult("off")
                missing = vector_index.missing(chunks)
                if missing > 0:
                    return SemanticResult("off")
            embedded = vector_index.lookup(chunks)
            if embedded is None:
                return SemanticResult("off")
            order = semantic_rank(question, embedded, embedder)
            return SemanticResult("ready", order)
        except Exception as exc:
            self._log_request_failure(exc)
            return SemanticResult("off")

    def start_build(self, index: ProjectIndex, files: Sequence[ProjectFile]) -> None:
        if not self._enabled:
            return
        try:
            chunks = self._chunks_for(index, files)
            if not chunks:
                return
            self._ensure_build(chunks)
        except Exception as exc:
            self._log_request_failure(exc)

    def progress(self) -> tuple[int, int] | None:
        with self._lock:
            if not self._running:
                return None
            return self._progress

    def wait_idle(self, timeout: float | None = None) -> bool:
        with self._lock:
            thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def _chunks_for(
        self, index: ProjectIndex, files: Sequence[ProjectFile]
    ) -> tuple[Chunk, ...]:
        with self._lock:
            cached = self._chunk_memo.get(index)
        if cached is not None:
            return cached
        chunks = chunk_index(index, files)
        with self._lock:
            existing = self._chunk_memo.get(index)
            if existing is not None:
                return existing
            self._chunk_memo[index] = chunks
        return chunks

    def _ensure_build(self, chunks: tuple[Chunk, ...]) -> None:
        vector_index = self._vector_index
        if vector_index is None:
            return
        with self._lock:
            if self._running:
                return
            if self._in_backoff():
                return
        # Still run ensure when nothing is missing so a restart logs the
        # reused rows (0 embedded) instead of skipping the build thread.
        missing = vector_index.missing(chunks)
        with self._lock:
            if self._running:
                return
            if self._in_backoff():
                return
            self._progress = (0, missing)
            self._running = True
            thread = threading.Thread(
                target=self._build,
                args=(chunks,),
                name="semantic-build",
                daemon=True,
            )
            self._thread = thread
            thread.start()

    def _build(self, chunks: tuple[Chunk, ...]) -> None:
        vector_index = self._vector_index
        embedder = self._embedder
        if vector_index is None or embedder is None:
            with self._lock:
                self._running = False
                self._thread = None
                self._progress = None
            return
        try:
            result = vector_index.ensure(
                chunks, embedder, progress=self._on_progress
            )
        except Exception as exc:
            self._log_build_failure(exc)
            with self._lock:
                self._failed_at = self._clock()
        else:
            logger.info(
                f"Semantic index ready: {result.live} chunks "
                f"({result.embedded} embedded, {result.reused} reused)."
            )
            with self._lock:
                self._failed_at = None
        finally:
            with self._lock:
                self._running = False
                self._thread = None
                self._progress = None

    def _on_progress(self, done: int, total: int) -> None:
        with self._lock:
            self._progress = (done, total)

    def _in_backoff(self) -> bool:
        if self._failed_at is None:
            return False
        return self._clock() - self._failed_at < RETRY_AFTER_FAILURE_S

    def _log_request_failure(self, exc: Exception) -> None:
        logger.warning(
            f"Semantic search failed ({type(exc).__name__}); using lexical ranking."
        )

    def _log_build_failure(self, exc: Exception) -> None:
        if isinstance(exc, EmbedderUnavailableError):
            detail = f"{type(exc).__name__}: {exc}"
        else:
            detail = type(exc).__name__
        logger.warning(
            f"Semantic index build failed ({detail}); using lexical ranking."
        )
