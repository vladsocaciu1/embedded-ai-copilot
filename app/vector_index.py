"""app.vector_index: chunk embeddings on disk, keyed by hash.

The request path does not import this module.
"""

import hashlib
import io
import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.chunker import CHUNK_FORMAT, unique_texts
from app.embedder import EmbedderUnavailableError, TextEmbedder
from app.models import Chunk
from app.semantic_search import EmbeddedChunks

logger = logging.getLogger(__name__)

VECTORS_FORMAT = "vectors-v1"
ENSURE_GROUP = 256
CHECKPOINT_EVERY = 2048
COMPACT_STALE_RATIO = 0.5

_SLUG_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_INDEX_FILES = ("vectors.npy", "keys.json", "meta.json")


def model_slug(model: str) -> str:
    slug = _SLUG_UNSAFE.sub("-", model).strip("-.")
    if not slug or slug in {".", ".."} or "/" in slug or "\\" in slug:
        return "model"
    return slug


def vector_dir(cache_dir: Path, model: str) -> Path:
    return Path(cache_dir) / "vectors" / model_slug(model)


@dataclass(frozen=True)
class EnsureResult:
    live: int
    reused: int
    embedded: int
    stale: int
    compacted: bool
    write_failed: bool
    seconds: float


@dataclass(frozen=True)
class _Snapshot:
    keys: tuple[str, ...]
    matrix: np.ndarray
    positions: dict[str, int]

    @staticmethod
    def empty() -> "_Snapshot":
        return _Snapshot.create((), np.zeros((0, 0), dtype=np.float32))

    @staticmethod
    def create(keys: tuple[str, ...], matrix: np.ndarray) -> "_Snapshot":
        stored = np.array(matrix, dtype=np.float32, copy=True)
        return _Snapshot(
            keys=keys,
            matrix=stored,
            positions={key: index for index, key in enumerate(keys)},
        )

    @property
    def dims(self) -> int:
        return int(self.matrix.shape[1])

    def with_rows(self, keys: tuple[str, ...], matrix: np.ndarray) -> "_Snapshot":
        if not self.keys:
            return _Snapshot.create(keys, matrix)
        stacked = np.vstack([self.matrix, matrix])
        return _Snapshot.create(self.keys + keys, stacked)


class _Corrupt(Exception):
    pass


def _at_least_one(value: object, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return default
    return value


def _optional_dims(embedder: TextEmbedder) -> int | None:
    value = getattr(embedder, "dims", None)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _write_temp_file(path: Path, data: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _payloads(
    snap: _Snapshot, model: str, chunk_format: str
) -> tuple[bytes, bytes, bytes]:
    keys_bytes = json.dumps(list(snap.keys)).encode("utf-8")
    buffer = io.BytesIO()
    np.save(buffer, snap.matrix, allow_pickle=False)
    vectors_bytes = buffer.getvalue()
    meta = {
        "format": VECTORS_FORMAT,
        "model": model,
        "dims": int(snap.matrix.shape[1]),
        "chunk_format": chunk_format,
        "count": len(snap.keys),
        "keys_sha256": hashlib.sha256(keys_bytes).hexdigest(),
        "vectors_sha256": hashlib.sha256(vectors_bytes).hexdigest(),
    }
    meta_bytes = json.dumps(meta).encode("utf-8")
    return keys_bytes, vectors_bytes, meta_bytes


def _group_matrix(result: object, rows: int, expected: int | None) -> np.ndarray:
    matrix = np.asarray(result)
    if getattr(matrix, "ndim", None) != 2 or matrix.shape[0] != rows:
        raise EmbedderUnavailableError(
            f"Embedding returned shape {getattr(matrix, 'shape', None)} for {rows} texts."
        )
    width = int(matrix.shape[1])
    if expected is not None and width != expected:
        raise EmbedderUnavailableError(
            f"Embedding returned {width} dims, expected {expected}."
        )
    return np.array(matrix, dtype=np.float32, copy=True)


class VectorIndex:
    def __init__(
        self, cache_dir: Path, model: str, chunk_format: str = CHUNK_FORMAT
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._model = model
        self._chunk_format = chunk_format
        self._directory = vector_dir(self._cache_dir, model)
        self._snapshot = _Snapshot.empty()
        self._loaded = False
        self._load_lock = threading.Lock()
        self._ensure_lock = threading.Lock()
        self._write_warned = False

    @property
    def directory(self) -> Path:
        return self._directory

    def __len__(self) -> int:
        self._load()
        return len(self._snapshot.keys)

    def missing(self, chunks: Sequence[Chunk]) -> int:
        self._load()
        snap = self._snapshot
        return sum(1 for key in unique_texts(chunks) if key not in snap.positions)

    def lookup(self, chunks: Sequence[Chunk]) -> EmbeddedChunks | None:
        self._load()
        snap = self._snapshot
        scoped = tuple(chunks)
        if not scoped:
            return EmbeddedChunks(
                chunks=(),
                keys=(),
                matrix=np.zeros((0, snap.dims), dtype=np.float32),
            )
        texts = unique_texts(scoped)
        if any(key not in snap.positions for key in texts):
            return None
        indices = [snap.positions[key] for key in texts]
        matrix = np.array(snap.matrix[indices], dtype=np.float32, copy=True)
        return EmbeddedChunks(chunks=scoped, keys=tuple(texts), matrix=matrix)

    def ensure(
        self,
        chunks: Sequence[Chunk],
        embedder: TextEmbedder,
        progress: Callable[[int, int], None] | None = None,
    ) -> EnsureResult:
        started = time.perf_counter()
        if len(chunks) == 0:
            return EnsureResult(
                live=0,
                reused=0,
                embedded=0,
                stale=0,
                compacted=False,
                write_failed=False,
                seconds=time.perf_counter() - started,
            )

        with self._ensure_lock:
            self._write_warned = False
            self._load()
            snap = self._reject_dims(self._snapshot, embedder)
            texts = unique_texts(chunks)
            live_keys = tuple(texts)
            live_set = set(live_keys)
            missing = [key for key in live_keys if key not in snap.positions]
            reused = len(live_keys) - len(missing)
            group_size = _at_least_one(
                getattr(embedder, "group_size", ENSURE_GROUP), ENSURE_GROUP
            )
            checkpoint_every = _at_least_one(
                getattr(embedder, "checkpoint_every", CHECKPOINT_EVERY),
                CHECKPOINT_EVERY,
            )
            embedder_dims = _optional_dims(embedder)
            expected = embedder_dims
            if expected is None and snap.keys:
                expected = snap.dims
            embedded = 0
            since_checkpoint = 0
            pending = False
            write_failed = False
            total = len(missing)

            try:
                for start in range(0, len(missing), group_size):
                    group = missing[start : start + group_size]
                    result = embedder.embed_documents([texts[key] for key in group])
                    matrix = _group_matrix(result, len(group), expected)
                    if expected is None:
                        expected = int(matrix.shape[1])
                    snap = snap.with_rows(tuple(group), matrix)
                    self._snapshot = snap
                    embedded += len(group)
                    since_checkpoint += len(group)
                    pending = True
                    if progress is not None:
                        progress(embedded, total)
                    if since_checkpoint >= checkpoint_every:
                        if not self._persist(snap):
                            write_failed = True
                        since_checkpoint = 0
                        pending = False
            except Exception:
                if pending:
                    if not self._persist(self._snapshot):
                        write_failed = True
                failed_stale = sum(
                    1 for key in self._snapshot.keys if key not in live_set
                )
                self._log_ensure(
                    len(live_keys), reused, embedded, failed_stale, False, started
                )
                raise

            if pending:
                if not self._persist(snap):
                    write_failed = True

            stale = sum(1 for key in snap.keys if key not in live_set)
            compacted = False
            if live_keys and stale > len(live_keys) * COMPACT_STALE_RATIO:
                keep = tuple(key for key in snap.keys if key in live_set)
                indices = [snap.positions[key] for key in keep]
                kept = np.array(snap.matrix[indices], dtype=np.float32, copy=True)
                snap = _Snapshot.create(keep, kept)
                self._snapshot = snap
                compacted = True
                stale = 0
                if not self._persist(snap):
                    write_failed = True

            seconds = self._log_ensure(
                len(live_keys), reused, embedded, stale, compacted, started
            )
            return EnsureResult(
                live=len(live_keys),
                reused=reused,
                embedded=embedded,
                stale=stale,
                compacted=compacted,
                write_failed=write_failed,
                seconds=seconds,
            )

    def _log_ensure(
        self,
        live: int,
        reused: int,
        embedded: int,
        stale: int,
        compacted: bool,
        started: float,
    ) -> float:
        seconds = time.perf_counter() - started
        compacted_word = "yes" if compacted else "no"
        logger.info(
            f"Vector index {model_slug(self._model)}: {live} chunks, "
            f"{reused} reused, {embedded} embedded, {stale} stale, "
            f"compacted {compacted_word}, {seconds:.1f} s."
        )
        return seconds

    def _reject_dims(self, snap: _Snapshot, embedder: TextEmbedder) -> _Snapshot:
        embedder_dims = _optional_dims(embedder)
        if embedder_dims is None or not snap.keys or snap.dims == embedder_dims:
            return snap
        logger.warning(
            f"Vector index {self.directory} ignored (incompatible dims); rebuilding."
        )
        empty = _Snapshot.empty()
        self._snapshot = empty
        return empty

    def _load(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            self._snapshot = self._read_snapshot()
            self._loaded = True

    def _read_snapshot(self) -> _Snapshot:
        directory = self.directory
        if not directory.exists():
            return _Snapshot.empty()
        if not directory.is_dir():
            return self._ignored("not a directory")
        _delete_temps(directory)
        paths = [directory / name for name in _INDEX_FILES]
        present = [path.is_file() for path in paths]
        if not any(present):
            return _Snapshot.empty()
        if not all(present):
            return self._ignored("incomplete index")
        try:
            return self._parse_index(paths[0], paths[1], paths[2])
        except _Corrupt as exc:
            return self._ignored(str(exc))

    def _ignored(self, reason: str) -> _Snapshot:
        logger.warning(
            f"Vector index {self.directory} ignored ({reason}); rebuilding."
        )
        return _Snapshot.empty()

    def _parse_index(
        self, vectors_path: Path, keys_path: Path, meta_path: Path
    ) -> _Snapshot:
        try:
            meta_bytes = meta_path.read_bytes()
            keys_bytes = keys_path.read_bytes()
            vectors_bytes = vectors_path.read_bytes()
        except OSError as exc:
            raise _Corrupt("unreadable index") from exc

        try:
            meta = json.loads(meta_bytes)
        except json.JSONDecodeError as exc:
            raise _Corrupt("invalid meta.json") from exc
        if not isinstance(meta, dict):
            raise _Corrupt("invalid meta.json")

        try:
            keys = json.loads(keys_bytes)
        except json.JSONDecodeError as exc:
            raise _Corrupt("invalid keys.json") from exc

        if meta.get("format") != VECTORS_FORMAT:
            raise _Corrupt("incompatible format")
        if meta.get("model") != self._model:
            raise _Corrupt("incompatible model")
        if meta.get("chunk_format") != self._chunk_format:
            raise _Corrupt("incompatible chunk_format")

        dims = meta.get("dims")
        count = meta.get("count")
        keys_hash = meta.get("keys_sha256")
        vectors_hash = meta.get("vectors_sha256")
        if (
            isinstance(dims, bool)
            or not isinstance(dims, int)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or not isinstance(keys_hash, str)
            or not isinstance(vectors_hash, str)
            or dims < 0
            or count < 0
        ):
            raise _Corrupt("invalid meta.json")

        if hashlib.sha256(keys_bytes).hexdigest() != keys_hash:
            raise _Corrupt("hash mismatch")
        if hashlib.sha256(vectors_bytes).hexdigest() != vectors_hash:
            raise _Corrupt("hash mismatch")

        if not isinstance(keys, list):
            raise _Corrupt("invalid keys.json")
        if any(not isinstance(key, str) for key in keys):
            raise _Corrupt("non-string keys")
        if len(keys) != count:
            raise _Corrupt("count mismatch")
        if len(set(keys)) != len(keys):
            raise _Corrupt("duplicate keys")

        try:
            matrix = np.load(io.BytesIO(vectors_bytes), allow_pickle=False)
        except Exception as exc:
            raise _Corrupt("unreadable vectors.npy") from exc

        if matrix.dtype != np.float32:
            raise _Corrupt("dtype")
        if matrix.ndim != 2:
            raise _Corrupt("ndim")
        if matrix.shape != (count, dims):
            raise _Corrupt("shape")
        return _Snapshot.create(tuple(keys), matrix)

    def _persist(self, snap: _Snapshot) -> bool:
        directory = self.directory
        temps: list[Path] = []
        try:
            directory.mkdir(parents=True, exist_ok=True)
            keys_bytes, vectors_bytes, meta_bytes = _payloads(
                snap, self._model, self._chunk_format
            )
            pid = os.getpid()
            payloads = (
                ("vectors.npy", vectors_bytes),
                ("keys.json", keys_bytes),
                ("meta.json", meta_bytes),
            )
            for index, (name, data) in enumerate(payloads):
                temp = directory / f"{name}.tmp-{pid}-{index}"
                temps.append(temp)
                _write_temp_file(temp, data)
            if any(not temp.is_file() for temp in temps):
                raise OSError("temp file missing before replace")
            for (name, _data), temp in zip(payloads, temps, strict=True):
                os.replace(temp, directory / name)
            return True
        except Exception:
            for temp in temps:
                try:
                    temp.unlink(missing_ok=True)
                except OSError:
                    pass
            if not self._write_warned:
                logger.warning(
                    f"Vector index {directory} write failed; keeping rows in memory."
                )
                self._write_warned = True
            return False


def _delete_temps(directory: Path) -> None:
    try:
        leftovers = list(directory.glob("*.tmp-*"))
    except OSError:
        return
    for path in leftovers:
        try:
            if path.is_file():
                path.unlink()
        except OSError:
            continue
