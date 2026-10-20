from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.chunker import unique_texts
from app.embedder import TextEmbedder
from app.models import Chunk


@dataclass(frozen=True)
class EmbeddedChunks:
    chunks: tuple[Chunk, ...]  # scoped chunks, may repeat keys
    keys: tuple[str, ...]  # unique keys, row order
    matrix: np.ndarray  # (len(keys), dims) float32, L2-normalized


def embed_chunks(chunks: Sequence[Chunk], embedder: TextEmbedder) -> EmbeddedChunks:
    scoped = tuple(chunks)
    if not scoped:
        dims = int(getattr(embedder, "dims", 0))
        return EmbeddedChunks(
            chunks=(),
            keys=(),
            matrix=np.zeros((0, dims), dtype=np.float32),
        )
    texts = unique_texts(scoped)
    keys = tuple(texts)
    matrix = np.asarray(
        embedder.embed_documents([texts[key] for key in keys]),
        dtype=np.float32,
    )
    return EmbeddedChunks(chunks=scoped, keys=keys, matrix=matrix)


def semantic_scores(
    question: str, embedded: EmbeddedChunks, embedder: TextEmbedder
) -> dict[Path, float]:
    if not embedded.chunks or embedded.matrix.shape[0] == 0:
        return {}
    query = np.asarray(embedder.embed_query(question), dtype=np.float32).reshape(-1)
    cosines = embedded.matrix @ query
    index_of = {key: index for index, key in enumerate(embedded.keys)}
    best: dict[Path, float] = {}
    for chunk in embedded.chunks:
        score = float(cosines[index_of[chunk.key]])
        current = best.get(chunk.path)
        if current is None or score > current:
            best[chunk.path] = score
    return best


def semantic_rank(
    question: str, embedded: EmbeddedChunks, embedder: TextEmbedder
) -> tuple[Path, ...]:
    scores = semantic_scores(question, embedded, embedder)
    ordered = sorted(scores, key=lambda path: (-scores[path], path.parts))
    return tuple(ordered)
