import hashlib
import re
from collections.abc import Sequence

import numpy as np


class FakeEmbedder:
    def __init__(self, dims: int = 64, model: str = "fake-bow") -> None:
        self._model_name = model
        self._dims = dims
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def loaded(self) -> bool:
        return True

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        items = list(texts)
        self.document_calls.append(items)
        if not items:
            return np.zeros((0, self._dims), dtype=np.float32)
        return np.vstack([self._vector(text) for text in items])

    def embed_query(self, text: str) -> np.ndarray:
        self.query_calls.append(text)
        return self._vector(text)

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dims, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            if len(token) < 2:
                continue
            index = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % self._dims
            vector[index] += np.float32(1.0)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return (vector / np.float32(norm)).astype(np.float32)
