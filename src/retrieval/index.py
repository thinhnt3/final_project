from __future__ import annotations

import numpy as np


class CosineIndex:
    """FAISS inner-product index with a NumPy fallback for portability."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = np.asarray(vectors, dtype="float32")
        try:
            import faiss
            self._index = faiss.IndexFlatIP(self.vectors.shape[1])
            self._index.add(self.vectors)
        except ImportError:
            self._index = None

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        query = np.asarray(query, dtype="float32").reshape(1, -1)
        if self._index is not None:
            scores, indices = self._index.search(query, k)
            return indices[0], scores[0]
        scores = self.vectors @ query[0]
        indices = np.argsort(-scores)[:k]
        return indices, scores[indices]
