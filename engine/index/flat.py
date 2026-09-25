"""Exact / flat dense search (FAISS IndexFlatIP on L2-normalized vectors)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from engine.types import Hit, Query


class FlatIndex:
    def __init__(self, vectors: np.ndarray, metric: str = "ip"):
        """
        vectors: (N, D) float32. For BGE, pass L2-normalized vectors and metric='ip'.
        """
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.n, self.dim = self.vectors.shape
        self.metric = metric
        self._index = None
        try:
            import faiss  # type: ignore

            if metric == "ip":
                index = faiss.IndexFlatIP(self.dim)
            else:
                index = faiss.IndexFlatL2(self.dim)
            index.add(self.vectors)
            self._index = index
        except ImportError:
            self._index = None

    def search(self, query: Query, k: int) -> list[Hit]:
        if query.vector is None:
            raise ValueError("FlatIndex requires query.vector")
        k = min(k, self.n)
        q = np.ascontiguousarray(query.vector.reshape(1, -1), dtype=np.float32)
        if self._index is not None:
            scores, idx = self._index.search(q, k)
            return [Hit(doc_id=int(i), score=float(s), source="dense") for i, s in zip(idx[0], scores[0])]
        # NumPy fallback
        if self.metric == "ip":
            scores = (self.vectors @ q.ravel())
        else:
            diff = self.vectors - q.ravel()
            scores = -np.sum(diff * diff, axis=1)
        top = np.argpartition(-scores, kth=k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [Hit(doc_id=int(i), score=float(scores[i]), source="dense") for i in top]

    def stats(self) -> dict:
        return {"index": "flat", "n": self.n, "dim": self.dim, "backend": "faiss" if self._index else "numpy"}
