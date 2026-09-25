"""Full-precision rescoring from on-disk / memmap vectors."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from engine.types import Hit, Query


class Rescorer:
    def __init__(self, vectors: np.ndarray, metric: str = "ip"):
        self.vectors = vectors  # memmap or ndarray (N, D)
        self.metric = metric

    def rescore(self, query: Query, candidates: Sequence[Hit], k: int) -> list[Hit]:
        if query.vector is None:
            raise ValueError("vector required")
        if not candidates:
            return []
        q = query.vector.astype(np.float32).ravel()
        ids = [h.doc_id for h in candidates]
        # grouped / sorted reads
        order = np.argsort(ids)
        sorted_ids = [ids[i] for i in order]
        mats = np.stack([np.asarray(self.vectors[i], dtype=np.float32) for i in sorted_ids])
        if self.metric == "ip":
            scores = mats @ q
        else:
            scores = -np.sum((mats - q) ** 2, axis=1)
        rescored = [
            Hit(doc_id=sorted_ids[j], score=float(scores[j]), source="dense")
            for j in range(len(sorted_ids))
        ]
        rescored.sort(key=lambda h: h.score, reverse=True)
        return rescored[:k]
