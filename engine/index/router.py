"""Centroid router: pick nprobe partitions for a query."""
from __future__ import annotations

from typing import Optional

import numpy as np


class Router:
    def __init__(self, centroids: np.ndarray):
        self.centroids = np.ascontiguousarray(centroids, dtype=np.float32)
        self.k, self.dim = self.centroids.shape

    def route(self, query: np.ndarray, nprobe: int, adaptive_gap: Optional[float] = None) -> np.ndarray:
        q = query.astype(np.float32).ravel()
        # Higher IP = closer for normalized vectors
        scores = self.centroids @ q
        order = np.argsort(-scores)
        if adaptive_gap is None:
            return order[:nprobe]
        chosen = [int(order[0])]
        for i in range(1, len(order)):
            if len(chosen) >= nprobe:
                break
            gap = scores[order[i - 1]] - scores[order[i]]
            chosen.append(int(order[i]))
            if gap > adaptive_gap and len(chosen) >= 1:
                # keep going until gap large AND we have at least 1; stop after adding when gap big
                if gap > adaptive_gap and len(chosen) >= max(1, nprobe // 4):
                    break
        return np.asarray(chosen[:nprobe], dtype=np.int64)
