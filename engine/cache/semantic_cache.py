"""Semantic cache: return cached results if query embedding is close enough."""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from engine.types import Hit, Query


class SemanticCache:
    def __init__(self, threshold: float = 0.95, max_entries: int = 5_000):
        self.threshold = threshold
        self.max_entries = max_entries
        self._vectors: Optional[np.ndarray] = None
        self._payloads: list[list[Hit]] = []
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, query: Query) -> Optional[list[Hit]]:
        if query.vector is None or self._vectors is None or len(self._payloads) == 0:
            self.misses += 1
            return None
        q = query.vector.astype(np.float32).ravel()
        scores = self._vectors @ q
        j = int(scores.argmax())
        if float(scores[j]) >= self.threshold:
            self.hits += 1
            return list(self._payloads[j])
        self.misses += 1
        return None

    def put(self, query: Query, hits: list[Hit]) -> None:
        if query.vector is None:
            return
        v = query.vector.astype(np.float32).ravel().reshape(1, -1)
        with self._lock:
            if self._vectors is None:
                self._vectors = v.copy()
            else:
                self._vectors = np.vstack([self._vectors, v])
            self._payloads.append(list(hits))
            if len(self._payloads) > self.max_entries:
                self._vectors = self._vectors[1:]
                self._payloads = self._payloads[1:]

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "semantic_cache_hit_rate": (self.hits / total) if total else 0.0,
            "semantic_cache_entries": len(self._payloads),
            "threshold": self.threshold,
        }
