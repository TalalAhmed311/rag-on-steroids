"""Prefetch strategies for partition loading."""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from engine.storage.graph_manager import GraphManager


class Prefetcher:
    def __init__(
        self,
        manager: GraphManager,
        centroids: np.ndarray,
        strategy: str = "none",
        neighbour_k: int = 2,
    ):
        self.manager = manager
        self.centroids = centroids
        self.strategy = strategy
        self.neighbour_k = neighbour_k
        self.coaccess: dict[int, dict[int, int]] = {}
        self.wasted = 0
        self.prefetched = 0
        self._lock = threading.Lock()

    def on_probe(self, probed: list[int]) -> None:
        if self.strategy == "none":
            return
        # update co-access
        with self._lock:
            for a in probed:
                for b in probed:
                    if a == b:
                        continue
                    self.coaccess.setdefault(a, {})
                    self.coaccess[a][b] = self.coaccess[a].get(b, 0) + 1

        targets: set[int] = set()
        if self.strategy in ("neighbour", "both"):
            for p in probed:
                scores = self.centroids @ self.centroids[p]
                order = np.argsort(-scores)
                for j in order[1 : 1 + self.neighbour_k]:
                    targets.add(int(j))
        if self.strategy in ("history", "both"):
            for p in probed:
                related = self.coaccess.get(p, {})
                for q, _cnt in sorted(related.items(), key=lambda x: -x[1])[: self.neighbour_k]:
                    targets.add(q)
        targets -= set(probed)
        for t in targets:
            self.prefetched += 1
            threading.Thread(target=self._bg_load, args=(t,), daemon=True).start()

    def _bg_load(self, partition_id: int) -> None:
        before = self.manager.hits + self.manager.loads
        self.manager.get(partition_id)
        # crude wasted: loaded but we don't know use — leave counter for harness
        _ = before
