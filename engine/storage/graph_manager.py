"""Partition cache: keep hot HNSW partitions in RAM under a byte budget."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

from engine.index.hnsw import HNSWIndex
from engine.storage.policies import POLICIES, EvictionPolicy


class GraphManager:
    def __init__(
        self,
        partition_dir: Path,
        n_partitions: int,
        budget_bytes: int,
        policy: str = "lru",
        ef_search: int = 64,
        pinned: Optional[set[int]] = None,
    ):
        self.partition_dir = Path(partition_dir)
        self.n_partitions = n_partitions
        self.budget_bytes = budget_bytes
        self.ef_search = ef_search
        self.pinned = pinned or set()
        self.policy: EvictionPolicy = POLICIES[policy]()
        self._cache: dict[int, HNSWIndex] = {}
        self._meta: dict[int, dict] = {}
        self._used = 0
        self._lock = threading.RLock()
        self.last_was_load = False
        self.loads = 0
        self.hits = 0
        self.evictions = 0

    def _size_of(self, index: HNSWIndex) -> int:
        # rough: n * dim * 4 * 1.5 for graph overhead
        return int(index.n * max(index.dim, 1) * 4 * 1.5) + (1 << 20)

    def get(self, partition_id: int) -> HNSWIndex:
        with self._lock:
            if partition_id in self._cache:
                self.hits += 1
                self._meta[partition_id]["freq"] += 1
                self._meta[partition_id]["last_used"] = time.time()
                self.last_was_load = False
                return self._cache[partition_id]
            self.last_was_load = True
            self.loads += 1
            index = HNSWIndex(index_path=self.partition_dir / f"part_{partition_id:05d}" / "hnsw.bin")
            index.set_ef(self.ef_search)
            size = self._size_of(index)
            while self._used + size > self.budget_bytes and len(self._cache) > 0:
                victim = self.policy.choose_victim(self._meta, self.pinned)
                self._evict(victim)
            self._cache[partition_id] = index
            self._meta[partition_id] = {
                "freq": 1,
                "last_used": time.time(),
                "size_bytes": size,
            }
            self._used += size
            return index

    def _evict(self, partition_id: int) -> None:
        if partition_id in self.pinned:
            return
        idx = self._cache.pop(partition_id, None)
        meta = self._meta.pop(partition_id, None)
        if meta:
            self._used -= meta["size_bytes"]
            self.evictions += 1
        del idx

    def stats(self) -> dict:
        total = self.hits + self.loads
        return {
            "cache_hit_rate": (self.hits / total) if total else 0.0,
            "partition_loads": self.loads,
            "evictions": self.evictions,
            "cached_partitions": len(self._cache),
            "used_bytes": self._used,
            "budget_bytes": self.budget_bytes,
        }
