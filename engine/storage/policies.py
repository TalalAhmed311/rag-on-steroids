"""Eviction policies for the graph manager."""
from __future__ import annotations

from typing import Dict


class EvictionPolicy:
    def choose_victim(self, meta: Dict[int, dict], pinned: set[int]) -> int:
        raise NotImplementedError


class LRU(EvictionPolicy):
    def choose_victim(self, meta: Dict[int, dict], pinned: set[int]) -> int:
        victims = [(m["last_used"], pid) for pid, m in meta.items() if pid not in pinned]
        if not victims:
            raise RuntimeError("no eviction candidate")
        return min(victims)[1]


class LFU(EvictionPolicy):
    def choose_victim(self, meta: Dict[int, dict], pinned: set[int]) -> int:
        victims = [(m["freq"], m["last_used"], pid) for pid, m in meta.items() if pid not in pinned]
        if not victims:
            raise RuntimeError("no eviction candidate")
        return min(victims)[2]


class CostAware(EvictionPolicy):
    """Evict lowest frequency / size_bytes (prefer evicting large rarely-used)."""

    def choose_victim(self, meta: Dict[int, dict], pinned: set[int]) -> int:
        best_pid, best_score = None, None
        for pid, m in meta.items():
            if pid in pinned:
                continue
            score = m["freq"] / max(m["size_bytes"], 1)
            if best_score is None or score < best_score:
                best_score, best_pid = score, pid
        if best_pid is None:
            raise RuntimeError("no eviction candidate")
        return best_pid


POLICIES = {"lru": LRU, "lfu": LFU, "cost": CostAware}
