"""Exact result cache (normalized query text → hits)."""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Optional

from engine.types import Hit, Query


def normalize_query(text: str) -> str:
    return " ".join(text.lower().split())


class ResultCache:
    def __init__(self, max_entries: int = 10_000):
        self.max_entries = max_entries
        self._store: OrderedDict[str, list[Hit]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, query: Query) -> Optional[list[Hit]]:
        if not query.text:
            return None
        key = normalize_query(query.text)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self.hits += 1
                return list(self._store[key])
            self.misses += 1
            return None

    def put(self, query: Query, hits: list[Hit]) -> None:
        if not query.text:
            return
        key = normalize_query(query.text)
        with self._lock:
            self._store[key] = list(hits)
            self._store.move_to_end(key)
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "result_cache_hit_rate": (self.hits / total) if total else 0.0,
            "result_cache_entries": len(self._store),
        }
