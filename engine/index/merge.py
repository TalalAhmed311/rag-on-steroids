"""Merge hits from multiple partitions with top-k heap + dedup."""
from __future__ import annotations

import heapq
from typing import Iterable

from engine.types import Hit


def merge_hits(hit_lists: Iterable[list[Hit]], k: int) -> list[Hit]:
    best: dict[int, Hit] = {}
    for hits in hit_lists:
        for h in hits:
            prev = best.get(h.doc_id)
            if prev is None or h.score > prev.score:
                best[h.doc_id] = h
    # top-k by score
    top = heapq.nlargest(k, best.values(), key=lambda h: h.score)
    return top
