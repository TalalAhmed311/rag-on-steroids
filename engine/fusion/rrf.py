"""Reciprocal Rank Fusion and weighted score fusion."""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from engine.types import Hit


def rrf(hit_lists: Sequence[list[Hit]], k: int = 60, top_k: int = 100) -> list[Hit]:
    scores: dict[int, float] = defaultdict(float)
    for hits in hit_lists:
        for rank, h in enumerate(hits, start=1):
            scores[h.doc_id] += 1.0 / (k + rank)
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
    return [Hit(doc_id=i, score=s, source="fused") for i, s in ranked]


def weighted_fusion(
    dense: list[Hit],
    sparse: list[Hit],
    w_dense: float = 0.5,
    w_sparse: float = 0.5,
    top_k: int = 100,
) -> list[Hit]:
    def norm(hits: list[Hit]) -> dict[int, float]:
        if not hits:
            return {}
        vals = [h.score for h in hits]
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0
        return {h.doc_id: (h.score - lo) / span for h in hits}

    d = norm(dense)
    s = norm(sparse)
    ids = set(d) | set(s)
    scored = [
        Hit(doc_id=i, score=w_dense * d.get(i, 0.0) + w_sparse * s.get(i, 0.0), source="fused")
        for i in ids
    ]
    scored.sort(key=lambda h: h.score, reverse=True)
    return scored[:top_k]
