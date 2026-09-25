"""Frozen measurement definitions for Stage 0+.

Do not change metric formulas after Stage 0 is frozen without human approval.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def recall_at_k(retrieved: Sequence[int], relevant: Sequence[int], k: int) -> float:
    """ANN / set recall: |top-k ∩ true| / k  (true set truncated to k if longer)."""
    if k <= 0:
        raise ValueError("k must be positive")
    truth = set(list(relevant)[:k])
    if not truth:
        return 0.0
    # Roadmap ANN definition: |returned top-k ∩ true top-k| / k
    top = list(retrieved)[:k]
    return len(set(top) & truth) / float(k)


def precision_at_k(retrieved: Sequence[int], relevant: Iterable[int], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    rel = set(relevant)
    top = list(retrieved)[:k]
    if not top:
        return 0.0
    return sum(1 for d in top if d in rel) / float(k)


def recall_at_k_qrels(retrieved: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Retrieval recall vs qrels: |top-k ∩ relevant| / |relevant|."""
    rel = set(relevant)
    if not rel:
        return 0.0
    top = set(list(retrieved)[:k])
    return len(top & rel) / float(len(rel))


def mrr_at_k(retrieved: Sequence[int], relevant: Iterable[int], k: int = 10) -> float:
    rel = set(relevant)
    for rank, doc_id in enumerate(list(retrieved)[:k], start=1):
        if doc_id in rel:
            return 1.0 / rank
    return 0.0


def dcg_at_k(grades: Sequence[float], k: int) -> float:
    total = 0.0
    for i, g in enumerate(list(grades)[:k]):
        total += (2.0 ** g - 1.0) / np.log2(i + 2.0)
    return float(total)


def ndcg_at_k(
    retrieved: Sequence[int],
    relevance: dict[int, float],
    k: int = 10,
) -> float:
    """nDCG@k using graded relevance map (missing docs = 0)."""
    grades = [float(relevance.get(int(d), 0.0)) for d in list(retrieved)[:k]]
    dcg = dcg_at_k(grades, k)
    ideal = sorted(relevance.values(), reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def latency_stats(latencies_ms: Sequence[float]) -> dict[str, float]:
    if not latencies_ms:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(arr.mean()),
    }


def mean_metric(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))
