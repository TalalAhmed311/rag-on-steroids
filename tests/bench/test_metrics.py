"""Unit tests for bench.metrics — hand-computed examples (Stage 0 gate)."""
from __future__ import annotations

import math

import pytest

from bench.metrics import (
    latency_stats,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    recall_at_k_qrels,
)


def test_recall_at_k_ann_perfect():
    # true top-3 = {1,2,3}; retrieved same order → 3/3
    assert recall_at_k([1, 2, 3, 9], [1, 2, 3], k=3) == 1.0


def test_recall_at_k_ann_partial():
    # 2 of top-3 truth in retrieved top-3 → 2/3
    assert recall_at_k([1, 9, 2], [1, 2, 3], k=3) == pytest.approx(2 / 3)


def test_precision_at_k():
    # 2 relevant in top-4 → 0.5
    assert precision_at_k([10, 11, 12, 13], {11, 13, 99}, k=4) == 0.5


def test_recall_at_k_qrels():
    # relevant={1,2,3}; retrieved top-2 hits one → 1/3
    assert recall_at_k_qrels([1, 9], {1, 2, 3}, k=2) == pytest.approx(1 / 3)


def test_mrr_at_10():
    # first relevant at rank 4 → 1/4
    assert mrr_at_k([9, 8, 7, 1, 2], {1, 2}, k=10) == 0.25
    assert mrr_at_k([9, 8, 7], {1}, k=10) == 0.0


def test_ndcg_at_k_hand():
    # retrieved grades: 3, 0  → DCG = 7/log2(2) + 0 = 7
    # ideal grades [3,1] → 7 + 1/log2(3)
    relevance = {1: 3.0, 2: 1.0}
    retrieved = [1, 99]
    dcg = (2**3 - 1) / math.log2(2) + 0.0
    idcg = (2**3 - 1) / math.log2(2) + (2**1 - 1) / math.log2(3)
    assert ndcg_at_k(retrieved, relevance, k=2) == pytest.approx(dcg / idcg)


def test_latency_stats():
    stats = latency_stats([10.0, 20.0, 30.0, 40.0, 50.0])
    assert stats["mean"] == pytest.approx(30.0)
    assert stats["p50"] == pytest.approx(30.0)
