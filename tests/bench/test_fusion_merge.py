"""Smoke tests for fusion / merge (no embeddings required)."""
from engine.fusion.rrf import rrf, weighted_fusion
from engine.index.merge import merge_hits
from engine.types import Hit


def test_merge_dedup_prefers_higher_score():
    a = [Hit(1, 0.5, "dense"), Hit(2, 0.9, "dense")]
    b = [Hit(1, 0.8, "dense"), Hit(3, 0.7, "dense")]
    out = merge_hits([a, b], k=3)
    by_id = {h.doc_id: h.score for h in out}
    assert by_id[1] == 0.8
    assert set(by_id) == {1, 2, 3}


def test_rrf_orders_shared_docs_higher():
    a = [Hit(1, 1.0, "dense"), Hit(2, 0.9, "dense")]
    b = [Hit(1, 1.0, "bm25"), Hit(3, 0.9, "bm25")]
    out = rrf([a, b], k=60, top_k=3)
    assert out[0].doc_id == 1


def test_weighted_fusion():
    dense = [Hit(1, 10.0, "dense"), Hit(2, 0.0, "dense")]
    sparse = [Hit(2, 5.0, "bm25"), Hit(1, 0.0, "bm25")]
    out = weighted_fusion(dense, sparse, 0.5, 0.5, top_k=2)
    assert {h.doc_id for h in out} == {1, 2}
