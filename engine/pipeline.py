"""Compose a Retriever pipeline from YAML config."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from engine.types import Hit, Query


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "name" not in cfg:
        raise ValueError(f"Config {path} must be a mapping with a 'name' field")
    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    blob = yaml.safe_dump(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


class Pipeline:
    def __init__(
        self,
        dense=None,
        sparse=None,
        fusion: str = "rrf",
        fusion_k: int = 60,
        w_dense: float = 0.5,
        w_sparse: float = 0.5,
        reranker=None,
        rerank_depth: int = 0,
        passages: Optional[list[str]] = None,
        result_cache=None,
        semantic_cache=None,
        rescorer=None,
        rescored_candidates: int = 0,
        candidate_k: int = 100,
    ):
        self.dense = dense
        self.sparse = sparse
        self.fusion = fusion
        self.fusion_k = fusion_k
        self.w_dense = w_dense
        self.w_sparse = w_sparse
        self.reranker = reranker
        self.rerank_depth = rerank_depth
        self.passages = passages or []
        self.result_cache = result_cache
        self.semantic_cache = semantic_cache
        self.rescorer = rescorer
        self.rescored_candidates = rescored_candidates
        self.candidate_k = candidate_k

    def search(self, query: Query, k: int) -> list[Hit]:
        if self.result_cache is not None:
            cached = self.result_cache.get(query)
            if cached is not None:
                return cached[:k]
        if self.semantic_cache is not None:
            cached = self.semantic_cache.get(query)
            if cached is not None:
                return cached[:k]

        fetch_k = max(k, self.candidate_k, self.rerank_depth, self.rescored_candidates or 0)
        dense_hits: list[Hit] = []
        sparse_hits: list[Hit] = []
        if self.dense is not None and query.vector is not None:
            dense_hits = self.dense.search(query, fetch_k)
            if self.rescorer is not None and self.rescored_candidates:
                dense_hits = self.rescorer.rescore(query, dense_hits[: self.rescored_candidates], fetch_k)
        if self.sparse is not None and query.text is not None:
            sparse_hits = self.sparse.search(query, fetch_k)

        if dense_hits and sparse_hits:
            from engine.fusion.rrf import rrf, weighted_fusion

            if self.fusion == "weighted":
                hits = weighted_fusion(dense_hits, sparse_hits, self.w_dense, self.w_sparse, top_k=fetch_k)
            else:
                hits = rrf([dense_hits, sparse_hits], k=self.fusion_k, top_k=fetch_k)
        elif dense_hits:
            hits = dense_hits
        else:
            hits = sparse_hits

        if self.reranker is not None and self.rerank_depth > 0:
            hits = self.reranker.rerank(query, hits[: self.rerank_depth], self.passages, top_k=k)
        else:
            hits = hits[:k]

        if self.result_cache is not None:
            self.result_cache.put(query, hits)
        if self.semantic_cache is not None:
            self.semantic_cache.put(query, hits)
        return hits

    def stats(self) -> dict:
        out: dict[str, Any] = {}
        if self.dense is not None and hasattr(self.dense, "stats"):
            out["dense"] = self.dense.stats()
        if self.sparse is not None and hasattr(self.sparse, "stats"):
            out["sparse"] = self.sparse.stats()
        if self.result_cache is not None:
            out.update(self.result_cache.stats())
        if self.semantic_cache is not None:
            out.update(self.semantic_cache.stats())
        return out


def _load_vectors(path: str | Path) -> np.ndarray:
    path = Path(path)
    return np.load(path, mmap_mode="r")


def build_pipeline(cfg: dict[str, Any], data_root: str | Path = "data") -> Pipeline:
    data_root = Path(data_root)
    dense_cfg = cfg.get("dense") or {}
    sparse_cfg = cfg.get("sparse")
    fusion_cfg = cfg.get("fusion")
    rerank_cfg = cfg.get("rerank")
    cache_cfg = cfg.get("cache")

    vectors_path = dense_cfg.get("vectors_path")
    vectors = _load_vectors(vectors_path) if vectors_path else None

    dense = None
    index_kind = (dense_cfg.get("index") or "flat") if dense_cfg else None
    if index_kind == "flat" and vectors is not None:
        from engine.index.flat import FlatIndex

        dense = FlatIndex(np.asarray(vectors), metric=dense_cfg.get("metric", "ip"))
    elif index_kind == "hnsw" and vectors is not None:
        from engine.index.hnsw import HNSWIndex

        hcfg = dense_cfg.get("hnsw") or {}
        path = dense_cfg.get("index_path")
        if path and Path(path).exists():
            dense = HNSWIndex(index_path=path)
            dense.set_ef(hcfg.get("ef_search", 64))
        else:
            dense = HNSWIndex(
                np.asarray(vectors),
                space=hcfg.get("space", "ip"),
                M=hcfg.get("M", 16),
                ef_construction=hcfg.get("ef_construction", 200),
                ef_search=hcfg.get("ef_search", 64),
            )
            if dense_cfg.get("save_path"):
                dense.save(dense_cfg["save_path"])
    elif index_kind == "partitioned":
        from engine.index.partitioned import PartitionedIndex
        from engine.storage.graph_manager import GraphManager

        pcfg = dense_cfg
        index_dir = pcfg.get("index_dir")
        hcfg = pcfg.get("hnsw") or {}
        if index_dir and Path(index_dir).exists():
            storage = (pcfg.get("storage") or {})
            lazy = storage.get("mode") == "managed"
            dense = PartitionedIndex.load(
                index_dir,
                nprobe=pcfg.get("nprobe", 8),
                ef_search=hcfg.get("ef_search", 64),
                lazy=lazy,
            )
        elif vectors is not None:
            dense = PartitionedIndex.build(
                np.asarray(vectors),
                n_partitions=pcfg.get("partitions", 256),
                overlap=pcfg.get("overlap", 1),
                nprobe=pcfg.get("nprobe", 8),
                M=hcfg.get("M", 16),
                ef_construction=hcfg.get("ef_construction", 200),
                ef_search=hcfg.get("ef_search", 64),
            )
            if pcfg.get("save_dir"):
                dense.save(pcfg["save_dir"])
            storage = (pcfg.get("storage") or {})
        else:
            storage = (pcfg.get("storage") or {})
        if storage.get("mode") == "managed" and pcfg.get("index_dir") and dense is not None:
            budget_gb = float(storage.get("budget_gb", 8))
            gm = GraphManager(
                Path(pcfg["index_dir"]),
                n_partitions=len(dense.local_to_global),
                budget_bytes=int(budget_gb * (1 << 30)),
                policy=storage.get("policy", "lru"),
                ef_search=hcfg.get("ef_search", 64),
            )
            dense.graph_manager = gm
            if storage.get("prefetch") and storage.get("prefetch") != "none":
                from engine.storage.prefetch import Prefetcher

                dense._prefetcher = Prefetcher(
                    gm, dense.router.centroids, strategy=storage.get("prefetch", "neighbour")
                )

    sparse = None
    passages: list[str] = []
    if sparse_cfg:
        from engine.sparse.bm25 import BM25Index
        import pyarrow.parquet as pq

        corpus_path = Path(sparse_cfg.get("corpus_path", data_root / "msmarco/subset_1m/corpus.parquet"))
        table = pq.read_table(corpus_path, columns=["title", "text"])
        passages = [
            f"{(ti or '')} {(tx or '')}".strip()
            for ti, tx in zip(table.column("title").to_pylist(), table.column("text").to_pylist())
        ]
        sparse = BM25Index(passages)

    fusion_name = "rrf"
    fusion_k = 60
    w_dense = w_sparse = 0.5
    if isinstance(fusion_cfg, dict):
        fusion_name = fusion_cfg.get("type", "rrf")
        fusion_k = fusion_cfg.get("k", 60)
        w_dense = fusion_cfg.get("w_dense", 0.5)
        w_sparse = fusion_cfg.get("w_sparse", 0.5)

    reranker = None
    rerank_depth = 0
    if rerank_cfg:
        from engine.rerank.cross_encoder import CrossEncoderReranker

        rerank_depth = int(rerank_cfg.get("depth", 50))
        reranker = CrossEncoderReranker(
            model_name=rerank_cfg.get("model", "BAAI/bge-reranker-base"),
            max_length=rerank_cfg.get("max_length", 256),
            backend=rerank_cfg.get("backend", "torch"),
        )
        if not passages and dense_cfg.get("corpus_path"):
            import pyarrow.parquet as pq

            table = pq.read_table(dense_cfg["corpus_path"], columns=["title", "text"])
            passages = [
                f"{(ti or '')} {(tx or '')}".strip()
                for ti, tx in zip(table.column("title").to_pylist(), table.column("text").to_pylist())
            ]

    result_cache = semantic_cache = None
    if cache_cfg:
        if cache_cfg.get("exact"):
            from engine.cache.result_cache import ResultCache

            result_cache = ResultCache(max_entries=int(cache_cfg.get("exact_size", 10_000)))
        if cache_cfg.get("semantic"):
            from engine.cache.semantic_cache import SemanticCache

            semantic_cache = SemanticCache(threshold=float(cache_cfg.get("semantic_threshold", 0.95)))

    rescorer = None
    rescored_candidates = 0
    if dense_cfg.get("rescoring") and vectors is not None:
        from engine.storage.rescoring import Rescorer

        rescorer = Rescorer(vectors, metric=dense_cfg.get("metric", "ip"))
        rescored_candidates = int(dense_cfg.get("rescored_candidates", 100))

    return Pipeline(
        dense=dense,
        sparse=sparse,
        fusion=fusion_name,
        fusion_k=fusion_k,
        w_dense=w_dense,
        w_sparse=w_sparse,
        reranker=reranker,
        rerank_depth=rerank_depth,
        passages=passages,
        result_cache=result_cache,
        semantic_cache=semantic_cache,
        rescorer=rescorer,
        rescored_candidates=rescored_candidates,
        candidate_k=int(cfg.get("candidate_k", 100)),
    )
