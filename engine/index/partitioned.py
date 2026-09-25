"""Multi-graph partitioned HNSW: k-means assign + one HNSW per partition."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np

from engine.index.hnsw import HNSWIndex
from engine.index.merge import merge_hits
from engine.index.router import Router
from engine.types import Hit, Query


def _kmeans(vectors: np.ndarray, k: int, seed: int = 42, niter: int = 25, sample: int = 200_000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = vectors.shape[0]
    idx = rng.choice(n, size=min(sample, n), replace=False)
    x = np.ascontiguousarray(vectors[idx], dtype=np.float32)
    try:
        import faiss

        kmeans = faiss.Kmeans(x.shape[1], k, niter=niter, verbose=False, seed=seed)
        kmeans.train(x)
        return np.array(kmeans.centroids, dtype=np.float32)
    except Exception:
        # fallback: random centroids refined by a few Lloyd iterations
        cents = x[rng.choice(len(x), size=k, replace=False)].copy()
        for _ in range(niter):
            scores = x @ cents.T
            assign = scores.argmax(axis=1)
            for j in range(k):
                mask = assign == j
                if mask.any():
                    cents[j] = x[mask].mean(axis=0)
                    nrm = np.linalg.norm(cents[j]) + 1e-8
                    cents[j] /= nrm
        return cents.astype(np.float32)


class PartitionedIndex:
    def __init__(
        self,
        partitions: list[HNSWIndex],
        local_to_global: list[np.ndarray],
        centroids: np.ndarray,
        nprobe: int = 8,
        adaptive_gap: Optional[float] = None,
        graph_manager=None,
    ):
        self.partitions = partitions
        self.local_to_global = local_to_global
        self.router = Router(centroids)
        self.nprobe = nprobe
        self.adaptive_gap = adaptive_gap
        self.graph_manager = graph_manager
        self._stats = {"partitions_probed": 0, "partition_loads": 0}

    @classmethod
    def build(
        cls,
        vectors: np.ndarray,
        n_partitions: int = 256,
        overlap: int = 1,
        nprobe: int = 8,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
        seed: int = 42,
    ) -> "PartitionedIndex":
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        centroids = _kmeans(vectors, n_partitions, seed=seed)
        # assign top-overlap centroids
        scores = vectors @ centroids.T  # (N, K) may be large — do in batches
        n = vectors.shape[0]
        assignments: list[list[int]] = [[] for _ in range(n_partitions)]
        bs = 50_000
        for start in range(0, n, bs):
            sl = vectors[start : start + bs]
            sc = sl @ centroids.T
            top = np.argpartition(-sc, kth=min(overlap, n_partitions) - 1, axis=1)[:, :overlap]
            for i, row in enumerate(top):
                for p in row:
                    assignments[int(p)].append(start + i)

        partitions: list[HNSWIndex] = []
        local_to_global: list[np.ndarray] = []
        for p, members in enumerate(assignments):
            ids = np.unique(np.asarray(members, dtype=np.int64))
            if len(ids) == 0:
                # empty — tiny dummy
                ids = np.array([0], dtype=np.int64)
            local_to_global.append(ids)
            part_vecs = vectors[ids]
            partitions.append(
                HNSWIndex(part_vecs, space="ip", M=M, ef_construction=ef_construction, ef_search=ef_search)
            )
        return cls(partitions, local_to_global, centroids, nprobe=nprobe)

    def search(self, query: Query, k: int) -> list[Hit]:
        if query.vector is None:
            raise ValueError("vector required")
        parts = self.router.route(query.vector, self.nprobe, self.adaptive_gap)
        self._stats["partitions_probed"] = len(parts)
        hit_lists = []
        loads = 0
        probed = [int(p) for p in parts]
        for p in probed:
            if self.graph_manager is not None:
                index = self.graph_manager.get(p)
                loads += 1 if self.graph_manager.last_was_load else 0
            else:
                index = self.partitions[p]
                if index is None:
                    raise RuntimeError("lazy partition without graph_manager")
            local_hits = index.search(query, k)
            mapped = [
                Hit(doc_id=int(self.local_to_global[p][h.doc_id]), score=h.score, source="dense")
                for h in local_hits
                if h.doc_id < len(self.local_to_global[p])
            ]
            hit_lists.append(mapped)
        self._stats["partition_loads"] = loads
        if getattr(self, "_prefetcher", None) is not None:
            self._prefetcher.on_probe(probed)
        return merge_hits(hit_lists, k)

    def save(self, out_dir: Union[str, Path]) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "centroids.npy", self.router.centroids)
        meta = {"n_partitions": len(self.partitions), "nprobe": self.nprobe}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        for i, (idx, idmap) in enumerate(zip(self.partitions, self.local_to_global)):
            pdir = out_dir / f"part_{i:05d}"
            pdir.mkdir(exist_ok=True)
            idx.save(pdir / "hnsw.bin")
            np.save(pdir / "idmap.npy", idmap)

    @classmethod
    def load(
        cls,
        out_dir: Union[str, Path],
        nprobe: int = 8,
        ef_search: int = 64,
        lazy: bool = False,
    ) -> "PartitionedIndex":
        """Load partitioned index. If lazy=True, only centroids+idmaps stay in RAM
        (graphs loaded on demand via GraphManager)."""
        out_dir = Path(out_dir)
        centroids = np.load(out_dir / "centroids.npy")
        meta = json.loads((out_dir / "meta.json").read_text())
        idmaps = []
        partitions: list = []
        for i in range(meta["n_partitions"]):
            pdir = out_dir / f"part_{i:05d}"
            idmaps.append(np.load(pdir / "idmap.npy"))
            if lazy:
                partitions.append(None)  # type: ignore
            else:
                h = HNSWIndex(index_path=pdir / "hnsw.bin")
                h.set_ef(ef_search)
                partitions.append(h)
        obj = cls(partitions, idmaps, centroids, nprobe=nprobe)
        obj._lazy = lazy
        obj._index_dir = out_dir
        obj._ef_search = ef_search
        return obj

    def stats(self) -> dict:
        out = {"index": "partitioned", **self._stats, "n_partitions": len(self.local_to_global), "nprobe": self.nprobe}
        if self.graph_manager is not None:
            out["graph_manager"] = self.graph_manager.stats()
        return out
