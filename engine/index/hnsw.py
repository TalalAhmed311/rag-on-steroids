"""Single HNSW graph in RAM (naive baseline). Uses hnswlib."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from engine.types import Hit, Query


class HNSWIndex:
    def __init__(
        self,
        vectors: Optional[np.ndarray] = None,
        space: str = "ip",
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
        index_path: Optional[Union[str, Path]] = None,
    ):
        import hnswlib

        self.space = space
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._index = None
        self.n = 0
        self.dim = 0

        if index_path is not None:
            self.load(index_path)
            return
        if vectors is None:
            raise ValueError("provide vectors or index_path")
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.n, self.dim = vectors.shape
        self._index = hnswlib.Index(space=space, dim=self.dim)
        self._index.init_index(max_elements=self.n, ef_construction=ef_construction, M=M)
        self._index.add_items(vectors, np.arange(self.n))
        self._index.set_ef(ef_search)

    def set_ef(self, ef: int) -> None:
        self.ef_search = ef
        if self._index is not None:
            self._index.set_ef(ef)

    def search(self, query: Query, k: int) -> list[Hit]:
        if query.vector is None:
            raise ValueError("HNSWIndex requires query.vector")
        k = min(k, self.n)
        q = np.ascontiguousarray(query.vector.reshape(1, -1), dtype=np.float32)
        labels, distances = self._index.knn_query(q, k=k)
        # hnswlib: for 'ip' space, distance = 1 - cosine for normalized? Actually for space='ip'
        # the returned distance is inner product distance depending on version.
        # We expose raw library scores as-is; higher IP is better when space=ip with cosine.
        hits = []
        for lab, dist in zip(labels[0], distances[0]):
            # For space='ip', hnswlib returns distance = 1 - ip for cosine-normalized in some builds;
            # convert to similarity-like score: use -dist so sorting by score desc works if needed.
            score = float(-dist) if self.space in ("l2", "cosine") else float(-dist)
            hits.append(Hit(doc_id=int(lab), score=score, source="dense"))
        return hits

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._index.save_index(str(path))
        meta = path.with_suffix(path.suffix + ".meta.npz")
        np.savez(meta, n=self.n, dim=self.dim, M=self.M, ef_construction=self.ef_construction,
                 ef_search=self.ef_search, space=np.array(self.space))

    def load(self, path: Union[str, Path]) -> None:
        import hnswlib

        path = Path(path)
        meta = np.load(str(path) + ".meta.npz", allow_pickle=True)
        self.n = int(meta["n"])
        self.dim = int(meta["dim"])
        self.M = int(meta["M"])
        self.ef_construction = int(meta["ef_construction"])
        self.ef_search = int(meta["ef_search"])
        self.space = str(meta["space"])
        self._index = hnswlib.Index(space=self.space, dim=self.dim)
        self._index.load_index(str(path), max_elements=self.n)
        self._index.set_ef(self.ef_search)

    def stats(self) -> dict:
        return {
            "index": "hnsw",
            "n": self.n,
            "dim": self.dim,
            "M": self.M,
            "ef_search": self.ef_search,
            "ef_construction": self.ef_construction,
        }
