"""OS-managed memory-mapped vector store (Stage 3 baseline)."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

from engine.index.hnsw import HNSWIndex
from engine.types import Hit, Query


class MmapVectorStore:
    def __init__(self, path: Union[str, Path], shape: tuple[int, int]):
        self.path = Path(path)
        self.shape = shape
        self.vectors = np.memmap(self.path, dtype=np.float32, mode="r", shape=shape)

    @staticmethod
    def write(path: Union[str, Path], vectors: np.ndarray) -> "MmapVectorStore":
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        mm = np.memmap(path, dtype=np.float32, mode="w+", shape=vectors.shape)
        mm[:] = vectors
        mm.flush()
        del mm
        return MmapVectorStore(path, vectors.shape)

    def row(self, idx: int) -> np.ndarray:
        return np.array(self.vectors[idx], dtype=np.float32)


class MmapHNSWRetriever:
    """HNSW whose vectors live in a memmap file (graph still in RAM via hnswlib index file)."""

    def __init__(self, hnsw: HNSWIndex, store: MmapVectorStore):
        self.hnsw = hnsw
        self.store = store

    def search(self, query: Query, k: int) -> list[Hit]:
        return self.hnsw.search(query, k)

    def stats(self) -> dict:
        s = self.hnsw.stats()
        s["storage"] = "mmap"
        s["mmap_path"] = str(self.store.path)
        return s
