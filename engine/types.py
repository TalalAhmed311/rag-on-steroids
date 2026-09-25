"""Shared query / hit types used by every retriever and the bench harness."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


@dataclass
class Query:
    id: str
    text: Optional[str] = None
    vector: Optional[np.ndarray] = None


@dataclass
class Hit:
    doc_id: int
    score: float
    source: str  # "dense", "bm25", "fused", "reranked"


class Retriever(Protocol):
    def search(self, query: Query, k: int) -> list[Hit]: ...

    def stats(self) -> dict: ...
