"""BM25 sparse retrieval. Uses rank_bm25 (pure Python) for Stage 8 portability.

Tantivy can replace this later for larger postings-on-disk deployments.
"""
from __future__ import annotations

import re
from typing import Optional, Sequence

from engine.types import Hit, Query

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    def __init__(self, texts: Sequence[str]):
        from rank_bm25 import BM25Okapi

        self.corpus_tokens = [tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(self.corpus_tokens)
        self.n = len(texts)

    def search(self, query: Query, k: int) -> list[Hit]:
        if not query.text:
            raise ValueError("BM25 requires query.text")
        scores = self._bm25.get_scores(tokenize(query.text))
        k = min(k, self.n)
        import numpy as np

        top = np.argpartition(-scores, kth=k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [Hit(doc_id=int(i), score=float(scores[i]), source="bm25") for i in top]

    def stats(self) -> dict:
        return {"index": "bm25", "n": self.n, "backend": "rank_bm25"}
