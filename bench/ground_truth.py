"""Exact kNN ground truth for ANN evaluation (Track S / dense Track Q).

BLOCKED until embeddings exist. This module is the Stage 0 interface; running
`main` without vector files exits with a clear error.

After Stage 0 freeze: do not change saved GT files or this computation without
human approval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def compute_exact_knn(
    corpus: np.ndarray,
    queries: np.ndarray,
    k: int = 100,
    batch_size: int = 64,
    metric: str = "ip",
) -> np.ndarray:
    """Exact top-k. Default metric=ip for L2-normalized BGE embeddings (cosine).

    Returns int64 array of shape (n_queries, k) with corpus indices.
    """
    try:
        import faiss  # type: ignore
    except ImportError:
        faiss = None

    n, dim = corpus.shape
    if queries.shape[1] != dim:
        raise ValueError("query/corpus dim mismatch")
    if k > n:
        raise ValueError(f"k={k} > corpus size {n}")

    corpus_f = np.ascontiguousarray(corpus, dtype=np.float32)
    queries_f = np.ascontiguousarray(queries, dtype=np.float32)

    if faiss is not None:
        if metric == "ip":
            index = faiss.IndexFlatIP(dim)
        else:
            index = faiss.IndexFlatL2(dim)
        index.add(corpus_f)
        _d, idx = index.search(queries_f, k)
        return idx.astype(np.int64)

    # NumPy fallback (slow; fine for tiny smoke tests only)
    out = np.empty((queries.shape[0], k), dtype=np.int64)
    for start in range(0, queries.shape[0], batch_size):
        q = queries_f[start : start + batch_size]
        if metric == "ip":
            scores = q @ corpus_f.T
            part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
            row = np.arange(q.shape[0])[:, None]
            ordered = np.take_along_axis(part, np.argsort(-scores[row, part], axis=1), axis=1)
        else:
            q2 = np.sum(q * q, axis=1, keepdims=True)
            c2 = np.sum(corpus_f * corpus_f, axis=1, keepdims=True).T
            dist = q2 + c2 - 2.0 * (q @ corpus_f.T)
            part = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
            row = np.arange(q.shape[0])[:, None]
            ordered = np.take_along_axis(part, np.argsort(dist[row, part], axis=1), axis=1)
        out[start : start + q.shape[0]] = ordered
    return out


def save_ground_truth(out_dir: Path, indices: np.ndarray, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    npy = out_dir / "neighbors.npy"
    np.save(npy, indices)
    meta = {
        **meta,
        "neighbors_file": npy.name,
        "shape": list(indices.shape),
        "sha256": sha256_file(npy),
    }
    (out_dir / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-vectors", type=Path, required=True, help="float32 .npy (N, D)")
    ap.add_argument("--query-vectors", type=Path, required=True, help="float32 .npy (Q, D)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--metric", choices=["ip", "l2"], default="ip")
    args = ap.parse_args()

    if not args.corpus_vectors.exists() or not args.query_vectors.exists():
        print(
            "ERROR: embedding files missing. Defer ground-truth until embeddings are built.\n"
            f"  corpus: {args.corpus_vectors}\n"
            f"  queries: {args.query_vectors}",
            file=sys.stderr,
        )
        sys.exit(2)

    print(f"Loading vectors (metric={args.metric}, k={args.k}) ...", flush=True)
    corpus = np.load(args.corpus_vectors)
    queries = np.load(args.query_vectors)
    print(f"  corpus={corpus.shape} queries={queries.shape}", flush=True)
    idx = compute_exact_knn(corpus, queries, k=args.k, metric=args.metric)
    save_ground_truth(
        args.out_dir,
        idx,
        {
            "k": args.k,
            "metric": args.metric,
            "corpus_vectors": str(args.corpus_vectors),
            "query_vectors": str(args.query_vectors),
            "n_corpus": int(corpus.shape[0]),
            "n_queries": int(queries.shape[0]),
            "dim": int(corpus.shape[1]),
        },
    )
    print(f"Wrote ground truth to {args.out_dir}")


if __name__ == "__main__":
    main()
