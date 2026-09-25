#!/usr/bin/env python3
"""Build Stage 1 HNSW index once embeddings exist."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from engine.index.hnsw import HNSWIndex


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", type=Path, default=Path("data/msmarco/subset_1m/embeddings/passages.f32.npy"))
    ap.add_argument("--out", type=Path, default=Path("data/indexes/stage01_hnsw/hnsw.bin"))
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--ef-search", type=int, default=64)
    args = ap.parse_args()
    vecs = np.load(args.vectors)
    print(f"Building HNSW on {vecs.shape} ...", flush=True)
    idx = HNSWIndex(vecs, space="ip", M=args.M, ef_construction=args.ef_construction, ef_search=args.ef_search)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    idx.save(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
