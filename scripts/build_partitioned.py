#!/usr/bin/env python3
"""Build Stage 4 partitioned HNSW on the 1M subset."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from engine.index.partitioned import PartitionedIndex


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", type=Path, default=Path("data/msmarco/subset_1m/embeddings/passages.f32.npy"))
    ap.add_argument("--out", type=Path, default=Path("data/indexes/stage04_partitioned"))
    ap.add_argument("--partitions", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=1)
    ap.add_argument("--nprobe", type=int, default=8)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--ef-search", type=int, default=64)
    args = ap.parse_args()

    vecs = np.load(args.vectors)
    print(f"Building partitioned index on {vecs.shape} K={args.partitions} ...", flush=True)
    t0 = time.time()
    idx = PartitionedIndex.build(
        vecs,
        n_partitions=args.partitions,
        overlap=args.overlap,
        nprobe=args.nprobe,
        M=args.M,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    idx.save(args.out)
    sizes = [len(m) for m in idx.local_to_global]
    print(
        f"Saved {args.out} in {(time.time() - t0) / 60:.1f} min | "
        f"part sizes min/median/max={min(sizes)}/{int(np.median(sizes))}/{max(sizes)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
