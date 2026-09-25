#!/usr/bin/env python3
"""Create a Qdrant collection and upsert MS MARCO 1M embeddings."""
from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest
except ImportError:
    raise SystemExit("pip install qdrant-client")


def wait_ready(url: str, timeout_s: float = 120.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url + "/readyz", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(1)
    raise RuntimeError("Qdrant not ready")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:6333")
    ap.add_argument("--collection", default="msmarco_1m")
    ap.add_argument("--vectors", type=Path, default=Path("data/msmarco/subset_1m/embeddings/passages.f32.npy"))
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--on-disk", action="store_true", help="Store vectors on disk (Qdrant memmap)")
    args = ap.parse_args()

    wait_ready(args.url)
    client = QdrantClient(url=args.url, prefer_grpc=False, timeout=60)
    vecs = np.load(args.vectors, mmap_mode="r")
    n, dim = vecs.shape
    print(f"Upserting {n} x {dim} into {args.collection} (on_disk={args.on_disk})")

    if args.collection in [c.name for c in client.get_collections().collections]:
        client.delete_collection(args.collection)

    client.create_collection(
        collection_name=args.collection,
        vectors_config=rest.VectorParams(
            size=dim,
            distance=rest.Distance.COSINE,
            on_disk=args.on_disk,
        ),
        optimizers_config=rest.OptimizersConfigDiff(indexing_threshold=20000),
        hnsw_config=rest.HnswConfigDiff(m=16, ef_construct=200),
    )

    t0 = time.time()
    for start in range(0, n, args.batch):
        end = min(start + args.batch, n)
        batch = np.asarray(vecs[start:end], dtype=np.float32)
        points = [
            rest.PointStruct(id=start + i, vector=batch[i].tolist())
            for i in range(end - start)
        ]
        client.upsert(collection_name=args.collection, points=points, wait=True)
        if end % (args.batch * 20) == 0 or end == n:
            print(f"  {end}/{n}  ({end / (time.time() - t0):.0f} pts/s)", flush=True)
    print(f"Done in {(time.time() - t0) / 60:.1f} min")
    info = client.get_collection(args.collection)
    print(info)


if __name__ == "__main__":
    main()
