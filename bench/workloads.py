"""Generate and freeze uniform / Zipf query streams (Stage 0).

These streams are IDs into a query set — no embeddings required.
After Stage 0 freeze, do not regenerate with different seeds without approval.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def uniform_stream(n_queries: int, pool_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, pool_size, size=n_queries, dtype=np.int32)


def zipf_stream(n_queries: int, pool_size: int, seed: int, s: float = 1.0) -> np.ndarray:
    """Sample query indices with Zipf(s) over ranks 1..pool_size (hot head)."""
    if s <= 0:
        raise ValueError("zipf s must be > 0")
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, pool_size + 1, dtype=np.float64)
    weights = ranks ** (-s)
    weights /= weights.sum()
    return rng.choice(pool_size, size=n_queries, replace=True, p=weights).astype(np.int32)


def write_workload(
    out_dir: Path,
    name: str,
    indices: np.ndarray,
    meta: dict[str, Any],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / f"{name}.npy"
    meta_path = out_dir / f"{name}.json"
    np.save(npy_path, indices)
    meta = {
        **meta,
        "name": name,
        "n_queries": int(indices.shape[0]),
        "file": npy_path.name,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return npy_path


def generate_default_workloads(
    out_dir: Path,
    pool_size: int,
    n_queries: int = 10_000,
    seed: int = 42,
) -> list[Path]:
    paths: list[Path] = []
    paths.append(
        write_workload(
            out_dir,
            "uniform",
            uniform_stream(n_queries, pool_size, seed),
            {"type": "uniform", "pool_size": pool_size, "seed": seed},
        )
    )
    for s in (0.8, 1.0, 1.2):
        tag = str(s).replace(".", "p")
        paths.append(
            write_workload(
                out_dir,
                f"zipf_s{tag}",
                zipf_stream(n_queries, pool_size, seed, s=s),
                {"type": "zipf", "s": s, "pool_size": pool_size, "seed": seed},
            )
        )
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("data/workloads/msmarco_dev"))
    ap.add_argument("--pool-size", type=int, default=6980, help="Dev query count")
    ap.add_argument("--n-queries", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    paths = generate_default_workloads(args.out_dir, args.pool_size, args.n_queries, args.seed)
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
