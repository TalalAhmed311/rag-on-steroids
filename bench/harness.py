"""Offline accuracy + latency harness."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from bench.metrics import (
    latency_stats,
    mean_metric,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    recall_at_k_qrels,
)
from bench.sysmon import SysMonitor
from engine.pipeline import build_pipeline, config_hash, load_config
from engine.types import Query


def git_commit() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def write_result(result: dict[str, Any], results_root: Path = Path("results")) -> Path:
    stage = result["stage"]
    name = result["config_name"]
    ts = result.get("timestamp") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = results_root / stage / name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ts}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return path


def _load_qrels(path: Path) -> dict[int, dict[int, float]]:
    import pyarrow.parquet as pq

    t = pq.read_table(path)
    out: dict[int, dict[int, float]] = {}
    qidx = t.column("qidx").to_numpy()
    # subset uses sub_idx when present
    doc_col = "sub_idx" if "sub_idx" in t.column_names else "idx"
    docs = t.column(doc_col).to_numpy()
    scores = t.column("score").to_numpy()
    for q, d, s in zip(qidx, docs, scores):
        out.setdefault(int(q), {})[int(d)] = float(s)
    return out


def run_eval(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    pipeline = build_pipeline(cfg, data_root=args.data_root)
    q_vectors = np.load(args.query_vectors, mmap_mode="r") if args.query_vectors else None
    q_texts = None
    if args.query_texts:
        import pyarrow.parquet as pq

        q_texts = pq.read_table(args.query_texts, columns=["text"]).column("text").to_pylist()

    workload = np.load(args.workload) if args.workload else np.arange(min(1000, len(q_vectors or [])))
    gt = np.load(args.ground_truth) if args.ground_truth else None
    qrels = _load_qrels(Path(args.qrels)) if args.qrels else None

    if args.warmup > 0:
        for i in workload[: args.warmup]:
            i = int(i)
            q = Query(
                id=str(i),
                text=None if q_texts is None else q_texts[i],
                vector=None if q_vectors is None else np.asarray(q_vectors[i]),
            )
            pipeline.search(q, k=args.k)

    mon = SysMonitor(disk_device=args.disk_device)
    mon.start()
    lat = []
    ann_recalls = {1: [], 10: [], 100: []}
    precs, mrrs, ndcgs, rq = [], [], [], []

    measure_n = min(args.n_queries, len(workload))
    for qi in range(measure_n):
        i = int(workload[qi])
        q = Query(
            id=str(i),
            text=None if q_texts is None else q_texts[i],
            vector=None if q_vectors is None else np.asarray(q_vectors[i]),
        )
        t0 = time.perf_counter()
        hits = pipeline.search(q, k=max(args.k, 100))
        lat.append((time.perf_counter() - t0) * 1000.0)
        docs = [h.doc_id for h in hits]
        if gt is not None:
            truth = gt[i].tolist()
            for kk in (1, 10, 100):
                ann_recalls[kk].append(recall_at_k(docs, truth, kk))
        if qrels is not None and i in qrels:
            rel = qrels[i]
            precs.append(precision_at_k(docs, rel, 10))
            mrrs.append(mrr_at_k(docs, rel, 10))
            ndcgs.append(ndcg_at_k(docs, rel, 10))
            rq.append(recall_at_k_qrels(docs, rel, 100))

    resources = mon.stop()
    result = {
        "stage": str(cfg.get("stage", "00")),
        "config_name": cfg["name"],
        "config_hash": config_hash(cfg),
        "git_commit": git_commit(),
        "server": cfg.get("server", {}),
        "memory_limit_gb": cfg.get("memory_limit_gb"),
        "dataset": cfg.get("dataset", {}),
        "workload": {
            "file": str(args.workload),
            "n_queries": measure_n,
        },
        "run_mode": args.run_mode,
        "repeat": args.repeat,
        "accuracy": {
            "recall@1": mean_metric(ann_recalls[1]) if ann_recalls[1] else None,
            "recall@10": mean_metric(ann_recalls[10]) if ann_recalls[10] else None,
            "recall@100": mean_metric(ann_recalls[100]) if ann_recalls[100] else None,
            "precision@10": mean_metric(precs) if precs else None,
            "mrr@10": mean_metric(mrrs) if mrrs else None,
            "ndcg@10": mean_metric(ndcgs) if ndcgs else None,
            "recall@100_qrels": mean_metric(rq) if rq else None,
        },
        "latency_ms": latency_stats(lat),
        "throughput": {"max_sustainable_qps": None, "slo_p99_ms": cfg.get("slo_p99_ms", 100)},
        "resources": resources,
        "engine_stats": pipeline.stats(),
        "index": {},
        "status": "ok",
        "error": None,
        "timestamp": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--results-root", type=Path, default=Path("results"))
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--query-vectors", type=Path, default=None)
    ap.add_argument("--query-texts", type=Path, default=None)
    ap.add_argument("--workload", type=Path, default=None)
    ap.add_argument("--ground-truth", type=Path, default=None)
    ap.add_argument("--qrels", type=Path, default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-queries", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--run-mode", default="warm")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--disk-device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.dry_run:
        from datetime import datetime, timezone

        result = {
            "stage": str(cfg.get("stage", "00")),
            "config_name": cfg["name"],
            "config_hash": config_hash(cfg),
            "git_commit": git_commit(),
            "status": "dry_run",
            "error": "dry-run only",
            "timestamp": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        }
        print(f"Wrote {write_result(result, args.results_root)}")
        return

    result = run_eval(cfg, args)
    path = write_result(result, args.results_root)
    print(json.dumps({"path": str(path), "accuracy": result["accuracy"], "latency_ms": result["latency_ms"]}, indent=2))


if __name__ == "__main__":
    main()
