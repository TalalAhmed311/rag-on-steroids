#!/usr/bin/env python3
"""
Capacity / load ramp against engine.serve_http.

Uses concurrent workers (open-loop style) to find the highest QPS where
p99 stays under --slo-p99-ms and error rate <= --max-error-rate.

Also works as a drop-in alternative to autocannon when Node is unavailable.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import psutil


def post_search(url: str, qidx: int, k: int = 10, timeout: float = 30.0) -> tuple[bool, float]:
    body = json.dumps({"qidx": int(qidx), "k": k}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
        return True, (time.perf_counter() - t0) * 1000.0
    except Exception:
        return False, (time.perf_counter() - t0) * 1000.0


def run_for(url: str, workload: np.ndarray, concurrency: int, duration_s: float, k: int) -> dict:
    end = time.time() + duration_s
    ok = err = 0
    lats: list[float] = []
    lock_idx = {"i": 0}

    def one(_):
        nonlocal ok, err
        while time.time() < end:
            i = lock_idx["i"]
            lock_idx["i"] = i + 1
            qidx = int(workload[i % len(workload)])
            good, ms = post_search(url, qidx, k=k)
            lats.append(ms)
            if good:
                ok += 1
            else:
                err += 1

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(one, i) for i in range(concurrency)]
        for f in as_completed(futs):
            f.result()
    elapsed = max(time.time() - t0, 1e-6)
    arr = np.asarray(lats, dtype=np.float64) if lats else np.array([0.0])
    total = ok + err
    return {
        "concurrency": concurrency,
        "duration_s": duration_s,
        "completed": ok,
        "errors": err,
        "error_rate": err / max(total, 1),
        "achieved_qps": ok / elapsed,
        "latency_ms": {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "mean": float(arr.mean()),
        },
        "client_rss_gb": psutil.Process().memory_info().rss / (1024**3),
    }


def fetch_server_stats(base: str) -> dict:
    try:
        with urllib.request.urlopen(base.replace("/search", "/stats"), timeout=5) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        return {"error": str(e)}


def ramp(url: str, workload: np.ndarray, concs: list[int], hold_s: float, slo_p99: float, max_err: float, k: int) -> dict:
    history = []
    best = None
    base = url
    for c in concs:
        # warm
        run_for(url, workload, min(c, 8), 3.0, k)
        r = run_for(url, workload, c, hold_s, k)
        r["server"] = fetch_server_stats(base)
        history.append(r)
        print(json.dumps({k: r[k] for k in ("concurrency", "achieved_qps", "error_rate", "latency_ms", "server")}, indent=2), flush=True)
        ok = r["latency_ms"]["p99"] <= slo_p99 and r["error_rate"] <= max_err
        if ok:
            best = r
        else:
            break
    return {"best": best, "history": history}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080/search")
    ap.add_argument("--workload", type=Path, default=Path("data/workloads/msmarco_dev/uniform.npy"))
    ap.add_argument("--hold-s", type=float, default=20.0)
    ap.add_argument("--slo-p99-ms", type=float, default=100.0)
    ap.add_argument("--max-error-rate", type=float, default=0.01)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--concurrencies", default="4,8,16,32,64,96,128")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    wl = np.load(args.workload)
    concs = [int(x) for x in args.concurrencies.split(",") if x.strip()]
    result = ramp(args.url, wl, concs, args.hold_s, args.slo_p99_ms, args.max_error_rate, args.k)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
