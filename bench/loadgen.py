"""Open-loop HTTP load generator against engine.server (/search)."""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from bench.metrics import latency_stats


def one_request(url: str, body: dict, timeout: float = 30.0) -> float:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp.read()
    return (time.perf_counter() - t0) * 1000.0


def run_fixed_rate(url: str, qindices: np.ndarray, rate: float, duration_s: float) -> dict:
    interval = 1.0 / rate
    end = time.perf_counter() + duration_s
    lats = []
    errors = 0
    i = 0
    next_t = time.perf_counter()
    while time.perf_counter() < end:
        now = time.perf_counter()
        if now < next_t:
            time.sleep(min(0.001, next_t - now))
            continue
        qidx = int(qindices[i % len(qindices)])
        i += 1
        next_t += interval
        try:
            lats.append(one_request(url, {"qidx": qidx, "k": 10}))
        except Exception:
            errors += 1
    stats = latency_stats(lats)
    return {
        "offered_qps": rate,
        "completed": len(lats),
        "errors": errors,
        "error_rate": errors / max(1, len(lats) + errors),
        "latency_ms": stats,
    }


def ramp(url: str, qindices: np.ndarray, start_qps: float, step: float, hold_s: float, max_qps: float, slo_p99: float) -> dict:
    rate = start_qps
    history = []
    best = None
    while rate <= max_qps:
        r = run_fixed_rate(url, qindices, rate, hold_s)
        history.append(r)
        ok = r["latency_ms"]["p99"] <= slo_p99 and r["error_rate"] <= 0.001
        if ok:
            best = r
            rate += step
        else:
            break
    return {"max_sustainable": best, "history": history}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8080/search")
    ap.add_argument("--workload", type=str, default="data/workloads/msmarco_dev/uniform.npy")
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--duration-s", type=float, default=30.0)
    ap.add_argument("--ramp", action="store_true")
    ap.add_argument("--start-qps", type=float, default=10.0)
    ap.add_argument("--step-qps", type=float, default=10.0)
    ap.add_argument("--hold-s", type=float, default=30.0)
    ap.add_argument("--max-qps", type=float, default=200.0)
    ap.add_argument("--slo-p99-ms", type=float, default=100.0)
    args = ap.parse_args()
    q = np.load(args.workload)
    if args.ramp:
        print(json.dumps(ramp(args.url, q, args.start_qps, args.step_qps, args.hold_s, args.max_qps, args.slo_p99_ms), indent=2))
    else:
        print(json.dumps(run_fixed_rate(args.url, q, args.rate, args.duration_s), indent=2))


if __name__ == "__main__":
    main()
