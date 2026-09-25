"""Multi-threaded HTTP retrieval server for capacity tests."""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import psutil

from engine.pipeline import build_pipeline, load_config
from engine.types import Query


class Stats:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.n = 0
        self.errors = 0
        self.lat_ms: list[float] = []
        self.started = time.time()

    def record(self, ms: float, ok: bool = True) -> None:
        with self.lock:
            self.n += 1
            if not ok:
                self.errors += 1
            if len(self.lat_ms) < 200_000:
                self.lat_ms.append(ms)

    def snapshot(self) -> dict:
        with self.lock:
            arr = np.asarray(self.lat_ms, dtype=np.float64) if self.lat_ms else np.array([0.0])
            elapsed = max(time.time() - self.started, 1e-6)
            proc = psutil.Process()
            return {
                "requests": self.n,
                "errors": self.errors,
                "qps": self.n / elapsed,
                "latency_ms": {
                    "p50": float(np.percentile(arr, 50)),
                    "p95": float(np.percentile(arr, 95)),
                    "p99": float(np.percentile(arr, 99)),
                    "mean": float(arr.mean()),
                },
                "rss_gb": proc.memory_info().rss / (1024**3),
                "cpu_pct": proc.cpu_percent(None),
                "uptime_s": elapsed,
            }


def make_handler(pipeline, query_vectors, query_texts, stats: Stats, config_name: str):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self._json(200, {"ok": True, "config": config_name})
                return
            if self.path == "/stats":
                self._json(200, {**stats.snapshot(), "engine": pipeline.stats()})
                return
            self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/search":
                self._json(404, {"error": "not found"})
                return
            t0 = time.perf_counter()
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                text = body.get("text")
                vec = body.get("vector")
                k = int(body.get("k", 10))
                if vec is None and body.get("qidx") is not None and query_vectors is not None:
                    qidx = int(body["qidx"]) % len(query_vectors)
                    vec = query_vectors[qidx]
                    if not text and query_texts and qidx < len(query_texts):
                        text = query_texts[qidx]
                query = Query(
                    id=str(body.get("id", "")),
                    text=text,
                    vector=None if vec is None else np.asarray(vec, dtype=np.float32),
                )
                hits = pipeline.search(query, k=k)
                payload = {
                    "hits": [{"doc_id": h.doc_id, "score": h.score, "source": h.source} for h in hits]
                }
                stats.record((time.perf_counter() - t0) * 1000.0, ok=True)
                self._json(200, payload)
            except Exception as e:
                stats.record((time.perf_counter() - t0) * 1000.0, ok=False)
                self._json(500, {"error": str(e)})

        def _json(self, code: int, obj: dict) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt, *args):
            return

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--query-vectors", type=Path, default=Path("data/msmarco/subset_1m/embeddings/queries.f32.npy"))
    ap.add_argument("--query-texts", type=Path, default=Path("data/msmarco/queries_dev.parquet"))
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"Building pipeline {cfg['name']} ...", flush=True)
    pipeline = build_pipeline(cfg, data_root=args.data_root)
    qv = np.load(args.query_vectors)
    qtexts = []
    if args.query_texts.exists():
        import pyarrow.parquet as pq

        qtexts = pq.read_table(args.query_texts, columns=["text"]).column("text").to_pylist()
    stats = Stats()
    handler = make_handler(pipeline, qv, qtexts, stats, cfg["name"])
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving {cfg['name']} on http://{args.host}:{args.port}  (/search /health /stats)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print(json.dumps(stats.snapshot(), indent=2))


if __name__ == "__main__":
    main()
