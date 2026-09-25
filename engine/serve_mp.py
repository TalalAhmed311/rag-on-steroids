#!/usr/bin/env python3
"""
Multi-process retrieval server (Experiment 2).

Each worker process loads its own pipeline copy and listens on the same port
via SO_REUSEPORT so the kernel distributes accepts. This bypasses the Python
GIL for search work across processes (at the cost of N× index RAM).

Usage:
  python3 -m engine.serve_mp --config configs/stage01_hnsw.yaml --workers 4 --port 8080
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from multiprocessing import Process
from pathlib import Path

import numpy as np
import psutil

from engine.pipeline import build_pipeline, load_config
from engine.types import Query


class ReuseThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Linux: multiple processes bind the same port; kernel load-balances accepts.
        if hasattr(socket, "SO_REUSEPORT"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        super().server_bind()


def make_handler(pipeline, query_vectors, query_texts, worker_id: int, config_name: str):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self._json(200, {"ok": True, "config": config_name, "worker": worker_id, "pid": os.getpid()})
                return
            if self.path == "/stats":
                proc = psutil.Process()
                self._json(
                    200,
                    {
                        "worker": worker_id,
                        "pid": os.getpid(),
                        "rss_gb": proc.memory_info().rss / (1024**3),
                        "cpu_pct": proc.cpu_percent(None),
                        "engine": pipeline.stats(),
                    },
                )
                return
            self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/search":
                self._json(404, {"error": "not found"})
                return
            t0 = time.perf_counter()
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                text = body.get("text")
                vec = body.get("vector")
                k = int(body.get("k", 10))
                if vec is None and body.get("qidx") is not None and query_vectors is not None:
                    qidx = int(body["qidx"]) % len(query_vectors)
                    vec = query_vectors[qidx]
                    if not text and query_texts and qidx < len(query_texts):
                        text = query_texts[qidx]
                hits = pipeline.search(
                    Query(
                        id=str(body.get("id", "")),
                        text=text,
                        vector=None if vec is None else np.asarray(vec, dtype=np.float32),
                    ),
                    k=k,
                )
                self._json(
                    200,
                    {
                        "hits": [{"doc_id": h.doc_id, "score": h.score, "source": h.source} for h in hits],
                        "worker": worker_id,
                        "latency_ms": (time.perf_counter() - t0) * 1000.0,
                    },
                )
            except Exception as e:
                self._json(500, {"error": str(e), "worker": worker_id})

        def _json(self, code: int, obj: dict) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            return

    return Handler


def worker_main(args_dict: dict, worker_id: int) -> None:
    cfg = load_config(args_dict["config"])
    print(f"[worker {worker_id} pid={os.getpid()}] loading {cfg['name']} ...", flush=True)
    pipeline = build_pipeline(cfg, data_root=args_dict["data_root"])
    qv = np.load(args_dict["query_vectors"])
    qtexts = []
    qt = Path(args_dict["query_texts"])
    if qt.exists():
        import pyarrow.parquet as pq

        qtexts = pq.read_table(qt, columns=["text"]).column("text").to_pylist()
    handler = make_handler(pipeline, qv, qtexts, worker_id, cfg["name"])
    httpd = ReuseThreadingHTTPServer((args_dict["host"], args_dict["port"]), handler)
    print(f"[worker {worker_id}] listening on {args_dict['host']}:{args_dict['port']}", flush=True)
    httpd.serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--query-vectors", type=Path, default=Path("data/msmarco/subset_1m/embeddings/queries.f32.npy"))
    ap.add_argument("--query-texts", type=Path, default=Path("data/msmarco/queries_dev.parquet"))
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    args = ap.parse_args()

    payload = {
        "config": str(args.config),
        "host": args.host,
        "port": args.port,
        "query_vectors": str(args.query_vectors),
        "query_texts": str(args.query_texts),
        "data_root": str(args.data_root),
    }

    procs: list[Process] = []

    def _shutdown(*_):
        for p in procs:
            if p.is_alive():
                p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"Starting {args.workers} workers on :{args.port} (SO_REUSEPORT)", flush=True)
    for i in range(args.workers):
        p = Process(target=worker_main, args=(payload, i), daemon=False)
        p.start()
        procs.append(p)
        time.sleep(0.5)  # stagger loads to avoid thundering herd on disk/RAM

    # wait until health responds
    import urllib.request

    for _ in range(180):
        try:
            with urllib.request.urlopen(f"http://{args.host}:{args.port}/health", timeout=1) as r:
                if r.status == 200:
                    print("cluster healthy", flush=True)
                    break
        except Exception:
            time.sleep(1)
    else:
        print("WARNING: health check never succeeded", flush=True)

    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
