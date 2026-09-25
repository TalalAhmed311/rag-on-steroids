#!/usr/bin/env python3
"""Thin HTTP proxy: same /search API as engine.serve_http, backed by Qdrant."""
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from qdrant_client import QdrantClient


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant", default="http://127.0.0.1:6333")
    ap.add_argument("--collection", default="msmarco_1m")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--query-vectors", default="data/msmarco/subset_1m/embeddings/queries.f32.npy")
    ap.add_argument("--ef", type=int, default=64)
    args = ap.parse_args()

    client = QdrantClient(url=args.qdrant, timeout=30)
    qv = np.load(args.query_vectors)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self._ok({"ok": True, "backend": "qdrant"})
                return
            if self.path == "/stats":
                info = client.get_collection(args.collection)
                self._ok({"collection": args.collection, "points": info.points_count})
                return
            self.send_error(404)

        def do_POST(self):
            if self.path != "/search":
                self.send_error(404)
                return
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            k = int(body.get("k", 10))
            if body.get("vector") is not None:
                vec = body["vector"]
            else:
                qidx = int(body.get("qidx", 0)) % len(qv)
                vec = qv[qidx].tolist()
            t0 = time.perf_counter()
            # qdrant-client >=1.14 renamed search -> query_points
            if hasattr(client, "query_points"):
                res = client.query_points(
                    collection_name=args.collection,
                    query=vec,
                    limit=k,
                    search_params={"hnsw_ef": args.ef},
                )
                points = res.points
            else:
                points = client.search(
                    collection_name=args.collection,
                    query_vector=vec,
                    limit=k,
                    search_params={"hnsw_ef": args.ef},
                )
            ms = (time.perf_counter() - t0) * 1000
            payload = {
                "hits": [{"doc_id": h.id, "score": h.score, "source": "qdrant"} for h in points],
                "latency_ms": ms,
            }
            self._ok(payload)

        def _ok(self, obj):
            data = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            return

    print(f"Qdrant proxy on http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
