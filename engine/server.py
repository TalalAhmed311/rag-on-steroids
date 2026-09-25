"""gRPC retrieval service (Stage 11)."""
from __future__ import annotations

import argparse
import json
from concurrent import futures
from pathlib import Path

import grpc
import numpy as np

from engine.pipeline import build_pipeline, load_config
from engine.types import Query

# Inline proto via generic servicer using JSON payloads to avoid codegen dependency at import.
# Full .proto lives in engine/proto/retriever.proto for future grpc_tools generation.


class RetrieverServicer:
    def __init__(self, pipeline, query_vectors: np.ndarray | None = None, query_texts: list[str] | None = None):
        self.pipeline = pipeline
        self.query_vectors = query_vectors
        self.query_texts = query_texts or []

    def Search(self, request: dict, context=None) -> dict:
        qid = request.get("id", "")
        text = request.get("text")
        vec = request.get("vector")
        k = int(request.get("k", 10))
        if vec is None and request.get("qidx") is not None and self.query_vectors is not None:
            qidx = int(request["qidx"])
            vec = self.query_vectors[qidx]
            if not text and qidx < len(self.query_texts):
                text = self.query_texts[qidx]
        query = Query(
            id=str(qid),
            text=text,
            vector=None if vec is None else np.asarray(vec, dtype=np.float32),
        )
        hits = self.pipeline.search(query, k=k)
        return {
            "hits": [{"doc_id": h.doc_id, "score": h.score, "source": h.source} for h in hits],
            "stats": self.pipeline.stats(),
        }


def serve_http_json(pipeline, host: str, port: int, query_vectors=None, query_texts=None):
    """Lightweight JSON-over-HTTP server (stdlib) so Stage 11 works without proto stubs."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    servicer = RetrieverServicer(pipeline, query_vectors, query_texts)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            if self.path != "/search":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            result = servicer.Search(body)
            payload = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt, *args):
            return

    httpd = HTTPServer((host, port), Handler)
    print(f"Serving on http://{host}:{port}  POST /search  GET /health", flush=True)
    httpd.serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--query-vectors", type=Path, default=None)
    ap.add_argument("--query-texts-parquet", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    pipeline = build_pipeline(cfg, data_root=args.data_root)
    qv = np.load(args.query_vectors) if args.query_vectors else None
    qtexts = []
    if args.query_texts_parquet:
        import pyarrow.parquet as pq

        qtexts = pq.read_table(args.query_texts_parquet, columns=["text"]).column("text").to_pylist()
    serve_http_json(pipeline, args.host, args.port, qv, qtexts)


if __name__ == "__main__":
    main()
