# Go + Qdrant search service

Lightweight HTTP search front-end in **Go** talking to **Qdrant** over gRPC.
Replaces the Python `serve_http` / `serve_mp` / proxy servers on this branch.

## Why

| | Python multi-process HNSW | Go → Qdrant |
|---|---|---|
| Process RSS | ~1.8 GB × workers | Go heap **~15–30 MB** + Qdrant ~200–500 MB |
| Concurrency model | GIL / process copies | Goroutines + Qdrant’s Rust threads |
| Role | Full custom engine | Optimized serving path |

## API (same as before)

- `GET /health`
- `GET /stats`
- `POST /search` JSON `{"qidx": 0, "k": 10}` or `{"vector":[...], "k":10}`

## Run

Qdrant must be up (`docker` container or local) with collection `msmarco_1m`.

```bash
# from repo root
./scripts/run_go_searchd.sh

# or
cd go && go build -o bin/searchd ./cmd/searchd
./bin/searchd --addr :8080 --qdrant-host 127.0.0.1 --qdrant-port 6334
```

Load test (existing Python client harness still works against any HTTP API):

```bash
python3 -m bench.capacity_test --url http://127.0.0.1:8080/search \
  --workload data/workloads/msmarco_dev/zipf_s1p0.npy \
  --hold-s 12 --slo-p99-ms 100 --concurrencies 8,16,32,64
```

## Layout

```
go/
  cmd/searchd/main.go
  internal/server/http.go
  internal/vectors/npy.go   # loads queries.f32.npy for qidx
  bin/searchd               # gitignored build artifact
```
