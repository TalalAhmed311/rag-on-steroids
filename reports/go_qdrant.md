# Go + Qdrant serving branch

**Branch:** `feat/go-qdrant-server`  
**Change:** All Python engine/bench/serving removed. Serving is Go `searchd` → Qdrant only.

## Layout
- `go/cmd/searchd` — HTTP `/health` `/stats` `/search`
- `go/cmd/loadtest` — concurrency ramp + p50/p95/p99
- `scripts/bench_go_qdrant.sh` — full capacity suite
- `scripts/run_go_searchd.sh` / `scripts/run_qdrant.sh`

## Measured (warm, this box)

| Metric | Value |
|---|---|
| SLO QPS (p99≤100ms) | **~1640** @ c=96 |
| Peak plateau | **~1650** |
| searchd RSS | **~52 MB** |
| Qdrant under load | **~646 MB** |
| Autocannon c=64 (fixed body) | ~2988 req/s |

Full comparison vs Python HNSW / MP / old proxy: **`reports/compare_go_qdrant.md`**.
