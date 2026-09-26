# Go + Qdrant branch progress

Branch: `feat/go-qdrant-server`

## Done
- [x] Removed all Python engine/bench/scripts from this branch
- [x] Go `searchd` → Qdrant gRPC (`go/cmd/searchd`)
- [x] Go `loadtest` ramp (`go/cmd/loadtest`)
- [x] Capacity comparison vs Python baselines (see `reports/compare_go_qdrant.md`)

## Measured QPS (warm)
- **Go+Qdrant:** ~**1640 QPS** sustainable (p99≤100ms, c=96); plateau ~1650
- vs Python HNSW 1p ~960 · MP w=4 ~2140 · Python→Qdrant proxy ~403
- searchd ~52 MB + Qdrant ~646 MB under load

## Runtime deps
- Qdrant (Docker) with `msmarco_1m` collection
- Go 1.23+
- Optional: autocannon (Node)
