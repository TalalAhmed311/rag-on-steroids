# Context Engine (Go + Qdrant serving branch)

Python serving/engine code was removed on `feat/go-qdrant-server`.

## Stack
- **Search API:** Go (`go/cmd/searchd`)
- **Vector DB:** Qdrant (Docker)
- **Load test:** Go (`go/cmd/loadtest`) + optional autocannon
- **Data:** existing MS MARCO embeddings under `data/` (already ingested)

## Quick start
```bash
# Qdrant must be running with collection msmarco_1m
./scripts/run_go_searchd.sh

# capacity suite
./scripts/bench_go_qdrant.sh
```

See `reports/compare_go_qdrant.md` for QPS vs the old Python stack.

Full write-up (every experiment, issue table, Python/Go code): **`reports/ARTICLE.md`**.

Consumer / laptop SIMD vector-DB design notes: **`reports/consumer-db.md`**.
