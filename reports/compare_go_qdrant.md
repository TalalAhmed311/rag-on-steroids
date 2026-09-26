# Go + Qdrant vs Python stack — QPS comparison

**Box:** 8 vCPU, 31 GB RAM · **Corpus:** MS MARCO 1M × 384-d · **ef=64**  
**This branch:** Python serving/engine removed; Go `searchd` → Qdrant gRPC.

Artifacts: `results/go_qdrant/` · harness: `scripts/bench_go_qdrant.sh`

---

## Headline

| Stack | Sustainable QPS (p99≤100ms) | Peak plateau | Process / backend RAM |
|---|---|---|---|
| **Go + Qdrant (this)** | **~1640 QPS** @ c=96 | **~1650 QPS** | searchd **~52 MB** + Qdrant **~646 MB** |
| Python HNSW 1-proc | ~960 QPS @ c=8 | ~1500 QPS | ~1.8 GB |
| Python HNSW MP w=4 | **~2140 QPS** @ c=64 | ~2275 QPS | ~7 GB |
| Python → Qdrant proxy | ~403 QPS @ c=8 | ~400 QPS | Qdrant ~175 MB (cold) |

**Verdict:** Go fronting Qdrant is **~4×** the old Python→Qdrant proxy and **~1.7×** single-process Python HNSW under the same p99≤100ms SLO, at **~700 MB** total vs **1.8–7 GB** for in-RAM HNSW. Absolute QPS leader on this box is still **Python HNSW MP w=4 (~2.1k)** — it buys that with **~10×** more RAM (full index copies).

---

## Go + Qdrant ramp (zipf-style rotating `qidx`)

### SLO stop (p99≤100ms)

| c | QPS | p50 | p95 | p99 |
|---|---|---|---|---|
| 8 | 1290 | 5.7 | 10.6 | 14.2 |
| 16 | 1410 | 10.8 | 18.5 | 23.2 |
| 32 | 1543 | 20.2 | 31.7 | 38.3 |
| 64 | 1617 | 39.2 | 57.1 | 65.8 |
| **96** | **1640** | 58.0 | 81.5 | **96.6** |
| 128 | 1645 | 77.2 | 105.9 | 126 → stop |

### Exhaust (no SLO)

Saturates ~**1620–1650 QPS** from c=64 upward; higher concurrency only inflates latency (p99 257 ms @ c=256).

### Autocannon (fixed body — optimistic)

| c | Avg req/s | p99 latency |
|---|---|---|
| 64 | **~2988** | ~39 ms |
| 128 | **~2759** | ~78 ms |

Same-query body overstates QPS vs the rotating-`qidx` loadtest; use **1640** as the fair number.

---

## Why the gap moved

1. **Python Qdrant proxy was the bottleneck** (~400 QPS), not Qdrant itself — Go gRPC client + goroutines unlock ~4×.
2. **Qdrant under load caches** into ~646 MB (was ~175 MB idle); still far below in-RAM HNSW.
3. **In-RAM HNSW MP** still wins raw QPS (~2.1k) because search stays in process memory with no gRPC hop — at the cost of N× index RAM.

---

## Reproduce

```bash
# Qdrant up with msmarco_1m
./scripts/bench_go_qdrant.sh
```
