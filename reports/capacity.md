# Capacity & resource report (this box)

**Machine:** 8 vCPU, 31 GB RAM, no swap, NVMe data disk  
**Corpus:** MS MARCO 1M × 384-d BGE embeddings  
**Load tools:** `bench/capacity_test.py` (thread ramp) + `autocannon`  
**Server:** `engine/serve_http.py` (ThreadingHTTPServer)

---

## Headline limits

| System | Sustainable QPS (p99 SLO) | Server RSS | Notes |
|---|---|---|---|
| **Our HNSW** | **~960 QPS** @ p99≤100ms (c=8) | **~1.8 GB** | Best dense path on this box |
| Our HNSW (exhaust) | ~1500 QPS peak | ~1.8 GB | p99 blows past 1s at c≥32 |
| Autocannon HNSW c=64 | ~1668 req/s | — | high latency / timeouts under overload |
| Partitioned (nprobe=16) | **~174 QPS** @ p99≤200ms | ~2.4 GB | All partitions in RAM |
| Graph manager (LRU 8GB) | **~166 QPS** @ p99≤300ms | ~2.4 GB | After warm cache |
| HNSW + rescoring | **~170 QPS** @ p99≤100ms | ~3.2 GB | Needs fp32 vectors mapped |
| HNSW + result/semantic cache | ~231 QPS @ p99≤100ms | ~1.8 GB | Zipf helps; GIL/lock limits |
| Flat (FAISS) | ~12 QPS | ~3.0 GB | Too slow for load tests |
| **Qdrant** (on-disk HNSW) | **~403 QPS** @ p99≤100ms (c=8) | **~175 MB** container | 1M points ingested |

**RAM headroom:** even under HNSW load, machine stays ~**1.8–3.2 GB** process RSS; **~29 GB available** system-wide. Bottleneck is **CPU + Python GIL / single-process threading**, not RAM for the 1M index.

---

## What this means

1. On **current resources**, the engine is **CPU-bound**, not memory-bound, for 1M vectors.
2. Single-process threaded HTTP tops out around **1–1.5k QPS** for HNSW before latency collapses.
3. Qdrant uses far less RAM (on-disk vectors) but lower QPS than our in-RAM HNSW on this box.
4. To push higher: multi-process workers (one index copy per process), or Rust/Stage 12 hot path, or more cores.

---

## Artifacts

- Capacity JSONs: `results/capacity/`
- Qdrant: Docker `qdrant` on `:6333`, collection `msmarco_1m`, storage under `data/qdrant/storage`
- Scripts: `scripts/run_capacity_v2.sh`, `scripts/autocannon_search.sh`, `scripts/serve_qdrant_proxy.py`, `bench/capacity_test.py`
- Stage 2 PQ smoke: recall@10≈0.68, index ~64 MB (`results/02/pq64_smoke.json`)

---

## Still not fully gated (need more time / spend)

- Full Stage 2/3 sweeps, Stage 6–7 measured gates, Stage 9 rerank load, Stage 11 multi-process server ladder, Track S 30M+.
