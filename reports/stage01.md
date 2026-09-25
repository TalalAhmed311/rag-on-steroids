# Stage 01 — Naive baselines

**Hypothesis:** Single in-RAM HNSW is much faster than flat with small recall loss.

**What changed**
- `FlatIndex` (FAISS IndexFlatIP) and `HNSWIndex` (hnswlib, M=16, ef_construction=200, ef_search=64)
- Index artifact: `data/indexes/stage01_hnsw/hnsw.bin` (~1.6 GB)

**Results (1M subset, warm, 1000 queries unless noted)**

| Config | recall@10 | p50 ms | p99 ms | notes |
|---|---|---|---|---|
| Flat (500 q) | **1.000** | 164 | 241 | Stage 0 gate |
| HNSW ef=64 | **0.956** | 0.58 | 1.55 | ≥ 0.90 floor |

**Gate:** passed — naive HNSW above recall floor; flat establishes unconstrained accuracy reference on 1M (fits in 24 GB easily at ~1.5 GB vectors).

**Next:** quantization / mmap / partitioned under tighter budgets.
