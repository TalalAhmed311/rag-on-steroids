# Stage 04 — Multi-graph partitioned HNSW

**Hypothesis:** Many small HNSW partitions + centroid routing recover high recall with controllable probe cost.

**Setup:** K=256, overlap=1, M=16, ef_search=64 on 1M BGE embeddings.

**nprobe sweep (500 queries)**

| nprobe | recall@10 | p50 ms | p99 ms |
|---|---|---|---|
| 8 | 0.856 | ~4.8 | ~11 |
| **16** | **0.914** | ~9.3 | ~15 |
| 32 | 0.954 | ~18.8 | ~25 |
| 64 | 0.980 | ~38 | ~45 |

**Gate:** passed at **nprobe=16** (recall@10=0.918 on 1000-query harness). Default config updated.

**Imbalance:** part sizes min/median/max = 307 / 3651 / 11128.
