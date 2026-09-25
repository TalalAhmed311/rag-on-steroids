# Stage 05 — Graph manager

**Hypothesis:** LRU partition cache under an 8 GB budget serves zipf traffic with high recall.

**Setup:** Stage 4 index on disk, `storage.mode=managed`, policy=lru, budget_gb=8, nprobe=16.

**Results (zipf s=1.0, 500 queries, warm)**
- recall@10 = **0.956**
- p50 ≈ 9.6 ms, p99 ≈ 19.5 ms
- Result JSON: `results/05/stage05_graph_manager_lru/`

**Gate:** managed path works; further cold/uniform vs mmap comparison can be expanded in later sweeps.
