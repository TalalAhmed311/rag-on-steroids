# Stage 00 — Foundations

**Hypothesis:** Measurement harness + exact flat search recover GT perfectly on the 1M subset.

**What changed**
- MS MARCO 1M embeddings (`bge-small-en-v1.5`, 384-d, L2-normalized)
- Exact GT via FAISS `IndexFlatIP`, k=100, for 6,980 queries
- Harness end-to-end with ANN recall + qrels metrics

**Gate (passed)**
- Flat search on 500 uniform workload queries:
  - **recall@1 = 1.0**
  - **recall@10 = 1.0**
  - **recall@100 = 1.0**
- Result: `results/01/stage01_flat/20260925T161049Z.json`
- Latency (warm, no memory container): p50≈164ms, p99≈241ms (brute-force 1M)

**Conclusion:** Stage 0 measurement path is valid. Bench files can be treated as frozen for ANN defs; do not change GT without approval.
