# Stage 08 — Hybrid BM25 + dense (smoke)

**Setup:** HNSW dense + `rank_bm25` + RRF (k=60), 200 queries.

**Quality (qrels):** nDCG@10≈0.64, MRR@10≈0.59, Recall@100_qrels≈0.99

**Note:** ANN recall@k vs dense GT is not the right metric for fused lists (expected lower). Latency is high (~2s) because pure-Python BM25 over 1M docs — replace with Tantivy for Stage 8 gate-quality runs.
