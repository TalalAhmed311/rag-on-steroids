# Project Progress

Source of truth for completed work. Updated after each finished task.

---

## Done on this box (8 vCPU / 31 GB)

### Data
- [x] 300 GB disk, MS MARCO 1M text + embeddings + GT (IP k=100)

### Measured stages
- [x] Stage 0/1 Flat + HNSW (recall gates passed)
- [x] Stage 2 PQ smoke (FAISS IndexPQ m=64, recall@10≈0.68)
- [x] Stage 4 Partitioned (nprobe=16)
- [x] Stage 5 Graph manager (lazy + LRU)
- [x] Stage 7 rescoring load path
- [x] Stage 8 hybrid smoke (slow BM25)
- [x] Stage 10 cache load path

### Serving / baselines
- [x] Threaded HTTP server `engine/serve_http.py`
- [x] Capacity ramp `bench/capacity_test.py` + **autocannon**
- [x] **Qdrant** Docker local + 1M on-disk ingest + proxy load test
- [x] Report: `reports/capacity.md` — HNSW ~960 QPS sustainable, ~1.8 GB RSS; Qdrant ~403 QPS @ ~175 MB
- [x] **Experiment 2 multi-process:** `engine/serve_mp.py` + `reports/capacity_mp.md`
  - HNSW w=4 → **~2140 QPS** (p99≤100); w=8 no gain, ~14 GB RAM
  - Partitioned/GM scale ~2× with workers but stay << HNSW
  - Qdrant still ~400 QPS, RAM-efficient

### Open
- [ ] Multi-process server (break GIL ceiling)
- [ ] Tantivy BM25 / Stage 9 rerank full curves
- [ ] Track S 30M+ / paid ladder / 1M QPS extrapolation

---

## Changelog

| When | Task |
|---|---|
| 2026-09-24–25 | Data, embeddings, GT, Stages 0/1/4/5 measured |
| 2026-09-25 | Qdrant local, capacity matrix, autocannon, capacity report |
