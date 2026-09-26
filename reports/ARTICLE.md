# Plain Dense Retrieval at Scale: Full Experiment Log

**One article. Every stage, every serving stack, every wall, every table we measured.**

This is the canonical write-up of the MS MARCO 1M retrieval capacity work: what we built, how Python search worked (GIL → multi-process), how Go + Qdrant changed the picture, what failed, and why.

---

## Table of contents

1. [Scope and ground rules](#1-scope-and-ground-rules)
2. [Hardware, corpus, and measurement method](#2-hardware-corpus-and-measurement-method)
3. [What “simple search” is (and is not)](#3-what-simple-search-is-and-is-not)
4. [Stage gates: quality before capacity](#4-stage-gates-quality-before-capacity)
5. [Python architecture and code](#5-python-architecture-and-code)
6. [Experiment 1 — Single-process capacity](#6-experiment-1--single-process-capacity)
7. [The GIL problem (why threads were not enough)](#7-the-gil-problem-why-threads-were-not-enough)
8. [Experiment 2 — Multi-process SO_REUSEPORT](#8-experiment-2--multi-process-soreuseport)
9. [Qdrant path: Python proxy → Go searchd](#9-qdrant-path-python-proxy--go-searchd)
10. [Go architecture and code](#10-go-architecture-and-code)
11. [Go capacity + exhaust matrix](#11-go-capacity--exhaust-matrix)
12. [Master comparison tables](#12-master-comparison-tables)
13. [Issue catalog (everything that hurt us)](#13-issue-catalog-everything-that-hurt-us)
14. [Walls: CPU, RAM, GIL, disk, proxy](#14-walls-cpu-ram-gil-disk-proxy)
15. [What improved results (and what did not)](#15-what-improved-results-and-what-did-not)
16. [Hardware limitations (deeper)](#16-hardware-limitations-deeper)
17. [When the DB is remote: network I/O & bandwidth math](#17-when-the-db-is-remote-network-io--bandwidth-math)
18. [Deeper mechanics we needed to build this](#18-deeper-mechanics-we-needed-to-build-this)
19. [Scaling fantasy check: 100k QPS](#19-scaling-fantasy-check-100k-qps)
20. [Final verdict](#20-final-verdict)
21. [Artifact index](#21-artifact-index)

---

## 1. Scope and ground rules

### In scope

- Dense vector retrieval only: **BGE-small 384-d**, L2-normalized, Cosine / inner-product style scoring
- **HNSW** (and flat / PQ / partitioned variants where we gated them)
- HTTP `POST /search` with precomputed query vectors (`qidx`) so **embedding time is excluded**
- Capacity: QPS, latency percentiles, RSS / Docker memory, worker counts
- Serving in **Python** (`master`) and **Go + Qdrant** (`feat/go-qdrant-server`)

### Out of scope for the capacity numbers we treat as “truth”

- Cross-encoder / LLM **reranking**
- Production query embedding at load (GPU)
- Multi-node distributed Qdrant clusters (we only *estimate* those)
- Track S (30M+ docs) full capacity — designed for, not fully load-tested here

### SLO used in reports

Unless a table says otherwise:

| Name | Definition |
|---|---|
| **Sustainable QPS** | Highest achieved QPS in a concurrency ramp where **p99 ≤ 100 ms** and error rate ≤ 5% |
| **Peak / exhaust QPS** | Highest QPS seen with SLO disabled (latency may be terrible) |
| **Autocannon QPS** | Fixed JSON body every request — **optimistic**, often cache-shaped |

---

## 2. Hardware, corpus, and measurement method

### Machine

| Resource | Value |
|---|---|
| vCPU | **8** |
| RAM | **31 GB** |
| Swap | **none** |
| Data disk | ~300 GB NVMe (`data/`) |
| OS | Amazon Linux 2023 |

### Corpus

| Item | Value |
|---|---|
| Passages | **1,000,000** (MS MARCO subset) |
| Embedding model | `bge-small-en-v1.5` |
| Dim | **384** |
| Normalization | L2 |
| Passage vectors on disk | ~**1.5 GB** `passages.f32.npy` |
| Dev queries with vectors | **6,980** |
| Ground truth | FAISS `IndexFlatIP`, k=100 |

### Load tools

| Tool | Role |
|---|---|
| `bench/capacity_test.py` | Python concurrency ramp + p50/p95/p99 + SLO stop |
| `go/cmd/loadtest` | Same idea in Go |
| `autocannon` | HTTP blast; fixed body unless customized |
| `/stats` + `ps` + `docker stats` | RSS / CPU samples |

### Request shape

```http
POST /search
Content-Type: application/json

{"qidx": 1234, "k": 10}
```

Server maps `qidx` → row in `queries.f32.npy` and runs ANN. This isolates **retrieval**, not the embedder.

---

## 3. What “simple search” is (and is not)

### Pipeline we capacity-tested as “plain”

```
qidx → float32[384] → HNSW search (ef=64, k=10) → JSON hits
```

### Pipeline stages that exist in the roadmap but are **not** in the Go hot path

| Stage | Idea | Capacity outcome on 1M |
|---|---|---|
| 0 Flat | Exact IP | Correct GT; ~12 QPS |
| 1 HNSW | Single in-RAM graph | **Best simple path** |
| 2 PQ | Product quantization | Recall fail + slow smoke |
| 4 Partitioned | K graphs + nprobe | Quality OK; **QPS much worse** |
| 5 Graph manager | LRU partitions under RAM budget | Quality OK; **QPS much worse** |
| Cache / rescore | Extra layers on HNSW | **Lower QPS** under load |
| 8 Hybrid BM25+RRF | Lexical + dense | Smoke only; BM25 ~2s (Python) |
| 9 Rerank | Cross-encoder | Not gated under load |

**Go + Qdrant today = Stage-1-class ANN only** (HNSW inside Qdrant). No rerank.

---

## 4. Stage gates: quality before capacity

### Stage 0 — Flat exact

| Metric | Result |
|---|---|
| recall@1 / @10 / @100 | **1.0 / 1.0 / 1.0** (500 queries) |
| p50 / p99 latency | ~164 / ~241 ms |
| Gate | **Passed** (harness + GT valid) |

### Stage 1 — Naive HNSW (`hnswlib`)

| Config | recall@10 | p50 ms | p99 ms | Notes |
|---|---|---|---|---|
| Flat (500 q) | 1.000 | 164 | 241 | Reference |
| HNSW M=16, ef_construction=200, **ef=64** | **0.956** | **0.58** | **1.55** | Floor ≥ 0.90 |
| Index artifact | — | — | — | ~**1.6 GB** `hnsw.bin` |

**Gate: passed.**

### Stage 2 — PQ smoke (FAISS IndexPQ m=64)

| Metric | Result |
|---|---|
| recall@10 | **0.678** (failed floor) |
| p50 / p99 | **33.9 / 55.5 ms** (slower than HNSW) |
| Index size | ~64 MB |

**Gate: failed.** Naive PQ was both worse quality and slower — warning against “quantize = free speed.”

### Stage 4 — Partitioned multi-graph HNSW

Setup: K=256 partitions, overlap=1, M=16, ef=64.

| nprobe | recall@10 | p50 ms | p99 ms |
|---|---|---|---|
| 8 | 0.856 | ~4.8 | ~11 |
| **16** | **0.914** | ~9.3 | ~15 |
| 32 | 0.954 | ~18.8 | ~25 |
| 64 | 0.980 | ~38 | ~45 |

Partition size imbalance: min / median / max = **307 / 3651 / 11128**.

**Gate: passed at nprobe=16** (quality). Capacity later showed this is **not** a QPS win on 1M.

### Stage 5 — Graph manager (LRU, 8 GB budget)

| Metric | Warm zipf (500 q) |
|---|---|
| recall@10 | **0.956** |
| p50 / p99 | ~9.6 / ~19.5 ms |
| Gate | Passed functionally |

Cold start / load-on-miss later destroyed p99 under load tests.

### Stage 8 — Hybrid BM25 + dense + RRF (smoke only)

| Metric | Value |
|---|---|
| nDCG@10 / MRR@10 | ~0.64 / ~0.59 |
| Latency | **~2 s**/query (pure-Python BM25 over 1M) |
| Status | Smoke — not a capacity candidate until Tantivy (or similar) |

---

## 5. Python architecture and code

### 5.1 End-to-end request path

```
Client
  → ThreadingHTTPServer (1 process)  OR  N× SO_REUSEPORT workers
    → JSON parse (Python, GIL)
    → query_vectors[qidx]
    → pipeline.search(...)
         → HNSWIndex.knn_query  (hnswlib C++, releases GIL during search)
    → JSON serialize (Python, GIL)
  ← hits
```

### 5.2 HNSW index (algorithm core)

From `engine/index/hnsw.py` on `master`:

```python
class HNSWIndex:
    def __init__(self, vectors=None, space="ip", M=16,
                 ef_construction=200, ef_search=64, index_path=None):
        import hnswlib
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.n, self.dim = vectors.shape
        self._index = hnswlib.Index(space=space, dim=self.dim)
        self._index.init_index(
            max_elements=self.n, ef_construction=ef_construction, M=M
        )
        self._index.add_items(vectors, np.arange(self.n))
        self._index.set_ef(ef_search)

    def search(self, query, k: int):
        q = np.ascontiguousarray(query.vector.reshape(1, -1), dtype=np.float32)
        labels, distances = self._index.knn_query(q, k=k)
        return [
            Hit(doc_id=int(lab), score=float(-dist), source="dense")
            for lab, dist in zip(labels[0], distances[0])
        ]
```

Index-level search is sub-millisecond when warm. **Serving** is the bottleneck for QPS.

### 5.3 Single-process HTTP server

From `engine/serve_http.py`:

```python
httpd = ThreadingHTTPServer((host, port), Handler)

# each request thread:
body = json.loads(self.rfile.read(length))
vec = query_vectors[int(body["qidx"]) % len(query_vectors)]
hits = pipeline.search(
    Query(vector=np.asarray(vec, dtype=np.float32)),
    k=int(body.get("k", 10)),
)
self._json(200, {"hits": [{"doc_id": h.doc_id, "score": h.score, ...} for h in hits]})
```

### 5.4 Multi-process server (GIL escape hatch)

From `engine/serve_mp.py`:

```python
class ReuseThreadingHTTPServer(ThreadingHTTPServer):
    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        super().server_bind()

# Each worker process:
pipeline = build_pipeline(cfg)   # FULL private HNSW copy (~1.8 GB)
httpd = ReuseThreadingHTTPServer((host, port), make_handler(...))
httpd.serve_forever()
```

```bash
python3 -m engine.serve_mp --config configs/stage01_hnsw.yaml --workers 4 --port 8080
```

### 5.5 Python → Qdrant proxy

```python
# scripts/serve_qdrant_proxy.py
res = client.query_points(
    collection_name="msmarco_1m",
    query=vec,
    limit=k,
    search_params={"hnsw_ef": 64},
)
```

Same HTTP API; ANN runs in Qdrant (Rust). Python still owns accept/JSON/GIL.

---

## 6. Experiment 1 — Single-process capacity

**Server:** `engine/serve_http.py` (threaded, 1 process)  
**Artifacts:** `results/capacity/`, `reports/capacity.md`

### Headline table (sustainable / noted)

| System | Sustainable QPS | SLO used | Server RSS | Notes |
|---|---|---|---|---|
| **Our HNSW** | **~960** | p99≤100ms @ c=8 | **~1.8 GB** | Best dense single-proc path early |
| Our HNSW (exhaust) | ~1500 peak | none | ~1.8 GB | p99 → seconds at high c |
| Autocannon HNSW c=64 | ~1668 req/s | — | — | Overloaded / optimistic |
| Partitioned nprobe=16 | **~174** | p99≤200ms | ~2.4 GB | All partitions in RAM |
| Graph manager LRU 8GB | **~166** | p99≤300ms | ~2.4 GB | After warm |
| HNSW + rescoring | **~170** | p99≤100ms | ~3.2 GB | Needs fp32 vectors |
| HNSW + result/semantic cache | ~231 | p99≤100ms | ~1.8 GB | Zipf helps; locks hurt |
| Flat (FAISS) | **~12** | — | ~3.0 GB | Unusable for load |
| **Qdrant** (Python proxy) | **~403** | p99≤100ms @ c=8 | Qdrant ~**175 MB** cold | RAM king, QPS weak via proxy |

### Raw JSON peaks (Exp 1 files)

| File | Peak QPS | c | p99 ms |
|---|---|---|---|
| `hnsw.json` | 1557 | 4 | 4.4 |
| `hnsw_exhaust_v2.json` | 1547 | 8 | 6.5 |
| `qdrant.json` | 408 | 4 | 15.9 |
| `hnsw_cache.json` | 231 | 8 | 90.6 |
| `hnsw_rescored.json` | 175 | 8 | 110.7 |
| `partitioned_v2.json` | 176 | 4 | 33.4 |
| `graph_manager_v2.json` | 168 | 4 | 34.8 |
| `partitioned.json` (early) | 79 | 4 | 105 |
| `graph_manager.json` (cold-ish) | 61 | 4 | 270 |
| `flat.json` | 12 | 4 | 533 |
| `go_qdrant.json` (cold smoke) | 342 | 8 | **523** | cold start anomaly |

### Exp 1 diagnosis

| Observation | Meaning |
|---|---|
| ~1.8 GB RSS, ~29 GB still free | **Not RAM-bound** for 1M HNSW |
| QPS stuck ~1–1.5k | **CPU + GIL / one process** |
| Flat ~12 QPS | Exact search dead for capacity |
| Partitioned / GM ≪ HNSW | Extra probes cost more than they save on 1M |
| Cache / rescore ≪ plain HNSW | Extra Python work + locks |

---

## 7. The GIL problem (why threads were not enough)

### What people expect

`ThreadingHTTPServer` → many threads → use all 8 cores → ~8× QPS.

### What actually happens

```
┌─────────────────────────────────────────────┐
│  One Python process                         │
│  ┌─────┐ ┌─────┐ ┌─────┐                    │
│  │ T1  │ │ T2  │ │ T3  │  ...               │
│  └──┬──┘ └──┬──┘ └──┬──┘                    │
│     │       │       │                       │
│     ▼       ▼       ▼                       │
│   JSON / routing / numpy glue  ←── GIL      │
│     │       │       │                       │
│     ▼       ▼       ▼                       │
│   hnswlib C++ knn_query  ←── can run parallel│
│     (GIL released in native code)           │
└─────────────────────────────────────────────┘
```

- Native HNSW work **can** overlap across threads.
- Request parsing, `pipeline` glue, JSON, stats locks still **serialize** on the GIL.
- Net effect: **sublinear** scaling; we plateaued near **~1k sustainable** / **~1.5k exhaust** in one process.

### Proof that GIL (process count) mattered

When we moved to **multi-process** (separate interpreters + separate indexes), QPS jumped toward **~2.1k** — until **cores and RAM** became the limits. That is the classic “GIL escape via processes” pattern.

---

## 8. Experiment 2 — Multi-process SO_REUSEPORT

**Server:** `python3 -m engine.serve_mp --workers N`  
**Artifacts:** `results/capacity_mp/`, `reports/capacity_mp.md`

### HNSW primary scaling curve

| Workers | Best under p99≤100ms (report) | Peak QPS (JSON) | Approx RAM |
|---|---|---|---|
| 1 | **~1507** @ c=16 | **1622** | ~1.8 GB |
| 2 | **~2013** @ c=32 | **2158** | ~3.6 GB |
| 4 | **~2139** @ c=64 | **2275** | ~7 GB |
| 8 | **~2138** @ c=64 | **2299** | **~14 GB** |

Autocannon HNSW w=4, c=64, 20s: ~**4211** req/s (fixed-body optimistic).

### Partitioned (nprobe=16) under MP

| Workers | Best QPS | SLO note |
|---|---|---|
| 2 | **~305** | p99≤200ms class |
| 4 | **~477** | still ≪ HNSW |

### Graph manager under MP

| Workers | Best QPS |
|---|---|
| 2 | **~290** |
| 4 | **~448** |

### HNSW + cache under MP w=4

| Result | Detail |
|---|---|
| Peak | ~**386** QPS |
| p99≤100ms | **Failed** (p99≈116 ms on first step) |
| Issue | Private caches per worker → diluted hits + lock overhead |

### Qdrant under Exp 2 probes

| Probe | Result |
|---|---|
| Proxy c=8 p99≤100 | ~**403** QPS, ~175 MB |
| Autocannon c=64 | ~**398** req/s, timeouts |
| `qdrant_high.json` | Collapsed (~**2.8** QPS, p99 ~6.6 **seconds**) — overload / timeout death spiral |

### Exp 2 issues (HNSW MP)

| Issue | Detail | Impact |
|---|---|---|
| Diminishing returns after w=4 | w8 ≈ w4 QPS on 8 cores | Wasted RAM |
| RAM × workers | Full HNSW copy per process | **Memory wall at w=8** |
| Latency cliff | Client concurrency ≫ useful parallelism | p99 → ~1s |
| No shared memory index | Each worker private | Cannot share 1.8 GB graph |
| Still one machine | No horizontal story | Caps at ~2.1k here |

### Cross-system bottleneck table (Exp 2)

| Bottleneck | Who hits it first |
|---|---|
| Python GIL / single process | Exp 1 HNSW (~1k) |
| Cores saturated + index copies | Exp 2 HNSW w≥4 (~2.1k plateau) |
| RAM × workers | HNSW w=8; partitioned/GM w=4 |
| Probe fan-out (nprobe) | Partitioned / GM |
| Disk / cold cache | Graph manager cold start |
| On-disk ANN + Python proxy | Qdrant ~400 QPS |
| Client overload / timeouts | Autocannon on slow backends; `qdrant_high` |

**Exp 2 verdict:** MP **~2×** HNSW QPS (1.5k → 2.1k) with sweet spot **w=4**. Beyond that: little QPS, lots of RAM.

---

## 9. Qdrant path: Python proxy → Go searchd

### Collection (shared)

| Setting | Value |
|---|---|
| Name | `msmarco_1m` (later also `msmarco_1m_ram`) |
| Points | 1,000,000 |
| Distance | Cosine |
| HNSW | M=16, ef_construct≈100–200 |
| Vectors on_disk (original) | **true** |
| Docker | `qdrant/qdrant:v1.13.2`, ports 6333/6334 |

### Why Python→Qdrant was slow (~400 QPS)

| Factor | Effect |
|---|---|
| Python ThreadingHTTPServer + GIL | Front-door serialization |
| Sync client + JSON | Extra overhead per query |
| On-disk vectors | Lower than in-RAM hnswlib |
| Misread | We first blamed Qdrant; Go proved Qdrant could do **~4×** more |

### What Go changed

| Before | After |
|---|---|
| Python proxy ~403 QPS | Go `searchd` ~**1640** QPS (p99≤100ms) |
| Proxy RSS tied to Python | searchd ~**52 MB** |
| Same Qdrant | Same Qdrant (now the limiter) |

---

## 10. Go architecture and code

### Layout

```
go/cmd/searchd          HTTP + SO_REUSEPORT flag
go/internal/server      /health /stats /search → Qdrant gRPC
go/cmd/loadtest         concurrency ramp
go/cmd/ingest           rebuild collections (RAM / shards)
scripts/bench_go_qdrant.sh
scripts/bench_exhaust.sh
scripts/run_go_searchd_mp.sh
```

### Search handler

```go
vec := vectors.Row(s.queries, s.nQueries, s.dim, *req.QIdx)

res, err := s.client.Query(ctx, &qdrant.QueryPoints{
    CollectionName: s.cfg.Collection,
    Query:          qdrant.NewQueryDense(vec),
    Limit:          qdrant.PtrOf(uint64(req.K)),
    WithPayload:    qdrant.NewWithPayloadEnable(false),
    Params: &qdrant.SearchParams{
        HnswEf: qdrant.PtrOf(s.cfg.EF),
    },
})
```

### Multi-process listen (optional)

```go
unix.SetsockoptInt(int(fd), unix.SOL_SOCKET, unix.SO_REUSEPORT, 1)
ln, err := lc.Listen(ctx, "tcp", *addr)
httpSrv.Serve(ln)
```

### Concurrency model difference

| | Python | Go |
|---|---|---|
| Unit of concurrency | Thread (+ GIL) or process | Goroutine (no GIL) |
| Front-door cost | High | Low (~52 MB RSS) |
| To use 8 cores on in-proc HNSW | **N processes × N RAM** | N/A on this branch (Qdrant owns ANN) |
| To use 8 cores with Qdrant | Proxy often limited first | Goroutines feed Qdrant until **Qdrant CPU** saturates |

---

## 11. Go capacity + exhaust matrix

### Fair loadtest (rotating qidx), p99≤100ms

| c | QPS | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|
| 8 | 1290 | 5.7 | 10.6 | 14.2 |
| 16 | 1410 | 10.8 | 18.5 | 23.2 |
| 32 | 1543 | 20.2 | 31.7 | 38.3 |
| 64 | 1617 | 39.2 | 57.1 | 65.8 |
| **96** | **1640** | 58.0 | 81.5 | **96.6** |
| 128 | 1645 | 77.2 | 105.9 | 126 → **stop** |

**Sustainable: ~1640 QPS @ c=96.**  
**Plateau even without SLO: ~1620–1650 QPS** (extra concurrency only inflates latency).

### Resources (warm / under load)

| Component | Memory |
|---|---|
| searchd | ~**52 MB** RSS |
| Qdrant under load | ~**500–650 MB** |
| Total vs Python HNSW MP w=4 | ~**0.7 GB** vs ~**7 GB** |

### Autocannon (fixed body — optimistic)

| c | Avg req/s | p99 latency |
|---|---|---|
| 64 | ~**2988** | ~39 ms |
| 128 | ~**2759** | ~78 ms |
| 256 (exhaust finale) | ~**2786** | ~177 ms |

### Exhaust matrix (`results/go_exhaust/`)

| Config | Peak QPS | c | p99 ms | Lesson |
|---|---|---|---|---|
| **w4 ef=32 RAM** | **1792** | 512 | 483 | Best exhaust; latency bad |
| w8 ef=32 RAM | 1776 | 512 | 521 | Same plateau |
| w1 ef=64 disk | 1713 | 256 | 226 | Strong; simple wins |
| w4 ef=64 disk | 1654 | 256 | 237 | More Go workers ≠ more QPS |
| w1 ef=64 RAM (4 shards) | 1229 | 384 | 494 | Sharded RAM coll. slower here |
| w2/w4/w8 ef=64 RAM | ~1194–1210 | high | 519–670 | Worse than disk coll. |

**Exhaust verdict:** More Go processes do **not** unlock large gains. Lowering `ef` buys a little peak QPS. Qdrant remains the CPU wall (~700% CPU observed while searching).

### Cold-start issue (early Go smoke)

| Run | QPS | p99 |
|---|---|---|
| First short ramp (cold) | ~342 | **523 ms** |
| After warm | ~1300+ @ c=8 | ~14 ms |

**Issue:** Never publish cold-start ramps as capacity truth.

---

## 12. Master comparison tables

### A. All serving stacks (headline)

| Stack | Sustainable QPS | Peak | RAM | Wall |
|---|---|---|---|---|
| Flat exact | ~12 | — | ~3 GB | CPU (brute force) |
| Python→Qdrant proxy | ~403 | ~400 | ~175 MB cold | Proxy + GIL |
| Python HNSW 1-proc | ~960–1507 | ~1550–1620 | ~1.8 GB | GIL / 1 proc |
| **Go + Qdrant** | **~1640** | ~1650–1792 | ~**0.7 GB** | Qdrant CPU |
| **Python HNSW MP w=4** | **~2140** | ~2275 | ~**7 GB** | 8 cores; RAM at w8 |

### B. Quality vs speed (index level)

| Index | recall@10 | Typical latency | Capacity role |
|---|---|---|---|
| Flat | 1.00 | ~164 ms | GT only |
| HNSW ef=64 | 0.956 | ~0.6 ms | Production simple path |
| Partitioned nprobe=16 | 0.914+ | ~9 ms | Scale/RAM designs |
| Graph manager warm | 0.956 | ~10 ms | When full graph won’t fit |
| PQ m=64 smoke | 0.678 | ~34 ms | Failed |

### C. “Fancy” layers vs plain HNSW (1-proc load)

| System | QPS vs plain HNSW |
|---|---|
| Plain HNSW | baseline |
| + cache | **worse** (~231) |
| + rescore | **worse** (~170) |
| Partitioned | **worse** (~174) |
| Graph manager | **worse** (~166) |
| Hybrid BM25 (smoke) | not capacity-viable (~2s) |

### D. Python vs Go for the **same** Qdrant

| Front door | QPS | Δ |
|---|---|---|
| Python proxy | ~403 | 1× |
| Go searchd | ~1640 | **~4.1×** |

### E. Cost of another ~500 QPS (MP HNSW vs Go+Qdrant)

| | Go+Qdrant | Python HNSW w=4 |
|---|---|---|
| QPS | ~1640 | ~2140 (+~500) |
| RAM | ~0.7 GB | ~7 GB (**~10×**) |

---

## 13. Issue catalog (everything that hurt us)

Operational and technical issues encountered across the project, with impact.

### Infrastructure / data

| # | Issue | What happened | Impact | Mitigation |
|---|---|---|---|---|
| I1 | Data disk not mounted | Large downloads/embeddings needed `data/` on xvdf | Blocked Stage 0 | Format + mount 300GB disk |
| I2 | Embedding time / CPU | 1M×384 BGE on CPU is slow | User interrupted; needed resume | 50k checkpoint flushes + progress log |
| I3 | No swap | Large peaks have no safety net | OOM risk at HNSW w=8 | Cap workers; watch `free` |
| I4 | `pip` / toolchain gaps | `hnswlib` needed g++ | Build failures | `dnf` install compilers |

### Measurement / process hygiene

| # | Issue | What happened | Impact | Mitigation |
|---|---|---|---|---|
| I5 | `pkill` matched script cmdline | Capacity jobs killed mid-run | Lost runs | Prefer `fuser -k port` / kill by PID file |
| I6 | Cold vs warm confusion | Go first ramp 342 QPS / p99 523 | False “Qdrant is slow” | Always warm; discard cold |
| I7 | Autocannon fixed body | Same `qidx` forever | Inflated QPS (2.8–4k) | Report zipf/`qidx` ramp as fair |
| I8 | SLO vs peak mixing | Exhaust QPS quoted as sustainable | Overclaim | Separate tables |
| I9 | Graph manager cold p99 | First loads from disk | Early GM numbers looked dead (~61 QPS) | Warm + v2 rerun |

### Python / GIL / MP

| # | Issue | What happened | Impact | Mitigation |
|---|---|---|---|---|
| I10 | **GIL + threaded server** | Threads ≠ multi-core for Python glue | ~1k QPS ceiling | `serve_mp` processes |
| I11 | **RAM × workers** | Each worker full HNSW | ~14 GB at w=8 | Sweet spot w=4 |
| I12 | Private caches under MP | Each process own cache | Cache MP failed SLO | Shared cache needs shm/redis |
| I13 | Stats lock contention | Thread-safe latency lists | Minor overhead | Keep sampling bounded |
| I14 | Pure-Python BM25 | Hybrid smoke ~2s/query | Unusable under load | Tantivy / C++ lexical later |

### Index design

| # | Issue | What happened | Impact | Mitigation |
|---|---|---|---|---|
| I15 | Partition imbalance | Sizes 307–11128 | Uneven latency | Accept or rebalance centroids |
| I16 | nprobe cost | Higher nprobe → higher recall **and** latency | QPS tanks vs single HNSW | Use only when RAM forces split |
| I17 | PQ without care | recall 0.68, slower | Failed Stage 2 | HNSW+quant+rescore if revisit |
| I18 | Rescore path | Extra fp32 touches | ~170 QPS | Skip for plain capacity |

### Qdrant / API / Go

| # | Issue | What happened | Impact | Mitigation |
|---|---|---|---|---|
| I19 | `client.search` removed | qdrant-client API rename | Proxy breakage | `query_points` / Go `Query` |
| I20 | Python proxy bottleneck | Blamed Qdrant for 400 QPS | Wrong architecture choice risk | Rewrite front door in Go |
| I21 | PATCH `on_disk` API shape | 400 Bad Request on naive PATCH | Couldn’t flip disk→RAM in place | Recreate collection via `ingest` |
| I22 | Overload death spiral | `qdrant_high` → ~3 QPS, multi-second p99 | Looks like total failure | Backpressure; don’t past plateau |
| I23 | Go MP no gain | w1≈w4≈w8 into Qdrant | Wasted complexity | Single searchd enough |
| I24 | 4-shard RAM collection slower | Peak ~1.2k vs ~1.7k disk | Surprising regression | Prefer simpler shard layout here |
| I25 | Errors under huge autocannon | searchd `errors` counter rose in long suite | Need monitoring | Timeouts / context cancel |

### Product / expectation issues

| # | Issue | What happened | Impact | Mitigation |
|---|---|---|---|---|
| I26 | “Add cache/partitions for QPS” | Intuition vs measurement | Would **reduce** QPS on 1M | Measure first |
| I27 | “Quantize for huge QPS” | Stage 2 smoke failed | False hope | Quant for RAM; expect ~1.5–2× QPS max |
| I28 | “100k QPS on one box” | Math: need ~50–80× this host | Unrealistic | Horizontal replicas |

### Hardware / network (local tests didn’t hit these; production will)

| # | Issue | What happened / risk | Impact | Mitigation |
|---|---|---|---|---|
| I29 | Colocated-only benches | All QPS on localhost | Hides NIC/RTT walls | Add remote-Qdrant soak before prod |
| I30 | Cross-AZ / cross-region DB | +1–80 ms RTT on every search | Destroys p99≤100ms | Same-AZ; cache; relax SLO |
| I31 | Raw vector over HTTP at high QPS | ~3 KB/req → ~2.4 Gbit/s @ 100k | NIC / LB saturation | `qidx`/id lookup; binary proto; 10 GbE |
| I32 | Snapshot / reindex transfer | Multi-GB index moves | Deploy windows blow up | Object storage + 10/25 GbE |
| I33 | Short-lived connections | SYN/PPS storm at high QPS | Soft NIC / LB failure | Keep-alive, HTTP2, gRPC pools |
| I34 | No swap + MP copies | w=8 ~14 GB on 31 GB box | OOM / reclaim jitter | Cap workers; dedicated mem |
---

## 14. Walls: CPU, RAM, GIL, disk, proxy

```
                    free RAM
                       │
     Exp1 HNSW ────────┤  still ~29 GB free  →  NOT memory wall
                       │
     Exp2 w=4 ─────────┤  ~7 GB used         →  OK
                       │
     Exp2 w=8 ─────────┤  ~14 GB + pressure  →  MEMORY WALL
                       │
     Go+Qdrant ────────┤  ~0.7 GB            →  RAM comfortable
                       │
                    CPU / GIL
                       │
     Python 1-proc ────┤  GIL + 8 cores underused → GIL WALL
     Python MP w=4 ────┤  cores full              → CPU WALL (~2.1k)
     Go+Qdrant ────────┤  Qdrant ~7× CPU          → CPU WALL (~1.6k)
     Go many workers ──┤  no extra QPS            → already at CPU WALL
```

| Wall name | Symptom | Approx QPS at wall | Fix |
|---|---|---|---|
| Brute-force CPU | Flat search | ~12 | HNSW |
| GIL / 1 process | Free RAM, stuck QPS | ~1.0–1.5k | Multi-process |
| Core saturation | w4≈w8 | ~2.1k | More machines |
| Memory × copies | w8 ~14 GB | same QPS as w4 | Fewer workers |
| Proxy language | Qdrant underused | ~400 | Go front door |
| Qdrant ANN CPU | Go plateau | ~1.6k | Replicas / bigger box |
| Disk cold cache | GM p99 spikes | unstable | Warm / mmap / keep hot |

---

## 15. What improved results (and what did not)

### Improvements that worked

| Change | From → To | Why it worked |
|---|---|---|
| Flat → HNSW | 12 → ~960+ QPS | ANN |
| HNSW 1-proc → MP w=4 | ~1.5k → ~2.1k | Bypass GIL; use cores |
| Python Qdrant proxy → Go | ~400 → ~1640 | Remove GIL proxy; gRPC |
| Warm-up before measure | 342 → 1300+ @ c=8 | Page cache / JIT / connections |
| ef 64 → 32 (exhaust) | ~1.6k → ~1.8k peak | Less graph work (recall trade) |

### Changes that did **not** improve QPS on 1M

| Change | Result |
|---|---|
| More threads in one Python process | Plateau |
| Partitioned multi-graph | Much lower QPS |
| Graph manager | Much lower QPS |
| Result/semantic cache (Python) | Lower / failed SLO |
| fp32 rescore layer | Lower QPS |
| More Go searchd workers | Flat plateau |
| Naive PQ | Worse recall + latency |
| Hybrid BM25 (pure Python) | Seconds per query |

---

## 16. Hardware limitations (deeper)

Our box looked “big” (31 GB RAM) but capacity was never a pure software story. Hardware layers bound us in a fixed order.

### 16.1 The resource stack (what actually constrained us)

| Layer | What it does in retrieval | On this box | Did it limit us? |
|---|---|---|---|
| **CPU cores / clocks** | HNSW distance + graph walk; JSON; gRPC | 8 vCPU | **Yes — primary wall** (~1.6–2.1k QPS) |
| **DRAM capacity** | Hold graph + vectors + OS page cache | 31 GB | Only when **copying** indexes (MP w=8) |
| **DRAM bandwidth** | Feed SIMD distance loops | Typical cloud ~50–100+ GB/s shared | Secondary; shows up with fat in-RAM HNSW |
| **Disk (NVMe)** | Cold load, on-disk Qdrant vectors, checkpoints | Fast local NVMe | Cold GM / first Qdrant warm; not steady-state HNSW |
| **NIC / network** | Client↔API; API↔remote DB | Localhost in our tests | **Not hit here**; becomes real when DB is off-box |
| **Swap** | Emergency paging | **None** | No soft landing — OOM or reclaim only |

### 16.2 Why 8 vCPUs capped us (back-of-envelope)

At Go+Qdrant sustainable ~**1640 QPS**, mean service time under load at c=96 was ~**58 ms** *waiting in the system* (queueing), while unloaded p50 was ~**6 ms**.

Little’s law (steady state):

\[
L \approx \lambda \cdot W
\]

Example at λ = 1640 QPS, W ≈ 0.058 s → L ≈ **95** requests in flight — matches our best concurrency step (**c=96**).

If each useful search needs roughly one “core-slice” of ANN work and Qdrant reported ~**7×** CPU under load:

\[
\text{QPS}_{\max} \lesssim \frac{\text{busy cores} \times 1000}{\text{CPU-ms per query}}
\]

With ~7 cores busy and ~1640 QPS → **~4.3 CPU-ms/query** of Qdrant work (order-of-magnitude). Doubling cores (16 vCPU), if linear, suggests ~**3k QPS** class — not 10k — unless the ANN itself gets cheaper (`ef`, quant, smaller k).

**Hardware lesson:** buying RAM alone would not have moved the Go+Qdrant plateau; **more cores (or more Qdrant replicas)** would.

### 16.3 Memory math for this corpus

| Object | Formula | Size |
|---|---|---|
| fp32 vectors | 1e6 × 384 × 4 | **~1.46 GB** |
| HNSW graph (hnswlib artifact) | measured | **~1.6 GB** |
| Python HNSW process RSS | measured | **~1.8 GB** |
| ×4 MP workers | 4 × ~1.8 | **~7 GB** |
| ×8 MP workers | 8 × ~1.8 | **~14 GB** → pressure on 31 GB |
| Qdrant under load | measured | **~0.5–0.65 GB** (on-disk vectors + graph + cache) |
| Query matrix (6980 × 384 × 4) | — | **~10.7 MB** (tiny) |

**Working set vs capacity:** 1M fits comfortably once. It stops fitting when software **duplicates** the working set per process.

### 16.4 Disk / page cache effects we saw

| Situation | Hardware effect | Symptom |
|---|---|---|
| First Go/Qdrant ramp | Cold page cache / cold connections | ~342 QPS, p99 **523 ms** |
| After warm | Vectors/graph in RAM cache | ~1300+ QPS @ c=8, p99 ~14 ms |
| Graph manager cold | Partition fault-in from NVMe | Early ~61 QPS, p99 ~270 ms |
| Embeddings / ingest | Sequential NVMe write + CPU | Long wall-clock; needed 50k checkpoints |
| No swap | Cannot hide RAM overcommit | Must size workers to RAM |

NVMe is fast, but **random cold HNSW touches** still spike p99. Steady-state capacity numbers are **warm-cache** numbers unless stated.

### 16.5 Hypervisor / noisy neighbor (cloud reality)

We ran on a cloud VM. Unmeasured but real for production:

| Effect | How it shows up |
|---|---|
| CPU steal / throttling | p99 grows without code changes |
| Shared NIC | Bandwidth and PPS limits below “spec” |
| EBS vs local NVMe | Remote disks add ms to cold paths |
| Credit-based burstable SKUs | Great smoke tests, false capacity |

**Rule:** treat one-box QPS as **±20–30%** soft unless you pin dedicated CPU / local SSD.

### 16.6 Hardware × software interaction matrix

| If you only upgrade… | Likely QPS effect on *this* design |
|---|---|
| RAM 31 → 64 GB | Helps **MP HNSW w=8+**; little for Go+Qdrant plateau |
| 8 → 16 vCPU | **Best single-knob** for Qdrant / HNSW |
| Faster NVMe | Helps cold start & on-disk vectors; small for warm HNSW |
| GPU | Helps **embedding**, not our `qidx` search path |
| 10/25/100 GbE | Matters only when **DB or clients are remote** (next section) |

---

## 17. When the DB is remote: network I/O & bandwidth math

All published QPS here used **localhost** (searchd ↔ Qdrant on the same machine). Production often puts the vector DB on another host or in another AZ. Then the **NIC** joins the wall list.

### 17.1 Deployment shapes

```
A) What we measured (colocated)
   Client ──HTTP──► searchd ──gRPC──► Qdrant     (loopback, µs RTT)

B) Split tier (common)
   Client ──HTTP──► searchd ──gRPC/HTTP──► Qdrant (other host)
                      ▲                         ▲
                   NIC #1                    NIC #2

C) Clients far away
   Clients ──WAN/LB──► searchd ──private net──► Qdrant
```

Every extra hop adds **RTT + serialization + bandwidth**.

### 17.2 Bytes per search (measured / derived)

From autocannon on this API: ~**2.04 MB/s** at ~**2988 req/s** ⇒ response ≈ **683 bytes**/req (top‑10 JSON hits).

| Path | Payload | Approx size |
|---|---|---|
| HTTP request (`qidx` only) | `{"qidx":N,"k":10}` | **~23–40 B** |
| HTTP request (raw vector) | 384 floats JSON | **~2.3 KB** |
| HTTP response (k=10) | hits JSON | **~650–700 B** |
| gRPC query vector | 384 × fp32 | **1536 B** raw (+ protobuf framing) |
| gRPC result | 10 × (id + score) | **~100–300 B** + framing |
| **Fair total HTTP (qidx)** | req+resp | **~700–750 B**/query |
| **HTTP with vector in body** | req+resp | **~3.0 KB**/query |
| **searchd↔Qdrant gRPC** | vector + hits | **~1.8–2.2 KB**/query (order) |

### 17.3 Bandwidth formulas

Let \(Q\) = queries/sec, \(B\) = bytes per query (both directions on that link).

\[
\text{MB/s} = \frac{Q \times B}{10^6}, \quad
\text{Gbit/s} = \frac{Q \times B \times 8}{10^9}
\]

Add **~20–40%** overhead for TCP/TLS/HTTP2 framing, retries, and stats — or use a **1.5× safety factor** for planning.

### 17.4 Bandwidth tables (planning)

**Link 1 — Clients → searchd (HTTP, qidx style, ~720 B/query):**

| Target QPS | Payload MB/s | ≈ Gbit/s raw | Plan with 1.5× |
|---|---|---|---|
| 1,640 (our Go SLO) | 1.2 | **0.009** | trivial (≪1 GbE) |
| 2,140 (HNSW MP) | 1.5 | 0.012 | trivial |
| 10,000 | 7.2 | 0.058 | still fine on 1 GbE |
| 50,000 | 36 | 0.29 | comfortable on 1 GbE |
| **100,000** | **72** | **0.58** | still **<1 GbE** for *this tiny JSON* |

**Link 1b — Clients send full vectors (~3.0 KB/query):**

| Target QPS | MB/s | ≈ Gbit/s | Plan 1.5× |
|---|---|---|---|
| 1,640 | 4.9 | 0.039 | fine |
| 10,000 | 30 | 0.24 | fine |
| 100,000 | **300** | **2.4** | need **≥5–10 GbE** or compress/binary |

**Link 2 — searchd → remote Qdrant (gRPC ~2 KB/query):**

| Target QPS | MB/s | ≈ Gbit/s | Plan 1.5× | Notes |
|---|---|---|---|---|
| 1,640 | 3.3 | 0.026 | ~0.04 | same-AZ 1 GbE OK |
| 10,000 | 20 | 0.16 | ~0.24 | 1 GbE OK if not shared heavily |
| 50,000 | 100 | 0.80 | ~1.2 | **≥10 GbE** safer |
| **100,000** | **200** | **1.6** | **~2.4** | **10 GbE minimum**; 25 GbE comfortable |

### 17.5 Latency budget if Qdrant is off-box

Unloaded colocated search p50 ≈ **6 ms**. That budget is mostly ANN.

| Path | Typical added RTT | Effect on p50 / p99 |
|---|---|---|
| Loopback | ~0.01–0.05 ms | Ignorable |
| Same AZ, same VPC | ~0.2–0.5 ms | Small |
| Cross-AZ | ~1–2 ms | Eats SLO headroom |
| Cross-region | ~20–80+ ms | **Breaks p99≤100ms** easily under load |

If you need p99 ≤ 100 ms and ANN already uses ~40–90 ms under load, **cross-region DB is incompatible** with this SLO without caches or relaxed SLOs.

**Queueing reminder:** remote RTT adds to service time \(W\). Same λ with larger \(W\) ⇒ either lower sustainable QPS or higher concurrency (worse tails).

Rough capacity derate if you add \(d\) ms fixed RTT to a system that was latency-bound:

\[
QPS_{\text{remote}} \approx QPS_{\text{local}} \times \frac{W_{\text{local}}}{W_{\text{local}} + d/1000}
\]

Example: \(W=0.006\) s, \(d=2\) ms → factor \(6/8=0.75\) → **~1640 → ~1230 QPS** before even counting bandwidth.

### 17.6 Packets/sec and connection mechanics

Bandwidth is not the only NIC limit — **PPS** and **connections** matter.

| Item | At 100k QPS (order) |
|---|---|
| HTTP requests/sec | 100k |
| If short-lived TCP/request | can require **huge** SYN rate — **don’t**; use keep-alive / HTTP2 / gRPC streams |
| Concurrent connections | For c≈ concurrency; at scale use LB with pooling |
| gRPC to Qdrant | Multiplex many RPCs on few connections (**required**) |

We already configured Go `http.Transport` with high `MaxIdleConnsPerHost` in loadtest — same idea must exist **searchd → Qdrant** and **LB → searchd**.

### 17.7 Ingest / index replication bandwidth (often forgotten)

Serving QPS is small JSON. **Building and syncing** indexes is the heavy network path.

| Transfer | Size | Time on 1 GbE (~100 MB/s useful) | Time on 10 GbE |
|---|---|---|---|
| fp32 vectors 1M | ~1.5 GB | ~15–20 s | ~2 s |
| HNSW artifact | ~1.6 GB | ~20 s | ~2 s |
| Qdrant snapshot / full reindex ship | few GB | minutes | tens of seconds |
| 30M vectors fp32 | ~44 GB | ~7–10 min | ~1 min |
| 100k QPS × 60 replicas resync | — | operationally painful | need object store + snap

**Issue:** teams provision NICs for query traffic (easy) and then fail on **snapshot restore / blue-green index swap**.

### 17.8 If DB is outside: recommended network design

| QPS class | Client→API | API→Qdrant | Placement |
|---|---|---|---|
| ≤5k | 1 GbE | 1 GbE | Same AZ |
| 5–20k | 1–10 GbE | **10 GbE** | Same AZ, pooled gRPC |
| 20–100k | 10 GbE + LB | **10–25 GbE**, sharded/replicated Qdrant | Same AZ; avoid cross-region |
| + embeddings | often dominates | binary protobuf, batch | GPU tier separate |

Also budget:

- **TLS CPU** (can steal cores from ANN if terminated on searchd)
- **LB bandwidth** equal to client aggregate
- **AZ egress $$** if cross-AZ (cloud bill wall ≠ NIC wall)

### 17.9 Did network limit *our* experiments?

| Link | In our runs | Limit? |
|---|---|---|
| Client → server | localhost / same host | **No** |
| searchd → Qdrant | localhost gRPC | **No** |
| Disk | local NVMe | Cold path only |

So every wall we hit was **CPU / GIL / RAM copies**, not NIC. The tables above are for **the next failure mode** once you split tiers or chase 100k QPS.

---

## 18. Deeper mechanics we needed to build this

Beyond “call HNSW,” the system required a stack of mechanics. These are the non-optional pieces that made experiments trustworthy.

### 18.1 Data & embedding mechanics

| Mechanic | Why required | What we did |
|---|---|---|
| Corpus subsetting | Full MS MARCO too big for first gates | 1M passage slice |
| Consistent embedding space | Mixed dims break indexes | BGE-small 384 everywhere |
| L2 normalize | Cosine ≡ IP | Normalize at embed time |
| Checkpointed embedding | CPU embeds take hours; crashes lose work | 50k flush + manifest |
| Separate query matrix | Load tests must not call the embedder | `queries.f32.npy` + `qidx` |
| Ground-truth flat index | Recall@k needs exact neighbor lists | FAISS IndexFlatIP k=100 |

### 18.2 Index mechanics

| Mechanic | Why | Notes |
|---|---|---|
| HNSW params (M, ef_construction, ef_search) | Recall/latency trade | M=16, ef_c=200, ef_s=64 default |
| Persist index (`hnsw.bin` + meta) | Don’t rebuild every run | ~1.6 GB |
| Qdrant collection lifecycle | Green status before load | Wait for `indexed_vectors_count` |
| Optional sharding / on_disk flags | RAM vs QPS experiments | 4-shard RAM coll. *hurt* here |
| Partition centroids + nprobe | Stage 4 | Quality OK, QPS bad on 1M |
| LRU graph manager | Stage 5 RAM budget | Cold-start p99 hazard |

### 18.3 Serving mechanics

| Mechanic | Python | Go |
|---|---|---|
| HTTP JSON API | `ThreadingHTTPServer` | `net/http` |
| `qidx` resolution | numpy row | `vectors.Row` subslice |
| Concurrency | threads / processes | goroutines |
| Multi-process accept | `SO_REUSEPORT` | same (`unix.SO_REUSEPORT`) |
| Backend protocol | in-proc hnswlib **or** HTTP Qdrant client | **gRPC** Qdrant |
| Timeouts | ad hoc | `context.WithTimeout` per search |
| Stats | locked lists + psutil | atomics + `runtime.MemStats` |

### 18.4 Load-test mechanics (why numbers are believable)

| Mechanic | Purpose |
|---|---|
| Concurrency ramp | Find SLO knee, not just max flood |
| Warmup step | Kill cold-cache lies |
| p50/p95/p99 | Tails matter more than mean |
| Error-rate cap | Don’t count timeout storms as QPS |
| Rotating `qidx` | Avoid single-key cache fantasy |
| Autocannon separate | Explicitly labeled optimistic |
| `fuser`/PID files | Avoid `pkill` collateral damage |
| Docker stats + RSS | Attribute CPU/RAM to the right process |

### 18.5 Correctness / gate mechanics

| Gate | Threshold | Role |
|---|---|---|
| Flat recall@10 | 1.0 | Harness trust |
| HNSW recall@10 | ≥ 0.90 | Ship floor |
| Stage 4 nprobe sweep | pick min nprobe hitting floor | Cost control |
| Capacity SLO | p99≤100ms | “Sustainable” definition |

Without gates, you can “win” QPS by destroying recall (PQ smoke: 0.68 recall).

### 18.6 OS mechanics we leaned on

| OS feature | Use |
|---|---|
| `SO_REUSEPORT` | MP accept balancing |
| Page cache | Warm Qdrant / mmap behavior |
| No swap | Forces honest RAM sizing |
| `docker` bridge + port publish | Qdrant isolation |
| `taskset` / CPU affinity (optional, unused) | Would reduce jitter in harder benches |

### 18.7 Mechanics we did *not* build (but 100k QPS would need)

| Missing piece | Why it matters at scale |
|---|---|
| Client-side / edge cache | Cut QPS to origin |
| Shared result cache (Redis/shm) | MP-safe hits |
| Admission control / queue limits | Prevent death spirals (`qdrant_high`) |
| Hedged requests / retries with budget | Tail latency |
| Service mesh mTLS offload | Save ANN CPU |
| Index blue/green + snapshot automation | Safe deploys |
| Multi-AZ failover with SLO math | Availability vs latency |
| Embed microservice + batching | Real user traffic |
| Autoscale on p99 not CPU avg | Match our actual wall |

### 18.8 End-to-end path (mechanics annotated)

```
[embed offline] → passages.f32.npy ─► build HNSW / upsert Qdrant
[embed offline] → queries.f32.npy  ─► loadtest picks qidx
                                         │
loadtest ─HTTP JSON─► searchd ─gRPC vector─► Qdrant HNSW
                         │                     │
                    goroutines              Rust threads
                    keep-alive               segments/shards
                         │                     │
                    JSON tops-k ◄──────────── ids+scores
                         │
                   p99 / QPS recorded
```

Every arrow is a place hardware (CPU, RAM, NIC, disk) can become the wall. On this box, arrows on **localhost** meant the wall sat inside **ANN CPU** and **Python process model**.

---

## 19. Scaling fantasy check: 100k QPS

From measured ~1640 (Go+Qdrant) or ~2140 (HNSW MP):

| Assumption | Math |
|---|---|
| Linear replicas of Go+Qdrant | 100000/1640 ≈ **61** boxes like this |
| With 75% efficiency | ~**80** boxes / ~**500–650 vCPU** |
| HNSW MP style | ~**47–55** fat-RAM nodes |
| Client HTTP (qidx) NIC | ~**0.6 Gbit/s** raw (~1 Gbit/s planned) — **not** the hard part |
| searchd→remote Qdrant | ~**1.6–2.4 Gbit/s** → plan **10 GbE** |
| Embedding at 100k QPS | **Separate, usually larger** (GPU) problem |

**Not solvable** by cache/partitions/quantize alone on one 8-vCPU host. Network only starts to dominate once you **split DB** or send **raw vectors** at very high QPS.

---

## 20. Final verdict

1. **Algorithm for capacity truth:** plain **HNSW top‑k**. No rerank in the numbers we trust.
2. **Highest QPS on this box:** Python **in-RAM HNSW, 4 processes** → **~2140** sustainable-class, ~7 GB RAM.
3. **Best QPS/RAM serving we shipped:** **Go → Qdrant** → **~1640** QPS at ~**0.7 GB**.
4. **Python’s main enemy:** the **GIL** for single-process serving; escape = **processes**, which hit a **memory wall** at w=8.
5. **Go’s main win vs Qdrant:** destroying the **Python proxy** bottleneck (~4×), not inventing a new ANN.
6. **Go’s ceiling here:** **Qdrant CPU**, not goroutines or searchd RAM.
7. **Hardware:** cores first; RAM when copying indexes; disk for cold paths; **NIC unused on localhost** but mandatory in the math once Qdrant is remote.
8. **Remote DB:** tiny JSON keeps client bandwidth easy even at 100k QPS; **gRPC to remote Qdrant** wants **10 GbE** class at that scale; cross-region RTT fights a 100 ms p99 SLO.
9. **Partitions / graph manager / Python caches:** useful for *fitting* big indexes, **harmful for QPS on 1M**.
10. **Ship recommendation:** Go `searchd` + Qdrant for efficient serving; keep Python HNSW MP notes as the “max QPS / max RAM” reference; design NICs/pooling before splitting tiers.

---

## 21. Artifact index

| Path | Contents |
|---|---|
| `reports/capacity.md` | Exp 1 single-process |
| `reports/capacity_mp.md` | Exp 2 multi-process |
| `reports/compare_go_qdrant.md` | Go vs Python headline |
| `reports/go_qdrant.md` | Go branch smoke/summary |
| `reports/stage00.md` … `stage08.md` | Quality gates |
| `results/capacity/*.json` | Exp 1 ramps |
| `results/capacity_mp/*.json` | Exp 2 ramps |
| `results/go_qdrant/` | Go SLO + exhaust + autocannon |
| `results/go_exhaust/` | Workers × ef × collection matrix |
| `go/cmd/searchd`, `go/internal/server` | Current serving code |
| `master:engine/serve_*.py`, `engine/index/hnsw.py` | Python serving (historical) |

---

*Generated from measured runs on the 8 vCPU / 31 GB MS MARCO 1M box. Prefer this file over scattered stage notes when explaining the full story.*
