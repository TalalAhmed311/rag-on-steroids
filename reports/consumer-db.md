# Consumer Vector DB (Laptop / SIMD-first)

Design notes for building **our own embeddable vector database** aimed at consumer apps on laptops — SIMD distance kernels, single-node, offline-friendly — informed by the MS MARCO 1M experiments in this repo (`reports/ARTICLE.md`).

---

## 1. Product thesis

Ship a **local library**, not a distributed Qdrant clone.

| Consumer DB | Server DB (what we ran with Go+Qdrant) |
|---|---|
| In-process / sidecar on the laptop | Separate daemon + gRPC |
| SIMD HNSW in one address space | Remote ANN + network |
| Optimize for **RAM fit + p99 UX** | Optimize for multi-tenant QPS |
| No Docker required | Docker/ops OK |
| Privacy / offline by default | Shared infrastructure |

**One-line pitch:** *Add / search / persist vectors on a laptop with one native index file, AVX/NEON-fast distances, and predictable recall.*

---

## 2. Why this repo says it’s feasible

Measured on **8 vCPU / 31 GB**, MS MARCO **1M × 384-d** BGE, plain HNSW (no rerank):

| Signal | Implication for laptops |
|---|---|
| In-RAM HNSW ~**1.6–1.8 GB** + sub-ms index search | 16 GB laptops can hold ~0.5–1M @ 384-d |
| Capacity wall was **CPU**, not NIC (localhost) | Local SIMD is the right bet |
| Python GIL forced **N× RAM** multi-process for ~2.1k QPS | Consumer core must be **native** (Rust/C++/Zig), one copy |
| Go+Qdrant ~**1640 QPS** @ ~0.7 GB | Fine for apps; overkill ops for “notes search on my Mac” |
| Partitions / graph-manager **hurt QPS** when data fits | Laptop default = **one graph** until RAM forces mmap/quant |
| PQ smoke failed recall (0.68) | Quantize carefully; always gate recall |

### Ballpark laptop QPS (order-of-magnitude)

Warm search, 384-d, ef≈64, k=10, native HNSW:

| Machine | Corpus | Expected behavior |
|---|---|---|
| 4–8 cores, 8–16 GB | 100k–500k | Comfortable interactive + light batch |
| 8 cores, 16–32 GB | ~1M | Similar to our single HNSW class (hundreds–1k+ QPS) |
| 8 GB RAM | >500k fp32 | Need **int8/PQ** or mmap — not full fp32 graph |

Interactive UI rarely needs >50–100 QPS; batch reindex / “search all photos” needs the headroom.

---

## 3. Non-goals (v1)

- Multi-node sharding / consensus  
- Python GIL-bound HTTP server as the core  
- Hybrid BM25 + cross-encoder rerank in the hot path (optional plugin later)  
- “100k QPS on a MacBook” marketing  
- Replacing cloud Qdrant for multi-tenant SaaS  

---

## 4. Target architecture

```
┌─────────────────────────────────────────────┐
│  App (Electron / mobile / CLI / Go service) │
│         thin binding (C ABI / FFI)          │
└────────────────────┬────────────────────────┘
                     ▼
┌─────────────────────────────────────────────┐
│  consumer-db engine (Rust or C++)           │
│  ┌─────────────┐  ┌──────────────────────┐  │
│  │ WAL + file  │  │ HNSW graph           │  │
│  │ format      │  │ + vector store       │  │
│  └─────────────┘  └──────────┬───────────┘  │
│                              │              │
│                    SIMD distance kernels    │
│                    (AVX2 / AVX-512 / NEON)  │
└─────────────────────────────────────────────┘
         ▲
         │ optional
    mmap / quant / background compact
```

### Process model

- **Default:** embed the engine in the app process (lowest latency, one RAM copy).  
- **Optional:** tiny local sidecar if the app language is hostile to FFI — still **one** index, not N Python workers.

---

## 5. Core technical design

### 5.1 Index

| Piece | v1 choice | Notes |
|---|---|---|
| Graph | **HNSW** | Same family as our Stage 1 winner |
| Metric | IP / Cosine (L2-normalize at insert) | Matches BGE-style embeds |
| Defaults | M=16, ef_construction=100–200, ef_search=32–64 | Tunable per device class |
| Small DB | Flat brute force below ~10–20k | Avoid HNSW overhead |
| Large DB | HNSW + **mmap** vectors or scalar/int8 quant | When RSS > ~30–40% of laptop RAM |

### 5.2 SIMD distance layer

This is the product’s “secret sauce” surface — not a new graph algorithm.

| ISA | Platform |
|---|---|
| AVX2 (FMA) | Most Intel/AMD laptops |
| AVX-512 | Some workstation / newer CPUs (runtime detect) |
| NEON | Apple Silicon / Android Arm |
| Scalar fallback | Always |

Kernels for: `f32` IP/L2, later `f16` / `int8` dot with rescore.

**What SIMD buys:** higher distance throughput inside HNSW candidate scoring.  
**What it doesn’t buy:** free memory; graph pointer-chasing still costs.

### 5.3 Memory modes (progressive)

| Mode | When | Trade |
|---|---|---|
| `ram` | Default small/medium | Fastest; RSS ≈ vectors + graph |
| `mmap` | Vectors or graph on SSD | Lower RSS; cold p99 spikes (we saw this with GM/Qdrant cold) |
| `quant` | Won’t fit fp32 | Smaller; must **rescore** top candidates to hold recall |

**Rule from experiments:** don’t introduce partitions until a single graph won’t fit. Partitions lost QPS badly on 1M that already fit.

### 5.4 Storage format (single-file goal)

Suggested layout (conceptual):

```
[header magic/version]
[metadata: dim, metric, count, params]
[vector block | quant codes]
[HNSW levels / neighbors]
[WAL / free list]
[footer checksum]
```

Requirements:

- Crash-safe **append + fsync** for inserts  
- Atomic replace on compact/optimize  
- Portable endianness / versioning  
- Optional encryption-at-rest (consumer privacy)

### 5.5 API sketch (C ABI + idiomatic wrappers)

```c
cdb_db* cdb_open(const char* path, cdb_options opts);
int     cdb_add(cdb_db*, uint64_t id, const float* vec, size_t dim);
int     cdb_search(cdb_db*, const float* q, int k, int ef,
                   cdb_hit* out, int* n_out);
int     cdb_delete(cdb_db*, uint64_t id);
int     cdb_flush(cdb_db*);
void    cdb_close(cdb_db*);
```

Wrappers: Go, Python (for tooling only), TypeScript (WASM or native addon).

HTTP is optional for debugging — **not** the primary laptop API.

---

## 6. Resource budgets (consumer SKUs)

### 6.1 fp32 footprint (approx)

\[
\text{vectors} \approx N \times d \times 4
\]

| N docs | d=384 fp32 vectors | + HNSW graph (rough) | Fits comfortably |
|---|---|---|---|
| 100k | ~150 MB | ~+100–200 MB | 8 GB laptop |
| 500k | ~750 MB | ~+0.5–1 GB | 16 GB |
| 1M | ~1.5 GB | ~+1.5–2 GB (our ~1.6–1.8 GB class) | 16–32 GB |
| 5M | ~7.5 GB | + multi-GB graph | needs quant/mmap |

### 6.2 Device profiles

| Profile | RAM | Mode | Target N @ 384-d |
|---|---|---|---|
| Light | 8 GB | mmap + int8 | ~200–500k |
| Standard | 16 GB | ram fp32 | ~0.5–1M |
| Heavy | 32 GB | ram fp32 | ~1–2M |
| Power-user | 32 GB+ | ram / quant hybrid | multi-M |

Always leave headroom for the **host app + OS + embedder model**.

### 6.3 Embedder is often the real laptop wall

ANN may do 1k QPS while **local BGE/ONNX** does tens of embeds/sec on CPU. Product UX must:

- Batch embed on ingest  
- Cache query embeddings  
- Offer “cloud embed / local search” as an option without forcing cloud search  

Our capacity tests used precomputed `qidx` — consumers will not, unless you design for it.

---

## 7. Quality gates (do not ship without)

Copied from what saved us in Stage 0/1:

| Gate | Threshold | Why |
|---|---|---|
| Flat recall@10 vs exact | 1.0 on harness | Trust measurement |
| HNSW recall@10 | **≥ 0.90** (configurable) | Same floor as this project |
| Quant + rescore recall@10 | ≥ floor | Prevent Stage-2 PQ failure mode |
| p95 search latency | e.g. &lt; 20–50 ms interactive | UX |
| Crash test | kill -9 mid-insert → recoverable | Consumer trust |
| Warm vs cold | Document both | Avoid our cold Go smoke trap |

---

## 8. Lessons from Python / Go experiments (apply directly)

| Lesson | Consumer-DB action |
|---|---|
| GIL + threads ≠ multi-core for Python servers | Core in **Rust/C++**; Python is binding only |
| MP HNSW = QPS win, **RAM × N** | Never fork N full graphs on a laptop |
| Go front door sped up Qdrant ~4× | If sidecar exists, use **native** HTTP/gRPC, not Python |
| Cache/partitions didn’t raise QPS on fitting data | v1 = plain HNSW; add features when RAM forces them |
| Autocannon fixed-body lied | Bench with diverse queries |
| No swap on server | Laptops have swap — still design so working set fits or mmap cleanly |

---

## 9. Differentiation vs existing options

You will be compared to: **USearch, FAISS, hnswlib, sqlite-vss, LanceDB, Chroma local, DuckDB VSS**.

Don’t pretend HNSW is new. Differentiate on:

| Angle | Idea |
|---|---|
| UX | One file, one `open()`, works offline |
| Safety | WAL + checksums + encryption |
| Footprint | Tiny static lib; runtime ISA detect |
| Opinionated profiles | `cdb.profile = "laptop-16g"` sets ef/mmap/quant |
| Privacy | No telemetry; local-only defaults |
| App kits | Electron/Tauri examples; “personal RAG” template |

---

## 10. MVP roadmap

### Phase 0 — Spike (1–2 weeks)

- f32 IP SIMD (AVX2 + NEON)  
- In-RAM HNSW insert/search  
- Save/load one file  
- Recall@10 harness on a 100k slice of our MS MARCO embeddings  

### Phase 1 — Consumer MVP

- WAL + crash recovery  
- mmap mode  
- Go + Python bindings  
- CLI: `cdb put / search / stats`  
- Bench pack: QPS + RSS + recall (reuse methodology from `ARTICLE.md`)  

### Phase 2 — Fit more data

- Scalar int8 + fp32 rescore  
- Background optimize/compact when on AC power  
- Optional SQLite FTS shim for hybrid (not pure-Python BM25)  

### Phase 3 — Product polish

- Encryption  
- Snapshot export  
- Mobile (iOS/Android) NEON builds  
- Embedder plugins (ONNX) **outside** the ANN hot path  

---

## 11. Suggested package layout (greenfield)

```
consumer-db/
  crates/ or src/     # engine
    simd/             # distance kernels + dispatch
    hnsw/             # graph
    store/            # file + WAL
    api/              # C ABI
  bindings/
    go/
    python/
  benches/            # recall + QPS
  examples/
    personal_search/
  docs/
    consumer-db.md    # this file (or symlink)
```

If built inside this monorepo later: `consumer-db/` at repo root, reuse `data/msmarco/.../embeddings` for gates only.

---

## 12. Success metrics (v1)

| Metric | Target |
|---|---|
| Lib size (stripped) | &lt; few MB native |
| Open 1M index on NVMe | &lt; few seconds warm path documented |
| Search p50 @ 1M ef=64 k=10 | mid–sub ms to low ms on modern laptop CPU |
| RSS @ 1M fp32 | ~2 GB class (honest docs) |
| Recall@10 | ≥ 0.90 vs flat GT |
| Crash recovery | 100% of fault-injection suite |

---

## 13. Bottom line

A **SIMD-first, single-node, embeddable HNSW engine** is the right consumer vector DB:

- Our measurements say **in-process native HNSW** wins on one machine for QPS/RAM.  
- **Go+Qdrant** wins as a slim *service*; laptops want the index **inside the app**.  
- Avoid Python multi-process copies, avoid partitions-until-needed, gate recall before quant.  
- Plan RAM for vectors+graph+embedder; plan SIMD for distance; plan mmap/quant for overflow.

**Next concrete step:** Phase 0 spike on AVX2/NEON + HNSW file + recall gate against existing `data/msmarco/subset_1m/embeddings`.

---

*Companion to `reports/ARTICLE.md` (server/Python/Go capacity log). This file is the consumer-product design track.*
