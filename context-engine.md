# Context Engine — Project Roadmap

A resource-constrained retrieval ("context") engine for RAG, built in stages. Each stage adds exactly one component, is benchmarked through the same harness, and is compared against a naive baseline and the previous stage. After the single-server design is exhausted, the same engine is run up a ladder of bigger servers to study how it scales.

This document is the source of truth for the agent building the project. Read it fully before starting. Follow the stage order. Do not skip gates.

---

## 1. Goals

1. Build a retrieval engine that does **hybrid search (dense + BM25) + reranking**.
2. Make it work well on a **24 GB RAM server** where the full-precision index does **not** fit in memory, using:
   - vector quantization,
   - a **multi-graph (partitioned) HNSW** index with a router and result merging,
   - a **graph manager** that keeps hot partitions in RAM and loads cold ones from NVMe on demand (disk as "virtual memory"),
   - prefetching, and full-precision rescoring from SSD.
3. Measure every stage for **accuracy** (recall, precision, nDCG) and **efficiency** (latency, QPS, RAM, disk reads).
4. Find the **max sustainable QPS** on the 24 GB server, identify the bottleneck, then repeat on bigger servers (vertical scaling study) and extrapolate what 1M QPS would require.

## 2. Non-goals

- Not building an LLM generation layer. Output of the engine is ranked context (passages + scores).
- Not writing HNSW graph traversal from scratch. Use `hnswlib` / FAISS per partition. Our work is the layer around it: routing, partition management, caching, merging, rescoring, fusion.
- Not optimizing prematurely. Python first; Rust only where profiling proves Python overhead dominates (Stage 12).

## 3. Operating rules for the agent

1. **Measurement is sacred.** After Stage 0 is frozen, do not modify `bench/ground_truth.py`, the saved ground-truth files, the query sets, or metric definitions in `bench/metrics.py`. If a change seems necessary, stop and report to the human with the reason.
2. **Every number comes from the harness.** No hand-run timings in reports. Every reported result must have a JSON file in `results/`.
3. **Never fabricate or estimate results.** If a run fails, record the failure and the error.
4. **One change per stage.** A stage adds one component (plus its config). Don't bundle unrelated changes.
5. **Every stage ends with a report** (`reports/stageNN.md`) containing: hypothesis, what changed, parameter sweep, results table vs baseline and previous stage, plots, conclusion, whether the gate passed.
6. **Stop and ask the human** before: renting or resizing servers, any action that spends money, deleting data or results, changing the memory budget, or changing the recall floor.
7. **Tear down** paid servers when a benchmarking session ends. Log start/stop times and cost in `infra/cost_log.md`.
8. **Reproducibility.** Every result JSON includes git commit, config hash, server spec, dataset version, and random seeds. Pin all dependency versions in a lock file.
9. **Anomalies.** If results look too good (e.g., recall jumps, latency drops 10×), assume a measurement bug first. Verify the page cache was dropped and the memory limit was active.
10. Commit code, configs, results, and reports to git after each stage.

## 4. Constraints and definitions

- **Memory budget:** the engine process runs in a container with a hard limit: `docker run --memory=24g --memory-swap=24g`. All "24 GB" results must be produced under this limit, even on bigger machines.
- **Cold run:** before the run, `sync; echo 3 > /proc/sys/vm/drop_caches`, restart the engine container.
- **Warm run:** run a warm-up of N queries (default 10,000, not measured), then measure.
- **Recall floor:** recall@10 ≥ **0.90** on the ANN track. A configuration below the floor is reported but does not count as a "win".
- **Latency SLO:** p99 ≤ **100 ms** for dense-only and hybrid; p99 ≤ **300 ms** for pipelines with reranking. (Human may adjust; record the values used in every result.)
- **Max sustainable QPS:** the highest offered load at which p99 stays within the SLO and error rate ≤ 0.1%, held for at least 3 minutes.
- **Repeats:** each measured configuration runs 3 times; report the median and min/max.

## 5. Tech stack

| Area | Choice |
|---|---|
| Language | Python 3.11+ (Rust via PyO3/maturin only in Stage 12) |
| HNSW | `hnswlib` (primary), FAISS for flat/IVF-PQ baselines and exact ground truth |
| Quantization | FAISS (PQ, SQ8), NumPy for binary codes |
| Disk storage | NumPy `memmap`, raw binary files, `os.pread` / optional io_uring later |
| BM25 | Tantivy (`tantivy` Python bindings) |
| Reranker | Small cross-encoder (e.g. `bge-reranker-base`) exported to ONNX, int8, ONNX Runtime on CPU; GPU variant in the server ladder |
| Embedding (offline) | Open embedding model (e.g. `bge-small-en-v1.5`, 384-d) via sentence-transformers or TEI on a rented GPU, run once |
| Serving | gRPC (`grpcio`) with multiple worker processes |
| Load generation | Custom async gRPC load generator in `bench/loadgen.py` (open-loop, fixed arrival rate) on a separate machine |
| Metrics / monitoring | Prometheus + Grafana, `psutil`, `/proc` disk stats, `iostat` |
| Plots | matplotlib |
| Infra | Docker, bash / Terraform scripts in `infra/` |
| External baselines | Qdrant (on-disk + quantization), FAISS IVF-PQ, DiskANN |

Verify current versions and availability of each package before pinning.

## 6. Datasets

Two tracks. Keep them separate in code and results.

### Track Q — Quality (retrieval relevance)
- **MS MARCO passage ranking** (~8.8M passages) with dev queries and qrels. Optionally BEIR datasets (e.g. SciFact, FiQA, NQ) for quick iteration.
- Embed passages and queries with the chosen embedding model, once, and store as `float32` `.npy` / `.fbin`.
- Metrics: Precision@k, Recall@100, MRR@10, nDCG@10.
- Used for: Stages 8–10 (hybrid, rerank, cache) and as a sanity check for all dense stages.

### Track S — Scale (ANN accuracy under memory pressure)
- A vector set big enough that fp32 does **not** fit in 24 GB: **30M–100M vectors**.
- Candidates: subsets of the big-ann-benchmarks datasets (e.g. Deep, MS SPACEV) or MS MARCO embeddings if large enough. Verify download sources and licenses before use.
- Ground truth must be **recomputed for the exact subset used** (subset ground truth ≠ full-dataset ground truth).
- Metrics: recall@1, recall@10, recall@100 vs exact neighbours.
- Used for: Stages 1–7, 11.

### Dev subset
- A 1M-vector subset of each track for fast local iteration. Every stage is first made correct on the dev subset, then run at full scale.

### Query workloads
- `uniform`: queries sampled uniformly from the test query set.
- `zipf`: queries sampled with a Zipf distribution (s = 1.0 default; also sweep 0.8 and 1.2) to simulate skewed real traffic where some topics are hot. Caching stages must be evaluated on both.
- Fixed seeds; workload files are generated once in Stage 0 and frozen.

## 7. Repository layout

```
context-engine/
  ROADMAP.md
  engine/
    index/
      flat.py            # exact search (FAISS IndexFlat)
      hnsw.py            # single HNSW in RAM (naive baseline)
      partitioned.py     # multi-graph: k-means partitions, one HNSW each
      router.py          # centroid routing, nprobe selection
      merge.py           # top-k heap merge across partitions
    storage/
      quantization.py    # SQ8, PQ, binary codes
      mmap_store.py      # naive memory-mapped storage (OS-managed)
      graph_manager.py   # partition cache: load/evict under memory budget
      policies.py        # LRU, LFU, cost-aware eviction
      prefetch.py        # prefetch strategies
      rescoring.py       # full-precision rescoring from SSD
    sparse/bm25.py
    fusion/rrf.py        # RRF + weighted score fusion
    rerank/cross_encoder.py
    cache/
      result_cache.py    # exact query cache
      semantic_cache.py  # embedding-similarity cache
    pipeline.py          # builds a pipeline from a config
    server.py            # gRPC service
  bench/
    ground_truth.py      # exact kNN for test queries (FROZEN after Stage 0)
    workloads.py         # uniform / zipf query streams (FROZEN)
    metrics.py           # recall, precision, MRR, nDCG, latency stats (FROZEN)
    harness.py           # offline accuracy + latency runs
    loadgen.py           # load ramp, max sustainable QPS
    sysmon.py            # CPU, RAM, disk IOPS, cache stats sampling
    plots.py
  configs/
    stage01_flat.yaml
    stage01_hnsw.yaml
    ...
  data/                  # gitignored; datasets, embeddings, indexes
  results/               # one JSON per run (committed)
  reports/               # stageNN.md reports (committed)
  infra/
    setup_server.sh
    run_engine.sh        # docker run with --memory limit
    drop_caches.sh
    teardown.sh
    cost_log.md
  rust/                  # Stage 12 only
  tests/
```

## 8. Pipeline interface

Every component plugs into the same interface so each stage is only a new config.

```python
class Retriever(Protocol):
    def search(self, query: Query, k: int) -> list[Hit]: ...
    def stats(self) -> dict: ...   # cache hits, disk reads, partitions probed, etc.

@dataclass
class Query:
    id: str
    text: str | None
    vector: np.ndarray | None

@dataclass
class Hit:
    doc_id: int
    score: float
    source: str   # "dense", "bm25", "fused", "reranked"
```

`pipeline.py` composes components from YAML, e.g.:

```yaml
name: stage05_graph_manager_lru
track: S
dense:
  index: partitioned
  partitions: 2048
  overlap: 1
  nprobe: 16
  hnsw: {M: 16, ef_construction: 200, ef_search: 64}
  vectors: {codes: pq64, rescoring: none}
  storage: {mode: managed, budget_gb: 12, policy: lru, prefetch: none}
sparse: null
fusion: null
rerank: null
cache: null
memory_limit_gb: 24
```

## 9. Result schema

Every run writes `results/{stage}/{config_name}/{timestamp}.json`:

```json
{
  "stage": "05",
  "config_name": "stage05_graph_manager_lru",
  "config_hash": "...",
  "git_commit": "...",
  "server": {"name": "...", "vcpu": 8, "ram_gb": 24, "disk": "NVMe", "gpu": null},
  "memory_limit_gb": 24,
  "dataset": {"track": "S", "name": "...", "n_vectors": 50000000, "dim": 96},
  "workload": {"type": "zipf", "s": 1.0, "n_queries": 10000, "seed": 42},
  "run_mode": "cold|warm",
  "repeat": 1,
  "accuracy": {"recall@1": 0.0, "recall@10": 0.0, "recall@100": 0.0,
               "precision@10": null, "mrr@10": null, "ndcg@10": null},
  "latency_ms": {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0},
  "throughput": {"max_sustainable_qps": null, "slo_p99_ms": 100},
  "resources": {"peak_rss_gb": 0.0, "avg_cpu_pct": 0.0,
                "disk_reads_per_query": 0.0, "disk_read_mb_per_query": 0.0,
                "disk_iops_peak": 0.0},
  "engine_stats": {"cache_hit_rate": null, "partitions_probed_avg": null,
                   "partition_loads_per_query": null},
  "index": {"build_time_s": 0.0, "ram_size_gb": 0.0, "disk_size_gb": 0.0},
  "status": "ok|failed",
  "error": null
}
```

---

## 10. Stages

Each stage lists: goal, tasks, experiments, deliverables, and a **gate** that must pass before moving on. Stages 1–7 run on Track S (plus a Track Q sanity check); 8–10 on Track Q; 11 on both.

### Stage 0 — Foundations and measurement harness

**Goal:** be able to measure anything correctly before building anything.

Tasks:
1. Repo skeleton, lock file, Docker image, `infra/run_engine.sh` enforcing the memory limit.
2. Download datasets; build dev subsets (1M) and full Track S subset; embed Track Q (on a rented GPU if needed; log cost).
3. `ground_truth.py`: exact top-100 neighbours via FAISS `IndexFlat` for 10,000 test queries on each dataset size used. Save as files with checksums.
4. `workloads.py`: generate and freeze uniform and zipf query streams.
5. `metrics.py`: recall@k (ANN), precision@k, recall@k, MRR@10, nDCG@10 (qrels), latency percentiles. Unit-test each against hand-computed examples.
6. `harness.py`: runs a config over a workload, cold or warm, N repeats, writes result JSON.
7. `sysmon.py`: samples RSS, CPU, disk IOPS/bytes (from `/proc/diskstats`) during runs.
8. `loadgen.py`: open-loop generator at fixed arrival rate; ramp mode (steps, hold duration) that finds max sustainable QPS per the definition in §4. Runs on a separate machine.
9. `plots.py`: recall vs latency, recall vs RAM, QPS vs parameter, cache hit rate vs budget.

Deliverables: working harness, frozen ground truth + workloads, tests passing, `reports/stage00.md`.

**Gate:** a trivial flat-search config on the dev subset gives recall@10 = 1.0 and the harness produces valid JSON and plots. Verify the memory limit works by deliberately exceeding it (the container must be OOM-killed). Then freeze the `bench/` measurement files.

### Stage 1 — Naive baselines

**Goal:** establish the reference points everything is compared against.

Tasks:
1. `flat.py`: exact search (only on subsets that fit in RAM).
2. `hnsw.py`: single HNSW graph, fp32, fully in RAM.
3. Run the single HNSW on the dev subset under 24 GB, and on the full Track S set **without** the memory limit on a large machine (only if the human approves renting it) to get the unconstrained upper bound. If not approved, use the largest subset that fits in 24 GB and document it.

Experiments: sweep `ef_search` ∈ {16, 32, 64, 128, 256}, `M` ∈ {16, 32}. Record recall, latency, RAM, build time.

Deliverables: `reports/stage01.md` with recall-vs-latency curve for the naive HNSW.

**Gate:** naive baseline numbers recorded; document at what dataset size naive HNSW stops fitting in 24 GB.

### Stage 2 — Quantization

**Goal:** fit more vectors in RAM; quantify the accuracy cost.

Tasks: implement SQ8 (int8), PQ (sweep code sizes), binary codes (Hamming search) as vector representations for search inside HNSW / flat.

Experiments: each quantization × `ef_search` sweep. Track RAM saved vs recall lost.

**Gate:** recall-vs-RAM table for all quantizations. Identify which quantizations let the full Track S set fit in 24 GB and at what recall.

### Stage 3 — Naive disk (OS-managed virtual memory)

**Goal:** baseline for disk-backed search where the OS decides what stays in RAM.

Tasks: `mmap_store.py` — store the single HNSW index / vectors in memory-mapped files on NVMe; run under the 24 GB limit with dataset larger than RAM.

Experiments: cold vs warm; uniform vs zipf; record page faults, disk reads/query, latency distribution.

**Gate:** documented behaviour of the OS-managed approach. This is the baseline the graph manager (Stage 5) must beat.

### Stage 4 — Multi-graph partitioned index

**Goal:** replace the single graph with many small HNSW graphs plus a router.

Tasks:
1. Train k-means on a sample; assign vectors to partitions.
2. Optional overlap: assign boundary vectors to their top-2/top-3 nearest centroids (duplicate IDs deduplicated at merge).
3. Build one HNSW per partition; save each as a separate file with metadata (size, centroid, vector count).
4. `router.py`: rank centroids for a query, choose `nprobe` partitions (fixed nprobe; also try adaptive: probe until centroid distance gap exceeds a threshold).
5. `merge.py`: search probed partitions, merge with a top-k heap, deduplicate.
6. In this stage all partitions are loaded in RAM (with quantization if needed) — storage management comes next.

Experiments: partitions K ∈ {256, 1024, 4096}; nprobe ∈ {1, 2, 4, 8, 16, 32, 64}; overlap ∈ {1, 2}; fixed vs adaptive nprobe. Plot recall vs nprobe and recall vs latency. Measure partition size distribution (imbalance).

**Gate:** a partitioned config reaches recall@10 ≥ 0.90; report the boundary-loss effect and the overlap tradeoff (recall vs extra space).

### Stage 5 — Graph manager (managed virtual memory)

**Goal:** keep only hot partitions in RAM under a fixed budget; load cold partitions from NVMe on demand.

Tasks:
1. `graph_manager.py`: partition registry, memory accounting, load on miss, evict when over budget, pin router data and centroids permanently.
2. `policies.py`: LRU, LFU, and a cost-aware policy (evict by frequency ÷ size or frequency × load cost).
3. Thread/process safety for concurrent queries; no duplicate loads of the same partition.
4. Instrument: hit rate, loads/query, bytes read/query, eviction count.

Experiments: budget_gb ∈ {4, 8, 12, 16}; each policy; uniform and zipf (s ∈ {0.8, 1.0, 1.2}); cold and warm. Compare directly against Stage 3 (OS mmap).

**Gate:** clear answer to "does the managed cache beat OS-managed mmap, under which workloads, and by how much?" If it does not, analyse why before continuing.

### Stage 6 — Prefetching

**Goal:** raise hit rate by loading partitions before they're needed.

Tasks: `prefetch.py` strategies:
1. Neighbour prefetch: when partition P is loaded, also load the partitions whose centroids are nearest to P's.
2. History prefetch: track partition co-access; prefetch frequent co-occurring partitions.
3. Background loading thread so prefetch doesn't block queries.

Experiments: each strategy vs none, on zipf workloads, at 2 budgets. Measure hit rate, p99, wasted loads (prefetched but not used).

**Gate:** documented whether prefetch helps and its cost in wasted I/O. Keep it off by default if it doesn't help.

### Stage 7 — Full-precision rescoring from SSD

**Goal:** recover recall lost to quantization.

Tasks: search on compressed codes to get top-R candidates; read their full-precision vectors from an on-disk file (grouped reads, sorted by offset); recompute exact distances; return top-k.

Experiments: R ∈ {20, 50, 100, 200, 500}; per quantization type. Measure recall gained vs disk reads and latency added.

**Gate:** best dense configuration on 24 GB selected (quantization + partitions + manager + rescoring). Compare it against external baselines: Qdrant (on-disk + quantization), FAISS IVF-PQ, DiskANN, all under the same 24 GB limit. Produce the headline plot: **recall@10 vs p99 latency at 24 GB**, one line per system.

### Stage 8 — Hybrid search (BM25 + dense)

**Goal:** improve relevance with lexical search. Track Q.

Tasks: Tantivy BM25 index over passages (postings on disk); `rrf.py` Reciprocal Rank Fusion (k = 60 default); weighted score fusion with normalized scores.

Experiments: dense-only vs BM25-only vs RRF vs weighted (sweep weights); candidate depth from each side ∈ {50, 100, 200}. Metrics: nDCG@10, MRR@10, Recall@100, latency.

**Gate:** hybrid nDCG@10 vs dense-only reported; pick the default fusion.

### Stage 9 — Reranking

**Goal:** improve precision at the top with a cross-encoder.

Tasks: ONNX int8 cross-encoder on CPU; batch query–passage pairs; truncate passages to a fixed max length.

Experiments: rerank depth ∈ {0, 20, 50, 100}; fp32 vs int8 ONNX. Metrics: nDCG@10, MRR@10, p50/p99 latency, CPU cost per query.

**Gate:** quality-vs-latency curve for rerank depth. Expect reranking to become the throughput bottleneck; document it.

### Stage 10 — Result caching

**Goal:** serve repeated queries cheaply.

Tasks: exact result cache (normalized query text → results, LRU with size limit); semantic cache (query embedding similarity ≥ threshold → cached results).

Experiments: uniform vs zipf; semantic threshold ∈ {0.90, 0.95, 0.98}. Measure hit rate, latency, and **quality impact** of semantic hits (nDCG of cached vs fresh results — wrong cache hits must be counted).

**Gate:** documented throughput gain and any quality loss from semantic caching.

### Stage 11 — Serving, saturation, and vertical scaling

**Goal:** find max sustainable QPS and bottlenecks, then climb the server ladder.

Tasks:
1. `server.py`: gRPC service, multiple worker processes, batch-search endpoint, health check, Prometheus metrics.
2. Grafana dashboard: QPS, latency percentiles, CPU, RSS, disk IOPS, cache hit rate, errors.
3. Load-ramp runs from a separate load-generator machine for three pipelines: **dense-only**, **hybrid**, **hybrid + rerank**; each with and without result cache; uniform and zipf.
4. At saturation, record which resource hit its limit first (CPU, RAM, disk IOPS, network).

Server ladder (change one resource at a time; human approves each rental):

| Step | Server | Question |
|---|---|---|
| L1 | 24 GB RAM, 8 vCPU, NVMe | Baseline: limit of the constrained design |
| L2 | 24 GB RAM, 16–32 vCPU | Does it scale with cores, or is disk the limit? |
| L3 | 64 GB RAM, 8 vCPU | Effect of a larger cache budget |
| L4 | 128+ GB RAM | Whole index in RAM: cost of disk |
| L5 | + GPU (L4/A10 class) | Reranker on GPU: does the main bottleneck move? |
| H  | 2–4 × best-value node | Horizontal scaling is linear? |

Per ladder step: same dataset, workloads, and configs; re-tune only documented knobs; 3 repeats.

**Gate / final deliverables:**
- Max sustainable QPS per pipeline per server, with bottleneck identified.
- Plots: QPS vs cores, QPS vs RAM, QPS per dollar per server, horizontal scaling curve.
- **1M QPS extrapolation:** nodes and cost required for dense-only, hybrid, and hybrid + rerank, based on measured per-node throughput and the horizontal scaling result.

### Stage 12 — Optimization (optional, profiling-driven)

**Goal:** remove Python overhead only where it measurably dominates.

Tasks: profile with `py-spy` under load; if the router → partition search → merge loop or graph-manager bookkeeping dominates, move it to Rust (PyO3 + maturin) or Numba. Consider batching queries per partition, and io_uring for disk reads.

**Gate:** before/after comparison with identical configs; accuracy unchanged, throughput/latency change reported.

### Stage 13 — Final write-up

Deliverables:
- `reports/final.md`: problem, design, stage-by-stage results, headline plots, bottleneck analysis, what worked and what didn't, 1M QPS extrapolation, cost summary.
- Architecture diagram.
- Instructions to reproduce any result from its JSON (config + commit + dataset).

---

## 11. Metrics reference

**ANN accuracy (Track S)**
- recall@k = |returned top-k ∩ true top-k| / k, averaged over queries.

**Retrieval quality (Track Q)**
- Precision@k = relevant docs in top-k / k.
- Recall@k = relevant docs in top-k / total relevant docs for the query.
- MRR@10 = mean of 1 / rank of first relevant doc (0 if none in top 10).
- nDCG@10 = DCG@10 / ideal DCG@10 using qrels grades.

**Efficiency**
- Latency p50 / p95 / p99 (ms), measured end to end at the client for load tests and inside the harness for offline runs (label which).
- Max sustainable QPS (definition in §4).
- Peak RSS (GB), disk reads per query, MB read per query, cache hit rate, partition loads per query.
- QPS per dollar = max sustainable QPS / server hourly price.

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| OS page cache inflates disk results | Drop caches before cold runs; enforce container memory limit; check disk read counters are non-zero |
| Ground truth mismatch after subsetting | Recompute exact kNN for every subset used; checksum files |
| Partition imbalance (some huge partitions) | Balanced k-means or split oversized partitions; report size distribution |
| Load generator becomes the bottleneck | Run on a separate machine; monitor its CPU; add generator machines |
| Python/GIL overhead masks design effects | Multi-process workers; compare designs within the same language layer; Stage 12 only if needed |
| Semantic cache returns wrong results | Measure quality of cache hits, not just hit rate |
| Cost overruns on rented servers | Human approval per rental; teardown script; cost log |
| Agent "improves" metrics by changing measurement | `bench/` frozen after Stage 0; rule 1 in §3 |

## 13. Milestone checklist

- [ ] Stage 0 — harness, ground truth, workloads frozen
- [ ] Stage 1 — naive baselines
- [ ] Stage 2 — quantization
- [ ] Stage 3 — naive disk (mmap)
- [ ] Stage 4 — multi-graph partitioned index
- [ ] Stage 5 — graph manager
- [ ] Stage 6 — prefetching
- [ ] Stage 7 — rescoring + external baselines comparison
- [ ] Stage 8 — hybrid search
- [ ] Stage 9 — reranking
- [ ] Stage 10 — result caching
- [ ] Stage 11 — serving, saturation, server ladder, 1M QPS extrapolation
- [ ] Stage 12 — profiling-driven optimization (optional)
- [ ] Stage 13 — final write-up