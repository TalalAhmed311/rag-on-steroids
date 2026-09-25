# Experiment 2 — Multi-process scaling

**Goal:** Raise QPS by running multiple OS processes (each with its own index copy) via `SO_REUSEPORT`, and record what breaks for each system.

**Server:** `python3 -m engine.serve_mp --workers N`  
**Artifacts:** `results/capacity_mp/`

---

## Our HNSW (primary scaling curve)

| Workers | Best under p99≤100ms | Peak QPS seen | Approx RAM cost |
|---|---|---|---|
| 1 | **1507 QPS** (c=16, p99≈7ms) | 1622 | ~1.8 GB |
| 2 | **2013 QPS** (c=32, p99≈22ms) | 2158 | ~3.6 GB |
| 4 | **2139 QPS** (c=64, p99≈67ms) | 2275 | ~7 GB |
| 8 | **2138 QPS** (c=64, p99≈66ms) | 2299 | **~14 GB** (near pressure) |

**Autocannon (HNSW w=4, c=64, 20s):** ~**4211 req/s** avg (same-query body; optimistic vs zipf mix).

### Problems — HNSW MP
- **Diminishing returns after 4 workers** on 8 cores: w8 ≈ w4 QPS but **2× RAM**.
- **Latency cliff** when client concurrency ≫ workers×useful parallelism (p99 jumps to ~1s).
- **RAM multiplies** with workers (full HNSW copy per process). w8 left the box with only hundreds of MB free at peak load.
- Still **single-machine**; no horizontal story yet.

---

## Partitioned (nprobe=16)

| Workers | Best (p99≤200ms) | Peak |
|---|---|---|
| 2 | **305 QPS** | 305 |
| 4 | **477 QPS** | 477 |

### Problems — Partitioned
- Scales with workers but **far below HNSW** (routing + 16 graph probes).
- **RAM × workers** for full in-RAM partition set (~2.4 GB × N).
- p99 sensitive once concurrency rises past ~16–32.

---

## Graph manager (lazy LRU, budget 8 GB)

| Workers | Best (p99≤300ms) | Peak |
|---|---|---|
| 2 | **290 QPS** | 290 |
| 4 | **448 QPS** | 448 |

### Problems — Graph manager
- **Cold / warm asymmetry:** first queries load partitions from disk → p99 spikes (seen in Exp 1).
- Under zipf + warm cache, behaves like partitioned throughput.
- Multi-process **duplicates caches** (no shared partition cache across processes) → more RAM, less cache efficiency than one fat process.
- Locking / load-on-miss can serialize under bursty unique partition access.

---

## HNSW + result/semantic cache (w=4)

- Peak ~**386 QPS** but **failed p99≤100ms** on first step (p99≈116ms).
- **Problems:** cache helps zipf repeats, but semantic cache + locks add overhead; under MP, each worker has a **private cache** (hit rate diluted).

---

## Qdrant (already multi-threaded Rust)

| Probe | Result |
|---|---|
| Exp 1 (proxy, c=8, p99≤100) | ~**403 QPS**, container RAM ~**175 MB** |
| Autocannon c=64 | ~**398 req/s**, some timeouts |
| Docker stats under test | ~175 MB / 31 GB |

### Problems — Qdrant
- **On-disk HNSW** → lower QPS than our in-RAM HNSW, much **lower RAM**.
- Python **proxy** (`serve_qdrant_proxy.py`) can become the bottleneck / error source (API quirks, GIL on JSON).
- High concurrency shows **timeouts** before CPU is saturated (I/O + search latency).
- Collection must be fully indexed (green) or latency is unstable.

---

## Cross-system diagnosis

| Bottleneck | Who hits it first |
|---|---|
| **Python GIL / single process** | Exp 1 HNSW (~1k QPS) |
| **Cores saturated + index copies** | Exp 2 HNSW w≥4 (~2.1k QPS plateau) |
| **RAM × workers** | HNSW w=8, partitioned/GM w=4 |
| **Probe fan-out (nprobe)** | Partitioned / GM |
| **Disk / cold cache** | Graph manager cold start |
| **On-disk ANN + proxy** | Qdrant |
| **Client overload (timeouts)** | Autocannon c=64 on slower backends |

**Verdict:** Multi-processing **~2×** HNSW QPS (1.5k → 2.1k sustainable) with **sweet spot w=4** on this 8-vCPU / 31 GB box. Beyond that you buy little QPS and burn RAM. Qdrant stays the **RAM-efficient** baseline (~400 QPS @ ~175 MB), not the throughput leader here.

---

## How to reproduce

```bash
# one stage
python3 -m engine.serve_mp --config configs/stage01_hnsw.yaml --workers 4 --port 8080
python3 -m bench.capacity_test --url http://127.0.0.1:8080/search \
  --workload data/workloads/msmarco_dev/zipf_s1p0.npy --hold-s 12 --slo-p99-ms 100 \
  --concurrencies 8,16,32,64,96

# full matrix
bash scripts/run_capacity_mp.sh
```
