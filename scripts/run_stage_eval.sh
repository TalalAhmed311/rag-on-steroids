#!/usr/bin/env bash
# Stage 0 / 1 evaluation helpers for the 1M MS MARCO subset.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

EMB=data/msmarco/subset_1m/embeddings
GT=data/msmarco/subset_1m/ground_truth/neighbors.npy
QRELS=data/msmarco/subset_1m/qrels_dev.parquet
WL=data/workloads/msmarco_dev/uniform.npy

run_one () {
  local cfg="$1"
  local n="${2:-1000}"
  python3 -m bench.harness \
    --config "$cfg" \
    --query-vectors "$EMB/queries.f32.npy" \
    --query-texts data/msmarco/queries_dev.parquet \
    --ground-truth "$GT" \
    --qrels "$QRELS" \
    --workload "$WL" \
    --n-queries "$n" \
    --k 10 \
    --warmup 50 \
    --run-mode warm \
    --disk-device xvdf
}

case "${1:-help}" in
  flat) run_one configs/stage01_flat.yaml "${2:-2000}" ;;
  hnsw) run_one configs/stage01_hnsw.yaml "${2:-2000}" ;;
  hybrid) run_one configs/stage08_hybrid_rrf.yaml "${2:-500}" ;;
  *) echo "Usage: $0 {flat|hnsw|hybrid} [n_queries]"; exit 1 ;;
esac
