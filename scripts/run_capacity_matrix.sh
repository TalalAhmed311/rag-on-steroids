#!/usr/bin/env bash
# Run capacity matrix across stage configs (+ optional Qdrant).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p results/capacity logs

QV=data/msmarco/subset_1m/embeddings/queries.f32.npy
WL=data/workloads/msmarco_dev/zipf_s1p0.npy
HOLD="${HOLD_S:-15}"
CONC="${CONCURRENCIES:-4,8,16,32,64,96}"

run_stage () {
  local name="$1"
  local cfg="$2"
  local port="$3"
  local slo="${4:-100}"
  echo "===== $name on :$port ====="
  pkill -f "serve_http.py --port $port" 2>/dev/null || true
  sleep 1
  python3 -m engine.serve_http --config "$cfg" --port "$port" --query-vectors "$QV" \
    > "logs/serve_${name}.log" 2>&1 &
  local spid=$!
  for i in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$port/health" >/dev/null && break
    sleep 1
  done
  # RSS after load
  sleep 2
  local rss
  rss=$(ps -o rss= -p $spid | awk '{printf "%.2f", $1/1024/1024}')
  echo "server_rss_gb=$rss"
  python3 -m bench.capacity_test \
    --url "http://127.0.0.1:$port/search" \
    --workload "$WL" \
    --hold-s "$HOLD" \
    --slo-p99-ms "$slo" \
    --concurrencies "$CONC" \
    --out "results/capacity/${name}.json"
  # final stats
  curl -sf "http://127.0.0.1:$port/stats" > "results/capacity/${name}_server_stats.json" || true
  kill $spid 2>/dev/null || true
  wait $spid 2>/dev/null || true
  sleep 2
}

# Dense stages that fit this box
run_stage hnsw configs/stage01_hnsw.yaml 8080 100
run_stage partitioned configs/stage04_partitioned.yaml 8080 100
run_stage graph_manager configs/stage05_graph_manager_lru.yaml 8080 100
run_stage hnsw_cache configs/stage10_hnsw_cache.yaml 8080 100
run_stage hnsw_rescored configs/stage07_hnsw_rescored.yaml 8080 100

# Flat is heavy — shorter ramp / looser SLO
HOLD_S=10 CONCURRENCIES=2,4,8,16 \
  run_stage flat configs/stage01_flat.yaml 8080 500 || true

echo "Capacity matrix done → results/capacity/"
