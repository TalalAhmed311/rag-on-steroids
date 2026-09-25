#!/usr/bin/env bash
# Experiment 2: multi-process QPS scaling across our stages + Qdrant.
set -euo pipefail
cd /home/ec2-user/rag-on-steroids
mkdir -p results/capacity_mp logs

fuser -k 8080/tcp 8081/tcp 2>/dev/null || true
sleep 2

QV=data/msmarco/subset_1m/embeddings/queries.f32.npy
WL=data/workloads/msmarco_dev/zipf_s1p0.npy
HOLD=12

run_mp() {
  local name="$1" cfg="$2" workers="$3" slo="$4" concs="$5"
  local tag="${name}_w${workers}"
  echo "===== $tag ====="
  fuser -k 8080/tcp 2>/dev/null || true
  sleep 1
  python3 -m engine.serve_mp --config "$cfg" --workers "$workers" --port 8080 \
    --query-vectors "$QV" > "logs/serve_mp_${tag}.log" 2>&1 &
  local master=$!
  for i in $(seq 1 180); do
    curl -sf http://127.0.0.1:8080/health >/dev/null && break
    sleep 1
  done
  # warm
  python3 -m bench.capacity_test --url http://127.0.0.1:8080/search --workload "$WL" \
    --hold-s 6 --slo-p99-ms 9999 --concurrencies 16 --out /tmp/warm_mp.json >/dev/null || true
  # measure
  python3 -m bench.capacity_test --url http://127.0.0.1:8080/search --workload "$WL" \
    --hold-s "$HOLD" --slo-p99-ms "$slo" --max-error-rate 0.05 \
    --concurrencies "$concs" --out "results/capacity_mp/${tag}.json"
  # process RSS sum
  ps -o rss= -C python3 2>/dev/null | awk '{s+=$1} END {printf "approx_python_rss_gb=%.2f\n", s/1024/1024}' || true
  free -h | head -2
  # kill master + children
  kill "$master" 2>/dev/null || true
  pkill -P "$master" 2>/dev/null || true
  # also kill any leftover serve_mp workers
  pgrep -af 'engine.serve_mp|serve_mp' | grep -v run_capacity_mp | awk '{print $1}' | xargs -r kill 2>/dev/null || true
  fuser -k 8080/tcp 2>/dev/null || true
  sleep 3
}

# Our HNSW — primary scaling study
for w in 1 2 4 8; do
  run_mp hnsw configs/stage01_hnsw.yaml "$w" 100 "8,16,32,64,96,128"
done

# Partitioned + graph manager at 2 and 4 workers (heavier RAM)
run_mp partitioned configs/stage04_partitioned.yaml 2 200 "8,16,32,64"
run_mp partitioned configs/stage04_partitioned.yaml 4 200 "8,16,32,64"
run_mp graph_manager configs/stage05_graph_manager_lru.yaml 2 300 "8,16,32,64"
run_mp graph_manager configs/stage05_graph_manager_lru.yaml 4 300 "8,16,32,64"

# HNSW + cache at 4 workers
run_mp hnsw_cache configs/stage10_hnsw_cache.yaml 4 100 "8,16,32,64,96"

# Qdrant — already multi-threaded Rust; push higher concurrency + autocannon
echo "===== qdrant high-concurrency ====="
python3 scripts/serve_qdrant_proxy.py --port 8081 > logs/qdrant_proxy_mp.log 2>&1 &
qpid=$!
sleep 2
python3 -m bench.capacity_test --url http://127.0.0.1:8081/search --workload "$WL" \
  --hold-s 15 --slo-p99-ms 100 --max-error-rate 0.05 \
  --concurrencies 8,16,32,64,96,128 --out results/capacity_mp/qdrant_high.json || true
autocannon -c 64 -d 15 -m POST -H 'Content-Type=application/json' -b '{"qidx":42,"k":10}' \
  http://127.0.0.1:8081/search 2>&1 | tee results/capacity_mp/qdrant_autocannon.txt || true
sudo docker stats qdrant --no-stream | tee results/capacity_mp/qdrant_docker_stats.txt || true
kill "$qpid" 2>/dev/null || true
fuser -k 8081/tcp 2>/dev/null || true

# Autocannon our best MP HNSW (4 workers)
echo "===== autocannon hnsw w4 ====="
python3 -m engine.serve_mp --config configs/stage01_hnsw.yaml --workers 4 --port 8080 \
  --query-vectors "$QV" > logs/serve_mp_hnsw_ac.log 2>&1 &
master=$!
for i in $(seq 1 120); do curl -sf http://127.0.0.1:8080/health >/dev/null && break; sleep 1; done
autocannon -c 64 -d 20 -m POST -H 'Content-Type=application/json' -b '{"qidx":123,"k":10}' \
  http://127.0.0.1:8080/search 2>&1 | tee results/capacity_mp/hnsw_w4_autocannon.txt || true
kill "$master" 2>/dev/null || true
pkill -P "$master" 2>/dev/null || true
fuser -k 8080/tcp 2>/dev/null || true

echo EXP2_DONE
