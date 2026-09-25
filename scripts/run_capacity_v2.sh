#!/usr/bin/env bash
set -euo pipefail
cd /home/ec2-user/rag-on-steroids
mkdir -p results/capacity logs

# Kill only live server PIDs listening on 8080/8081 (not this script)
for port in 8080 8081; do
  fuser -k ${port}/tcp 2>/dev/null || true
done
sleep 2

run_one() {
  local name="$1" cfg="$2" slo="$3" concs="$4"
  echo "===== $name ====="
  python3 -m engine.serve_http --config "$cfg" --port 8080 \
    --query-vectors data/msmarco/subset_1m/embeddings/queries.f32.npy \
    > "logs/serve_${name}_v2.log" 2>&1 &
  local spid=$!
  for i in $(seq 1 120); do
    if curl -sf http://127.0.0.1:8080/health >/dev/null; then break; fi
    sleep 1
  done
  echo "up pid=$spid"
  python3 -m bench.capacity_test --url http://127.0.0.1:8080/search \
    --workload data/workloads/msmarco_dev/zipf_s1p0.npy --hold-s 8 --slo-p99-ms 9999 \
    --concurrencies 8 --out "/tmp/warm_${name}.json" >/dev/null || true
  python3 -m bench.capacity_test --url http://127.0.0.1:8080/search \
    --workload data/workloads/msmarco_dev/zipf_s1p0.npy --hold-s 15 --slo-p99-ms "$slo" \
    --concurrencies "$concs" --out "results/capacity/${name}_v2.json"
  curl -sf http://127.0.0.1:8080/stats > "results/capacity/${name}_v2_stats.json" || true
  ps -o rss= -p "$spid" | awk -v n="$name" '{printf "%s_rss_gb=%.2f\n", n, $1/1024/1024}'
  kill "$spid" 2>/dev/null || true
  wait "$spid" 2>/dev/null || true
  fuser -k 8080/tcp 2>/dev/null || true
  sleep 2
}

run_one partitioned configs/stage04_partitioned.yaml 200 4,8,16,32
run_one graph_manager configs/stage05_graph_manager_lru.yaml 300 4,8,16,32
run_one hnsw_exhaust configs/stage01_hnsw.yaml 99999 8,16,32,64,96,128

echo "===== autocannon hnsw ====="
python3 -m engine.serve_http --config configs/stage01_hnsw.yaml --port 8080 \
  --query-vectors data/msmarco/subset_1m/embeddings/queries.f32.npy > logs/serve_hnsw_ac.log 2>&1 &
spid=$!
for i in $(seq 1 60); do curl -sf http://127.0.0.1:8080/health >/dev/null && break; sleep 1; done
autocannon -c 64 -d 20 -m POST -H 'Content-Type=application/json' -b '{"qidx":123,"k":10}' \
  http://127.0.0.1:8080/search | tee results/capacity/hnsw_autocannon.txt
kill "$spid" 2>/dev/null || true
wait "$spid" 2>/dev/null || true
fuser -k 8080/tcp 2>/dev/null || true

echo "===== qdrant ====="
python3 scripts/serve_qdrant_proxy.py --port 8081 > logs/qdrant_proxy.log 2>&1 &
qpid=$!
sleep 3
python3 -m bench.capacity_test --url http://127.0.0.1:8081/search \
  --workload data/workloads/msmarco_dev/zipf_s1p0.npy --hold-s 15 --slo-p99-ms 100 \
  --concurrencies 4,8,16,32 --out results/capacity/qdrant.json || true
kill "$qpid" 2>/dev/null || true
fuser -k 8081/tcp 2>/dev/null || true
echo ALL_DONE
