#!/usr/bin/env bash
# Full Go+Qdrant capacity suite (no Python).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH=/usr/local/go/bin:$PATH
cd "$ROOT"
mkdir -p results/go_qdrant logs

fuser -k 8080/tcp 2>/dev/null || true
sleep 1

cd "$ROOT/go"
go build -o bin/searchd ./cmd/searchd
go build -o bin/loadtest ./cmd/loadtest

"$ROOT/go/bin/searchd" \
  --addr :8080 \
  --qdrant-host 127.0.0.1 \
  --qdrant-port 6334 \
  --collection msmarco_1m \
  --query-vectors "$ROOT/data/msmarco/subset_1m/embeddings/queries.f32.npy" \
  --ef 64 \
  > "$ROOT/logs/go_searchd_bench.log" 2>&1 &
spid=$!

for i in $(seq 1 60); do
  curl -sf http://127.0.0.1:8080/health >/dev/null && break
  sleep 0.5
done

echo "=== Go loadtest SLO p99<=100ms ==="
"$ROOT/go/bin/loadtest" \
  --url http://127.0.0.1:8080/search \
  --hold 15s --warmup 8s \
  --slo-p99-ms 100 --max-error-rate 0.05 \
  --concurrencies 8,16,32,64,96,128,192,256 \
  --out "$ROOT/results/go_qdrant/loadtest_slo100.json"

echo "=== Go loadtest exhaust (no SLO stop) ==="
"$ROOT/go/bin/loadtest" \
  --url http://127.0.0.1:8080/search \
  --hold 12s --warmup 5s \
  --slo-p99-ms 99999 --max-error-rate 1.0 \
  --concurrencies 8,16,32,64,96,128,192,256 \
  --out "$ROOT/results/go_qdrant/loadtest_exhaust.json"

echo "=== autocannon c=64 ==="
if command -v autocannon >/dev/null; then
  autocannon -c 64 -d 20 -m POST \
    -H 'Content-Type=application/json' \
    -b '{"qidx":123,"k":10}' \
    http://127.0.0.1:8080/search \
    | tee "$ROOT/results/go_qdrant/autocannon_c64.txt"
fi

echo "=== autocannon c=128 ==="
if command -v autocannon >/dev/null; then
  autocannon -c 128 -d 20 -m POST \
    -H 'Content-Type=application/json' \
    -b '{"qidx":456,"k":10}' \
    http://127.0.0.1:8080/search \
    | tee "$ROOT/results/go_qdrant/autocannon_c128.txt"
fi

curl -sf http://127.0.0.1:8080/stats | tee "$ROOT/results/go_qdrant/server_stats.json"
echo
sudo docker stats qdrant --no-stream | tee "$ROOT/results/go_qdrant/qdrant_docker_stats.txt" || true
ps -o pid,rss,cmd -p "$spid" | tee "$ROOT/results/go_qdrant/searchd_ps.txt"

kill "$spid" 2>/dev/null || true
fuser -k 8080/tcp 2>/dev/null || true
echo GO_BENCH_DONE
