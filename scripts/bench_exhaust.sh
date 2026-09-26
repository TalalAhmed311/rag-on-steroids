#!/usr/bin/env bash
# Exhaust CPU/RAM: recreate RAM collection, sweep workers×concurrency, track peak QPS.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH=/usr/local/go/bin:$PATH
OUT="$ROOT/results/go_exhaust"
mkdir -p "$OUT" "$ROOT/logs" "$ROOT/go/bin"
cd "$ROOT/go"

echo "=== build ==="
go build -o bin/searchd ./cmd/searchd
go build -o bin/loadtest ./cmd/loadtest
go build -o bin/ingest ./cmd/ingest

echo "=== retune Qdrant container (8 search threads) ==="
if docker ps --format '{{.Names}}' | grep -qx qdrant; then
  docker rm -f qdrant >/dev/null
fi
docker run -d --name qdrant --restart unless-stopped \
  -p 6333:6333 -p 6334:6334 \
  -v "$ROOT/data/qdrant/storage:/qdrant/storage" \
  -e QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS=8 \
  -e QDRANT__STORAGE__PERFORMANCE__MAX_OPTIMIZATION_THREADS=4 \
  qdrant/qdrant:v1.13.2 >/dev/null

for _ in $(seq 1 60); do
  curl -sf http://127.0.0.1:6333/readyz >/dev/null && break
  sleep 0.5
done
echo "qdrant ready"

echo "=== ingest RAM collection (shards=4, on_disk=false) ==="
./bin/ingest \
  --collection msmarco_1m_ram \
  --passages "$ROOT/data/msmarco/subset_1m/embeddings/passages.f32.npy" \
  --shards 4 --batch 1024 --on-disk=false --recreate \
  | tee "$OUT/ingest_ram.log"

stop_workers() {
  if [[ -f "$ROOT/logs/searchd_mp.pids" ]]; then
    while read -r p; do kill "$p" 2>/dev/null || true; done < "$ROOT/logs/searchd_mp.pids"
    rm -f "$ROOT/logs/searchd_mp.pids"
  fi
  fuser -k 8080/tcp 2>/dev/null || true
  sleep 0.5
}

run_one() {
  local workers=$1 ef=$2 collection=$3 tag=$4
  stop_workers
  WORKERS="$workers" EF="$ef" COLLECTION="$collection" \
    bash "$ROOT/scripts/run_go_searchd_mp.sh"

  # warm
  ./bin/loadtest --url http://127.0.0.1:8080/search --hold 3s --warmup 5s \
    --slo-p99-ms 99999 --max-error-rate 1 --concurrencies 32 >/dev/null

  echo "=== loadtest workers=$workers ef=$ef collection=$collection ==="
  ./bin/loadtest \
    --url http://127.0.0.1:8080/search \
    --hold 12s --warmup 4s \
    --slo-p99-ms 99999 --max-error-rate 1.0 \
    --concurrencies 16,32,64,96,128,192,256,384,512 \
    --out "$OUT/${tag}.json" | tee "$OUT/${tag}.log"

  # sample resource use
  {
    echo "tag=$tag workers=$workers"
    ps -o pid,rss,pcpu,cmd -p "$(tr '\n' ',' < "$ROOT/logs/searchd_mp.pids" | sed 's/,$//')" || true
    docker stats qdrant --no-stream || true
    free -h | head -2
  } | tee "$OUT/${tag}_resources.txt"
}

# CPU sampler during peak run
(
  for i in $(seq 1 200); do
    date +%s
    grep 'cpu ' /proc/stat | awk '{u=$2+$4; t=$2+$3+$4+$5; if(NR==1){u1=u;t1=t} else print (u-u1)*100/(t-t1); u1=u;t1=t}'
    sleep 2
  done
) > "$OUT/cpu_sample.txt" &
CPUPID=$!

# Matrix: on-disk orig collection vs RAM; workers 1/2/4/8; ef 64 and 32
run_one 1 64 msmarco_1m       w1_ef64_disk
run_one 4 64 msmarco_1m       w4_ef64_disk
run_one 1 64 msmarco_1m_ram   w1_ef64_ram
run_one 2 64 msmarco_1m_ram   w2_ef64_ram
run_one 4 64 msmarco_1m_ram   w4_ef64_ram
run_one 8 64 msmarco_1m_ram   w8_ef64_ram
run_one 4 32 msmarco_1m_ram   w4_ef32_ram
run_one 8 32 msmarco_1m_ram   w8_ef32_ram

# final autocannon blast on best-ish config (w8 ef32)
stop_workers
WORKERS=8 EF=32 COLLECTION=msmarco_1m_ram bash "$ROOT/scripts/run_go_searchd_mp.sh"
./bin/loadtest --url http://127.0.0.1:8080/search --hold 3s --warmup 4s --slo-p99-ms 99999 --max-error-rate 1 --concurrencies 64 >/dev/null
if command -v autocannon >/dev/null; then
  autocannon -c 256 -d 20 -m POST -H 'Content-Type: application/json' \
    -b '{"qidx":42,"k":10}' http://127.0.0.1:8080/search \
    | tee "$OUT/autocannon_w8_ef32_c256.txt"
fi

kill "$CPUPID" 2>/dev/null || true
stop_workers

# summarize peaks
python3 - <<'PY' | tee "$OUT/summary.txt"
import json, glob, os
root = os.environ.get("OUT", "/home/ec2-user/rag-on-steroids/results/go_exhaust")
rows = []
for path in sorted(glob.glob(root + "/*.json")):
    with open(path) as f:
        d = json.load(f)
    hist = d.get("history") or []
    if not hist:
        continue
    best = max(hist, key=lambda h: h["achieved_qps"])
    rows.append((os.path.basename(path), best["achieved_qps"], best["concurrency"], best["p99_ms"], best["p50_ms"]))
rows.sort(key=lambda r: -r[1])
print(f"{'config':<28} {'peak_qps':>10} {'c':>6} {'p99':>8} {'p50':>8}")
for name, qps, c, p99, p50 in rows:
    print(f"{name:<28} {qps:10.1f} {c:6d} {p99:8.1f} {p50:8.1f}")
if rows:
    print("\nBEST:", rows[0][0], f"{rows[0][1]:.0f} QPS")
PY

echo EXHAUST_DONE
