#!/usr/bin/env bash
# Launch N searchd workers on one port via SO_REUSEPORT.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH=/usr/local/go/bin:$PATH
WORKERS="${WORKERS:-4}"
ADDR="${ADDR:-:8080}"
COLLECTION="${COLLECTION:-msmarco_1m_ram}"
EF="${EF:-64}"
QDRANT_HOST="${QDRANT_HOST:-127.0.0.1}"
QDRANT_GRPC_PORT="${QDRANT_GRPC_PORT:-6334}"
PIDFILE="${PIDFILE:-$ROOT/logs/searchd_mp.pids}"

mkdir -p "$ROOT/logs" "$ROOT/go/bin"
cd "$ROOT/go"
[[ -x bin/searchd ]] || go build -o bin/searchd ./cmd/searchd

fuser -k 8080/tcp 2>/dev/null || true
sleep 0.5
: > "$PIDFILE"

for i in $(seq 1 "$WORKERS"); do
  "$ROOT/go/bin/searchd" \
    --addr "$ADDR" \
    --reuseport \
    --qdrant-host "$QDRANT_HOST" \
    --qdrant-port "$QDRANT_GRPC_PORT" \
    --collection "$COLLECTION" \
    --query-vectors "$ROOT/data/msmarco/subset_1m/embeddings/queries.f32.npy" \
    --ef "$EF" \
    >> "$ROOT/logs/go_searchd_w${i}.log" 2>&1 &
  echo $! >> "$PIDFILE"
done

for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1${ADDR}/health" >/dev/null && break
  sleep 0.25
done
echo "searchd MP workers=$WORKERS collection=$COLLECTION pids=$(tr '\n' ' ' < "$PIDFILE")"
curl -sf "http://127.0.0.1${ADDR}/health"; echo
