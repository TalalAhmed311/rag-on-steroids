#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH=/usr/local/go/bin:$PATH
cd "$ROOT/go"
[[ -x bin/searchd ]] || go build -o bin/searchd ./cmd/searchd
exec ./bin/searchd \
  --addr "${ADDR:-:8080}" \
  --qdrant-host "${QDRANT_HOST:-127.0.0.1}" \
  --qdrant-port "${QDRANT_GRPC_PORT:-6334}" \
  --collection "${COLLECTION:-msmarco_1m}" \
  --query-vectors "${ROOT}/data/msmarco/subset_1m/embeddings/queries.f32.npy" \
  --ef "${EF:-64}"
