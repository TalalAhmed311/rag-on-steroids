#!/usr/bin/env bash
# Autocannon load test against a running /search endpoint.
# Body cycles qidx via a tiny node wrapper is awkward; we POST fixed JSON and
# let the server map missing diversity — for zipf-like load use bench/capacity_test.py.
set -euo pipefail
URL="${1:-http://127.0.0.1:8080/search}"
CONN="${2:-32}"
DUR="${3:-20}"
BODY='{"qidx":42,"k":10}'
echo "autocannon $URL connections=$CONN duration=${DUR}s"
autocannon -c "$CONN" -d "$DUR" -m POST \
  -H "Content-Type=application/json" \
  -b "$BODY" \
  "$URL"
