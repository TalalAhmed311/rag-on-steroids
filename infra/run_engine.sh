#!/usr/bin/env bash
# Run the engine container with a hard memory limit (default 24g).
set -euo pipefail

MEMORY_LIMIT="${MEMORY_LIMIT:-24g}"
IMAGE="${IMAGE:-context-engine:latest}"
DATA_DIR="${DATA_DIR:-$(cd "$(dirname "$0")/.." && pwd)/data}"
NAME="${NAME:-context-engine}"

docker run --rm -it \
  --name "$NAME" \
  --memory="$MEMORY_LIMIT" \
  --memory-swap="$MEMORY_LIMIT" \
  -v "$DATA_DIR:/data:rw" \
  -v "$(cd "$(dirname "$0")/.." && pwd):/workspace:rw" \
  -w /workspace \
  "$IMAGE" \
  "$@"
