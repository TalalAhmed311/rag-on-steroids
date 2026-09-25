#!/usr/bin/env bash
# Install + run Qdrant locally (binary, no Docker required).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
QDIR="${QDRANT_DIR:-$ROOT/data/qdrant}"
VER="${QDRANT_VERSION:-v1.13.2}"
mkdir -p "$QDIR"
cd "$QDIR"

if [[ ! -x ./qdrant ]]; then
  ARCH=$(uname -m)
  case "$ARCH" in
    x86_64) ASSET="qdrant-x86_64-unknown-linux-gnu.tar.gz" ;;
    aarch64) ASSET="qdrant-aarch64-unknown-linux-gnu.tar.gz" ;;
    *) echo "unsupported arch $ARCH"; exit 1 ;;
  esac
  URL="https://github.com/qdrant/qdrant/releases/download/${VER}/${ASSET}"
  echo "Downloading $URL"
  curl -L --fail -o qdrant.tgz "$URL"
  tar -xzf qdrant.tgz
  chmod +x qdrant
fi

export QDRANT__SERVICE__HTTP_PORT="${QDRANT_HTTP_PORT:-6333}"
export QDRANT__SERVICE__GRPC_PORT="${QDRANT_GRPC_PORT:-6334}"
export QDRANT__STORAGE__STORAGE_PATH="$QDIR/storage"
mkdir -p "$QDIR/storage"
echo "Starting Qdrant on :$QDRANT__SERVICE__HTTP_PORT (storage=$QDIR/storage)"
exec ./qdrant
