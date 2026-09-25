#!/usr/bin/env bash
# Placeholder teardown for rented servers — extend when infra is provisioned.
set -euo pipefail
echo "teardown: no managed cloud resources yet. Stop local containers:"
docker rm -f context-engine 2>/dev/null || true
