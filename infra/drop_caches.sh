#!/usr/bin/env bash
# Drop OS page cache before a cold bench run. Requires root.
set -euo pipefail
sync
echo 3 > /proc/sys/vm/drop_caches
echo "Caches dropped."
