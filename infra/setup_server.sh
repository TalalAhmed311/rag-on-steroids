#!/usr/bin/env bash
# Local package install for Stage 0.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python3 -m pip install --user -r requirements.txt
python3 -m pip install --user -e ".[dev]"
echo "Setup complete."
