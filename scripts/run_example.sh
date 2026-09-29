#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PYTHON:-python}" scripts/run.py --arm full --output "${1:-runs/example}"
