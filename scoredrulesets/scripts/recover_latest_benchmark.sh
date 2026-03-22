#!/usr/bin/env bash
set -euo pipefail

# Run from repository root regardless of current working directory.
cd "$(dirname "$0")/.."

# Use local venv if present.
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

# Always recover from the latest benchmark_full_report log.
python examples/recover_heatmap_from_log.py --latest-log "$@"

