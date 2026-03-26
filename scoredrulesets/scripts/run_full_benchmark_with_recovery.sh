#!/usr/bin/env bash
set -euo pipefail

# Run from repository root regardless of current working directory.
cd "$(dirname "$0")/.."

# Use local venv if present.
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

python3 examples/benchmark_full_report.py "$@"

