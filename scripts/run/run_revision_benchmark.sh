#!/usr/bin/env bash
# Run the revision benchmark (10 repeats, separate checkpoint).
#
# Old paper results are never touched:
#   benchmarks/checkpoint_paper.jsonl  <- submission checkpoint (read-only)
#   benchmarks/paper/                  <- submission figures and reports
#
# Revision outputs go to:
#   benchmarks/checkpoint_revision.jsonl  <- can be interrupted and resumed
#   benchmarks/revision/
#
# After completion, generate the Friedman/Nemenyi LaTeX table with:
#   python examples/benchmarks/benchmark_revision_artifacts.py \
#       --input benchmarks/revision/benchmark_results.json \
#       --output-dir benchmarks/revision/revision_artifacts
set -euo pipefail

cd "$(dirname "$0")/../.."

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

python3 examples/benchmarks/benchmark_revision.py "$@"
