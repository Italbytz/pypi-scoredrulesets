#!/usr/bin/env bash
set -euo pipefail

# Run from repository root regardless of current working directory.
cd "$(dirname "$0")/../.."

# Use local venv if present.
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

echo "═══════════════════════════════════════════════════════════════════"
echo "  ExSTraCS Shrinking-Varianten Benchmark"
echo "  Ziel: Varianten reduzieren / zusammenfassen"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "  Argumente:   $*"
echo "  --quick      Schnellmodus (1 Repeat, nur sklearn-Datensätze)"
echo "  --repeats N  Anzahl Wiederholungen (default: 3)"
echo "  --datasets   z.B. sklearn_iris,sklearn_wine"
echo ""

python3 -u tools/analysis/benchmark_exstracs_shrinking.py "$@"

