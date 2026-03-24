"""
Biomedizinischer Benchmark: Synthetische SNP-/Epistasie-Datensaetze.

Testet, wie gut verschiedene Regellernsysteme Feature-Interaktionen (Epistasie)
und hochdimensionale Daten mit wenig Signal erkennen.  Diese Datensaetze sind
speziell darauf ausgelegt, Unterschiede zwischen den Algorithmen aufzuzeigen:

  - Epistasie: Nur Interaktionen, kein Haupteffekt → logicGP/Michigan sollten
    baumbasierte Methoden (CART) deutlich schlagen.
  - XOR/Paritaet: Nicht-linear trennbar, erfordert konjunktive Regeln.
  - Hochdimensional (p >> n): Overfitting-Verhalten wird sichtbar.
  - Imbalanced: Seltene-Krankheit-Szenario, F1-macro zeigt Unterschiede.

Aufruf:
    python examples/benchmark_biomedical.py
    python examples/benchmark_biomedical.py --datasets synth_epistasis_2way_easy synth_xor_3bit
    python examples/benchmark_biomedical.py --include-pmlb
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from scoredrulesets.benchmarking import (
    BenchmarkConfig,
    run_benchmarks,
    aggregate_benchmark_results,
    aggregated_results_as_dicts,
    format_benchmark_leaderboard_table,
    format_benchmark_report_markdown,
)
from scoredrulesets.benchmarking.runner import results_as_dicts

# Kern-Estimatoren fuer den Vergleich
_DEFAULT_ESTIMATORS = [
    "wrapper_cart",
    "wrapper_hs",
    "wrapper_pittsburgh",
    "wrapper_logicgp",
    "wrapper_rulefit",
    "gp",
]


def main():
    parser = argparse.ArgumentParser(
        description="Biomedizinischer Benchmark (Epistasie, XOR, hochdimensional, imbalanced)"
    )
    parser.add_argument(
        "--estimators",
        nargs="+",
        default=None,
        help=f"Estimator names (default: {', '.join(_DEFAULT_ESTIMATORS)})",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset names or group alias: 'synthetic', 'epistasis', 'pmlb' (default: synthetic)",
    )
    parser.add_argument(
        "--include-pmlb",
        action="store_true",
        default=False,
        help="Lade auch pmlb-Datensaetze (erfordert: pip install pmlb)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Anzahl Wiederholungen pro Datensatz/Estimator (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/biomedical"),
        help="Ausgabeverzeichnis (default: benchmarks/biomedical)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Timeout in Sekunden pro Einzellauf (default: 300). 0 = kein Timeout.",
    )
    args = parser.parse_args()

    estimator_names = args.estimators or _DEFAULT_ESTIMATORS
    dataset_names = args.datasets or ["synthetic"]

    config = BenchmarkConfig(
        dataset_names=dataset_names,
        estimator_names=estimator_names,
        repeats=args.repeats,
        random_state=42,
        show_progress=True,
        include_online_uci=False,
        include_pmlb=args.include_pmlb,
        timeout_seconds=args.timeout if args.timeout > 0 else None,
    )

    print("=" * 70)
    print("BIOMEDIZINISCHER BENCHMARK")
    print("  Datensaetze:", ", ".join(dataset_names))
    print("  Estimatoren:", ", ".join(estimator_names))
    print("  Wiederholungen:", args.repeats)
    if args.include_pmlb:
        print("  pmlb-Datensaetze: aktiviert")
    print("=" * 70)

    t0 = time.time()
    results = run_benchmarks(config)
    elapsed = time.time() - t0
    print(f"\nBenchmark abgeschlossen in {elapsed:.1f}s ({len(results)} Laeufe)")

    # Aggregation
    agg = aggregate_benchmark_results(results)

    # Ausgabe
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    raw_path = outdir / "biomedical_results.json"
    with open(raw_path, "w") as f:
        json.dump(results_as_dicts(results), f, indent=2, default=str)
    print(f"  Rohergebnisse → {raw_path}")

    agg_path = outdir / "biomedical_results_aggregated.json"
    with open(agg_path, "w") as f:
        json.dump(aggregated_results_as_dicts(agg), f, indent=2, default=str)
    print(f"  Aggregiert   → {agg_path}")

    md_path = outdir / "biomedical_report.md"
    md = format_benchmark_report_markdown(agg)
    md_path.write_text(md)
    print(f"  Report       → {md_path}")

    # Leaderboard in Terminal
    print("\n" + format_benchmark_leaderboard_table(agg))


if __name__ == "__main__":
    main()

