"""
Multiplexer-Benchmark: Ziel F1=1.0 auf dem vollstaendigen Datensatz (kein Train/Test-Split).

Klassische boolesche Benchmark-Datensaetze (6-MUX, 11-MUX, 20-MUX) fuer
Regellernsysteme.  Der Multiplexer ist ein Standardproblem aus der LCS-Literatur
(Learning Classifier Systems), bei dem der Ausgabewert der Datenbit an der
durch die Adressbits codierten Position ist.

Aufruf:
    python examples/benchmark_multiplexer.py
    python examples/benchmark_multiplexer.py --estimators wrapper_logicgp_mux wrapper_cart_mux
    python examples/benchmark_multiplexer.py --datasets mux_6 mux_11
"""

import argparse
import json
import sys
import time
from pathlib import Path

from scoredrulesets.benchmarking.estimators import default_estimator_specs
from scoredrulesets.benchmarking.datasets import load_multiplexer_datasets
from scoredrulesets.benchmarking import (
    BenchmarkConfig,
    run_benchmarks,
    aggregate_benchmark_results,
    aggregated_results_as_dicts,
    build_benchmark_leaderboard,
    format_benchmark_leaderboard_table,
)
from scoredrulesets.benchmarking.runner import results_as_dicts


# Standard-Estimatoren fuer Multiplexer-Benchmarks
_DEFAULT_MUX_ESTIMATORS = [
    "wrapper_logicgp_mux",
    "wrapper_logicgp_mux_rlcw",
    "wrapper_cart_mux",
]


def main():
    parser = argparse.ArgumentParser(description="Multiplexer benchmark (target: F1=1.0)")
    parser.add_argument(
        "--estimators",
        nargs="+",
        default=None,
        help=f"Estimator names (default: {', '.join(_DEFAULT_MUX_ESTIMATORS)})",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="MUX dataset names, e.g. mux_6 mux_11 (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/2026-03-multiplexer"),
        help="Output directory for results",
    )
    args = parser.parse_args()

    estimator_names = args.estimators or _DEFAULT_MUX_ESTIMATORS
    mux_datasets = load_multiplexer_datasets()
    dataset_names = args.datasets or list(mux_datasets.keys())

    # Pruefe, ob alle angeforderten Datasets verfuegbar sind
    for ds in dataset_names:
        if ds not in mux_datasets:
            print(f"[ERROR] Unbekannter Datensatz: {ds}")
            print(f"  Verfuegbar: {', '.join(mux_datasets.keys())}")
            sys.exit(1)

    # Pruefe, ob alle angeforderten Estimatoren registriert sind
    all_specs = default_estimator_specs()
    for est in estimator_names:
        if est not in all_specs:
            print(f"[ERROR] Unbekannter Estimator: {est}")
            print(f"  Verfuegbar: {', '.join(sorted(all_specs.keys()))}")
            sys.exit(1)

    print("=" * 70)
    print("MULTIPLEXER BENCHMARK (Ziel: F1 = 1.0, kein Train/Test-Split)")
    print("=" * 70)
    for ds_name in dataset_names:
        bundle = mux_datasets[ds_name]
        print(f"  {ds_name}: {bundle.X.shape[0]} Instanzen, {bundle.X.shape[1]} Features")
    print(f"  Estimatoren: {', '.join(estimator_names)}")
    print()

    config = BenchmarkConfig(
        dataset_names=dataset_names,
        estimator_names=estimator_names,
        repeats=1,  # deterministisch, kein Split → 1 Wiederholung reicht
        random_state=0,
        show_progress=True,
    )

    t0 = time.time()
    results = run_benchmarks(config)
    elapsed = time.time() - t0

    # Ergebnisse ausgeben
    print()
    print("=" * 70)
    print(f"ERGEBNISSE (Laufzeit: {elapsed:.1f}s)")
    print("=" * 70)

    aggregated = aggregate_benchmark_results(results)
    leaderboard = build_benchmark_leaderboard(aggregated)

    print(format_benchmark_leaderboard_table(leaderboard))
    print()

    # Perfekte Scores pruefen
    perfect_count = 0
    total_count = 0
    for r in results:
        if r.status == "ok" and r.f1_macro is not None:
            total_count += 1
            if abs(r.f1_macro - 1.0) < 1e-9:
                perfect_count += 1
                print(f"  ✅ PERFEKT: {r.estimator} auf {r.dataset} (F1=1.0000)")
            else:
                print(f"  ❌ {r.estimator} auf {r.dataset}: F1={r.f1_macro:.4f}")

    print(f"\n  Perfekte Scores: {perfect_count}/{total_count}")

    # Ergebnisse speichern
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = results_as_dicts(results)
    (output_dir / "mux_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    agg_payload = aggregated_results_as_dicts(aggregated)
    (output_dir / "mux_results_aggregated.json").write_text(
        json.dumps(agg_payload, indent=2), encoding="utf-8"
    )

    print(f"\n  Ergebnisse gespeichert in: {output_dir}/")


if __name__ == "__main__":
    main()


