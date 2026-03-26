"""Normal-Lite Benchmark: wie Standard, aber mit weniger Repeats fuer schnellere Iteration.

Profil:
- gleiche Datensaetze wie Standard
- gleicher Estimator-Katalog wie Standard
- default: 2 Repeats (statt 3)
- eigene Artefakte in benchmarks/normal_lite/
- eigener Checkpoint: benchmarks/checkpoint_normal_lite.jsonl

Aufruf:
    python examples/benchmarks/benchmark_normal_lite.py
    python examples/benchmarks/benchmark_normal_lite.py --repeats 2 --timeout 180
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from benchmark_standard import _maybe_tee_to_file, main as run_standard


def _default_log_path() -> Path:
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("benchmarks") / "logs" / f"benchmark_normal_lite_{ts}.log"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normal-Lite Benchmark: Standard-Konfiguration mit weniger Repeats.",
    )
    parser.add_argument(
        "--log-file", type=Path, default=None,
        help="Console-Log-Datei (default: auto-generiert).",
    )
    parser.add_argument(
        "--no-log", action="store_true",
        help="Kein Log-File schreiben.",
    )
    parser.add_argument(
        "--datasets", type=str, default="",
        help="Kommaseparierte Dataset-Liste (default: Standard-Datasets).",
    )
    parser.add_argument(
        "--estimators", type=str, default="",
        help="Kommaseparierte Schaetzer-Liste (default: alle ohne MUX).",
    )
    parser.add_argument(
        "--repeats", type=int, default=2,
        help="Wiederholungen pro Kombination (default: 2).",
    )
    parser.add_argument(
        "--timeout", type=float, default=300.0,
        help="Timeout pro Einzellauf in Sekunden (default: 300). 0 = kein Timeout.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="benchmarks/checkpoint_normal_lite.jsonl",
        help="Checkpoint-Datei fuer Resume (default: benchmarks/checkpoint_normal_lite.jsonl).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="benchmarks/normal_lite",
        help="Ausgabeverzeichnis fuer Reports/Artefakte (default: benchmarks/normal_lite).",
    )
    parser.add_argument(
        "--no-checkpoint", action="store_true",
        help="Checkpoint/Resume deaktivieren.",
    )
    args = parser.parse_args()

    if args.no_log:
        selected_log_file: Path | None = None
    elif args.log_file is not None:
        selected_log_file = args.log_file
    else:
        selected_log_file = _default_log_path()

    ds_names = [x.strip() for x in args.datasets.split(",") if x.strip()] or None
    est_names = [x.strip() for x in args.estimators.split(",") if x.strip()] or None
    timeout = args.timeout if args.timeout > 0 else None
    ckpt = None if args.no_checkpoint else args.checkpoint

    with _maybe_tee_to_file(selected_log_file):
        run_standard(
            log_file=selected_log_file,
            dataset_names=ds_names,
            estimator_names=est_names,
            repeats=args.repeats,
            timeout_seconds=timeout,
            checkpoint_path=ckpt,
            output_dir=args.output_dir,
            console_title="NORMAL-LITE BENCHMARK: Alle Schaetzer, ausgewaehlte Datensaetze, reduzierte Repeats",
            report_title="ScoredRuleSets Normal-Lite Benchmark",
        )


if __name__ == "__main__":
    main()
