"""Normal-Lite benchmark: same as standard, but with fewer repeats for faster iteration.

Profile:
- same datasets as standard
- same estimator catalog as standard
- default: 2 repeats (instead of 3)
- dedicated artifacts in benchmarks/normal_lite/
- dedicated checkpoint: benchmarks/checkpoint_normal_lite.jsonl

Usage:
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
        description="Normal-Lite Benchmark: standard configuration with fewer repeats.",
    )
    parser.add_argument(
        "--log-file", type=Path, default=None,
        help="Console log file (default: auto-generated).",
    )
    parser.add_argument(
        "--no-log", action="store_true",
        help="Do not write a log file.",
    )
    parser.add_argument(
        "--datasets", type=str, default="",
        help="Comma-separated Dataset list (default: standard datasets).",
    )
    parser.add_argument(
        "--estimators", type=str, default="",
        help="Comma-separated Estimator list (default: all without MUX).",
    )
    parser.add_argument(
        "--repeats", type=int, default=2,
        help="Repeats per combination (default: 2).",
    )
    parser.add_argument(
        "--timeout", type=float, default=300.0,
        help="Timeout per single run in seconds (default: 300). 0 = no timeout.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="benchmarks/checkpoint_normal_lite.jsonl",
        help="Checkpoint file for resume (default: benchmarks/checkpoint_normal_lite.jsonl).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="benchmarks/normal_lite",
        help="output directory for reports/artifacts (default: benchmarks/normal_lite).",
    )
    parser.add_argument(
        "--no-checkpoint", action="store_true",
        help="Disable checkpoint/resume.",
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
            console_title="NORMAL-LITE BENCHMARK: all estimators, selected datasets, reduced repeats",
            report_title="ScoredRuleSets Normal-Lite Benchmark",
        )


if __name__ == "__main__":
    main()
