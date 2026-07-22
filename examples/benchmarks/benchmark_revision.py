"""Revision benchmark for the ECTA paper: 10 repeats on 10 datasets.

This script is a thin wrapper around ``benchmark_paper.main()`` with
revision-specific defaults. It produces a statistically stronger replication
that supports a Friedman test with post-hoc Nemenyi correction (10 datasets
as blocks, 10 repeats for stable cell means).

Relationship to the original paper benchmark
--------------------------------------------
* **Estimators and datasets** are identical to ``benchmark_paper.py``.
* **Checkpoint** is written to ``benchmarks/checkpoint_revision.jsonl``
  — completely separate from ``benchmarks/checkpoint_paper.jsonl``.
  The original paper results are therefore never touched or overwritten.
* **Output** goes to ``benchmarks/revision/`` instead of
  ``benchmarks/paper/``.

Fallback to old results
-----------------------
The submission results remain in::

    benchmarks/checkpoint_paper.jsonl
    benchmarks/paper/

They can be replayed at any time via ``benchmark_paper.py`` (or
``replot_paper_figures.py``) without re-running any estimator.

Friedman / Nemenyi post-processing
-----------------------------------
After this script completes, run::

    python examples/benchmarks/benchmark_revision_artifacts.py \\
        --input benchmarks/revision/benchmark_results.json \\
        --output-dir benchmarks/revision/revision_artifacts

to produce the LaTeX Friedman table and critical-difference values.

Usage
-----
    # From the pypi-scoredrulesets root:
    python examples/benchmarks/benchmark_revision.py

    # Custom options:
    python examples/benchmarks/benchmark_revision.py \\
        --repeats 10 \\
        --timeout 300 \\
        --checkpoint benchmarks/checkpoint_revision.jsonl \\
        --output-dir benchmarks/revision

    # Resume an interrupted run (checkpoint already exists):
    python examples/benchmarks/benchmark_revision.py
    # (checkpoint_revision.jsonl is read automatically; completed runs are skipped)

    # Single estimator smoke-test:
    python examples/benchmarks/benchmark_revision.py \\
        --estimators paper_ruleGP --repeats 2 --no-checkpoint

Expected runtime
----------------
10 repeats x 10 datasets x 8 estimators = 800 runs.
With ruleNSGA-II (~43 s) and ExSTraCS+LRC (~128 s) as worst cases,
total wall time is roughly 3–8 hours depending on hardware.
Use ``scripts/run/run_revision_benchmark.sh`` for a supervised run with
logging.
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

# Re-use everything from the paper benchmark — estimators, datasets, helpers.
from benchmark_paper import (  # type: ignore[import-not-found]
    PAPER_ESTIMATOR_NAMES,
    _PAPER_SPECS,  # noqa: F401  (side-effect: registers specs in ESTIMATOR_SPECS)
    _default_log_path,
    _maybe_tee_to_file,
    main,
)

# ---------------------------------------------------------------------------
# Revision defaults (differ from the paper benchmark)
# ---------------------------------------------------------------------------

REVISION_REPEATS: int = 10
REVISION_CHECKPOINT: str = "benchmarks/checkpoint_revision.jsonl"
REVISION_OUTPUT_DIR: str = "benchmarks/revision"
REVISION_CONSOLE_TITLE: str = "REVISION BENCHMARK: 8 methods, 10 datasets, 10 repeats"
REVISION_REPORT_TITLE: str = (
    "ScoredRuleSets Revision Benchmark – Extended Replication Study (10 repeats)"
)


def _default_revision_log_path() -> Path:
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("benchmarks") / "logs" / f"benchmark_revision_{ts}.log"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Revision benchmark: same estimators and datasets as the paper benchmark, "
            "but 10 repeats and a separate checkpoint / output directory."
        ),
        epilog=(
            "Old paper results remain untouched in "
            "benchmarks/checkpoint_paper.jsonl and benchmarks/paper/. "
            "Run benchmark_revision_artifacts.py afterwards to produce "
            "the Friedman / Nemenyi LaTeX table."
        ),
    )
    parser.add_argument(
        "--log-file", type=Path, default=None,
        help="Console log file (default: auto-generated under benchmarks/logs/).",
    )
    parser.add_argument(
        "--no-log", action="store_true",
        help="Do not write a log file.",
    )
    parser.add_argument(
        "--datasets", type=str, default="",
        help="Comma-separated dataset list (default: same 10 as paper benchmark).",
    )
    parser.add_argument(
        "--estimators", type=str, default="",
        help=(
            "Comma-separated internal estimator keys (default: all paper estimators). "
            "Available: " + ", ".join(_PAPER_SPECS.keys())
        ),
    )
    parser.add_argument(
        "--repeats", type=int, default=REVISION_REPEATS,
        help=f"Repeats per combination (default: {REVISION_REPEATS}).",
    )
    parser.add_argument(
        "--timeout", type=float, default=300.0,
        help="Timeout per single run in seconds (default: 300). 0 = no timeout.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=REVISION_CHECKPOINT,
        help=f"Checkpoint file for resume (default: {REVISION_CHECKPOINT}).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=REVISION_OUTPUT_DIR,
        help=f"Output directory (default: {REVISION_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--no-checkpoint", action="store_true",
        help="Disable checkpoint/resume.",
    )
    args = parser.parse_args()

    selected_log_file = (
        None if args.no_log
        else (args.log_file if args.log_file is not None else _default_revision_log_path())
    )

    ds_names = [x.strip() for x in args.datasets.split(",") if x.strip()] or None
    est_names = [x.strip() for x in args.estimators.split(",") if x.strip()] or None
    timeout = args.timeout if args.timeout > 0 else None
    ckpt = None if args.no_checkpoint else args.checkpoint

    with _maybe_tee_to_file(selected_log_file):
        main(
            log_file=selected_log_file,
            dataset_names=ds_names,
            estimator_names=est_names,
            repeats=args.repeats,
            timeout_seconds=timeout,
            checkpoint_path=ckpt,
            output_dir=args.output_dir,
            console_title=REVISION_CONSOLE_TITLE,
            report_title=REVISION_REPORT_TITLE,
        )
