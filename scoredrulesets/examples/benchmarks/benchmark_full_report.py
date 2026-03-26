"""Comprehensive benchmark with full reporting.

- Uses all available estimators from `benchmarking/estimators.py`
- Uses all datasets: sklearn, Paper-UCI, and synthetic groups
- Per-run timeout (default: 300s) prevents hanging runs
- Prints progress and intermediate status
- Creates Markdown/HTML reports and CSV/JSON artifacts

Runtime can be very long depending on configuration.

Usage:
    python examples/benchmarks/benchmark_full_report.py
    python examples/benchmarks/benchmark_full_report.py --timeout 120
    python examples/benchmarks/benchmark_full_report.py --datasets cart_hard,ruleset_hard --estimators wrapper_cart,gp
    python examples/benchmarks/benchmark_full_report.py --skip-synthetic
"""

import argparse
import contextlib
import datetime as dt
import json
import sys
import time
from pathlib import Path

from scoredrulesets.benchmarking.estimators import default_estimator_specs
from scoredrulesets.benchmarking import (
    BenchmarkConfig,
    run_benchmarks,
    aggregate_benchmark_results,
    aggregated_results_as_dicts,
    build_benchmark_leaderboard,
    format_benchmark_report_markdown,
    format_benchmark_report_html,
    format_benchmark_leaderboard_table,
    plot_benchmark_results,
    plot_benchmark_heatmap,
    plot_benchmark_heatmap_combined,
    plot_critical_difference_diagram,
    plot_combined_dot,
    plot_efficiency_summary,
    plot_pareto_front,
    plot_win_tie_loss_pareto_matrix,
    plot_win_tie_loss_size_matrix,
    plot_win_tie_loss_triangular_matrix,
    plot_win_tie_loss_matrix,
)
from scoredrulesets.benchmarking.runner import results_as_dicts


class _TeeStream:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


def _default_log_path() -> Path:
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("benchmarks") / "logs" / f"benchmark_full_report_{ts}.log"


@contextlib.contextmanager
def _maybe_tee_to_file(log_file: Path | None):
    if log_file is None:
        yield
        return

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as f:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = _TeeStream(old_stdout, f)
        sys.stderr = _TeeStream(old_stderr, f)
        try:
            print(f"[LOG] writing console output to: {log_file}")
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def main(
    log_file: Path | None = None,
    *,
    dataset_names: list[str] | None = None,
    estimator_names: list[str] | None = None,
    repeats: int = 3,
    timeout_seconds: float | None = 300.0,
    skip_synthetic: bool = False,
    include_pmlb: bool = False,
    checkpoint_path: str | Path | None = "benchmarks/checkpoint.jsonl",
):
    # Collect all estimators and datasets
    all_estimators = list(default_estimator_specs().keys())
    # Multiplexer/MUX-specific variants are only meaningful on MUX datasets
    # and would bias the general ranking -> exclude by default.
    _MUX_ONLY = {"wrapper_logicgp_mux", "wrapper_logicgp_mux_rlcw", "wrapper_cart_mux"}
    if estimator_names is None:
        estimator_names = [e for e in all_estimators if e not in _MUX_ONLY]

    # Without explicit dataset selection: use all available sources.
    if dataset_names is None:
        dataset_names = None  # run_benchmarks will use the full registry

    dn_display = ", ".join(dataset_names) if dataset_names else "(all available)"
    en_display = ", ".join(estimator_names)
    timeout_display = f"{timeout_seconds:.0f}s" if timeout_seconds else "disabled"
    ckpt_display = str(checkpoint_path) if checkpoint_path else "disabled"

    print(f"Starting benchmark with {len(estimator_names)} estimators...")
    print(f"  Estimators:       {en_display}")
    print(f"  Datasets:       {dn_display}")
    print(f"  Repeats:    {repeats}")
    print(f"  Timeout per run:  {timeout_display}")
    print(f"  Checkpoint:        {ckpt_display}")
    print(f"  synthetic:       {'yes' if not skip_synthetic else 'no'}")
    print(f"  PMLB:              {'yes' if include_pmlb else 'no'}")

    config = BenchmarkConfig(
        dataset_names=dataset_names,
        estimator_names=estimator_names,
        use_paper_split_policy=True,
        include_online_uci=True,
        include_synthetic=not skip_synthetic,
        include_pmlb=include_pmlb,
        paper_uci_strict=False,
        repeats=repeats,
        random_state=42,
        show_progress=True,
        timeout_seconds=timeout_seconds,
        checkpoint_path=checkpoint_path,
    )

    # Fortschrittsanzeige
    print("\n[1/3] Run benchmarks...")
    t0 = time.time()
    results = run_benchmarks(config)
    t1 = time.time()
    print(f"Benchmarks completed in {t1-t0:.1f} seconds.")

    print("\n[2/3] Aggregate and generate reports...")
    payload = results_as_dicts(results)
    aggregated = aggregate_benchmark_results(results, error_bar="std")
    leaderboard = build_benchmark_leaderboard(aggregated)
    aggregated_payload = aggregated_results_as_dicts(aggregated)

    # Schreibe CSV/JSON
    Path("benchmark_results.csv").write_text(_csv_string(payload), encoding="utf-8")
    Path("benchmark_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    Path("benchmark_results_aggregated.csv").write_text(_csv_string(aggregated_payload), encoding="utf-8")
    Path("benchmark_results_aggregated.json").write_text(json.dumps(aggregated_payload, indent=2), encoding="utf-8")

    # Plots and Reports
    png_path, pdf_path = plot_benchmark_results(results, output_base=Path("benchmark_results"), aggregate_repeats=True)
    heatmap_png, heatmap_pdf = plot_benchmark_heatmap(results, output_base=Path("benchmark_results_heatmap"))
    combined_png, combined_pdf = plot_benchmark_heatmap_combined(results, output_base=Path("benchmark_results_heatmap_combined"))
    dot_png, dot_pdf = plot_combined_dot(results, output_base=Path("benchmark_results_combined_dot"))
    pareto_png, pareto_pdf = plot_pareto_front(results, output_base=Path("benchmark_results_pareto"))
    cd_png, cd_pdf = plot_critical_difference_diagram(results, output_base=Path("benchmark_results_cd"))
    wtl_png, wtl_pdf = plot_win_tie_loss_matrix(results, output_base=Path("benchmark_results_wtl"))
    wtl_size_png, wtl_size_pdf = plot_win_tie_loss_size_matrix(results, output_base=Path("benchmark_results_wtl_size"))
    wtl_pareto_png, wtl_pareto_pdf = plot_win_tie_loss_pareto_matrix(results, output_base=Path("benchmark_results_wtl_pareto"))
    wtl_tri_png, wtl_tri_pdf = plot_win_tie_loss_triangular_matrix(results, output_base=Path("benchmark_results_wtl_triangular"))
    eff_png, eff_pdf = plot_efficiency_summary(results, output_base=Path("benchmark_results_efficiency"))
    md_report = format_benchmark_report_markdown(
        leaderboard,
        title="ScoredRuleSets Benchmark Report",
        config={
            "datasets": dn_display,
            "estimators": en_display,
            "repeats": repeats,
            "timeout_seconds": timeout_display,
        },
        artifact_paths={
            "raw_csv": "benchmark_results.csv",
            "raw_json": "benchmark_results.json",
            "aggregated_csv": "benchmark_results_aggregated.csv",
            "aggregated_json": "benchmark_results_aggregated.json",
            "plot_png": str(png_path),
            "plot_pdf": str(pdf_path),
            "heatmap_png": str(heatmap_png),
            "heatmap_pdf": str(heatmap_pdf),
            "combined_heatmap_png": str(combined_png),
            "combined_heatmap_pdf": str(combined_pdf),
            "combined_dot_png": str(dot_png),
            "combined_dot_pdf": str(dot_pdf),
            "pareto_png": str(pareto_png),
            "pareto_pdf": str(pareto_pdf),
            "cd_png": str(cd_png),
            "cd_pdf": str(cd_pdf),
            "wtl_png": str(wtl_png),
            "wtl_pdf": str(wtl_pdf),
            "wtl_size_png": str(wtl_size_png),
            "wtl_size_pdf": str(wtl_size_pdf),
            "wtl_pareto_png": str(wtl_pareto_png),
            "wtl_pareto_pdf": str(wtl_pareto_pdf),
            "wtl_triangular_png": str(wtl_tri_png),
            "wtl_triangular_pdf": str(wtl_tri_pdf),
            "efficiency_png": str(eff_png),
            "efficiency_pdf": str(eff_pdf),
        },
        notes=[
            "All estimators and datasets (sklearn, Paper-UCI, synthetic: "
            "cart_hard, ruleset_hard, rule_hard), Paper-Split-Policy.",
            f"Timeout per single run: {timeout_display}.",
            f"{repeats} Repeats.",
            "Runtime and complexity can vary significantly by system.",
        ],
    )
    Path("benchmark_leaderboard.md").write_text(md_report, encoding="utf-8")
    html_report = format_benchmark_report_html(
        leaderboard,
        title="ScoredRuleSets Benchmark Report",
        config={
            "datasets": dn_display,
            "estimators": en_display,
            "repeats": repeats,
            "timeout_seconds": timeout_display,
        },
        artifact_paths={
            "raw_csv": "benchmark_results.csv",
            "raw_json": "benchmark_results.json",
            "aggregated_csv": "benchmark_results_aggregated.csv",
            "aggregated_json": "benchmark_results_aggregated.json",
            "plot_png": str(png_path),
            "plot_pdf": str(pdf_path),
            "heatmap_png": str(heatmap_png),
            "heatmap_pdf": str(heatmap_pdf),
            "combined_heatmap_png": str(combined_png),
            "combined_heatmap_pdf": str(combined_pdf),
            "combined_dot_png": str(dot_png),
            "combined_dot_pdf": str(dot_pdf),
            "pareto_png": str(pareto_png),
            "pareto_pdf": str(pareto_pdf),
            "cd_png": str(cd_png),
            "cd_pdf": str(cd_pdf),
            "wtl_png": str(wtl_png),
            "wtl_pdf": str(wtl_pdf),
            "wtl_size_png": str(wtl_size_png),
            "wtl_size_pdf": str(wtl_size_pdf),
            "wtl_pareto_png": str(wtl_pareto_png),
            "wtl_pareto_pdf": str(wtl_pareto_pdf),
            "wtl_triangular_png": str(wtl_tri_png),
            "wtl_triangular_pdf": str(wtl_tri_pdf),
            "efficiency_png": str(eff_png),
            "efficiency_pdf": str(eff_pdf),
        },
        notes=[
            "All estimators and datasets (sklearn, Paper-UCI, synthetic: "
            "cart_hard, ruleset_hard, rule_hard), Paper-Split-Policy.",
            f"Timeout per single run: {timeout_display}.",
            f"{repeats} Repeats.",
            "Runtime and complexity can vary significantly by system.",
        ],
    )
    Path("benchmark_leaderboard.html").write_text(html_report, encoding="utf-8")

    print("\n[3/3] Done! Results and reports were written.")
    print("Key files:")
    print("- benchmark_results.csv / .json (Raw data)")
    print("- benchmark_results_aggregated.csv / .json (Aggregated)")
    print("- benchmark_leaderboard.md / .html (Report)")
    print("- benchmark_results.png / .pdf (Plots)")
    print("- benchmark_results_heatmap.png / .pdf (Heatmap)")
    print("- benchmark_results_heatmap_combined.png / .pdf (Combined multi-metric heatmap)")
    print("- benchmark_results_combined_dot.png / .pdf (Combined dot plot: F1 + model size)")
    print("- benchmark_results_pareto.png / .pdf (Pareto front: F1 vs model size)")
    print("- benchmark_results_cd.png / .pdf (Critical Difference Diagram)")
    print("- benchmark_results_wtl.png / .pdf (Win/Tie/Loss Matrix)")
    print("- benchmark_results_wtl_size.png / .pdf (Win/Tie/Loss Matrix: model size)")
    print("- benchmark_results_wtl_pareto.png / .pdf (Pareto Win/Tie/Loss Matrix)")
    print("- benchmark_results_wtl_triangular.png / .pdf (Triangular W/T/L Matrix)")
    print("- benchmark_results_efficiency.png / .pdf (Executive Efficiency Summary)")

    print("\nLeaderboard (Top 10):")
    print(format_benchmark_leaderboard_table(leaderboard)[:2000])

    if log_file is not None:
        print(f"\n[LOG] completed. Full log saved to: {log_file}")


def _csv_string(rows):
    if not rows:
        return ""
    import io, csv
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run full benchmark report with all estimator/dataset combinations.",
        epilog="Dataset-Gruppen: cart_hard, ruleset_hard, rule_hard, synthetic, epistasis, paper_uci, multiplexer, pmlb",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path for console log file. Default: benchmarks/logs/benchmark_full_report_<timestamp>.log",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable automatic console log file output.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
           help="Comma-separated list of datasets or group aliases "
               "(e.g. cart_hard,ruleset_hard,rule_hard). Empty = all.",
    )
    parser.add_argument(
        "--estimators",
        type=str,
        default="",
        help="Comma-separated list of estimators. Empty = all (without MUX variants).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeats per dataset/estimator (default: 3).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Timeout in seconds pro Einzellauf (default: 300). 0 = no timeout.",
    )
    parser.add_argument(
        "--skip-synthetic",
        action="store_true",
        help="synthetice Datasets skip.",
    )
    parser.add_argument(
        "--include-pmlb",
        action="store_true",
        help="PMLB-Datasets include (requires: pip install pmlb).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="benchmarks/checkpoint.jsonl",
        help="JSONL-Checkpoint file for resume (default: benchmarks/checkpoint.jsonl).",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpoint/resume.",
    )
    args = parser.parse_args()

    selected_log_file: Path | None
    if args.no_log:
        selected_log_file = None
    elif args.log_file is not None:
        selected_log_file = args.log_file
    else:
        selected_log_file = _default_log_path()

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
            skip_synthetic=args.skip_synthetic,
            include_pmlb=args.include_pmlb,
            checkpoint_path=ckpt,
        )

