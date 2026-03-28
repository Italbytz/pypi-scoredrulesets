"""Standard benchmark: all estimator variants on selected datasets.

This combines the full estimator catalog (as in ``benchmark_full_report``)
with a focused dataset selection.

- **Estimators:** all from ``default_estimator_specs()`` except
    MUX-specific variants (same set as the full benchmark).
- **Datasets:** 10 datasets with the highest type discrimination.

This allows direct comparison of all estimator variants without the runtime
cost of the full benchmark (which runs on all datasets).

Overall: ~N estimators x 10 datasets x 3 repeats
(instead of N x all datasets in the full benchmark).

Produces the same reports as the full benchmark:
    CSV, JSON, aggregations, heatmaps, Pareto front, leaderboard (MD + HTML).

Usage:
    python examples/benchmarks/benchmark_standard.py
    python examples/benchmarks/benchmark_standard.py --repeats 5
    python examples/benchmarks/benchmark_standard.py --timeout 120
    python examples/benchmarks/benchmark_standard.py --estimators wrapper_cart,gp,wrapper_rulegp
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import sys
import time
from pathlib import Path

from scoredrulesets.benchmarking import (
    BenchmarkConfig,
    aggregate_benchmark_results,
    aggregated_results_as_dicts,
    build_benchmark_leaderboard,
    format_benchmark_report_markdown,
    format_benchmark_report_html,
    format_benchmark_leaderboard_table,
    plot_benchmark_heatmap,
    plot_benchmark_heatmap_combined,
    plot_benchmark_results,
    plot_critical_difference_diagram,
    plot_combined_dot,
    plot_efficiency_summary,
    plot_pareto_front,
    plot_win_tie_loss_pareto_matrix,
    plot_win_tie_loss_size_matrix,
    plot_win_tie_loss_triangular_matrix,
    plot_win_tie_loss_matrix,
    run_benchmarks,
)
from scoredrulesets.benchmarking.estimators import default_estimator_specs
from scoredrulesets.benchmarking.runner import results_as_dicts

# ---------------------------------------------------------------------------
# Dataset selection (focused selection with high type discrimination)
# ---------------------------------------------------------------------------
# 10 datasets with the highest type discrimination (TypeSpread >= 0.39),
# plus one real-world UCI multiclass dataset.
STANDARD_DATASETS: list[str] = [
    # synthetic – Top Diskriminierung
    "synth_dnf_3x2",              # DNF-concept, Spread 0.604
    "synth_xor_3bit",             # XOR/parity, Spread 0.588
    "mux_11",                     # 11-Bit Multiplexer, Spread 0.525
    "synth_monk3",                # MONK-3 + noise, Spread 0.512
    "synth_checkerboard_4x4",     # Checkerboard, Spread 0.491
    "synth_overlap_4rules",       # Overlapping rules, Spread 0.479
    "synth_highdim_p500_n120",    # 500 Features / 120 Samples, Spread 0.418
    "synth_imbalanced_10pct",     # 10% minority class, Spread 0.408
    "synth_epistasis_2way_easy",  # 2-way epistasis, Spread 0.393
    # Real-World
    "uci_car_evaluation",         # 4-class, categorical, Paper-UCI
]

# ---------------------------------------------------------------------------
# Estimator selection: all from default_estimator_specs() without MUX variants
# ---------------------------------------------------------------------------
# MUX-specific variants are excluded because they are only meaningful on
# multiplexer datasets and would bias the general ranking.
_MUX_ONLY: set[str] = {
    "wrapper_logicgp_mux",
    "wrapper_logicgp_mux_rlcw",
    "wrapper_logicgp_mux_rlcw_macro",
    "wrapper_cart_mux",
}


def _standard_estimator_names() -> list[str]:
    """All registered estimators without MUX-specific variants."""
    return [e for e in default_estimator_specs() if e not in _MUX_ONLY]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    return Path("benchmarks") / "logs" / f"benchmark_standard_{ts}.log"


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


def _csv_string(rows):
    if not rows:
        return ""
    import io, csv
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def main(
    log_file: Path | None = None,
    *,
    dataset_names: list[str] | None = None,
    estimator_names: list[str] | None = None,
    repeats: int = 3,
    timeout_seconds: float | None = 300.0,
    checkpoint_path: str | Path | None = "benchmarks/checkpoint_standard.jsonl",
    output_dir: str | Path = "benchmarks/standard",
    console_title: str = "STANDARD BENCHMARK: all estimators, selected datasets",
    report_title: str = "ScoredRuleSets Standard Benchmark - all estimators, selected datasets",
):
    """Run the standard benchmark and generate all reports."""

    ds_names = dataset_names or STANDARD_DATASETS
    est_names = estimator_names or _standard_estimator_names()

    dn_display = ", ".join(ds_names)
    en_display = ", ".join(est_names)
    timeout_display = f"{timeout_seconds:.0f}s" if timeout_seconds else "disabled"
    ckpt_display = str(checkpoint_path) if checkpoint_path else "disabled"
    total_runs = len(ds_names) * len(est_names) * repeats

    print("=" * 70)
    print(console_title)
    print("=" * 70)
    print(f"  Datasets ({len(ds_names):2d}): {dn_display}")
    print(f"  Estimators   ({len(est_names):2d}): {en_display}")
    print(f"  Repeats:    {repeats}")
    print(f"  Timeout per run:  {timeout_display}")
    print(f"  Checkpoint:        {ckpt_display}")
    print(f"  Total runs:     {total_runs}")
    print("=" * 70)

    # output directory
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = BenchmarkConfig(
        dataset_names=ds_names,
        estimator_names=est_names,
        use_paper_split_policy=True,
        include_online_uci=True,
        include_synthetic=True,
        include_pmlb=False,
        paper_uci_strict=False,
        repeats=repeats,
        random_state=42,
        show_progress=True,
        timeout_seconds=timeout_seconds,
        checkpoint_path=checkpoint_path,
    )

    # ------- Phase 1: Run benchmarks -------
    print("\n[1/3] Running benchmarks...")
    t0 = time.time()
    results = run_benchmarks(config)
    t1 = time.time()
    print(f"Benchmarks completed in {t1 - t0:.1f} seconds ({total_runs} runs).")

    # ------- Phase 2: Aggregate and generate reports -------
    print("\n[2/3] Aggregate and generate reports...")
    payload = results_as_dicts(results)
    aggregated = aggregate_benchmark_results(results, error_bar="std")
    leaderboard = build_benchmark_leaderboard(aggregated)
    aggregated_payload = aggregated_results_as_dicts(aggregated)

    # CSV / JSON
    (out_dir / "benchmark_results.csv").write_text(
        _csv_string(payload), encoding="utf-8"
    )
    (out_dir / "benchmark_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (out_dir / "benchmark_results_aggregated.csv").write_text(
        _csv_string(aggregated_payload), encoding="utf-8"
    )
    (out_dir / "benchmark_results_aggregated.json").write_text(
        json.dumps(aggregated_payload, indent=2), encoding="utf-8"
    )

    # Plots
    base = out_dir / "benchmark_results"
    png_path, pdf_path = plot_benchmark_results(
        results, output_base=base, aggregate_repeats=True
    )
    heatmap_png, heatmap_pdf = plot_benchmark_heatmap(
        results, output_base=out_dir / "benchmark_results_heatmap"
    )
    combined_png, combined_pdf = plot_benchmark_heatmap_combined(
        results, output_base=out_dir / "benchmark_results_heatmap_combined"
    )
    dot_png, dot_pdf = plot_combined_dot(
        results, output_base=out_dir / "benchmark_results_combined_dot"
    )
    pareto_png, pareto_pdf = plot_pareto_front(
        results, output_base=out_dir / "benchmark_results_pareto"
    )
    cd_png, cd_pdf = plot_critical_difference_diagram(
        results, output_base=out_dir / "benchmark_results_cd"
    )
    wtl_png, wtl_pdf = plot_win_tie_loss_matrix(
        results, output_base=out_dir / "benchmark_results_wtl"
    )
    wtl_size_png, wtl_size_pdf = plot_win_tie_loss_size_matrix(
        results, output_base=out_dir / "benchmark_results_wtl_size"
    )
    wtl_pareto_png, wtl_pareto_pdf = plot_win_tie_loss_pareto_matrix(
        results, output_base=out_dir / "benchmark_results_wtl_pareto"
    )
    wtl_tri_png, wtl_tri_pdf = plot_win_tie_loss_triangular_matrix(
        results, output_base=out_dir / "benchmark_results_wtl_triangular"
    )
    eff_png, eff_pdf = plot_efficiency_summary(
        results, output_base=out_dir / "benchmark_results_efficiency"
    )

    # Markdown-Report
    md_report = format_benchmark_report_markdown(
        leaderboard,
        title=report_title,
        config={
            "datasets": dn_display,
            "estimators": en_display,
            "repeats": repeats,
            "timeout_seconds": timeout_display,
            "design": f"{len(est_names)} estimators (all without MUX variants) x "
                      f"{len(ds_names)} datasets (highest type discrimination)",
        },
        artifact_paths={
            "raw_csv": "benchmark_results.csv",
            "raw_json": "benchmark_results.json",
            "aggregated_csv": "benchmark_results_aggregated.csv",
            "aggregated_json": "benchmark_results_aggregated.json",
            "plot_png": str(png_path.name),
            "plot_pdf": str(pdf_path.name),
            "heatmap_png": str(heatmap_png.name),
            "heatmap_pdf": str(heatmap_pdf.name),
            "combined_heatmap_png": str(combined_png.name),
            "combined_heatmap_pdf": str(combined_pdf.name),
            "combined_dot_png": str(dot_png.name),
            "combined_dot_pdf": str(dot_pdf.name),
            "pareto_png": str(pareto_png.name),
            "pareto_pdf": str(pareto_pdf.name),
            "cd_png": str(cd_png.name),
            "cd_pdf": str(cd_pdf.name),
            "wtl_png": str(wtl_png.name),
            "wtl_pdf": str(wtl_pdf.name),
            "wtl_size_png": str(wtl_size_png.name),
            "wtl_size_pdf": str(wtl_size_pdf.name),
            "wtl_pareto_png": str(wtl_pareto_png.name),
            "wtl_pareto_pdf": str(wtl_pareto_pdf.name),
            "wtl_triangular_png": str(wtl_tri_png.name),
            "wtl_triangular_pdf": str(wtl_tri_pdf.name),
            "efficiency_png": str(eff_png.name),
            "efficiency_pdf": str(eff_pdf.name),
        },
        notes=[
            "Standard benchmark: all estimator variants on selected datasets.",
            "Dataset-selection basiert auf TypeSpread-Analyse (Pareto-Checkpoint, "
            "2189 runs, 24 datasets, 31 estimators).",
            "Estimator selection: full catalog from default_estimator_specs() "
            "without MUX-specific variants.",
            f"Timeout per single run: {timeout_display}.",
            f"{repeats} Repeats with random_state=42.",
        ],
    )
    (out_dir / "benchmark_report.md").write_text(md_report, encoding="utf-8")

    html_report = format_benchmark_report_html(
        leaderboard,
        title=report_title,
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
            "plot_png": str(png_path.name),
            "plot_pdf": str(pdf_path.name),
            "heatmap_png": str(heatmap_png.name),
            "heatmap_pdf": str(heatmap_pdf.name),
            "combined_heatmap_png": str(combined_png.name),
            "combined_heatmap_pdf": str(combined_pdf.name),
            "combined_dot_png": str(dot_png.name),
            "combined_dot_pdf": str(dot_pdf.name),
            "pareto_png": str(pareto_png.name),
            "pareto_pdf": str(pareto_pdf.name),
            "cd_png": str(cd_png.name),
            "cd_pdf": str(cd_pdf.name),
            "wtl_png": str(wtl_png.name),
            "wtl_pdf": str(wtl_pdf.name),
            "wtl_size_png": str(wtl_size_png.name),
            "wtl_size_pdf": str(wtl_size_pdf.name),
            "wtl_pareto_png": str(wtl_pareto_png.name),
            "wtl_pareto_pdf": str(wtl_pareto_pdf.name),
            "wtl_triangular_png": str(wtl_tri_png.name),
            "wtl_triangular_pdf": str(wtl_tri_pdf.name),
            "efficiency_png": str(eff_png.name),
            "efficiency_pdf": str(eff_pdf.name),
        },
        notes=[
            "Standard benchmark: all estimator variants, selected datasets.",
            f"Timeout per single run: {timeout_display}.",
            f"{repeats} Repeats.",
        ],
    )
    (out_dir / "benchmark_report.html").write_text(html_report, encoding="utf-8")

    # ------- Phase 3: Summary -------
    print(f"\n[3/3] Done! Results in: {out_dir}/")
    print("Key files:")
    print(f"  {out_dir}/benchmark_results.csv / .json (Raw data)")
    print(f"  {out_dir}/benchmark_results_aggregated.csv / .json (Aggregated)")
    print(f"  {out_dir}/benchmark_report.md / .html (Report)")
    print(f"  {out_dir}/benchmark_results.png / .pdf (Plots)")
    print(f"  {out_dir}/benchmark_results_heatmap*.png / .pdf (Heatmaps)")
    print(f"  {out_dir}/benchmark_results_pareto.png / .pdf (Pareto front)")
    print(f"  {out_dir}/benchmark_results_cd.png / .pdf (Critical Difference Diagram)")
    print(f"  {out_dir}/benchmark_results_wtl.png / .pdf (Win/Tie/Loss Matrix)")
    print(f"  {out_dir}/benchmark_results_wtl_size.png / .pdf (Win/Tie/Loss Matrix: model size)")
    print(f"  {out_dir}/benchmark_results_wtl_pareto.png / .pdf (Pareto Win/Tie/Loss Matrix)")
    print(f"  {out_dir}/benchmark_results_wtl_triangular.png / .pdf (Triangular W/T/L Matrix)")
    print(f"  {out_dir}/benchmark_results_efficiency.png / .pdf (Executive Efficiency Summary)")

    print("\nLeaderboard (Top 15):")
    print(format_benchmark_leaderboard_table(leaderboard)[:3000])

    if log_file is not None:
        print(f"\n[LOG] completed. Full log saved to: {log_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standard benchmark: all estimators on selected datasets.",
        epilog=(
            "Design: 10 datasets (highest TypeSpread), all estimators "
            "(without MUX variants) from default_estimator_specs()."
        ),
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
        help="Comma-separated Dataset list (default: STANDARD_DATASETS).",
    )
    parser.add_argument(
        "--estimators", type=str, default="",
        help="Comma-separated Estimator list (default: all without MUX).",
    )
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="Repeats per combination (default: 3).",
    )
    parser.add_argument(
        "--timeout", type=float, default=300.0,
        help="Timeout per single run in seconds (default: 300). 0 = no timeout.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="benchmarks/checkpoint_standard.jsonl",
        help="Checkpoint file for resume (default: benchmarks/checkpoint_standard.jsonl).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="benchmarks/standard",
        help="output directory for reports/artifacts (default: benchmarks/standard).",
    )
    parser.add_argument(
        "--no-checkpoint", action="store_true",
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
            checkpoint_path=ckpt,
            output_dir=args.output_dir,
        )

