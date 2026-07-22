"""Paper-Benchmark: 7 rule-learning-Methoden auf 10 selected datasets.

The goal of this benchmark is a direct method comparison for publication.
Estimators are shown with concise display names so they can be used directly
in paper tables and figures.

Estimators (short name -> internal wrapper):
    HS              <- wrapper_hs_pruned
    RuleKit         <- wrapper_rulekit_native
    ExSTraCS        <- wrapper_exstracs
    ExSTraCS (LRC)  <- wrapper_exstracs_compact (Lossy Rule Compaction)
    ruleGP          <- wrapper_rulegp_strong (logicGP mode)
    ruleNSGA-II     <- wrapper_rulensga2_strong
    ruleNLN         <- wrapper_rulenln_strong
    rulePLCS        <- wrapper_ruleplcs_strong

Naming migration:
    old ruleGP  -> rulensga2
    old ruleGP2 -> ruleGP

Datasets (10):
  Real-World (4): sklearn_breast_cancer, sklearn_wine, uci_car_evaluation,
                  uci_heart_disease
  synthetic (6): synth_dnf_3x2, synth_overlap_4rules, synth_monk3,
                   synth_xor_3bit, synth_imbalanced_10pct, synth_checkerboard_4x4

Rationale for dataset selection:
  - Real-World-Datasets provide readers with reference points from well-known benchmarks.
  - synth_dnf_3x2 / synth_overlap_4rules / synth_monk3: concept learning, bei dem
    rule-learning semantische Vorteile hat.
  - synth_xor_3bit: nonlinear, separates GP/LCS von linearen Methoden.
  - synth_imbalanced_10pct: Robustheit against classimbalance.
  - synth_checkerboard_4x4: geometrically komplex, no simple rules.

Usage:
    python examples/benchmarks/benchmark_paper.py
    python examples/benchmarks/benchmark_paper.py --repeats 5
    python examples/benchmarks/benchmark_paper.py --timeout 300
    python examples/benchmarks/benchmark_paper.py --no-checkpoint
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
    plot_dual_cd_diagram,
    plot_efficiency_summary,
    plot_2d_rank_plot,
    plot_pareto_front,
    plot_win_tie_loss_pareto_matrix,
    plot_win_tie_loss_size_matrix,
    plot_win_tie_loss_triangular_matrix,
    plot_win_tie_loss_matrix,
    run_benchmarks,
)
from scoredrulesets.benchmarking.estimators import (
    EstimatorSpec,
    ESTIMATOR_SPECS,
    default_estimator_specs,
)
from scoredrulesets.benchmarking.runner import results_as_dicts
from scoredrulesets.estimators.sklearn_wrapper import ScoredRuleSetClassifier

# ---------------------------------------------------------------------------
# Paper-Estimators: short names + identical Hyperparameter as die Wrapper
# ---------------------------------------------------------------------------
# Die Factories sind 1:1 from default_estimator_specs() imported.
# spec.name defines the display name in alln Reports, Plots and Tabellen.

_PAPER_SPECS: dict[str, EstimatorSpec] = {
    "paper_HS": EstimatorSpec(
        name="HS",
        factory=lambda: ScoredRuleSetClassifier(
            backend="hs",
            transform_params={"prune_atoms": True},
            random_state=0,
        ),
    ),
    "paper_RuleKit": EstimatorSpec(
        name="RuleKit",
        factory=lambda: ScoredRuleSetClassifier(
            backend="rulekit_native",
            backend_params={
                "max_rules": 30,
                "max_conditions": 7,
                "min_samples_leaf": 3,
                "enable_pruning": True,
                "pruning_fraction": 0.25,
            },
            random_state=0,
        ),
    ),
    "paper_ExSTraCS": EstimatorSpec(
        name="ExSTraCS",
        factory=lambda: ScoredRuleSetClassifier(
            backend="exstracs",
            backend_params={"learning_iterations": 10000},
            random_state=0,
        ),
    ),
    "paper_ExSTraCS_LRC": EstimatorSpec(
        name="ExSTraCS (LRC)",
        factory=lambda: ScoredRuleSetClassifier(
            backend="exstracs",
            backend_params={"learning_iterations": 10000},
            exstracs_params={
                "interval_merge": True,
                "interval_merge_iou_threshold": 0.3,
                "conservative_prune": True,
            },
            random_state=0,
        ),
    ),
    "paper_ruleGP": EstimatorSpec(
        name="ruleGP",
        factory=lambda: ScoredRuleSetClassifier(
            backend="rulegp",
            backend_params={
                "f1_averaging": "macro",
                "atom_space_strategy": "hybrid",
                "atom_preselection_strategy": "logicgp_binned_sets",
                "max_generations": 500,
                "stagnation_generations": 80,
                "population_size": 120,
                "n_adaptations_per_gen": 20,
            },
            max_fit_seconds=240,
            random_state=0,
        ),
    ),
    "paper_ruleNLN": EstimatorSpec(
        name="ruleNLN",
        factory=lambda: ScoredRuleSetClassifier(
            backend="rulenln",
            backend_params={
                "n_rules": 20,
                "n_bins": 6,
                "learning_rate": 0.15,
                "l1_conj": 0.0003,
                "l1_score": 0.00015,
                "epochs": 600,
                "early_stopping_rounds": 50,
                "atom_threshold": 0.06,
            },
            random_state=0,
        ),
    ),
    "paper_rulePLCS": EstimatorSpec(
        name="rulePLCS",
        factory=lambda: ScoredRuleSetClassifier(
            backend="ruleplcs",
            backend_params={
                "population_size": 200,
                "n_iterations": 50,
                "n_repetitions": 2,
                "tournament_size": 4,
                "crossover_prob": 0.6,
                "mutation_prob": 0.6,
                "max_rules": 15,
                "default_class_policy": "major",
            },
            random_state=0,
        ),
    ),
    "paper_ruleNSGA2": EstimatorSpec(
        name="ruleNSGA-II",
        factory=lambda: ScoredRuleSetClassifier(
            backend="rulensga2",
            backend_params={
                "population_size": 150,
                "generations": 250,
                "max_rules": 12,
                "max_atoms_per_rule": 5,
                "tournament_size": 4,
                "early_stopping_rounds": 30,
                "enable_compaction": True,
            },
            random_state=0,
        ),
    ),
}

# Register paper estimators globally so run_benchmarks() can find them
ESTIMATOR_SPECS.update(_PAPER_SPECS)

PAPER_ESTIMATOR_NAMES: list[str] = list(_PAPER_SPECS.keys())

# ---------------------------------------------------------------------------
# Dataset-selection
# ---------------------------------------------------------------------------
PAPER_DATASETS: list[str] = [
    # Real-world (4) - known reference benchmarks
    "sklearn_breast_cancer",    # binary, 569 x 30, medical
    "sklearn_wine",             # 3-class, 178 x 13, chemical
    "uci_car_evaluation",       # 4-class, 1728 x 6, categorical
    "uci_heart_disease",        # binary, 303 x 13, medical
    # Synthetic - concept learning with clear rule structures
    "synth_dnf_3x2",            # DNF concept (spread 0.604), rule learning should dominate
    "synth_overlap_4rules",     # Overlapping rules (Spread 0.479), conflict-resolution stress test
    "synth_monk3",              # MONK-3 + noise (spread 0.512), classic ML benchmark
    # Synthetic - challenging
    "synth_xor_3bit",           # nonlinear / parity (Spread 0.588), separates GP/LCS
    "synth_imbalanced_10pct",   # 10% minority class (spread 0.408), imbalance robustness
    "synth_checkerboard_4x4",   # geometrically complex (spread 0.491), no simple rules
]

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
    return Path("benchmarks") / "logs" / f"benchmark_paper_{ts}.log"


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
    checkpoint_path: str | Path | None = "benchmarks/checkpoint_paper.jsonl",
    output_dir: str | Path = "benchmarks/paper",
    console_title: str = "PAPER BENCHMARK: 8 methods, 10 datasets",
    report_title: str = "ScoredRuleSets Paper Benchmark - Rule-Based Classifiers Comparison",
):
    """Run the paper benchmark and generate all reports."""

    ds_names = dataset_names or PAPER_DATASETS
    est_names = estimator_names or PAPER_ESTIMATOR_NAMES

    # Display names (spec.name) for the estimator list in output
    registry = default_estimator_specs()
    display_names = [registry[e].name for e in est_names if e in registry]

    dn_display = ", ".join(ds_names)
    en_display = ", ".join(display_names)
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
    print(f"  Total runs:        {total_runs}")
    print("=" * 70)

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
    dual_cd_png, dual_cd_pdf = plot_dual_cd_diagram(
        results, output_base=out_dir / "benchmark_results_dual_cd"
    )
    rank2d_png, rank2d_pdf = plot_2d_rank_plot(
        results, output_base=out_dir / "benchmark_results_rank2d"
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
            "design": (
                f"8 Rule-Based Classifiers x {len(ds_names)} Datasets "
                f"({len([d for d in ds_names if d.startswith('sklearn_') or d.startswith('uci_')])} real-world, "
                f"{len([d for d in ds_names if d.startswith('synth_')])} synthetic), "
                f"{repeats} repeats"
            ),
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
            "dual_cd_png": str(dual_cd_png.name),
            "dual_cd_pdf": str(dual_cd_pdf.name),
            "rank2d_png": str(rank2d_png.name),
            "rank2d_pdf": str(rank2d_pdf.name),
        },
        notes=[
            "Paper Benchmark: 7 rule-based classifiers on 10 selected datasets.",
            "Real-world datasets: sklearn_breast_cancer, sklearn_wine, "
            "uci_car_evaluation, uci_heart_disease.",
            "Synthetic datasets chosen for concept diversity: DNF rules, overlapping "
            "rules, MONK-3 noise, XOR/parity, class imbalance, geometric complexity.",
            "ExSTraCS (LRC) applies Lossy Rule Compaction post-hoc "
            "(interval merge + conservative pruning; 0-6% F1 loss, 29-98% rule reduction).",
            f"Timeout per run: {timeout_display}. {repeats} repeats, random_state=42.",
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
            "dual_cd_png": str(dual_cd_png.name),
            "dual_cd_pdf": str(dual_cd_pdf.name),
            "rank2d_png": str(rank2d_png.name),
            "rank2d_pdf": str(rank2d_pdf.name),
        },
        notes=[
            "Paper Benchmark: 7 rule-based classifiers on 10 selected datasets.",
            f"Timeout per run: {timeout_display}. {repeats} repeats.",
        ],
    )
    (out_dir / "benchmark_report.html").write_text(html_report, encoding="utf-8")

    # ------- Phase 3: Summary -------
    print(f"\n[3/3] Done! Results in: {out_dir}/")
    print("Key files:")
    print(f"  {out_dir}/benchmark_results.csv / .json          (Raw data)")
    print(f"  {out_dir}/benchmark_results_aggregated.csv / .json (Aggregated)")
    print(f"  {out_dir}/benchmark_report.md / .html            (Report)")
    print(f"  {out_dir}/benchmark_results.png / .pdf           (Plots)")
    print(f"  {out_dir}/benchmark_results_heatmap*.png / .pdf  (Heatmaps)")
    print(f"  {out_dir}/benchmark_results_pareto.png / .pdf    (Pareto front)")
    print(f"  {out_dir}/benchmark_results_cd.png / .pdf        (Critical Difference Diagram)")
    print(f"  {out_dir}/benchmark_results_wtl.png / .pdf       (Win/Tie/Loss Matrix)")
    print(f"  {out_dir}/benchmark_results_wtl_size.png / .pdf  (Win/Tie/Loss Matrix: model size)")
    print(f"  {out_dir}/benchmark_results_wtl_pareto.png / .pdf (Pareto Win/Tie/Loss Matrix)")
    print(f"  {out_dir}/benchmark_results_wtl_triangular.png / .pdf (Triangular W/T/L Matrix)")
    print(f"  {out_dir}/benchmark_results_efficiency.png / .pdf (Executive Efficiency Summary)")
    print(f"  {out_dir}/benchmark_results_dual_cd.png / .pdf (Dual CD Diagram: F1 + Complexity)")
    print(f"  {out_dir}/benchmark_results_rank2d.png / .pdf (2D Rank Plot: Quality vs. Complexity)")

    print("\nLeaderboard (Top 10):")
    print(format_benchmark_leaderboard_table(leaderboard)[:3000])

    if log_file is not None:
        print(f"\n[LOG] completed. Full log saved to: {log_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Paper-Benchmark: 7 rule-based classifiers on 10 selected datasets.",
        epilog=(
            "Estimators: HS, RuleKit, ExSTraCS, ExSTraCS (LRC), ruleGP, ruleNSGA-II, ruleNLN, rulePLCS. "
            "10 Datasets: 4 real-world (sklearn/UCI) + 6 synthetic."
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
        help="Comma-separated Dataset list (default: PAPER_DATASETS).",
    )
    parser.add_argument(
        "--estimators", type=str, default="",
        help=(
            "Comma-separated internal estimator keys (default: all paper estimators). "
            "Available: " + ", ".join(_PAPER_SPECS.keys())
        ),
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
        "--checkpoint", type=str, default="benchmarks/checkpoint_paper.jsonl",
        help="Checkpoint file for resume (default: benchmarks/checkpoint_paper.jsonl).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="benchmarks/paper",
        help="output directory (default: benchmarks/paper).",
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
