"""Generate per-dataset pareto plots and a rank2d plot for the DATA 2026 talk.

Produces one pareto-front PNG/PDF per UCI dataset plus an aggregated rank2d plot.
Outputs go to articles/2026/scoredrulesets/talk_figures/.

Usage (from pypi-scoredrulesets root):
    python examples/benchmarks/talk_pareto_per_dataset.py
    python examples/benchmarks/talk_pareto_per_dataset.py \\
        --checkpoint benchmarks/checkpoint_paper.jsonl \\
        --output-dir articles/2026/scoredrulesets/talk_figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scoredrulesets.benchmarking.runner import (
    load_checkpoint_results,
    aggregate_benchmark_results,
    build_pareto_per_dataset,
)
from scoredrulesets.benchmarking.plotting import plot_2d_rank_plot


def _estimator_color_map(estimator_names: list[str]) -> dict[str, tuple]:
    import matplotlib.cm as cm
    cmap = cm.get_cmap("tab10")
    return {name: cmap(i % 10) for i, name in enumerate(estimator_names)}


def plot_single_dataset_pareto(
    aggregated,
    ds_name: str,
    output_base: Path,
    size_metric: str = "n_atoms",
) -> tuple[Path, Path]:
    """Plot pareto front for one dataset as a standalone figure."""
    ds_results = [r for r in aggregated if r.dataset == ds_name]
    if not ds_results:
        raise ValueError(f"No results for dataset '{ds_name}'")

    pareto_per_ds = build_pareto_per_dataset(
        aggregated,
        quality_attr="f1_macro_mean",
        size_attr=f"{size_metric}_mean",
    )
    pareto_set = {r.estimator for r in pareto_per_ds.get(ds_name, [])}
    estimator_names = sorted({r.estimator for r in aggregated})
    estimator_colors = _estimator_color_map(estimator_names)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    size_col = f"{size_metric}_mean"
    x_all, y_all, est_all = [], [], []
    x_pareto, y_pareto, labels_pareto = [], [], []

    for r in ds_results:
        sz = getattr(r, size_col)
        f1 = r.f1_macro_mean
        if sz is None or f1 is None:
            continue
        x_all.append(float(sz))
        y_all.append(float(f1))
        est_all.append(r.estimator)
        if r.estimator in pareto_set:
            x_pareto.append(float(sz))
            y_pareto.append(float(f1))
            labels_pareto.append(r.estimator)

    for estimator in estimator_names:
        ex = [x for x, e in zip(x_all, est_all) if e == estimator and e not in pareto_set]
        ey = [y for y, e in zip(y_all, est_all) if e == estimator and e not in pareto_set]
        if ex:
            ax.scatter(ex, ey, s=55, c=[estimator_colors[estimator]], alpha=0.3,
                       edgecolors="none", zorder=2)

    for estimator in estimator_names:
        ex = [x for x, e in zip(x_pareto, labels_pareto) if e == estimator]
        ey = [y for y, e in zip(y_pareto, labels_pareto) if e == estimator]
        if ex:
            ax.scatter(ex, ey, s=160, c=[estimator_colors[estimator]],
                       edgecolors="black", linewidths=1.2, zorder=4)
            for xi, yi in zip(ex, ey):
                ax.annotate(estimator, (xi, yi), textcoords="offset points",
                            xytext=(5, 3), fontsize=8, color=estimator_colors[estimator])

    if len(x_pareto) > 1:
        order = sorted(range(len(x_pareto)), key=lambda i: x_pareto[i])
        ax.step([x_pareto[i] for i in order], [y_pareto[i] for i in order],
                where="post", linestyle="--", color="black", linewidth=1.2,
                alpha=0.7, zorder=1)

    ax.set_title(ds_name, fontsize=14, fontweight="bold")
    ax.set_xlabel(f"Model size ({size_metric})", fontsize=11)
    ax.set_ylabel("F1-macro", fontsize=11)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    if x_all:
        ax.set_xscale("symlog", linthresh=1.0, linscale=1.0)
        xmax = max(x_all) * 1.1
        ax.set_xlim(left=0.0, right=xmax)
        tick_candidates = [0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0]
        tick_values = [t for t in tick_candidates if t <= xmax]
        if len(tick_values) < 2:
            tick_values = [0.0, max(1.0, xmax)]
        ax.set_xticks(tick_values)
        ax.set_xticklabels([f"{int(t)}" if t >= 1 else "0" for t in tick_values])

    # Legend at bottom
    handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=estimator_colors[e], markeredgecolor="black",
               markeredgewidth=0.7, markersize=8, label=e)
        for e in estimator_names
        if any(est == e for est in est_all)
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              frameon=True, framealpha=0.8, ncol=2)

    fig.tight_layout()
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main(checkpoint_path: str | Path, output_dir: str | Path) -> None:
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results, _ = load_checkpoint_results(checkpoint_path)
    ok_results = [r for r in results if r.status == "ok"]
    print(f"Loaded {len(ok_results)} ok results from {checkpoint_path}")

    aggregated = aggregate_benchmark_results(ok_results, error_bar="std")
    datasets = sorted({r.dataset for r in aggregated})
    print(f"Datasets: {datasets}")

    # Per-dataset pareto plots
    for ds in datasets:
        safe_name = ds.replace(" ", "_").lower()
        out_base = output_dir / f"pareto_{safe_name}"
        png, pdf = plot_single_dataset_pareto(aggregated, ds, out_base)
        print(f"  {ds}: {pdf}")

    # Rank2D across all datasets
    rank2d_png, rank2d_pdf = plot_2d_rank_plot(
        ok_results, output_base=output_dir / "rank2d"
    )
    print(f"Rank2D: {rank2d_pdf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate per-dataset pareto plots for the DATA 2026 talk."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="benchmarks/checkpoint_paper.jsonl",
        help="Path to checkpoint JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../../articles/2026/scoredrulesets/talk_figures",
        help="Output directory for generated figures.",
    )
    args = parser.parse_args()
    main(args.checkpoint, args.output_dir)
