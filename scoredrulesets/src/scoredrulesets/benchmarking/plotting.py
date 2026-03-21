from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .runner import AggregatedBenchmarkResult, BenchmarkResult, aggregate_benchmark_results


def plot_benchmark_results(
    results: Iterable[BenchmarkResult],
    output_base: str | Path,
    size_metric: str = "n_rules",
    aggregate_repeats: bool = False,
    error_bar: str = "std",
) -> tuple[Path, Path]:
    """Erzeugt einen Scatter-Plot fuer Benchmark-Ergebnisse und speichert PNG/PDF.

    x-Achse: Modellgroesse (`n_rules`, `n_atoms`, `ruleset_json_bytes`)
    y-Achse: `f1_macro`
    Farbe: `fit_seconds`
    Punktlabel: `<dataset>/<estimator>#<repeat>`
    """
    valid_size_metrics = {"n_rules", "n_atoms", "ruleset_json_bytes"}
    if size_metric not in valid_size_metrics:
        raise ValueError(
            f"Unknown size_metric '{size_metric}'. Expected one of {sorted(valid_size_metrics)}"
        )

    raw_results = list(results)
    ok_results = [result for result in raw_results if result.status == "ok"]
    if not ok_results:
        raise ValueError("No successful benchmark results available for plotting")

    fig, ax = plt.subplots(figsize=(10, 6))
    if aggregate_repeats:
        aggregated = aggregate_benchmark_results(ok_results, error_bar=error_bar)
        scatter = _plot_aggregated(ax, aggregated, size_metric)
        ax.set_title(
            f"Benchmark (aggregiert): F1 vs Modellgroesse (Farbe = Fit-Zeit, Fehler = {error_bar})"
        )
    else:
        scatter = _plot_raw(ax, ok_results, size_metric)
        ax.set_title("Benchmark: F1 vs Modellgroesse (Farbe = Fit-Zeit)")

    ax.set_xlabel(size_metric)
    ax.set_ylabel("f1_macro")
    ax.grid(True, alpha=0.3)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("fit_seconds")
    fig.tight_layout()

    base = Path(output_base)
    png_path = base.with_suffix(".png")
    pdf_path = base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def _plot_raw(ax, ok_results: list[BenchmarkResult], size_metric: str):
    x_values = [getattr(result, size_metric) for result in ok_results]
    y_values = [result.f1_macro for result in ok_results]
    fit_values = [result.fit_seconds for result in ok_results]
    labels = [f"{result.dataset}/{result.estimator}#{result.repeat}" for result in ok_results]
    scatter = ax.scatter(x_values, y_values, c=fit_values, cmap="viridis", s=90, edgecolors="black")
    for x, y, label in zip(x_values, y_values, labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    return scatter


def _plot_aggregated(ax, aggregated: list[AggregatedBenchmarkResult], size_metric: str):
    metric_name = f"{size_metric}_mean"
    metric_err = f"{size_metric}_error"
    x_values = [getattr(result, metric_name) for result in aggregated]
    x_errors = [getattr(result, metric_err) for result in aggregated]
    y_values = [result.f1_macro_mean for result in aggregated]
    y_errors = [result.f1_macro_error for result in aggregated]
    fit_values = [result.fit_seconds_mean for result in aggregated]
    labels = [f"{result.dataset}/{result.estimator}" for result in aggregated]

    scatter = ax.scatter(x_values, y_values, c=fit_values, cmap="viridis", s=110, edgecolors="black")
    for x, y, xerr, yerr, label in zip(x_values, y_values, x_errors, y_errors, labels):
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor="gray", alpha=0.7, capsize=4)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    return scatter


