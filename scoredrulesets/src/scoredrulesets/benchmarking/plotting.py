from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from .runner import AggregatedBenchmarkResult, BenchmarkResult, aggregate_benchmark_results


def plot_benchmark_results(
    results: Iterable[BenchmarkResult],
    output_base: str | Path,
    size_metric: str = "n_rules",
    aggregate_repeats: bool = False,
    error_bar: str = "std",
) -> tuple[Path, Path]:
    """Create a scatter plot for benchmark results and save it as PNG/PDF.

    x-axis: model size (`n_rules`, `n_atoms`, `ruleset_json_bytes`)
    y-axis: `f1_macro`
    color: `fit_seconds`
    point label: `<dataset>/<estimator>#<repeat>`
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

    if aggregate_repeats:
        aggregated_results = aggregate_benchmark_results(ok_results, error_bar=error_bar)
        dataset_names = sorted({result.dataset for result in aggregated_results})
        fit_values = [float(result.fit_seconds_mean or 0.0) for result in aggregated_results]
        title = (
            f"Benchmark (aggregated, per dataset): F1 vs model size "
            f"(color = fit time, error = {error_bar})"
        )
    else:
        aggregated_results = None
        dataset_names = sorted({result.dataset for result in ok_results})
        fit_values = [float(result.fit_seconds or 0.0) for result in ok_results]
        title = "Benchmark (per dataset): F1 vs model size (color = fit time)"

    fig, axes = _build_dataset_axes(len(dataset_names))
    fig.suptitle(title)
    fit_norm = _build_fit_norm(fit_values)

    first_scatter = None
    used_axes = []
    for ax, dataset_name in zip(axes, dataset_names):
        used_axes.append(ax)
        if aggregate_repeats:
            dataset_results = [
                result for result in aggregated_results or [] if result.dataset == dataset_name
            ]
            scatter = _plot_aggregated(ax, dataset_results, size_metric, fit_norm=fit_norm)
        else:
            dataset_results = [result for result in ok_results if result.dataset == dataset_name]
            scatter = _plot_raw(ax, dataset_results, size_metric, fit_norm=fit_norm)
        if first_scatter is None:
            first_scatter = scatter

        ax.set_title(dataset_name)
        ax.set_xlabel(size_metric)
        ax.set_ylabel("f1_macro")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(dataset_names):]:
        ax.set_visible(False)

    if first_scatter is None:
        raise ValueError("No plottable benchmark results available")

    cbar = fig.colorbar(first_scatter, ax=used_axes)
    cbar.set_label("fit_seconds")
    fig.subplots_adjust(top=0.88, wspace=0.28, hspace=0.35)

    base = Path(output_base)
    png_path = base.with_suffix(".png")
    pdf_path = base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def plot_benchmark_heatmap(
    results: Iterable[BenchmarkResult],
    output_base: str | Path,
    error_bar: str = "std",
) -> tuple[Path, Path]:
    """Create a heatmap for aggregated benchmark results.

    rows: datasets
    columns: estimators
    color: mean F1-macro
    annotation: `f1±err` and mean fit time
    """
    raw_results = list(results)
    ok_results = [result for result in raw_results if result.status == "ok"]
    if not ok_results:
        raise ValueError("No successful benchmark results available for heatmap plotting")

    aggregated = aggregate_benchmark_results(ok_results, error_bar=error_bar)
    if not aggregated:
        raise ValueError("No aggregated benchmark results available for heatmap plotting")

    dataset_names = sorted({result.dataset for result in aggregated})
    estimator_names = _sort_estimators_for_heatmap(aggregated)
    matrix = np.full((len(dataset_names), len(estimator_names)), np.nan)
    annotations = np.full((len(dataset_names), len(estimator_names)), "", dtype=object)

    dataset_index = {name: idx for idx, name in enumerate(dataset_names)}
    estimator_index = {name: idx for idx, name in enumerate(estimator_names)}

    for result in aggregated:
        row = dataset_index[result.dataset]
        col = estimator_index[result.estimator]
        matrix[row, col] = float(result.f1_macro_mean or np.nan)
        annotations[row, col] = (
            f"{_fmt_metric(result.f1_macro_mean)}±{_fmt_metric(result.f1_macro_error)}\n"
            f"fit {_fmt_metric(result.fit_seconds_mean)}s"
        )

    fig_width = max(7.5, 1.0 + 1.2 * len(estimator_names))
    fig_height = max(4.5, 1.2 + 0.8 * len(dataset_names))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#e5e7eb")
    im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_title("Benchmark comparison (heatmap): mean F1-macro per dataset/estimator")
    ax.set_xticks(range(len(estimator_names)), estimator_names, rotation=45, ha="right")
    ax.set_yticks(range(len(dataset_names)), dataset_names)
    ax.set_xlabel("Estimator")
    ax.set_ylabel("Dataset")

    for row_idx in range(len(dataset_names)):
        for col_idx in range(len(estimator_names)):
            if np.isnan(matrix[row_idx, col_idx]):
                ax.text(col_idx, row_idx, "n/a", ha="center", va="center", fontsize=8, color="black")
                continue
            text_color = "white" if matrix[row_idx, col_idx] < 0.55 else "black"
            ax.text(
                col_idx,
                row_idx,
                annotations[row_idx, col_idx],
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("f1_macro_mean")
    fig.subplots_adjust(left=0.18, right=0.95, bottom=0.28, top=0.90)

    base = Path(output_base)
    png_path = base.with_suffix(".png")
    pdf_path = base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def _plot_raw(
    ax,
    ok_results: list[BenchmarkResult],
    size_metric: str,
    *,
    fit_norm: Normalize,
):
    x_values = [getattr(result, size_metric) for result in ok_results]
    y_values = [result.f1_macro for result in ok_results]
    fit_values = [result.fit_seconds for result in ok_results]
    labels = [f"{result.estimator}#{result.repeat}" for result in ok_results]
    scatter = ax.scatter(
        x_values,
        y_values,
        c=fit_values,
        cmap="viridis",
        norm=fit_norm,
        s=90,
        edgecolors="black",
    )
    for x, y, label in zip(x_values, y_values, labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    return scatter


def _plot_aggregated(
    ax,
    aggregated: list[AggregatedBenchmarkResult],
    size_metric: str,
    *,
    fit_norm: Normalize,
):
    metric_name = f"{size_metric}_mean"
    metric_err = f"{size_metric}_error"
    x_values = [getattr(result, metric_name) for result in aggregated]
    x_errors = [getattr(result, metric_err) for result in aggregated]
    y_values = [result.f1_macro_mean for result in aggregated]
    y_errors = [result.f1_macro_error for result in aggregated]
    fit_values = [result.fit_seconds_mean for result in aggregated]
    labels = [result.estimator for result in aggregated]

    scatter = ax.scatter(
        x_values,
        y_values,
        c=fit_values,
        cmap="viridis",
        norm=fit_norm,
        s=110,
        edgecolors="black",
    )
    for x, y, xerr, yerr, label in zip(x_values, y_values, x_errors, y_errors, labels):
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor="gray", alpha=0.7, capsize=4)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    return scatter


def _build_dataset_axes(n_datasets: int):
    if n_datasets <= 0:
        raise ValueError("Expected at least one dataset for plotting")
    ncols = 1 if n_datasets == 1 else 2
    nrows = math.ceil(n_datasets / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(7 * ncols, 4.8 * nrows),
        squeeze=False,
    )
    return fig, list(axes.flatten())


def _build_fit_norm(fit_values: list[float]) -> Normalize:
    if not fit_values:
        return Normalize(vmin=0.0, vmax=1.0)
    vmin = float(min(fit_values))
    vmax = float(max(fit_values))
    if math.isclose(vmin, vmax):
        vmax = vmin + 1e-9
    return Normalize(vmin=vmin, vmax=vmax)


def _sort_estimators_for_heatmap(
    aggregated: list[AggregatedBenchmarkResult],
) -> list[str]:
    estimator_scores: dict[str, list[float]] = {}
    for result in aggregated:
        if result.f1_macro_mean is None:
            continue
        estimator_scores.setdefault(result.estimator, []).append(float(result.f1_macro_mean))

    return sorted(
        estimator_scores,
        key=lambda name: (-float(np.mean(estimator_scores[name])), name),
    )


def _fmt_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


