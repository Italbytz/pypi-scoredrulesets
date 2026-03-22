from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, LogNorm
from matplotlib.patches import Rectangle
from matplotlib import cm

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


# ---------------------------------------------------------------------------
# Combined multi-metric heatmap
# ---------------------------------------------------------------------------


def plot_benchmark_heatmap_combined(
    results: Iterable[BenchmarkResult],
    output_base: str | Path,
    error_bar: str = "std",
) -> tuple[Path, Path]:
    """Create a combined heatmap with three metrics per cell.

    Each cell is split into three rectangles:
      - **Top half**: F1-macro (green colour scale)
      - **Bottom-left quarter**: total atom count (blue colour scale, log)
      - **Bottom-right quarter**: fit time in seconds (orange colour scale, log)

    Each sub-rectangle is annotated with its numeric value.

    Returns (png_path, pdf_path).
    """
    raw_results = list(results)
    ok_results = [r for r in raw_results if r.status == "ok"]
    if not ok_results:
        raise ValueError("No successful benchmark results for combined heatmap")

    aggregated = aggregate_benchmark_results(ok_results, error_bar=error_bar)
    if not aggregated:
        raise ValueError("No aggregated results for combined heatmap")

    dataset_names = sorted({r.dataset for r in aggregated})
    estimator_names = _sort_estimators_for_heatmap(aggregated)

    n_rows = len(dataset_names)
    n_cols = len(estimator_names)

    # Build lookup  ─────────────────────────────────────────────────────────
    lookup: dict[tuple[str, str], AggregatedBenchmarkResult] = {}
    for r in aggregated:
        lookup[(r.dataset, r.estimator)] = r

    # Collect raw metric arrays for normalisation  ──────────────────────────
    f1_vals: list[float] = []
    atom_vals: list[float] = []
    fit_vals: list[float] = []
    for r in aggregated:
        if r.f1_macro_mean is not None:
            f1_vals.append(float(r.f1_macro_mean))
        if r.n_atoms_mean is not None and r.n_atoms_mean > 0:
            atom_vals.append(float(r.n_atoms_mean))
        if r.fit_seconds_mean is not None and r.fit_seconds_mean > 0:
            fit_vals.append(float(r.fit_seconds_mean))

    # Colour maps & norms  ─────────────────────────────────────────────────
    f1_cmap = plt.colormaps["Greens"]
    f1_norm = Normalize(vmin=0.0, vmax=1.0)

    atom_cmap = plt.colormaps["Blues"]
    if atom_vals:
        atom_norm = LogNorm(
            vmin=max(1.0, min(atom_vals)),
            vmax=max(max(atom_vals), 2.0),
        )
    else:
        atom_norm = Normalize(vmin=1, vmax=100)

    fit_cmap = plt.colormaps["Oranges"]
    if fit_vals:
        fit_norm = LogNorm(
            vmin=max(0.001, min(fit_vals)),
            vmax=max(max(fit_vals), 0.002),
        )
    else:
        fit_norm = Normalize(vmin=0.001, vmax=10)

    # Figure layout  ───────────────────────────────────────────────────────
    cell_w = max(1.4, min(2.0, 26.0 / max(n_cols, 1)))
    cell_h = max(0.9, min(1.3, 18.0 / max(n_rows, 1)))
    fig_w = 2.0 + cell_w * n_cols + 3.0        # left margin + cells + legend
    fig_h = 1.6 + cell_h * n_rows + 1.2        # top/bottom margins
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)          # row 0 at top
    ax.set_aspect("equal")
    ax.set_frame_on(False)
    ax.tick_params(length=0)

    # Axis labels  ─────────────────────────────────────────────────────────
    ax.set_xticks([c + 0.5 for c in range(n_cols)])
    ax.set_xticklabels(estimator_names, rotation=55, ha="right", fontsize=8)
    ax.set_yticks([r + 0.5 for r in range(n_rows)])
    ax.set_yticklabels(dataset_names, fontsize=9)
    ax.set_xlabel("Estimator", fontsize=10)
    ax.set_ylabel("Dataset", fontsize=10)
    ax.set_title(
        "Combined benchmark heatmap: F1 (top), atoms (bottom-left), fit time (bottom-right)",
        fontsize=11,
        pad=12,
    )

    # Draw cells  ──────────────────────────────────────────────────────────
    for ri, ds in enumerate(dataset_names):
        for ci, est in enumerate(estimator_names):
            entry = lookup.get((ds, est))
            _draw_combined_cell(
                ax, ci, ri, entry,
                f1_cmap, f1_norm,
                atom_cmap, atom_norm,
                fit_cmap, fit_norm,
            )

    # Grid lines  ──────────────────────────────────────────────────────────
    for r in range(n_rows + 1):
        ax.axhline(r, color="white", linewidth=1.2)
    for c in range(n_cols + 1):
        ax.axvline(c, color="white", linewidth=1.2)

    # Colour-bar legends (right side)  ─────────────────────────────────────
    _add_legend_bar(fig, f1_cmap, f1_norm, label="F1-macro", position=[0.92, 0.55, 0.015, 0.30])
    _add_legend_bar(fig, atom_cmap, atom_norm, label="Atoms", position=[0.92, 0.15, 0.015, 0.30])
    _add_legend_bar(fig, fit_cmap, fit_norm, label="Fit (s)", position=[0.96, 0.15, 0.015, 0.30])

    fig.subplots_adjust(left=0.15, right=0.90, bottom=0.22, top=0.92)

    base = Path(output_base)
    png_path = base.with_suffix(".png")
    pdf_path = base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def _draw_combined_cell(
    ax,
    col: int,
    row: int,
    entry: AggregatedBenchmarkResult | None,
    f1_cmap, f1_norm,
    atom_cmap, atom_norm,
    fit_cmap, fit_norm,
):
    """Draw one cell with three coloured sub-rectangles + text annotations."""
    x0, y0 = float(col), float(row)

    if entry is None:
        # Grey "n/a" cell
        ax.add_patch(Rectangle((x0, y0), 1.0, 1.0, facecolor="#e5e7eb", edgecolor="none"))
        ax.text(x0 + 0.5, y0 + 0.5, "n/a", ha="center", va="center", fontsize=7, color="#888")
        return

    # ── Top half: F1 ──────────────────────────────────────────────────────
    f1_val = entry.f1_macro_mean
    if f1_val is not None:
        f1_color = f1_cmap(f1_norm(float(f1_val)))
    else:
        f1_color = "#e5e7eb"
    ax.add_patch(Rectangle((x0, y0), 1.0, 0.5, facecolor=f1_color, edgecolor="none"))

    f1_text = f"{f1_val:.2f}" if f1_val is not None else "n/a"
    f1_err = entry.f1_macro_error
    if f1_err is not None and f1_val is not None:
        f1_text += f"\n±{f1_err:.2f}"
    text_lum = _luminance(f1_color) if f1_val is not None else 0.7
    ax.text(
        x0 + 0.5, y0 + 0.25, f1_text,
        ha="center", va="center",
        fontsize=7, fontweight="bold",
        color="white" if text_lum < 0.5 else "black",
    )

    # ── Bottom-left quarter: Atoms ────────────────────────────────────────
    atom_val = entry.n_atoms_mean
    if atom_val is not None and atom_val > 0:
        atom_color = atom_cmap(atom_norm(float(atom_val)))
    else:
        atom_color = "#e5e7eb"
    ax.add_patch(Rectangle((x0, y0 + 0.5), 0.5, 0.5, facecolor=atom_color, edgecolor="none"))

    atom_text = _fmt_compact(atom_val)
    atom_lum = _luminance(atom_color) if (atom_val is not None and atom_val > 0) else 0.7
    ax.text(
        x0 + 0.25, y0 + 0.75, atom_text,
        ha="center", va="center",
        fontsize=6,
        color="white" if atom_lum < 0.5 else "black",
    )

    # ── Bottom-right quarter: Fit time ────────────────────────────────────
    fit_val = entry.fit_seconds_mean
    if fit_val is not None and fit_val > 0:
        fit_color = fit_cmap(fit_norm(float(fit_val)))
    else:
        fit_color = "#e5e7eb"
    ax.add_patch(Rectangle((x0 + 0.5, y0 + 0.5), 0.5, 0.5, facecolor=fit_color, edgecolor="none"))

    fit_text = _fmt_duration_short(fit_val)
    fit_lum = _luminance(fit_color) if (fit_val is not None and fit_val > 0) else 0.7
    ax.text(
        x0 + 0.75, y0 + 0.75, fit_text,
        ha="center", va="center",
        fontsize=6,
        color="white" if fit_lum < 0.5 else "black",
    )


def _add_legend_bar(fig, cmap, norm, *, label: str, position: list[float]):
    """Add a small colour-bar to the figure at `position = [left, bottom, width, height]`."""
    cax = fig.add_axes(position)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)


def _fmt_compact(value: float | None) -> str:
    """Format a number compactly (e.g. 1234 → '1.2k')."""
    if value is None:
        return "n/a"
    v = float(value)
    if v >= 10_000:
        return f"{v / 1000:.0f}k"
    if v >= 1_000:
        return f"{v / 1000:.1f}k"
    if v >= 100:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def _fmt_duration_short(seconds: float | None) -> str:
    """Format seconds compactly (e.g. 0.03 → '30ms', 5.2 → '5.2s')."""
    if seconds is None:
        return "n/a"
    s = float(seconds)
    if s < 0.01:
        return f"{s * 1000:.0f}ms"
    if s < 1.0:
        return f"{s * 1000:.0f}ms"
    if s < 60:
        return f"{s:.1f}s"
    m = s / 60
    return f"{m:.1f}m"


def _luminance(color) -> float:
    """Approximate perceived luminance of an RGBA tuple or hex colour."""
    import matplotlib.colors as mcolors
    try:
        rgba = mcolors.to_rgba(color)
    except Exception:
        return 0.5
    # Rec. 709 luminance
    return 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
