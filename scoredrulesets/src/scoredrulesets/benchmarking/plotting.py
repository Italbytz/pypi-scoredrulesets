from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib import cm

from .runner import (
    AggregatedBenchmarkResult,
    BenchmarkResult,
    aggregate_benchmark_results,
    build_pareto_per_dataset,
)


_PAPER_ESTIMATOR_DISPLAY_ORDER: tuple[str, ...] = (
    "HS",
    "RuleKit",
    "ExSTraCS",
    "ExSTraCS (LRC)",
    "logicGP",
    "ruleGP",
    "ruleNLN",
    "ruleLCS",
)


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
    color: estimator
    legend: shared across subplots
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
        estimator_names = _estimator_display_order({result.estimator for result in aggregated_results})
        title = (
            f"Benchmark (aggregated, per dataset): F1 vs model size "
            f"(color = estimator, error = {error_bar})"
        )
    else:
        aggregated_results = None
        dataset_names = sorted({result.dataset for result in ok_results})
        estimator_names = _estimator_display_order({result.estimator for result in ok_results})
        title = "Benchmark (per dataset): F1 vs model size (color = estimator)"

    estimator_colors = _estimator_color_map(estimator_names)

    fig, axes = _build_dataset_axes(len(dataset_names))
    fig.suptitle(title)

    for ax, dataset_name in zip(axes, dataset_names):
        if aggregate_repeats:
            dataset_results = [
                result for result in aggregated_results or [] if result.dataset == dataset_name
            ]
            _plot_aggregated(ax, dataset_results, size_metric, estimator_colors=estimator_colors)
        else:
            dataset_results = [result for result in ok_results if result.dataset == dataset_name]
            _plot_raw(ax, dataset_results, size_metric, estimator_colors=estimator_colors)

        ax.set_title(dataset_name)
        ax.set_xlabel(size_metric)
        ax.set_ylabel("f1_macro")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(dataset_names):]:
        ax.set_visible(False)

    if not estimator_names:
        raise ValueError("No plottable benchmark results available")

    legend_handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=estimator_colors[est],
            markeredgecolor="black",
            markeredgewidth=0.7,
            markersize=7,
            label=est,
        )
        for est in estimator_names
    ]
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.86, 0.5),
        frameon=False,
        title="Estimator",
        fontsize=8,
        title_fontsize=9,
    )
    fig.subplots_adjust(top=0.88, right=0.83, wspace=0.28, hspace=0.35)

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

    cmap = plt.get_cmap("plasma").copy()
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
    estimator_colors: dict[str, tuple[float, float, float, float]],
):
    if not ok_results:
        return

    by_estimator: dict[str, list[BenchmarkResult]] = {}
    for result in ok_results:
        by_estimator.setdefault(result.estimator, []).append(result)

    for estimator, est_results in by_estimator.items():
        x_values = [getattr(result, size_metric) for result in est_results]
        y_values = [result.f1_macro for result in est_results]
        ax.scatter(
            x_values,
            y_values,
            c=[estimator_colors.get(estimator, "#4b5563")],
            s=70,
            edgecolors="black",
            linewidths=0.7,
            alpha=0.9,
        )


def _plot_aggregated(
    ax,
    aggregated: list[AggregatedBenchmarkResult],
    size_metric: str,
    *,
    estimator_colors: dict[str, tuple[float, float, float, float]],
):
    metric_name = f"{size_metric}_mean"
    metric_err = f"{size_metric}_error"
    if not aggregated:
        return

    for result in aggregated:
        x = getattr(result, metric_name)
        xerr = getattr(result, metric_err)
        y = result.f1_macro_mean
        yerr = result.f1_macro_error
        if x is None or y is None:
            continue

        color = estimator_colors.get(result.estimator, "#4b5563")
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            capsize=4,
            markersize=6,
            linewidth=1.0,
            color=color,
            markeredgecolor="black",
            markeredgewidth=0.7,
            ecolor=color,
            alpha=0.9,
        )


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

    ordered_names = _estimator_display_order(estimator_scores.keys())
    return sorted(
        ordered_names,
        key=lambda name: (_estimator_order_rank(name), -float(np.mean(estimator_scores[name])), name),
    )


def _estimator_order_rank(name: str) -> int:
    try:
        return _PAPER_ESTIMATOR_DISPLAY_ORDER.index(name)
    except ValueError:
        return len(_PAPER_ESTIMATOR_DISPLAY_ORDER)


def _estimator_display_order(names: Iterable[str]) -> list[str]:
    names_set = set(names)
    ordered: list[str] = [n for n in _PAPER_ESTIMATOR_DISPLAY_ORDER if n in names_set]
    ordered.extend(sorted(n for n in names_set if n not in ordered))
    return ordered


def _estimator_color_map(estimator_names: Iterable[str]) -> dict[str, tuple[float, float, float, float]]:
    # Fixed paper palette (colorblind-friendly) + stable fallback colors.
    paper_palette = {
        "HS": "#0072B2",              # blue
        "RuleKit": "#E69F00",         # orange
        "ExSTraCS": "#009E73",        # green
        "ExSTraCS (LRC)": "#D55E00",  # vermillion
        "logicGP": "#56B4E9",         # sky blue
        "ruleGP": "#CC79A7",          # purple
        "ruleNLN": "#F0E442",         # yellow
        "ruleLCS": "#7F7F7F",         # gray
    }
    fallback_palette = [
        "#332288",  # dark blue
        "#88CCEE",  # light blue
        "#44AA99",  # teal
        "#117733",  # green
        "#999933",  # olive
        "#DDCC77",  # sand
        "#CC6677",  # rose
        "#882255",  # wine
        "#AA4499",  # magenta
        "#661100",  # brown
    ]
    ordered = _estimator_display_order(estimator_names)
    color_map: dict[str, tuple[float, float, float, float]] = {}
    fallback_idx = 0
    for est in ordered:
        if est in paper_palette:
            color_map[est] = matplotlib.colors.to_rgba(paper_palette[est])
        else:
            color_map[est] = matplotlib.colors.to_rgba(
                fallback_palette[fallback_idx % len(fallback_palette)]
            )
            fallback_idx += 1
    return color_map


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
      - **Top half**: F1-macro
      - **Bottom-left quarter**: total atom count
      - **Bottom-right quarter**: fit time in seconds

    All three sub-rectangles share the same colour scale. Darker colour means
    better relative performance for the respective metric; for size and fit time
    the scale is therefore inverted so that smaller values appear better.

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

    # Colour map & norms  ──────────────────────────────────────────────────
    combined_cmap = plt.colormaps["plasma"]
    f1_norm = Normalize(vmin=0.0, vmax=1.0)

    atom_norm = _build_positive_log_norm(atom_vals, min_floor=1.0, default_max=100.0)

    fit_norm = _build_positive_log_norm(fit_vals, min_floor=0.001, default_max=10.0)
    legend_norm = Normalize(vmin=0.0, vmax=1.0)

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
                combined_cmap,
                f1_norm,
                atom_norm,
                fit_norm,
            )

    # Grid lines  ──────────────────────────────────────────────────────────
    for r in range(n_rows + 1):
        ax.axhline(r, color="white", linewidth=1.2)
    for c in range(n_cols + 1):
        ax.axvline(c, color="white", linewidth=1.2)

    # Shared colour-bar legend (right side)  ───────────────────────────────
    _add_legend_bar(
        fig,
        combined_cmap,
        legend_norm,
        position=[0.92, 0.20, 0.018, 0.56],
        label="Relative performance",
        ticks=[0.0, 1.0],
        ticklabels=["worse", "better"],
        show_ticks=True,
    )

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
    cmap,
    f1_norm,
    atom_norm,
    fit_norm,
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
        f1_score = _metric_score(float(f1_val), f1_norm)
        f1_color = cmap(f1_score)
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
        atom_score = _metric_score(float(atom_val), atom_norm, invert=True)
        atom_color = cmap(atom_score)
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
        fit_score = _metric_score(float(fit_val), fit_norm, invert=True)
        fit_color = cmap(fit_score)
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


def _add_legend_bar(
    fig,
    cmap,
    norm,
    *,
    position: list[float],
    label: str | None = None,
    ticks: list[float] | None = None,
    ticklabels: list[str] | None = None,
    show_ticks: bool = True,
):
    """Add a small colour-bar to the figure at `position = [left, bottom, width, height]`."""
    cax = fig.add_axes(position)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    if label:
        cb.set_label(label, fontsize=8)
    if ticks is not None:
        cb.set_ticks(ticks)
    if ticklabels is not None:
        cb.set_ticklabels(ticklabels)
    if show_ticks:
        cb.ax.tick_params(labelsize=7)
    else:
        cb.set_ticks([])
        cb.ax.tick_params(length=0)


def _build_positive_log_norm(values: list[float], *, min_floor: float, default_max: float) -> LogNorm:
    """Build a positive log norm with a safe non-zero span."""
    if not values:
        return LogNorm(vmin=min_floor, vmax=default_max)

    vmin = max(min_floor, min(values))
    vmax = max(max(values), min_floor * 2.0)
    if vmax <= vmin:
        vmax = vmin * 1.01
    return LogNorm(vmin=vmin, vmax=vmax)


def _metric_score(value: float, norm, *, invert: bool = False) -> float:
    """Map a metric value to a shared 0..1 colour score."""
    score = float(norm(value))
    if not math.isfinite(score):
        score = 0.0
    score = min(1.0, max(0.0, score))
    return 1.0 - score if invert else score


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


# ---------------------------------------------------------------------------
# Combined dot plot (F1 left, model size right) – inspired by imodels
# ---------------------------------------------------------------------------


def plot_combined_dot(
    results: Iterable[BenchmarkResult],
    output_base: str | Path,
    error_bar: str = "std",
    size_metric: str = "n_atoms",
) -> tuple[Path, Path]:
    """Create a combined two-panel dot plot (F1 left, model size right).

    Y-axis: datasets, each estimator shown as a coloured dot.
    Left panel: F1-macro (0–1).
    Right panel: model size on symlog scale.
    Legend on the far right.

    Returns *(png_path, pdf_path)*.
    """
    raw_results = list(results)
    ok_results = [r for r in raw_results if r.status == "ok"]
    if not ok_results:
        raise ValueError("No successful benchmark results for combined dot plot")

    aggregated = aggregate_benchmark_results(ok_results, error_bar=error_bar)
    if not aggregated:
        raise ValueError("No aggregated results for combined dot plot")

    dataset_names = sorted({r.dataset for r in aggregated})
    estimator_names = _sort_estimators_for_heatmap(aggregated)

    n_datasets = len(dataset_names)
    n_estimators = len(estimator_names)

    # Build lookup
    lookup: dict[tuple[str, str], AggregatedBenchmarkResult] = {}
    for r in aggregated:
        lookup[(r.dataset, r.estimator)] = r

    # Figure layout
    fig_width = max(10.0, 6.0 + 0.4 * n_estimators)
    fig_height = max(4.0, 1.2 * n_datasets)
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.35])
    ax_f1 = fig.add_subplot(gs[0, 0])
    ax_size = fig.add_subplot(gs[0, 1], sharey=ax_f1)
    ax_legend = fig.add_subplot(gs[0, 2])

    y_base = np.arange(n_datasets, dtype=float)
    offsets = (
        np.array([0.0])
        if n_estimators <= 1
        else np.linspace(-0.3, 0.3, num=n_estimators)
    )

    estimator_colors = _estimator_color_map(estimator_names)

    size_mean_col = f"{size_metric}_mean"
    size_err_col = f"{size_metric}_error"

    for idx, est in enumerate(estimator_names):
        color = estimator_colors.get(est, "#4b5563")
        f1_vals = np.full(n_datasets, np.nan)
        f1_errs = np.full(n_datasets, 0.0)
        size_vals = np.full(n_datasets, np.nan)
        size_errs = np.full(n_datasets, 0.0)

        for di, ds in enumerate(dataset_names):
            entry = lookup.get((ds, est))
            if entry is None:
                continue
            if entry.f1_macro_mean is not None:
                f1_vals[di] = entry.f1_macro_mean
            if entry.f1_macro_error is not None:
                f1_errs[di] = entry.f1_macro_error
            sz = getattr(entry, size_mean_col, None)
            if sz is not None:
                size_vals[di] = sz
            se = getattr(entry, size_err_col, None)
            if se is not None:
                size_errs[di] = se

        y = y_base + offsets[idx]
        mask_f1 = ~np.isnan(f1_vals)
        mask_sz = ~np.isnan(size_vals)

        if mask_f1.any():
            ax_f1.errorbar(
                f1_vals[mask_f1], y[mask_f1],
                xerr=f1_errs[mask_f1],
                fmt="o", capsize=3, markersize=5, linewidth=1,
                color=color, label=est,
            )
        if mask_sz.any():
            ax_size.errorbar(
                size_vals[mask_sz], y[mask_sz],
                xerr=size_errs[mask_sz],
                fmt="o", capsize=3, markersize=5, linewidth=1,
                color=color, label=est,
            )

    # Axis config
    ax_f1.set_yticks(y_base)
    ax_f1.set_yticklabels(dataset_names, fontsize=9)
    ax_f1.invert_yaxis()
    ax_f1.set_xlim(-0.02, 1.02)
    ax_f1.set_xlabel("F1-macro (mean ± std)", fontsize=10)
    ax_f1.set_ylabel("Dataset", fontsize=10)
    ax_f1.grid(axis="x", alpha=0.3)
    ax_f1.set_axisbelow(True)

    ax_size.set_xscale("symlog", linthresh=1.0, linscale=1.0)
    ax_size.set_xlabel(f"{size_metric} (mean ± std)", fontsize=10)
    ax_size.grid(axis="x", alpha=0.3)
    ax_size.set_axisbelow(True)
    plt.setp(ax_size.get_yticklabels(), visible=False)

    # Background bands
    for di in range(n_datasets):
        if di % 2 == 1:
            for ax in (ax_f1, ax_size):
                ax.axhspan(di - 0.5, di + 0.5, facecolor="0.96", edgecolor="none", zorder=-1)

    # Legend
    handles, labels = ax_f1.get_legend_handles_labels()
    ax_legend.axis("off")
    if handles:
        ax_legend.legend(
            handles, labels,
            loc="center left",
            ncol=1,
            frameon=False,
            title="Estimator",
            fontsize=9,
            title_fontsize=10,
        )

    fig.suptitle("Combined benchmark: F1 (left) and model size (right)", fontsize=12, y=1.01)

    base = Path(output_base)
    png_path = base.with_suffix(".png")
    pdf_path = base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


# ---------------------------------------------------------------------------
# Pareto front scatter plot (F1 vs model size)
# ---------------------------------------------------------------------------


def plot_pareto_front(
    results: Iterable[BenchmarkResult],
    output_base: str | Path,
    error_bar: str = "std",
    size_metric: str = "n_atoms",
) -> tuple[Path, Path]:
    """Create a scatter plot highlighting the Pareto front per dataset.

    x-axis: model size  (n_atoms by default)
    y-axis: F1-macro
    Pareto-optimal estimators are highlighted with a larger marker and
    connected by a dashed step line.

    Returns *(png_path, pdf_path)*.
    """
    raw_results = list(results)
    ok_results = [r for r in raw_results if r.status == "ok"]
    if not ok_results:
        raise ValueError("No successful benchmark results for Pareto front plot")

    aggregated = aggregate_benchmark_results(ok_results, error_bar=error_bar)
    if not aggregated:
        raise ValueError("No aggregated results for Pareto front plot")

    pareto_per_ds = build_pareto_per_dataset(
        aggregated,
        quality_attr="f1_macro_mean",
        size_attr=f"{size_metric}_mean",
    )

    dataset_names = sorted({r.dataset for r in aggregated})
    fig, axes = _build_dataset_axes(len(dataset_names))
    fig.suptitle("Pareto front: F1 vs model size per dataset", fontsize=13)

    used_axes = []
    for ax, ds_name in zip(axes, dataset_names):
        used_axes.append(ax)
        ds_results = [r for r in aggregated if r.dataset == ds_name]
        pareto_set = {r.estimator for r in pareto_per_ds.get(ds_name, [])}

        size_col = f"{size_metric}_mean"
        size_err_col = f"{size_metric}_error"

        x_all, y_all, labels_all = [], [], []
        x_pareto, y_pareto, labels_pareto = [], [], []

        for r in ds_results:
            sz = getattr(r, size_col)
            f1 = r.f1_macro_mean
            if sz is None or f1 is None:
                continue
            x_all.append(float(sz))
            y_all.append(float(f1))
            labels_all.append(r.estimator)
            if r.estimator in pareto_set:
                x_pareto.append(float(sz))
                y_pareto.append(float(f1))
                labels_pareto.append(r.estimator)

        # All estimators (grey)
        ax.scatter(x_all, y_all, s=50, c="lightgray", edgecolors="gray", zorder=2)
        for xv, yv, lbl in zip(x_all, y_all, labels_all):
            ax.annotate(lbl, (xv, yv), textcoords="offset points", xytext=(4, 4), fontsize=6, color="gray")

        # Pareto-optimal (red, larger)
        ax.scatter(x_pareto, y_pareto, s=120, c="tomato", edgecolors="darkred", zorder=3, label="Pareto-optimal")
        for xv, yv, lbl in zip(x_pareto, y_pareto, labels_pareto):
            ax.annotate(
                lbl, (xv, yv), textcoords="offset points", xytext=(5, 5),
                fontsize=7, fontweight="bold", color="darkred",
            )

        # Step line connecting Pareto front
        if len(x_pareto) > 1:
            order = sorted(range(len(x_pareto)), key=lambda i: x_pareto[i])
            ax.step(
                [x_pareto[i] for i in order],
                [y_pareto[i] for i in order],
                where="post",
                linestyle="--",
                color="tomato",
                linewidth=1.2,
                alpha=0.7,
                zorder=1,
            )

        ax.set_title(ds_name, fontsize=10)
        ax.set_xlabel(size_metric)
        ax.set_ylabel("F1-macro")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        if x_all:
            ax.set_xscale("symlog", linthresh=1.0, linscale=1.0)

    for ax in axes[len(dataset_names):]:
        ax.set_visible(False)

    fig.subplots_adjust(top=0.88, wspace=0.30, hspace=0.40)

    base = Path(output_base)
    png_path = base.with_suffix(".png")
    pdf_path = base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path

