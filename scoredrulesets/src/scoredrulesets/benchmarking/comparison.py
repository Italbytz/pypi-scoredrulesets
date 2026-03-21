from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .reporting import (
    _html_document,
    _html_kv_list,
    _html_section,
    _html_table,
    format_benchmark_leaderboard_html,
    format_benchmark_leaderboard_markdown,
)
from .runner import AggregatedBenchmarkResult, build_benchmark_leaderboard


@dataclass
class BenchmarkComparisonRow:
    dataset: str
    core_best_estimator: str | None
    core_best_f1: float | None
    compare_best_estimator: str | None
    compare_best_f1: float | None
    compare_minus_core_best_f1: float | None
    focus_estimator: str
    focus_f1: float | None
    baseline_estimator: str
    baseline_f1: float | None
    focus_minus_baseline_f1: float | None
    focus_minus_baseline_fit_seconds: float | None
    focus_minus_baseline_rules: float | None


def load_aggregated_results_json(path: str | Path) -> list[AggregatedBenchmarkResult]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [AggregatedBenchmarkResult(**row) for row in payload]


def compare_benchmark_runs(
    core_results: list[AggregatedBenchmarkResult],
    compare_results: list[AggregatedBenchmarkResult],
    *,
    focus_estimator: str = "wrapper_hs",
    baseline_estimator: str = "wrapper_cart",
) -> list[BenchmarkComparisonRow]:
    core_grouped = _group_by_dataset(core_results)
    compare_grouped = _group_by_dataset(compare_results)
    datasets = sorted(set(core_grouped.keys()) & set(compare_grouped.keys()))

    rows: list[BenchmarkComparisonRow] = []
    for dataset in datasets:
        core_best = _best_result(core_grouped[dataset])
        compare_best = _best_result(compare_grouped[dataset])
        focus = _find_estimator(compare_grouped[dataset], focus_estimator)
        baseline = _find_estimator(compare_grouped[dataset], baseline_estimator)

        rows.append(
            BenchmarkComparisonRow(
                dataset=dataset,
                core_best_estimator=core_best.estimator if core_best else None,
                core_best_f1=core_best.f1_macro_mean if core_best else None,
                compare_best_estimator=compare_best.estimator if compare_best else None,
                compare_best_f1=compare_best.f1_macro_mean if compare_best else None,
                compare_minus_core_best_f1=_delta(
                    compare_best.f1_macro_mean if compare_best else None,
                    core_best.f1_macro_mean if core_best else None,
                ),
                focus_estimator=focus_estimator,
                focus_f1=focus.f1_macro_mean if focus else None,
                baseline_estimator=baseline_estimator,
                baseline_f1=baseline.f1_macro_mean if baseline else None,
                focus_minus_baseline_f1=_delta(
                    focus.f1_macro_mean if focus else None,
                    baseline.f1_macro_mean if baseline else None,
                ),
                focus_minus_baseline_fit_seconds=_delta(
                    focus.fit_seconds_mean if focus else None,
                    baseline.fit_seconds_mean if baseline else None,
                ),
                focus_minus_baseline_rules=_delta(
                    focus.n_rules_mean if focus else None,
                    baseline.n_rules_mean if baseline else None,
                ),
            )
        )
    return rows


def format_benchmark_comparison_report_markdown(
    rows: list[BenchmarkComparisonRow],
    *,
    title: str = "Core vs HS Benchmark Comparison",
    core_label: str = "core",
    compare_label: str = "hs_compare",
    core_artifacts: dict[str, str] | None = None,
    compare_artifacts: dict[str, str] | None = None,
) -> str:
    lines = [f"# {title}", ""]
    lines.extend(_comparison_summary(rows))

    lines.extend(["## Inputs", ""])
    lines.append(f"- **core_label**: `{core_label}`")
    lines.append(f"- **compare_label**: `{compare_label}`")
    if rows:
        lines.append(f"- **focus_estimator**: `{rows[0].focus_estimator}`")
        lines.append(f"- **baseline_estimator**: `{rows[0].baseline_estimator}`")
    if core_artifacts:
        lines.append(f"- **core_artifacts**: `{core_artifacts}`")
    if compare_artifacts:
        lines.append(f"- **compare_artifacts**: `{compare_artifacts}`")
    lines.append("")

    lines.extend(["## Dataset Comparison", ""])
    lines.extend(_comparison_table_markdown(rows))
    lines.append("")

    lines.extend(["## Interpretation", ""])
    lines.append("- `compare_minus_core_best_f1 > 0` bedeutet: der beste Lauf im Vergleichsset war besser als der beste Lauf im Core-Set.")
    lines.append("- `focus_minus_baseline_f1 > 0` bedeutet: der Fokus-Estimator (z. B. `wrapper_hs`) war auf diesem Datensatz besser als der Baseline-Estimator (z. B. `wrapper_cart`).")
    lines.append("- Positive `focus_minus_baseline_fit_seconds` bedeuten, dass der Fokus-Estimator langsamer war.")
    lines.append("- Positive `focus_minus_baseline_rules` bedeuten, dass der Fokus-Estimator mehr Regeln erzeugt hat.")
    lines.append("")
    return "\n".join(lines)


def format_benchmark_comparison_report_html(
    rows: list[BenchmarkComparisonRow],
    *,
    title: str = "Core vs HS Benchmark Comparison",
    core_label: str = "core",
    compare_label: str = "hs_compare",
    core_artifacts: dict[str, str] | None = None,
    compare_artifacts: dict[str, str] | None = None,
) -> str:
    sections = []
    sections.append(_html_section("Executive Summary", _html_kv_list(_comparison_summary_items(rows))))

    input_items = [("core_label", core_label), ("compare_label", compare_label)]
    if rows:
        input_items.extend(
            [
                ("focus_estimator", rows[0].focus_estimator),
                ("baseline_estimator", rows[0].baseline_estimator),
            ]
        )
    if core_artifacts:
        input_items.append(("core_artifacts", str(core_artifacts)))
    if compare_artifacts:
        input_items.append(("compare_artifacts", str(compare_artifacts)))
    sections.append(_html_section("Inputs", _html_kv_list(input_items)))
    sections.append(_html_section("Dataset Comparison", _comparison_table_html(rows)))
    sections.append(
        _html_section(
            "Interpretation",
            _html_kv_list(
                [
                    ("delta_best", "compare_minus_core_best_f1 > 0 => Vergleichsset besser als Core-Set"),
                    ("focus_vs_baseline", "focus_minus_baseline_f1 > 0 => Fokus-Estimator besser als Baseline"),
                    ("delta_fit_s", "positive Werte => Fokus-Estimator langsamer"),
                    ("delta_rules", "positive Werte => Fokus-Estimator groesser"),
                ]
            ),
        )
    )
    return _html_document(title, sections)


def format_benchmark_meta_report_markdown(
    core_results: list[AggregatedBenchmarkResult],
    compare_results: list[AggregatedBenchmarkResult],
    comparison_rows: list[BenchmarkComparisonRow],
    *,
    title: str = "Combined Benchmark Meta Report",
    core_label: str = "core",
    compare_label: str = "hs_compare",
    core_artifacts: dict[str, str] | None = None,
    compare_artifacts: dict[str, str] | None = None,
    comparison_artifacts: dict[str, str] | None = None,
) -> str:
    core_leaderboard = build_benchmark_leaderboard(core_results, primary_metric="f1_macro_mean")
    compare_leaderboard = build_benchmark_leaderboard(compare_results, primary_metric="f1_macro_mean")

    lines = [f"# {title}", ""]
    lines.extend(_meta_summary_section(core_leaderboard, compare_leaderboard, comparison_rows, core_label, compare_label))

    lines.extend(["## Inputs", ""])
    lines.append(f"- **core_label**: `{core_label}`")
    lines.append(f"- **compare_label**: `{compare_label}`")
    if comparison_rows:
        lines.append(f"- **focus_estimator**: `{comparison_rows[0].focus_estimator}`")
        lines.append(f"- **baseline_estimator**: `{comparison_rows[0].baseline_estimator}`")
    lines.append("")

    lines.extend(["## Artifacts", ""])
    if core_artifacts:
        lines.append(f"- **core**: `{core_artifacts}`")
    if compare_artifacts:
        lines.append(f"- **compare**: `{compare_artifacts}`")
    if comparison_artifacts:
        lines.append(f"- **comparison**: `{comparison_artifacts}`")
    lines.append("")

    if comparison_artifacts:
        lines.extend(["## Comparison Parameters", ""])
        for key, value in comparison_artifacts.items():
            lines.append(f"- **{key}**: `{value}`")
        lines.append("")

    lines.extend([f"## Snapshot: {core_label}", "", format_benchmark_leaderboard_markdown(core_leaderboard), ""])
    lines.extend([f"## Snapshot: {compare_label}", "", format_benchmark_leaderboard_markdown(compare_leaderboard), ""])
    lines.extend(["## Direct Comparison", ""])
    lines.extend(_comparison_table_markdown(comparison_rows))
    lines.append("")
    lines.extend(["## Per-Dataset Takeaways", ""])
    lines.extend(_meta_takeaways(comparison_rows))
    return "\n".join(lines)


def format_benchmark_meta_report_html(
    core_results: list[AggregatedBenchmarkResult],
    compare_results: list[AggregatedBenchmarkResult],
    comparison_rows: list[BenchmarkComparisonRow],
    *,
    title: str = "Combined Benchmark Meta Report",
    core_label: str = "core",
    compare_label: str = "hs_compare",
    core_artifacts: dict[str, str] | None = None,
    compare_artifacts: dict[str, str] | None = None,
    comparison_artifacts: dict[str, str] | None = None,
) -> str:
    core_leaderboard = build_benchmark_leaderboard(core_results, primary_metric="f1_macro_mean")
    compare_leaderboard = build_benchmark_leaderboard(compare_results, primary_metric="f1_macro_mean")

    sections = []
    sections.append(
        _html_section(
            "Executive Summary",
            _html_kv_list(_meta_summary_items(core_leaderboard, compare_leaderboard, comparison_rows, core_label, compare_label)),
        )
    )

    input_items = [("core_label", core_label), ("compare_label", compare_label)]
    if comparison_rows:
        input_items.extend(
            [
                ("focus_estimator", comparison_rows[0].focus_estimator),
                ("baseline_estimator", comparison_rows[0].baseline_estimator),
            ]
        )
    sections.append(_html_section("Inputs", _html_kv_list(input_items)))

    artifact_items = []
    if core_artifacts:
        artifact_items.append(("core", str(core_artifacts)))
    if compare_artifacts:
        artifact_items.append(("compare", str(compare_artifacts)))
    if comparison_artifacts:
        artifact_items.append(("comparison", str(comparison_artifacts)))
    if artifact_items:
        sections.append(_html_section("Artifacts", _html_kv_list(artifact_items)))
        sections.append(_html_section("Comparison Parameters", _html_kv_list([(k, str(v)) for k, v in (comparison_artifacts or {}).items()])))

    sections.append(_html_section(f"Snapshot: {core_label}", format_benchmark_leaderboard_html(core_leaderboard)))
    sections.append(_html_section(f"Snapshot: {compare_label}", format_benchmark_leaderboard_html(compare_leaderboard)))
    sections.append(_html_section("Direct Comparison", _comparison_table_html(comparison_rows)))
    sections.append(_html_section("Per-Dataset Takeaways", _html_kv_list([(row.dataset, f"best(core)={row.core_best_estimator}, best(compare)={row.compare_best_estimator}, delta_best_f1={_fmt(row.compare_minus_core_best_f1)}, {row.focus_estimator} vs {row.baseline_estimator}={_fmt(row.focus_minus_baseline_f1)}") for row in comparison_rows])))
    return _html_document(title, sections)


def _comparison_summary(rows: list[BenchmarkComparisonRow]) -> list[str]:
    lines = ["## Executive Summary", ""]
    for key, value in _comparison_summary_items(rows):
        lines.append(f"- **{key}**: `{value}`")
    lines.append("")
    return lines


def _comparison_summary_items(rows: list[BenchmarkComparisonRow]) -> list[tuple[str, str]]:
    compare_beats_core = sum(1 for row in rows if (row.compare_minus_core_best_f1 or 0.0) > 0)
    focus_beats_baseline = sum(1 for row in rows if (row.focus_minus_baseline_f1 or 0.0) > 0)
    top_focus = max(rows, key=lambda row: row.focus_minus_baseline_f1 or float("-inf")) if rows else None
    items = [
        ("datasets_compared", str(len(rows))),
        ("compare_best_beats_core_best", str(compare_beats_core)),
        ("focus_beats_baseline", str(focus_beats_baseline)),
    ]
    if top_focus and top_focus.focus_minus_baseline_f1 is not None:
        items.append(("largest_focus_gain_vs_baseline", f"{top_focus.dataset} (delta_f1={_fmt(top_focus.focus_minus_baseline_f1)})"))
    return items


def _meta_summary_section(
    core_rows: list[AggregatedBenchmarkResult],
    compare_rows: list[AggregatedBenchmarkResult],
    comparison_rows: list[BenchmarkComparisonRow],
    core_label: str,
    compare_label: str,
) -> list[str]:
    lines = ["## Executive Summary", ""]
    for key, value in _meta_summary_items(core_rows, compare_rows, comparison_rows, core_label, compare_label):
        lines.append(f"- **{key}**: `{value}`")
    lines.append("")
    return lines


def _meta_summary_items(
    core_rows: list[AggregatedBenchmarkResult],
    compare_rows: list[AggregatedBenchmarkResult],
    comparison_rows: list[BenchmarkComparisonRow],
    core_label: str,
    compare_label: str,
) -> list[tuple[str, str]]:
    core_top = core_rows[0] if core_rows else None
    compare_top = compare_rows[0] if compare_rows else None
    compare_beats_core = sum(1 for row in comparison_rows if (row.compare_minus_core_best_f1 or 0.0) > 0)
    focus_beats_baseline = sum(1 for row in comparison_rows if (row.focus_minus_baseline_f1 or 0.0) > 0)
    items: list[tuple[str, str]] = []
    if core_top is not None:
        items.append((f"best_in_{core_label}", f"{core_top.dataset} / {core_top.estimator} (f1={_fmt(core_top.f1_macro_mean)}, rules={_fmt(core_top.n_rules_mean)}, fit_s={_fmt(core_top.fit_seconds_mean)})"))
    if compare_top is not None:
        items.append((f"best_in_{compare_label}", f"{compare_top.dataset} / {compare_top.estimator} (f1={_fmt(compare_top.f1_macro_mean)}, rules={_fmt(compare_top.n_rules_mean)}, fit_s={_fmt(compare_top.fit_seconds_mean)})"))
    items.append((f"datasets_where_{compare_label}_best_beats_{core_label}_best", str(compare_beats_core)))
    items.append(("datasets_where_focus_beats_baseline", str(focus_beats_baseline)))
    return items


def _meta_takeaways(rows: list[BenchmarkComparisonRow]) -> list[str]:
    lines: list[str] = []
    for row in rows:
        cmp_delta = _fmt(row.compare_minus_core_best_f1)
        focus_delta = _fmt(row.focus_minus_baseline_f1)
        lines.append(
            f"- **{row.dataset}**: best(core) = `{row.core_best_estimator}` vs best(compare) = `{row.compare_best_estimator}` "
            f"(delta_best_f1={cmp_delta}); `{row.focus_estimator}` vs `{row.baseline_estimator}` = {focus_delta} f1."
        )
    lines.append("")
    return lines


def _comparison_table_markdown(rows: list[BenchmarkComparisonRow]) -> list[str]:
    lines = [
        "| dataset | core_best | core_f1 | compare_best | compare_f1 | delta_best | focus_f1 | baseline_f1 | focus-baseline Δf1 | Δfit_s | Δrules |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.dataset,
                    row.core_best_estimator or "",
                    _fmt(row.core_best_f1),
                    row.compare_best_estimator or "",
                    _fmt(row.compare_best_f1),
                    _fmt(row.compare_minus_core_best_f1),
                    _fmt(row.focus_f1),
                    _fmt(row.baseline_f1),
                    _fmt(row.focus_minus_baseline_f1),
                    _fmt(row.focus_minus_baseline_fit_seconds),
                    _fmt(row.focus_minus_baseline_rules),
                ]
            )
            + " |"
        )
    return lines


def _comparison_table_html(rows: list[BenchmarkComparisonRow]) -> str:
    headers = ["dataset", "core_best", "core_f1", "compare_best", "compare_f1", "delta_best", "focus_f1", "baseline_f1", "focus-baseline Δf1", "Δfit_s", "Δrules"]
    data_rows = []
    for row in rows:
        data_rows.append(
            [
                row.dataset,
                row.core_best_estimator or "",
                _fmt(row.core_best_f1),
                row.compare_best_estimator or "",
                _fmt(row.compare_best_f1),
                _fmt(row.compare_minus_core_best_f1),
                _fmt(row.focus_f1),
                _fmt(row.baseline_f1),
                _fmt(row.focus_minus_baseline_f1),
                _fmt(row.focus_minus_baseline_fit_seconds),
                _fmt(row.focus_minus_baseline_rules),
            ]
        )
    return _html_table(headers, data_rows)


def _group_by_dataset(
    results: list[AggregatedBenchmarkResult],
) -> dict[str, list[AggregatedBenchmarkResult]]:
    grouped: dict[str, list[AggregatedBenchmarkResult]] = {}
    for result in results:
        if result.status != "ok":
            continue
        grouped.setdefault(result.dataset, []).append(result)
    return grouped


def _best_result(
    results: list[AggregatedBenchmarkResult],
) -> AggregatedBenchmarkResult | None:
    if not results:
        return None
    return max(
        results,
        key=lambda item: (
            float(item.f1_macro_mean or float("-inf")),
            -float(item.fit_seconds_mean or float("inf")),
            -float(item.n_rules_mean or float("inf")),
            item.estimator,
        ),
    )


def _find_estimator(
    results: list[AggregatedBenchmarkResult],
    estimator_name: str,
) -> AggregatedBenchmarkResult | None:
    for result in results:
        if result.estimator == estimator_name:
            return result
    return None


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"

