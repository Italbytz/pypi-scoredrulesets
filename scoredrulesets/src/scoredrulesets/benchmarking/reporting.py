from __future__ import annotations

from pathlib import Path

from .runner import AggregatedBenchmarkResult


def format_benchmark_leaderboard_table(
    results: list[AggregatedBenchmarkResult],
) -> str:
    headers = [
        "rank",
        "dataset",
        "estimator",
        "repeats",
        "f1_mean",
        "f1_err",
        "fit_s",
        "rules",
        "atoms",
    ]
    rows: list[list[str]] = []
    for rank, result in enumerate(results, start=1):
        rows.append(
            [
                str(rank),
                result.dataset,
                result.estimator,
                str(result.n_repeats),
                _fmt_float(result.f1_macro_mean),
                _fmt_float(result.f1_macro_error),
                _fmt_float(result.fit_seconds_mean),
                _fmt_float(result.n_rules_mean),
                _fmt_float(result.n_atoms_mean),
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def fmt_row(values: list[str]) -> str:
        return "| " + " | ".join(values[i].ljust(widths[i]) for i in range(len(values))) + " |"

    sep = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    lines = [sep, fmt_row(headers), sep]
    for row in rows:
        lines.append(fmt_row(row))
    lines.append(sep)
    return "\n".join(lines)


def format_benchmark_leaderboard_markdown(
    results: list[AggregatedBenchmarkResult],
) -> str:
    lines = [
        "| rank | dataset | estimator | repeats | f1_mean | f1_err | fit_s | rules | atoms |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, result in enumerate(results, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    result.dataset,
                    result.estimator,
                    str(result.n_repeats),
                    _fmt_float(result.f1_macro_mean),
                    _fmt_float(result.f1_macro_error),
                    _fmt_float(result.fit_seconds_mean),
                    _fmt_float(result.n_rules_mean),
                    _fmt_float(result.n_atoms_mean),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def format_benchmark_report_markdown(
    results: list[AggregatedBenchmarkResult],
    *,
    title: str = "Benchmark Report",
    config: dict[str, object] | None = None,
    artifact_paths: dict[str, str | Path] | None = None,
    notes: list[str] | None = None,
) -> str:
    sections = [f"# {title}", ""]

    if results:
        sections.extend(_summary_section(results))

    if config:
        sections.extend(["## Configuration", ""])
        for key, value in config.items():
            sections.append(f"- **{key}**: `{value}`")
        sections.append("")

    if artifact_paths:
        sections.extend(["## Artifacts", ""])
        for label, path in artifact_paths.items():
            path_text = str(path)
            sections.append(f"- **{label}**: [{path_text}]({path_text})")
        sections.append("")
        plot_png = artifact_paths.get("plot_png")
        if plot_png:
            plot_text = str(plot_png)
            sections.extend(["## Plot Preview", "", f"![Benchmark plot]({plot_text})", ""])

    if notes:
        sections.extend(["## Notes", ""])
        for note in notes:
            sections.append(f"- {note}")
        sections.append("")

    if results:
        sections.extend(_dataset_overview_section(results))

    sections.extend(["## Leaderboard", "", format_benchmark_leaderboard_markdown(results), ""])
    if results:
        sections.extend(_dataset_sections(results))
    return "\n".join(sections)


def _summary_section(results: list[AggregatedBenchmarkResult]) -> list[str]:
    top = results[0]
    n_datasets = len({result.dataset for result in results})
    n_estimators = len({result.estimator for result in results})
    lines = ["## Summary", ""]
    lines.append(f"- **datasets**: `{n_datasets}`")
    lines.append(f"- **estimators**: `{n_estimators}`")
    lines.append(
        "- **top_1_model**: "
        f"`{top.dataset} / {top.estimator}` "
        f"(f1={_fmt_float(top.f1_macro_mean)}, rules={_fmt_float(top.n_rules_mean)}, fit_s={_fmt_float(top.fit_seconds_mean)})"
    )
    lines.append("")
    return lines


def _dataset_sections(results: list[AggregatedBenchmarkResult]) -> list[str]:
    grouped: dict[str, list[AggregatedBenchmarkResult]] = {}
    for result in results:
        grouped.setdefault(result.dataset, []).append(result)

    sections: list[str] = []
    seen: set[str] = set()
    for result in results:
        dataset = result.dataset
        if dataset in seen:
            continue
        seen.add(dataset)
        dataset_results = grouped[dataset]
        sections.extend(
            [
                f"## Dataset: {dataset}",
                "",
            ]
        )
        sections.extend(_dataset_summary_section(dataset_results))
        sections.extend(
            [
                format_benchmark_leaderboard_markdown(dataset_results),
                "",
            ]
        )
    return sections


def _dataset_overview_section(results: list[AggregatedBenchmarkResult]) -> list[str]:
    grouped: dict[str, list[AggregatedBenchmarkResult]] = {}
    for result in results:
        grouped.setdefault(result.dataset, []).append(result)

    lines = ["## Top per Dataset", ""]
    seen: set[str] = set()
    for result in results:
        dataset = result.dataset
        if dataset in seen:
            continue
        seen.add(dataset)
        summary = _select_dataset_summary(grouped[dataset])
        lines.append(f"### {dataset}")
        lines.append("")
        lines.append(
            "- **best_model**: "
            f"`{summary['best'].estimator}` (f1={_fmt_float(summary['best'].f1_macro_mean)}, rules={_fmt_float(summary['best'].n_rules_mean)}, fit_s={_fmt_float(summary['best'].fit_seconds_mean)})"
        )
        lines.append(
            "- **smallest_model**: "
            f"`{summary['smallest'].estimator}` (rules={_fmt_float(summary['smallest'].n_rules_mean)}, atoms={_fmt_float(summary['smallest'].n_atoms_mean)}, f1={_fmt_float(summary['smallest'].f1_macro_mean)})"
        )
        lines.append(
            "- **fastest_model**: "
            f"`{summary['fastest'].estimator}` (fit_s={_fmt_float(summary['fastest'].fit_seconds_mean)}, f1={_fmt_float(summary['fastest'].f1_macro_mean)}, rules={_fmt_float(summary['fastest'].n_rules_mean)})"
        )
        lines.append("")
    return lines


def _dataset_summary_section(results: list[AggregatedBenchmarkResult]) -> list[str]:
    summary = _select_dataset_summary(results)
    best = summary["best"]
    smallest = summary["smallest"]
    fastest = summary["fastest"]

    return [
        "- **best_model**: "
        f"`{best.estimator}` (f1={_fmt_float(best.f1_macro_mean)}, rules={_fmt_float(best.n_rules_mean)}, fit_s={_fmt_float(best.fit_seconds_mean)})",
        "- **smallest_model**: "
        f"`{smallest.estimator}` (rules={_fmt_float(smallest.n_rules_mean)}, atoms={_fmt_float(smallest.n_atoms_mean)}, f1={_fmt_float(smallest.f1_macro_mean)})",
        "- **fastest_model**: "
        f"`{fastest.estimator}` (fit_s={_fmt_float(fastest.fit_seconds_mean)}, f1={_fmt_float(fastest.f1_macro_mean)}, rules={_fmt_float(fastest.n_rules_mean)})",
        "",
    ]


def _select_dataset_summary(
    results: list[AggregatedBenchmarkResult],
) -> dict[str, AggregatedBenchmarkResult]:
    best = max(
        results,
        key=lambda item: (
            _value_or_neg_inf(item.f1_macro_mean),
            -_value_or_inf(item.fit_seconds_mean),
            -_value_or_inf(item.n_rules_mean),
            item.estimator,
        ),
    )
    smallest = min(
        results,
        key=lambda item: (
            _value_or_inf(item.n_rules_mean),
            _value_or_inf(item.n_atoms_mean),
            _value_or_inf(item.fit_seconds_mean),
            item.estimator,
        ),
    )
    fastest = min(
        results,
        key=lambda item: (
            _value_or_inf(item.fit_seconds_mean),
            _value_or_inf(item.n_rules_mean),
            -_value_or_neg_inf(item.f1_macro_mean),
            item.estimator,
        ),
    )
    return {
        "best": best,
        "smallest": smallest,
        "fastest": fastest,
    }


def _value_or_inf(value: float | None) -> float:
    if value is None:
        return float("inf")
    return float(value)


def _value_or_neg_inf(value: float | None) -> float:
    if value is None:
        return float("-inf")
    return float(value)


def _fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"

