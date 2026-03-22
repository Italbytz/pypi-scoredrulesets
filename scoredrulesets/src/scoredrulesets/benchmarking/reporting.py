from __future__ import annotations

import html
from pathlib import Path

from .runner import AggregatedBenchmarkResult


HEATMAP_PREVIEW_NOTE = (
    "Heatmap add-on: compact overview of aggregated F1 values and fit times "
    "per dataset/estimator."
)


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
                _display_estimator(result),
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
                    _display_estimator(result),
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


def format_benchmark_leaderboard_html(
    results: list[AggregatedBenchmarkResult],
) -> str:
    headers = ["rank", "dataset", "estimator", "repeats", "f1_mean", "f1_err", "fit_s", "rules", "atoms"]
    rows = []
    for rank, result in enumerate(results, start=1):
        rows.append(
            [
                str(rank),
                result.dataset,
                _display_estimator(result),
                str(result.n_repeats),
                _fmt_float(result.f1_macro_mean),
                _fmt_float(result.f1_macro_error),
                _fmt_float(result.fit_seconds_mean),
                _fmt_float(result.n_rules_mean),
                _fmt_float(result.n_atoms_mean),
            ]
        )
    return _html_table(headers, rows)


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
        sections.extend(_warnings_section_markdown(results))

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
        heatmap_png = artifact_paths.get("heatmap_png")
        if plot_png or heatmap_png:
            sections.extend(["## Plot Preview", ""])
            if plot_png:
                plot_text = str(plot_png)
                sections.append(f"![Benchmark plot]({plot_text})")
                sections.append("")
            if heatmap_png:
                heatmap_text = str(heatmap_png)
                sections.append(
                    f"_{HEATMAP_PREVIEW_NOTE}_"
                )
                sections.append("")
                sections.append(f"![Benchmark heatmap]({heatmap_text})")
                sections.append("")

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


def format_benchmark_report_html(
    results: list[AggregatedBenchmarkResult],
    *,
    title: str = "Benchmark Report",
    config: dict[str, object] | None = None,
    artifact_paths: dict[str, str | Path] | None = None,
    notes: list[str] | None = None,
) -> str:
    sections: list[str] = []

    if results:
        top = results[0]
        sections.append(
            _html_section(
                "Summary",
                _html_kv_list(
                    [
                        ("datasets", str(len({result.dataset for result in results}))),
                        ("estimators", str(len({result.estimator for result in results}))),
                        ("warning_runs", str(sum(result.validation_warning_count for result in results))),
                        ("warning_models", str(sum(1 for result in results if result.validation_warning_count > 0))),
                        (
                            "top_1_model",
                            f"{top.dataset} / {_display_estimator(top)} (f1={_fmt_float(top.f1_macro_mean)}, rules={_fmt_float(top.n_rules_mean)}, fit_s={_fmt_float(top.fit_seconds_mean)})",
                        ),
                    ]
                ),
            )
        )
        warnings_html = _warnings_section_html(results)
        if warnings_html:
            sections.append(_html_section("Warnings", warnings_html))

    if config:
        sections.append(
            _html_section(
                "Configuration",
                _html_kv_list([(key, str(value)) for key, value in config.items()]),
            )
        )

    if artifact_paths:
        artifact_items = []
        for label, path in artifact_paths.items():
            path_text = str(path)
            artifact_items.append((label, _html_link(path_text, path_text)))
        sections.append(_html_section("Artifacts", _html_kv_list(artifact_items, escape_values=False)))
        plot_png = artifact_paths.get("plot_png")
        heatmap_png = artifact_paths.get("heatmap_png")
        if plot_png or heatmap_png:
            preview_parts: list[str] = []
            if plot_png:
                preview_parts.append(_html_image(str(plot_png), "Benchmark plot"))
            if heatmap_png:
                preview_parts.append(
                    f"<p><em>{html.escape(HEATMAP_PREVIEW_NOTE)}</em></p>"
                )
                preview_parts.append(_html_image(str(heatmap_png), "Benchmark heatmap"))
            sections.append(_html_section("Plot Preview", "".join(preview_parts)))

    if notes:
        sections.append(_html_section("Notes", _html_list(notes)))

    if results:
        sections.append(_html_section("Top per Dataset", _dataset_overview_html(results)))
        sections.append(_html_section("Leaderboard", format_benchmark_leaderboard_html(results)))
        seen: set[str] = set()
        for result in results:
            dataset = result.dataset
            if dataset in seen:
                continue
            seen.add(dataset)
            dataset_results = [item for item in results if item.dataset == dataset]
            sections.append(_dataset_section_html(dataset, dataset_results))

    return _html_document(title, sections)


def _summary_section(results: list[AggregatedBenchmarkResult]) -> list[str]:
    top = results[0]
    n_datasets = len({result.dataset for result in results})
    n_estimators = len({result.estimator for result in results})
    warning_runs = sum(result.validation_warning_count for result in results)
    warning_models = sum(1 for result in results if result.validation_warning_count > 0)
    lines = ["## Summary", ""]
    lines.append(f"- **datasets**: `{n_datasets}`")
    lines.append(f"- **estimators**: `{n_estimators}`")
    lines.append(f"- **warning_runs**: `{warning_runs}`")
    lines.append(f"- **warning_models**: `{warning_models}`")
    lines.append(
        "- **top_1_model**: "
        f"`{top.dataset} / {_display_estimator(top)}` "
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
            f"`{_display_estimator(summary['best'])}` (f1={_fmt_float(summary['best'].f1_macro_mean)}, rules={_fmt_float(summary['best'].n_rules_mean)}, fit_s={_fmt_float(summary['best'].fit_seconds_mean)})"
        )
        lines.append(
            "- **smallest_model**: "
            f"`{_display_estimator(summary['smallest'])}` (rules={_fmt_float(summary['smallest'].n_rules_mean)}, atoms={_fmt_float(summary['smallest'].n_atoms_mean)}, f1={_fmt_float(summary['smallest'].f1_macro_mean)})"
        )
        lines.append(
            "- **fastest_model**: "
            f"`{_display_estimator(summary['fastest'])}` (fit_s={_fmt_float(summary['fastest'].fit_seconds_mean)}, f1={_fmt_float(summary['fastest'].f1_macro_mean)}, rules={_fmt_float(summary['fastest'].n_rules_mean)})"
        )
        warned = [item for item in grouped[dataset] if item.validation_warning_count > 0]
        if warned:
            lines.append(f"- **warnings**: `{len(warned)}` estimator(s) with validation warnings")
        lines.append("")
    return lines


def _dataset_overview_html(results: list[AggregatedBenchmarkResult]) -> str:
    grouped: dict[str, list[AggregatedBenchmarkResult]] = {}
    for result in results:
        grouped.setdefault(result.dataset, []).append(result)

    blocks = []
    seen: set[str] = set()
    for result in results:
        dataset = result.dataset
        if dataset in seen:
            continue
        seen.add(dataset)
        summary = _select_dataset_summary(grouped[dataset])
        blocks.append(
            "<div class='dataset-card'>"
            f"<h3>{html.escape(dataset)}</h3>"
            + _html_kv_list(
                [
                    (
                        "best_model",
                        f"{_display_estimator(summary['best'])} (f1={_fmt_float(summary['best'].f1_macro_mean)}, rules={_fmt_float(summary['best'].n_rules_mean)}, fit_s={_fmt_float(summary['best'].fit_seconds_mean)})",
                    ),
                    (
                        "smallest_model",
                        f"{_display_estimator(summary['smallest'])} (rules={_fmt_float(summary['smallest'].n_rules_mean)}, atoms={_fmt_float(summary['smallest'].n_atoms_mean)}, f1={_fmt_float(summary['smallest'].f1_macro_mean)})",
                    ),
                    (
                        "fastest_model",
                        f"{_display_estimator(summary['fastest'])} (fit_s={_fmt_float(summary['fastest'].fit_seconds_mean)}, f1={_fmt_float(summary['fastest'].f1_macro_mean)}, rules={_fmt_float(summary['fastest'].n_rules_mean)})",
                    ),
                    (
                        "warnings",
                        str(sum(1 for item in grouped[dataset] if item.validation_warning_count > 0)),
                    ),
                ]
            )
            + "</div>"
        )
    return "".join(blocks)


def _dataset_summary_section(results: list[AggregatedBenchmarkResult]) -> list[str]:
    summary = _select_dataset_summary(results)
    best = summary["best"]
    smallest = summary["smallest"]
    fastest = summary["fastest"]

    lines = [
        "- **best_model**: "
        f"`{_display_estimator(best)}` (f1={_fmt_float(best.f1_macro_mean)}, rules={_fmt_float(best.n_rules_mean)}, fit_s={_fmt_float(best.fit_seconds_mean)})",
        "- **smallest_model**: "
        f"`{_display_estimator(smallest)}` (rules={_fmt_float(smallest.n_rules_mean)}, atoms={_fmt_float(smallest.n_atoms_mean)}, f1={_fmt_float(smallest.f1_macro_mean)})",
        "- **fastest_model**: "
        f"`{_display_estimator(fastest)}` (fit_s={_fmt_float(fastest.fit_seconds_mean)}, f1={_fmt_float(fastest.f1_macro_mean)}, rules={_fmt_float(fastest.n_rules_mean)})",
    ]
    warned = [item for item in results if item.validation_warning_count > 0]
    if warned:
        lines.append(f"- **warnings**: `{len(warned)}` estimator(s) with validation warnings")
    lines.append("")
    return lines


def _dataset_section_html(dataset: str, results: list[AggregatedBenchmarkResult]) -> str:
    summary = _select_dataset_summary(results)
    return (
        f"<section data-dataset='{html.escape(dataset, quote=True)}'>"
        f"<h2>Dataset: {html.escape(dataset)}</h2>"
        + _html_kv_list(
            [
                (
                    "best_model",
                    f"{_display_estimator(summary['best'])} (f1={_fmt_float(summary['best'].f1_macro_mean)}, rules={_fmt_float(summary['best'].n_rules_mean)}, fit_s={_fmt_float(summary['best'].fit_seconds_mean)})",
                ),
                (
                    "smallest_model",
                    f"{_display_estimator(summary['smallest'])} (rules={_fmt_float(summary['smallest'].n_rules_mean)}, atoms={_fmt_float(summary['smallest'].n_atoms_mean)}, f1={_fmt_float(summary['smallest'].f1_macro_mean)})",
                ),
                (
                    "fastest_model",
                    f"{_display_estimator(summary['fastest'])} (fit_s={_fmt_float(summary['fastest'].fit_seconds_mean)}, f1={_fmt_float(summary['fastest'].f1_macro_mean)}, rules={_fmt_float(summary['fastest'].n_rules_mean)})",
                ),
                (
                    "warnings",
                    str(sum(1 for item in results if item.validation_warning_count > 0)),
                ),
            ]
        )
        + format_benchmark_leaderboard_html(results)
        + "</section>"
    )


def _display_estimator(result: AggregatedBenchmarkResult) -> str:
    if getattr(result, "validation_warning_count", 0) > 0:
        return f"{result.estimator} ⚠x{int(result.validation_warning_count)}"
    return result.estimator


def _warnings_section_markdown(results: list[AggregatedBenchmarkResult]) -> list[str]:
    warned = [result for result in results if getattr(result, "validation_warning_count", 0) > 0]
    if not warned:
        return []
    lines = ["## Warnings", ""]
    for result in warned:
        example = result.validation_warning_example or "warning present"
        lines.append(
            f"- `{result.dataset} / {result.estimator}`: {int(result.validation_warning_count)} warning run(s). Example: {example}"
        )
    lines.append("")
    return lines


def _warnings_section_html(results: list[AggregatedBenchmarkResult]) -> str:
    warned = [result for result in results if getattr(result, "validation_warning_count", 0) > 0]
    if not warned:
        return ""
    return "<ul>" + "".join(
        f"<li><strong>{html.escape(result.dataset)} / {html.escape(result.estimator)}</strong>: "
        f"{int(result.validation_warning_count)} warning run(s). Example: {html.escape(result.validation_warning_example or 'warning present')}</li>"
        for result in warned
    ) + "</ul>"


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


def _html_document(title: str, sections: list[str]) -> str:
    escaped_title = html.escape(title)
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{escaped_title}</title>"
        "<style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;line-height:1.45;margin:2rem;max-width:1200px;}"
        "h1,h2,h3{margin-top:1.4rem;}"
        "table{border-collapse:collapse;width:100%;margin:1rem 0;}"
        "th,td{border:1px solid #d0d7de;padding:0.45rem 0.6rem;text-align:left;}"
        "th{background:#f6f8fa;}"
        "code{background:#f6f8fa;padding:0.1rem 0.25rem;border-radius:4px;}"
        "ul{padding-left:1.25rem;}"
        "img{max-width:100%;height:auto;border:1px solid #d0d7de;border-radius:6px;}"
        ".dataset-card{border:1px solid #d0d7de;border-radius:6px;padding:0.75rem 1rem;margin:0.75rem 0;background:#fafbfc;}"
        "</style></head><body>"
        f"<h1>{escaped_title}</h1>"
        + "".join(sections)
        + "</body></html>"
    )


def _html_section(title: str, body: str) -> str:
    return f"<section><h2>{html.escape(title)}</h2>{body}</section>"


def _html_kv_list(items: list[tuple[str, str]], *, escape_values: bool = True) -> str:
    rows = []
    for key, value in items:
        safe_value = html.escape(value) if escape_values else value
        rows.append(f"<li><strong>{html.escape(key)}</strong>: {safe_value}</li>")
    return "<ul>" + "".join(rows) + "</ul>"


def _html_list(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in items) + "</ul>"


def _html_table(headers: list[str], rows: list[list[str]]) -> str:
    thead = "<thead><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in headers) + "</tr></thead>"
    body_rows = []
    for row in rows:
        body_rows.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>")
    return "<table>" + thead + "<tbody>" + "".join(body_rows) + "</tbody></table>"


def _html_link(href: str, label: str) -> str:
    return f"<a href='{html.escape(href, quote=True)}'>{html.escape(label)}</a>"


def _html_image(src: str, alt: str) -> str:
    return f"<img src='{html.escape(src, quote=True)}' alt='{html.escape(alt, quote=True)}'>"
