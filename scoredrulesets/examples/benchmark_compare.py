from __future__ import annotations

import argparse
from pathlib import Path

from scoredrulesets.benchmarking import (
    compare_benchmark_runs,
    format_benchmark_comparison_report_html,
    format_benchmark_comparison_report_markdown,
    load_aggregated_results_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two aggregated benchmark runs")
    parser.add_argument("--core-aggregated-json", required=True, type=str)
    parser.add_argument("--compare-aggregated-json", required=True, type=str)
    parser.add_argument("--output-markdown", required=True, type=str)
    parser.add_argument("--output-html", default="", type=str)
    parser.add_argument("--focus-estimator", default="wrapper_hs", type=str)
    parser.add_argument("--baseline-estimator", default="wrapper_cart", type=str)
    parser.add_argument("--title", default="Core vs HS Benchmark Comparison", type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    core_path = Path(args.core_aggregated_json)
    compare_path = Path(args.compare_aggregated_json)
    output_path = Path(args.output_markdown)
    output_html_path = Path(args.output_html) if args.output_html else None

    core_results = load_aggregated_results_json(core_path)
    compare_results = load_aggregated_results_json(compare_path)
    rows = compare_benchmark_runs(
        core_results,
        compare_results,
        focus_estimator=args.focus_estimator,
        baseline_estimator=args.baseline_estimator,
    )
    report = format_benchmark_comparison_report_markdown(
        rows,
        title=args.title,
        core_label=core_path.parent.name,
        compare_label=compare_path.parent.name,
        core_artifacts={"aggregated_json": str(core_path)},
        compare_artifacts={"aggregated_json": str(compare_path)},
    )
    output_path.write_text(report, encoding="utf-8")
    print(f"Markdown comparison report: {output_path}")

    if output_html_path is not None:
        html_report = format_benchmark_comparison_report_html(
            rows,
            title=args.title,
            core_label=core_path.parent.name,
            compare_label=compare_path.parent.name,
            core_artifacts={"aggregated_json": str(core_path)},
            compare_artifacts={"aggregated_json": str(compare_path)},
        )
        output_html_path.write_text(html_report, encoding="utf-8")
        print(f"HTML comparison report: {output_html_path}")


if __name__ == "__main__":
    main()

