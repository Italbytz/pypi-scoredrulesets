from __future__ import annotations

import argparse
import csv
import json
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
    plot_benchmark_results,
    run_benchmarks,
)
from scoredrulesets.benchmarking.runner import results_as_dicts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark fuer Scored Rule Set Estimatoren")
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Kommagetrennte Liste, z.B. sklearn_iris,sklearn_wine oder paper_uci",
    )
    parser.add_argument(
        "--paper-uci",
        action="store_true",
        help="Fuegt den Paper-UCI-Katalog als Dataset-Auswahl hinzu (Alias: paper_uci).",
    )
    parser.add_argument(
        "--paper-uci-strict",
        action="store_true",
        help="Fehlschlag, wenn bei paper_uci nicht alle Paper-Datensaetze verfuegbar sind.",
    )
    parser.add_argument(
        "--offline-uci",
        action="store_true",
        help="Deaktiviert Online-UCI-Loader (ucimlrepo/OpenML) und nutzt nur lokale Registry-Quellen.",
    )
    parser.add_argument(
        "--estimators",
        type=str,
        default="",
        help="Kommagetrennte Liste, z.B. wrapper_cart,native,gp,gp_fast,wrapper_hs",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--paper-split-policy",
        action="store_true",
        help="Aktiviert die im Paper verwendete Split-Policy (n<500:0.30, 500-4999:0.25, >=5000:0.20).",
    )
    parser.add_argument("--output-csv", type=str, default="benchmark_results.csv")
    parser.add_argument("--output-json", type=str, default="benchmark_results.json")
    parser.add_argument("--output-plot-base", type=str, default="benchmark_results")
    parser.add_argument(
        "--plot-size-metric",
        type=str,
        default="n_rules",
        choices=["n_rules", "n_atoms", "ruleset_json_bytes"],
    )
    parser.add_argument("--aggregate-repeats", action="store_true")
    parser.add_argument(
        "--error-bar",
        type=str,
        default="std",
        choices=["std", "sem"],
    )
    parser.add_argument(
        "--leaderboard-primary-metric",
        type=str,
        default="f1_macro_mean",
        choices=[
            "f1_macro_mean",
            "fit_seconds_mean",
            "predict_seconds_mean",
            "n_rules_mean",
            "n_atoms_mean",
            "ruleset_json_bytes_mean",
        ],
    )
    parser.add_argument("--output-markdown", type=str, default="")
    parser.add_argument("--output-html", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_names = [x.strip() for x in args.datasets.split(",") if x.strip()]
    if args.paper_uci:
        dataset_names.append("paper_uci")
    dataset_names = dataset_names or None
    estimator_names = [x.strip() for x in args.estimators.split(",") if x.strip()] or None
    aggregated_csv = _sibling_output(Path(args.output_csv), "_aggregated")
    aggregated_json = _sibling_output(Path(args.output_json), "_aggregated")

    config = BenchmarkConfig(
        dataset_names=dataset_names,
        estimator_names=estimator_names,
        use_paper_split_policy=bool(args.paper_split_policy),
        include_online_uci=not bool(args.offline_uci),
        paper_uci_strict=bool(args.paper_uci_strict),
        repeats=args.repeats,
        random_state=args.random_state,
    )
    results = run_benchmarks(config)
    payload = results_as_dicts(results)
    aggregated_payload = None
    leaderboard = None
    markdown_path = None
    html_path = None
    heatmap_png_path = None
    heatmap_pdf_path = None
    if args.aggregate_repeats:
        aggregated = aggregate_benchmark_results(results, error_bar=args.error_bar)
        leaderboard = build_benchmark_leaderboard(
            aggregated,
            primary_metric=args.leaderboard_primary_metric,
        )
        aggregated_payload = aggregated_results_as_dicts(aggregated)

    _write_csv(Path(args.output_csv), payload)
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if aggregated_payload is not None:
        _write_csv(aggregated_csv, aggregated_payload)
        aggregated_json.write_text(json.dumps(aggregated_payload, indent=2), encoding="utf-8")
        markdown_path = Path(args.output_markdown) if args.output_markdown else _sibling_output(Path(args.output_json), "_leaderboard").with_suffix(".md")
    png_path, pdf_path = plot_benchmark_results(
        results,
        output_base=Path(args.output_plot_base),
        size_metric=args.plot_size_metric,
        aggregate_repeats=args.aggregate_repeats,
        error_bar=args.error_bar,
    )
    if aggregated_payload is not None:
        heatmap_base = Path(args.output_plot_base).with_name(
            f"{Path(args.output_plot_base).name}_heatmap"
        )
        heatmap_png_path, heatmap_pdf_path = plot_benchmark_heatmap(
            results,
            output_base=heatmap_base,
            error_bar=args.error_bar,
        )
    if aggregated_payload is not None:
        aggregated_csv_name = aggregated_csv.name
        aggregated_json_name = aggregated_json.name
        artifact_paths = {
            "raw_csv": Path(args.output_csv).name,
            "raw_json": Path(args.output_json).name,
            "aggregated_csv": aggregated_csv_name,
            "aggregated_json": aggregated_json_name,
            "plot_png": png_path.name,
            "plot_pdf": pdf_path.name,
        }
        if heatmap_png_path and heatmap_pdf_path:
            artifact_paths["heatmap_png"] = heatmap_png_path.name
            artifact_paths["heatmap_pdf"] = heatmap_pdf_path.name
        report = format_benchmark_report_markdown(
            leaderboard or [],
            title="ScoredRuleSets Benchmark Report",
            config={
                "datasets": ",".join(dataset_names or []),
                "estimators": ",".join(estimator_names or []),
                "repeats": args.repeats,
                "random_state": args.random_state,
                "paper_split_policy": bool(args.paper_split_policy),
                "paper_uci_strict": bool(args.paper_uci_strict),
                "offline_uci": bool(args.offline_uci),
                "plot_size_metric": args.plot_size_metric,
                "error_bar": args.error_bar,
                "leaderboard_primary_metric": args.leaderboard_primary_metric,
            },
            artifact_paths=artifact_paths,
            notes=[
                "Leaderboard ist ueber aggregierte Ergebnisse sortiert.",
                "Plot ist nach Datensaetzen in einzelne Panels aufgeteilt.",
                "Plot zeigt Mittelwerte pro (dataset, estimator) mit Fehlerbalken, wenn --aggregate-repeats aktiv ist.",
                "Die Heatmap bietet eine kompakte Gesamtansicht fuer mittlere F1-Werte und Fit-Zeiten.",
            ],
        )
        markdown_path.write_text(report, encoding="utf-8")
        html_path = Path(args.output_html) if args.output_html else _sibling_output(Path(args.output_json), "_leaderboard").with_suffix(".html")
        html_artifact_paths = {
            "raw_csv": Path(args.output_csv).name,
            "raw_json": Path(args.output_json).name,
            "aggregated_csv": aggregated_csv.name,
            "aggregated_json": aggregated_json.name,
            "plot_png": png_path.name,
            "plot_pdf": pdf_path.name,
        }
        if heatmap_png_path and heatmap_pdf_path:
            html_artifact_paths["heatmap_png"] = heatmap_png_path.name
            html_artifact_paths["heatmap_pdf"] = heatmap_pdf_path.name
        html_report = format_benchmark_report_html(
            leaderboard or [],
            title="ScoredRuleSets Benchmark Report",
            config={
                "datasets": ",".join(dataset_names or []),
                "estimators": ",".join(estimator_names or []),
                "repeats": args.repeats,
                "random_state": args.random_state,
                "paper_split_policy": bool(args.paper_split_policy),
                "paper_uci_strict": bool(args.paper_uci_strict),
                "offline_uci": bool(args.offline_uci),
                "plot_size_metric": args.plot_size_metric,
                "error_bar": args.error_bar,
                "leaderboard_primary_metric": args.leaderboard_primary_metric,
            },
            artifact_paths=html_artifact_paths,
            notes=[
                "Leaderboard ist ueber aggregierte Ergebnisse sortiert.",
                "Plot ist nach Datensaetzen in einzelne Panels aufgeteilt.",
                "Plot zeigt Mittelwerte pro (dataset, estimator) mit Fehlerbalken, wenn --aggregate-repeats aktiv ist.",
                "Die Heatmap bietet eine kompakte Gesamtansicht fuer mittlere F1-Werte und Fit-Zeiten.",
            ],
        )
        html_path.write_text(html_report, encoding="utf-8")

    print(_render_console_table(payload))
    if aggregated_payload is not None:
        print()
        print(format_benchmark_leaderboard_table(leaderboard or []))
    print(f"CSV: {args.output_csv}")
    print(f"JSON: {args.output_json}")
    if aggregated_payload is not None:
        print(f"CSV (aggregated): {aggregated_csv}")
        print(f"JSON (aggregated): {aggregated_json}")
        print(f"Markdown (report): {markdown_path}")
        print(f"HTML (report): {html_path}")
        if heatmap_png_path and heatmap_pdf_path:
            print(f"PNG (heatmap): {heatmap_png_path}")
            print(f"PDF (heatmap): {heatmap_pdf_path}")
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _sibling_output(path: Path, suffix_stub: str) -> Path:
    return path.with_name(f"{path.stem}{suffix_stub}{path.suffix}")


def _render_console_table(rows: list[dict]) -> str:
    if not rows:
        return "<no rows>"

    if "repeat" in rows[0]:
        visible_keys = [
            "dataset",
            "estimator",
            "repeat",
            "status",
            "f1_macro",
            "fit_seconds",
            "predict_seconds",
            "n_rules",
            "n_atoms",
            "ruleset_json_bytes",
        ]
    else:
        visible_keys = [
            "dataset",
            "estimator",
            "n_repeats",
            "status",
            "f1_macro_mean",
            "f1_macro_error",
            "fit_seconds_mean",
            "n_rules_mean",
            "n_atoms_mean",
            "ruleset_json_bytes_mean",
        ]
    cols = {k: [k] for k in visible_keys}
    for row in rows:
        for key in visible_keys:
            value = row.get(key)
            if isinstance(value, float):
                cols[key].append(f"{value:.4f}")
            else:
                cols[key].append("" if value is None else str(value))

    widths = {k: max(len(v) for v in values) for k, values in cols.items()}
    sep = "+-" + "-+-".join("-" * widths[k] for k in visible_keys) + "-+"

    def fmt(values: list[str]) -> str:
        return "| " + " | ".join(values[i].ljust(widths[k]) for i, k in enumerate(visible_keys)) + " |"

    lines = [sep, fmt(visible_keys), sep]
    for row in rows:
        display = []
        for key in visible_keys:
            val = row.get(key)
            if isinstance(val, float):
                display.append(f"{val:.4f}")
            else:
                display.append("" if val is None else str(val))
        lines.append(fmt(display))
    lines.append(sep)
    return "\n".join(lines)


if __name__ == "__main__":
    main()

