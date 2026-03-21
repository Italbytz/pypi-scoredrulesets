"""
Umfangreicher Benchmark aller relevanten Schätzer und Datensätze mit Fortschrittsanzeige und Report.

- Nutzt alle verfügbaren Estimatoren aus benchmarking/estimators.py
- Nutzt alle Paper-UCI- und sklearn-Datensätze
- Gibt Fortschritt und Zwischenstände aus
- Erstellt Markdown- und HTML-Report sowie CSV/JSON

Laufzeit: Kann je nach Konfiguration sehr lang sein!
"""

from pathlib import Path
from scoredrulesets.benchmarking.estimators import default_estimator_specs
from scoredrulesets.benchmarking.datasets import load_sklearn_datasets, load_online_paper_uci_datasets
from scoredrulesets.benchmarking import (
    BenchmarkConfig,
    run_benchmarks,
    aggregate_benchmark_results,
    aggregated_results_as_dicts,
    build_benchmark_leaderboard,
    format_benchmark_report_markdown,
    format_benchmark_report_html,
    format_benchmark_leaderboard_table,
    plot_benchmark_results,
    plot_benchmark_heatmap,
)
from scoredrulesets.benchmarking.runner import results_as_dicts
import json
import time
from tqdm import tqdm


def main():
    # Alle Estimatoren und Datensätze sammeln
    estimator_specs = default_estimator_specs()
    estimator_names = list(estimator_specs.keys())
    datasets = load_sklearn_datasets()
    datasets.update(load_online_paper_uci_datasets())
    dataset_names = list(datasets.keys())

    print(f"Starte Benchmark mit {len(estimator_names)} Schätzern und {len(dataset_names)} Datensätzen...")
    print(f"Estimatoren: {', '.join(estimator_names)}")
    print(f"Datensätze: {', '.join(dataset_names)}")

    config = BenchmarkConfig(
        dataset_names=dataset_names,
        estimator_names=estimator_names,
        use_paper_split_policy=True,
        include_online_uci=True,
        paper_uci_strict=False,
        repeats=3,
        random_state=42,
    )

    # Fortschrittsanzeige
    print("\n[1/3] Führe Benchmarks aus...")
    t0 = time.time()
    results = run_benchmarks(config, progress_callback=lambda msg: print(f"[Progress] {msg}"))
    t1 = time.time()
    print(f"Benchmarks abgeschlossen in {t1-t0:.1f} Sekunden.")

    print("\n[2/3] Aggregiere und erstelle Reports...")
    payload = results_as_dicts(results)
    aggregated = aggregate_benchmark_results(results, error_bar="std")
    leaderboard = build_benchmark_leaderboard(aggregated)
    aggregated_payload = aggregated_results_as_dicts(aggregated)

    # Schreibe CSV/JSON
    Path("benchmark_results.csv").write_text(_csv_string(payload), encoding="utf-8")
    Path("benchmark_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    Path("benchmark_results_aggregated.csv").write_text(_csv_string(aggregated_payload), encoding="utf-8")
    Path("benchmark_results_aggregated.json").write_text(json.dumps(aggregated_payload, indent=2), encoding="utf-8")

    # Plots und Reports
    png_path, pdf_path = plot_benchmark_results(results, output_base=Path("benchmark_results"), aggregate_repeats=True)
    heatmap_png, heatmap_pdf = plot_benchmark_heatmap(results, output_base=Path("benchmark_results_heatmap"))
    md_report = format_benchmark_report_markdown(
        leaderboard,
        title="ScoredRuleSets Benchmark Report",
        config={"datasets": ",".join(dataset_names), "estimators": ",".join(estimator_names)},
        artifact_paths={
            "raw_csv": "benchmark_results.csv",
            "raw_json": "benchmark_results.json",
            "aggregated_csv": "benchmark_results_aggregated.csv",
            "aggregated_json": "benchmark_results_aggregated.json",
            "plot_png": str(png_path),
            "plot_pdf": str(pdf_path),
            "heatmap_png": str(heatmap_png),
            "heatmap_pdf": str(heatmap_pdf),
        },
        notes=[
            "Alle Schätzer und Datensätze, 3 Wiederholungen, Paper-Split-Policy.",
            "Laufzeit und Komplexität können je nach System stark variieren.",
        ],
    )
    Path("benchmark_leaderboard.md").write_text(md_report, encoding="utf-8")
    html_report = format_benchmark_report_html(
        leaderboard,
        title="ScoredRuleSets Benchmark Report",
        config={"datasets": ",".join(dataset_names), "estimators": ",".join(estimator_names)},
        artifact_paths={
            "raw_csv": "benchmark_results.csv",
            "raw_json": "benchmark_results.json",
            "aggregated_csv": "benchmark_results_aggregated.csv",
            "aggregated_json": "benchmark_results_aggregated.json",
            "plot_png": str(png_path),
            "plot_pdf": str(pdf_path),
            "heatmap_png": str(heatmap_png),
            "heatmap_pdf": str(heatmap_pdf),
        },
        notes=[
            "Alle Schätzer und Datensätze, 3 Wiederholungen, Paper-Split-Policy.",
            "Laufzeit und Komplexität können je nach System stark variieren.",
        ],
    )
    Path("benchmark_leaderboard.html").write_text(html_report, encoding="utf-8")

    print("\n[3/3] Fertig! Ergebnisse und Reports wurden geschrieben.")
    print("Wichtige Dateien:")
    print("- benchmark_results.csv / .json (Rohdaten)")
    print("- benchmark_results_aggregated.csv / .json (Aggregiert)")
    print("- benchmark_leaderboard.md / .html (Report)")
    print("- benchmark_results.png / .pdf (Plots)")
    print("- benchmark_results_heatmap.png / .pdf (Heatmap)")

    print("\nLeaderboard (Top 10):")
    print(format_benchmark_leaderboard_table(leaderboard)[:2000])


def _csv_string(rows):
    if not rows:
        return ""
    import io, csv
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


if __name__ == "__main__":
    main()

