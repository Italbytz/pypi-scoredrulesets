# -*- coding: utf-8 -*-
"""Generiere Reports aus vorhandenen Benchmark-Ergebnissen (JSON-Dateien).

Nach einem fertigen Benchmark-Lauf können die Reports unabhängig neu erstellt
werden, ohne die Benchmarks erneut zu laufen. Dies ist nützlich für:

- Neue Report-Versionen/Layouts generieren
- Reports mit anderen Aggregationsmethoden
- Dokumentation aktualisieren
- Verschiedene Report-Formate

Aufruf:
    python examples/benchmarks/generate_reports.py normal-lite
    python examples/benchmarks/generate_reports.py standard
    python examples/benchmarks/generate_reports.py full
    python examples/benchmarks/generate_reports.py --input-dir benchmarks/normal_lite --output-dir benchmarks/normal_lite
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scoredrulesets.benchmarking import (
    aggregate_benchmark_results,
    aggregated_results_as_dicts,
    build_benchmark_leaderboard,
    format_benchmark_leaderboard_table,
    format_benchmark_report_markdown,
    format_benchmark_report_html,
    plot_benchmark_heatmap,
    plot_benchmark_heatmap_combined,
    plot_benchmark_results,
    plot_combined_dot,
    plot_pareto_front,
)
from scoredrulesets.benchmarking.runner import BenchmarkResult


def load_benchmark_results_from_json(json_file: Path):
    """Lade Benchmark-Ergebnisse aus JSON-Datei.
    
    Erwartet das Format, das von benchmark_*.py generiert wird:
    ```
    [
      {"dataset": "...", "estimator": "...", "repeat": 0, "f1_macro": ...},
      ...
    ]
    ```
    """
    with json_file.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    results = []
    for item in raw_data:
        results.append(
            BenchmarkResult(
                dataset=item.get("dataset"),
                estimator=item.get("estimator"),
                repeat=item.get("repeat", 0),
                status=item.get("status", "ok"),
                skip_reason=item.get("skip_reason"),
                error=item.get("error"),
                f1_macro=item.get("f1_macro"),
                fit_seconds=item.get("fit_seconds", item.get("training_time")),
                predict_seconds=item.get("predict_seconds", item.get("prediction_time")),
                n_rules=item.get("n_rules"),
                n_atoms=item.get("n_atoms"),
                ruleset_json_bytes=item.get("ruleset_json_bytes"),
                n_train=item.get("n_train"),
                n_test=item.get("n_test"),
                n_classes=item.get("n_classes"),
                validation_action=item.get("validation_action"),
                validation_message=item.get("validation_message"),
            )
        )
    
    return results


def generate_reports(
    input_dir: Path,
    output_dir: Path,
    title: str = "Benchmark Report",
    config_desc: str = "",
    design_desc: str = "",
):
    """Generiere alle Reports aus Benchmark-Ergebnissen.
    
    Args:
        input_dir: Verzeichnis mit benchmark_results.json
        output_dir: Ausgabeverzeichnis für Reports
        title: Report-Titel
        config_desc: Beschreibung der Konfiguration
        design_desc: Beschreibung des Experimental-Designs
    """
    
    json_file = input_dir / "benchmark_results.json"
    if not json_file.exists():
        print(f"ERROR: {json_file} nicht gefunden")
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[1/3] Lade Ergebnisse aus: {json_file}")
    raw_data = load_benchmark_results_from_json(json_file)
    print(f"      {len(raw_data)} Einträge geladen")
    
    # Aggregiere Ergebnisse
    print(f"[2/3] Aggregiere und erstelle Reports...")
    aggregated = aggregate_benchmark_results(raw_data, error_bar="std")
    leaderboard = build_benchmark_leaderboard(aggregated)
    aggregated_payload = aggregated_results_as_dicts(aggregated)
    
    # Generiere Plots
    print(f"      Erstelle Visualisierungen...")
    base = output_dir / "benchmark_results"
    png_path, pdf_path = plot_benchmark_results(
        raw_data, output_base=base, aggregate_repeats=True
    )
    heatmap_png, heatmap_pdf = plot_benchmark_heatmap(
        raw_data, output_base=output_dir / "benchmark_results_heatmap"
    )
    combined_png, combined_pdf = plot_benchmark_heatmap_combined(
        raw_data, output_base=output_dir / "benchmark_results_heatmap_combined"
    )
    dot_png, dot_pdf = plot_combined_dot(
        raw_data, output_base=output_dir / "benchmark_results_combined_dot"
    )
    pareto_png, pareto_pdf = plot_pareto_front(
        raw_data, output_base=output_dir / "benchmark_results_pareto"
    )
    
    # Markdown-Report
    print(f"      Schreibe Markdown-Report...")
    md_report = format_benchmark_report_markdown(
        leaderboard,
        title=title,
        config={
            "config": config_desc,
        },
        artifact_paths={
            "raw_csv": "benchmark_results.csv",
            "raw_json": "benchmark_results.json",
            "aggregated_csv": "benchmark_results_aggregated.csv",
            "aggregated_json": "benchmark_results_aggregated.json",
            "plot_png": str(png_path.name),
            "plot_pdf": str(pdf_path.name),
            "heatmap_png": str(heatmap_png.name),
            "heatmap_pdf": str(heatmap_pdf.name),
            "combined_heatmap_png": str(combined_png.name),
            "combined_heatmap_pdf": str(combined_pdf.name),
            "combined_dot_png": str(dot_png.name),
            "combined_dot_pdf": str(dot_pdf.name),
            "pareto_png": str(pareto_png.name),
            "pareto_pdf": str(pareto_pdf.name),
        },
        notes=[
            f"Reports regeneriert von vorhandenen Benchmark-Ergebnissen.",
            design_desc,
        ] if design_desc else [],
    )
    (output_dir / "benchmark_report.md").write_text(md_report, encoding="utf-8")
    
    # HTML-Report
    print(f"      Schreibe HTML-Report...")
    html_report = format_benchmark_report_html(
        leaderboard,
        title=title,
        config={
            "config": config_desc,
        },
        artifact_paths={
            "raw_csv": "benchmark_results.csv",
            "raw_json": "benchmark_results.json",
            "aggregated_csv": "benchmark_results_aggregated.csv",
            "aggregated_json": "benchmark_results_aggregated.json",
            "plot_png": str(png_path.name),
            "plot_pdf": str(pdf_path.name),
            "heatmap_png": str(heatmap_png.name),
            "heatmap_pdf": str(heatmap_pdf.name),
            "combined_heatmap_png": str(combined_png.name),
            "combined_heatmap_pdf": str(combined_pdf.name),
            "combined_dot_png": str(dot_png.name),
            "combined_dot_pdf": str(dot_pdf.name),
            "pareto_png": str(pareto_png.name),
            "pareto_pdf": str(pareto_pdf.name),
        },
    )
    (output_dir / "benchmark_report.html").write_text(html_report, encoding="utf-8")
    
    # Aggregierte Daten exportieren
    print(f"      Schreibe aggregierte Daten...")
    (output_dir / "benchmark_results_aggregated.json").write_text(
        json.dumps(aggregated_payload, indent=2), encoding="utf-8"
    )
    
    # Zusammenfassung
    print(f"\n[3/3] Fertig! Reports in: {output_dir}/")
    print(f"  benchmark_report.md / .html (Reports)")
    print(f"  benchmark_results.png / .pdf (Plots)")
    print(f"  benchmark_results_heatmap*.png / .pdf (Heatmaps)")
    print(f"  benchmark_results_combined_dot.png / .pdf (Dot-Plot Vergleich)")
    print(f"  benchmark_results_pareto.png / .pdf (Pareto-Front)")
    
    print(f"\nLeaderboard (Top 15):")
    print(format_benchmark_leaderboard_table(leaderboard)[:3000])
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generiere Reports aus vorhandenen Benchmark-Ergebnissen",
        epilog="Nach einem fertigen Benchmark können die Reports unabhängig neu erzeugt werden.",
    )
    
    # Preset-Modes
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["normal-lite", "standard", "full"],
        help="Vordefinierter Mode (normal-lite/standard/full) oder --input-dir verwenden",
    )
    
    # Custom Directories
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Eingabeverzeichnis mit benchmark_results.json (overrides mode)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Ausgabeverzeichnis (default: input-dir)",
    )
    
    # Custom Descriptions
    parser.add_argument(
        "--title",
        type=str,
        default="ScoredRuleSets Benchmark Report",
        help="Report-Titel",
    )
    parser.add_argument(
        "--design",
        type=str,
        help="Design-Beschreibung für Report",
    )
    
    args = parser.parse_args()
    
    # Determine input/output directories
    if args.input_dir:
        input_dir = args.input_dir
        output_dir = args.output_dir or input_dir
    elif args.mode:
        mode = args.mode
        mode_to_dir = {
            "normal-lite": "normal_lite",
            "standard": "standard",
            "full": "full",
        }
        input_dir = Path("benchmarks") / mode_to_dir[mode]
        output_dir = args.output_dir or input_dir
        
        # Set defaults for mode
        if mode == "normal-lite":
            args.title = "ScoredRuleSets Normal-Lite Benchmark"
            args.design = "10 Datasets, voller Estimator-Katalog (ohne MUX), reduzierte Repeats"
        elif mode == "standard":
            args.title = "ScoredRuleSets Standard Benchmark"
            args.design = "10 reale Datasets, breite Schaetzer-Auswahl"
        elif mode == "full":
            args.title = "ScoredRuleSets Full Benchmark – Komplette Auswertung"
            args.design = "Alle verfuegbaren Datasets und Schaetzer"
    else:
        parser.print_help()
        sys.exit(1)
    
    # Generate reports
    success = generate_reports(
        input_dir=input_dir,
        output_dir=output_dir,
        title=args.title,
        config_desc=f"Mode: {args.mode or 'custom'}" if args.mode else "Custom",
        design_desc=args.design or "",
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
