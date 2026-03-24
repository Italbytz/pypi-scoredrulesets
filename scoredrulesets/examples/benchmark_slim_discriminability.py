"""Schlanker Benchmark mit maximaler Unterscheidbarkeit zwischen Schaetzertypen.

Auswahlkriterien:
- **Datasets:** 10 Datensaetze mit hoechster Typ-Diskriminierung (aus Pareto-Analyse
  der 2189-Run-Checkpoint-Daten), ergaenzt um 2 reale UCI-Datensaetze.
- **Estimators:** 11 Schaetzer – ein starker Vertreter pro Typ-Familie plus
  die Bug-Fix-relevanten Varianten (gp vs gp_residual, rlcw_macro vs rlcw_micro).

Insgesamt: 11 Schaetzer × 10 Datasets × 5 Repeats = **550 Laeufe**
(statt 31 × 24 × N im vollen Benchmark).

Produziert die gleichen Reports wie der volle Benchmark:
  CSV, JSON, Aggregationen, Heatmaps, Pareto-Front, Leaderboard (MD + HTML).

Aufruf:
    python examples/benchmark_slim_discriminability.py
    python examples/benchmark_slim_discriminability.py --repeats 10
    python examples/benchmark_slim_discriminability.py --timeout 120
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import sys
import time
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
    plot_benchmark_heatmap_combined,
    plot_benchmark_results,
    plot_combined_dot,
    plot_pareto_front,
    run_benchmarks,
)
from scoredrulesets.benchmarking.runner import results_as_dicts

# ---------------------------------------------------------------------------
# Dataset-Auswahl: maximale Typ-Diskriminierung
# ---------------------------------------------------------------------------
# Sortiert nach Type-Spread (max - min der Typ-Mittelwerte) aus der
# Checkpoint-Analyse mit 2189 Laeufen, 24 Datasets, 31 Schaetzern.
#
# | Dataset                    | TypeSpread | Warum ausgewaehlt?                      |
# |----------------------------|------------|-----------------------------------------|
# | synth_dnf_3x2              | 0.604      | Hoechste Diskriminierung, reines DNF    |
# | synth_xor_3bit             | 0.588      | XOR-Interaktion, grosse Typunterschiede |
# | mux_11                     | 0.525      | Boolesche Logik, mittlere Komplexitaet  |
# | synth_monk3                | 0.512      | Klassiker mit Rauschen                  |
# | synth_checkerboard_4x4     | 0.491      | Nicht-achsenparallele Grenzen           |
# | synth_overlap_4rules       | 0.479      | Ueberlappende Subgruppen                |
# | synth_highdim_p500_n120    | 0.418      | Hochdimensional, wenig Samples          |
# | synth_imbalanced_10pct     | 0.408      | Klassenungleichgewicht                  |
# | synth_epistasis_2way_easy  | 0.393      | Epistatische Interaktionen              |
# | uci_car_evaluation         | –          | Realer Multiclass-Datensatz (Paper-UCI) |
#
SLIM_DATASETS: list[str] = [
    # Synthetisch – Top Diskriminierung
    "synth_dnf_3x2",              # DNF-Konzept, Spread 0.604
    "synth_xor_3bit",             # XOR/Paritaet, Spread 0.588
    "mux_11",                     # 11-Bit Multiplexer, Spread 0.525
    "synth_monk3",                # MONK-3 + Rauschen, Spread 0.512
    "synth_checkerboard_4x4",     # Schachbrett, Spread 0.491
    "synth_overlap_4rules",       # Ueberlappende Regeln, Spread 0.479
    "synth_highdim_p500_n120",    # 500 Features / 120 Samples, Spread 0.418
    "synth_imbalanced_10pct",     # 10% Minoritaetsklasse, Spread 0.408
    "synth_epistasis_2way_easy",  # 2-Wege-Epistasie, Spread 0.393
    # Real-World
    "uci_car_evaluation",         # 4-Klassen, kategorial, Paper-UCI
]

# ---------------------------------------------------------------------------
# Estimator-Auswahl: ein starker Vertreter pro Typ-Familie
# ---------------------------------------------------------------------------
# Typ-Familie          | Vertreter                | Grund
# ---------------------|--------------------------|-------------------------------------
# tree (pruned)        | wrapper_cart_pruned       | Dominiert wrapper_cart strikt
# tree (HS)            | wrapper_hs_pruned         | Dominiert wrapper_hs strikt
# rulekit              | wrapper_rulekit           | Bester Allrounder (F1 0.723)
# rulefit              | wrapper_rulefit           | Solide (compact dominiert)
# exstracs             | exstracs_shrink_filter    | Bester F1 (0.778)
# logicgp (macro)      | logicgp_rlcw_macro        | *Bug-Fix*: jetzt mit Macro-F1
# logicgp (micro)      | logicgp_rlcw_micro        | *Bug-Fix*: jetzt mit Micro-F1
# pittsburgh           | pittsburgh_fast           | Ultra-kompakt, selten dominiert
# gp (base)            | gp                        | *Bug-Fix*: jetzt single_rule
# gp (residual)        | gp_residual               | *Bug-Fix*: residual_covering
# native               | native                    | Baseline-Referenz
#
SLIM_ESTIMATORS: list[str] = [
    # Tree-basiert
    "wrapper_cart_pruned",
    "wrapper_hs_pruned",
    # Externe Regelinduktion
    "wrapper_rulekit",
    "wrapper_rulefit",
    "wrapper_exstracs_shrink_filter",
    # LogicGP – Bug-Fix-Vergleich Macro vs Micro
    "wrapper_logicgp_rlcw_macro",
    "wrapper_logicgp_rlcw_micro",
    # Pittsburgh (kompakteste Familie)
    "wrapper_pittsburgh_fast",
    # GP nativ – Bug-Fix-Vergleich single_rule vs residual_covering
    "gp",
    "gp_residual",
    # Baseline
    "native",
]


# ---------------------------------------------------------------------------
# Hilfsfunktionen (identisch mit benchmark_full_report.py)
# ---------------------------------------------------------------------------

class _TeeStream:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


def _default_log_path() -> Path:
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("benchmarks") / "logs" / f"benchmark_slim_{ts}.log"


@contextlib.contextmanager
def _maybe_tee_to_file(log_file: Path | None):
    if log_file is None:
        yield
        return
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as f:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = _TeeStream(old_stdout, f)
        sys.stderr = _TeeStream(old_stderr, f)
        try:
            print(f"[LOG] writing console output to: {log_file}")
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def _csv_string(rows):
    if not rows:
        return ""
    import io, csv
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------

def main(
    log_file: Path | None = None,
    *,
    dataset_names: list[str] | None = None,
    estimator_names: list[str] | None = None,
    repeats: int = 5,
    timeout_seconds: float | None = 300.0,
    checkpoint_path: str | Path | None = "benchmarks/checkpoint_slim.jsonl",
):
    """Fuehrt den schlanken Benchmark aus und erzeugt alle Reports."""

    ds_names = dataset_names or SLIM_DATASETS
    est_names = estimator_names or SLIM_ESTIMATORS

    dn_display = ", ".join(ds_names)
    en_display = ", ".join(est_names)
    timeout_display = f"{timeout_seconds:.0f}s" if timeout_seconds else "deaktiviert"
    ckpt_display = str(checkpoint_path) if checkpoint_path else "deaktiviert"
    total_runs = len(ds_names) * len(est_names) * repeats

    print("=" * 70)
    print("SLIM BENCHMARK: Maximale Typ-Diskriminierung")
    print("=" * 70)
    print(f"  Datensaetze ({len(ds_names):2d}): {dn_display}")
    print(f"  Schaetzer   ({len(est_names):2d}): {en_display}")
    print(f"  Wiederholungen:    {repeats}")
    print(f"  Timeout pro Lauf:  {timeout_display}")
    print(f"  Checkpoint:        {ckpt_display}")
    print(f"  Gesamt-Laeufe:     {total_runs}")
    print("=" * 70)

    # Ausgabeverzeichnis
    out_dir = Path("benchmarks") / "slim"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = BenchmarkConfig(
        dataset_names=ds_names,
        estimator_names=est_names,
        use_paper_split_policy=True,
        include_online_uci=True,
        include_synthetic=True,
        include_pmlb=False,
        paper_uci_strict=False,
        repeats=repeats,
        random_state=42,
        show_progress=True,
        timeout_seconds=timeout_seconds,
        checkpoint_path=checkpoint_path,
    )

    # ------- Phase 1: Benchmarks ausfuehren -------
    print("\n[1/3] Fuehre Benchmarks aus...")
    t0 = time.time()
    results = run_benchmarks(config)
    t1 = time.time()
    print(f"Benchmarks abgeschlossen in {t1 - t0:.1f} Sekunden ({total_runs} Laeufe).")

    # ------- Phase 2: Aggregieren und Reports -------
    print("\n[2/3] Aggregiere und erstelle Reports...")
    payload = results_as_dicts(results)
    aggregated = aggregate_benchmark_results(results, error_bar="std")
    leaderboard = build_benchmark_leaderboard(aggregated)
    aggregated_payload = aggregated_results_as_dicts(aggregated)

    # CSV / JSON
    (out_dir / "benchmark_results.csv").write_text(
        _csv_string(payload), encoding="utf-8"
    )
    (out_dir / "benchmark_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (out_dir / "benchmark_results_aggregated.csv").write_text(
        _csv_string(aggregated_payload), encoding="utf-8"
    )
    (out_dir / "benchmark_results_aggregated.json").write_text(
        json.dumps(aggregated_payload, indent=2), encoding="utf-8"
    )

    # Plots
    base = out_dir / "benchmark_results"
    png_path, pdf_path = plot_benchmark_results(
        results, output_base=base, aggregate_repeats=True
    )
    heatmap_png, heatmap_pdf = plot_benchmark_heatmap(
        results, output_base=out_dir / "benchmark_results_heatmap"
    )
    combined_png, combined_pdf = plot_benchmark_heatmap_combined(
        results, output_base=out_dir / "benchmark_results_heatmap_combined"
    )
    dot_png, dot_pdf = plot_combined_dot(
        results, output_base=out_dir / "benchmark_results_combined_dot"
    )
    pareto_png, pareto_pdf = plot_pareto_front(
        results, output_base=out_dir / "benchmark_results_pareto"
    )

    # Markdown-Report
    md_report = format_benchmark_report_markdown(
        leaderboard,
        title="ScoredRuleSets Slim Benchmark – Maximale Typ-Diskriminierung",
        config={
            "datasets": dn_display,
            "estimators": en_display,
            "repeats": repeats,
            "timeout_seconds": timeout_display,
            "design": "10 Datasets mit hoechster Typ-Diskriminierung (TypeSpread ≥ 0.39), "
                      "11 Schaetzer (ein Vertreter pro Typ + Bug-Fix-Vergleiche)",
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
            "Schlanker Benchmark, optimiert auf maximale Unterscheidbarkeit der Schaetzer-Typen.",
            "Datensatz-Auswahl basiert auf TypeSpread-Analyse (Pareto-Checkpoint, 2189 Laeufe, 24 Datasets, 31 Schaetzer).",
            "Bug-Fix-Vergleiche: gp (single_rule) vs gp_residual (residual_covering); "
            "rlcw_macro (F1-macro) vs rlcw_micro (F1-micro).",
            f"Timeout pro Einzellauf: {timeout_display}.",
            f"{repeats} Wiederholungen mit random_state=42.",
        ],
    )
    (out_dir / "benchmark_report.md").write_text(md_report, encoding="utf-8")

    html_report = format_benchmark_report_html(
        leaderboard,
        title="ScoredRuleSets Slim Benchmark – Maximale Typ-Diskriminierung",
        config={
            "datasets": dn_display,
            "estimators": en_display,
            "repeats": repeats,
            "timeout_seconds": timeout_display,
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
            "Schlanker Benchmark, optimiert auf maximale Unterscheidbarkeit der Schaetzer-Typen.",
            "Bug-Fix-Vergleiche: gp vs gp_residual; rlcw_macro vs rlcw_micro.",
        ],
    )
    (out_dir / "benchmark_report.html").write_text(html_report, encoding="utf-8")

    # ------- Phase 3: Zusammenfassung -------
    print(f"\n[3/3] Fertig! Ergebnisse in: {out_dir}/")
    print("Wichtige Dateien:")
    print(f"  {out_dir}/benchmark_results.csv / .json (Rohdaten)")
    print(f"  {out_dir}/benchmark_results_aggregated.csv / .json (Aggregiert)")
    print(f"  {out_dir}/benchmark_report.md / .html (Report)")
    print(f"  {out_dir}/benchmark_results.png / .pdf (Plots)")
    print(f"  {out_dir}/benchmark_results_heatmap*.png / .pdf (Heatmaps)")
    print(f"  {out_dir}/benchmark_results_pareto.png / .pdf (Pareto-Front)")

    print("\nLeaderboard (Top 15):")
    print(format_benchmark_leaderboard_table(leaderboard)[:3000])

    if log_file is not None:
        print(f"\n[LOG] completed. Full log saved to: {log_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Schlanker Benchmark fuer maximale Schaetzer-Typ-Diskriminierung.",
        epilog=(
            "Design: 10 Datasets (hoechster TypeSpread), 11 Schaetzer "
            "(ein Vertreter pro Typ + Bug-Fix-Vergleiche)."
        ),
    )
    parser.add_argument(
        "--log-file", type=Path, default=None,
        help="Console-Log-Datei (default: auto-generiert).",
    )
    parser.add_argument(
        "--no-log", action="store_true",
        help="Kein Log-File schreiben.",
    )
    parser.add_argument(
        "--datasets", type=str, default="",
        help="Kommaseparierte Dataset-Liste (default: SLIM_DATASETS).",
    )
    parser.add_argument(
        "--estimators", type=str, default="",
        help="Kommaseparierte Schaetzer-Liste (default: SLIM_ESTIMATORS).",
    )
    parser.add_argument(
        "--repeats", type=int, default=5,
        help="Wiederholungen pro Kombination (default: 5).",
    )
    parser.add_argument(
        "--timeout", type=float, default=300.0,
        help="Timeout pro Einzellauf in Sekunden (default: 300). 0 = kein Timeout.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="benchmarks/checkpoint_slim.jsonl",
        help="Checkpoint-Datei fuer Resume (default: benchmarks/checkpoint_slim.jsonl).",
    )
    parser.add_argument(
        "--no-checkpoint", action="store_true",
        help="Checkpoint/Resume deaktivieren.",
    )
    args = parser.parse_args()

    selected_log_file: Path | None
    if args.no_log:
        selected_log_file = None
    elif args.log_file is not None:
        selected_log_file = args.log_file
    else:
        selected_log_file = _default_log_path()

    ds_names = [x.strip() for x in args.datasets.split(",") if x.strip()] or None
    est_names = [x.strip() for x in args.estimators.split(",") if x.strip()] or None
    timeout = args.timeout if args.timeout > 0 else None
    ckpt = None if args.no_checkpoint else args.checkpoint

    with _maybe_tee_to_file(selected_log_file):
        main(
            log_file=selected_log_file,
            dataset_names=ds_names,
            estimator_names=est_names,
            repeats=args.repeats,
            timeout_seconds=timeout,
            checkpoint_path=ckpt,
        )

