"""Standard-Benchmark: Alle Schaetzer-Varianten auf ausgewaehlten Datensaetzen.

Kombination aus dem vollen Schaetzer-Katalog (wie ``benchmark_full_report``)
und einer fokussierten Datensatz-Auswahl.

- **Estimators:** Alle aus ``default_estimator_specs()`` abzueglich
  MUX-spezifischer Varianten (identisch mit dem Full-Benchmark).
- **Datasets:** 10 Datensaetze mit hoechster Typ-Diskriminierung.

Damit koennen alle Schaetzer-Varianten direkt verglichen werden,
ohne die Laufzeit des Full-Benchmarks (mit allen Datensaetzen) in Kauf
zu nehmen.

Insgesamt: ~N Schaetzer x 10 Datasets x 3 Repeats
(statt N x alle Datasets im Full-Benchmark).

Produziert die gleichen Reports wie der volle Benchmark:
  CSV, JSON, Aggregationen, Heatmaps, Pareto-Front, Leaderboard (MD + HTML).

Aufruf:
    python examples/benchmark_standard.py
    python examples/benchmark_standard.py --repeats 5
    python examples/benchmark_standard.py --timeout 120
    python examples/benchmark_standard.py --estimators wrapper_cart,gp,wrapper_rulegp
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
from scoredrulesets.benchmarking.estimators import default_estimator_specs
from scoredrulesets.benchmarking.runner import results_as_dicts

# ---------------------------------------------------------------------------
# Dataset-Auswahl (fokussierte Auswahl mit hoher Typ-Diskriminierung)
# ---------------------------------------------------------------------------
# 10 Datensaetze mit hoechster Typ-Diskriminierung (TypeSpread >= 0.39),
# plus ein realer UCI-Multiclass-Datensatz.
STANDARD_DATASETS: list[str] = [
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
# Estimator-Auswahl: Alle aus default_estimator_specs() ohne MUX-Varianten
# ---------------------------------------------------------------------------
# MUX-spezifische Varianten werden ausgeschlossen, da sie nur auf
# Multiplexer-Datensaetzen sinnvoll sind und das allgemeine Ranking verzerren.
_MUX_ONLY: set[str] = {
    "wrapper_logicgp_mux",
    "wrapper_logicgp_mux_rlcw",
    "wrapper_logicgp_mux_rlcw_macro",
    "wrapper_cart_mux",
}


def _standard_estimator_names() -> list[str]:
    """Alle registrierten Schaetzer ohne MUX-spezifische Varianten."""
    return [e for e in default_estimator_specs() if e not in _MUX_ONLY]


# ---------------------------------------------------------------------------
# Hilfsfunktionen
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
    return Path("benchmarks") / "logs" / f"benchmark_standard_{ts}.log"


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
    repeats: int = 3,
    timeout_seconds: float | None = 300.0,
    checkpoint_path: str | Path | None = "benchmarks/checkpoint_standard.jsonl",
    output_dir: str | Path = "benchmarks/standard",
    console_title: str = "STANDARD BENCHMARK: Alle Schaetzer, ausgewaehlte Datensaetze",
    report_title: str = "ScoredRuleSets Standard Benchmark – Alle Schaetzer, ausgewaehlte Datensaetze",
):
    """Fuehrt den Standard-Benchmark aus und erzeugt alle Reports."""

    ds_names = dataset_names or STANDARD_DATASETS
    est_names = estimator_names or _standard_estimator_names()

    dn_display = ", ".join(ds_names)
    en_display = ", ".join(est_names)
    timeout_display = f"{timeout_seconds:.0f}s" if timeout_seconds else "deaktiviert"
    ckpt_display = str(checkpoint_path) if checkpoint_path else "deaktiviert"
    total_runs = len(ds_names) * len(est_names) * repeats

    print("=" * 70)
    print(console_title)
    print("=" * 70)
    print(f"  Datensaetze ({len(ds_names):2d}): {dn_display}")
    print(f"  Schaetzer   ({len(est_names):2d}): {en_display}")
    print(f"  Wiederholungen:    {repeats}")
    print(f"  Timeout pro Lauf:  {timeout_display}")
    print(f"  Checkpoint:        {ckpt_display}")
    print(f"  Gesamt-Laeufe:     {total_runs}")
    print("=" * 70)

    # Ausgabeverzeichnis
    out_dir = Path(output_dir)
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
        title=report_title,
        config={
            "datasets": dn_display,
            "estimators": en_display,
            "repeats": repeats,
            "timeout_seconds": timeout_display,
            "design": f"{len(est_names)} Schaetzer (alle ohne MUX-Varianten) x "
                      f"{len(ds_names)} Datensaetze (hoechste Typ-Diskriminierung)",
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
            "Standard-Benchmark: alle Schaetzer-Varianten auf ausgewaehlten Datensaetzen.",
            "Datensatz-Auswahl basiert auf TypeSpread-Analyse (Pareto-Checkpoint, "
            "2189 Laeufe, 24 Datasets, 31 Schaetzer).",
            "Schaetzer-Auswahl: vollstaendiger Katalog aus default_estimator_specs() "
            "ohne MUX-spezifische Varianten.",
            f"Timeout pro Einzellauf: {timeout_display}.",
            f"{repeats} Wiederholungen mit random_state=42.",
        ],
    )
    (out_dir / "benchmark_report.md").write_text(md_report, encoding="utf-8")

    html_report = format_benchmark_report_html(
        leaderboard,
        title=report_title,
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
            "Standard-Benchmark: alle Schaetzer-Varianten, ausgewaehlte Datensaetze.",
            f"Timeout pro Einzellauf: {timeout_display}.",
            f"{repeats} Wiederholungen.",
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
        description="Standard-Benchmark: alle Schaetzer auf ausgewaehlten Datensaetzen.",
        epilog=(
            "Design: 10 Datasets (hoechster TypeSpread), alle Schaetzer "
            "(ohne MUX-Varianten) aus default_estimator_specs()."
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
        help="Kommaseparierte Dataset-Liste (default: STANDARD_DATASETS).",
    )
    parser.add_argument(
        "--estimators", type=str, default="",
        help="Kommaseparierte Schaetzer-Liste (default: alle ohne MUX).",
    )
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="Wiederholungen pro Kombination (default: 3).",
    )
    parser.add_argument(
        "--timeout", type=float, default=300.0,
        help="Timeout pro Einzellauf in Sekunden (default: 300). 0 = kein Timeout.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="benchmarks/checkpoint_standard.jsonl",
        help="Checkpoint-Datei fuer Resume (default: benchmarks/checkpoint_standard.jsonl).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="benchmarks/standard",
        help="Ausgabeverzeichnis fuer Reports/Artefakte (default: benchmarks/standard).",
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
            output_dir=args.output_dir,
        )

