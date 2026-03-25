"""LogicGP-Varianten-Benchmark: Welche Konfigurationen sind redundant?

Ziel: Systematischer Vergleich aller LogicGP-Achsen, um Varianten zu
identifizieren, die zusammengefasst oder entfernt werden koennen.

Vergleichsachsen
----------------
1. **Trainer-Typ:**    FLCW  vs  RLCW  (Suchraum + Pareto-Dominanz)
2. **F1-Averaging:**   Macro vs Micro   (Fitness-Ziel)
3. **Budget:**         Standard (200/500 Gen.) vs Fast (50 Gen.)
4. **Literale:**       Full (alle Teilmengen) vs Singleton (nur Einzelwerte)
5. **Bins:**           3 vs 5 vs 7       (Diskretisierungsfeinheit)

Datensaetze:  8 synthetische + 1 realer (car_evaluation) aus dem Slim-Benchmark.
              Gewaehlt fuer hohe Typ-Diskriminierung bei LogicGP.

Insgesamt:  12 Varianten x 9 Datasets x 3 Repeats = **324 Laeufe**

Aufruf:
    python examples/benchmark_logicgp_variants.py
    python examples/benchmark_logicgp_variants.py --repeats 5
    python examples/benchmark_logicgp_variants.py --timeout 240
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import csv
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
# Dataset-Auswahl
# ---------------------------------------------------------------------------
# Datensaetze, die LogicGP-Varianten gut unterscheiden (aus der Slim-Analyse):
# - synth_dnf_3x2:           LogicGP excellent (F1=1.0), Achsenunterschiede sichtbar
# - synth_xor_3bit:          Interaktionsdatensatz, micro >> macro teils
# - mux_11:                  Skalierungsstress (macro timeout-anfaellig)
# - synth_monk3:             Klein, macro~=micro
# - synth_overlap_4rules:    Overlap, macro vs micro Unterschied
# - synth_checkerboard_4x4:  Schwer fuer LogicGP, Skalierungsstress
# - synth_imbalanced_10pct:  Klassenungleichgewicht
# - synth_epistasis_2way:    Epistatisch, moderate Schwierigkeit
# - uci_car_evaluation:      Real, 4-Klassen, LogicGP eher schwach
LOGICGP_DATASETS: list[str] = [
    "synth_dnf_3x2",
    "synth_xor_3bit",
    "mux_11",
    "synth_monk3",
    "synth_overlap_4rules",
    "synth_checkerboard_4x4",
    "synth_imbalanced_10pct",
    "synth_epistasis_2way_easy",
    "uci_car_evaluation",
]

# ---------------------------------------------------------------------------
# LogicGP-Varianten: systematische Abdeckung aller Achsen
# ---------------------------------------------------------------------------
#
# Die Varianten sind so gewaehlt, dass sich jeweils genau eine Achse
# aendert und per Paarvergleich eine Entscheidung getroffen werden kann.
#
# | Name                       | Trainer    | Avg   | Gens | Stag | Pop | Adapt | Bins | Lits     | min_mw | Budget |
# |----------------------------|------------|-------|------|------|-----|-------|------|----------|--------|--------|
# | lgp_rlcw_macro             | rlcw_macro | macro | 200  | 40   | 50  | 12    | 5    | full     | 0.10   | 240s   |
# | lgp_rlcw_micro             | rlcw_micro | micro | 200  | 40   | 50  | 12    | 5    | full     | 0.10   | 240s   |
# | lgp_flcw_macro             | flcw_macro | macro | 200  | 40   | 50  | 8     | 5    | full     | 0.00   | 240s   |
# | lgp_flcw_micro             | flcw_micro | micro | 200  | 40   | 50  | 8     | 5    | full     | 0.00   | 240s   |
# | lgp_rlcw_macro_fast        | rlcw_macro | macro |  50  | 15   | 30  | 12    | 5    | full     | 0.10   | 240s   |
# | lgp_rlcw_macro_singleton   | rlcw_macro | macro | 200  | 40   | 50  | 12    | 5    | singleton| 0.10   | 240s   |
# | lgp_rlcw_macro_bins3       | rlcw_macro | macro | 200  | 40   | 50  | 12    | 3    | full     | 0.10   | 240s   |
# | lgp_rlcw_macro_bins7       | rlcw_macro | macro | 200  | 40   | 50  | 12    | 7    | full     | 0.10   | 240s   |
# | lgp_flcw_macro_fast        | flcw_macro | macro |  50  | 15   | 30  | 8     | 5    | full     | 0.00   | 240s   |
# | lgp_rlcw_macro_nofilt      | rlcw_macro | macro | 200  | 40   | 50  | 12    | 5    | full     | 0.00   | 240s   |
# | lgp_rlcw_macro_bigpop      | rlcw_macro | macro | 200  | 40   | 80  | 12    | 5    | full     | 0.10   | 240s   |
# | lgp_rlcw_macro_strongfilt  | rlcw_macro | macro | 200  | 40   | 50  | 12    | 5    | full     | 0.25   | 240s   |
#
# Paarvergleiche:
#   rlcw_macro vs rlcw_micro            -> Achse: F1-Averaging
#   rlcw_macro vs flcw_macro            -> Achse: Trainer-Typ (RLCW vs FLCW)
#   flcw_macro vs flcw_micro            -> Achse: F1-Averaging bei FLCW
#   rlcw_macro vs rlcw_macro_fast       -> Achse: Budget (viele vs wenige Generationen)
#   flcw_macro vs flcw_macro_fast       -> Achse: Budget bei FLCW
#   rlcw_macro vs rlcw_macro_singleton  -> Achse: Literal-Generator
#   rlcw_macro vs rlcw_macro_bins3      -> Achse: Bins  (5 vs 3)
#   rlcw_macro vs rlcw_macro_bins7      -> Achse: Bins  (5 vs 7)
#   rlcw_macro vs rlcw_macro_nofilt     -> Achse: min_max_weight  (0.1 vs 0.0)
#   rlcw_macro vs rlcw_macro_bigpop     -> Achse: Populationsgroesse (50 vs 80)
#   rlcw_macro vs rlcw_macro_strongfilt -> Achse: min_max_weight  (0.1 vs 0.25)

LOGICGP_ESTIMATORS: list[str] = [
    "lgp_rlcw_macro",
    "lgp_rlcw_micro",
    "lgp_flcw_macro",
    "lgp_flcw_micro",
    "lgp_rlcw_macro_fast",
    "lgp_rlcw_macro_singleton",
    "lgp_rlcw_macro_bins3",
    "lgp_rlcw_macro_bins7",
    "lgp_flcw_macro_fast",
    "lgp_rlcw_macro_nofilt",
    "lgp_rlcw_macro_bigpop",
    "lgp_rlcw_macro_strongfilt",
]


# ---------------------------------------------------------------------------
# Registrierung der Varianten als EstimatorSpecs
# ---------------------------------------------------------------------------

def _register_logicgp_estimators():
    """Registriert die LogicGP-Varianten im globalen Estimator-Registry."""
    from scoredrulesets.benchmarking.estimators import EstimatorSpec, ESTIMATOR_SPECS
    from scoredrulesets.estimators import ScoredRuleSetClassifier

    _TIME_BUDGET = 240  # Sekunden, unter dem 300s-Benchmark-Timeout

    specs = {
        # --- Basis: RLCW Macro (Referenz) ---
        "lgp_rlcw_macro": dict(
            trainer="rlcw", f1_averaging="macro", max_generations=200, stagnation_generations=40,
            population_size=50, n_adaptations_per_gen=12, n_bins=5,
            min_max_weight=0.1, max_fit_seconds=_TIME_BUDGET,
            literal_generator="full", random_state=0,
        ),
        # --- Achse: F1-Averaging ---
        "lgp_rlcw_micro": dict(
            trainer="rlcw", f1_averaging="micro", max_generations=200, stagnation_generations=40,
            population_size=50, n_adaptations_per_gen=12, n_bins=5,
            min_max_weight=0.1, max_fit_seconds=_TIME_BUDGET,
            literal_generator="full", random_state=0,
        ),
        # --- Achse: Trainer-Typ (FLCW vs RLCW) ---
        "lgp_flcw_macro": dict(
            trainer="flcw", f1_averaging="macro", max_generations=200, stagnation_generations=40,
            population_size=50, n_adaptations_per_gen=8, n_bins=5,
            min_max_weight=0.0, max_fit_seconds=_TIME_BUDGET,
            literal_generator="full", random_state=0,
        ),
        "lgp_flcw_micro": dict(
            trainer="flcw", f1_averaging="micro", max_generations=200, stagnation_generations=40,
            population_size=50, n_adaptations_per_gen=8, n_bins=5,
            min_max_weight=0.0, max_fit_seconds=_TIME_BUDGET,
            literal_generator="full", random_state=0,
        ),
        # --- Achse: Budget (Fast) ---
        "lgp_rlcw_macro_fast": dict(
            trainer="rlcw", f1_averaging="macro", max_generations=50, stagnation_generations=15,
            population_size=30, n_adaptations_per_gen=12, n_bins=5,
            min_max_weight=0.1, max_fit_seconds=_TIME_BUDGET,
            literal_generator="full", random_state=0,
        ),
        "lgp_flcw_macro_fast": dict(
            trainer="flcw", f1_averaging="macro", max_generations=50, stagnation_generations=15,
            population_size=30, n_adaptations_per_gen=8, n_bins=5,
            min_max_weight=0.0, max_fit_seconds=_TIME_BUDGET,
            literal_generator="full", random_state=0,
        ),
        # --- Achse: Literal-Generator ---
        "lgp_rlcw_macro_singleton": dict(
            trainer="rlcw", f1_averaging="macro", max_generations=200, stagnation_generations=40,
            population_size=50, n_adaptations_per_gen=12, n_bins=5,
            min_max_weight=0.1, max_fit_seconds=_TIME_BUDGET,
            literal_generator="singleton", random_state=0,
        ),
        # --- Achse: Bins ---
        "lgp_rlcw_macro_bins3": dict(
            trainer="rlcw", f1_averaging="macro", max_generations=200, stagnation_generations=40,
            population_size=50, n_adaptations_per_gen=12, n_bins=3,
            min_max_weight=0.1, max_fit_seconds=_TIME_BUDGET,
            literal_generator="full", random_state=0,
        ),
        "lgp_rlcw_macro_bins7": dict(
            trainer="rlcw", f1_averaging="macro", max_generations=200, stagnation_generations=40,
            population_size=50, n_adaptations_per_gen=12, n_bins=7,
            min_max_weight=0.1, max_fit_seconds=_TIME_BUDGET,
            literal_generator="full", random_state=0,
        ),
        # --- Achse: min_max_weight (Literal-Filterung) ---
        "lgp_rlcw_macro_nofilt": dict(
            trainer="rlcw", f1_averaging="macro", max_generations=200, stagnation_generations=40,
            population_size=50, n_adaptations_per_gen=12, n_bins=5,
            min_max_weight=0.0, max_fit_seconds=_TIME_BUDGET,
            literal_generator="full", random_state=0,
        ),
        "lgp_rlcw_macro_strongfilt": dict(
            trainer="rlcw", f1_averaging="macro", max_generations=200, stagnation_generations=40,
            population_size=50, n_adaptations_per_gen=12, n_bins=5,
            min_max_weight=0.25, max_fit_seconds=_TIME_BUDGET,
            literal_generator="full", random_state=0,
        ),
        # --- Achse: Populationsgroesse ---
        "lgp_rlcw_macro_bigpop": dict(
            trainer="rlcw", f1_averaging="macro", max_generations=200, stagnation_generations=40,
            population_size=80, n_adaptations_per_gen=12, n_bins=5,
            min_max_weight=0.1, max_fit_seconds=_TIME_BUDGET,
            literal_generator="full", random_state=0,
        ),
    }

    for name, params in specs.items():
        if name not in ESTIMATOR_SPECS:
            ESTIMATOR_SPECS[name] = EstimatorSpec(
                name=name,
                factory=lambda p=params: ScoredRuleSetClassifier(
                    backend="logicgp",
                    backend_params=dict(p),
                    random_state=0,
                ),
            )


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
    return Path("benchmarks") / "logs" / f"benchmark_logicgp_{ts}.log"


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
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Analyse: automatische Paarvergleiche
# ---------------------------------------------------------------------------

def _run_pairwise_analysis(payload: list[dict], out_dir: Path):
    """Fuehrt automatische Paarvergleiche durch und schreibt Analyse-Report.

    Fuer jedes Vergleichspaar wird pro Datensatz der Mittelwert verglichen.
    Ergebnis: Empfehlung pro Achse (behalten / zusammenfassen / entfernen).
    """
    from collections import defaultdict
    from statistics import mean

    PAIRS = [
        # (name_a, name_b, achse, beschreibung)
        ("lgp_rlcw_macro", "lgp_rlcw_micro", "F1-Averaging (RLCW)", "Macro vs Micro bei RLCW"),
        ("lgp_flcw_macro", "lgp_flcw_micro", "F1-Averaging (FLCW)", "Macro vs Micro bei FLCW"),
        ("lgp_rlcw_macro", "lgp_flcw_macro", "Trainer-Typ (Macro)", "RLCW vs FLCW bei Macro"),
        ("lgp_rlcw_micro", "lgp_flcw_micro", "Trainer-Typ (Micro)", "RLCW vs FLCW bei Micro"),
        ("lgp_rlcw_macro", "lgp_rlcw_macro_fast", "Budget (RLCW)", "Standard vs Fast bei RLCW"),
        ("lgp_flcw_macro", "lgp_flcw_macro_fast", "Budget (FLCW)", "Standard vs Fast bei FLCW"),
        ("lgp_rlcw_macro", "lgp_rlcw_macro_singleton", "Literal-Generator", "Full vs Singleton"),
        ("lgp_rlcw_macro", "lgp_rlcw_macro_bins3", "Bins (5 vs 3)", "n_bins=5 vs n_bins=3"),
        ("lgp_rlcw_macro", "lgp_rlcw_macro_bins7", "Bins (5 vs 7)", "n_bins=5 vs n_bins=7"),
        ("lgp_rlcw_macro", "lgp_rlcw_macro_nofilt", "Literal-Filter (0.1 vs 0)", "min_max_weight 0.1 vs 0.0"),
        ("lgp_rlcw_macro", "lgp_rlcw_macro_strongfilt", "Literal-Filter (0.1 vs 0.25)", "min_max_weight 0.1 vs 0.25"),
        ("lgp_rlcw_macro", "lgp_rlcw_macro_bigpop", "Populationsgroesse", "pop=50 vs pop=80"),
    ]

    # Gruppiere Ergebnisse: (dataset, estimator) -> [records]
    groups = defaultdict(list)
    for r in payload:
        if r.get("status") == "ok" and r.get("f1_macro") is not None:
            groups[(r["dataset"], r["estimator"])].append(r)

    lines = []
    lines.append("# LogicGP Varianten -- Paarvergleich-Analyse\n")
    lines.append(f"Generiert: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    lines.append("Fuer jedes Vergleichspaar wird pro Datensatz der Mittelwert (ueber Repeats) verglichen.\n")
    lines.append("**Legende:** A > B = A hat hoehere mittlere F1; Delta = A-F1 minus B-F1; t = Avg Fit-Sekunden\n")

    summary_rows = []

    for name_a, name_b, achse, beschreibung in PAIRS:
        lines.append(f"\n## {achse}: `{name_a}` vs `{name_b}`")
        lines.append(f"_{beschreibung}_\n")
        lines.append(f"| Dataset | {name_a} F1 | {name_b} F1 | Delta F1 | {name_a} Atoms | {name_b} Atoms | {name_a} Fit(s) | {name_b} Fit(s) | Besser |")
        lines.append("|---------|------------|------------|------|---------------|---------------|----------------|----------------|--------|")

        wins_a, wins_b, ties = 0, 0, 0
        all_delta_f1 = []
        all_delta_time = []

        for ds in LOGICGP_DATASETS:
            recs_a = groups.get((ds, name_a), [])
            recs_b = groups.get((ds, name_b), [])

            if not recs_a and not recs_b:
                lines.append(f"| {ds} | -- | -- | -- | -- | -- | -- | -- | -- |")
                continue

            f1_a = mean([r["f1_macro"] for r in recs_a]) if recs_a else float("nan")
            f1_b = mean([r["f1_macro"] for r in recs_b]) if recs_b else float("nan")
            at_a = mean([r["n_atoms"] for r in recs_a]) if recs_a else float("nan")
            at_b = mean([r["n_atoms"] for r in recs_b]) if recs_b else float("nan")
            t_a = mean([r["fit_seconds"] for r in recs_a]) if recs_a else float("nan")
            t_b = mean([r["fit_seconds"] for r in recs_b]) if recs_b else float("nan")

            if not recs_a:
                label = f"**{name_b}** (A fehlt)"
                wins_b += 1
            elif not recs_b:
                label = f"**{name_a}** (B fehlt)"
                wins_a += 1
            else:
                delta = f1_a - f1_b
                all_delta_f1.append(delta)
                all_delta_time.append(t_a - t_b)
                if abs(delta) < 0.01:
                    label = "~="
                    ties += 1
                elif delta > 0:
                    label = f"**{name_a}**"
                    wins_a += 1
                else:
                    label = f"**{name_b}**"
                    wins_b += 1

            def _fmt(v):
                if v != v:  # NaN check
                    return "--"
                return f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}"

            delta_str = _fmt(f1_a - f1_b) if recs_a and recs_b else "--"
            lines.append(
                f"| {ds} | {_fmt(f1_a)} | {_fmt(f1_b)} | {delta_str} | "
                f"{_fmt(at_a)} | {_fmt(at_b)} | {_fmt(t_a)} | {_fmt(t_b)} | {label} |"
            )

        # Zusammenfassung
        mean_delta_f1 = mean(all_delta_f1) if all_delta_f1 else 0
        mean_delta_time = mean(all_delta_time) if all_delta_time else 0
        total = wins_a + wins_b + ties
        lines.append(f"\n**Ergebnis:** {name_a} gewinnt {wins_a}x, {name_b} gewinnt {wins_b}x, ~= {ties}x (Delta<0.01)")
        lines.append(f"**Avg Delta F1:** {mean_delta_f1:+.4f}   **Avg Delta Fit-Zeit:** {mean_delta_time:+.1f}s")

        if total > 0:
            if wins_a >= 2 * wins_b and mean_delta_f1 > 0.01:
                emp = f"-> **{name_a}** klar besser. `{name_b}` ist Kandidat fuer Entfernung."
            elif wins_b >= 2 * wins_a and mean_delta_f1 < -0.01:
                emp = f"-> **{name_b}** klar besser. `{name_a}` ist Kandidat fuer Entfernung."
            elif abs(mean_delta_f1) < 0.02 and abs(mean_delta_time) < 10:
                emp = f"-> Kein klarer Unterschied. Varianten koennen zusammengefasst werden."
            else:
                emp = f"-> Situationsabhaengig. Beide behalten oder nach Anwendungsfall waehlen."
        else:
            emp = "-> Keine Daten fuer Vergleich."
        lines.append(f"\n**Empfehlung:** {emp}\n")

        summary_rows.append({
            "achse": achse,
            "name_a": name_a,
            "name_b": name_b,
            "wins_a": wins_a,
            "wins_b": wins_b,
            "ties": ties,
            "mean_delta_f1": round(mean_delta_f1, 4),
            "mean_delta_time_s": round(mean_delta_time, 1),
            "empfehlung": emp.replace("-> ", "").strip("*"),
        })

    # Gesamtuebersicht
    lines.append("\n---\n")
    lines.append("# Gesamtuebersicht\n")
    lines.append("| Achse | A | B | A-Wins | B-Wins | Ties | Avg Delta F1 | Avg Delta t(s) | Empfehlung |")
    lines.append("|-------|---|---|--------|--------|------|--------|----------|------------|")
    for r in summary_rows:
        lines.append(
            f"| {r['achse']} | {r['name_a']} | {r['name_b']} | {r['wins_a']} | "
            f"{r['wins_b']} | {r['ties']} | {r['mean_delta_f1']:+.4f} | "
            f"{r['mean_delta_time_s']:+.1f} | {r['empfehlung'][:60]} |"
        )

    report = "\n".join(lines) + "\n"
    (out_dir / "logicgp_pairwise_analysis.md").write_text(report, encoding="utf-8")
    (out_dir / "logicgp_pairwise_summary.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  Paarvergleich-Analyse: {out_dir}/logicgp_pairwise_analysis.md")
    print(f"  Paarvergleich-Summary: {out_dir}/logicgp_pairwise_summary.json")
    return summary_rows


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
    checkpoint_path: str | Path | None = "benchmarks/checkpoint_logicgp.jsonl",
):
    """Fuehrt den LogicGP-Varianten-Benchmark aus und erzeugt Reports + Analyse."""

    # Registriere die LogicGP-Varianten
    _register_logicgp_estimators()

    ds_names = dataset_names or LOGICGP_DATASETS
    est_names = estimator_names or LOGICGP_ESTIMATORS

    dn_display = ", ".join(ds_names)
    en_display = ", ".join(est_names)
    timeout_display = f"{timeout_seconds:.0f}s" if timeout_seconds else "deaktiviert"
    ckpt_display = str(checkpoint_path) if checkpoint_path else "deaktiviert"
    total_runs = len(ds_names) * len(est_names) * repeats

    print("=" * 72)
    print("LOGICGP VARIANTEN-BENCHMARK")
    print("Ziel: Varianten reduzieren / zusammenfassen")
    print("=" * 72)
    print(f"  Datensaetze   ({len(ds_names):2d}): {dn_display}")
    print(f"  Varianten     ({len(est_names):2d}): {en_display}")
    print(f"  Wiederholungen:      {repeats}")
    print(f"  Timeout pro Lauf:    {timeout_display}")
    print(f"  Checkpoint:          {ckpt_display}")
    print(f"  Gesamt-Laeufe:       {total_runs}")
    print("=" * 72)

    out_dir = Path("benchmarks") / "logicgp-variants"
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
    print("\n[1/4] Fuehre Benchmarks aus...")
    t0 = time.time()
    results = run_benchmarks(config)
    t1 = time.time()
    print(f"Benchmarks abgeschlossen in {t1 - t0:.1f}s ({total_runs} Laeufe).")

    # ------- Phase 2: Aggregieren und Reports -------
    print("\n[2/4] Aggregiere und erstelle Reports...")
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
        title="LogicGP Varianten-Benchmark -- Konsolidierungsanalyse",
        config={
            "datasets": dn_display,
            "estimators": en_display,
            "repeats": repeats,
            "timeout_seconds": timeout_display,
            "design": "12 LogicGP-Varianten ueber 5 Vergleichsachsen "
                      "(Trainer, F1-Avg, Budget, Literale, Bins, Filter, Population) "
                      f"auf {len(ds_names)} Datensaetzen.",
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
            "LogicGP-Varianten-Benchmark zur Konsolidierung.",
            "12 Varianten ueber 5 Achsen: Trainer (FLCW/RLCW), F1-Averaging (Macro/Micro), "
            "Budget (Standard/Fast), Literal-Generator (Full/Singleton), Bins (3/5/7), "
            "Literal-Filter (0/0.1/0.25), Populationsgroesse (50/80).",
            f"Alle Varianten mit max_fit_seconds=240 (unter {timeout_display} Timeout).",
            f"{repeats} Wiederholungen mit random_state=42.",
        ],
    )
    (out_dir / "benchmark_report.md").write_text(md_report, encoding="utf-8")

    html_report = format_benchmark_report_html(
        leaderboard,
        title="LogicGP Varianten-Benchmark -- Konsolidierungsanalyse",
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
            "LogicGP-Varianten-Benchmark zur Konsolidierung.",
            "12 Varianten ueber 5 Achsen.",
        ],
    )
    (out_dir / "benchmark_report.html").write_text(html_report, encoding="utf-8")

    # ------- Phase 3: Paarvergleich-Analyse -------
    print("\n[3/4] Fuehre Paarvergleich-Analyse durch...")
    _run_pairwise_analysis(payload, out_dir)

    # ------- Phase 4: Zusammenfassung -------
    print(f"\n[4/4] Fertig! Ergebnisse in: {out_dir}/")
    print("Wichtige Dateien:")
    print(f"  {out_dir}/benchmark_results.csv / .json          (Rohdaten)")
    print(f"  {out_dir}/benchmark_results_aggregated.csv / .json (Aggregiert)")
    print(f"  {out_dir}/benchmark_report.md / .html             (Report)")
    print(f"  {out_dir}/logicgp_pairwise_analysis.md            (Achsen-Analyse)")
    print(f"  {out_dir}/logicgp_pairwise_summary.json           (Maschinen-lesbar)")
    print(f"  {out_dir}/benchmark_results_pareto.png / .pdf     (Pareto-Front)")

    print("\nLeaderboard (alle):")
    print(format_benchmark_leaderboard_table(leaderboard)[:4000])

    if log_file is not None:
        print(f"\n[LOG] completed. Full log saved to: {log_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LogicGP-Varianten-Benchmark zur Konsolidierung.",
        epilog=(
            "Vergleicht 12 LogicGP-Varianten ueber 5 Achsen (Trainer, F1-Avg, "
            "Budget, Literale, Bins, Filter, Population) auf 9 Datensaetzen."
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
        help="Kommaseparierte Dataset-Liste (default: LOGICGP_DATASETS).",
    )
    parser.add_argument(
        "--estimators", type=str, default="",
        help="Kommaseparierte Varianten-Liste (default: LOGICGP_ESTIMATORS).",
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
        "--checkpoint", type=str, default="benchmarks/checkpoint_logicgp.jsonl",
        help="Checkpoint-Datei fuer Resume (default: benchmarks/checkpoint_logicgp.jsonl).",
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

