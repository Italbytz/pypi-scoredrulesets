#!/usr/bin/env python -u
"""
ExSTraCS Shrinking-Varianten Benchmark
=======================================

Systematischer Vergleich **aller** ExSTraCS-Shrinking-Strategien und ihrer
Kombinationen auf mehreren Datensätzen.

Ziel:
    Identifizieren, welche Varianten redundant sind oder zusammengefasst
    werden können, um die Anzahl der Estimator-Specs in
    ``benchmarking/estimators.py`` zu reduzieren.

Gemessene Metriken pro Variante × Dataset:
    - F1-macro (Test)
    - F1-macro (Train)
    - Anzahl Regeln (ohne Default)
    - Anzahl Atome
    - Laufzeit (fit + shrinking)

Ausgabe:
    - Konsolen-Tabelle mit Ergebnissen
    - CSV-Datei  (benchmarks/exstracs_shrinking_benchmark.csv)
    - Paarweise Redundanz-Analyse
    - Zusammenfassungs-Empfehlungen

Aufruf:
    python tools/analysis/benchmark_exstracs_shrinking.py
    python tools/analysis/benchmark_exstracs_shrinking.py --repeats 5
    python tools/analysis/benchmark_exstracs_shrinking.py --datasets sklearn_iris,sklearn_wine
    python tools/analysis/benchmark_exstracs_shrinking.py --quick   # nur sklearn-Datensätze, 1 Repeat
"""
from __future__ import annotations

import argparse
import csv
import io
import itertools
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

from scoredrulesets.estimators.sklearn_wrapper import ScoredRuleSetClassifier

# ---------------------------------------------------------------------------
# Datensätze
# ---------------------------------------------------------------------------

SKLEARN_DATASETS: list[tuple[str, object]] = [
    ("sklearn_iris", load_iris),
    ("sklearn_wine", load_wine),
    ("sklearn_breast_cancer", load_breast_cancer),
]


def _load_datasets(
    names: list[str] | None,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Lade Datensätze.  Ohne explizite Auswahl → alle sklearn-Datensätze."""
    if names is not None:
        # Versuche zuerst aus sklearn_datasets
        sklearn_map = {n: loader for n, loader in SKLEARN_DATASETS}
        selected = []
        for name in names:
            if name in sklearn_map:
                X, y = sklearn_map[name](return_X_y=True)
                selected.append((name, np.asarray(X), np.asarray(y)))
            else:
                # Lade über die allgemeine Dataset-Registry
                try:
                    from scoredrulesets.benchmarking.datasets import load_dataset_registry
                    registry = load_dataset_registry(
                        include_online_uci=True,
                        include_synthetic=True,
                        include_pmlb=False,
                    )
                    if name in registry:
                        b = registry[name]
                        selected.append((name, np.asarray(b.X), np.asarray(b.y)))
                    else:
                        print(f"[WARN] Datensatz '{name}' nicht gefunden – überspringe.")
                except Exception as e:
                    print(f"[WARN] Fehler beim Laden von '{name}': {e}")
        return selected

    return [
        (name, *loader(return_X_y=True))
        for name, loader in SKLEARN_DATASETS
    ]


# ---------------------------------------------------------------------------
# Shrinking-Varianten (alle sinnvollen Einzelstrategien + Kombinationen)
# ---------------------------------------------------------------------------

# Aufbau: (Name, exstracs_params_dict | None, Beschreibung)
SHRINKING_VARIANTS: list[tuple[str, dict | None, str]] = [
    # ── Baseline ──
    ("baseline", None, "Kein Shrinking"),

    # ── Einzelstrategien ──
    ("filter_p20", {
        "filter_weak_rules": True,
        "min_fitness_percentile": 0.2,
    }, "Filter schwache Regeln (top 80%)"),

    ("filter_p40", {
        "filter_weak_rules": True,
        "min_fitness_percentile": 0.4,
    }, "Filter schwache Regeln (top 60%)"),

    ("conservative", {
        "conservative_prune": True,
    }, "Conservative Pruning"),

    ("aggressive_1pct", {
        "aggressive_prune": True,
        "max_f1_loss": 0.01,
    }, "Aggressive Pruning (1% loss)"),

    ("aggressive_2pct", {
        "aggressive_prune": True,
        "max_f1_loss": 0.02,
    }, "Aggressive Pruning (2% loss)"),

    ("consolidate", {
        "consolidate_similar": True,
        "similarity_threshold": 0.8,
    }, "Consolidate Similar Rules"),

    ("imerge_0.3", {
        "interval_merge": True,
        "interval_merge_iou_threshold": 0.3,
    }, "Interval-Merge (IoU ≥ 0.3)"),

    ("imerge_0.1", {
        "interval_merge": True,
        "interval_merge_iou_threshold": 0.1,
    }, "Interval-Merge (IoU ≥ 0.1)"),

    # ── Sinnvolle 2er-Kombinationen ──
    ("filter+conservative", {
        "filter_weak_rules": True,
        "min_fitness_percentile": 0.2,
        "conservative_prune": True,
    }, "Filter + Conservative"),

    ("filter+aggressive", {
        "filter_weak_rules": True,
        "min_fitness_percentile": 0.2,
        "aggressive_prune": True,
        "max_f1_loss": 0.01,
    }, "Filter + Aggressive (1%)"),

    ("imerge+conservative", {
        "interval_merge": True,
        "interval_merge_iou_threshold": 0.3,
        "conservative_prune": True,
    }, "IMerge + Conservative"),

    ("imerge+consolidate", {
        "interval_merge": True,
        "interval_merge_iou_threshold": 0.3,
        "consolidate_similar": True,
    }, "IMerge + Consolidate"),

    ("conservative+consolidate", {
        "conservative_prune": True,
        "consolidate_similar": True,
    }, "Conservative + Consolidate"),

    # ── Registrierte Estimator-Konfigurationen (aus estimators.py) ──
    ("est:shrink_conservative", {
        "conservative_prune": True,
    }, "[estimators.py] shrink_conservative"),

    ("est:shrink_aggressive", {
        "aggressive_prune": True,
        "max_f1_loss": 0.01,
    }, "[estimators.py] shrink_aggressive"),

    ("est:shrink_filter", {
        "filter_weak_rules": True,
        "min_fitness_percentile": 0.4,
        "conservative_prune": True,
        "consolidate_similar": True,
        "interval_merge": True,
        "aggressive_prune": True,
        "max_f1_loss": 0.02,
    }, "[estimators.py] shrink_filter (Vollpipeline)"),

    ("est:shrink_all", {
        "conservative_prune": True,
        "filter_weak_rules": True,
        "consolidate_similar": True,
        "aggressive_prune": True,
        "max_f1_loss": 0.01,
    }, "[estimators.py] shrink_all"),

    # ── Empfohlene Kandidaten zum Zusammenfassen ──
    ("proposed:light", {
        "filter_weak_rules": True,
        "min_fitness_percentile": 0.2,
        "interval_merge": True,
        "interval_merge_iou_threshold": 0.3,
    }, "VORSCHLAG: Light (Filter + IMerge)"),

    ("proposed:standard", {
        "filter_weak_rules": True,
        "min_fitness_percentile": 0.2,
        "interval_merge": True,
        "interval_merge_iou_threshold": 0.3,
        "conservative_prune": True,
        "consolidate_similar": True,
    }, "VORSCHLAG: Standard (Filter+IMerge+Cons+Consol)"),

    ("proposed:aggressive", {
        "filter_weak_rules": True,
        "min_fitness_percentile": 0.3,
        "interval_merge": True,
        "interval_merge_iou_threshold": 0.3,
        "conservative_prune": True,
        "consolidate_similar": True,
        "aggressive_prune": True,
        "max_f1_loss": 0.02,
    }, "VORSCHLAG: Aggressive (Alles an, 2% loss)"),
]


# ---------------------------------------------------------------------------
# Ergebnis-Datenstruktur
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    dataset: str
    variant: str
    description: str
    repeat: int
    f1_test: float
    f1_train: float
    n_rules: int
    n_atoms: int
    avg_atoms: float
    fit_seconds: float
    error: str | None = None


# ---------------------------------------------------------------------------
# Einzellauf
# ---------------------------------------------------------------------------

def _run_single(
    ds_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    variant_name: str,
    exstracs_params: dict | None,
    description: str,
    repeat: int,
) -> RunResult:
    """Trainiere einen ExSTraCS-Classifier mit gegebenen Shrinking-Params und messe."""
    t0 = time.time()
    try:
        clf = ScoredRuleSetClassifier(
            backend="exstracs",
            backend_params={},
            exstracs_params=exstracs_params,
            random_state=42 + repeat,
        )
        clf.fit(X_train, y_train)
        elapsed = time.time() - t0

        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)
        f1_test = f1_score(y_test, y_pred_test, average="macro", zero_division=0)
        f1_train = f1_score(y_train, y_pred_train, average="macro", zero_division=0)

        rs = clf.to_ruleset()
        n_rules = len([r for r in rs.rules if r.atoms])
        n_atoms = sum(len(r.atoms) for r in rs.rules)
        avg_atoms = n_atoms / max(n_rules, 1)

        return RunResult(
            dataset=ds_name,
            variant=variant_name,
            description=description,
            repeat=repeat,
            f1_test=f1_test,
            f1_train=f1_train,
            n_rules=n_rules,
            n_atoms=n_atoms,
            avg_atoms=avg_atoms,
            fit_seconds=elapsed,
        )
    except Exception as e:
        elapsed = time.time() - t0
        return RunResult(
            dataset=ds_name,
            variant=variant_name,
            description=description,
            repeat=repeat,
            f1_test=0.0,
            f1_train=0.0,
            n_rules=0,
            n_atoms=0,
            avg_atoms=0.0,
            fit_seconds=elapsed,
            error=str(e)[:120],
        )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

@dataclass
class AggResult:
    dataset: str
    variant: str
    description: str
    n_repeats: int
    f1_test_mean: float
    f1_test_std: float
    f1_train_mean: float
    n_rules_mean: float
    n_atoms_mean: float
    avg_atoms_mean: float
    fit_seconds_mean: float
    errors: int


def _aggregate(results: list[RunResult]) -> list[AggResult]:
    """Aggregiere Ergebnisse über Wiederholungen, gruppiert nach (dataset, variant)."""
    groups: dict[tuple[str, str], list[RunResult]] = {}
    for r in results:
        key = (r.dataset, r.variant)
        groups.setdefault(key, []).append(r)

    agg: list[AggResult] = []
    for (ds, var), runs in groups.items():
        valid = [r for r in runs if r.error is None]
        errs = len(runs) - len(valid)
        if not valid:
            agg.append(AggResult(
                dataset=ds, variant=var, description=runs[0].description,
                n_repeats=len(runs), f1_test_mean=0.0, f1_test_std=0.0,
                f1_train_mean=0.0, n_rules_mean=0.0, n_atoms_mean=0.0,
                avg_atoms_mean=0.0, fit_seconds_mean=0.0, errors=errs,
            ))
            continue
        agg.append(AggResult(
            dataset=ds,
            variant=var,
            description=valid[0].description,
            n_repeats=len(runs),
            f1_test_mean=float(np.mean([r.f1_test for r in valid])),
            f1_test_std=float(np.std([r.f1_test for r in valid])),
            f1_train_mean=float(np.mean([r.f1_train for r in valid])),
            n_rules_mean=float(np.mean([r.n_rules for r in valid])),
            n_atoms_mean=float(np.mean([r.n_atoms for r in valid])),
            avg_atoms_mean=float(np.mean([r.avg_atoms for r in valid])),
            fit_seconds_mean=float(np.mean([r.fit_seconds for r in valid])),
            errors=errs,
        ))
    return agg


# ---------------------------------------------------------------------------
# Redundanz-Analyse
# ---------------------------------------------------------------------------

def _redundancy_analysis(agg: list[AggResult]) -> str:
    """Analysiere paarweise Redundanz zwischen Varianten.

    Zwei Varianten gelten als *redundant*, wenn sie sich auf KEINEM Dataset
    signifikant in F1 oder Modellgröße unterscheiden.

    Schwellenwerte:
        - |ΔF1| ≤ 0.01  UND
        - |ΔAtome| / max(Atome) ≤ 0.10
    """
    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("REDUNDANZ-ANALYSE")
    lines.append("=" * 100)
    lines.append("Paare, die sich auf keinem Datensatz wesentlich unterscheiden")
    lines.append("(|ΔF1| ≤ 0.01 UND relative Atom-Differenz ≤ 10%)")
    lines.append("-" * 100)

    # Gruppiere nach Dataset
    by_ds: dict[str, dict[str, AggResult]] = {}
    for a in agg:
        by_ds.setdefault(a.dataset, {})[a.variant] = a

    all_variants = sorted({a.variant for a in agg})
    datasets = sorted(by_ds.keys())

    redundant_pairs: list[tuple[str, str]] = []

    for v1, v2 in itertools.combinations(all_variants, 2):
        is_redundant = True
        for ds in datasets:
            r1 = by_ds.get(ds, {}).get(v1)
            r2 = by_ds.get(ds, {}).get(v2)
            if r1 is None or r2 is None:
                continue
            delta_f1 = abs(r1.f1_test_mean - r2.f1_test_mean)
            max_atoms = max(r1.n_atoms_mean, r2.n_atoms_mean, 1)
            delta_atoms_rel = abs(r1.n_atoms_mean - r2.n_atoms_mean) / max_atoms
            if delta_f1 > 0.01 or delta_atoms_rel > 0.10:
                is_redundant = False
                break
        if is_redundant:
            redundant_pairs.append((v1, v2))

    if redundant_pairs:
        for v1, v2 in redundant_pairs:
            lines.append(f"  ≈  {v1:<35} ↔ {v2}")
    else:
        lines.append("  Keine vollständig redundanten Paare gefunden.")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Empfehlungen
# ---------------------------------------------------------------------------

def _recommendations(agg: list[AggResult]) -> str:
    """Generiere Empfehlungen zur Konsolidierung der Varianten."""
    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("EMPFEHLUNGEN")
    lines.append("=" * 100)

    # Finde Baseline-Ergebnisse
    baselines = {a.dataset: a for a in agg if a.variant == "baseline"}

    # Pro Dataset: Ranke Varianten nach Pareto-Kriterium (F1 hoch, Atome niedrig)
    by_ds: dict[str, list[AggResult]] = {}
    for a in agg:
        by_ds.setdefault(a.dataset, []).append(a)

    # Über alle Datasets gemittelte Scores
    variant_scores: dict[str, list[float]] = {}  # Variant → list of relative scores
    variant_reductions: dict[str, list[float]] = {}  # Variant → list of atom reductions

    for ds, results in by_ds.items():
        bl = baselines.get(ds)
        if bl is None:
            continue
        for r in results:
            if r.variant == "baseline":
                continue
            f1_diff = r.f1_test_mean - bl.f1_test_mean
            atom_reduction = (
                (1 - r.n_atoms_mean / bl.n_atoms_mean) * 100
                if bl.n_atoms_mean > 0
                else 0
            )
            # Score: +1 Punkt pro 1% Atom-Reduktion, -10 Punkte pro 1% F1-Verlust
            score = atom_reduction + f1_diff * 1000
            variant_scores.setdefault(r.variant, []).append(score)
            variant_reductions.setdefault(r.variant, []).append(atom_reduction)

    # Mittlerer Score über alle Datasets
    ranked = sorted(
        variant_scores.items(),
        key=lambda x: np.mean(x[1]),
        reverse=True,
    )

    lines.append("")
    lines.append("Ranking nach mittlerem Score (Atom-Reduktion belohnt, F1-Verlust bestraft):")
    lines.append(f"  {'Rang':>4}  {'Variante':<35}  {'Score':>8}  {'Atom-Red.':>10}")
    lines.append(f"  {'-' * 4}  {'-' * 35}  {'-' * 8}  {'-' * 10}")
    for rank, (variant, scores) in enumerate(ranked, 1):
        mean_score = np.mean(scores)
        mean_red = np.mean(variant_reductions.get(variant, [0]))
        lines.append(f"  {rank:>4}  {variant:<35}  {mean_score:>8.1f}  {mean_red:>9.1f}%")

    # Explizite Empfehlung
    lines.append("")
    lines.append("-" * 100)
    lines.append("Vorschlag zur Konsolidierung der estimators.py ExSTraCS-Specs:")
    lines.append("")
    lines.append("  Registrierte ExSTraCS-Varianten in estimators.py (nach Konsolidierung):")
    lines.append("    1. wrapper_exstracs                    (Baseline)")
    lines.append("    2. wrapper_exstracs_pruned             (conservative_prune, F1-erhaltend)")
    lines.append("    3. wrapper_exstracs_compact            (interval_merge + conservative, max. Kompression)")
    lines.append("")
    lines.append("  Prüfe oben:")
    lines.append("    - Welche 'est:' Varianten sich kaum unterscheiden (→ zusammenfassen)")
    lines.append("    - Welche 'proposed:' Varianten die besten Trade-offs bieten")
    lines.append("    - Ob 'shrink_all' und 'shrink_filter' wirklich verschieden sind")
    lines.append("=" * 100)
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV-Export
# ---------------------------------------------------------------------------

def _write_csv(agg: list[AggResult], path: Path) -> None:
    """Schreibe aggregierte Ergebnisse als CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset", "variant", "description", "n_repeats",
            "f1_test_mean", "f1_test_std", "f1_train_mean",
            "n_rules_mean", "n_atoms_mean", "avg_atoms_mean",
            "fit_seconds_mean", "errors",
        ])
        for a in agg:
            writer.writerow([
                a.dataset, a.variant, a.description, a.n_repeats,
                f"{a.f1_test_mean:.4f}", f"{a.f1_test_std:.4f}", f"{a.f1_train_mean:.4f}",
                f"{a.n_rules_mean:.1f}", f"{a.n_atoms_mean:.1f}", f"{a.avg_atoms_mean:.1f}",
                f"{a.fit_seconds_mean:.1f}", a.errors,
            ])
    print(f"\n  CSV geschrieben: {path}")


# ---------------------------------------------------------------------------
# Konsolen-Tabelle
# ---------------------------------------------------------------------------

def _print_dataset_table(
    ds_name: str,
    agg_ds: list[AggResult],
    baseline: AggResult | None,
) -> None:
    """Drucke Ergebnis-Tabelle für ein Dataset."""
    header = (
        f"  {'Variante':<35} {'F1-test':>8} {'±std':>6} "
        f"{'ΔF1':>6} {'Rules':>6} {'Atoms':>6} {'Atom↓%':>7} {'t(s)':>7}"
    )
    sep = f"  {'-' * 35} {'-' * 8} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 7} {'-' * 7}"

    print(f"\n{'─' * 100}")
    print(f"  Dataset: {ds_name}")
    print(f"{'─' * 100}")
    print(header)
    print(sep)

    for a in agg_ds:
        if baseline is not None and a.variant != "baseline":
            delta_f1 = a.f1_test_mean - baseline.f1_test_mean
            atom_red = (
                (1 - a.n_atoms_mean / baseline.n_atoms_mean) * 100
                if baseline.n_atoms_mean > 0 else 0
            )
            delta_str = f"{delta_f1:>+.3f}"
            red_str = f"{atom_red:>6.1f}%"
        else:
            delta_str = "  ref"
            red_str = "    ref"

        err_mark = " ✗" if a.errors > 0 else ""
        print(
            f"  {a.variant:<35} {a.f1_test_mean:>8.4f} {a.f1_test_std:>6.3f} "
            f"{delta_str:>6} {a.n_rules_mean:>6.0f} {a.n_atoms_mean:>6.0f} "
            f"{red_str:>7} {a.fit_seconds_mean:>7.1f}{err_mark}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="ExSTraCS Shrinking-Varianten Benchmark",
    )
    parser.add_argument(
        "--datasets", type=str, default="",
        help="Komma-separierte Datensatz-Namen (default: sklearn_iris,sklearn_wine,sklearn_breast_cancer)",
    )
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="Anzahl Wiederholungen pro Variante/Dataset (default: 3)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Schnellmodus: nur sklearn-Datensätze, 1 Repeat",
    )
    parser.add_argument(
        "--output", type=str, default="benchmarks/exstracs_shrinking_benchmark.csv",
        help="CSV-Ausgabepfad",
    )
    args = parser.parse_args(argv)

    if args.quick:
        args.repeats = 1

    ds_names = [x.strip() for x in args.datasets.split(",") if x.strip()] or None
    datasets = _load_datasets(ds_names)
    repeats = max(1, args.repeats)

    n_total = len(datasets) * len(SHRINKING_VARIANTS) * repeats
    print("=" * 100)
    print("ExSTraCS SHRINKING-VARIANTEN BENCHMARK")
    print("=" * 100)
    print(f"  Datensätze:      {len(datasets)} ({', '.join(d[0] for d in datasets)})")
    print(f"  Varianten:       {len(SHRINKING_VARIANTS)}")
    print(f"  Wiederholungen:  {repeats}")
    print(f"  Gesamtläufe:     {n_total}")
    print(f"  CSV-Ausgabe:     {args.output}")
    print("=" * 100)

    all_results: list[RunResult] = []
    run_idx = 0
    t_global = time.time()

    for ds_name, X, y in datasets:
        for repeat in range(repeats):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42 + repeat, stratify=y,
            )
            for variant_name, exstracs_params, description in SHRINKING_VARIANTS:
                run_idx += 1
                label = f"[{run_idx:>4}/{n_total}]"
                print(
                    f"  {label} {ds_name:<25} r={repeat} {variant_name:<35}",
                    end=" ",
                    flush=True,
                )
                result = _run_single(
                    ds_name=ds_name,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    variant_name=variant_name,
                    exstracs_params=exstracs_params,
                    description=description,
                    repeat=repeat,
                )
                all_results.append(result)
                if result.error:
                    print(f"✗ {result.error[:60]}")
                else:
                    print(
                        f"F1={result.f1_test:.4f} "
                        f"R={result.n_rules:>3} A={result.n_atoms:>4} "
                        f"t={result.fit_seconds:.1f}s"
                    )

    elapsed_total = time.time() - t_global
    print(f"\n  Gesamtlaufzeit: {elapsed_total:.0f}s ({elapsed_total / 60:.1f} min)")

    # Aggregation
    agg = _aggregate(all_results)

    # Pro-Dataset Tabellen
    datasets_seen = list(dict.fromkeys(a.dataset for a in agg))
    for ds in datasets_seen:
        agg_ds = [a for a in agg if a.dataset == ds]
        baseline = next((a for a in agg_ds if a.variant == "baseline"), None)
        _print_dataset_table(ds, agg_ds, baseline)

    # Über alle Datasets gemittelt
    print(f"\n{'━' * 100}")
    print("  ÜBER ALLE DATASETS GEMITTELT")
    print(f"{'━' * 100}")

    # Gruppiere agg nach Variante
    by_variant: dict[str, list[AggResult]] = {}
    for a in agg:
        by_variant.setdefault(a.variant, []).append(a)

    baseline_all = by_variant.get("baseline", [])
    bl_f1_mean = np.mean([a.f1_test_mean for a in baseline_all]) if baseline_all else 0

    header = f"  {'Variante':<35} {'F1-mean':>8} {'ΔF1':>6} {'Rules':>6} {'Atoms':>6} {'Atom↓%':>7}"
    print(header)
    print(f"  {'-' * 35} {'-' * 8} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 7}")

    for variant, results in by_variant.items():
        f1_m = np.mean([a.f1_test_mean for a in results])
        rules_m = np.mean([a.n_rules_mean for a in results])
        atoms_m = np.mean([a.n_atoms_mean for a in results])
        bl_atoms = np.mean([a.n_atoms_mean for a in baseline_all]) if baseline_all else atoms_m

        delta_f1 = f1_m - bl_f1_mean if variant != "baseline" else 0
        atom_red = (1 - atoms_m / bl_atoms) * 100 if bl_atoms > 0 and variant != "baseline" else 0

        delta_str = f"{delta_f1:>+.3f}" if variant != "baseline" else "  ref"
        red_str = f"{atom_red:>6.1f}%" if variant != "baseline" else "    ref"

        print(
            f"  {variant:<35} {f1_m:>8.4f} {delta_str:>6} "
            f"{rules_m:>6.0f} {atoms_m:>6.0f} {red_str:>7}"
        )

    # Redundanz-Analyse
    print()
    print(_redundancy_analysis(agg))

    # Empfehlungen
    print(_recommendations(agg))

    # CSV schreiben
    _write_csv(agg, Path(args.output))

    print("Benchmark abgeschlossen.\n")


if __name__ == "__main__":
    main()

