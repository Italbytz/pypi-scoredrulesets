#!/usr/bin/env python
"""
Gezielter Benchmark-Vergleich: Java-RuleKit vs. Python-RuleKit-Native.

Dieses Skript vergleicht die beiden RuleKit-Varianten auf mehreren
Standard-Datensätzen und gibt eine detaillierte Analyse aus.
Falls das Java-Backend (wrapper_rulekit) nicht verfügbar ist (kein JDK),
wird dies protokolliert und nur das native Backend getestet.

Aufruf:
    python tools/analysis/benchmark_rulekit_comparison.py
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

# --------------------------------------------------------------------------
# Projekt-Imports
# --------------------------------------------------------------------------
from scoredrulesets import ScoredRuleSetClassifier, RuleKitNativeClassifier
from scoredrulesets.formatting import format_ruleset_table


# --------------------------------------------------------------------------
# Datensätze
# --------------------------------------------------------------------------

def _load_datasets() -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Lade Standard-Datensätze (keine externen Downloads nötig)."""
    datasets: list[tuple[str, np.ndarray, np.ndarray]] = []

    X, y = load_iris(return_X_y=True)
    datasets.append(("iris", X, y))

    X, y = load_wine(return_X_y=True)
    datasets.append(("wine", X, y))

    X, y = load_breast_cancer(return_X_y=True)
    datasets.append(("breast_cancer", X, y))

    return datasets


# --------------------------------------------------------------------------
# Estimator-Konfigurationen
# --------------------------------------------------------------------------

ESTIMATOR_CONFIGS: dict[str, dict] = {
    "rulekit_native": {
        "backend": "rulekit_native",
        "backend_params": {
            "max_rules": 20,
            "max_conditions": 5,
            "min_samples_leaf": 5,
            "enable_pruning": True,
        },
    },
    "rulekit_native_no_prune": {
        "backend": "rulekit_native",
        "backend_params": {
            "max_rules": 20,
            "max_conditions": 5,
            "min_samples_leaf": 5,
            "enable_pruning": False,
        },
    },
    "rulekit_native_strong": {
        "backend": "rulekit_native",
        "backend_params": {
            "max_rules": 30,
            "max_conditions": 7,
            "min_samples_leaf": 3,
            "enable_pruning": True,
            "pruning_fraction": 0.25,
        },
    },
    "rulekit_java": {
        "backend": "rulekit",
        "backend_params": {},
    },
}


# --------------------------------------------------------------------------
# Evaluierung
# --------------------------------------------------------------------------

def evaluate_estimator(
    name: str,
    config: dict,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Evaluiere einen Estimator via Stratified-K-Fold CV."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    f1_scores = []
    acc_scores = []
    fit_times = []
    pred_times = []
    n_rules_list = []
    n_atoms_list = []
    errors: list[str] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        try:
            clf = ScoredRuleSetClassifier(random_state=random_state, **config)

            t0 = time.perf_counter()
            clf.fit(X_train, y_train)
            fit_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            y_pred = clf.predict(X_test)
            pred_time = time.perf_counter() - t0

            f1 = f1_score(y_test, y_pred, average="macro")
            acc = accuracy_score(y_test, y_pred)

            ruleset = clf.to_ruleset()
            non_default_rules = [r for r in ruleset.rules if r.atoms]
            total_atoms = sum(len(r.atoms) for r in ruleset.rules)

            f1_scores.append(f1)
            acc_scores.append(acc)
            fit_times.append(fit_time)
            pred_times.append(pred_time)
            n_rules_list.append(len(non_default_rules))
            n_atoms_list.append(total_atoms)

        except Exception as e:
            errors.append(f"Fold {fold_idx}: {type(e).__name__}: {e}")

    return {
        "name": name,
        "f1_macro_mean": float(np.mean(f1_scores)) if f1_scores else None,
        "f1_macro_std": float(np.std(f1_scores)) if f1_scores else None,
        "accuracy_mean": float(np.mean(acc_scores)) if acc_scores else None,
        "accuracy_std": float(np.std(acc_scores)) if acc_scores else None,
        "fit_time_mean": float(np.mean(fit_times)) if fit_times else None,
        "fit_time_std": float(np.std(fit_times)) if fit_times else None,
        "pred_time_mean": float(np.mean(pred_times)) if pred_times else None,
        "n_rules_mean": float(np.mean(n_rules_list)) if n_rules_list else None,
        "n_atoms_mean": float(np.mean(n_atoms_list)) if n_atoms_list else None,
        "n_folds_ok": len(f1_scores),
        "n_folds_total": n_splits,
        "errors": errors,
    }


# --------------------------------------------------------------------------
# Regelinspektion
# --------------------------------------------------------------------------

def inspect_rules(name: str, config: dict, X: np.ndarray, y: np.ndarray) -> str | None:
    """Trainiere auf dem vollen Datensatz und gib die Regeln aus."""
    try:
        clf = ScoredRuleSetClassifier(random_state=42, **config)
        clf.fit(X, y)
        ruleset = clf.to_ruleset()
        return format_ruleset_table(ruleset)
    except Exception as e:
        return f"  [FEHLER: {type(e).__name__}: {e}]"


# --------------------------------------------------------------------------
# Hauptprogramm
# --------------------------------------------------------------------------

def main() -> None:
    print("=" * 78)
    print("  RuleKit Java vs. RuleKit Native — Gezielter Benchmark-Vergleich")
    print("=" * 78)
    print()

    datasets = _load_datasets()

    # Prüfe ob Java-RuleKit verfügbar ist
    java_available = True
    try:
        test_clf = ScoredRuleSetClassifier(backend="rulekit", random_state=0)
        X_test, y_test = load_iris(return_X_y=True)
        test_clf.fit(X_test[:20], y_test[:20])
        print("[INFO] Java-RuleKit Backend ist verfügbar.\n")
    except Exception as e:
        java_available = False
        print(f"[WARNUNG] Java-RuleKit Backend NICHT verfügbar: {type(e).__name__}: {e}")
        print("[INFO] Es wird nur das native Python-Backend getestet.\n")

    # Welche Estimatoren testen?
    configs_to_test = {k: v for k, v in ESTIMATOR_CONFIGS.items()
                       if java_available or v["backend"] != "rulekit"}

    # ---------- Pro Datensatz evaluieren ----------
    all_results: dict[str, list[dict]] = {}

    for ds_name, X, y in datasets:
        print(f"\n{'─' * 78}")
        print(f"  Datensatz: {ds_name}  (n={X.shape[0]}, p={X.shape[1]}, "
              f"Klassen={len(np.unique(y))})")
        print(f"{'─' * 78}")

        ds_results = []
        for est_name, config in configs_to_test.items():
            print(f"  ▸ {est_name:30s} ... ", end="", flush=True)
            result = evaluate_estimator(est_name, config, X, y)
            ds_results.append(result)

            if result["f1_macro_mean"] is not None:
                print(
                    f"F1={result['f1_macro_mean']:.4f}±{result['f1_macro_std']:.4f}  "
                    f"Acc={result['accuracy_mean']:.4f}  "
                    f"Rules={result['n_rules_mean']:.1f}  "
                    f"Atoms={result['n_atoms_mean']:.1f}  "
                    f"Fit={result['fit_time_mean']:.3f}s"
                )
            else:
                print(f"FEHLER ({len(result['errors'])} Folds fehlgeschlagen)")
                for err in result["errors"][:3]:
                    print(f"    {err}")

        all_results[ds_name] = ds_results

    # ---------- Zusammenfassung ----------
    print(f"\n\n{'=' * 78}")
    print("  ZUSAMMENFASSUNG")
    print(f"{'=' * 78}\n")

    # Sammle Ergebnisse in einer Tabelle
    header = f"{'Datensatz':20s} {'Estimator':30s} {'F1-macro':>12s} {'Accuracy':>12s} {'#Rules':>8s} {'#Atoms':>8s} {'Fit(s)':>10s}"
    print(header)
    print("─" * len(header))

    for ds_name, ds_results in all_results.items():
        for r in ds_results:
            if r["f1_macro_mean"] is not None:
                print(
                    f"{ds_name:20s} {r['name']:30s} "
                    f"{r['f1_macro_mean']:>7.4f}±{r['f1_macro_std']:.3f} "
                    f"{r['accuracy_mean']:>7.4f}±{r['accuracy_std']:.3f} "
                    f"{r['n_rules_mean']:>7.1f} "
                    f"{r['n_atoms_mean']:>7.1f} "
                    f"{r['fit_time_mean']:>9.3f}"
                )
            else:
                print(f"{ds_name:20s} {r['name']:30s} {'FEHLER':>12s}")

    # ---------- Qualitative Regelinspektion ----------
    print(f"\n\n{'=' * 78}")
    print("  REGELINSPEKTION (Iris, voller Datensatz)")
    print(f"{'=' * 78}")

    for est_name, config in configs_to_test.items():
        print(f"\n─── {est_name} ───")
        X_iris, y_iris = load_iris(return_X_y=True)
        table = inspect_rules(est_name, config, X_iris, y_iris)
        if table:
            print(table)

    # ---------- Empfehlung ----------
    print(f"\n\n{'=' * 78}")
    print("  EMPFEHLUNG")
    print(f"{'=' * 78}\n")

    if not java_available:
        print("  Das Java-RuleKit Backend konnte nicht geladen werden.")
        print("  → Das native Python-Backend (rulekit_native) kann als Ersatz dienen.")
        print("  → Zum Vergleich muss Java (JDK 11+) installiert sein.")
    else:
        # Vergleiche F1-Scores
        diffs = []
        for ds_name, ds_results in all_results.items():
            java_r = next((r for r in ds_results if r["name"] == "rulekit_java"), None)
            native_r = next((r for r in ds_results if r["name"] == "rulekit_native"), None)
            if java_r and native_r and java_r["f1_macro_mean"] is not None and native_r["f1_macro_mean"] is not None:
                diff = abs(java_r["f1_macro_mean"] - native_r["f1_macro_mean"])
                diffs.append((ds_name, diff, java_r["f1_macro_mean"], native_r["f1_macro_mean"]))

        if diffs:
            max_diff = max(d[1] for d in diffs)
            mean_diff = np.mean([d[1] for d in diffs])
            print(f"  F1-Differenz (Java vs. Native) über {len(diffs)} Datensätze:")
            for ds, diff, java_f1, native_f1 in diffs:
                marker = "≈" if diff < 0.02 else ("△" if diff < 0.05 else "▲")
                print(f"    {marker} {ds:20s}: Java={java_f1:.4f}  Native={native_f1:.4f}  Δ={diff:.4f}")
            print()
            if max_diff < 0.02:
                print("  ✅ Die Ergebnisse sind praktisch identisch (max Δ < 0.02).")
                print("  → Das Java-RuleKit Backend kann sicher entfernt werden.")
            elif max_diff < 0.05:
                print("  ⚠️  Kleine Unterschiede vorhanden (max Δ < 0.05).")
                print("  → Prüfe ob die Unterschiede für Dein Projekt akzeptabel sind.")
            else:
                print("  ❌ Signifikante Unterschiede vorhanden!")
                print("  → Beide Backends sollten vorerst erhalten bleiben.")
        else:
            print("  Kein Vergleich möglich (fehlende Ergebnisse).")

    print()


if __name__ == "__main__":
    main()

