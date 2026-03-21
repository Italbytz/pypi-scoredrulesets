#!/usr/bin/env python3
"""Example: Pittsburgh-style scored rule set backend.

This script demonstrates three things:
1. Training `PittsburghRuleSetClassifier` directly
2. Inspecting the learned `ScoredRuleSet` metadata and rule table
3. Running a small comparison against `NativeScoredRuleSetClassifier`
   and `GeneticScoredRuleSetClassifier`
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.datasets import load_iris
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from scoredrulesets import (
    GeneticScoredRuleSetClassifier,
    NativeScoredRuleSetClassifier,
    PittsburghRuleSetClassifier,
    format_ruleset_table,
)


def _evaluate_classifier(name: str, clf, X_train, X_test, y_train, y_test) -> dict[str, object]:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_true_norm = [str(v) for v in y_test]
    y_pred_norm = [str(v) for v in y_pred]
    f1 = float(f1_score(y_true_norm, y_pred_norm, average="macro"))
    ruleset = clf.to_ruleset()
    return {
        "name": name,
        "f1_macro": f1,
        "n_rules": len(ruleset.rules),
        "n_atoms": sum(len(rule.atoms) for rule in ruleset.rules),
        "metadata": ruleset.metadata,
        "ruleset": ruleset,
    }


def run_demo(random_state: int = 42) -> dict[str, object]:
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.30,
        random_state=random_state,
        stratify=iris.target,
    )

    pittsburgh = PittsburghRuleSetClassifier(
        max_rules=5,
        candidate_pool_size=20,
        beam_width=6,
        max_iterations=12,
        validation_fraction=0.2,
        complexity_penalty=0.01,
        random_state=random_state,
    )
    pittsburgh_result = _evaluate_classifier(
        "pittsburgh",
        pittsburgh,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    comparison_estimators = [
        (
            "native",
            NativeScoredRuleSetClassifier(
                max_rules=6,
                min_samples_leaf=3,
            ),
        ),
        (
            "gp",
            GeneticScoredRuleSetClassifier(
                population_size=20,
                generations=8,
                max_rules=4,
                early_stopping_rounds=3,
                validation_fraction=0.2,
                random_state=random_state,
            ),
        ),
    ]

    comparison_results: list[dict[str, object]] = []
    for name, estimator in comparison_estimators:
        comparison_results.append(
            _evaluate_classifier(name, estimator, X_train, X_test, y_train, y_test)
        )

    return {
        "dataset": "sklearn_iris",
        "train_size": len(y_train),
        "test_size": len(y_test),
        "pittsburgh": pittsburgh_result,
        "comparison": comparison_results,
    }


def main() -> None:
    result = run_demo(random_state=42)
    pittsburgh = cast(dict[str, Any], result["pittsburgh"])

    print("=" * 80)
    print("Pittsburgh backend example")
    print("=" * 80)
    print(f"Dataset: {result['dataset']}")
    print(f"Train size: {result['train_size']} | Test size: {result['test_size']}")
    print()

    print("Direct PittsburghRuleSetClassifier run")
    print("-" * 80)
    print(f"F1 macro: {pittsburgh['f1_macro']:.4f}")
    print(f"Rules:    {pittsburgh['n_rules']}")
    print(f"Atoms:    {pittsburgh['n_atoms']}")
    print("Metadata:")
    for key, value in sorted(pittsburgh["metadata"].items()):
        print(f"  - {key}: {value}")
    print()
    print("Learned ruleset")
    print("-" * 80)
    print(format_ruleset_table(pittsburgh["ruleset"]))
    print()

    print("Mini comparison against native and gp")
    print("-" * 80)
    print(f"{'estimator':16} {'f1_macro':>10} {'rules':>8} {'atoms':>8}")
    print("-" * 80)
    comparison_rows = cast(list[dict[str, Any]], result["comparison"])
    all_rows = [pittsburgh] + comparison_rows
    for row in all_rows:
        print(
            f"{row['name']:16} {row['f1_macro']:10.4f} {row['n_rules']:8d} {row['n_atoms']:8d}"
        )
    print("-" * 80)


if __name__ == "__main__":
    main()



