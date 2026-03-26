#!/usr/bin/env python3
"""Example: RuleLCS-style scored rule set backend.

This script demonstrates three things:
1. Training `RuleLCSClassifier` directly
2. Inspecting the learned `ScoredRuleSet` metadata and rule table
3. Running a small comparison against `ScoredRuleSetClassifier(backend="cart")`
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.datasets import load_iris
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from scoredrulesets import (
    RuleLCSClassifier,
    ScoredRuleSetClassifier,
    format_ruleset_table,
)


PROFILE_BACKEND_PARAMS: dict[str, dict[str, Any]] = {
    "default": {
        "max_rules": 5,
        "candidate_pool_size": 20,
        "beam_width": 6,
        "max_iterations": 12,
        "validation_fraction": 0.2,
        "complexity_penalty": 0.01,
    },
    "fast": {
        "max_rules": 4,
        "candidate_pool_size": 12,
        "beam_width": 4,
        "max_iterations": 6,
        "validation_fraction": 0.15,
        "complexity_penalty": 0.012,
    },
    "strong": {
        "max_rules": 6,
        "candidate_pool_size": 32,
        "beam_width": 10,
        "max_iterations": 20,
        "validation_fraction": 0.25,
        "complexity_penalty": 0.008,
    },
    "diverse": {
        "max_rules": 7,
        "min_samples_leaf": 3,
        "candidate_pool_size": 36,
        "beam_width": 10,
        "max_iterations": 18,
        "validation_fraction": 0.25,
        "complexity_penalty": 0.01,
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the direct RuleLCS estimator example")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_BACKEND_PARAMS.keys()),
        default="default",
        help="RuleLCS estimator profile to run",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


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


def run_demo(random_state: int = 42, profile: str = "default") -> dict[str, object]:
    if profile not in PROFILE_BACKEND_PARAMS:
        raise ValueError(
            f"Unknown profile '{profile}'. Available: {sorted(PROFILE_BACKEND_PARAMS)}"
        )

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.30,
        random_state=random_state,
        stratify=iris.target,
    )

    rulelcs = RuleLCSClassifier(
        **dict(PROFILE_BACKEND_PARAMS[profile]),
        random_state=random_state,
    )
    rulelcs_result = _evaluate_classifier(
        "rulelcs",
        rulelcs,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    comparison_estimators = [
        (
            "cart",
            ScoredRuleSetClassifier(
                backend="cart",
                backend_params={"max_depth": 4},
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
        "profile": profile,
        "train_size": len(y_train),
        "test_size": len(y_test),
        "rulelcs": rulelcs_result,
        "comparison": comparison_results,
    }


def main() -> None:
    args = _parse_args()
    result = run_demo(random_state=args.random_state, profile=args.profile)
    rulelcs = cast(dict[str, Any], result["rulelcs"])

    print("=" * 80)
    print("RuleLCS backend example")
    print("=" * 80)
    print(f"Dataset: {result['dataset']}")
    print(f"Profile: {result['profile']}")
    print(f"Train size: {result['train_size']} | Test size: {result['test_size']}")
    print()

    print("Direct RuleLCSClassifier run")
    print("-" * 80)
    print(f"F1 macro: {rulelcs['f1_macro']:.4f}")
    print(f"Rules:    {rulelcs['n_rules']}")
    print(f"Atoms:    {rulelcs['n_atoms']}")
    print("Metadata:")
    for key, value in sorted(rulelcs["metadata"].items()):
        print(f"  - {key}: {value}")
    print()
    print("Learned ruleset")
    print("-" * 80)
    print(format_ruleset_table(rulelcs["ruleset"]))
    print()

    print("Mini comparison against native and gp")
    print("-" * 80)
    print(f"{'estimator':16} {'f1_macro':>10} {'rules':>8} {'atoms':>8}")
    print("-" * 80)
    comparison_rows = cast(list[dict[str, Any]], result["comparison"])
    all_rows = [rulelcs] + comparison_rows
    for row in all_rows:
        print(
            f"{row['name']:16} {row['f1_macro']:10.4f} {row['n_rules']:8d} {row['n_atoms']:8d}"
        )
    print("-" * 80)


if __name__ == "__main__":
    main()



