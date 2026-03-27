#!/usr/bin/env python3
"""Example: RuleLCS backend via ScoredRuleSetClassifier wrapper.

This script demonstrates:
1. Training through `ScoredRuleSetClassifier(backend="rulelcs")`
2. Inspecting scored-ruleset metadata and a compact rules table
3. Comparing wrapper-rulelcs against wrapper-cart and native
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.datasets import load_iris
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from scoredrulesets import ScoredRuleSetClassifier, format_ruleset_table


PROFILE_BACKEND_PARAMS: dict[str, dict[str, Any]] = {
    "default": {
        "population_size": 100,
        "n_iterations": 30,
        "n_repetitions": 1,
        "max_rules": 10,
    },
    "fast": {
        "population_size": 50,
        "n_iterations": 15,
        "n_repetitions": 1,
        "max_rules": 8,
    },
    "strong": {
        "population_size": 200,
        "n_iterations": 50,
        "n_repetitions": 2,
        "tournament_size": 4,
        "crossover_prob": 0.6,
        "mutation_prob": 0.6,
        "max_rules": 15,
        "default_class_policy": "major",
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RuleLCS wrapper example")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_BACKEND_PARAMS.keys()),
        default="default",
        help="RuleLCS backend profile to run",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _evaluate_wrapper(
    name: str,
    clf: ScoredRuleSetClassifier,
    X_train,
    X_test,
    y_train,
    y_test,
) -> dict[str, Any]:
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
        "metadata": dict(ruleset.metadata),
        "ruleset": ruleset,
    }


def run_demo(random_state: int = 42, profile: str = "default") -> dict[str, Any]:
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

    wrapper_rulelcs = ScoredRuleSetClassifier(
        backend="rulelcs",
        backend_params=dict(PROFILE_BACKEND_PARAMS[profile]),
        random_state=random_state,
    )
    rulelcs_result = _evaluate_wrapper(
        "wrapper_rulelcs",
        wrapper_rulelcs,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    comparison_estimators = [
        (
            "wrapper_cart_d2",
            ScoredRuleSetClassifier(
                backend="cart",
                backend_params={"max_depth": 2},
                random_state=random_state,
            ),
        ),
        (
            "wrapper_cart_d4",
            ScoredRuleSetClassifier(
                backend="cart",
                backend_params={"max_depth": 4},
                random_state=random_state,
            ),
        ),
    ]

    comparison_results: list[dict[str, Any]] = []
    for name, estimator in comparison_estimators:
        comparison_results.append(
            _evaluate_wrapper(name, estimator, X_train, X_test, y_train, y_test)
        )

    return {
        "dataset": "sklearn_iris",
        "profile": profile,
        "train_size": len(y_train),
        "test_size": len(y_test),
        "wrapper_rulelcs": rulelcs_result,
        "comparison": comparison_results,
    }


def main() -> None:
    args = _parse_args()
    result = run_demo(random_state=args.random_state, profile=args.profile)
    rulelcs = result["wrapper_rulelcs"]

    print("=" * 80)
    print("RuleLCS wrapper example")
    print("=" * 80)
    print(f"Dataset: {result['dataset']}")
    print(f"Profile: {result['profile']}")
    print(f"Train size: {result['train_size']} | Test size: {result['test_size']}")
    print()

    print("ScoredRuleSetClassifier(backend='rulelcs')")
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

    print("Mini wrapper comparison")
    print("-" * 80)
    print(f"{'estimator':20} {'f1_macro':>10} {'rules':>8} {'atoms':>8}")
    print("-" * 80)
    rows = [rulelcs] + result["comparison"]
    for row in rows:
        print(f"{row['name']:20} {row['f1_macro']:10.4f} {row['n_rules']:8d} {row['n_atoms']:8d}")
    print("-" * 80)


if __name__ == "__main__":
    main()


