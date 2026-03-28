"""
Example: Neural Logic Network (NLN) backend for Scored Rule Sets.

This example shows how to use the NLN backend, inspired by
Payani & Fekri ("Learning Algorithms via Neural Logic Networks").

The NLN backend:
  - Discretizes features into binary propositions (quantile thresholds)
  - Learns conjunction rules via differentiable logic (gradient descent)
  - Extracts interpretable rules as a ScoredRuleSet
  - Requires only NumPy (no PyTorch/TensorFlow)

Run:
    python examples/estimators/example_rulenln_backend.py
"""

import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

from scoredrulesets.estimators.rulenln import RuleNLNClassifier
from scoredrulesets.estimators.sklearn_wrapper import ScoredRuleSetClassifier


def print_ruleset(rs, title="Rule Set"):
    """Pretty-print a ScoredRuleSet."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(f"  Classes:  {rs.class_labels}")
    print(f"  Features: {rs.feature_names}")
    print(f"  Rules:    {len(rs.rules)}")
    print()

    for i, rule in enumerate(rs.rules):
        if not rule.atoms:
            scores_str = ", ".join(f"{s:+.3f}" for s in rule.scores)
            print(f"  [{rule.rule_id or 'default'}]  DEFAULT → scores=[{scores_str}]")
        else:
            conditions = " AND ".join(
                f"{a.feature} {a.op} {a.value:.3f}" for a in rule.atoms
            )
            scores_str = ", ".join(f"{s:+.3f}" for s in rule.scores)
            print(f"  [{rule.rule_id}]  IF {conditions}")
            print(f"    {'':>12}THEN scores=[{scores_str}]")
    print(f"{'=' * 70}")


def demo_native_iris():
    """Apply NLN directly to Iris."""
    print("\n" + "=" * 70)
    print("  Demo 1: native NLN on Iris (3 classes)")
    print("=" * 70)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y,
    )

    clf = RuleNLNClassifier(
        n_rules=12,
        n_bins=5,
        learning_rate=0.3,
        l1_conj=0.002,
        l1_score=0.001,
        epochs=300,
        early_stopping_rounds=30,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"\nF1 (macro): {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(classification_report(y_test, y_pred, target_names=["setosa", "versicolor", "virginica"]))

    rs = clf.to_ruleset()
    print_ruleset(rs, "NLN Rule Set: Iris")

    return clf


def demo_wrapper_breast_cancer():
    """Apply NLN via ScoredRuleSetClassifier wrapper on Breast Cancer."""
    print("\n" + "=" * 70)
    print("  Demo 2: NLN wrapper on Breast Cancer (binary)")
    print("=" * 70)

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y,
    )

    clf = ScoredRuleSetClassifier(
        backend="rulenln",
        backend_params={
            "n_rules": 10,
            "n_bins": 4,
            "learning_rate": 0.3,
            "l1_conj": 0.002,
            "l1_score": 0.001,
            "epochs": 300,
            "early_stopping_rounds": 30,
        },
        random_state=42,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"\nF1 (macro): {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(classification_report(y_test, y_pred, target_names=["malignant", "benign"]))

    rs = clf.to_ruleset()
    print_ruleset(rs, "NLN Rule Set: Breast Cancer")

    return clf


def demo_sparsity_control():
    """Show the effect of L1 regularization on rule complexity."""
    print("\n" + "=" * 70)
    print("  Demo 3: L1 regularization and sparsity")
    print("=" * 70)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y,
    )

    configs = [
        ("Low (l1=0.0005)", 0.0005, 0.0002),
        ("Medium (l1=0.005)",   0.005,  0.002),
        ("High (l1=0.05)",     0.05,   0.02),
    ]

    for label, l1_c, l1_s in configs:
        clf = RuleNLNClassifier(
            n_rules=12, n_bins=5, epochs=200,
            learning_rate=0.3, l1_conj=l1_c, l1_score=l1_s,
            early_stopping_rounds=30, random_state=42,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        rs = clf.to_ruleset()

        n_rules = len([r for r in rs.rules if r.atoms])
        n_atoms = sum(len(r.atoms) for r in rs.rules)
        f1 = f1_score(y_test, y_pred, average="macro")

        print(f"  {label:30s}  -> {n_rules:2d} rules, {n_atoms:3d} atoms, F1={f1:.4f}")


if __name__ == "__main__":
    demo_native_iris()
    demo_wrapper_breast_cancer()
    demo_sparsity_control()
    print("\nDone.")

