"""
Beispiel: Neural Logic Network (NLN) Backend fuer Scored Rule Sets.

Dieses Beispiel zeigt, wie das NLN-Backend verwendet wird, das auf den Ideen
von Payani & Fekri ("Learning Algorithms via Neural Logic Networks") basiert.

Das NLN-Backend:
  - Diskretisiert Features in binaere Propositionen (Quantil-Schwellen)
  - Lernt Konjunktionsregeln via differenzierbarer Logik (Gradientenabstieg)
  - Extrahiert interpretierbare Regeln als ScoredRuleSet
  - Benötigt nur NumPy (kein PyTorch/TensorFlow)

Aufruf:
    python examples/estimators/example_nln_backend.py
"""

import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

from scoredrulesets.estimators.nln import NeuralLogicNetClassifier
from scoredrulesets.estimators.sklearn_wrapper import ScoredRuleSetClassifier


def print_ruleset(rs, title="Rule Set"):
    """Schöne Darstellung eines ScoredRuleSets."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(f"  Klassen:  {rs.class_labels}")
    print(f"  Features: {rs.feature_names}")
    print(f"  Regeln:   {len(rs.rules)}")
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
    """NLN direkt auf Iris anwenden."""
    print("\n" + "=" * 70)
    print("  Demo 1: NLN nativ auf Iris (3-Klassen)")
    print("=" * 70)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y,
    )

    clf = NeuralLogicNetClassifier(
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
    """NLN über den ScoredRuleSetClassifier-Wrapper auf Breast Cancer."""
    print("\n" + "=" * 70)
    print("  Demo 2: NLN-Wrapper auf Breast Cancer (binär)")
    print("=" * 70)

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y,
    )

    clf = ScoredRuleSetClassifier(
        backend="nln",
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
    """Zeigt den Effekt der L1-Regularisierung auf die Regelkomplexität."""
    print("\n" + "=" * 70)
    print("  Demo 3: L1-Regularisierung und Sparsity")
    print("=" * 70)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y,
    )

    configs = [
        ("Schwach (l1=0.0005)", 0.0005, 0.0002),
        ("Mittel (l1=0.005)",   0.005,  0.002),
        ("Stark (l1=0.05)",     0.05,   0.02),
    ]

    for label, l1_c, l1_s in configs:
        clf = NeuralLogicNetClassifier(
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

        print(f"  {label:30s}  → {n_rules:2d} Regeln, {n_atoms:3d} Atome, F1={f1:.4f}")


if __name__ == "__main__":
    demo_native_iris()
    demo_wrapper_breast_cancer()
    demo_sparsity_control()
    print("\nDone.")

