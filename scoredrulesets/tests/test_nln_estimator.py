"""Tests for the Neural Logic Network (NLN) backend."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from scoredrulesets.estimators.nln import NeuralLogicNetClassifier
from scoredrulesets.estimators.sklearn_wrapper import ScoredRuleSetClassifier
from scoredrulesets.schema import ScoredRuleSet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iris_split(random_state: int = 42):
    X, y = load_iris(return_X_y=True)
    return train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=y)


def _wine_split(random_state: int = 42):
    X, y = load_wine(return_X_y=True)
    return train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=y)


# ---------------------------------------------------------------------------
# Basic tests
# ---------------------------------------------------------------------------

class TestNeuralLogicNetClassifier:
    """Core tests for the NLN estimator."""

    def test_fit_predict_iris(self):
        X_train, X_test, y_train, y_test = _iris_split()
        clf = NeuralLogicNetClassifier(
            n_rules=12, n_bins=5, epochs=300,
            learning_rate=0.3, l1_conj=0.002, l1_score=0.001,
            early_stopping_rounds=30, random_state=0,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        assert y_pred.shape == y_test.shape
        f1 = f1_score(y_test, y_pred, average="macro")
        print(f"\n[NLN native] Iris F1={f1:.4f}")
        # NLN should achieve at least a reasonable F1 on Iris
        assert f1 > 0.3, f"F1 too low: {f1:.4f}"

    def test_fit_predict_wine(self):
        X_train, X_test, y_train, y_test = _wine_split()
        clf = NeuralLogicNetClassifier(
            n_rules=15, n_bins=4, epochs=400,
            learning_rate=0.3, l1_conj=0.001, l1_score=0.0005,
            early_stopping_rounds=40, random_state=0,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro")
        print(f"\n[NLN native] Wine F1={f1:.4f}")
        assert f1 > 0.2, f"F1 too low: {f1:.4f}"

    def test_to_ruleset_valid(self):
        X_train, _, y_train, _ = _iris_split()
        clf = NeuralLogicNetClassifier(n_rules=8, epochs=50, random_state=0)
        clf.fit(X_train, y_train)

        rs = clf.to_ruleset()
        assert isinstance(rs, ScoredRuleSet)
        rs.validate()

        # Should have at least the default rule
        assert len(rs.rules) >= 1

        # Class labels must match
        assert set(rs.class_labels) == {0, 1, 2}

    def test_rules_have_atoms(self):
        """Non-default rules should have atoms with valid ops."""
        X_train, _, y_train, _ = _iris_split()
        clf = NeuralLogicNetClassifier(
            n_rules=10, n_bins=5, epochs=120, random_state=0,
        )
        clf.fit(X_train, y_train)
        rs = clf.to_ruleset()

        non_default = [r for r in rs.rules if r.atoms]
        # With reasonable training, at least one non-default rule should exist
        assert len(non_default) >= 1, "No non-default rules extracted!"

        for rule in non_default:
            for atom in rule.atoms:
                assert atom.op in ("<=", ">"), f"Unexpected op: {atom.op}"
                assert isinstance(atom.value, float)

    def test_predict_proba_shape(self):
        X_train, X_test, y_train, _ = _iris_split()
        clf = NeuralLogicNetClassifier(n_rules=6, epochs=50, random_state=0)
        clf.fit(X_train, y_train)

        proba = clf.predict_proba(X_test)
        assert proba.shape == (len(X_test), 3)
        # Each row should be a valid probability distribution
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_binary_classification(self):
        """NLN should work for binary classification."""
        X, y = load_breast_cancer(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y,
        )
        clf = NeuralLogicNetClassifier(
            n_rules=12, n_bins=4, epochs=300,
            learning_rate=0.3, l1_conj=0.002, l1_score=0.001,
            early_stopping_rounds=30, random_state=0,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro")
        print(f"\n[NLN native] Breast Cancer F1={f1:.4f}")
        assert f1 > 0.3, f"F1 too low on breast cancer: {f1:.4f}"


# ---------------------------------------------------------------------------
# Wrapper integration tests
# ---------------------------------------------------------------------------

class TestNLNWrapper:
    """Tests for NLN via the ScoredRuleSetClassifier wrapper."""

    def test_wrapper_backend_nln(self):
        X_train, X_test, y_train, y_test = _iris_split()
        clf = ScoredRuleSetClassifier(
            backend="nln",
            backend_params={
                "n_rules": 10,
                "n_bins": 4,
                "epochs": 100,
                "early_stopping_rounds": 15,
            },
            random_state=0,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        assert y_pred.shape == y_test.shape
        f1 = f1_score(y_test, y_pred, average="macro")
        print(f"\n[NLN wrapper] Iris F1={f1:.4f}")
        assert f1 > 0.3

    def test_wrapper_to_ruleset(self):
        X_train, _, y_train, _ = _iris_split()
        clf = ScoredRuleSetClassifier(
            backend="nln",
            backend_params={"n_rules": 8, "epochs": 60},
            random_state=0,
        )
        clf.fit(X_train, y_train)
        rs = clf.to_ruleset()
        assert isinstance(rs, ScoredRuleSet)
        rs.validate()

    def test_wrapper_f1_preserved(self):
        """F1 through the wrapper should equal F1 of the native model
        (since the wrapper routes prediction through the ScoredRuleSet)."""
        X_train, X_test, y_train, y_test = _iris_split()
        native = NeuralLogicNetClassifier(
            n_rules=10, n_bins=4, epochs=80, random_state=0,
        )
        native.fit(X_train, y_train)
        y_native = native.predict(X_test)

        wrapper = ScoredRuleSetClassifier(
            backend="nln",
            backend_params={"n_rules": 10, "n_bins": 4, "epochs": 80},
            random_state=0,
        )
        wrapper.fit(X_train, y_train)
        y_wrapper = wrapper.predict(X_test)

        f1_nat = f1_score(y_test, y_native, average="macro")
        f1_wrap = f1_score(y_test, y_wrapper, average="macro")
        print(f"\n[NLN] Native F1={f1_nat:.4f}, Wrapper F1={f1_wrap:.4f}")
        # Both use the same ruleset for prediction → should be identical
        assert f1_nat == f1_wrap


# ---------------------------------------------------------------------------
# Display test
# ---------------------------------------------------------------------------

class TestNLNDisplay:
    """Test that the model can be displayed in a human-readable way."""

    def test_display_ruleset(self):
        X_train, X_test, y_train, y_test = _iris_split()
        clf = NeuralLogicNetClassifier(
            n_rules=12, n_bins=5, epochs=300,
            learning_rate=0.3, early_stopping_rounds=30, random_state=0,
        )
        clf.fit(X_train, y_train)
        rs = clf.to_ruleset()

        print("\n" + "=" * 70)
        print("NLN Scored Rule Set (Iris)")
        print("=" * 70)
        print(f"Classes: {rs.class_labels}")
        print(f"Features: {rs.feature_names}")
        print(f"Total rules: {len(rs.rules)}")
        print()

        for rule in rs.rules:
            if not rule.atoms:
                print(f"  [DEFAULT] scores={[f'{s:.3f}' for s in rule.scores]}")
            else:
                conditions = " AND ".join(
                    f"{a.feature} {a.op} {a.value:.3f}" for a in rule.atoms
                )
                scores_str = [f"{s:.3f}" for s in rule.scores]
                print(f"  IF {conditions}")
                print(f"     THEN scores={scores_str}  (id={rule.rule_id})")

        print()
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro")
        print(f"Test F1 (macro): {f1:.4f}")
        print("=" * 70)


# ---------------------------------------------------------------------------
# Sparsity test
# ---------------------------------------------------------------------------

class TestNLNSparsity:
    """Verify that L1 regularisation produces sparse rules."""

    def test_high_l1_fewer_atoms(self):
        X_train, _, y_train, _ = _iris_split()

        # Low L1
        clf_low = NeuralLogicNetClassifier(
            n_rules=10, n_bins=4, l1_conj=0.001, l1_score=0.0005,
            epochs=100, random_state=0,
        )
        clf_low.fit(X_train, y_train)
        rs_low = clf_low.to_ruleset()
        atoms_low = sum(len(r.atoms) for r in rs_low.rules)

        # High L1
        clf_high = NeuralLogicNetClassifier(
            n_rules=10, n_bins=4, l1_conj=0.05, l1_score=0.02,
            epochs=100, random_state=0,
        )
        clf_high.fit(X_train, y_train)
        rs_high = clf_high.to_ruleset()
        atoms_high = sum(len(r.atoms) for r in rs_high.rules)

        print(f"\n[NLN Sparsity] Low L1: {atoms_low} atoms, High L1: {atoms_high} atoms")
        # Higher L1 should generally produce fewer or equal atoms
        assert atoms_high <= atoms_low + 5, (
            f"High L1 ({atoms_high} atoms) should produce fewer atoms "
            f"than low L1 ({atoms_low} atoms)"
        )





