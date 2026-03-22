"""Tests for the RuleFit estimator backend and ScoredRuleSet transformation."""

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split(dataset_loader, random_state=42):
    X, y = dataset_loader(return_X_y=True)
    return train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=y)


# ---------------------------------------------------------------------------
# Direct RuleFitClassifier tests
# ---------------------------------------------------------------------------


class TestRuleFitClassifierDirect:
    """Test RuleFitClassifier without the ScoredRuleSetClassifier wrapper."""

    def test_binary_fit_predict(self):
        X_train, X_test, y_train, y_test = _split(load_breast_cancer)
        from scoredrulesets.estimators.rulefit import RuleFitClassifier

        clf = RuleFitClassifier(max_rules=20, tree_size=4, random_state=0)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro")
        assert f1 > 0.80, f"Binary F1 too low: {f1:.4f}"

    def test_multiclass_fit_predict(self):
        X_train, X_test, y_train, y_test = _split(load_iris)
        from scoredrulesets.estimators.rulefit import RuleFitClassifier

        clf = RuleFitClassifier(max_rules=30, tree_size=4, random_state=0)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro")
        assert f1 > 0.70, f"Multiclass F1 too low: {f1:.4f}"

    def test_predict_proba_shape(self):
        X_train, X_test, y_train, y_test = _split(load_iris)
        from scoredrulesets.estimators.rulefit import RuleFitClassifier

        clf = RuleFitClassifier(max_rules=20, tree_size=4, random_state=0)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        assert proba.shape == (len(X_test), 3)
        # Probabilities should sum to ≈1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_get_rules(self):
        X_train, _, y_train, _ = _split(load_breast_cancer)
        from scoredrulesets.estimators.rulefit import RuleFitClassifier

        clf = RuleFitClassifier(max_rules=20, tree_size=4, random_state=0)
        clf.fit(X_train, y_train)
        rules_df = clf.get_rules(class_index=0)
        assert len(rules_df) > 0, "get_rules() returned empty DataFrame"
        assert "rule" in rules_df.columns
        assert "coef" in rules_df.columns


# ---------------------------------------------------------------------------
# ScoredRuleSet transformation tests
# ---------------------------------------------------------------------------


class TestRuleFitTransformation:
    """Test the RuleFitClassifier → ScoredRuleSet transformation."""

    def test_binary_to_ruleset(self):
        X_train, X_test, y_train, y_test = _split(load_breast_cancer)
        from scoredrulesets.estimators.rulefit import RuleFitClassifier

        clf = RuleFitClassifier(max_rules=30, tree_size=4, random_state=0)
        clf.fit(X_train, y_train)

        ruleset = clf.to_ruleset()
        assert len(ruleset.rules) > 0
        assert len(ruleset.class_labels) == 2
        # All rules must have correct number of scores
        for rule in ruleset.rules:
            assert len(rule.scores) == 2

    def test_multiclass_to_ruleset(self):
        X_train, X_test, y_train, y_test = _split(load_iris)
        from scoredrulesets.estimators.rulefit import RuleFitClassifier

        clf = RuleFitClassifier(max_rules=30, tree_size=4, random_state=0)
        clf.fit(X_train, y_train)

        ruleset = clf.to_ruleset()
        assert len(ruleset.rules) > 0
        assert len(ruleset.class_labels) == 3
        for rule in ruleset.rules:
            assert len(rule.scores) == 3

    def test_binary_transformation_quality(self):
        """ScoredRuleSet F1 should be close to native F1 (lossy but bounded)."""
        X_train, X_test, y_train, y_test = _split(load_breast_cancer)
        from scoredrulesets.estimators.rulefit import RuleFitClassifier
        from scoredrulesets.runtime import predict as predict_from_ruleset

        clf = RuleFitClassifier(max_rules=30, tree_size=4, random_state=0)
        clf.fit(X_train, y_train)

        y_native = clf.predict(X_test)
        f1_native = f1_score(y_test, y_native, average="macro")

        ruleset = clf.to_ruleset()
        y_rs = predict_from_ruleset(ruleset, X_test)
        y_test_s = np.asarray(y_test, dtype=str)
        y_rs_s = np.asarray(y_rs, dtype=str)
        f1_rs = f1_score(y_test_s, y_rs_s, average="macro")

        # Tolerate up to 10% absolute loss (transformation is lossy)
        assert f1_rs > f1_native - 0.10, (
            f"ScoredRuleSet F1 too far below native: {f1_rs:.4f} vs {f1_native:.4f}"
        )

    def test_has_default_rule(self):
        X_train, _, y_train, _ = _split(load_breast_cancer)
        from scoredrulesets.estimators.rulefit import RuleFitClassifier

        clf = RuleFitClassifier(max_rules=20, tree_size=4, random_state=0)
        clf.fit(X_train, y_train)

        ruleset = clf.to_ruleset()
        default_rules = [r for r in ruleset.rules if len(r.atoms) == 0]
        assert len(default_rules) == 1, f"Expected 1 default rule, got {len(default_rules)}"

    def test_atom_operators_valid(self):
        """All atoms must use supported operators from schema.SUPPORTED_OPS."""
        from scoredrulesets.schema import SUPPORTED_OPS
        X_train, _, y_train, _ = _split(load_iris)
        from scoredrulesets.estimators.rulefit import RuleFitClassifier

        clf = RuleFitClassifier(max_rules=30, tree_size=4, random_state=0)
        clf.fit(X_train, y_train)

        ruleset = clf.to_ruleset()
        for rule in ruleset.rules:
            for atom in rule.atoms:
                assert atom.op in SUPPORTED_OPS, (
                    f"Atom op '{atom.op}' not in SUPPORTED_OPS"
                )


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


class TestRuleFitJSONRoundTrip:
    """Test save/load of RuleFit ScoredRuleSets."""

    def test_json_roundtrip_binary(self, tmp_path):
        X_train, X_test, y_train, y_test = _split(load_breast_cancer)
        from scoredrulesets.estimators.rulefit import RuleFitClassifier
        from scoredrulesets.io import dump_ruleset_json, load_ruleset_json
        from scoredrulesets.runtime import predict as predict_from_ruleset

        clf = RuleFitClassifier(max_rules=20, tree_size=4, random_state=0)
        clf.fit(X_train, y_train)
        ruleset = clf.to_ruleset()

        path = tmp_path / "rulefit_binary.json"
        dump_ruleset_json(ruleset, path)
        loaded = load_ruleset_json(path)

        y_orig = predict_from_ruleset(ruleset, X_test)
        y_loaded = predict_from_ruleset(loaded, X_test)
        np.testing.assert_array_equal(y_orig, y_loaded)

    def test_json_roundtrip_multiclass(self, tmp_path):
        X_train, X_test, y_train, y_test = _split(load_iris)
        from scoredrulesets.estimators.rulefit import RuleFitClassifier
        from scoredrulesets.io import dump_ruleset_json, load_ruleset_json
        from scoredrulesets.runtime import predict as predict_from_ruleset

        clf = RuleFitClassifier(max_rules=20, tree_size=4, random_state=0)
        clf.fit(X_train, y_train)
        ruleset = clf.to_ruleset()

        path = tmp_path / "rulefit_multi.json"
        dump_ruleset_json(ruleset, path)
        loaded = load_ruleset_json(path)

        y_orig = predict_from_ruleset(ruleset, X_test)
        y_loaded = predict_from_ruleset(loaded, X_test)
        np.testing.assert_array_equal(y_orig, y_loaded)


# ---------------------------------------------------------------------------
# Wrapper integration (ScoredRuleSetClassifier with backend='rulefit')
# ---------------------------------------------------------------------------


class TestRuleFitWrapper:
    """Test via the main ScoredRuleSetClassifier entry point."""

    def test_wrapper_binary(self):
        X_train, X_test, y_train, y_test = _split(load_breast_cancer)
        from scoredrulesets import ScoredRuleSetClassifier

        clf = ScoredRuleSetClassifier(
            backend="rulefit",
            backend_params={"max_rules": 30, "tree_size": 4},
            random_state=0,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro")
        assert f1 > 0.80, f"Wrapper binary F1 too low: {f1:.4f}"

        ruleset = clf.to_ruleset()
        assert len(ruleset.rules) > 0
        assert ruleset.metadata.get("transform") == "rulefit_to_scored_ruleset"

    def test_wrapper_multiclass(self):
        X_train, X_test, y_train, y_test = _split(load_iris)
        from scoredrulesets import ScoredRuleSetClassifier

        clf = ScoredRuleSetClassifier(
            backend="rulefit",
            backend_params={"max_rules": 30, "tree_size": 4},
            random_state=0,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro")
        assert f1 > 0.60, f"Wrapper multiclass F1 too low: {f1:.4f}"

    def test_wrapper_save_and_reload(self, tmp_path):
        X_train, X_test, y_train, y_test = _split(load_iris)
        from scoredrulesets import ScoredRuleSetClassifier

        clf = ScoredRuleSetClassifier(
            backend="rulefit",
            backend_params={"max_rules": 20, "tree_size": 4},
            random_state=0,
        )
        clf.fit(X_train, y_train)
        y_pred_orig = clf.predict(X_test)

        path = tmp_path / "wrapper_rulefit.json"
        clf.save_ruleset(str(path))

        clf2 = ScoredRuleSetClassifier.from_ruleset_json(str(path))
        y_pred_loaded = clf2.predict(X_test)
        np.testing.assert_array_equal(y_pred_orig, y_pred_loaded)

    def test_wrapper_lossy_flag(self):
        """RuleFit wrapper should set transformation_lossy_=True."""
        X_train, _, y_train, _ = _split(load_iris)
        from scoredrulesets import ScoredRuleSetClassifier

        clf = ScoredRuleSetClassifier(
            backend="rulefit",
            backend_params={"max_rules": 20, "tree_size": 4},
            random_state=0,
        )
        clf.fit(X_train, y_train)
        assert getattr(clf, "transformation_lossy_", False) is True


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------


class TestRuleFitFormatting:
    """Test console table output for RuleFit rule sets."""

    def test_format_table(self):
        X_train, _, y_train, _ = _split(load_breast_cancer)
        from scoredrulesets.estimators.rulefit import RuleFitClassifier
        from scoredrulesets.formatting import format_ruleset_table

        clf = RuleFitClassifier(max_rules=20, tree_size=4, random_state=0)
        clf.fit(X_train, y_train)
        ruleset = clf.to_ruleset()

        table = format_ruleset_table(ruleset)
        assert len(table) > 0
        assert "rulefit_default" in table
        # Should contain header row markers
        assert "+" in table or "|" in table or "rule_id" in table

