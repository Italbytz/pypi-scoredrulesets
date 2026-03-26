"""Tests for preprocessing (feature selection, threshold budget) in ScoredRuleSetClassifier."""

import numpy as np
import pytest
from sklearn.datasets import load_iris

from scoredrulesets.estimators.sklearn_wrapper import ScoredRuleSetClassifier


@pytest.fixture
def iris_data():
    X, y = load_iris(return_X_y=True)
    return X, y


# ---------------------------------------------------------------------------
# Feature Selection (Step 1)
# ---------------------------------------------------------------------------


class TestFeatureSelection:
    """Feature selection via preprocessing dict."""

    def test_kbest_reduces_features(self, iris_data):
        X, y = iris_data
        clf = ScoredRuleSetClassifier(
            backend="cart",
            backend_params={"max_depth": 3},
            preprocessing={"feature_selection": "kbest", "k": 2},
            random_state=0,
        )
        clf.fit(X, y)

        # Should have stored selected feature indices
        assert clf.selected_feature_indices_ is not None
        assert len(clf.selected_feature_indices_) == 2

        # Feature names should reflect only selected features
        assert len(clf.feature_names_in_) == 2

        # Predict should work with FULL-width X (slicing happens internally)
        preds = clf.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset(set(y))

    def test_no_preprocessing_passes_all_features(self, iris_data):
        X, y = iris_data
        clf = ScoredRuleSetClassifier(
            backend="cart",
            backend_params={"max_depth": 3},
            random_state=0,
        )
        clf.fit(X, y)

        assert clf.selected_feature_indices_ is None
        assert len(clf.feature_names_in_) == X.shape[1]

    def test_kbest_k_larger_than_features(self, iris_data):
        """k > n_features should keep all features."""
        X, y = iris_data
        clf = ScoredRuleSetClassifier(
            backend="cart",
            backend_params={"max_depth": 3},
            preprocessing={"feature_selection": "kbest", "k": 100},
            random_state=0,
        )
        clf.fit(X, y)
        # Should keep all 4 features
        assert len(clf.feature_names_in_) == X.shape[1]

    def test_predict_proba_with_feature_selection(self, iris_data):
        X, y = iris_data
        clf = ScoredRuleSetClassifier(
            backend="cart",
            backend_params={"max_depth": 3},
            preprocessing={"feature_selection": "kbest", "k": 2},
            random_state=0,
        )
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(y), 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=0.01)


# ---------------------------------------------------------------------------
# Threshold Budget (Step 2)
# ---------------------------------------------------------------------------


class TestThresholdBudget:
    """max_thresholds_per_feature for native backends."""


    def test_rulelcs_threshold_cap_fit(self, iris_data):
        """RuleLCSClassifier with max_thresholds_per_feature fits OK."""
        X, y = iris_data
        clf = ScoredRuleSetClassifier(
            backend="rulelcs",
            backend_params={"max_thresholds_per_feature": 3, "max_rules": 4},
            random_state=0,
        )
        clf.fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == len(y)

    def test_nln_threshold_cap(self, iris_data):
        """NLN respects max_thresholds_per_feature."""
        from scoredrulesets.estimators.rulenln import RuleNLNClassifier

        X, y = iris_data
        clf = RuleNLNClassifier(
            n_bins=10,  # would normally produce many thresholds
            max_thresholds_per_feature=3,
            epochs=5,
            random_state=0,
        )
        clf.fit(X, y)
        for thr in clf._thresholds_:
            assert len(thr) <= 3

    def test_preprocessing_threshold_via_wrapper(self, iris_data):
        """max_thresholds_per_feature set via preprocessing dict."""
        X, y = iris_data
        clf = ScoredRuleSetClassifier(
            backend="rulelcs",
            backend_params={"max_rules": 4},
            preprocessing={"max_thresholds_per_feature": 3},
            random_state=0,
        )
        clf.fit(X, y)
        # Should fit without error and produce predictions
        preds = clf.predict(X)
        assert len(preds) == len(y)


# ---------------------------------------------------------------------------
# Combined: Feature Selection + Threshold Budget
# ---------------------------------------------------------------------------


class TestCombinedPreprocessing:
    def test_feature_selection_and_threshold_budget(self, iris_data):
        X, y = iris_data
        clf = ScoredRuleSetClassifier(
            backend="rulelcs",
            backend_params={"max_rules": 4},
            preprocessing={
                "feature_selection": "kbest",
                "k": 2,
                "max_thresholds_per_feature": 3,
            },
            random_state=0,
        )
        clf.fit(X, y)
        assert clf.selected_feature_indices_ is not None
        assert len(clf.selected_feature_indices_) == 2
        preds = clf.predict(X)
        assert len(preds) == len(y)

