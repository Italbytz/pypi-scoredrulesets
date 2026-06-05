"""Tests for preprocessing (feature selection, threshold budget) in ScoredRuleSetClassifier."""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer

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

    def test_custom_feature_selector_object(self, iris_data):
        X, y = iris_data
        selector = SelectKBest(score_func=f_classif, k=2)
        clf = ScoredRuleSetClassifier(
            backend="cart",
            backend_params={"max_depth": 3},
            preprocessing={"feature_selector": selector},
            random_state=0,
        )
        clf.fit(X, y)

        assert clf.feature_selector_ is not None
        assert len(clf.feature_names_in_) == 2
        preds = clf.predict(X)
        assert len(preds) == len(y)

    def test_feature_selection_params_are_applied(self, iris_data):
        X, y = iris_data
        clf = ScoredRuleSetClassifier(
            backend="cart",
            backend_params={"max_depth": 3},
            preprocessing={
                "feature_selection": "rfe",
                "k": 2,
                "feature_selection_params": {"step": 2},
            },
            random_state=0,
        )
        clf.fit(X, y)
        assert len(clf.feature_names_in_) == 2

    def test_feature_selection_and_feature_selector_are_mutually_exclusive(self, iris_data):
        X, y = iris_data
        with pytest.raises(ValueError, match="Use either"):
            ScoredRuleSetClassifier(
                backend="cart",
                backend_params={"max_depth": 3},
                preprocessing={
                    "feature_selection": "kbest",
                    "k": 2,
                    "feature_selector": SelectKBest(score_func=f_classif, k=2),
                },
                random_state=0,
            ).fit(X, y)


# ---------------------------------------------------------------------------
# Threshold Budget (Step 2)
# ---------------------------------------------------------------------------


class TestThresholdBudget:
    """max_thresholds_per_feature for native backends."""


    def test_ruleplcs_fit(self, iris_data):
        """RulePLCSClassifier (BioHEL) fits OK via wrapper."""
        X, y = iris_data
        clf = ScoredRuleSetClassifier(
            backend="ruleplcs",
            backend_params={
                "population_size": 50,
                "n_iterations": 10,
                "n_repetitions": 1,
                "max_rules": 4,
            },
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
            backend="ruleplcs",
            backend_params={"max_rules": 4},
            preprocessing={"max_thresholds_per_feature": 3},
            random_state=0,
        )
        clf.fit(X, y)
        # Should fit without error and produce predictions
        preds = clf.predict(X)
        assert len(preds) == len(y)

    def test_wrapper_max_fit_seconds_is_forwarded_to_ruleplcs(self, iris_data):
        X, y = iris_data
        clf = ScoredRuleSetClassifier(
            backend="ruleplcs",
            backend_params={
                "population_size": 20,
                "n_iterations": 5,
                "n_repetitions": 1,
                "max_rules": 3,
            },
            max_fit_seconds=1.25,
            random_state=0,
        )
        clf.fit(X, y)

        assert hasattr(clf.estimator_, "max_fit_seconds")
        assert clf.estimator_.max_fit_seconds == 1.25
        assert clf.to_ruleset().metadata["max_fit_seconds"] == 1.25


# ---------------------------------------------------------------------------
# Combined: Feature Selection + Threshold Budget
# ---------------------------------------------------------------------------


class TestCombinedPreprocessing:
    def test_feature_selection_and_threshold_budget(self, iris_data):
        X, y = iris_data
        clf = ScoredRuleSetClassifier(
            backend="ruleplcs",
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


class TestPipelineSteps:
    def test_pipeline_steps_impute_and_scale(self, iris_data):
        X, y = iris_data
        X_nan = X.copy().astype(float)
        X_nan[:10, 0] = np.nan

        clf = ScoredRuleSetClassifier(
            backend="cart",
            backend_params={"max_depth": 3},
            preprocessing={
                "pipeline_steps": [
                    {"name": "impute", "params": {"strategy": "median"}},
                    {"name": "standard_scale"},
                ]
            },
            random_state=0,
        )
        clf.fit(X_nan, y)
        assert clf.preprocess_pipeline_ is not None
        pred = clf.predict(X_nan)
        assert pred.shape == y.shape

    def test_pipeline_steps_support_custom_transformer(self, iris_data):
        X, y = iris_data
        clf = ScoredRuleSetClassifier(
            backend="cart",
            backend_params={"max_depth": 3},
            preprocessing={
                "pipeline_steps": [
                    {
                        "id": "identity",
                        "transformer": FunctionTransformer(lambda x: x, validate=False),
                    }
                ]
            },
            random_state=0,
        )
        clf.fit(X, y)
        pred = clf.predict(X)
        assert pred.shape == y.shape

    def test_pipeline_steps_allow_feature_count_changes(self, iris_data):
        X, y = iris_data
        clf = ScoredRuleSetClassifier(
            backend="cart",
            backend_params={"max_depth": 3},
            preprocessing={
                "pipeline_steps": [
                    {"transformer": PCA(n_components=2, random_state=0)}
                ]
            },
            random_state=0,
        )
        clf.fit(X, y)
        pred = clf.predict(X)
        assert pred.shape == y.shape
        assert len(clf.feature_names_in_) == 2

