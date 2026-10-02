"""Tests for AutoScoredRuleSetClassifier (AutoML meta-estimator)."""

import numpy as np
import pytest
from sklearn.datasets import load_iris

from scoredrulesets.estimators.auto import AutoScoredRuleSetClassifier


@pytest.fixture
def iris_data():
    X, y = load_iris(return_X_y=True)
    return X, y


class TestAutoEstimator:

    def test_basic_fit_predict(self, iris_data):
        X, y = iris_data
        clf = AutoScoredRuleSetClassifier(
            candidate_backends=["cart", "hs"],
            cv=3,
            scoring="f1_weighted",
            random_state=0,
        )
        clf.fit(X, y)

        assert clf.best_backend_ in ("cart", "hs")
        assert isinstance(clf.cv_results_, dict)
        assert len(clf.cv_results_) == 2
        assert clf.best_score_ > 0

        preds = clf.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset(set(y))

    def test_to_ruleset(self, iris_data):
        X, y = iris_data
        clf = AutoScoredRuleSetClassifier(
            candidate_backends=["cart"],
            cv=2,
            random_state=0,
        )
        clf.fit(X, y)
        ruleset = clf.to_ruleset()
        assert ruleset is not None
        assert len(ruleset.rules) > 0

    def test_predict_proba(self, iris_data):
        X, y = iris_data
        clf = AutoScoredRuleSetClassifier(
            candidate_backends=["cart", "hs"],
            cv=2,
            random_state=0,
        )
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(y), 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=0.01)

    def test_with_preprocessing(self, iris_data):
        X, y = iris_data
        clf = AutoScoredRuleSetClassifier(
            candidate_backends=["cart"],
            cv=2,
            preprocessing={"feature_selection": "kbest", "k": 2},
            random_state=0,
        )
        clf.fit(X, y)
        assert clf.best_backend_ == "cart"
        preds = clf.predict(X)
        assert len(preds) == len(y)

    def test_per_backend_params(self, iris_data):
        X, y = iris_data
        clf = AutoScoredRuleSetClassifier(
            candidate_backends=["cart", "ruleplcs"],
            backend_params={
                "cart": {"max_depth": 2},
                "ruleplcs": {"max_rules": 3},
            },
            cv=2,
            random_state=0,
        )
        clf.fit(X, y)
        assert clf.best_backend_ in ("cart", "ruleplcs")

    def test_failing_backend_skipped(self, iris_data):
        """A backend that fails during CV should be skipped with a warning."""
        X, y = iris_data
        clf = AutoScoredRuleSetClassifier(
            candidate_backends=["cart", "nonexistent_backend"],
            cv=2,
            random_state=0,
        )
        with pytest.warns(UserWarning, match="failed during CV"):
            clf.fit(X, y)

        assert clf.best_backend_ == "cart"
        assert clf.cv_results_["nonexistent_backend"] == float("-inf")

    def test_default_backends(self, iris_data):
        """Default candidate_backends should work when no backends specified."""
        X, y = iris_data
        clf = AutoScoredRuleSetClassifier(cv=2, random_state=0)
        clf.fit(X, y)
        assert clf.best_backend_ in ("greedy_pareto", "cart", "hs", "ruleplcs")

    def test_pareto_fusion_and_spectrum(self, iris_data):
        """Test Master Pareto Archive spectrum, active model switching, and index card export."""
        X, y = iris_data
        clf = AutoScoredRuleSetClassifier(
            candidate_backends=["greedy_pareto", "cart"],
            preference="balanced",
            enable_pareto_fusion=True,
            cv=2,
            random_state=0,
        )
        clf.fit(X, y)

        spectrum = clf.get_pareto_spectrum()
        assert isinstance(spectrum, list)
        assert len(spectrum) >= 1

        # Check structure of spectrum entries
        first = spectrum[0]
        assert "complexity" in first
        assert "score" in first
        assert "backend" in first
        assert "ruleset" in first

        # Test index card export
        card = clf.export_index_card()
        assert isinstance(card, str)
        assert len(card) > 0

        # Test active model switching
        orig_ruleset = clf.to_ruleset()
        target_complexity = spectrum[-1]["complexity"]
        clf.set_active_model(target_complexity)
        assert clf.to_ruleset() is not None

        # Verify predictions work
        preds = clf.predict(X)
        assert len(preds) == len(y)

    def test_preferences(self, iris_data):
        """Test different user intent profiles ('compact', 'balanced', 'accuracy')."""
        X, y = iris_data
        for pref in ("compact", "balanced", "accuracy"):
            clf = AutoScoredRuleSetClassifier(
                candidate_backends=["greedy_pareto", "cart"],
                preference=pref,
                enable_pareto_fusion=True,
                cv=2,
                random_state=0,
            )
            clf.fit(X, y)
            preds = clf.predict(X)
            assert len(preds) == len(y)

    def test_auto_regressor(self):
        """Test AutoScoredRuleSetRegressor on continuous target."""
        from sklearn.datasets import make_regression
        from scoredrulesets.estimators.auto import AutoScoredRuleSetRegressor

        X, y = make_regression(n_samples=60, n_features=4, noise=0.1, random_state=42)
        reg = AutoScoredRuleSetRegressor(
            candidate_backends=["greedy_cascaded", "cart"],
            cv=2,
            random_state=0,
        )
        reg.fit(X, y)

        assert reg.best_backend_ in ("greedy_cascaded", "cart")
        assert reg.ruleset_ is not None
        preds = reg.predict(X)
        assert len(preds) == len(y)

        spectrum = reg.get_pareto_spectrum()
        assert isinstance(spectrum, list)
        assert len(spectrum) >= 1


