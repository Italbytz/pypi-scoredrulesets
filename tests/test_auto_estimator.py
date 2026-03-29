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
        assert clf.best_backend_ in ("cart", "hs", "ruleplcs")


