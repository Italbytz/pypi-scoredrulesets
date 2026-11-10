from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_iris, make_regression
from sklearn.metrics import f1_score, r2_score

from scoredrulesets.estimators import (
    AutoScoredRuleSetClassifier,
    AutoScoredRuleSetRegressor,
    ScoredRuleSetClassifier,
    ScoredRuleSetRegressor,
)
from scoredrulesets.schema import ScoredRuleSet


class TestBackendADD:
    """Test suite for the Additive Algebraic Decision Diagram (backend='add') integration."""

    def test_classifier_binary(self):
        data = load_breast_cancer()
        X, y = data.data[:150], data.target[:150]

        clf = ScoredRuleSetClassifier(
            backend="add",
            backend_params={"n_estimators": 10, "max_depth": 2, "random_state": 42},
            random_state=42,
        )
        clf.fit(X, y)

        assert hasattr(clf, "ruleset_")
        assert isinstance(clf.ruleset_, ScoredRuleSet)
        assert clf.ruleset_.task_type == "classification"
        assert clf.ruleset_.validate() is None

        # Verify predictions
        preds = clf.predict(X)
        assert len(preds) == len(y)
        f1 = f1_score(y, preds, average="macro")
        assert f1 > 0.80

        proba = clf.predict_proba(X)
        assert proba.shape == (len(y), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

        # Verify pareto archive was populated
        assert hasattr(clf, "pareto_archive_")
        assert isinstance(clf.pareto_archive_, dict)
        assert len(clf.pareto_archive_) >= 2

    def test_classifier_multiclass(self):
        data = load_iris()
        X, y = data.data, data.target

        clf = ScoredRuleSetClassifier(
            backend="add",
            backend_params={"n_estimators": 8, "max_depth": 2, "random_state": 42},
            random_state=42,
        )
        clf.fit(X, y)

        assert hasattr(clf, "ruleset_")
        assert isinstance(clf.ruleset_, ScoredRuleSet)
        assert clf.ruleset_.task_type == "classification"
        assert clf.ruleset_.aggregation.type == "argmax_sum"
        assert len(clf.ruleset_.class_labels) == 3
        assert clf.ruleset_.validate() is None

        preds = clf.predict(X)
        assert len(preds) == len(y)
        f1 = f1_score(y, preds, average="macro")
        assert f1 > 0.80

        proba = clf.predict_proba(X)
        assert proba.shape == (len(y), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_regressor(self):
        X, y = make_regression(n_samples=120, n_features=6, noise=0.1, random_state=42)

        reg = ScoredRuleSetRegressor(
            backend="add",
            backend_params={"n_estimators": 12, "max_depth": 2, "random_state": 42},
            random_state=42,
        )
        reg.fit(X, y)

        assert hasattr(reg, "ruleset_")
        assert isinstance(reg.ruleset_, ScoredRuleSet)
        assert reg.ruleset_.task_type == "regression"
        assert reg.ruleset_.validate() is None

        preds = reg.predict(X)
        assert len(preds) == len(y)
        r2 = r2_score(y, preds)
        assert r2 > 0.40

        # Verify pareto archive
        assert hasattr(reg, "pareto_archive_")
        assert len(reg.pareto_archive_) >= 2

    def test_auto_classifier_with_add(self):
        data = load_iris()
        X, y = data.data, data.target

        auto_clf = AutoScoredRuleSetClassifier(
            candidate_backends=["cart", "add"],
            cv=2,
            random_state=42,
        )
        auto_clf.fit(X, y)

        assert auto_clf.best_backend_ in ("cart", "add")
        assert len(auto_clf.master_archive_.candidates_) >= 1

        # Check if ADD contributed candidates to the harvested candidates pool
        add_harvested = [c for c in auto_clf.all_harvested_candidates_ if c.backend == "add"]
        assert len(add_harvested) >= 1

        preds = auto_clf.predict(X)
        assert len(preds) == len(y)
        assert f1_score(y, preds, average="macro") > 0.85

    def test_auto_classifier_pareto_front_on_wine(self):
        from sklearn.datasets import load_wine
        X, y = load_wine(return_X_y=True)

        auto_clf = AutoScoredRuleSetClassifier(
            candidate_backends=["cart", "add"],
            cv=2,
            random_state=42,
        )
        auto_clf.fit(X, y)

        assert auto_clf.best_backend_ == "add"
        add_candidates = [c for c in auto_clf.master_archive_.candidates_ if c.backend == "add"]
        assert len(add_candidates) >= 1

    def test_auto_regressor_with_add(self):
        X, y = make_regression(n_samples=80, n_features=5, noise=0.1, random_state=42)

        auto_reg = AutoScoredRuleSetRegressor(
            candidate_backends=["cart", "add"],
            cv=2,
            random_state=42,
        )
        auto_reg.fit(X, y)

        assert auto_reg.best_backend_ in ("cart", "add")
        assert len(auto_reg.master_archive_.candidates_) >= 1

        preds = auto_reg.predict(X)
        assert len(preds) == len(y)

    def test_memetic_add_classifier(self):
        data = load_breast_cancer()
        X, y = data.data[:100], data.target[:100]

        clf = ScoredRuleSetClassifier(
            backend="memetic_add",
            backend_params={"n_estimators": 4, "max_depth": 2, "population_size": 15, "n_generations": 5, "random_state": 42},
            random_state=42,
        )
        clf.fit(X, y)

        assert hasattr(clf, "ruleset_")
        assert isinstance(clf.ruleset_, ScoredRuleSet)
        assert clf.ruleset_.task_type == "classification"
        assert clf.ruleset_.validate() is None

        preds = clf.predict(X)
        assert len(preds) == len(y)
        f1 = f1_score(y, preds, average="macro")
        assert f1 > 0.80

        proba = clf.predict_proba(X)
        assert proba.shape == (len(y), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_memetic_add_regressor(self):
        X, y = make_regression(n_samples=80, n_features=5, noise=0.1, random_state=42)

        reg = ScoredRuleSetRegressor(
            backend="memetic_add",
            backend_params={"n_estimators": 4, "max_depth": 2, "population_size": 15, "n_generations": 5, "random_state": 42},
            random_state=42,
        )
        reg.fit(X, y)

        assert hasattr(reg, "ruleset_")
        assert isinstance(reg.ruleset_, ScoredRuleSet)
        assert reg.ruleset_.task_type == "regression"
        assert reg.ruleset_.validate() is None

        preds = reg.predict(X)
        assert len(preds) == len(y)
        r2 = r2_score(y, preds)
        assert r2 > 0.20
