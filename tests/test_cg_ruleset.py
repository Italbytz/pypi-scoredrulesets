"""Unit tests for Column Generation backend (CG-Rule)."""

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, make_classification, make_regression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from scoredrulesets.estimators import (
    ColumnGenerationClassifier,
    ColumnGenerationRegressor,
    RuleGPClassifier,
    ScoredRuleSetClassifier,
    ScoredRuleSetRegressor,
)
from scoredrulesets.runtime import predict as predict_from_ruleset
from scoredrulesets.runtime import predict_regression as predict_regression_from_ruleset
from scoredrulesets.schema import ScoredRuleSet


def test_cg_classifier_synthetic_binary():
    X, y = make_classification(
        n_samples=150,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=42,
    )
    clf = ColumnGenerationClassifier(
        max_rules=10,
        max_depth=3,
        min_support=3,
        c_reg=1.0,
        random_state=42,
    )
    clf.fit(X, y)

    rs = clf.to_ruleset()
    assert isinstance(rs, ScoredRuleSet)
    assert len(rs.rules) > 1

    preds = clf.predict(X)
    assert preds.shape == y.shape
    acc = accuracy_score(y, preds)
    assert acc > 0.70

    proba = clf.predict_proba(X)
    assert proba.shape == (150, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_cg_classifier_multiclass():
    iris = load_iris()
    clf = ColumnGenerationClassifier(
        max_rules=8,
        max_depth=2,
        min_support=3,
        c_reg=1.0,
        random_state=42,
    )
    clf.fit(iris.data, iris.target)

    preds = clf.predict(iris.data)
    acc = accuracy_score(iris.target, preds)
    assert acc > 0.85

    rs = clf.to_ruleset()
    assert rs.class_labels == [0, 1, 2]
    # Prediction directly via runtime
    runtime_preds = predict_from_ruleset(rs, iris.data)
    assert np.array_equal(preds, runtime_preds)


def test_cg_backend_via_scoredruleset_classifier():
    data = load_breast_cancer()
    X = data.data[:120]
    y = data.target[:120]

    clf = ScoredRuleSetClassifier(
        backend="cg",
        backend_params={"max_rules": 10, "max_depth": 2, "c_reg": 1.0},
        random_state=42,
    )
    clf.fit(X, y)

    assert hasattr(clf, "ruleset_")
    assert isinstance(clf.ruleset_, ScoredRuleSet)
    preds = clf.predict(X)
    assert len(preds) == len(y)
    acc = accuracy_score(y, preds)
    assert acc > 0.85


def test_cg_regressor_synthetic():
    X, y = make_regression(
        n_samples=150,
        n_features=6,
        n_informative=4,
        noise=0.1,
        random_state=42,
    )
    reg = ColumnGenerationRegressor(
        max_rules=15,
        max_depth=3,
        alpha=0.01,
        random_state=42,
    )
    reg.fit(X, y)

    rs = reg.to_ruleset()
    assert isinstance(rs, ScoredRuleSet)
    assert rs.task_type == "regression"
    assert len(rs.rules) > 1

    preds = reg.predict(X)
    assert preds.shape == y.shape
    r2 = r2_score(y, preds)
    assert r2 > 0.40, f"Expected R2 > 0.40, got {r2}"

    # Verify runtime prediction consistency
    runtime_preds = predict_regression_from_ruleset(rs, X)
    assert np.allclose(preds, runtime_preds)


def test_cg_regressor_backend_via_scoredruleset_regressor():
    data = load_diabetes()
    X = data.data[:120]
    y = data.target[:120]

    reg = ScoredRuleSetRegressor(
        backend="cg",
        backend_params={"max_rules": 12, "max_depth": 2, "alpha": 0.5},
        random_state=42,
    )
    reg.fit(X, y)

    assert hasattr(reg, "ruleset_")
    assert reg.ruleset_.task_type == "regression"
    preds = reg.predict(X)
    assert len(preds) == len(y)
    r2 = r2_score(y, preds)
    assert r2 > 0.25


def test_rulegp_warmstart_cg():
    data = load_breast_cancer()
    X = data.data[:100]
    y = data.target[:100]

    clf = RuleGPClassifier(
        max_generations=3,
        population_size=20,
        warmstart_strategy="cg",
        warmstart_max_rules=5,
        random_state=42,
    )
    clf.fit(X, y)

    preds = clf.predict(X)
    assert len(preds) == len(y)
    acc = accuracy_score(y, preds)
    assert acc > 0.85


def test_cg_classifier_pareto_archive():
    data = load_breast_cancer()
    X, y = data.data[:140], data.target[:140]
    clf = ColumnGenerationClassifier(max_rules=10, max_depth=2, c_reg=1.0, random_state=42)
    clf.fit(X, y)

    assert hasattr(clf, "pareto_archive_")
    assert len(clf.pareto_archive_) >= 2
    for n_atoms, rs in clf.pareto_archive_.items():
        assert isinstance(rs, ScoredRuleSet)
        assert sum(len(r.atoms) for r in rs.rules) == n_atoms


def test_cg_regressor_pareto_archive():
    data = load_diabetes()
    X, y = data.data[:140], data.target[:140]
    reg = ColumnGenerationRegressor(max_rules=12, max_depth=2, alpha=0.5, random_state=42)
    reg.fit(X, y)

    assert hasattr(reg, "pareto_archive_")
    assert len(reg.pareto_archive_) >= 2
    for n_atoms, rs in reg.pareto_archive_.items():
        assert isinstance(rs, ScoredRuleSet)
        assert rs.task_type == "regression"
        assert sum(len(r.atoms) for r in rs.rules) == n_atoms


def test_cg_in_automl():
    from scoredrulesets.estimators import AutoScoredRuleSetClassifier

    data = load_breast_cancer()
    X, y = data.data[:120], data.target[:120]
    auto_clf = AutoScoredRuleSetClassifier(
        candidate_backends=["cg"],
        cv=2,
        random_state=42,
    )
    auto_clf.fit(X, y)
    spectrum = auto_clf.get_pareto_spectrum()
    assert len(spectrum) >= 1
    assert accuracy_score(y, auto_clf.predict(X)) > 0.80


