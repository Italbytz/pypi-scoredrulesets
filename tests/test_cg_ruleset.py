"""Unit tests for Column Generation backend (CG-Rule)."""

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_iris, make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from scoredrulesets.estimators import ColumnGenerationClassifier, ScoredRuleSetClassifier
from scoredrulesets.runtime import predict as predict_from_ruleset
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
