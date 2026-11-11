"""Tests for RuleGP extensions (tree_splits, subsample, class_weight) via ScoredRuleSetClassifier."""

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, make_classification

from scoredrulesets import ScoredRuleSetClassifier


def test_rulegp_backend_tree_splits_and_subsample():
    X, y = make_classification(
        n_samples=250, n_features=6, n_informative=4, random_state=42
    )
    clf = ScoredRuleSetClassifier(
        backend="rulegp",
        backend_params={
            "max_generations": 25,
            "subsample": 100,
            "continuous_threshold_strategy": "tree_splits",
        },
        random_state=42,
    )
    clf.fit(X, y)
    assert hasattr(clf, "ruleset_")
    preds = clf.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})


def test_rulegp_backend_class_weight_balanced():
    # 90% Class 0, 10% Class 1
    X, y = make_classification(
        n_samples=300,
        n_features=6,
        weights=[0.9, 0.1],
        random_state=42,
    )
    clf = ScoredRuleSetClassifier(
        backend="rulegp",
        backend_params={
            "max_generations": 25,
            "class_weight": "balanced",
        },
        random_state=42,
    )
    clf.fit(X, y)
    assert hasattr(clf.estimator_, "threshold_")
    assert 0.0 < clf.estimator_.threshold_ < 1.0
    preds = clf.predict(X)
    assert len(preds) == len(y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(y), 2)


def test_logicgp_backend_forwarding():
    X, y = load_breast_cancer(return_X_y=True)
    clf = ScoredRuleSetClassifier(
        backend="logicgp",
        backend_params={
            "max_generations": 20,
            "subsample": 150,
            "class_weight": "balanced",
            "feature_encoding_strategy": "tree_splits",
        },
        random_state=42,
    )
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == len(y)
