"""
Tests for GPASClassifier (binary logicGP / GPAS variant).
"""
from __future__ import annotations

import numpy as np
import pytest


def _make_binary_data(n: int = 60, n_features: int = 4, seed: int = 42):
    """Small binary classification dataset with categorical features."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 3, size=(n, n_features)).astype(object)
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


# ---------------------------------------------------------------------------
# Basic fit / predict
# ---------------------------------------------------------------------------

def test_gpas_fit_predict():
    from scoredrulesets import GPASClassifier

    X, y = _make_binary_data()
    clf = GPASClassifier(
        max_generations=20,
        stagnation_generations=5,
        n_bins=3,
        random_state=0,
    )
    clf.fit(X, y)

    pred = clf.predict(X[:10])
    proba = clf.predict_proba(X[:10])

    assert pred.shape == (10,)
    assert proba.shape == (10, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(pred).issubset({0, 1})


def test_gpas_ruleset_structure():
    from scoredrulesets import GPASClassifier

    X, y = _make_binary_data()
    clf = GPASClassifier(
        max_generations=15,
        stagnation_generations=5,
        n_bins=3,
        random_state=1,
    )
    clf.fit(X, y)

    rs = clf.to_ruleset()
    assert rs.metadata["source"] == "logicgp"
    assert rs.metadata["trainer"] == "flcw"
    assert rs.metadata["f1_averaging"] == "micro"
    assert len(rs.class_labels) == 2
    assert any(r.rule_id == "logicgp_default" for r in rs.rules)


def test_gpas_initial_monomials_have_two_literals():
    """Verify that the GPAS initialization produces 2-literal monomials."""
    from scoredrulesets.estimators.logicgp import (
        GPASClassifier,
        _generate_literals,
        _discretize_features,
    )

    X, y = _make_binary_data(n=40, n_features=4)
    clf = GPASClassifier(max_generations=1, stagnation_generations=1, random_state=7)

    # Replicate the internal setup to inspect the initial population.
    clf._rng_ = np.random.default_rng(7)
    X_disc, _, _ = _discretize_features(X, n_bins=3)
    literals = _generate_literals(X_disc)
    population = clf._init_population(literals, n_classes=2, X_disc=X_disc,
                                      y_idx=np.where(y == 0, 0, 1).astype(int))

    # Every non-seed polynomial should have exactly one monomial with 2 literals.
    # (Seeds may differ, so we only check the first len(literals) entries.)
    n_seed_offset = len(literals)
    for poly in population[:n_seed_offset]:
        assert len(poly.monomials) == 1, "Initial GPAS individual should have 1 monomial"
        assert len(poly.monomials[0].literals) == 2, (
            "Initial GPAS monomial should have exactly 2 literals"
        )


# ---------------------------------------------------------------------------
# Binary-only constraint
# ---------------------------------------------------------------------------

def test_gpas_rejects_multiclass():
    from scoredrulesets import GPASClassifier

    X = np.array([[0, 1], [1, 0], [2, 2], [0, 2]], dtype=object)
    y = np.array([0, 1, 2, 0])

    clf = GPASClassifier(max_generations=5, random_state=0)
    with pytest.raises(ValueError, match="2 classes"):
        clf.fit(X, y)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_gpas_reproducible():
    from scoredrulesets import GPASClassifier

    X, y = _make_binary_data()

    def run():
        clf = GPASClassifier(
            max_generations=10, stagnation_generations=3, n_bins=3, random_state=99
        )
        clf.fit(X, y)
        return clf.predict(X).tolist()

    assert run() == run()
