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


def test_gpas_selector_shortest_zero_train_mcr_prefers_shortest_perfect_train_model():
    from scoredrulesets.estimators.logicgp import (
        _Fitness,
        _Monomial,
        _Polynomial,
        _SetLiteral,
        _select_model_shortest_zero_train_mcr,
    )

    lit_a = _SetLiteral(0, frozenset({0}), (0, 1))
    lit_b = _SetLiteral(1, frozenset({1}), (0, 1))
    unif = np.array([0.5, 0.5], dtype=float)

    small = _Polynomial(monomials=[_Monomial([lit_a], unif.copy())], default_weights=unif.copy())
    large = _Polynomial(
        monomials=[_Monomial([lit_a, lit_b], unif.copy())],
        default_weights=unif.copy(),
    )

    fit_small = _Fitness(objectives=np.array([1.0, 1.0]), size=small.size)
    fit_large = _Fitness(objectives=np.array([1.0, 1.0]), size=large.size)

    candidates = [
        (large, fit_large, 0.95, 0.0),
        (small, fit_small, 0.90, 0.0),
    ]

    selected = _select_model_shortest_zero_train_mcr(candidates)
    assert selected is small


def test_gpas_selector_shortest_zero_train_mcr_falls_back_when_no_perfect_train_model():
    from scoredrulesets.estimators.logicgp import (
        _Fitness,
        _Monomial,
        _Polynomial,
        _SetLiteral,
        _select_model_shortest_zero_train_mcr,
    )

    lit_a = _SetLiteral(0, frozenset({0}), (0, 1))
    lit_b = _SetLiteral(1, frozenset({1}), (0, 1))
    unif = np.array([0.5, 0.5], dtype=float)

    small = _Polynomial(monomials=[_Monomial([lit_a], unif.copy())], default_weights=unif.copy())
    large = _Polynomial(
        monomials=[_Monomial([lit_a, lit_b], unif.copy())],
        default_weights=unif.copy(),
    )

    fit_small = _Fitness(objectives=np.array([0.8, 0.8]), size=small.size)
    fit_large = _Fitness(objectives=np.array([0.9, 0.9]), size=large.size)

    candidates = [
        (small, fit_small, 0.70, 0.10),
        (large, fit_large, 0.85, 0.05),
    ]

    selected = _select_model_shortest_zero_train_mcr(candidates)
    assert selected is large


def test_gpas_accepts_shortest_zero_train_mcr_mode_and_exposes_metadata():
    from scoredrulesets import GPASClassifier

    X, y = _make_binary_data(n=40, n_features=4, seed=123)
    clf = GPASClassifier(
        max_generations=10,
        stagnation_generations=3,
        n_bins=3,
        model_selection="shortest_zero_train_mcr",
        random_state=11,
    )
    clf.fit(X, y)

    meta = clf.to_ruleset().metadata
    assert meta["model_selection"] == "shortest_zero_train_mcr"
