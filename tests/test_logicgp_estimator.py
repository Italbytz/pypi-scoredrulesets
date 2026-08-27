"""
Tests for LogicGPClassifier (FLCW and RLCW variants).
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_iris


def _make_simple_data():
    """Small, quickly trainable dataset."""
    rng = np.random.default_rng(42)
    X = rng.integers(0, 3, size=(60, 4)).astype(object)
    y = np.array([0] * 20 + [1] * 20 + [2] * 20)
    return X, y


# ---------------------------------------------------------------------------
# Basic tests (FLCW)
# ---------------------------------------------------------------------------

def test_logicgp_flcw_fit_predict():
    from scoredrulesets import LogicGPClassifier

    X, y = _make_simple_data()
    clf = LogicGPClassifier(
        trainer="flcw",
        f1_averaging="macro",
        max_generations=20,
        stagnation_generations=5,
        n_bins=3,
        random_state=0,
    )
    clf.fit(X, y)

    pred = clf.predict(X[:10])
    proba = clf.predict_proba(X[:10])

    assert pred.shape == (10,)
    assert proba.shape == (10, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_logicgp_flcw_ruleset_structure():
    from scoredrulesets import LogicGPClassifier

    X, y = _make_simple_data()
    clf = LogicGPClassifier(
        max_generations=15,
        stagnation_generations=5,
        n_bins=3,
        random_state=1,
    )
    clf.fit(X, y)

    ruleset = clf.to_ruleset()
    assert ruleset.metadata["source"] == "logicgp"
    assert ruleset.metadata["trainer"] == "rlcw"
    assert ruleset.metadata["f1_averaging"] == "micro"
    assert any(r.rule_id == "logicgp_default" for r in ruleset.rules)
    assert len(ruleset.class_labels) == 3


def test_logicgp_flcw_metadata_fields():
    from scoredrulesets import LogicGPClassifier

    X, y = _make_simple_data()
    clf = LogicGPClassifier(
        max_generations=10,
        stagnation_generations=5,
        population_size=20,
        n_adaptations_per_gen=6,
        n_bins=3,
        random_state=2,
    )
    clf.fit(X, y)

    meta = clf.to_ruleset().metadata
    assert meta["population_size"] == 20
    assert meta["n_adaptations_per_gen"] == 6
    assert meta["max_model_size"] is None


# ---------------------------------------------------------------------------
# RLCW-specific tests
# ---------------------------------------------------------------------------

def test_logicgp_rlcw_macro_fit_predict():
    from scoredrulesets import LogicGPClassifier

    X, y = _make_simple_data()
    clf = LogicGPClassifier(
        trainer="rlcw",
        f1_averaging="macro",
        max_generations=30,
    )
    clf.fit(X, y)

    pred = clf.predict(X[:10])
    proba = clf.predict_proba(X[:10])

    assert pred.shape == (10,)
    assert proba.shape == (10, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_logicgp_rlcw_micro_fit_predict():
    from scoredrulesets import LogicGPClassifier

    X, y = _make_simple_data()
    clf = LogicGPClassifier(
        trainer="rlcw",
        f1_averaging="micro",
        max_generations=25,
        stagnation_generations=8,
        n_bins=3,
        population_size=15,
        n_adaptations_per_gen=12,
        min_max_weight=0.15,
        random_state=3,
    )
    clf.fit(X, y)

    meta = clf.to_ruleset().metadata
    assert meta["trainer"] == "rlcw"
    assert meta["f1_averaging"] == "micro"
    assert meta["population_size"] == 15


def test_logicgp_rlcw_trainer_in_metadata():
    from scoredrulesets import LogicGPClassifier

    X, y = _make_simple_data()
    clf = LogicGPClassifier(
        trainer="rlcw",
        f1_averaging="macro",
        max_generations=20,
        stagnation_generations=5,
        n_bins=3,
        random_state=4,
    )
    clf.fit(X, y)

    meta = clf.to_ruleset().metadata
    assert meta["trainer"] == "rlcw"
    assert meta["f1_averaging"] == "macro"
    assert meta["source"] == "logicgp"


def test_logicgp_rlcw_max_model_size_constraint():
    from scoredrulesets import LogicGPClassifier

    X, y = _make_simple_data()
    clf = LogicGPClassifier(
        trainer="rlcw",
        f1_averaging="macro",
        max_generations=30,
        stagnation_generations=8,
        n_bins=3,
        max_model_size=2,
        random_state=5,
    )
    clf.fit(X, y)

    ruleset = clf.to_ruleset()
    meta = ruleset.metadata
    assert meta["max_model_size"] == 2
    # Model size must be <= max_model_size
    assert meta["model_size"] <= 2


def test_logicgp_rlcw_population_size_limits_growth():
    from scoredrulesets import LogicGPClassifier

    X, y = _make_simple_data()
    clf = LogicGPClassifier(
        trainer="rlcw",
        f1_averaging="macro",
        max_generations=20,
        stagnation_generations=5,
        n_bins=3,
        population_size=10,
        tournament_size=3,
        random_state=6,
    )
    clf.fit(X, y)
    # Should not raise an exception
    assert clf.to_ruleset() is not None


# ---------------------------------------------------------------------------
# n_adaptations_per_gen Tests
# ---------------------------------------------------------------------------

def test_logicgp_more_adaptations_per_gen():
    from scoredrulesets import LogicGPClassifier

    X, y = _make_simple_data()
    clf = LogicGPClassifier(
        trainer="rlcw",
        f1_averaging="macro",
        max_generations=15,
        stagnation_generations=5,
        n_bins=3,
        n_adaptations_per_gen=18,
        population_size=25,
        random_state=7,
    )
    clf.fit(X, y)
    meta = clf.to_ruleset().metadata
    assert meta["n_adaptations_per_gen"] == 18


# ---------------------------------------------------------------------------
# ScoredRuleSetClassifier wrapper (backend="logicgp")
# ---------------------------------------------------------------------------

def test_sklearn_wrapper_logicgp_backend_basic():
    from scoredrulesets import ScoredRuleSetClassifier

    X, y = load_iris(return_X_y=True)
    clf = ScoredRuleSetClassifier(
        backend="logicgp",
        backend_params={
            "trainer": "flcw",
            "f1_averaging": "macro",
            "max_generations": 20,
            "stagnation_generations": 5,
            "population_size": 20,
            "n_bins": 5,
            "random_state": 0,
        },
        random_state=0,
    )
    clf.fit(X, y)

    pred = clf.predict(X[:5])
    proba = clf.predict_proba(X[:5])

    assert pred.shape == (5,)
    assert proba.shape == (5, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_sklearn_wrapper_logicgp_rlcw_backend():
    from scoredrulesets import ScoredRuleSetClassifier

    X, y = _make_simple_data()
    clf = ScoredRuleSetClassifier(
        backend="logicgp",
        backend_params={
            "trainer": "rlcw",
            "f1_averaging": "macro",
            "max_generations": 25,
            "stagnation_generations": 8,
            "population_size": 20,
            "n_adaptations_per_gen": 12,
            "min_max_weight": 0.1,
            "n_bins": 3,
            "random_state": 1,
        },
        random_state=0,
    )
    clf.fit(X, y)

    pred = clf.predict(X[:10])
    assert pred.shape == (10,)
    assert clf.to_ruleset().metadata["trainer"] == "rlcw"
    assert clf.to_ruleset().metadata["f1_averaging"] == "macro"


# ---------------------------------------------------------------------------
# Benchmark estimators for logicGP
# ---------------------------------------------------------------------------

def test_benchmarking_estimator_specs_include_logicgp():
    from scoredrulesets.benchmarking.estimators import default_estimator_specs

    specs = default_estimator_specs()
    assert "wrapper_logicgp" in specs
    assert "wrapper_logicgp_strong" in specs
    assert "wrapper_logicgp_mux" in specs
    assert "wrapper_logicgp_mux_rlcw_macro" in specs


# ---------------------------------------------------------------------------
# Time budget (max_fit_seconds)
# ---------------------------------------------------------------------------

def test_logicgp_max_fit_seconds_raises_on_setup_timeout(monkeypatch):
    """A budget too small for even one generation must raise, not return a model."""
    import scoredrulesets.estimators._time_budget as time_budget_module
    from scoredrulesets import FitBudgetExceededError, LogicGPClassifier

    X, y = _make_simple_data()

    ticks = iter(float(i) for i in range(10_000))
    monkeypatch.setattr(time_budget_module.time, "monotonic", lambda: next(ticks))

    clf = LogicGPClassifier(
        trainer="rlcw",
        f1_averaging="macro",
        max_generations=1000,
        stagnation_generations=1000,
        n_bins=3,
        population_size=25,
        max_fit_seconds=0.5,
        random_state=0,
    )
    with pytest.raises(FitBudgetExceededError):
        clf.fit(X, y)



