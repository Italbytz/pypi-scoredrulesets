import numpy as np
import pytest
from sklearn.datasets import load_iris

import scoredrulesets.estimators._time_budget as time_budget_module
from scoredrulesets import RulePLCSClassifier, ScoredRuleSetClassifier, register_atom_selection_strategy
from scoredrulesets.benchmarking.estimators import default_estimator_specs


def test_ruleplcs_estimator_fit_predict_and_ruleset():
    X, y = load_iris(return_X_y=True)
    clf = RulePLCSClassifier(
        population_size=50,
        n_iterations=10,
        n_repetitions=1,
        max_rules=5,
        random_state=0,
    )
    clf.fit(X, y)

    pred = clf.predict(X[:10])
    proba = clf.predict_proba(X[:10])
    ruleset = clf.to_ruleset()

    assert pred.shape == (10,)
    assert proba.shape == (10, len(clf.classes_))
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert ruleset.metadata["estimator"] == "RulePLCSClassifier"
    assert ruleset.metadata["algorithm"] == "BioHEL-inspired IRL"
    assert any(rule.rule_id == "default" for rule in ruleset.rules)
    assert len(ruleset.rules) >= 2  # at least one learned rule + default


def test_sklearn_wrapper_ruleplcs_backend():
    X, y = load_iris(return_X_y=True)
    clf = ScoredRuleSetClassifier(
        backend="ruleplcs",
        backend_params={
            "population_size": 50,
            "n_iterations": 10,
            "n_repetitions": 1,
            "max_rules": 5,
        },
        random_state=0,
    )
    clf.fit(X, y)

    pred = clf.predict(X[:6])
    proba = clf.predict_proba(X[:6])
    assert pred.shape == (6,)
    assert proba.shape == (6, 3)
    assert clf.to_ruleset().metadata["estimator"] == "RulePLCSClassifier"


def test_benchmarking_estimator_specs_include_ruleplcs():
    specs = default_estimator_specs()
    assert "wrapper_ruleplcs" in specs
    assert "wrapper_ruleplcs_strong" in specs
    assert "wrapper_ruleplcs_topc2" not in specs


def test_ruleplcs_benchmark_profiles_use_expected_backend_and_budget():
    specs = default_estimator_specs()

    base = specs["wrapper_ruleplcs"].factory()
    strong = specs["wrapper_ruleplcs_strong"].factory()
    assert base.backend == "ruleplcs"
    assert base.backend_params["max_rules"] >= 10
    assert base.max_fit_seconds == 240

    assert strong.backend == "ruleplcs"
    assert strong.backend_params["population_size"] >= 200
    assert strong.backend_params["max_rules"] >= 15
    assert strong.max_fit_seconds == 240


def test_ruleplcs_registered_strategy_is_optional_and_recorded_in_metadata():
    X, y = load_iris(return_X_y=True)

    def _first_k_strategy(candidates, _y_idx, _n_classes, min_samples_leaf, top_k):
        selected = set()
        for signature, mask in candidates:
            if int(np.sum(mask)) < int(min_samples_leaf):
                continue
            selected.add(signature)
            if len(selected) >= int(top_k):
                break
        return selected

    register_atom_selection_strategy("top_c2_private_ut", _first_k_strategy, overwrite=True)

    clf = RulePLCSClassifier(
        population_size=40,
        n_iterations=8,
        n_repetitions=1,
        max_rules=4,
        atom_preselection_strategy="top_c2_private_ut",
        atom_preselection_top_k=32,
        random_state=0,
    )
    clf.fit(X, y)
    ruleset = clf.to_ruleset()

    assert ruleset.metadata["atom_preselection_strategy"] == "top_c2_private_ut"
    assert ruleset.metadata["atom_preselection_top_k"] == 32


def test_ruleplcs_registered_strategy_requires_positive_top_k():
    register_atom_selection_strategy(
        "top_c2_private_ut",
        lambda candidates, y_idx, n_classes, min_samples_leaf, top_k: set(),
        overwrite=True,
    )
    with pytest.raises(ValueError, match="atom_preselection_top_k"):
        RulePLCSClassifier(atom_preselection_strategy="top_c2_private_ut")


def test_ruleplcs_max_fit_seconds_raises_on_setup_timeout(monkeypatch):
    """A budget too small for even one rule search must raise, not return a model.

    When ``max_fit_seconds`` is exhausted during setup (before the rule-learning
    GA runs even once) no meaningful rule exists, so the estimator surfaces a
    clear timeout instead of a degenerate default-class-only model.
    """
    from scoredrulesets import FitBudgetExceededError

    X, y = load_iris(return_X_y=True)

    ticks = iter(float(i) for i in range(1000))
    monkeypatch.setattr(time_budget_module.time, "monotonic", lambda: next(ticks))

    clf = RulePLCSClassifier(
        population_size=30,
        n_iterations=20,
        n_repetitions=2,
        max_rules=5,
        max_fit_seconds=0.5,
        random_state=0,
    )
    with pytest.raises(FitBudgetExceededError):
        clf.fit(X, y)


def test_ruleplcs_run_ga_stops_during_population_setup():
    """A deadline crossed inside a GA run must stop cleanly, not overrun.

    ``_smart_init`` scans the whole feature space per individual, so on wide
    inputs the population setup can dominate the fit time.  When the deadline is
    already in the past, ``_run_ga`` must honour it during setup and return a
    valid individual instead of building the full population.
    """
    from scoredrulesets.estimators.ruleplcs import _run_ga

    X, y = load_iris(return_X_y=True)
    X = X.astype(float)
    y = y.astype(int)
    classes = np.unique(y)
    feature_info = RulePLCSClassifier()._build_feature_info(X)

    rng = np.random.default_rng(0)
    past_deadline = time_budget_module.time.monotonic() - 1.0

    rule = _run_ga(
        rng, X, y, feature_info, classes, None,
        population_size=50,
        n_iterations=20,
        tournament_size=3,
        crossover_prob=0.6,
        mutation_prob=0.3,
        prob_generalize=0.5,
        prob_specialize=0.5,
        prob_include=0.5,
        prob_one=0.5,
        coverage_break=0.1,
        coverage_ratio=0.9,
        mdl_initial_ratio=0.25,
        mdl_activate_iter=5,
        mdl_relax_factor=0.9,
        n_strata=2,
        allowed_feature_indices=None,
        fit_deadline=past_deadline,
    )

    assert rule is not None
    assert int(rule.class_value) in set(int(c) for c in classes)

