import numpy as np
from sklearn.datasets import load_iris

from scoredrulesets import RulePLCSClassifier, ScoredRuleSetClassifier
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


def test_ruleplcs_benchmark_profiles_use_expected_backend_and_budget():
    specs = default_estimator_specs()

    base = specs["wrapper_ruleplcs"].factory()
    strong = specs["wrapper_ruleplcs_strong"].factory()

    assert base.backend == "ruleplcs"
    assert base.backend_params["max_rules"] >= 10

    assert strong.backend == "ruleplcs"
    assert strong.backend_params["population_size"] >= 200
    assert strong.backend_params["max_rules"] >= 15

