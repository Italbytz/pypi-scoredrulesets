import numpy as np
import runpy
from pathlib import Path

from sklearn.datasets import load_iris

from scoredrulesets import PittsburghRuleSetClassifier, ScoredRuleSetClassifier
from scoredrulesets.benchmarking.estimators import default_estimator_specs


def test_pittsburgh_estimator_fit_predict_and_ruleset():
    X, y = load_iris(return_X_y=True)
    clf = PittsburghRuleSetClassifier(
        max_rules=4,
        candidate_pool_size=16,
        beam_width=5,
        max_iterations=8,
        random_state=0,
    )
    clf.fit(X, y)

    pred = clf.predict(X[:10])
    proba = clf.predict_proba(X[:10])
    ruleset = clf.to_ruleset()

    assert pred.shape == (10,)
    assert proba.shape == (10, len(clf.classes_))
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert any(rule.rule_id == "pittsburgh_default_prior" for rule in ruleset.rules)
    assert any(rule.rule_id and rule.rule_id.startswith("pittsburgh_rule_") for rule in ruleset.rules)
    assert ruleset.metadata["source"] == "pittsburgh"
    assert ruleset.metadata["selected_rule_count"] <= 4


def test_pittsburgh_estimator_handles_categorical_rules():
    X = np.array(
        [
            ["red", "small"],
            ["red", "large"],
            ["blue", "small"],
            ["blue", "large"],
            ["green", "small"],
            ["green", "large"],
            ["red", "small"],
            ["blue", "large"],
        ],
        dtype=object,
    )
    y = np.array([1, 1, 0, 0, 0, 0, 1, 0])

    clf = PittsburghRuleSetClassifier(
        max_rules=3,
        min_samples_leaf=1,
        candidate_pool_size=12,
        beam_width=4,
        max_iterations=6,
        random_state=3,
    )
    clf.fit(X, y)

    ruleset = clf.to_ruleset()
    assert any(atom.op in {"==", "in"} for rule in ruleset.rules for atom in rule.atoms)



def test_sklearn_wrapper_pittsburgh_backend():
    X, y = load_iris(return_X_y=True)
    clf = ScoredRuleSetClassifier(
        backend="pittsburgh",
        backend_params={
            "max_rules": 4,
            "candidate_pool_size": 16,
            "beam_width": 5,
            "max_iterations": 8,
        },
        random_state=0,
    )
    clf.fit(X, y)

    pred = clf.predict(X[:6])
    proba = clf.predict_proba(X[:6])
    assert pred.shape == (6,)
    assert proba.shape == (6, 3)
    assert clf.to_ruleset().metadata["source"] == "pittsburgh"



def test_benchmarking_estimator_specs_include_pittsburgh():
    specs = default_estimator_specs()
    assert "wrapper_pittsburgh" in specs
    assert "wrapper_pittsburgh_fast" in specs
    assert "wrapper_pittsburgh_strong" in specs
    assert "wrapper_pittsburgh_diverse" in specs


def test_pittsburgh_benchmark_profiles_use_expected_backend_and_budget():
    specs = default_estimator_specs()

    strong = specs["wrapper_pittsburgh_strong"].factory()
    diverse = specs["wrapper_pittsburgh_diverse"].factory()

    assert strong.backend == "pittsburgh"
    assert strong.backend_params["beam_width"] >= 10
    assert strong.backend_params["candidate_pool_size"] >= 32

    assert diverse.backend == "pittsburgh"
    assert diverse.backend_params["max_rules"] >= 7
    assert diverse.backend_params["candidate_pool_size"] >= 36


def test_pittsburgh_example_run_demo_smoke():
    example_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "example_pittsburgh_backend.py"
    )
    module_globals = runpy.run_path(str(example_path))
    result = module_globals["run_demo"](random_state=0)

    assert result["dataset"] == "sklearn_iris"
    assert result["pittsburgh"]["metadata"]["source"] == "pittsburgh"
    assert result["pittsburgh"]["n_rules"] > 0
    assert len(result["comparison"]) == 2
    assert {row["name"] for row in result["comparison"]} == {"native", "gp"}


def test_pittsburgh_wrapper_example_run_demo_smoke():
    example_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "example_pittsburgh_wrapper.py"
    )
    module_globals = runpy.run_path(str(example_path))
    result = module_globals["run_demo"](random_state=0)

    assert result["dataset"] == "sklearn_iris"
    assert result["wrapper_pittsburgh"]["metadata"]["source"] == "pittsburgh"
    assert result["wrapper_pittsburgh"]["n_rules"] > 0
    assert len(result["comparison"]) == 2
    assert {row["name"] for row in result["comparison"]} == {"wrapper_cart_d2", "wrapper_cart_d4"}


