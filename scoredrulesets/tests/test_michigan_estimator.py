import numpy as np
import runpy
from pathlib import Path

from sklearn.datasets import load_iris

from scoredrulesets import MichiganRuleSetClassifier, ScoredRuleSetClassifier
from scoredrulesets.benchmarking.estimators import default_estimator_specs


def test_michigan_estimator_fit_predict_and_ruleset():
    X, y = load_iris(return_X_y=True)
    clf = MichiganRuleSetClassifier(
        population_size=48,
        epochs=6,
        max_atoms_per_rule=2,
        random_state=0,
    )
    clf.fit(X, y)

    pred = clf.predict(X[:12])
    proba = clf.predict_proba(X[:12])
    ruleset = clf.to_ruleset()

    assert pred.shape == (12,)
    assert proba.shape == (12, len(clf.classes_))
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert any(rule.rule_id == "michigan_default_prior" for rule in ruleset.rules)
    assert any(rule.rule_id and rule.rule_id.startswith("michigan_rule_") for rule in ruleset.rules)
    assert ruleset.metadata["source"] == "michigan_lcs"



def test_michigan_estimator_handles_categorical_data():
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

    clf = MichiganRuleSetClassifier(
        population_size=30,
        epochs=5,
        max_atoms_per_rule=2,
        min_samples_leaf=1,
        random_state=3,
    )
    clf.fit(X, y)

    ruleset = clf.to_ruleset()
    assert any(atom.op == "==" for rule in ruleset.rules for atom in rule.atoms)



def test_sklearn_wrapper_michigan_backend():
    X, y = load_iris(return_X_y=True)
    clf = ScoredRuleSetClassifier(
        backend="michigan",
        backend_params={
            "population_size": 40,
            "epochs": 6,
            "max_atoms_per_rule": 2,
        },
        random_state=0,
    )
    clf.fit(X, y)

    pred = clf.predict(X[:6])
    proba = clf.predict_proba(X[:6])
    assert pred.shape == (6,)
    assert proba.shape == (6, 3)
    assert clf.to_ruleset().metadata["source"] == "michigan_lcs"



def test_benchmarking_estimator_specs_include_michigan_profiles():
    specs = default_estimator_specs()
    assert "wrapper_michigan" in specs
    assert "wrapper_michigan_fast" in specs
    assert "wrapper_michigan_strong" in specs
    assert "wrapper_michigan_compact" in specs


def test_michigan_estimator_compact_max_final_rules():
    X, y = load_iris(return_X_y=True)
    clf = MichiganRuleSetClassifier(
        population_size=100,
        epochs=12,
        max_atoms_per_rule=3,
        max_final_rules=12,
        random_state=0,
    )
    clf.fit(X, y)

    ruleset = clf.to_ruleset()
    non_default = [r for r in ruleset.rules if r.rule_id != "michigan_default_prior"]
    assert len(non_default) <= 12
    assert ruleset.metadata["max_final_rules"] == 12


def test_michigan_wrapper_example_run_demo_smoke():
    example_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "example_michigan_wrapper.py"
    )
    module_globals = runpy.run_path(str(example_path))
    result = module_globals["run_demo"](random_state=0)

    assert result["dataset"] == "sklearn_iris"
    assert result["profile"] == "default"
    assert result["wrapper_michigan"]["metadata"]["source"] == "michigan_lcs"
    assert result["wrapper_michigan"]["n_rules"] > 0
    assert {row["name"] for row in result["comparison"]} == {"wrapper_cart_d2", "wrapper_cart_d4"}


def test_michigan_wrapper_example_profile_smoke():
    example_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "example_michigan_wrapper.py"
    )
    module_globals = runpy.run_path(str(example_path))
    result = module_globals["run_demo"](random_state=0, profile="compact")

    assert result["profile"] == "compact"
    meta = result["wrapper_michigan"]["metadata"]
    assert meta["source"] == "michigan_lcs"
    assert meta["epochs"] >= 8
    assert meta["max_final_rules"] == 20


def test_michigan_example_run_demo_smoke():
    example_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "example_michigan_backend.py"
    )
    module_globals = runpy.run_path(str(example_path))
    result = module_globals["run_demo"](random_state=0)

    assert result["dataset"] == "sklearn_iris"
    assert result["profile"] == "default"
    assert result["michigan"]["metadata"]["source"] == "michigan_lcs"
    assert result["michigan"]["n_rules"] > 0
    assert {row["name"] for row in result["comparison"]} == {"native", "gp"}


def test_michigan_example_profile_smoke():
    example_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "example_michigan_backend.py"
    )
    module_globals = runpy.run_path(str(example_path))
    result = module_globals["run_demo"](random_state=0, profile="compact")

    assert result["profile"] == "compact"
    meta = result["michigan"]["metadata"]
    assert meta["source"] == "michigan_lcs"
    assert meta["epochs"] >= 8
    assert meta["max_final_rules"] == 20


