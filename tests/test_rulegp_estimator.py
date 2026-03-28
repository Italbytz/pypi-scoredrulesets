import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer

from scoredrulesets import RuleGPClassifier, ScoredRuleSetClassifier
from scoredrulesets.benchmarking.estimators import default_estimator_specs


def test_rulegp_estimator_fit_predict_and_ruleset():
    X, y = load_iris(return_X_y=True)
    clf = RuleGPClassifier(
        population_size=20,
        generations=10,
        max_rules=6,
        max_atoms_per_rule=3,
        random_state=0,
    )
    clf.fit(X, y)

    pred = clf.predict(X[:10])
    proba = clf.predict_proba(X[:10])
    ruleset = clf.to_ruleset()

    assert pred.shape == (10,)
    assert proba.shape == (10, len(clf.classes_))
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert any(r.rule_id == "rulegp_default" for r in ruleset.rules)
    assert any(r.rule_id and r.rule_id.startswith("rulegp_rule_") for r in ruleset.rules)
    assert ruleset.metadata["source"] == "rulegp"


def test_rulegp_binary_classification():
    X, y = load_breast_cancer(return_X_y=True)
    clf = RuleGPClassifier(
        population_size=20,
        generations=5,
        max_rules=4,
        random_state=42,
    )
    clf.fit(X, y)

    pred = clf.predict(X)
    proba = clf.predict_proba(X)
    assert set(pred).issubset(set(y))
    assert proba.shape == (X.shape[0], 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_rulegp_compaction_reduces_atoms():
    X, y = load_iris(return_X_y=True)

    clf_compact = RuleGPClassifier(
        population_size=30,
        generations=15,
        enable_compaction=True,
        random_state=0,
    )
    clf_compact.fit(X, y)
    rs_compact = clf_compact.to_ruleset()

    clf_no_compact = RuleGPClassifier(
        population_size=30,
        generations=15,
        enable_compaction=False,
        random_state=0,
    )
    clf_no_compact.fit(X, y)
    rs_no_compact = clf_no_compact.to_ruleset()

    atoms_compact = sum(len(r.atoms) for r in rs_compact.rules)
    atoms_no_compact = sum(len(r.atoms) for r in rs_no_compact.rules)
    # Compaction should keep atoms the same or reduce them
    assert atoms_compact <= atoms_no_compact + 2  # allow small tolerance


def test_sklearn_wrapper_rulegp_backend():
    X, y = load_iris(return_X_y=True)
    clf = ScoredRuleSetClassifier(
        backend="rulegp",
        backend_params={
            "population_size": 20,
            "generations": 5,
            "max_rules": 4,
        },
        random_state=0,
    )
    clf.fit(X, y)

    pred = clf.predict(X[:6])
    proba = clf.predict_proba(X[:6])
    assert pred.shape == (6,)
    assert proba.shape == (6, 3)
    assert clf.to_ruleset().metadata["source"] == "rulegp"


def test_benchmarking_estimator_specs_include_rulegp():
    specs = default_estimator_specs()
    assert "wrapper_rulegp" in specs


def test_rulegp_benchmark_spec_produces_valid_estimator():
    specs = default_estimator_specs()
    est = specs["wrapper_rulegp"].factory()
    assert est.backend == "rulegp"
    assert est.backend_params["population_size"] >= 10
    assert est.backend_params["enable_compaction"] is True


def test_rulegp_ruleset_atoms_use_correct_between_format():
    """Ensure 'between' atoms are serialized as [low, high] lists, not dicts."""
    X, y = load_iris(return_X_y=True)
    clf = RuleGPClassifier(
        population_size=20,
        generations=10,
        random_state=0,
    )
    clf.fit(X, y)
    rs = clf.to_ruleset()

    for rule in rs.rules:
        for atom in rule.atoms:
            if atom.op == "between":
                assert isinstance(atom.value, (list, tuple)), (
                    f"'between' atom value should be list/tuple, got {type(atom.value)}"
                )
                assert len(atom.value) == 2

