import numpy as np
import pytest
from sklearn.datasets import load_iris

from scoredrulesets import GeneticScoredRuleSetClassifier


def test_gp_native_estimator_fit_predict_and_ruleset():
    X, y = load_iris(return_X_y=True)
    clf = GeneticScoredRuleSetClassifier(
        population_size=20,
        generations=8,
        max_rules=4,
        random_state=0,
    )
    clf.fit(X, y)

    pred = clf.predict(X[:10])
    proba = clf.predict_proba(X[:10])
    ruleset = clf.to_ruleset()

    assert pred.shape == (10,)
    assert proba.shape == (10, len(clf.classes_))
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert any(rule.rule_id == "gp_default_prior" for rule in ruleset.rules)
    assert any(rule.rule_id and rule.rule_id.startswith("gp_rule_") for rule in ruleset.rules)


def test_gp_native_estimator_handles_categorical_data():
    X = np.array(
        [
            ["red", 1.0],
            ["red", 1.2],
            ["blue", 3.1],
            ["blue", 2.9],
            ["green", 2.7],
            ["green", 2.8],
        ],
        dtype=object,
    )
    y = np.array([1, 1, 0, 0, 0, 0])

    clf = GeneticScoredRuleSetClassifier(
        population_size=16,
        generations=6,
        max_rules=3,
        min_samples_leaf=1,
        random_state=3,
    )
    clf.fit(X, y)

    ruleset = clf.to_ruleset()
    assert any(atom.op == "==" for rule in ruleset.rules for atom in rule.atoms)


def test_gp_native_estimator_score_mode_proba():
    X, y = load_iris(return_X_y=True)
    clf = GeneticScoredRuleSetClassifier(
        population_size=16,
        generations=6,
        max_rules=3,
        score_mode="proba",
        random_state=7,
    )
    clf.fit(X, y)

    first_rule = next(rule for rule in clf.to_ruleset().rules if rule.rule_id != "gp_default_prior")
    assert all(0.0 <= s <= 1.0 for s in first_rule.scores)
    assert np.isclose(sum(first_rule.scores), 1.0)


def test_gp_native_estimator_score_mode_log_proba():
    X, y = load_iris(return_X_y=True)
    clf = GeneticScoredRuleSetClassifier(
        population_size=16,
        generations=6,
        max_rules=3,
        score_mode="log_proba",
        random_state=11,
    )
    clf.fit(X, y)

    first_rule = next(rule for rule in clf.to_ruleset().rules if rule.rule_id != "gp_default_prior")
    assert all(s <= 0.0 for s in first_rule.scores)


def test_gp_native_estimator_invalid_score_mode_raises():
    X, y = load_iris(return_X_y=True)
    clf = GeneticScoredRuleSetClassifier(score_mode="invalid", random_state=0)
    with pytest.raises(ValueError, match="Invalid score_mode"):
        clf.fit(X, y)


def test_gp_native_estimator_score_mode_auto_metadata():
    X, y = load_iris(return_X_y=True)
    clf = GeneticScoredRuleSetClassifier(score_mode="auto", aggregation="softmax_sum", random_state=2)
    clf.fit(X, y)
    assert clf.to_ruleset().metadata["score_mode"] == "log_proba"


def test_gp_native_estimator_can_sample_between_atoms():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    clf = GeneticScoredRuleSetClassifier(random_state=0)
    clf._rng_ = np.random.default_rng(0)
    specs = clf._build_feature_specs(X)

    ops = [clf._random_atom(specs).op for _ in range(120)]
    assert "between" in ops


def test_gp_native_estimator_can_sample_in_atoms():
    X = np.array([["red"], ["blue"], ["green"], ["yellow"]], dtype=object)
    clf = GeneticScoredRuleSetClassifier(random_state=0)
    clf._rng_ = np.random.default_rng(1)
    specs = clf._build_feature_specs(X)

    ops = [clf._random_atom(specs).op for _ in range(120)]
    assert "in" in ops



