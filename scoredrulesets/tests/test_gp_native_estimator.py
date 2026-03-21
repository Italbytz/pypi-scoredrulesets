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


def test_gp_native_estimator_reports_validation_and_generations_ran():
    X, y = load_iris(return_X_y=True)
    clf = GeneticScoredRuleSetClassifier(
        population_size=18,
        generations=25,
        early_stopping_rounds=2,
        validation_fraction=0.25,
        random_state=4,
    )
    clf.fit(X, y)

    meta = clf.to_ruleset().metadata
    assert meta["used_validation"] is True
    assert 1 <= meta["generations_ran"] <= 25
    assert isinstance(meta["early_stopped"], bool)


def test_gp_native_estimator_without_validation_fraction_uses_train_only():
    X, y = load_iris(return_X_y=True)
    clf = GeneticScoredRuleSetClassifier(
        generations=5,
        validation_fraction=0.0,
        random_state=5,
    )
    clf.fit(X, y)
    assert clf.to_ruleset().metadata["used_validation"] is False


def test_gp_native_estimator_selection_mode_pareto_metadata():
    X, y = load_iris(return_X_y=True)
    clf = GeneticScoredRuleSetClassifier(
        selection_mode="pareto",
        generations=6,
        random_state=9,
    )
    clf.fit(X, y)
    assert clf.to_ruleset().metadata["selection_mode"] == "pareto"


def test_gp_native_estimator_final_rule_selection_diverse_metadata():
    X, y = load_iris(return_X_y=True)
    clf = GeneticScoredRuleSetClassifier(
        final_rule_selection="diverse",
        generations=6,
        random_state=12,
    )
    clf.fit(X, y)
    assert clf.to_ruleset().metadata["final_rule_selection"] == "diverse"


def test_gp_native_estimator_final_rule_selection_contribution_metadata():
    X, y = load_iris(return_X_y=True)
    clf = GeneticScoredRuleSetClassifier(
        final_rule_selection="contribution",
        generations=6,
        random_state=13,
    )
    clf.fit(X, y)
    assert clf.to_ruleset().metadata["final_rule_selection"] == "contribution"


def test_gp_native_estimator_evolution_fitness_mode_residual_metadata():
    X, y = load_iris(return_X_y=True)
    clf = GeneticScoredRuleSetClassifier(
        evolution_fitness_mode="residual_covering",
        generations=6,
        random_state=14,
    )
    clf.fit(X, y)
    meta = clf.to_ruleset().metadata
    assert meta["evolution_fitness_mode"] == "residual_covering"
    assert meta["evolution_context_size"] >= 0


def test_gp_native_estimator_invalid_selection_mode_raises():
    X, y = load_iris(return_X_y=True)
    clf = GeneticScoredRuleSetClassifier(selection_mode="invalid", random_state=0)
    with pytest.raises(ValueError, match="Invalid selection_mode"):
        clf.fit(X, y)


def test_gp_native_estimator_invalid_final_rule_selection_raises():
    X, y = load_iris(return_X_y=True)
    clf = GeneticScoredRuleSetClassifier(final_rule_selection="invalid", random_state=0)
    with pytest.raises(ValueError, match="Invalid final_rule_selection"):
        clf.fit(X, y)


def test_gp_native_estimator_invalid_evolution_fitness_mode_raises():
    X, y = load_iris(return_X_y=True)
    clf = GeneticScoredRuleSetClassifier(evolution_fitness_mode="invalid", random_state=0)
    with pytest.raises(ValueError, match="Invalid evolution_fitness_mode"):
        clf.fit(X, y)


def test_gp_native_estimator_mask_jaccard_helper():
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    assert np.isclose(GeneticScoredRuleSetClassifier._mask_jaccard(a, b), 1 / 3)


def test_pareto_front_ranks_helper():
    objectives = [
        (0.90, 3),
        (0.85, 1),
        (0.80, 4),
        (0.88, 3),
    ]
    ranks = GeneticScoredRuleSetClassifier._pareto_front_ranks(objectives)
    assert ranks[0] == 0
    assert ranks[1] == 0
    assert ranks[3] == 1
    assert ranks[2] >= 1


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



