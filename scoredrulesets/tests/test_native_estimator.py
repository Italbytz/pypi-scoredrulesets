import numpy as np
from sklearn.datasets import load_iris

from scoredrulesets import NativeScoredRuleSetClassifier


def test_native_estimator_fit_predict_and_ruleset():
    X, y = load_iris(return_X_y=True)
    clf = NativeScoredRuleSetClassifier(
        aggregation="softmax_sum",
        max_rules=4,
        min_samples_leaf=5,
    )
    clf.fit(X, y)

    pred = clf.predict(X[:7])
    proba = clf.predict_proba(X[:7])
    ruleset = clf.to_ruleset()

    assert pred.shape == (7,)
    assert proba.shape == (7, len(clf.classes_))
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert len(ruleset.rules) >= 2
    assert any(r.rule_id == "native_default_prior" for r in ruleset.rules)
    assert any(r.rule_id and r.rule_id.startswith("native_f") for r in ruleset.rules)


def test_native_estimator_learns_categorical_rules():
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

    clf = NativeScoredRuleSetClassifier(
        aggregation="softmax_sum",
        max_rules=6,
        min_samples_leaf=1,
    )
    clf.fit(X, y)

    ruleset = clf.to_ruleset()
    pred = clf.predict(X)
    assert pred.shape == (8,)
    assert any(atom.op == "==" for rule in ruleset.rules for atom in rule.atoms)


def test_native_estimator_can_disable_categorical_rules():
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

    clf = NativeScoredRuleSetClassifier(
        enable_categorical_rules=False,
        max_rules=6,
        min_samples_leaf=1,
    )
    clf.fit(X, y)

    ruleset = clf.to_ruleset()
    assert not any(atom.op == "==" for rule in ruleset.rules for atom in rule.atoms)


