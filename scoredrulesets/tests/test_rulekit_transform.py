import numpy as np

from scoredrulesets.estimators.ruleset_transform import _condition_to_atoms, rulekit_to_scored_ruleset


class _FakeValueSet:
    def __init__(self, left, right, left_sign="(", right_sign="]"):
        self._left = left
        self._right = right
        self._left_sign = left_sign
        self._right_sign = right_sign

    def getLeft(self):
        return self._left

    def getRight(self):
        return self._right

    def getLeftSign(self):
        return self._left_sign

    def getRightSign(self):
        return self._right_sign


class _FakeCondition:
    def __init__(self, attr, value_set):
        self._attr = attr
        self._value_set = value_set

    def getAttribute(self):
        return self._attr

    def getValueSet(self):
        return self._value_set


class _FakeRule:
    def __init__(self, decision_class, conditions, confidence=None, support=None, weight=None):
        self.decision_class = decision_class
        self.conditions = conditions
        self.confidence = confidence
        self.support = support
        self.weight = weight


class _FakeEstimator:
    def __init__(self, rules):
        self.rules = rules


def test_rulekit_condition_interval_maps_to_two_boundary_atoms():
    condition = _FakeCondition("att1", _FakeValueSet(1.0, 3.0, left_sign="(", right_sign="]"))

    atoms = _condition_to_atoms(condition, ["f0", "f1"])

    assert len(atoms) == 2
    assert atoms[0].feature == "f0"
    assert atoms[0].op == ">"
    assert atoms[0].value == 1.0
    assert atoms[1].feature == "f0"
    assert atoms[1].op == "<="
    assert atoms[1].value == 3.0


def test_rulekit_transform_uses_rule_strength_and_prior_default():
    rules = [
        _FakeRule(
            decision_class=1,
            conditions=[_FakeCondition("att1", _FakeValueSet(0.0, 10.0))],
            confidence=0.8,
            support=0.5,
        )
    ]
    estimator = _FakeEstimator(rules)
    y_train = np.array([0, 0, 0, 1])

    ruleset = rulekit_to_scored_ruleset(
        estimator=estimator,
        class_labels=[0, 1],
        feature_names=["f0"],
        y_train=y_train,
    )

    assert len(ruleset.rules) == 2
    default_rule = ruleset.rules[0]
    transformed_rule = ruleset.rules[1]

    assert default_rule.atoms == []
    assert default_rule.scores[0] == 0.75
    assert default_rule.scores[1] == 0.25

    assert transformed_rule.scores[0] == 0.0
    assert transformed_rule.scores[1] == 0.4

