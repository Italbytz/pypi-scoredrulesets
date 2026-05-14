import numpy as np
import pytest

from scoredrulesets.runtime import predict, predict_regression
from scoredrulesets.schema import AggregationSpec, Atom, Rule, ScoredRuleSet


def _make_regression_ruleset(aggregation: str) -> ScoredRuleSet:
    return ScoredRuleSet(
        class_labels=[],
        task_type="regression",
        feature_names=["x"],
        aggregation=AggregationSpec(type=aggregation),
        rules=[
            Rule(atoms=[Atom(feature="x", op=">", value=0.5)], scores=[2.0]),
            Rule(atoms=[Atom(feature="x", op=">", value=1.0)], scores=[3.0]),
            Rule(atoms=[], scores=[10.0]),
        ],
    )


def test_predict_regression_default_plus_sum():
    ruleset = _make_regression_ruleset("default_plus_sum")
    X = np.asarray([[0.2], [0.8], [1.2]], dtype=float)

    y = predict_regression(ruleset, X)
    assert y.tolist() == [10.0, 12.0, 15.0]


def test_predict_regression_mean_active_with_default_fallback():
    ruleset = _make_regression_ruleset("mean_active")
    X = np.asarray([[0.2], [0.8], [1.2]], dtype=float)

    y = predict_regression(ruleset, X)
    assert y.tolist() == [10.0, 2.0, 2.5]


def test_predict_rejects_regression_ruleset():
    ruleset = _make_regression_ruleset("weighted_sum")
    X = np.asarray([[0.2], [0.8]], dtype=float)

    with pytest.raises(ValueError, match="task_type='classification'"):
        predict(ruleset, X)
