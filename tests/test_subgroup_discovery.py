import numpy as np
import pytest
from sklearn.datasets import make_regression

from scoredrulesets.estimators.evosubgroup import RuleEvoSubgroup
from scoredrulesets.estimators.rulegp_subgroup import RuleGPSubgroup


@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=100, n_features=4, random_state=42, noise=0.1)
    return X, y


def test_ruleevo_subgroup_basic(regression_data):
    X, y = regression_data
    model = RuleEvoSubgroup(population_size=20, max_generations=5, random_state=42)
    model.fit(X, y)
    
    # Subgroup discovery is exploratory, predict() should raise NotImplementedError
    with pytest.raises(NotImplementedError):
        model.predict(X)

    rs = model.to_ruleset()
    assert rs.task_type == "regression"
    assert rs.aggregation.type == "independent"
    assert len(rs.rules) > 0

    for rule in rs.rules:
        assert "quality" in rule.metadata
        assert "coverage" in rule.metadata
        assert "local_mean" in rule.metadata
        assert "global_mean" in rule.metadata
        assert 0.0 <= rule.metadata["coverage"] <= 1.0


def test_rulegp_subgroup_basic(regression_data):
    X, y = regression_data
    model = RuleGPSubgroup(population_size=20, max_generations=5, random_state=42)
    model.fit(X, y)
    
    with pytest.raises(NotImplementedError):
        model.predict(X)

    rs = model.to_ruleset()
    assert rs.task_type == "regression"
    assert rs.aggregation.type == "independent"
    assert len(rs.rules) > 0

    for rule in rs.rules:
        assert "quality" in rule.metadata
        assert "coverage" in rule.metadata
        assert "local_mean" in rule.metadata
        assert "global_mean" in rule.metadata
        assert 0.0 <= rule.metadata["coverage"] <= 1.0
