import numpy as np
import pytest
from sklearn.datasets import make_regression

from scoredrulesets.estimators.rulegp_regressor import RuleGPRegressor
from scoredrulesets.estimators.rulensga2_regressor import RuleNSGA2Regressor
from scoredrulesets.runtime import predict_regression


@pytest.fixture
def linear_data():
    X, y = make_regression(n_samples=100, n_features=3, random_state=42, noise=0.1)
    return X, y


def test_rulegp_takagi_sugeno(linear_data):
    X, y = linear_data
    model = RuleGPRegressor(prediction_type="linear", population_size=20, max_generations=5, random_state=42)
    model.fit(X, y)
    
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert not np.isnan(preds).any()

    rs = model.to_ruleset()
    assert rs.task_type == "regression"
    assert rs.aggregation.type == "takagi_sugeno"
    
    # Each rule should have len(feature_names) + 1 scores (slopes + intercept)
    expected_scores_len = X.shape[1] + 1
    for rule in rs.rules:
        assert len(rule.scores) == expected_scores_len

    # Validate predict_regression from runtime produces identical predictions
    rt_preds = predict_regression(rs, X)
    np.testing.assert_allclose(preds, rt_preds, rtol=1e-5)


def test_rulensga2_takagi_sugeno(linear_data):
    X, y = linear_data
    model = RuleNSGA2Regressor(prediction_type="linear", population_size=20, max_generations=5, random_state=42)
    model.fit(X, y)
    
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert not np.isnan(preds).any()

    rs = model.to_ruleset()
    assert rs.task_type == "regression"
    assert rs.aggregation.type == "takagi_sugeno"
    
    expected_scores_len = X.shape[1] + 1
    for rule in rs.rules:
        assert len(rule.scores) == expected_scores_len

    rt_preds = predict_regression(rs, X)
    np.testing.assert_allclose(preds, rt_preds, rtol=1e-5)
