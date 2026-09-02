import numpy as np
import pytest
from sklearn.datasets import make_regression

from scoredrulesets.estimators.rulegp_regressor import RuleGPRegressor
from scoredrulesets.runtime import predict_regression


@pytest.fixture
def multioutput_data():
    X, y = make_regression(n_samples=100, n_features=4, n_targets=3, random_state=42, noise=0.1)
    return X, y


def test_rulegp_multioutput(multioutput_data):
    X, y = multioutput_data
    model = RuleGPRegressor(population_size=20, max_generations=5, random_state=42)
    model.fit(X, y)
    
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert not np.isnan(preds).any()

    rs = model.to_ruleset()
    assert rs.task_type == "regression"
    assert rs.aggregation.type == "mean_active"
    
    # Each rule should have 3 scores (one per target)
    for rule in rs.rules:
        assert len(rule.scores) == 3

    # Validate predict_regression from runtime produces identical predictions
    rt_preds = predict_regression(rs, X)
    np.testing.assert_allclose(preds, rt_preds, rtol=1e-5)
