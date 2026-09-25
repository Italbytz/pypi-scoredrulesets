import numpy as np
import pytest
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from scoredrulesets.estimators import CascadedRuleGPRegressor
from scoredrulesets.runtime import predict_regression


def test_cascaded_rulegp_regressor_basic():
    X, y = load_diabetes(return_X_y=True)

    model = CascadedRuleGPRegressor(
        max_rules_stage1=2,
        max_rules_stage2=2,
        max_generations=5,
        population_size=15,
        random_state=42,
    )
    model.fit(X, y)

    preds = model.predict(X)
    assert preds.shape == (X.shape[0],)
    assert np.isfinite(preds).all()

    # Verify score method
    r2 = model.score(X, y)
    assert r2 > -1.0

    # Verify schema export & runtime consistency
    rs = model.to_ruleset()
    assert rs.task_type == "regression"
    assert rs.aggregation.type == "cascaded_sum"
    assert len(rs.rules) >= 2

    runtime_preds = predict_regression(rs, X)
    np.testing.assert_allclose(preds, runtime_preds, atol=1e-5)


def test_cascaded_rulegp_linear():
    X, y = load_diabetes(return_X_y=True)

    model = CascadedRuleGPRegressor(
        max_rules_stage1=2,
        max_rules_stage2=2,
        prediction_type="linear",
        max_generations=5,
        population_size=15,
        random_state=42,
    )
    model.fit(X, y)

    preds = model.predict(X)
    assert preds.shape == (X.shape[0],)
    assert np.isfinite(preds).all()
