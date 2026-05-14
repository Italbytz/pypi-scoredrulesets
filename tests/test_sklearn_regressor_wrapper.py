from pathlib import Path

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor

from scoredrulesets import ScoredRuleSetRegressor


def test_cart_regressor_backend_and_ruleset_reload(tmp_path: Path):
    X, y = load_diabetes(return_X_y=True)

    reg = ScoredRuleSetRegressor(
        backend="cart",
        backend_params={"max_depth": 4},
        random_state=0,
    )
    reg.fit(X, y)

    pred = reg.predict(X[:8])
    assert pred.shape == (8,)

    ruleset = reg.to_ruleset()
    assert ruleset.task_type == "regression"
    assert len(ruleset.rules) > 0

    model_file = tmp_path / "diabetes_ruleset_regression.json"
    reg.save_ruleset(model_file)

    loaded = ScoredRuleSetRegressor.from_ruleset_json(model_file)
    pred_loaded = loaded.predict(X[:8])
    assert pred_loaded.shape == (8,)
    np.testing.assert_allclose(pred_loaded, pred, rtol=1e-12, atol=1e-12)


def test_custom_tree_estimator_path_matches_sklearn_tree_predictions():
    X, y = load_diabetes(return_X_y=True)

    tree = DecisionTreeRegressor(max_depth=3, random_state=7)
    reg = ScoredRuleSetRegressor(estimator=tree)
    reg.fit(X, y)

    tree_pred = reg.estimator_.predict(X[:12])
    ruleset_pred = reg.predict(X[:12])

    np.testing.assert_allclose(ruleset_pred, tree_pred, rtol=1e-12, atol=1e-12)
