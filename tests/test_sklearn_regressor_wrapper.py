from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor

from scoredrulesets import ScoredRuleSetRegressor
from scoredrulesets.schema import AggregationSpec, Atom, Rule, ScoredRuleSet


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


@pytest.mark.parametrize("backend", ["rulegp", "rulensga2"])
def test_rule_backends_regression_projection_path(monkeypatch, backend: str):
    captured: dict[str, np.ndarray] = {}

    class _FakeRuleBackend:
        def fit(self, X, y):
            captured["y_encoded"] = np.asarray(y)
            self.classes_ = np.unique(y)
            self.ruleset_ = ScoredRuleSet(
                class_labels=self.classes_.tolist(),
                feature_names=["f0"],
                rules=[
                    Rule(
                        atoms=[Atom(feature="f0", op=">", value=0.0)],
                        scores=[0.0, 1.0, 0.0],
                        rule_id="r1",
                    ),
                    Rule(atoms=[], scores=[1.0, 0.0, 0.0], rule_id="default"),
                ],
                aggregation=AggregationSpec(type="argmax_sum", temperature=1.0),
            )
            return self

    def _fake_builder(*, backend, backend_params, random_state):
        assert backend in {"rulegp", "rulensga2"}
        return _FakeRuleBackend()

    monkeypatch.setattr(
        "scoredrulesets.estimators.sklearn_wrapper.build_backend_estimator",
        _fake_builder,
    )

    X = np.array([[-1.0], [0.2], [0.6], [1.5], [2.0], [3.0]], dtype=float)
    y = np.array([10.0, 12.0, 15.0, 30.0, 35.0, 40.0], dtype=float)

    reg = ScoredRuleSetRegressor(
        backend=backend,
        target_bins=3,
        random_state=0,
    )
    reg.fit(X, y)

    assert "y_encoded" in captured
    assert np.issubdtype(captured["y_encoded"].dtype, np.integer)
    assert reg.to_ruleset().task_type == "regression"
    assert reg.to_ruleset().metadata.get("source_backend") == backend

    pred = reg.predict(np.array([[-2.0], [1.0]], dtype=float))
    assert pred.shape == (2,)
    assert np.isfinite(pred).all()
