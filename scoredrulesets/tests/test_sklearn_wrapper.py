from pathlib import Path

from sklearn.datasets import load_iris

from scoredrulesets import ScoredRuleSetClassifier


def test_cart_backend_and_ruleset_reload(tmp_path: Path):
    X, y = load_iris(return_X_y=True)
    clf = ScoredRuleSetClassifier(
        backend="cart",
        backend_params={"max_depth": 3},
        transform_params={"depth_decay_lambda": 2.0},
        random_state=0,
    )
    clf.fit(X, y)

    pred = clf.predict(X[:5])
    assert pred.shape == (5,)

    ruleset = clf.to_ruleset()
    assert len(ruleset.rules) > 0

    model_file = tmp_path / "iris_ruleset.json"
    clf.save_ruleset(model_file)

    loaded = ScoredRuleSetClassifier.from_ruleset_json(model_file)
    pred_loaded = loaded.predict(X[:5])
    assert pred_loaded.shape == (5,)

