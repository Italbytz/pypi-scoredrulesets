import json
from pathlib import Path
import warnings

import numpy as np
from sklearn.datasets import load_iris

from scoredrulesets import ScoredRuleSetClassifier, format_ruleset_table
from scoredrulesets.schema import AggregationSpec, Rule, ScoredRuleSet


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


def test_cart_backend_prints_ruleset_table(capsys):
    X, y = load_iris(return_X_y=True)
    clf = ScoredRuleSetClassifier(
        backend="cart",
        backend_params={"max_depth": 2},
        random_state=0,
    )
    clf.fit(X, y)

    # Modell am Ende als Tabelle ausgeben (mit -s sichtbar in pytest).
    print(format_ruleset_table(clf.to_ruleset()))
    captured = capsys.readouterr()
    assert "| idx " in captured.out
    assert "| rule_id " in captured.out
    assert "| condition " in captured.out
    assert "| scores " in captured.out


def test_cart_backend_writes_ruleset_output_file(tmp_path: Path):
    X, y = load_iris(return_X_y=True)
    clf = ScoredRuleSetClassifier(
        backend="cart",
        backend_params={"max_depth": 2},
        random_state=0,
    )
    clf.fit(X, y)

    ruleset_json = json.dumps(clf.to_ruleset().to_dict(), indent=2, sort_keys=True)
    output_file = tmp_path / "printed_ruleset.json"
    output_file.write_text(ruleset_json, encoding="utf-8")

    loaded_payload = json.loads(output_file.read_text(encoding="utf-8"))
    assert loaded_payload["format"] == "scoredrulesets"
    assert isinstance(loaded_payload["rules"], list)
    assert len(loaded_payload["rules"]) > 0


def test_exstracs_params_unknown_keys_are_filtered():
    # Kein ExSTraCS-Backend nötig: wir testen die Param-Säuberung direkt.
    raw_params = {
        "conservative_prune": True,
        "prune_atoms": False,  # Legacy/invalid key -> must be ignored
        "unknown_key": 123,
    }

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        sanitized = ScoredRuleSetClassifier._sanitize_exstracs_params(raw_params)

    assert sanitized == {"conservative_prune": True}
    assert any("Ignoring unknown exstracs_params keys" in str(w.message) for w in captured)


def test_cart_pruned_keeps_non_empty_rules_on_iris():
    X, y = load_iris(return_X_y=True)
    clf = ScoredRuleSetClassifier(
        backend="cart",
        backend_params={"max_depth": 4},
        transform_params={
            "prune_atoms": True,
            "prune_lambda": 1.5,
        },
        random_state=0,
    )
    clf.fit(X, y)

    ruleset = clf.to_ruleset()
    n_atoms = sum(len(rule.atoms) for rule in ruleset.rules)

    # Regression guard: pruning must not collapse all non-default rules to atoms=[]
    assert n_atoms > 0
    assert all(len(rule.atoms) > 0 for rule in ruleset.rules)


def test_rulekit_wrapper_passes_training_labels_to_transform(monkeypatch):
    captured: dict[str, np.ndarray] = {}

    class _FakeRuleKitEstimator:
        def fit(self, X, y):
            self.classes_ = np.unique(y)
            return self

    def _fake_build_backend_estimator(*, backend, backend_params, random_state):
        assert backend == "rulekit"
        return _FakeRuleKitEstimator()

    def _fake_rulekit_to_scored_ruleset(estimator, class_labels, feature_names, y_train=None):
        assert estimator is not None
        captured["y_train"] = np.asarray(y_train)
        return ScoredRuleSet(
            class_labels=class_labels,
            feature_names=feature_names,
            rules=[
                Rule(atoms=[], scores=[1.0 / len(class_labels)] * len(class_labels), rule_id="default")
            ],
            aggregation=AggregationSpec(type="argmax_sum", temperature=1.0),
        )

    monkeypatch.setattr(
        "scoredrulesets.estimators.sklearn_wrapper.build_backend_estimator",
        _fake_build_backend_estimator,
    )
    monkeypatch.setattr(
        "scoredrulesets.estimators.sklearn_wrapper.rulekit_to_scored_ruleset",
        _fake_rulekit_to_scored_ruleset,
    )

    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    clf = ScoredRuleSetClassifier(backend="rulekit", random_state=0)
    clf.fit(X, y)

    assert "y_train" in captured
    np.testing.assert_array_equal(captured["y_train"], y)


