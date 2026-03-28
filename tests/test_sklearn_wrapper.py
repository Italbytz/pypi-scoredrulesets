import json
from pathlib import Path
import warnings

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

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

    # Print the final model as a table (visible in pytest with -s).
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
    # No ExSTraCS backend required: test parameter sanitization directly.
    raw_params = {
        "conservative_prune": True,
        "prune_atoms": False,  # invalid key -> must be ignored
        "unknown_key": 123,
    }

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        sanitized = ScoredRuleSetClassifier._sanitize_exstracs_params(raw_params)

    assert sanitized == {"conservative_prune": True}
    assert any("Ignoring unknown exstracs_params keys" in str(w.message) for w in captured)


def test_cart_pruned_keeps_non_empty_rules_on_iris():
    X, y = load_iris(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    # Baseline without pruning
    clf_base = ScoredRuleSetClassifier(
        backend="cart",
        backend_params={"max_depth": 4},
        random_state=0,
    )
    clf_base.fit(X_tr, y_tr)
    y_pred_base = clf_base.predict(X_te)
    n_atoms_base = sum(len(r.atoms) for r in clf_base.to_ruleset().rules)

    # Auto-pruning (automatically scans multiple lambdas)
    clf_pruned = ScoredRuleSetClassifier(
        backend="cart",
        backend_params={"max_depth": 4},
        transform_params={"prune_atoms": True},
        random_state=0,
    )
    clf_pruned.fit(X_tr, y_tr)
    y_pred_pruned = clf_pruned.predict(X_te)
    ruleset = clf_pruned.to_ruleset()
    n_atoms_pruned = sum(len(r.atoms) for r in ruleset.rules)

    # Pruning must not create empty non-default rules
    assert n_atoms_pruned > 0
    assert all(len(rule.atoms) > 0 for rule in ruleset.rules)

    # Pruning must preserve predictions (F1 must not decrease)
    assert np.array_equal(y_pred_pruned, y_pred_base)

    # Pruning must reduce atoms (or at least not increase them)
    assert n_atoms_pruned <= n_atoms_base


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
        "scoredrulesets.estimators.ruleset_transform.rulekit_to_scored_ruleset",
        _fake_rulekit_to_scored_ruleset,
    )

    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    clf = ScoredRuleSetClassifier(backend="rulekit", random_state=0)
    clf.fit(X, y)

    assert "y_train" in captured
    np.testing.assert_array_equal(captured["y_train"], y)


