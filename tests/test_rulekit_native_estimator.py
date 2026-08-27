import numpy as np
import pytest
from sklearn.datasets import make_classification

from scoredrulesets import available_atom_selection_strategies
from scoredrulesets import register_atom_selection_strategy
import scoredrulesets.estimators.rulekit_native as rulekit_native_module
from scoredrulesets.estimators.rulekit_native import RuleKitNativeClassifier


def _make_dataset(seed: int = 0):
    X, y = make_classification(
        n_samples=220,
        n_features=12,
        n_informative=4,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        random_state=seed,
    )
    return X.astype(float), y.astype(int)


def test_rulekit_native_registered_atom_preselection_records_metadata():
    X, y = _make_dataset(seed=42)

    def _first_k_strategy(candidates, _y_idx, _n_classes, min_samples_leaf, top_k):
        selected = set()
        for signature, mask in candidates:
            if int(np.sum(mask)) < int(min_samples_leaf):
                continue
            selected.add(signature)
            if len(selected) >= int(top_k):
                break
        return selected

    register_atom_selection_strategy("top_c2_private_ut", _first_k_strategy, overwrite=True)

    clf = RuleKitNativeClassifier(
        max_rules=12,
        max_conditions=4,
        min_samples_leaf=3,
        atom_preselection_strategy="top_c2_private_ut",
        atom_preselection_top_k=30,
        random_state=1,
    )
    clf.fit(X, y)
    ruleset = clf.to_ruleset()

    assert ruleset.metadata is not None
    assert ruleset.metadata.get("atom_preselection_strategy") == "top_c2_private_ut"
    assert ruleset.metadata.get("atom_preselection_top_k") == 30

    induced_rules = [
        r for r in ruleset.rules if r.metadata.get("kind") != "class_prior"
    ]
    assert len(induced_rules) > 0

    for rule in induced_rules:
        preselected = rule.metadata.get("preselected_atoms")
        assert isinstance(preselected, int)
        assert 1 <= preselected <= 30


def test_rulekit_native_rejects_missing_top_k_for_registered_strategy():
    register_atom_selection_strategy(
        "top_c2_private_ut",
        lambda candidates, y_idx, n_classes, min_samples_leaf, top_k: set(),
        overwrite=True,
    )
    with pytest.raises(ValueError, match="atom_preselection_top_k"):
        RuleKitNativeClassifier(atom_preselection_strategy="top_c2_private_ut")


def test_rulekit_native_rejects_unknown_atom_preselection_strategy():
    with pytest.raises(ValueError, match="atom_preselection_strategy"):
        RuleKitNativeClassifier(atom_preselection_strategy="unknown", atom_preselection_top_k=10)


def test_rulekit_native_accepts_registered_custom_atom_selection_strategy():
    X, y = _make_dataset(seed=3)

    def _first_k_strategy(candidates, _y_idx, _n_classes, min_samples_leaf, top_k):
        selected = set()
        for signature, mask in candidates:
            if int(np.sum(mask)) < int(min_samples_leaf):
                continue
            selected.add(signature)
            if len(selected) >= int(top_k):
                break
        return selected

    register_atom_selection_strategy("first_k_unit_test", _first_k_strategy, overwrite=True)

    clf = RuleKitNativeClassifier(
        max_rules=8,
        max_conditions=3,
        min_samples_leaf=3,
        atom_preselection_strategy="first_k_unit_test",
        atom_preselection_top_k=12,
        random_state=2,
    )
    clf.fit(X, y)
    ruleset = clf.to_ruleset()

    assert "first_k_unit_test" in available_atom_selection_strategies()
    assert ruleset.metadata is not None
    assert ruleset.metadata.get("atom_preselection_strategy") == "first_k_unit_test"


def test_rulekit_native_max_fit_seconds_raises_on_setup_timeout(monkeypatch):
    from scoredrulesets import FitBudgetExceededError

    X, y = _make_dataset(seed=7)
    ticks = iter(float(i) for i in range(2000))
    monkeypatch.setattr(rulekit_native_module.time, "monotonic", lambda: next(ticks))

    clf = RuleKitNativeClassifier(
        max_rules=10,
        max_conditions=4,
        max_fit_seconds=0.5,
        random_state=0,
    )
    with pytest.raises(FitBudgetExceededError):
        clf.fit(X, y)
