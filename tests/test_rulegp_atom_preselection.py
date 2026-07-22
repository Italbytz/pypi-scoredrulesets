import numpy as np
import pytest

from scoredrulesets import register_atom_selection_strategy


def test_rulegp_logicgp_binned_sets_preselection_builds_eq_in_between_atoms():
    from scoredrulesets.estimators.rulegp import RuleGPClassifier
    from scoredrulesets.estimators.rulegp import _build_feature_specs

    # Already binned-like data with ordered bin ids.
    X = np.array(
        [
            [0, 0],
            [1, 0],
            [2, 1],
            [3, 1],
            [0, 1],
            [1, 1],
            [2, 0],
            [3, 0],
        ],
        dtype=float,
    )
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=int)

    clf = RuleGPClassifier(
        atom_space_strategy="categorical_low_cardinality_only",
        atom_preselection_strategy="logicgp_binned_sets",
        min_samples_leaf=1,
        random_state=3,
    )

    specs = _build_feature_specs(
        X,
        atom_space_strategy="categorical_low_cardinality_only",
    )
    pool = clf._build_atom_pool(specs, X, y, n_classes=2)

    ops = {a.op for atoms in pool.values() for a in atoms}
    assert "==" in ops
    assert "in" in ops
    assert "between" in ops


def test_rulegp_registered_strategy_preselection_records_metadata():
    from sklearn.datasets import load_iris

    from scoredrulesets.estimators.rulegp import RuleGPClassifier

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

    X, y = load_iris(return_X_y=True)
    clf = RuleGPClassifier(
        atom_preselection_strategy="top_c2_private_ut",
        atom_preselection_top_k=64,
        max_generations=20,
        stagnation_generations=6,
        population_size=30,
        n_adaptations_per_gen=8,
        random_state=2,
    )
    clf.fit(X, y)

    rs = clf.to_ruleset()
    assert rs.metadata["atom_preselection_strategy"] == "top_c2_private_ut"
    assert rs.metadata["atom_preselection_top_k"] == 64


def test_rulegp_registered_strategy_requires_positive_top_k():
    from scoredrulesets.estimators.rulegp import RuleGPClassifier

    register_atom_selection_strategy(
        "top_c2_private_ut",
        lambda candidates, y_idx, n_classes, min_samples_leaf, top_k: set(),
        overwrite=True,
    )

    with pytest.raises(ValueError, match="atom_preselection_top_k"):
        RuleGPClassifier(atom_preselection_strategy="top_c2_private_ut")


def test_rulegp_public_benchmark_profile_does_not_expose_top_c2():
    from scoredrulesets.benchmarking.estimators import default_estimator_specs

    specs = default_estimator_specs()
    assert "wrapper_rulegp_topc2" not in specs
