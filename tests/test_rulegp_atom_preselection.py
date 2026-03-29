import numpy as np


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
