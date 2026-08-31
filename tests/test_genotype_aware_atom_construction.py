import numpy as np
import pytest

from scoredrulesets.estimators.atom_space import (
    build_native_feature_specs,
    genotype_feature_spec,
    genotype_levels,
)


def test_genotype_levels_detects_snp_encoding():
    col = np.array([0, 1, 2, 1, 0, 2, 1, 0], dtype=float)
    assert genotype_levels(col) == [0.0, 1.0, 2.0]


def test_genotype_levels_detects_ordinal_counts():
    col = np.array([0, 1, 2, 3, 4, 2, 1, 0], dtype=float)
    assert genotype_levels(col) == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_genotype_levels_rejects_continuous_feature():
    col = np.array([0.1, 1.7, 2.3, 0.9, 1.1], dtype=float)
    assert genotype_levels(col) is None


def test_genotype_levels_rejects_negative_values():
    col = np.array([-1, 0, 1, 2], dtype=float)
    assert genotype_levels(col) is None


def test_genotype_levels_rejects_single_level():
    col = np.array([1, 1, 1, 1], dtype=float)
    assert genotype_levels(col) is None


def test_genotype_levels_respects_max_levels():
    col = np.arange(20, dtype=float)
    assert genotype_levels(col, max_levels=10) is None
    assert genotype_levels(col, max_levels=20) == col.tolist()


def test_genotype_feature_spec_thresholds_for_snp():
    spec = genotype_feature_spec(3, [0.0, 1.0, 2.0])
    assert spec == {
        "idx": 3,
        "kind": "num",
        "thresholds": [0.5, 1.5],
        "intervals": [],
    }


def test_build_specs_genotype_aware_uses_compact_genetic_model_atoms():
    # Column 0: SNP genotype {0,1,2}. Column 1: continuous.
    X = np.array(
        [
            [0, 0.13],
            [1, 1.90],
            [2, 2.35],
            [1, 0.77],
            [0, 1.42],
            [2, 3.10],
            [1, 2.01],
            [0, 0.44],
        ],
        dtype=float,
    )
    specs = build_native_feature_specs(X, strategy="genotype_aware")

    genotype_spec = specs[0]
    assert genotype_spec["kind"] == "num"
    assert genotype_spec["thresholds"] == [0.5, 1.5]
    assert genotype_spec["intervals"] == []

    continuous_spec = specs[1]
    assert continuous_spec["kind"] == "num"
    # Continuous features keep the richer threshold/interval construction.
    assert len(continuous_spec["thresholds"]) >= 2
    assert len(continuous_spec["intervals"]) >= 1


def test_build_specs_genotype_aware_keeps_categorical_string_features():
    X = np.array(
        [
            [0, "a"],
            [1, "b"],
            [2, "a"],
            [1, "c"],
        ],
        dtype=object,
    )
    specs = build_native_feature_specs(X, strategy="genotype_aware")
    assert specs[0]["kind"] == "num"
    assert specs[0]["thresholds"] == [0.5, 1.5]
    assert specs[1]["kind"] == "cat"
    assert set(specs[1]["categories"]) == {"a", "b", "c"}


def test_build_specs_unknown_strategy_raises():
    X = np.zeros((4, 2), dtype=float)
    with pytest.raises(ValueError, match="Unknown native atom-space strategy"):
        build_native_feature_specs(X, strategy="not_a_strategy")  # type: ignore[arg-type]


def test_rulegp_genotype_aware_builds_dominant_and_recessive_atoms():
    from scoredrulesets.estimators.rulegp import RuleGPClassifier, _build_feature_specs

    X = np.array(
        [
            [0, 0],
            [1, 0],
            [2, 1],
            [1, 1],
            [0, 2],
            [2, 2],
            [1, 0],
            [0, 1],
        ],
        dtype=float,
    )
    y = np.array([0, 0, 1, 1, 0, 1, 0, 0], dtype=int)

    clf = RuleGPClassifier(
        atom_space_strategy="genotype_aware",
        min_samples_leaf=1,
        random_state=0,
    )
    specs = _build_feature_specs(X, atom_space_strategy="genotype_aware")
    pool = clf._build_atom_pool(specs, X, y, n_classes=2)

    atoms = [a for atoms in pool.values() for a in atoms]
    # Dominant model (x >= 1) is the "> 0.5" atom; recessive (x == 2) is "> 1.5".
    thresholds_gt = {a.value for a in atoms if a.op == ">"}
    assert 0.5 in thresholds_gt
    assert 1.5 in thresholds_gt
    # No subset ("in") or interval ("between") atoms for genotype features.
    ops = {a.op for a in atoms}
    assert ops <= {"<=", ">"}


def test_rulegp_genotype_aware_fits_and_predicts():
    from scoredrulesets.estimators.rulegp import RuleGPClassifier

    rng = np.random.default_rng(0)
    X = rng.integers(0, 3, size=(60, 5)).astype(float)
    # Label depends on a dominant model of feature 0.
    y = (X[:, 0] >= 1).astype(int)

    clf = RuleGPClassifier(
        atom_space_strategy="genotype_aware",
        min_samples_leaf=1,
        random_state=0,
    )
    clf.fit(X, y)
    pred = clf.predict(X)
    assert pred.shape == (60,)
    assert set(np.unique(pred)).issubset({0, 1})
