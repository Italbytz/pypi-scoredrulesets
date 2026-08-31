import numpy as np
import pytest

from scoredrulesets.estimators.atom_space import (
    build_native_feature_specs,
    build_ruleplcs_feature_info,
    fayyad_irani_cut_points,
)


def test_fayyad_irani_finds_clean_separation_cut():
    # Class 0 for x < 5, class 1 for x >= 5: one clean cut around 4.5.
    x = np.arange(10, dtype=float)
    y = (x >= 5).astype(int)
    cuts = fayyad_irani_cut_points(x, y)
    assert len(cuts) == 1
    assert 4.0 < cuts[0] < 5.0


def test_fayyad_irani_two_thresholds_for_three_intervals():
    x = np.arange(30, dtype=float)
    y = np.where(x < 10, 0, np.where(x < 20, 1, 2))
    cuts = fayyad_irani_cut_points(x, y)
    assert len(cuts) == 2
    assert 9.0 < cuts[0] < 10.0
    assert 19.0 < cuts[1] < 20.0


def test_fayyad_irani_no_cut_for_pure_noise():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = rng.integers(0, 2, size=200)
    cuts = fayyad_irani_cut_points(x, y)
    assert cuts == []


def test_fayyad_irani_no_cut_for_single_class():
    x = np.arange(10, dtype=float)
    y = np.zeros(10, dtype=int)
    assert fayyad_irani_cut_points(x, y) == []


def test_fayyad_irani_respects_max_cut_points():
    x = np.arange(40, dtype=float)
    y = (x // 10).astype(int)  # 4 classes -> up to 3 cuts
    cuts = fayyad_irani_cut_points(x, y, max_cut_points=2)
    assert len(cuts) <= 2


def test_build_specs_supervised_mdl_uses_informative_thresholds():
    n = 120
    x0 = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]) * 3.0 + np.linspace(
        0, 0.4, n
    )
    # Feature 1 is pure noise, should yield no supervised threshold atoms.
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=n)
    X = np.column_stack([x0, x1])
    y = (np.arange(n) >= n // 2).astype(int)

    specs = build_native_feature_specs(
        X, strategy="numeric_only", y=y, continuous_threshold_strategy="supervised_mdl"
    )
    # Informative feature gets at least one supervised cut point.
    assert len(specs[0]["thresholds"]) >= 1
    # Noise feature gets no supervised threshold atoms.
    assert specs[1]["thresholds"] == []


def test_build_specs_supervised_mdl_without_labels_falls_back_to_quantile():
    X = np.linspace(0, 1, 50).reshape(-1, 1)
    specs = build_native_feature_specs(
        X,
        strategy="numeric_only",
        y=None,
        continuous_threshold_strategy="supervised_mdl",
    )
    # With no labels, supervised discretization is inactive -> quantile thresholds.
    assert len(specs[0]["thresholds"]) >= 1


def test_build_specs_unknown_continuous_strategy_raises():
    X = np.zeros((4, 1), dtype=float)
    with pytest.raises(ValueError, match="Unknown continuous threshold strategy"):
        build_native_feature_specs(
            X, continuous_threshold_strategy="nope"  # type: ignore[arg-type]
        )


def test_rulegp_supervised_mdl_fits_and_predicts():
    from scoredrulesets.estimators.rulegp import RuleGPClassifier

    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    clf = RuleGPClassifier(
        continuous_threshold_strategy="supervised_mdl",
        min_samples_leaf=1,
        random_state=0,
    )
    clf.fit(X, y)
    pred = clf.predict(X)
    assert pred.shape == (80,)
    assert set(np.unique(pred)).issubset({0, 1})


def test_ruleplcs_feature_info_genotype_aware_marks_snp_categorical():
    # Feature 0: SNP genotype {0,1,2}. Feature 1: continuous.
    rng = np.random.default_rng(0)
    X = np.column_stack(
        [
            rng.integers(0, 3, size=60).astype(float),
            rng.normal(size=60),
        ]
    )
    info = build_ruleplcs_feature_info(
        X, low_cardinality_threshold=10, strategy="genotype_aware"
    )
    assert info[0]["numeric"] is False
    assert info[0]["values"] == {0, 1, 2}
    assert info[1]["numeric"] is True


def test_ruleplcs_genotype_aware_end_to_end():
    from scoredrulesets.estimators.ruleplcs import RulePLCSClassifier

    rng = np.random.default_rng(0)
    X = rng.integers(0, 3, size=(80, 5)).astype(float)
    y = (X[:, 0] >= 1).astype(int)

    clf = RulePLCSClassifier(
        feature_typing_strategy="genotype_aware",
        random_state=0,
    )
    clf.fit(X, y)
    pred = clf.predict(X)
    assert pred.shape == (80,)
    assert set(np.unique(pred)).issubset({0, 1})
