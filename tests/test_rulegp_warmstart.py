import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_iris


def test_rulegp_warmstart_none_default():
    from scoredrulesets.estimators.rulegp import RuleGPClassifier

    X, y = load_iris(return_X_y=True)
    clf = RuleGPClassifier(
        warmstart_strategy="none",
        max_generations=5,
        population_size=20,
        random_state=42,
    )
    clf.fit(X, y)
    rs = clf.to_ruleset()
    assert rs.metadata["warmstart_strategy"] == "none"
    assert hasattr(clf, "ruleset_")
    preds = clf.predict(X)
    assert preds.shape == (X.shape[0],)


def test_rulegp_warmstart_invalid_raises():
    from scoredrulesets.estimators.rulegp import RuleGPClassifier

    with pytest.raises(ValueError, match="warmstart_strategy"):
        RuleGPClassifier(warmstart_strategy="invalid_strategy")


def test_rulegp_warmstart_rulefit_execution():
    from scoredrulesets.estimators.rulegp import RuleGPClassifier

    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = [f"feat_{i}" for i in range(X.shape[1])]

    clf = RuleGPClassifier(
        warmstart_strategy="rulefit",
        warmstart_max_rules=20,
        warmstart_jaccard_max=0.8,
        max_generations=10,
        stagnation_generations=5,
        population_size=30,
        n_adaptations_per_gen=10,
        feature_names=feature_names,
        random_state=42,
    )
    clf.fit(X, y)

    rs = clf.to_ruleset()
    assert rs.metadata["warmstart_strategy"] == "rulefit"
    assert rs.metadata["warmstart_max_rules"] == 20
    assert len(rs.rules) >= 1

    # Predict checks
    preds = clf.predict(X)
    assert preds.shape == (X.shape[0],)
    assert set(np.unique(preds)).issubset({0, 1})

    probs = clf.predict_proba(X)
    assert probs.shape == (X.shape[0], 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_extract_rulefit_components_direct():
    from scoredrulesets.warmstart.rulefit_warmstart import extract_rulefit_components

    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = [f"feat_{i}" for i in range(X.shape[1])]

    curated_atoms, seed_rules = extract_rulefit_components(
        X_train=X,
        y_train=y,
        feature_names=feature_names,
        max_rules=15,
        min_samples_split=2,
        max_thresholds_per_feature=3,
        max_rules_seed=10,
        jaccard_max_sim=0.8,
        random_state=42,
    )

    assert isinstance(curated_atoms, list)
    assert isinstance(seed_rules, list)
    assert len(curated_atoms) > 0
    assert len(seed_rules) > 0

    # Ensure atoms format: (feature_idx, op, threshold)
    for fi, op, thr in curated_atoms:
        assert isinstance(fi, int)
        assert 0 <= fi < X.shape[1]
        assert op in ("<=", "<", ">=", ">", "==")
        assert isinstance(thr, float)


def test_rulegp_warmstart_ablation_modes():
    from scoredrulesets.estimators.rulegp import RuleGPClassifier

    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = [f"feat_{i}" for i in range(X.shape[1])]

    for mode in ("rulefit_atoms_only", "rulefit_seeds_only"):
        clf = RuleGPClassifier(
            warmstart_strategy=mode,
            warmstart_max_rules=15,
            feature_names=feature_names,
            max_generations=5,
            population_size=20,
            random_state=42,
        )
        clf.fit(X, y)
        rs = clf.to_ruleset()
        assert rs.metadata["warmstart_strategy"] == mode
        preds = clf.predict(X)
        assert preds.shape == (X.shape[0],)
