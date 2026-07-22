import numpy as np

from scoredrulesets import ScoredRuleSetClassifier


def test_rulegp_logicgp_singleton_preselection_emits_only_eq_atoms():
    from scoredrulesets.estimators.rulegp import RuleGPClassifier

    # Low-cardinality integer features emulate discretized logicGP inputs.
    X = np.array(
        [
            [0, 0, 1],
            [0, 1, 1],
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 0],
            [1, 1, 1],
            [0, 1, 0],
            [1, 0, 1],
        ],
        dtype=float,
    )
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=int)

    clf = RuleGPClassifier(
        atom_space_strategy="categorical_low_cardinality_only",
        atom_preselection_strategy="logicgp_singleton",
        max_generations=30,
        stagnation_generations=10,
        population_size=40,
        n_adaptations_per_gen=10,
        random_state=7,
    )
    clf.fit(X, y)

    ruleset = clf.to_ruleset()
    for rule in ruleset.rules:
        for atom in rule.atoms:
            assert atom.op == "=="


def test_rulegp_logicgp_binned_sets_stays_close_to_logicgp_rlcw_macro_f1():
    from sklearn.datasets import load_wine
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    from scoredrulesets.estimators.logicgp import LogicGPClassifier, _discretize_features
    from scoredrulesets.estimators.rulegp import RuleGPClassifier

    X, y = load_wine(return_X_y=True)
    seeds = [0, 1, 2, 3]
    diffs: list[float] = []

    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=seed,
            stratify=y,
        )

        logicgp = LogicGPClassifier(
            trainer="rlcw",
            f1_averaging="macro",
            n_bins=5,
            feature_encoding_strategy="auto_low_cardinality",
            literal_generator="singleton",
            max_generations=160,
            stagnation_generations=35,
            n_adaptations_per_gen=16,
            population_size=90,
            random_state=seed,
        )
        logicgp.fit(X_train, y_train)
        f1_logicgp = f1_score(y_test, logicgp.predict(X_test), average="macro")

        X_train_disc, binners, cat_masks = _discretize_features(
            X_train,
            n_bins=5,
            strategy="auto_low_cardinality",
        )
        X_test_disc, _, _ = _discretize_features(
            X_test,
            fitted_binners=binners,
            cat_masks=cat_masks,
            strategy="auto_low_cardinality",
        )

        rulegp = RuleGPClassifier(
            f1_averaging="macro",
            atom_space_strategy="hybrid",
            atom_preselection_strategy="logicgp_binned_sets",
            max_generations=160,
            stagnation_generations=35,
            n_adaptations_per_gen=16,
            population_size=90,
            random_state=seed,
        )
        rulegp.fit(X_train_disc, y_train)
        f1_rulegp = f1_score(y_test, rulegp.predict(X_test_disc), average="macro")

        diffs.append(float(f1_logicgp - f1_rulegp))

    abs_diffs = np.abs(np.asarray(diffs, dtype=float))
    # Keep this guardrail tolerant to stochastic GP variance, but fail if
    # compatibility drifts strongly.
    assert float(abs_diffs.mean()) <= 0.10
    assert float(abs_diffs.max()) <= 0.16


def test_wrapper_rulegp_logicgp_mode_smoke_runs_on_raw_numeric_data():
    from sklearn.datasets import load_wine

    X, y = load_wine(return_X_y=True)
    clf = ScoredRuleSetClassifier(
        backend="rulegp",
        backend_params={
            "atom_space_strategy": "hybrid",
            "atom_preselection_strategy": "logicgp_binned_sets",
            "max_generations": 40,
            "stagnation_generations": 15,
            "population_size": 40,
            "n_adaptations_per_gen": 10,
        },
        random_state=0,
    )
    clf.fit(X, y)

    pred = clf.predict(X[:12])
    proba = clf.predict_proba(X[:12])
    assert pred.shape == (12,)
    assert proba.shape[0] == 12
    assert clf.to_ruleset().metadata["source"] == "rulegp"


def test_rulegp_supports_shortest_zero_train_mcr_mode():
    from scoredrulesets.estimators.rulegp import RuleGPClassifier

    X = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 0],
            [1, 1],
            [0, 1],
            [1, 0],
        ],
        dtype=float,
    )
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=int)

    clf = RuleGPClassifier(
        model_selection="shortest_zero_train_mcr",
        validation_fraction=0.0,
        max_generations=20,
        stagnation_generations=5,
        population_size=30,
        n_adaptations_per_gen=8,
        random_state=3,
    )
    clf.fit(X, y)

    assert clf.to_ruleset().metadata["model_selection"] == "shortest_zero_train_mcr"
