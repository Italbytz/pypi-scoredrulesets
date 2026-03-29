import numpy as np


def test_rulegp2_logicgp_singleton_preselection_emits_only_eq_atoms():
    from scoredrulesets.estimators.rulegp2 import RuleGP2Classifier

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

    clf = RuleGP2Classifier(
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


def test_rulegp2_logicgp_binned_sets_stays_close_to_logicgp_rlcw_macro_f1():
    from sklearn.datasets import load_wine
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    from scoredrulesets.estimators.logicgp import LogicGPClassifier, _discretize_features
    from scoredrulesets.estimators.rulegp2 import RuleGP2Classifier

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

        rulegp2 = RuleGP2Classifier(
            f1_averaging="macro",
            atom_space_strategy="hybrid",
            atom_preselection_strategy="logicgp_binned_sets",
            max_generations=160,
            stagnation_generations=35,
            n_adaptations_per_gen=16,
            population_size=90,
            random_state=seed,
        )
        rulegp2.fit(X_train_disc, y_train)
        f1_rulegp2 = f1_score(y_test, rulegp2.predict(X_test_disc), average="macro")

        diffs.append(float(f1_logicgp - f1_rulegp2))

    abs_diffs = np.abs(np.asarray(diffs, dtype=float))
    # Keep this guardrail tolerant to stochastic GP variance, but fail if
    # compatibility drifts strongly.
    assert float(abs_diffs.mean()) <= 0.10
    assert float(abs_diffs.max()) <= 0.16
