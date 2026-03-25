"""Diagnose: logicGP final selection gap – consolidated recall vs actual F1."""
import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from scoredrulesets.estimators.logicgp import (
    LogicGPClassifier,
    _final_model_selection,
    _compute_weights,
    _evaluate_fitness,
    _discretize_features,
)
from scoredrulesets.runtime import predict as predict_from_ruleset


def diagnose(name, X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    clf = LogicGPClassifier(
        trainer="flcw",
        f1_averaging="macro",
        max_generations=50,
        stagnation_generations=20,
        population_size=30,
        n_adaptations_per_gen=6,
        n_bins=5,
        random_state=0,
    )
    clf.fit(X_tr, y_tr)

    # Vorhersage über ScoredRuleSet (wie im Benchmark)
    y_pred = clf.predict(X_te)
    f1_selected = f1_score(y_te, y_pred, average="macro")

    # Jetzt: alle Kandidaten aus dem GP-Lauf auswerten
    # Wir patchen _run_gp um die all_candidates zu sammeln
    print(f"\n=== {name} ===")
    print(f"  F1 (selected model, test): {f1_selected:.4f}")
    print(f"  Rules: {len(clf.ruleset_.rules)}, Atoms: {sum(len(r.atoms) for r in clf.ruleset_.rules)}")

    # Vergleich: was wäre F1 mit längerer Laufzeit?
    for gens, stag in [(200, 80), (500, 200), (1000, 500)]:
        clf2 = LogicGPClassifier(
            trainer="flcw",
            f1_averaging="macro",
            max_generations=gens,
            stagnation_generations=stag,
            population_size=30,
            n_adaptations_per_gen=6,
            n_bins=5,
            random_state=0,
        )
        clf2.fit(X_tr, y_tr)
        y_pred2 = clf2.predict(X_te)
        f1_2 = f1_score(y_te, y_pred2, average="macro")
        n_rules = len(clf2.ruleset_.rules)
        n_atoms = sum(len(r.atoms) for r in clf2.ruleset_.rules)
        print(f"  gens={gens:5d} stag={stag:4d}: F1={f1_2:.4f}  rules={n_rules}  atoms={n_atoms}")


for name, loader in [("iris", load_iris), ("wine", load_wine)]:
    X, y = loader(return_X_y=True)
    diagnose(name, X, y)

