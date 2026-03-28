# -*- coding: utf-8 -*-
"""Benchmark: klassenbalancierte Gewichte auf car_evaluation."""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from scoredrulesets.benchmarking.datasets import load_dataset_registry
from scoredrulesets.estimators.logicgp import LogicGPClassifier

reg = load_dataset_registry(include_online_uci=True, include_synthetic=True)
b = reg["uci_car_evaluation"]
X, y = b.X, b.y
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

configs = [
    ("FLCW pop=None",  dict(trainer="flcw", population_size=None)),
    ("FLCW pop=50",    dict(trainer="flcw", population_size=50)),
    ("RLCW pop=None",  dict(trainer="rlcw", population_size=None)),
    ("RLCW pop=50",    dict(trainer="rlcw", population_size=50)),
]

print(f"{'Config':<20} {'F1-macro':>8} {'F1-micro':>8} {'Mons':>5} {'Atoms':>6}")
print("-" * 55)

for name, kwargs in configs:
    clf = LogicGPClassifier(
        f1_averaging="macro",
        max_generations=100, stagnation_generations=30,
        n_adaptations_per_gen=12,
        n_bins=5, random_state=0,
        **kwargs,
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    f1_mac = f1_score(y_te, y_pred, average="macro")
    f1_mic = f1_score(y_te, y_pred, average="micro")
    rs = clf.to_ruleset()
    n_mons = len([r for r in rs.rules if r.atoms])
    n_atoms = sum(len(r.atoms) for r in rs.rules)
    print(f"{name:<20} {f1_mac:>8.4f} {f1_mic:>8.4f} {n_mons:>5} {n_atoms:>6}")

# Detaillierter Report fuer den besten
print("\n--- Detaillierter Report: FLCW pop=None ---")
clf_best = LogicGPClassifier(
    trainer="flcw", f1_averaging="macro",
    max_generations=100, stagnation_generations=30,
    population_size=None,
    n_adaptations_per_gen=12,
    n_bins=5, random_state=0,
)
clf_best.fit(X_tr, y_tr)
y_pred_best = clf_best.predict(X_te)
print(classification_report(y_te, y_pred_best))
