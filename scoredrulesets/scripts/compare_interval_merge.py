#!/usr/bin/env python -u
"""Kurzer Vergleich: Modellgroesse und F1 mit/ohne Interval-Merge."""
from __future__ import annotations

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from scoredrulesets.estimators.sklearn_wrapper import ScoredRuleSetClassifier


DATASETS = [
    ("Iris", load_iris),
    ("Wine", load_wine),
    ("Breast Cancer", load_breast_cancer),
]

VARIANTS = [
    ("baseline", None),
    ("interval_merge_0.3", {"interval_merge": True, "interval_merge_iou_threshold": 0.3}),
    ("interval_merge_0.1", {"interval_merge": True, "interval_merge_iou_threshold": 0.1}),
    ("interval_merge_0.5", {"interval_merge": True, "interval_merge_iou_threshold": 0.5}),
    ("imerge+filter", {"interval_merge": True, "interval_merge_iou_threshold": 0.3,
                        "filter_weak_rules": True, "min_fitness_percentile": 0.2}),
    ("imerge+cons+consol", {"interval_merge": True, "interval_merge_iou_threshold": 0.3,
                             "conservative_prune": True, "consolidate_similar": True}),
    ("shrink_all (alt)", {"conservative_prune": True, "filter_weak_rules": True,
                          "consolidate_similar": True, "aggressive_prune": True,
                          "max_f1_loss": 0.01}),
]


def count_rules_atoms(clf: ScoredRuleSetClassifier):
    rs = clf.to_ruleset()
    n_rules = len([r for r in rs.rules if r.atoms])
    n_atoms = sum(len(r.atoms) for r in rs.rules)
    return n_rules, n_atoms


def main():
    print("=" * 100)
    print("ExSTraCS INTERVAL-MERGE VERGLEICH: Modellgroesse & F1")
    print("=" * 100)

    for ds_name, loader in DATASETS:
        X, y = loader(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y,
        )
        feature_names = [f"f{i}" for i in range(X.shape[1])]

        print(f"\n{'─' * 100}")
        print(f"  {ds_name}: {X_train.shape[0]} train / {X_test.shape[0]} test, {X.shape[1]} features")
        print(f"{'─' * 100}")
        print(f"  {'Variante':<25} {'Rules':>6} {'Atoms':>6} {'F1-test':>8} {'F1-train':>9} {'Zeit (s)':>9}")
        print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*8} {'-'*9} {'-'*9}")

        for variant_name, exstracs_params in VARIANTS:
            try:
                clf = ScoredRuleSetClassifier(
                    backend="exstracs",
                    backend_params={},
                    exstracs_params=exstracs_params,
                    random_state=0,
                )
                t0 = time.time()
                clf.fit(X_train, y_train)
                elapsed = time.time() - t0

                y_pred_test = clf.predict(X_test)
                y_pred_train = clf.predict(X_train)
                f1_test = f1_score(y_test, y_pred_test, average="macro", zero_division=0)
                f1_train = f1_score(y_train, y_pred_train, average="macro", zero_division=0)
                n_rules, n_atoms = count_rules_atoms(clf)

                print(f"  {variant_name:<25} {n_rules:>6} {n_atoms:>6} {f1_test:>8.4f} {f1_train:>9.4f} {elapsed:>9.1f}")
            except Exception as e:
                print(f"  {variant_name:<25} ERROR: {e}")

    print(f"\n{'=' * 100}")
    print("Fertig.")


if __name__ == "__main__":
    main()

