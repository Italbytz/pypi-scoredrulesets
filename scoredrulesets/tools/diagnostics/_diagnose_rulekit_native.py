#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Detailed diagnostics for rulekit_native on Iris."""
import os
os.environ['JAVA_HOME'] = os.popen('/usr/libexec/java_home').read().strip()

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from scoredrulesets.formatting import format_ruleset_table
from scoredrulesets.runtime import predict as predict_from_ruleset
from scoredrulesets.estimators.rulekit_native import RuleKitNativeClassifier

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y,
)

print("=" * 70)
print("PARAMETER SWEEP: rulekit_native on Iris")
print("=" * 70)

configs = [
    ("default", dict(max_rules=20, max_conditions=5, enable_pruning=True, random_state=42)),
    ("no_prune", dict(max_rules=20, max_conditions=5, enable_pruning=False, random_state=42)),
    ("min_leaf=2", dict(max_rules=20, max_conditions=5, enable_pruning=True,
                        min_samples_leaf=2, min_rule_covered=2, random_state=42)),
    ("min_leaf=1", dict(max_rules=20, max_conditions=5, enable_pruning=True,
                        min_samples_leaf=1, min_rule_covered=1, random_state=42)),
    ("no_intervals", dict(max_rules=20, max_conditions=5, enable_pruning=True,
                          enable_intervals=False, random_state=42)),
    ("split_prune", dict(max_rules=20, max_conditions=5, enable_pruning=True,
                         pruning_mode="split", pruning_fraction=0.33, random_state=42)),
    ("min1+no_prune", dict(max_rules=20, max_conditions=8, enable_pruning=False,
                           min_samples_leaf=1, min_rule_covered=1, random_state=42)),
    ("min2+maxcond8", dict(max_rules=20, max_conditions=8, enable_pruning=True,
                           min_samples_leaf=2, min_rule_covered=2, random_state=42)),
]

# Java RuleKit reference
from rulekit.classification import RuleClassifier
from scoredrulesets.estimators.ruleset_transform import rulekit_to_scored_ruleset
rk = RuleClassifier()
rk.fit(X_train, y_train)
rk_pred = rk.predict(X_test)
rk_f1 = f1_score(y_test, rk_pred, average="weighted")
print(f"\n  {'Java RuleKit':<25} F1={rk_f1:.4f}  rules=6")

for name, kwargs in configs:
    try:
        clf = RuleKitNativeClassifier(**kwargs)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        f1 = f1_score(y_test, pred, average="weighted")
        rs = clf.to_ruleset()
        n_rules = len(rs.rules) - 1  # exclude default
        n_atoms = sum(len(r.atoms) for r in rs.rules)
        print(f"  {name:<25} F1={f1:.4f}  rules={n_rules}  atoms={n_atoms}")
        if f1 >= 0.92:
            print(f"    -> GOOD!")
            print(format_ruleset_table(rs))
    except Exception as e:
        print(f"  {name:<25} ERROR: {e}")

# Detailed look at best config
print("\n" + "=" * 70)
print("DETAILED: min_leaf=2 config")
print("=" * 70)
clf = RuleKitNativeClassifier(
    max_rules=20, max_conditions=5, enable_pruning=True,
    min_samples_leaf=2, min_rule_covered=2, random_state=42,
)
clf.fit(X_train, y_train)
rs = clf.to_ruleset()
print(format_ruleset_table(rs))
pred = clf.predict(X_test)
f1 = f1_score(y_test, pred, average="weighted")
print(f"F1={f1:.4f}")
print("\nPer-class accuracy:")
for c in sorted(set(y_test)):
    m = y_test == c
    print(f"  class {c}: {np.mean(pred[m] == y_test[m]):.3f}")

