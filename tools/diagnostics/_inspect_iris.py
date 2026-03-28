#!/usr/bin/env python
"""Quick rule inspection for rulekit_native on Iris."""
import numpy as np
from sklearn.datasets import load_iris
from scoredrulesets.estimators.rulekit_native import RuleKitNativeClassifier
from scoredrulesets.formatting import format_ruleset_table

X, y = load_iris(return_X_y=True)

for label, kwargs in [
    ("default (pruning)", dict(max_rules=20, max_conditions=5, enable_pruning=True, random_state=42)),
    ("no pruning", dict(max_rules=20, max_conditions=5, enable_pruning=False, random_state=42)),
]:
    clf = RuleKitNativeClassifier(**kwargs)
    clf.fit(X, y)
    rs = clf.to_ruleset()
    preds = clf.predict(X)
    acc = np.mean(preds == y)
    print(f"\n=== {label} === (accuracy={acc:.4f}, {len(rs.rules)} rules)")
    print(format_ruleset_table(rs))
    print("Per-class accuracy:")
    for c in sorted(set(y)):
        m = y == c
        print(f"  class {c}: {np.mean(preds[m] == y[m]):.4f} ({m.sum()} examples)")


