#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare Java RuleKit vs rulekit_native on Iris to diagnose differences."""
import os
os.environ['JAVA_HOME'] = os.popen('/usr/libexec/java_home').read().strip()

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from scoredrulesets.formatting import format_ruleset_table
from scoredrulesets.runtime import predict as predict_from_ruleset

# ---- Load data ----
X, y = load_iris(return_X_y=True)
feature_names = [f"f{i}" for i in range(X.shape[1])]
class_labels = sorted(list(set(y)))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y,
)

print("=" * 70)
print("COMPARISON: Java RuleKit vs rulekit_native on Iris")
print("=" * 70)
print(f"Train: {len(X_train)} | Test: {len(X_test)} | Classes: {class_labels}")
print()

# ---- Java RuleKit ----
print("-" * 70)
print("1) JAVA RULEKIT (via rulekit package)")
print("-" * 70)
from rulekit.classification import RuleClassifier
from scoredrulesets.estimators.ruleset_transform import rulekit_to_scored_ruleset

rk = RuleClassifier()
rk.fit(X_train, y_train)
rk_native_pred = rk.predict(X_test)
rk_native_f1 = f1_score(y_test, rk_native_pred, average="weighted")
rk_native_acc = accuracy_score(y_test, rk_native_pred)
print(f"Java RuleKit native:  F1={rk_native_f1:.4f}  Acc={rk_native_acc:.4f}")

# Show Java RuleKit rules
try:
    rule_source = None
    if hasattr(rk, "model") and hasattr(rk.model, "rules"):
        rule_source = rk.model.rules
    elif hasattr(rk, "rules_"):
        rule_source = rk.rules_
    if rule_source:
        print(f"  Java RuleKit produced {len(list(rule_source))} rules")
        for i, rule in enumerate(rule_source):
            print(f"  Rule {i}: {rule}")
except Exception as e:
    print(f"  (could not inspect Java rules: {e})")

# Transform
rs_rk = rulekit_to_scored_ruleset(rk, class_labels, feature_names, y_train=y_train)
print(f"\nTransformed ScoredRuleSet ({len(rs_rk.rules)} rules):")
print(format_ruleset_table(rs_rk))

rs_rk_pred = predict_from_ruleset(rs_rk, np.asarray(X_test))
rs_rk_f1 = f1_score(y_test, rs_rk_pred, average="weighted")
rs_rk_acc = accuracy_score(y_test, rs_rk_pred)
print(f"Java RuleKit transformed: F1={rs_rk_f1:.4f}  Acc={rs_rk_acc:.4f}")
print(f"F1 loss from transformation: {rk_native_f1 - rs_rk_f1:.4f}")

# Per-class
print("\nPer-class accuracy:")
for c in sorted(set(y_test)):
    m = y_test == c
    acc_native = np.mean(rk_native_pred[m] == y_test[m])
    acc_trans = np.mean(rs_rk_pred[m] == y_test[m])
    print(f"  class {c}: native={acc_native:.3f}  transformed={acc_trans:.3f}")

# Mismatches
mismatches = rk_native_pred != rs_rk_pred
if mismatches.any():
    print(f"\nMismatches (native vs transformed): {mismatches.sum()}/{len(y_test)}")
    idx = np.where(mismatches)[0][:10]
    for i in idx:
        print(f"  sample {i}: true={y_test[i]} native={rk_native_pred[i]} transformed={rs_rk_pred[i]}")
else:
    print("\nNo mismatches between native and transformed predictions!")

# ---- rulekit_native ----
print()
print("-" * 70)
print("2) RULEKIT_NATIVE (pure Python)")
print("-" * 70)
from scoredrulesets.estimators.rulekit_native import RuleKitNativeClassifier

rn = RuleKitNativeClassifier(
    max_rules=20, max_conditions=5, enable_pruning=True, random_state=42,
)
rn.fit(X_train, y_train)
rn_pred = rn.predict(X_test)
rn_f1 = f1_score(y_test, rn_pred, average="weighted")
rn_acc = accuracy_score(y_test, rn_pred)
print(f"rulekit_native: F1={rn_f1:.4f}  Acc={rn_acc:.4f}")

rn_rs = rn.to_ruleset()
print(f"\nScoredRuleSet ({len(rn_rs.rules)} rules):")
print(format_ruleset_table(rn_rs))

# Per-class
print("Per-class accuracy:")
for c in sorted(set(y_test)):
    m = y_test == c
    acc = np.mean(rn_pred[m] == y_test[m])
    print(f"  class {c}: {acc:.3f}")

# ---- Summary ----
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Java RuleKit native:      F1={rk_native_f1:.4f}  Acc={rk_native_acc:.4f}")
print(f"Java RuleKit transformed: F1={rs_rk_f1:.4f}  Acc={rs_rk_acc:.4f}  (loss={rk_native_f1 - rs_rk_f1:.4f})")
print(f"rulekit_native:           F1={rn_f1:.4f}  Acc={rn_acc:.4f}")
print(f"Java rules: {len(rs_rk.rules)}  |  Native rules: {len(rn_rs.rules)}")

