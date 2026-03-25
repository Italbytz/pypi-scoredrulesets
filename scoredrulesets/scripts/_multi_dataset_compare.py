#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Multi-dataset comparison: Java RuleKit vs rulekit_native."""
import os
os.environ['JAVA_HOME'] = os.popen('/usr/libexec/java_home').read().strip()

import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from scoredrulesets.estimators.rulekit_native import RuleKitNativeClassifier
from rulekit.classification import RuleClassifier

datasets = [
    ("Iris", load_iris),
    ("Wine", load_wine),
    ("Breast Cancer", load_breast_cancer),
]

header = "{:<16} {:>12} {:>13} {:>8}".format(
    "Dataset", "Java RuleKit", "Native (new)", "Gap",
)
print(header)
print("-" * 55)

for name, loader in datasets:
    X, y = loader(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y,
    )

    # Java RuleKit
    rk = RuleClassifier()
    rk.fit(X_tr, y_tr)
    rk_f1 = f1_score(y_te, rk.predict(X_te), average="weighted")

    # Native
    rn = RuleKitNativeClassifier(
        max_rules=20, max_conditions=5, enable_pruning=True, random_state=42,
    )
    rn.fit(X_tr, y_tr)
    rn_f1 = f1_score(y_te, rn.predict(X_te), average="weighted")
    rs = rn.to_ruleset()
    n_rules = len(rs.rules) - 1

    gap = rn_f1 - rk_f1
    row = "{:<16} {:>12.4f} {:>10.4f} ({}) {:>+8.4f}".format(
        name, rk_f1, rn_f1, n_rules, gap,
    )
    print(row)

