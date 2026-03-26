#!/usr/bin/env python
"""Diagnose why growing fails for class 2 on Iris."""
import numpy as np
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

# Simulate after removing class-0 (rule f2<=2.45) and class-1 positives
mask0_covered = (y == 0) & (X[:, 2] <= 2.45)
mask1_covered = (y == 1) & (X[:, 3] <= 1.65) & (X[:, 2] <= 4.95)
uncovered = ~mask0_covered & ~mask1_covered

X_unc = X[uncovered]
y_unc = y[uncovered]
print(f"Uncovered: n={len(y_unc)}, classes={np.bincount(y_unc, minlength=3)}")

P2, N2 = 50.0, 100.0  # full-dataset counts for class 2
p_all = float(np.sum(y_unc == 2))
n_all = float(np.sum(y_unc != 2))
q_empty = (p_all - n_all) / (P2 + N2)
print(f"Class 2 empty-rule quality: p={p_all}, n={n_all}, q={q_empty:.4f}")

print("\nClass 2 -- f3 threshold scan:")
for thr in [1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85]:
    mask = X_unc[:, 3] > thr
    p = float(np.sum(y_unc[mask] == 2))
    n = float(np.sum(y_unc[mask] != 2))
    q = (p - n) / 150.0
    print(f"  f3 > {thr:.2f}: p={p:.0f} n={n:.0f} q={q:.4f} delta={q-q_empty:+.4f}")

print("\nRemaining class-1 in uncovered:")
c1 = y_unc == 1
print(f"  count={c1.sum()}")
print(f"  f3 = {X_unc[c1, 3]}")
print(f"  f2 = {X_unc[c1, 2]}")

print("\n--- This confirms: empty-rule quality is HIGH, no single condition improves it.")
print("--- FIX: Initialize growing quality to -inf (like Java RuleKit).")

