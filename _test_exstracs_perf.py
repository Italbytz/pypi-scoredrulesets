"""Temporary test to measure ExSTraCS training + pruning performance on breast_cancer."""
import time
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from skExSTraCS import ExSTraCS
from scoredrulesets.estimators.ruleset_transform import exstracs_to_scored_ruleset
from scoredrulesets.estimators.exstracs_shrinking import exstracs_prune_conservative

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Features: {Xtr.shape[1]}, Train: {Xtr.shape[0]}")

print("Training (5000 iters, N=500)...")
t0 = time.time()
est = ExSTraCS(learning_iterations=5000, N=500, random_state=0)
est.fit(Xtr, ytr)
print(f"  done in {time.time()-t0:.1f}s")

print("Transforming...")
t0 = time.time()
rs = exstracs_to_scored_ruleset(estimator=est, class_labels=[0, 1],
                                feature_names=[f"f{i}" for i in range(30)])
n_rules = len(rs.rules)
n_atoms = sum(len(r.atoms) for r in rs.rules)
n_multi = sum(1 for r in rs.rules if len(r.atoms) > 1)
print(f"  done in {time.time()-t0:.1f}s, rules={n_rules}, atoms={n_atoms}, multi_atom_rules={n_multi}")

print("Conservative pruning (with ref data)...")
t0 = time.time()
rs2 = exstracs_prune_conservative(rs, X_ref=Xtr, y_ref=ytr)
elapsed = time.time() - t0
print(f"  done in {elapsed:.1f}s, rules={len(rs2.rules)}, atoms={sum(len(r.atoms) for r in rs2.rules)}")

print("\nNow test with DEFAULT ExSTraCS params (100K iters, N=1000)...")
print("Training (100000 iters, N=1000)...")
t0 = time.time()
est2 = ExSTraCS(learning_iterations=100000, N=1000, random_state=0)
est2.fit(Xtr, ytr)
elapsed_train = time.time() - t0
n_rules2 = len(est2.population.popSet)
print(f"  done in {elapsed_train:.1f}s, rules={n_rules2}")

print("Transforming (default)...")
t0 = time.time()
rs3 = exstracs_to_scored_ruleset(estimator=est2, class_labels=[0, 1],
                                 feature_names=[f"f{i}" for i in range(30)])
n_atoms3 = sum(len(r.atoms) for r in rs3.rules)
print(f"  done in {time.time()-t0:.1f}s, rules={len(rs3.rules)}, atoms={n_atoms3}")

print("Conservative pruning (default, with ref data)...")
t0 = time.time()
rs4 = exstracs_prune_conservative(rs3, X_ref=Xtr, y_ref=ytr)
elapsed_prune = time.time() - t0
print(f"  done in {elapsed_prune:.1f}s, rules={len(rs4.rules)}, atoms={sum(len(r.atoms) for r in rs4.rules)}")

print("\nDone!")

