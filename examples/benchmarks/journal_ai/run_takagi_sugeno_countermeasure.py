"""Takagi-Sugeno collapse countermeasure experiment.

The multi-objective evolutionary search minimizes a Pareto trade-off between
RMSE and rule-set size. With Takagi-Sugeno (linear) consequents, a near-global
single-region model already attains linear-model accuracy, so the size
objective drives the premise to its minimal extent (mean complexity ~1 atom).
The resulting model effectively collapses to a single global linear consequent.

This experiment tests whether *enforcing* a genuine, non-trivial partition
recovers accuracy above the collapsed / global-linear model. We construct a
Takagi-Sugeno scored rule set with an enforced K-region partition: a shallow
axis-aligned decision tree of depth d defines K = 2^d crisp leaf regions (each
leaf path is a conjunction of threshold atoms, i.e., a valid scored rule), and a
local ridge regression is fitted inside every region as the TS consequent. By
sweeping the enforced depth d in {0, 1, 2, 3} we obtain a controllable
region-count floor and can read off whether extra regions help.

d = 0 is the degenerate single-region case = global ridge (the collapse target).

Protocol: 5-fold cross-validation, seed 42, matching the Takagi-Sugeno benchmark
it extends (plain KFold, no feature standardization); the sklearn Diabetes
features are already normalized, and the Friedman generators are well
conditioned, so the reported global-ridge scores reproduce the collapse targets
cited in the manuscript exactly.
"""

from __future__ import annotations

import sys
import unittest.mock
from pathlib import Path

# Mock catgen if not installed in current environment
sys.modules["catgen"] = unittest.mock.MagicMock()
sys.modules["catgen.datasets"] = unittest.mock.MagicMock()

_SCRIPT_DIR = Path(__file__).resolve().parent
_PKG_SRC = _SCRIPT_DIR.parents[2] / "src"
if _PKG_SRC.exists() and str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import load_diabetes, make_friedman1, make_friedman2
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

from scoredrulesets.estimators.rulegp_regressor import RuleGPRegressor

RANDOM_STATE = 42
N_SPLITS = 5


class PartitionedTSRuleSet(BaseEstimator, RegressorMixin):
    """Takagi-Sugeno scored rule set with an enforced K-region partition.

    A depth-``d`` axis-aligned decision tree partitions the input space into
    K = 2**d crisp leaf regions (each leaf path is a conjunction of threshold
    atoms). A local ridge regression is fitted inside every region as the TS
    linear consequent. ``d = 0`` degenerates to a single global ridge model.
    """

    def __init__(self, depth: int = 2, alpha: float = 1.0, random_state: int | None = None):
        self.depth = depth
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.depth <= 0:
            self._leaf_ids_ = None
            self._global_ = Ridge(alpha=self.alpha).fit(X, y)
            self._models_ = {}
            self.n_regions_ = 1
            self.n_atoms_ = 0
            return self
        self._tree_ = DecisionTreeRegressor(
            max_depth=self.depth, random_state=self.random_state
        ).fit(X, y)
        leaf_ids = self._tree_.apply(X)
        self._global_ = Ridge(alpha=self.alpha).fit(X, y)  # fallback
        self._models_ = {}
        for leaf in np.unique(leaf_ids):
            mask = leaf_ids == leaf
            if mask.sum() > 1:
                self._models_[leaf] = Ridge(alpha=self.alpha).fit(X[mask], y[mask])
            else:
                self._models_[leaf] = self._global_
        self.n_regions_ = int(np.unique(leaf_ids).size)
        # Atom count = number of internal split nodes actually used along paths
        # (a proxy for total premise complexity of the induced rule set).
        self.n_atoms_ = int((self._tree_.tree_.children_left != -1).sum())
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.depth <= 0:
            return self._global_.predict(X)
        leaf_ids = self._tree_.apply(X)
        preds = np.empty(X.shape[0], dtype=float)
        for leaf in np.unique(leaf_ids):
            mask = leaf_ids == leaf
            model = self._models_.get(leaf, self._global_)
            preds[mask] = model.predict(X[mask])
        return preds


def get_datasets():
    datasets = []
    X_f1, y_f1 = make_friedman1(n_samples=400, n_features=10, noise=1.0, random_state=42)
    datasets.append((X_f1, y_f1, "Friedman #1"))
    X_f2, y_f2 = make_friedman2(n_samples=400, noise=1.0, random_state=42)
    datasets.append((X_f2, y_f2, "Friedman #2"))
    X_diab, y_diab = load_diabetes(return_X_y=True)
    datasets.append((X_diab, y_diab, "Diabetes"))
    return datasets


def _rulegp_atoms(model) -> float:
    if hasattr(model, "to_ruleset"):
        rs = model.to_ruleset()
        return float(sum(len(r.atoms) for r in rs.rules))
    return float("nan")


def evaluate_cv(model_fn, X, y, atom_fn):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    r2s, atoms = [], []
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        model = model_fn()
        model.fit(X_tr, y_tr)
        r2s.append(r2_score(y_te, model.predict(X_te)))
        atoms.append(atom_fn(model))
    return float(np.mean(r2s)), float(np.std(r2s)), float(np.nanmean(atoms))


def run():
    records = []
    for X, y, name in get_datasets():
        print(f"\n=== {name} (N={X.shape[0]}, D={X.shape[1]}) ===")

        # Collapse target: global linear ridge (= enforced depth 0).
        configs = [
            ("Global Ridge (collapse target)",
             lambda: PartitionedTSRuleSet(depth=0), lambda m: m.n_atoms_),
            ("Evolutionary TS (unconstrained)",
             lambda: RuleGPRegressor(prediction_type="linear", max_generations=40,
                                     population_size=60, random_state=42),
             _rulegp_atoms),
            ("Enforced TS (2 regions, d=1)",
             lambda: PartitionedTSRuleSet(depth=1, random_state=42), lambda m: m.n_atoms_),
            ("Enforced TS (4 regions, d=2)",
             lambda: PartitionedTSRuleSet(depth=2, random_state=42), lambda m: m.n_atoms_),
            ("Enforced TS (8 regions, d=3)",
             lambda: PartitionedTSRuleSet(depth=3, random_state=42), lambda m: m.n_atoms_),
        ]
        for label, fn, atom_fn in configs:
            r2_m, r2_s, atoms_m = evaluate_cv(fn, X, y, atom_fn)
            print(f"{label:<34}: R2 = {r2_m:6.3f} +/- {r2_s:5.3f} | atoms ~ {atoms_m:4.1f}")
            records.append({
                "Dataset": name,
                "Model": label,
                "R2_Mean": round(r2_m, 4),
                "R2_Std": round(r2_s, 4),
                "Atoms": round(atoms_m, 2),
            })

    df = pd.DataFrame(records)
    out_dir = Path(__file__).resolve().parent / "results_regression"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "takagi_sugeno_countermeasure_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    run()
