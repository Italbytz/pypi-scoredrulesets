"""Multi-Target Regression on a real UCI dataset with baselines.

Extends the multi-target evaluation beyond synthetic/Linnerud data to the UCI
Energy Efficiency dataset (768 buildings, 8 features, 2 correlated targets:
heating load Y1 and cooling load Y2). The vector-scored ruleGP regressor is
compared against a global-linear Ridge baseline and a depth-4 CART regressor,
all under identical 5-fold cross-validation (seed 42, per-fold standardization).
"""

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PKG_SRC = _SCRIPT_DIR.parents[2] / "src"
if _PKG_SRC.exists() and str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from scoredrulesets.estimators.rulegp_regressor import RuleGPRegressor

RANDOM_STATE = 42
N_SPLITS = 5


def load_energy_efficiency():
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=242)  # Energy efficiency
    X = ds.data.features.to_numpy(dtype=float)
    y = ds.data.targets.to_numpy(dtype=float)  # (n, 2): Y1 heating, Y2 cooling
    return X, y


def n_atoms(model, X, y):
    if hasattr(model, "to_ruleset"):
        return sum(len(r.atoms) for r in model.to_ruleset().rules)
    if hasattr(model, "tree_"):  # native multi-output CART
        return int(model.tree_.node_count)
    if hasattr(model, "estimators_") and all(hasattr(e, "tree_") for e in model.estimators_):
        return int(sum(e.tree_.node_count for e in model.estimators_))
    if hasattr(model, "estimators_"):  # per-target linear model: count coefficients
        return int(sum(int(np.count_nonzero(e.coef_)) + 1 for e in model.estimators_))
    return X.shape[1] * y.shape[1]


def evaluate(model_fn, X, y):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    pooled, per_target, atoms = [], [], []
    for tr, te in kf.split(X):
        scaler = StandardScaler().fit(X[tr])
        X_tr, X_te = scaler.transform(X[tr]), scaler.transform(X[te])
        model = model_fn()
        model.fit(X_tr, y[tr])
        pred = model.predict(X_te)
        raw = r2_score(y[te], pred, multioutput="raw_values")
        per_target.append(raw)
        pooled.append(float(np.mean(raw)))
        atoms.append(n_atoms(model, X, y))
    return (float(np.mean(pooled)), float(np.std(pooled)),
            np.mean(np.array(per_target), axis=0), float(np.mean(atoms)))


def run():
    X, y = load_energy_efficiency()
    print(f"Energy Efficiency: N={X.shape[0]}, D={X.shape[1]}, targets={y.shape[1]}")

    models = {
        "Ridge (per-target)": lambda: MultiOutputRegressor(Ridge()),
        "CART (d=4)": lambda: DecisionTreeRegressor(max_depth=4, random_state=RANDOM_STATE),
        "ruleGP (vector score)": lambda: RuleGPRegressor(
            prediction_type="constant", max_generations=40,
            population_size=60, random_state=RANDOM_STATE),
    }

    records = []
    for name, fn in models.items():
        pooled, std, per_t, atoms = evaluate(fn, X, y)
        pts = ", ".join(f"{v:.3f}" for v in per_t)
        print(f"  {name:<22}: pooled R2={pooled:.3f}±{std:.3f} | per-target=[{pts}] | atoms={atoms:.1f}")
        records.append({"model": name, "pooled_r2": pooled, "pooled_r2_std": std,
                        "per_target_r2": pts, "mean_atoms": atoms})

    out = _SCRIPT_DIR / "results_regression" / "multioutput_energy_efficiency.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out, index=False)
    print(f"\nSaved CSV to {out}")


if __name__ == "__main__":
    run()
