"""Subgroup Discovery & Exceptional Model Mining Benchmark.

Demonstrates evolutionary subgroup discovery on tabular regression datasets
using RuleEvoSubgroup (NSGA-II) and RuleGPSubgroup (GP).
"""

from __future__ import annotations

import sys
import unittest.mock
from pathlib import Path

# Mock catgen if not installed in current environment
sys.modules['catgen'] = unittest.mock.MagicMock()
sys.modules['catgen.datasets'] = unittest.mock.MagicMock()

# Ensure local scoredrulesets package is available
_SCRIPT_DIR = Path(__file__).resolve().parent
_PKG_SRC = _SCRIPT_DIR.parents[2] / "src"
if _PKG_SRC.exists() and str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_california_housing

from scoredrulesets.estimators import RuleNSGA2Subgroup, RuleGPSubgroup


def format_subgroups(ruleset, n_samples):
    rows = []
    for r in ruleset.rules:
        cond_str = " AND ".join([f"{a.feature} {a.op} {a.value:.4f}" if isinstance(a.value, float) else f"{a.feature} {a.op} {a.value}" for a in r.atoms]) if r.atoms else "DEFAULT"
        cov = r.metadata.get("coverage", 0.0)
        q = r.metadata.get("quality", 0.0)
        mean = r.metadata.get("local_mean", 0.0)
        gmean = r.metadata.get("global_mean", 0.0)
        dev = mean - gmean
        count = int(round(cov * n_samples))
        rows.append({
            "Subgroup Rule": cond_str,
            "Coverage (%)": cov * 100.0,
            "Sample Count": count,
            "Local Mean": mean,
            "Deviation": dev,
            "Quality Score": q,
            "Atom Count": len(r.atoms)
        })
    return pd.DataFrame(rows)


def run_benchmark():
    datasets = []
    # 1. Diabetes
    diab = load_diabetes()
    datasets.append((diab.data, diab.target, "Diabetes", list(diab.feature_names)))

    # 2. California Housing (Subsample for fast demonstration)
    cal = fetch_california_housing()
    np.random.seed(42)
    idx = np.random.choice(len(cal.data), size=500, replace=False)
    datasets.append((cal.data[idx], cal.target[idx], "California Housing (N=500)", list(cal.feature_names)))

    out_dir = Path(__file__).resolve().parent / "results_regression"
    out_dir.mkdir(exist_ok=True)

    for X, y, ds_name, feat_names in datasets:
        print(f"\n==================================================================")
        print(f"Dataset: {ds_name} (N={X.shape[0]}, D={X.shape[1]}, Global Mean={np.mean(y):.3f})")
        print(f"==================================================================")

        # 1. RuleNSGA2Subgroup (NSGA-II)
        print("\n--- RuleNSGA2Subgroup (NSGA-II) ---")
        evo_sg = RuleNSGA2Subgroup(population_size=50, max_generations=40, random_state=42)
        evo_sg.fit(X, y)
        evo_sg.feature_names_in_ = np.array(feat_names)
        rs_evo = evo_sg._to_schema_ruleset(evo_sg.ruleset_internal_, X)
        df_evo = format_subgroups(rs_evo, X.shape[0])
        print(df_evo.to_string(index=False))

        # 2. RuleGPSubgroup (GP)
        print("\n--- RuleGPSubgroup (RuleGP) ---")
        gp_sg = RuleGPSubgroup(population_size=50, max_generations=40, random_state=42)
        gp_sg.fit(X, y)
        gp_sg.feature_names_in_ = np.array(feat_names)
        rs_gp = gp_sg._to_schema_ruleset(gp_sg.ruleset_internal_, X)
        df_gp = format_subgroups(rs_gp, X.shape[0])
        print(df_gp.to_string(index=False))

        # Save CSV
        clean_ds = ds_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        df_evo.to_csv(out_dir / f"subgroups_nsga2_{clean_ds}.csv", index=False)
        df_gp.to_csv(out_dir / f"subgroups_rulegp_{clean_ds}.csv", index=False)

    print(f"\nSubgroup Discovery results saved to {out_dir}/")


if __name__ == "__main__":
    run_benchmark()
