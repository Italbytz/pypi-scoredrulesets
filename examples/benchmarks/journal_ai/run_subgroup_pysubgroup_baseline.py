"""Subgroup Discovery baseline: pysubgroup vs. RuleNSGA2Subgroup on Diabetes.

Provides an established external baseline (pysubgroup's beam search with the
standard numeric quality function) for the native RuleNSGA2Subgroup results
reported on the Diabetes dataset, using the same q = |I|^a * |mean_local -
mean_global| effect-size objective (a = 0.5).
"""

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PKG_SRC = _SCRIPT_DIR.parents[2] / "src"
if _PKG_SRC.exists() and str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

import numpy as np
import pandas as pd
import pysubgroup as ps
from sklearn.datasets import load_diabetes

RANDOM_STATE = 42


def run():
    diab = load_diabetes()
    df = pd.DataFrame(diab.data, columns=diab.feature_names)
    df["target"] = diab.target
    global_mean = float(df["target"].mean())
    n = len(df)
    print(f"Diabetes: N={n}, global target mean={global_mean:.3f}")

    target = ps.NumericTarget("target")
    searchspace = ps.create_selectors(df, ignore=["target"], nbins=5)
    task = ps.SubgroupDiscoveryTask(
        df, target, searchspace,
        result_set_size=5, depth=2,
        qf=ps.StandardQFNumeric(a=0.5),
    )
    result = ps.BeamSearch().execute(task)
    res_df = result.to_dataframe()

    records = []
    print("\n--- pysubgroup (BeamSearch, StandardQFNumeric a=0.5) top subgroups ---")
    for _, row in res_df.iterrows():
        sg = row["subgroup"]
        cover = sg.covers(df)
        cov_pct = 100.0 * cover.sum() / n
        local_mean = float(df.loc[cover, "target"].mean())
        print(f"  {str(sg):<45} cov={cov_pct:5.1f}%  mean={local_mean:7.2f}  "
              f"dev={local_mean - global_mean:+7.2f}  q={row['quality']:.2f}")
        records.append({
            "subgroup": str(sg),
            "coverage_pct": cov_pct,
            "sample_count": int(cover.sum()),
            "local_mean": local_mean,
            "deviation": local_mean - global_mean,
            "quality": float(row["quality"]),
        })

    out = _SCRIPT_DIR / "results_regression" / "subgroups_pysubgroup_diabetes.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out, index=False)
    print(f"\nSaved CSV to {out}")


if __name__ == "__main__":
    run()
