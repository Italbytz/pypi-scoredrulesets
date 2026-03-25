"""Analyse der Standard-Benchmark-Ergebnisse."""
import csv
from collections import defaultdict

rows = []
with open("benchmarks/standard/benchmark_results_aggregated.csv") as f:
    reader = csv.DictReader(f)
    for r in reader:
        if r["status"] != "ok":
            continue
        rows.append({
            "ds": r["dataset"],
            "est": r["estimator"],
            "f1": float(r["f1_macro_mean"]),
            "atoms": float(r["n_atoms_mean"]),
            "rules": float(r["n_rules_mean"]),
            "fit_s": float(r["fit_seconds_mean"]),
        })

datasets = sorted(set(r["ds"] for r in rows))
estimators = sorted(set(r["est"] for r in rows))

# --- Mean F1 and atoms across all datasets ---
print("\n=== MITTLERE F1 UND MODELLGROESSE UEBER ALLE DATASETS ===")
print(f"{'Estimator':<35} {'F1_mean':>8} {'Atoms':>7} {'Rules':>7} {'Fit_s':>8}")
print("-" * 70)
for est in sorted(estimators, key=lambda e: -sum(r["f1"] for r in rows if r["est"]==e)/max(sum(1 for r in rows if r["est"]==e),1)):
    est_rows = [r for r in rows if r["est"] == est]
    n = len(est_rows)
    f1_avg = sum(r["f1"] for r in est_rows) / n
    atoms_avg = sum(r["atoms"] for r in est_rows) / n
    rules_avg = sum(r["rules"] for r in est_rows) / n
    fit_avg = sum(r["fit_s"] for r in est_rows) / n
    print(f"{est:<35} {f1_avg:>8.4f} {atoms_avg:>7.1f} {rules_avg:>7.1f} {fit_avg:>8.2f}")

# --- Varianten-Vergleich innerhalb von Familien ---
families = {
    "pittsburgh": ["wrapper_pittsburgh", "wrapper_pittsburgh_fast", "wrapper_pittsburgh_diverse", "wrapper_pittsburgh_strong"],
    "rulekit_native": ["rulekit_native", "wrapper_rulekit_native", "wrapper_rulekit_native_fast", "wrapper_rulekit_native_strong"],
    "gp": ["gp", "gp_fast", "gp_diverse", "gp_contrib", "gp_residual"],
    "nln": ["nln_native", "wrapper_nln", "wrapper_nln_fast", "wrapper_nln_strong"],
    "logicgp": ["wrapper_logicgp", "wrapper_logicgp_flcw", "wrapper_logicgp_rlcw_macro"],
    "exstracs": ["wrapper_exstracs", "wrapper_exstracs_compact", "wrapper_exstracs_pruned"],
    "rulefit": ["wrapper_rulefit", "wrapper_rulefit_compact"],
    "cart": ["wrapper_cart", "wrapper_cart_pruned"],
    "hs": ["wrapper_hs", "wrapper_hs_pruned"],
}

print("\n=== VARIANTEN-VERGLEICH INNERHALB FAMILIEN ===")
for fam_name, members in families.items():
    print(f"\n--- {fam_name} ---")
    print(f"  {'Variante':<35} {'F1':>7} {'Atoms':>7} {'Fit_s':>8}")
    for est in members:
        est_rows = [r for r in rows if r["est"] == est]
        if not est_rows:
            continue
        n = len(est_rows)
        f1 = sum(r["f1"] for r in est_rows) / n
        atoms = sum(r["atoms"] for r in est_rows) / n
        fit = sum(r["fit_s"] for r in est_rows) / n
        print(f"  {est:<35} {f1:>7.4f} {atoms:>7.1f} {fit:>8.2f}")

# --- Top-5 pro Dataset ---
print("\n=== TOP-5 PRO DATASET (F1) ===")
for ds in datasets:
    ds_rows = sorted([r for r in rows if r["ds"] == ds], key=lambda r: -r["f1"])
    print(f"\n{ds}:")
    for i, r in enumerate(ds_rows[:5]):
        print(f"  {i+1}. {r['est']:<35} F1={r['f1']:.4f} atoms={r['atoms']:.0f} fit={r['fit_s']:.1f}s")
    print(f"  ... worst 3:")
    for r in ds_rows[-3:]:
        print(f"     {r['est']:<35} F1={r['f1']:.4f} atoms={r['atoms']:.0f} fit={r['fit_s']:.1f}s")

# --- Identische Varianten ---
print("\n=== IDENTISCHE/REDUNDANTE VARIANTEN (gleiche F1 auf allen Datasets) ===")
from itertools import combinations
for e1, e2 in combinations(estimators, 2):
    same = True
    count = 0
    for ds in datasets:
        r1 = next((r for r in rows if r["ds"]==ds and r["est"]==e1), None)
        r2 = next((r for r in rows if r["ds"]==ds and r["est"]==e2), None)
        if r1 and r2:
            count += 1
            if abs(r1["f1"] - r2["f1"]) > 0.001:
                same = False
                break
    if same and count >= 5:
        print(f"  {e1} == {e2}  ({count} datasets)")

# --- NLN duplicates check ---
print("\n=== NLN vs NLN_NATIVE Check ===")
for ds in datasets:
    r1 = next((r for r in rows if r["ds"]==ds and r["est"]=="nln_native"), None)
    r2 = next((r for r in rows if r["ds"]==ds and r["est"]=="wrapper_nln"), None)
    if r1 and r2:
        diff = abs(r1["f1"] - r2["f1"])
        print(f"  {ds}: nln_native={r1['f1']:.4f} wrapper_nln={r2['f1']:.4f} diff={diff:.4f}")

# --- Dataset difficulty ranking ---
print("\n=== DATASET-SCHWIERIGKEIT (mittlerer max-F1 aller Schaetzer) ===")
for ds in sorted(datasets, key=lambda d: max(r["f1"] for r in rows if r["ds"]==d)):
    best = max(r["f1"] for r in rows if r["ds"] == ds)
    avg = sum(r["f1"] for r in rows if r["ds"]==ds) / sum(1 for r in rows if r["ds"]==ds)
    print(f"  {ds:<30} best={best:.4f}  avg={avg:.4f}")

