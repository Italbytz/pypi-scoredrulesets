"""Analyze benchmark checkpoint data."""
import json
import numpy as np
from collections import defaultdict

data = []
with open("benchmarks/checkpoint.jsonl Kopie", "r") as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))

print(f"Total records: {len(data)}")

# Filter only OK records
ok = [r for r in data if r["status"] == "ok"]
err = [r for r in data if r["status"] != "ok"]
print(f"OK: {len(ok)}, Errors/Skipped: {len(err)}")

datasets = sorted(set(r["dataset"] for r in ok))
estimators = sorted(set(r["estimator"] for r in ok))
print(f"\nDatasets ({len(datasets)}): {datasets}")
print(f"\nEstimators ({len(estimators)}):")
for e in estimators:
    print(f"  {e}")

# Error summary
print(f"\nError/Skip details:")
for r in err:
    print(f"  {r['dataset']:25s} {r['estimator']:35s} status={r['status']} err={str(r.get('error',''))[:80]}")

# Aggregate: mean F1 and model size per estimator per dataset
agg = defaultdict(lambda: defaultdict(list))
for r in ok:
    key = r["estimator"]
    ds = r["dataset"]
    agg[key][ds].append({
        "f1": r["f1_macro"],
        "rules": r["n_rules"],
        "atoms": r["n_atoms"],
        "fit_s": r["fit_seconds"],
    })

# Print aggregated results per dataset
print("\n" + "="*120)
print("AGGREGATED RESULTS (mean over repeats)")
print("="*120)

for ds in datasets:
    print(f"\n--- {ds} ---")
    rows = []
    for est in estimators:
        if ds in agg[est]:
            vals = agg[est][ds]
            f1s = [v["f1"] for v in vals]
            rules = [v["rules"] for v in vals]
            atoms = [v["atoms"] for v in vals]
            fit_s = [v["fit_s"] for v in vals]
            model_size = np.mean([r + a for r, a in zip(rules, atoms)])
            rows.append({
                "est": est,
                "n": len(vals),
                "f1_mean": np.mean(f1s),
                "f1_std": np.std(f1s),
                "rules_mean": np.mean(rules),
                "atoms_mean": np.mean(atoms),
                "model_size": model_size,
                "fit_s": np.mean(fit_s),
            })
    # Sort by F1 desc
    rows.sort(key=lambda x: x["f1_mean"], reverse=True)
    print(f"  {'Estimator':40s} {'n':>3s} {'F1':>7s} {'±std':>6s} {'rules':>6s} {'atoms':>6s} {'size':>6s} {'fit_s':>8s}")
    for r in rows:
        print(f"  {r['est']:40s} {r['n']:3d} {r['f1_mean']:7.4f} {r['f1_std']:6.4f} {r['rules_mean']:6.1f} {r['atoms_mean']:6.1f} {r['model_size']:6.1f} {r['fit_s']:8.2f}")

# Cross-dataset summary: mean F1 across all datasets
print("\n" + "="*120)
print("CROSS-DATASET SUMMARY (mean F1 across all datasets where estimator ran)")
print("="*120)
est_summary = []
for est in estimators:
    ds_means = []
    ds_sizes = []
    for ds in datasets:
        if ds in agg[est]:
            vals = agg[est][ds]
            ds_means.append(np.mean([v["f1"] for v in vals]))
            ds_sizes.append(np.mean([v["rules"] + v["atoms"] for v in vals]))
    if ds_means:
        est_summary.append({
            "est": est,
            "n_datasets": len(ds_means),
            "f1_grand_mean": np.mean(ds_means),
            "f1_min": np.min(ds_means),
            "f1_max": np.max(ds_means),
            "size_mean": np.mean(ds_sizes),
        })

est_summary.sort(key=lambda x: x["f1_grand_mean"], reverse=True)
print(f"  {'Estimator':40s} {'#DS':>4s} {'F1_mean':>8s} {'F1_min':>8s} {'F1_max':>8s} {'size':>8s}")
for r in est_summary:
    print(f"  {r['est']:40s} {r['n_datasets']:4d} {r['f1_grand_mean']:8.4f} {r['f1_min']:8.4f} {r['f1_max']:8.4f} {r['size_mean']:8.1f}")

# Pareto dominance analysis
print("\n" + "="*120)
print("PARETO DOMINANCE ANALYSIS (F1 maximize, model_size minimize)")
print("Per dataset: A dominates B if A.f1 >= B.f1 AND A.size <= B.size (strict in at least one)")
print("="*120)

# Count how often each estimator is dominated per dataset
dom_count = defaultdict(int)  # est -> number of (dataset) where it's dominated
dom_by = defaultdict(lambda: defaultdict(int))  # dominated_est -> dominator_est -> count
total_datasets_per_est = defaultdict(int)

for ds in datasets:
    # Collect (est, f1_mean, model_size_mean) for this dataset
    ds_data = []
    for est in estimators:
        if ds in agg[est]:
            vals = agg[est][ds]
            f1 = np.mean([v["f1"] for v in vals])
            size = np.mean([v["rules"] + v["atoms"] for v in vals])
            ds_data.append((est, f1, size))
            total_datasets_per_est[est] += 1

    # Check Pareto dominance
    for i, (est_a, f1_a, size_a) in enumerate(ds_data):
        is_dominated = False
        for j, (est_b, f1_b, size_b) in enumerate(ds_data):
            if i == j:
                continue
            # B dominates A?
            if f1_b >= f1_a and size_b <= size_a and (f1_b > f1_a or size_b < size_a):
                is_dominated = True
                dom_by[est_a][est_b] += 1
        if is_dominated:
            dom_count[est_a] += 1

print(f"\n  {'Estimator':40s} {'#DS':>4s} {'#dominated':>11s} {'dom_rate':>9s}  dominated by (top-3)")
for est in sorted(estimators, key=lambda e: dom_count.get(e, 0) / max(total_datasets_per_est.get(e, 1), 1), reverse=True):
    n_ds = total_datasets_per_est.get(est, 0)
    n_dom = dom_count.get(est, 0)
    rate = n_dom / max(n_ds, 1)
    top_dominators = sorted(dom_by.get(est, {}).items(), key=lambda x: x[1], reverse=True)[:3]
    top_str = ", ".join(f"{d}({c})" for d, c in top_dominators)
    marker = " *** CANDIDATE FOR REMOVAL" if rate >= 0.8 else (" ** OFTEN DOMINATED" if rate >= 0.5 else "")
    print(f"  {est:40s} {n_ds:4d} {n_dom:11d} {rate:9.2%}{marker}  {top_str}")

