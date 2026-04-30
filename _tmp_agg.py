import json, statistics
from pathlib import Path
from collections import defaultdict

raw = json.loads(Path("benchmarks/paper/benchmark_results.json").read_text())

f1_data = defaultdict(list)
atoms_data = defaultdict(list)
time_data = defaultdict(list)
for r in raw:
    if r.get("status") != "ok":
        continue
    key = (r["dataset"], r["estimator"])
    if r.get("f1_macro") is not None:
        f1_data[key].append(r["f1_macro"])
    if r.get("n_atoms") is not None:
        atoms_data[key].append(r["n_atoms"])
    if r.get("fit_seconds") is not None:
        time_data[key].append(r["fit_seconds"])

ds_names = [
    "uci_breast_cancer_wisconsin_diagnostic", "uci_wine",
    "uci_car_evaluation", "uci_heart_disease",
    "synth_dnf_3x2", "synth_xor_3bit",
    "synth_checkerboard_4x4", "synth_overlap_4rules",
    "synth_monk3", "synth_imbalanced_10pct",
]
ds_short = {
    "uci_breast_cancer_wisconsin_diagnostic": "breast_cancer",
    "uci_wine": "wine",
    "uci_car_evaluation": "car_evaluation",
    "uci_heart_disease": "heart_disease",
    "synth_dnf_3x2": "dnf_3x2",
    "synth_xor_3bit": "xor_3bit",
    "synth_checkerboard_4x4": "checkerboard",
    "synth_overlap_4rules": "overlap_4rules",
    "synth_monk3": "monk3",
    "synth_imbalanced_10pct": "imbalanced",
}
est_order = ["HS", "RuleKit", "ExSTraCS", "ExSTraCS (LRC)", "ruleGP", "ruleNSGA-II", "ruleNLN", "rulePLCS"]

print("=== F1 scores ===")
header = f"{'Dataset':20s}" + "".join(f"{e:>14s}" for e in est_order)
print(header)
for ds in ds_names:
    row = f"{ds_short[ds]:20s}"
    for est in est_order:
        vals = f1_data.get((ds, est), [])
        if vals:
            row += f"{statistics.mean(vals):14.3f}"
        else:
            row += f"{'---':>14s}"
    print(row)

print("\n=== Atom counts ===")
print(header)
for ds in ds_names:
    row = f"{ds_short[ds]:20s}"
    for est in est_order:
        vals = atoms_data.get((ds, est), [])
        if vals:
            row += f"{statistics.mean(vals):14.1f}"
        else:
            row += f"{'---':>14s}"
    print(row)

print("\n=== Fit times (s) ===")
print(header)
for ds in ds_names:
    row = f"{ds_short[ds]:20s}"
    for est in est_order:
        vals = time_data.get((ds, est), [])
        if vals:
            row += f"{statistics.mean(vals):14.1f}"
        else:
            row += f"{'---':>14s}"
    print(row)
