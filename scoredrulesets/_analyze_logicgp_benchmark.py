"""Analyze the LogicGP variants benchmark log."""
import re
import collections
from statistics import mean, stdev

LOG = "benchmarks/logs/benchmark_logicgp_20260324-224305.log"

with open(LOG, "r") as f:
    lines = f.readlines()

# Count statuses
ok = sum(1 for l in lines if "status=OK" in l)
timeout = sum(1 for l in lines if "TIMEOUT" in l)
done = sum(1 for l in lines if "DONE" in l)
print(f"DONE lines: {done}, OK: {ok}, TIMEOUT: {timeout}")

# Parse BENCHMARK lines -- handle both pipe-separated and space-separated
results = collections.defaultdict(lambda: collections.defaultdict(list))
for l in lines:
    # Try pipe-separated format
    m = re.search(
        r"\[BENCHMARK\]\s+(\S+)\s+\|\s+(\S+)\s+\|\s+F1=([0-9.]+).*?Regeln=(\d+)\s+\|\s+Atome=(\d+)\s+\|\s+fit=(\S+)",
        l,
    )
    if not m:
        # Try space-separated format
        m = re.search(
            r"\[BENCHMARK\]\s+(\S+)\s+(\S+)\s+F1=([0-9.]+).*?Regeln=(\d+).*?Atome=(\d+).*?fit=(\S+)",
            l,
        )
    if m:
        est, ds, f1, rules, atoms, fit = m.groups()
        fit_s = float(fit.rstrip("s"))
        results[ds][est].append(
            {"f1": float(f1), "rules": int(rules), "atoms": int(atoms), "fit": fit_s}
        )

total_parsed = sum(len(v) for d in results.values() for v in d.values())
print(f"Parsed BENCHMARK entries: {total_parsed}")

# === Per-dataset leaderboard ===
for ds in sorted(results.keys()):
    print(f"\n{'='*70}")
    print(f"  {ds}")
    print(f"{'='*70}")
    est_avgs = []
    for est in sorted(results[ds].keys()):
        runs = results[ds][est]
        avg_f1 = mean([r["f1"] for r in runs])
        avg_rules = mean([r["rules"] for r in runs])
        avg_atoms = mean([r["atoms"] for r in runs])
        avg_fit = mean([r["fit"] for r in runs])
        est_avgs.append((avg_f1, est, avg_rules, avg_atoms, avg_fit, len(runs)))
    est_avgs.sort(reverse=True)
    print(f"  {'Estimator':32s}  {'F1':>7s}  {'Rules':>5s}  {'Atoms':>5s}  {'Fit(s)':>7s}  n")
    print(f"  {'-'*32}  {'-'*7}  {'-'*5}  {'-'*5}  {'-'*7}  --")
    for f1, est, rules, atoms, fit, n in est_avgs:
        print(f"  {est:32s}  {f1:.4f}  {rules:5.1f}  {atoms:5.1f}  {fit:7.2f}  {n}")

# === Cross-dataset aggregation per estimator ===
print(f"\n{'='*70}")
print("  GLOBAL AVERAGES (over all datasets)")
print(f"{'='*70}")

global_agg = collections.defaultdict(list)
for ds in results:
    for est in results[ds]:
        runs = results[ds][est]
        avg_f1 = mean([r["f1"] for r in runs])
        avg_atoms = mean([r["atoms"] for r in runs])
        avg_fit = mean([r["fit"] for r in runs])
        global_agg[est].append({"f1": avg_f1, "atoms": avg_atoms, "fit": avg_fit, "ds": ds})

global_avgs = []
for est, ds_list in global_agg.items():
    n_ds = len(ds_list)
    avg_f1 = mean([d["f1"] for d in ds_list])
    avg_atoms = mean([d["atoms"] for d in ds_list])
    avg_fit = mean([d["fit"] for d in ds_list])
    global_avgs.append((avg_f1, est, avg_atoms, avg_fit, n_ds))

global_avgs.sort(reverse=True)
print(f"  {'Estimator':32s}  {'F1':>7s}  {'Atoms':>5s}  {'Fit(s)':>7s}  DS")
print(f"  {'-'*32}  {'-'*7}  {'-'*5}  {'-'*7}  --")
for f1, est, atoms, fit, n_ds in global_avgs:
    print(f"  {est:32s}  {f1:.4f}  {atoms:5.1f}  {fit:7.2f}  {n_ds}")

# === Pairwise comparisons ===
PAIRS = [
    ("lgp_rlcw_macro", "lgp_rlcw_micro", "F1-Averaging (RLCW): Macro vs Micro"),
    ("lgp_flcw_macro", "lgp_flcw_micro", "F1-Averaging (FLCW): Macro vs Micro"),
    ("lgp_rlcw_macro", "lgp_flcw_macro", "Trainer-Typ: RLCW vs FLCW (Macro)"),
    ("lgp_rlcw_micro", "lgp_flcw_micro", "Trainer-Typ: RLCW vs FLCW (Micro)"),
    ("lgp_rlcw_macro", "lgp_rlcw_macro_fast", "Budget: Standard vs Fast (RLCW)"),
    ("lgp_flcw_macro", "lgp_flcw_macro_fast", "Budget: Standard vs Fast (FLCW)"),
    ("lgp_rlcw_macro", "lgp_rlcw_macro_singleton", "Literal-Gen: Full vs Singleton"),
    ("lgp_rlcw_macro", "lgp_rlcw_macro_bins3", "Bins: 5 vs 3"),
    ("lgp_rlcw_macro", "lgp_rlcw_macro_bins7", "Bins: 5 vs 7"),
    ("lgp_rlcw_macro", "lgp_rlcw_macro_nofilt", "Filter: 0.1 vs 0.0"),
    ("lgp_rlcw_macro", "lgp_rlcw_macro_strongfilt", "Filter: 0.1 vs 0.25"),
    ("lgp_rlcw_macro", "lgp_rlcw_macro_bigpop", "Population: 50 vs 80"),
]

print(f"\n{'='*70}")
print("  PAIRWISE COMPARISON SUMMARY")
print(f"{'='*70}")

for name_a, name_b, desc in PAIRS:
    wins_a, wins_b, ties = 0, 0, 0
    deltas = []
    for ds in sorted(results.keys()):
        runs_a = results[ds].get(name_a, [])
        runs_b = results[ds].get(name_b, [])
        if not runs_a or not runs_b:
            continue
        f1_a = mean([r["f1"] for r in runs_a])
        f1_b = mean([r["f1"] for r in runs_b])
        delta = f1_a - f1_b
        deltas.append(delta)
        if abs(delta) < 0.01:
            ties += 1
        elif delta > 0:
            wins_a += 1
        else:
            wins_b += 1

    avg_delta = mean(deltas) if deltas else 0
    total = wins_a + wins_b + ties
    if total == 0:
        verdict = "NO DATA"
    elif wins_a >= 2 * max(wins_b, 1) and avg_delta > 0.01:
        verdict = f"{name_a} CLEARLY BETTER -> remove {name_b}"
    elif wins_b >= 2 * max(wins_a, 1) and avg_delta < -0.01:
        verdict = f"{name_b} CLEARLY BETTER -> remove {name_a}"
    elif abs(avg_delta) < 0.02:
        verdict = "~EQUIVALENT -> merge candidates"
    else:
        verdict = "MIXED -> keep both"

    print(f"\n  {desc}")
    print(f"    {name_a}: {wins_a} wins | {name_b}: {wins_b} wins | ties: {ties} | avg dF1: {avg_delta:+.4f}")
    print(f"    --> {verdict}")

# === Timeout analysis ===
print(f"\n{'='*70}")
print("  TIMEOUT ANALYSIS")
print(f"{'='*70}")
for l in lines:
    if "TIMEOUT" in l and "DONE" in l:
        m2 = re.search(r"dataset=(\S+).*estimator=(\S+)", l)
        if m2:
            print(f"  TIMEOUT: {m2.group(2):32s} on {m2.group(1)}")

