"""Trace Pareto front dynamics during LogicGP evolution on car_evaluation."""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from scoredrulesets.benchmarking.datasets import load_dataset_registry
from scoredrulesets.estimators.logicgp import (
    LogicGPClassifier,
    _compute_weights,
    _evaluate_fitness,
    _evaluate_fitness_rlcw,
    _pareto_front,
    _pareto_front_rlcw,
    _tournament_trim,
    _Polynomial,
)

reg = load_dataset_registry(include_online_uci=True, include_synthetic=True)
b = reg["uci_car_evaluation"]
X, y = b.X, b.y
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
n_classes = len(np.unique(y_tr))

# Patch _run_gp to trace evolution
original_run_gp = LogicGPClassifier._run_gp
gen_stats = []

def traced_run_gp(self, population, all_literals, X_disc, y_idx, n_classes_,
                  *, X_val=None, y_val=None):
    """Patched _run_gp that collects per-generation statistics."""
    import time
    use_rlcw, f1_average = self._resolve_trainer_config()
    evaluate_fn = _evaluate_fitness_rlcw if use_rlcw else _evaluate_fitness
    pareto_fn = _pareto_front_rlcw if use_rlcw else _pareto_front

    for poly in population:
        _compute_weights(poly, X_disc, y_idx, n_classes_)

    evaluated = [(poly, evaluate_fn(poly, X_disc, y_idx, n_classes_)) for poly in population]
    
    # Initial population stats
    front_before_trim = pareto_fn(evaluated)
    gen_stats.append({
        "gen": -1,
        "pop_before_pareto": len(evaluated),
        "pareto_front_size": len(front_before_trim),
        "pop_after_trim": min(len(front_before_trim), self.population_size or 9999),
        "multi_mono_before": sum(1 for p, _ in evaluated if len(p.monomials) > 1),
        "multi_mono_after_pareto": sum(1 for p, _ in front_before_trim if len(p.monomials) > 1),
        "max_monomials": max(len(p.monomials) for p, _ in evaluated),
        "best_consolidated": max(f.consolidated for _, f in evaluated),
        "sizes": sorted(set(f.size for _, f in front_before_trim)),
    })

    evaluated = front_before_trim
    if self.population_size is not None and len(evaluated) > self.population_size:
        evaluated = _tournament_trim(evaluated, self.population_size, self.tournament_size, self._rng_)

    # Now delegate to original for the rest
    return original_run_gp(self, population, all_literals, X_disc, y_idx, n_classes_,
                           X_val=X_val, y_val=y_val)

# Don't patch - instead, do a manual step-through to get detailed stats
# Let's just run the original and also capture what the Pareto front looks like

# 1) FLCW: per-class recall + size  (n_classes + 1 = 5 objectives)
print("=" * 70)
print("FLCW (per-class recall + size) on car_evaluation")
print("=" * 70)

from scoredrulesets.estimators.logicgp import _discretize_features, _generate_literals

X_disc, binners, cat_masks = _discretize_features(X_tr, n_bins=5)
all_lits = _generate_literals(X_disc)

class_to_idx = {label: idx for idx, label in enumerate(np.unique(y_tr))}
y_idx = np.array([class_to_idx[v] for v in y_tr])

# Build initial population (1 polynomial per literal + seeds)
clf = LogicGPClassifier(
    trainer="flcw", f1_averaging="macro",
    max_generations=200, stagnation_generations=40,
    population_size=50, n_adaptations_per_gen=12,
    n_bins=5, random_state=0,
)
clf._rng_ = np.random.default_rng(0)
population = clf._init_population(all_lits, n_classes, X_disc, y_idx)
print(f"Initial population: {len(population)} individuals")
print(f"  Multi-monomial: {sum(1 for p in population if len(p.monomials) > 1)}")
print(f"  Max monomials: {max(len(p.monomials) for p in population)}")

# Evaluate initial population
for poly in population:
    _compute_weights(poly, X_disc, y_idx, n_classes)

evaluated_flcw = [(p, _evaluate_fitness(p, X_disc, y_idx, n_classes)) for p in population]

# Check Pareto front
front_flcw = _pareto_front(evaluated_flcw)
print(f"\nFLCW Pareto front from initial population:")
print(f"  Front size: {len(front_flcw)}")
print(f"  Multi-monomial in front: {sum(1 for p, _ in front_flcw if len(p.monomials) > 1)}")
print(f"  Model sizes in front: {sorted(set(f.size for _, f in front_flcw))}")
print(f"  Best consolidated: {max(f.consolidated for _, f in front_flcw):.4f}")

# After tournament_trim with pop_size=50
if len(front_flcw) > 50:
    trimmed_flcw = _tournament_trim(front_flcw, 50, 5, np.random.default_rng(0))
    print(f"\nAfter tournament_trim to 50:")
    print(f"  Multi-monomial: {sum(1 for p, _ in trimmed_flcw if len(p.monomials) > 1)}")
    print(f"  Model sizes: {sorted(set(f.size for _, f in trimmed_flcw))}")
else:
    print(f"\n  (front already <= 50, no trimming needed)")

# Detailed: show recalls and sizes for front members
print(f"\nFront details (first 20):")
for i, (poly, fit) in enumerate(front_flcw[:20]):
    n_mon = len(poly.monomials)
    recalls = fit.objectives
    print(f"  [{i}] size={fit.size}, monomials={n_mon}, recalls={np.round(recalls, 3)}, consolidated={fit.consolidated:.4f}")

# ---- RLCW comparison ----
print("\n" + "=" * 70)
print("RLCW (max_recall + mean_other + size, class-bound) on car_evaluation")
print("=" * 70)

evaluated_rlcw = [(p, _evaluate_fitness_rlcw(p, X_disc, y_idx, n_classes)) for p in population]
front_rlcw = _pareto_front_rlcw(evaluated_rlcw)

print(f"RLCW Pareto front from initial population:")
print(f"  Front size: {len(front_rlcw)}")
print(f"  Multi-monomial in front: {sum(1 for p, _ in front_rlcw if len(p.monomials) > 1)}")

# Show by best_class
from collections import Counter
best_class_counts = Counter(f.best_class for _, f in front_rlcw)
print(f"  By best_class: {dict(best_class_counts)}")
print(f"  Model sizes: {sorted(set(f.size for _, f in front_rlcw))}")

print(f"\nRLCW Front details (first 20):")
for i, (poly, fit) in enumerate(front_rlcw[:20]):
    n_mon = len(poly.monomials)
    print(f"  [{i}] size={fit.size}, monomials={n_mon}, best_class={fit.best_class}, "
          f"max_recall={fit.max_recall:.3f}, mean_other={fit.mean_other:.3f}, "
          f"consolidated={fit.consolidated:.4f}")

# ---- Compare: what if no population limit (original paper behavior)? ----
print("\n" + "=" * 70)
print("Impact of population_size limit")
print("=" * 70)

print(f"\nFLCW front: {len(front_flcw)} individuals → trim to 50 → loses {max(0, len(front_flcw)-50)} individuals")
print(f"RLCW front: {len(front_rlcw)} individuals → trim to 50 → loses {max(0, len(front_rlcw)-50)} individuals")

# Full run without pop limit
clf_no_limit = LogicGPClassifier(
    trainer="flcw", f1_averaging="macro",
    max_generations=100, stagnation_generations=30,
    population_size=None,  # No limit - original paper behavior
    n_adaptations_per_gen=12,
    n_bins=5, random_state=0,
)
clf_no_limit.fit(X_tr, y_tr)
y_pred_nolimit = clf_no_limit.predict(X_te)
f1_nolimit = f1_score(y_te, y_pred_nolimit, average="macro")
rs_nolimit = clf_no_limit.to_ruleset()
n_atoms_nolimit = sum(len(r.atoms) for r in rs_nolimit.rules)
n_mons_nolimit = len([r for r in rs_nolimit.rules if r.atoms])
print(f"\nFLCW population_size=None: F1={f1_nolimit:.4f}, monomials={n_mons_nolimit}, atoms={n_atoms_nolimit}")

clf_limited = LogicGPClassifier(
    trainer="flcw", f1_averaging="macro",
    max_generations=100, stagnation_generations=30,
    population_size=50,
    n_adaptations_per_gen=12,
    n_bins=5, random_state=0,
)
clf_limited.fit(X_tr, y_tr)
y_pred_limited = clf_limited.predict(X_te)
f1_limited = f1_score(y_te, y_pred_limited, average="macro")
rs_limited = clf_limited.to_ruleset()
n_atoms_limited = sum(len(r.atoms) for r in rs_limited.rules)
n_mons_limited = len([r for r in rs_limited.rules if r.atoms])
print(f"FLCW population_size=50:   F1={f1_limited:.4f}, monomials={n_mons_limited}, atoms={n_atoms_limited}")

# RLCW same comparison
clf_rlcw_nolimit = LogicGPClassifier(
    trainer="rlcw", f1_averaging="macro",
    max_generations=100, stagnation_generations=30,
    population_size=None,
    n_adaptations_per_gen=12,
    n_bins=5, random_state=0,
)
clf_rlcw_nolimit.fit(X_tr, y_tr)
y_pred_rlcw_nl = clf_rlcw_nolimit.predict(X_te)
f1_rlcw_nl = f1_score(y_te, y_pred_rlcw_nl, average="macro")
rs_rlcw_nl = clf_rlcw_nolimit.to_ruleset()
n_mons_rlcw_nl = len([r for r in rs_rlcw_nl.rules if r.atoms])
print(f"\nRLCW population_size=None: F1={f1_rlcw_nl:.4f}, monomials={n_mons_rlcw_nl}")

clf_rlcw_limited = LogicGPClassifier(
    trainer="rlcw", f1_averaging="macro",
    max_generations=100, stagnation_generations=30,
    population_size=50,
    n_adaptations_per_gen=12,
    n_bins=5, random_state=0,
)
clf_rlcw_limited.fit(X_tr, y_tr)
y_pred_rlcw_l = clf_rlcw_limited.predict(X_te)
f1_rlcw_l = f1_score(y_te, y_pred_rlcw_l, average="macro")
rs_rlcw_l = clf_rlcw_limited.to_ruleset()
n_mons_rlcw_l = len([r for r in rs_rlcw_l.rules if r.atoms])
print(f"RLCW population_size=50:   F1={f1_rlcw_l:.4f}, monomials={n_mons_rlcw_l}")

