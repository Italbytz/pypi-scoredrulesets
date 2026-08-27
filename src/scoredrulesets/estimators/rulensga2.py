"""
ruleGP – Genetic Programming on full Scored Rule Set individuals
================================================================

Combines ideas from:
- **logicGP**: Pareto-based GP, FLCW weight computation, model-size objective
- **ruleLCS**: post-hoc compaction (backward elimination)
- **NSGA-II**: crowding-distance for diversity on the Pareto front

Key differences from existing estimators:
- Each GP individual is a *complete rule set* (multiple rules + default),
  not a single rule or a polynomial with set-literals (logicGP).
- Three-objective NSGA-II: maximize F1, maximize mean-other-class F1,
  minimize model complexity (total atoms), with class-bound dominance.
- Works natively on numeric and categorical features (no forced discretisation).
- Post-hoc backward elimination + atom pruning for maximum compactness.
- Score refit after every structural mutation (FLCW-style).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import f1_score
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ..runtime import predict as predict_from_ruleset
from ..runtime import predict_proba as predict_proba_from_ruleset
from ..schema import AggregationSpec, Atom, Rule, ScoredRuleSet
from ._time_budget import deadline_reached, resolve_deadline
from .atom_selection import (
    available_atom_selection_strategies,
    is_atom_selection_strategy_available,
    iter_native_candidate_signatures,
    select_signatures_by_strategy,
    signature_key,
)
from .atom_space import NativeAtomSpaceStrategy
from .atom_space import build_native_feature_specs
from .base import BaseRuleSetEstimator


# ---------------------------------------------------------------------------
# Internal gene representations
# ---------------------------------------------------------------------------

@dataclass
class _AtomGene:
    feature_idx: int
    op: str
    value: object


@dataclass
class _RuleGene:
    atoms: list[_AtomGene]


@dataclass
class _Individual:
    """A full rule set individual: list of rule-genes + cached fitness."""
    rules: list[_RuleGene]
    fitness_f1: float = -1.0
    fitness_complexity: int = 999_999
    fitness_other_class_f1: float = 0.0
    fitness_best_class: int = 0
    fitness_class_f1: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def total_atoms(self) -> int:
        return sum(len(r.atoms) for r in self.rules)

    def clone(self) -> "_Individual":
        return _Individual(
            rules=[_RuleGene(atoms=list(r.atoms)) for r in self.rules],
        )


# ---------------------------------------------------------------------------
# NSGA-II helpers
# ---------------------------------------------------------------------------

def _fast_nondominated_sort(pop: list[_Individual]) -> list[list[int]]:
    """NSGA-II fast non-dominated sort.

    Objectives: max F1, max mean-other-class F1, min complexity.
    """
    n = len(pop)
    domination_count = [0] * n
    dominated_set: list[list[int]] = [[] for _ in range(n)]
    fronts: list[list[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(pop[p], pop[q]):
                dominated_set[p].append(q)
            elif _dominates(pop[q], pop[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front: list[int] = []
        for p in fronts[i]:
            for q in dominated_set[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    # Remove trailing empty front
    if not fronts[-1]:
        fronts.pop()
    return fronts


def _dominates(a: _Individual, b: _Individual) -> bool:
    """Class-bound Pareto dominance (RLCW-style).

    A dominates B only if they share the same best-predicted class,
    A is at least as good on all objectives, and strictly better on at
    least one. This creates per-class niches and encourages the population
    to maintain class-diverse solutions.
    """
    if a.fitness_best_class != b.fitness_best_class:
        return False
    f1_ge = a.fitness_f1 >= b.fitness_f1 - 1e-12
    oc_ge = a.fitness_other_class_f1 >= b.fitness_other_class_f1 - 1e-12
    cx_ge = a.fitness_complexity <= b.fitness_complexity
    f1_gt = a.fitness_f1 > b.fitness_f1 + 1e-12
    oc_gt = a.fitness_other_class_f1 > b.fitness_other_class_f1 + 1e-12
    cx_gt = a.fitness_complexity < b.fitness_complexity
    return f1_ge and oc_ge and cx_ge and (f1_gt or oc_gt or cx_gt)


def _crowding_distance(pop: list[_Individual], front: list[int]) -> dict[int, float]:
    """Compute crowding distance for a single front."""
    n = len(front)
    if n <= 2:
        return {idx: float("inf") for idx in front}

    distances: dict[int, float] = {idx: 0.0 for idx in front}

    for key_fn in [
        lambda ind: ind.fitness_f1,
        lambda ind: ind.fitness_other_class_f1,
        lambda ind: -ind.fitness_complexity,
    ]:
        sorted_front = sorted(front, key=lambda i: key_fn(pop[i]))
        obj_min = key_fn(pop[sorted_front[0]])
        obj_max = key_fn(pop[sorted_front[-1]])
        span = obj_max - obj_min
        if span < 1e-12:
            continue
        distances[sorted_front[0]] = float("inf")
        distances[sorted_front[-1]] = float("inf")
        for k in range(1, n - 1):
            distances[sorted_front[k]] += (
                key_fn(pop[sorted_front[k + 1]]) - key_fn(pop[sorted_front[k - 1]])
            ) / span

    return distances


def _nsga2_select(pop: list[_Individual], n_select: int) -> list[_Individual]:
    """Select n_select individuals using NSGA-II ranking + crowding distance."""
    if len(pop) <= n_select:
        return list(pop)

    fronts = _fast_nondominated_sort(pop)
    selected: list[int] = []

    for front in fronts:
        if len(selected) + len(front) <= n_select:
            selected.extend(front)
        else:
            remaining = n_select - len(selected)
            cd = _crowding_distance(pop, front)
            ranked = sorted(front, key=lambda i: cd[i], reverse=True)
            selected.extend(ranked[:remaining])
            break

    return [pop[i] for i in selected]


# ---------------------------------------------------------------------------
# Feature spec builder
# ---------------------------------------------------------------------------

def _build_feature_specs(
    X: np.ndarray,
    max_thresholds: int | None = None,
    low_cardinality_threshold: int = 10,
    atom_space_strategy: NativeAtomSpaceStrategy = "hybrid",
) -> list[dict]:
    """Build per-feature specs describing the operators and values RuleGP may use.

    Parameters
    ----------
    low_cardinality_threshold : int
        Numeric features with at most this many unique values are additionally
        treated as categorical (``kind="both"``).  This generates ``==`` atoms
        alongside the usual ``<=``/``>``/``between`` atoms, which is essential
        for datasets where categorical features are integer-encoded (e.g.
        MONK, car_evaluation).
    """
    return build_native_feature_specs(
        X,
        max_thresholds=max_thresholds,
        low_cardinality_threshold=low_cardinality_threshold,
        strategy=atom_space_strategy,
    )


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

class RuleNSGA2Classifier(BaseRuleSetEstimator):
    """Genetic Programming on full Scored Rule Set individuals.

    Evolves a population of complete rule sets using NSGA-II multi-objective
    optimisation (F1 vs. model complexity).  After evolution, backward
    elimination and atom pruning compact the final model.

    Parameters
    ----------
    population_size : int
        Number of individuals in the GP population.
    generations : int
        Maximum number of GP generations.
    max_rules : int
        Maximum number of rules per individual (excluding default).
    max_atoms_per_rule : int
        Maximum number of atoms per rule.
    mutation_rate : float
        Probability of mutation per offspring.
    crossover_rate : float
        Probability of crossover when producing offspring.
    tournament_size : int
        Tournament size for parent selection.
    min_samples_leaf : int
        Minimum samples that a rule must cover.
    validation_fraction : float
        Fraction of training data held out for fitness evaluation.
    early_stopping_rounds : int
        Stop after this many generations without Pareto-front improvement.
    enable_compaction : bool
        Run backward elimination + atom pruning on the final model.
    max_fit_seconds : float or None
        Maximum wall-clock runtime for the complete fit in seconds
        (``None`` = unlimited). The budget covers data setup and the GP loop;
        if it is exhausted the loop stops cleanly and the best model found so
        far is returned.
    min_max_weight : float
        If > 0, prefilters atom candidates by their maximal class-balanced
        weight on training data. Higher values reduce the search space.
    atom_preselection_strategy : str
        Optional atom preselection strategy. ``"none"`` keeps all atoms;
        any registered plugin strategy can preselect candidate atoms.
    atom_preselection_top_k : int | None
        Number of atoms retained when a preselection strategy is used.
    random_state : int or None
        Random seed.
    """

    def __init__(
        self,
        population_size: int = 80,
        generations: int = 100,
        max_rules: int = 6,
        max_atoms_per_rule: int = 3,
        mutation_rate: float = 0.35,
        crossover_rate: float = 0.6,
        tournament_size: int = 3,
        min_samples_leaf: int = 3,
        validation_fraction: float = 0.2,
        early_stopping_rounds: int = 15,
        enable_compaction: bool = True,
        max_fit_seconds: float | None = None,
        min_max_weight: float = 0.0,
        max_thresholds_per_feature: int | None = None,
        atom_space_strategy: NativeAtomSpaceStrategy = "hybrid",
        atom_preselection_strategy: str = "none",
        atom_preselection_top_k: int | None = None,
        random_state: int | None = None,
    ):
        self.population_size = population_size
        self.generations = generations
        self.max_rules = max_rules
        self.max_atoms_per_rule = max_atoms_per_rule
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.min_samples_leaf = min_samples_leaf
        self.validation_fraction = validation_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.enable_compaction = enable_compaction
        self.max_fit_seconds = max_fit_seconds
        self.min_max_weight = min_max_weight
        self.max_thresholds_per_feature = max_thresholds_per_feature
        self.atom_space_strategy = atom_space_strategy
        self.atom_preselection_strategy = atom_preselection_strategy
        self.atom_preselection_top_k = atom_preselection_top_k
        self.random_state = random_state
        self._atom_pool_: dict[int, list[_AtomGene]] = {}

        if self.atom_preselection_strategy != "none" and not is_atom_selection_strategy_available(
            self.atom_preselection_strategy
        ):
            available = ", ".join(available_atom_selection_strategies())
            raise ValueError(
                "atom_preselection_strategy must be 'none' or a registered "
                f"atom-selection strategy. Available registered strategies: [{available}]"
            )
        if self.atom_preselection_strategy != "none":
            if self.atom_preselection_top_k is None or int(self.atom_preselection_top_k) <= 0:
                raise ValueError(
                    "atom_preselection_top_k must be a positive integer when "
                    "atom_preselection_strategy != 'none'."
                )

    # ------------------------------------------------------------------
    # sklearn interface
    # ------------------------------------------------------------------

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=None)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(
            [f"f{i}" for i in range(self.n_features_in_)], dtype=object
        )
        self.classes_ = unique_labels(y)
        n_classes = len(self.classes_)
        rng = np.random.default_rng(self.random_state)

        # Absolute fit-time deadline (covers setup + GP loop). ``None`` = no limit.
        self._fit_deadline_ = resolve_deadline(self.max_fit_seconds)

        class_to_idx = {lab: i for i, lab in enumerate(self.classes_)}
        y_idx = np.asarray([class_to_idx[v] for v in y], dtype=int)

        # Train / val split (stratified)
        train_idx, val_idx = self._stratified_split(y_idx, rng)
        X_train, y_train = X[train_idx], y_idx[train_idx]
        X_eval = X[val_idx] if val_idx is not None else X_train
        y_eval = y_idx[val_idx] if val_idx is not None else y_train

        specs = _build_feature_specs(
            X,
            self.max_thresholds_per_feature,
            atom_space_strategy=self.atom_space_strategy,
        )
        self._atom_pool_ = self._build_atom_pool(specs, X_train, y_train, n_classes)

        # ---------- Initialise population ----------
        pop = self._init_population(specs, X_train, y_train, n_classes, rng)
        for ind in pop:
            self._evaluate(ind, X_eval, y_eval, n_classes)

        # ---------- GP loop ----------
        best_front_size = 0
        best_front_f1 = -1.0
        stale = 0
        _deadline = self._fit_deadline_
        gen_ran = 0

        for gen in range(max(1, self.generations)):
            gen_ran = gen + 1

            # Time budget check
            if deadline_reached(_deadline):
                break

            # Produce offspring
            offspring: list[_Individual] = []
            while len(offspring) < self.population_size:
                p1 = self._tournament(pop, rng)
                p2 = self._tournament(pop, rng)
                child = self._crossover(p1, p2, rng)
                child = self._mutate(child, specs, rng)
                # Refit scores for the child
                self._refit_scores(child, X_train, y_train, n_classes)
                self._evaluate(child, X_eval, y_eval, n_classes)
                offspring.append(child)
                # Stop building offspring as soon as the budget is exhausted;
                # the partial offspring set is still valid for selection.
                if deadline_reached(_deadline):
                    break

            # NSGA-II environmental selection
            combined = pop + offspring
            pop = _nsga2_select(combined, self.population_size)

            # Early stopping: track Pareto-front improvement
            front = _fast_nondominated_sort(pop)[0]
            front_f1 = max(pop[i].fitness_f1 for i in front)
            if front_f1 > best_front_f1 + 1e-6 or len(front) > best_front_size:
                best_front_f1 = front_f1
                best_front_size = len(front)
                stale = 0
            else:
                stale += 1
                if stale >= self.early_stopping_rounds:
                    break

        # ---------- Select best individual ----------
        best = self._select_best(pop)

        # ---------- Post-hoc compaction ----------
        if self.enable_compaction:
            best = self._compact(best, X_eval, y_eval, n_classes)

        # ---------- Refit scores on ALL data ----------
        self._refit_scores(best, X, y_idx, n_classes)
        self._evaluate(best, X, y_idx, n_classes)

        # ---------- Build ScoredRuleSet ----------
        self.ruleset_ = self._to_ruleset(best, X, y_idx, n_classes, gen_ran)
        self.ruleset_.validate()
        return self

    def predict(self, X):
        check_is_fitted(self, "ruleset_")
        X = check_array(X, dtype=None)
        return predict_from_ruleset(self.ruleset_, X)

    def predict_proba(self, X):
        check_is_fitted(self, "ruleset_")
        X = check_array(X, dtype=None)
        return predict_proba_from_ruleset(self.ruleset_, X)

    def to_ruleset(self) -> ScoredRuleSet:
        check_is_fitted(self, "ruleset_")
        return self.ruleset_

    # ------------------------------------------------------------------
    # Population initialisation
    # ------------------------------------------------------------------

    def _init_population(
        self,
        specs: list[dict],
        X: np.ndarray,
        y: np.ndarray,
        n_classes: int,
        rng: np.random.Generator,
    ) -> list[_Individual]:
        pop: list[_Individual] = []
        n_init = max(2, self.population_size)

        # (a) Systematic single-atom seeds (analogous to logicGP's per-literal seeding)
        for ind in self._seed_systematic(specs, X, y, n_classes, rng):
            pop.append(ind)

        # (b) Class-discriminative seeds: one rule per class
        for _ in range(min(n_init // 4, n_classes * 2)):
            ind = self._seed_class_discriminative(specs, X, y, n_classes, rng)
            if ind is not None:
                self._refit_scores(ind, X, y, n_classes)
                pop.append(ind)

        # (c) Greedy sequential-cover seed
        for _ in range(min(3, n_init // 4)):
            ind = self._seed_greedy_cover(specs, X, y, n_classes, rng)
            self._refit_scores(ind, X, y, n_classes)
            pop.append(ind)

        # (d) Random individuals to fill the rest
        while len(pop) < n_init:
            ind = self._random_individual(specs, rng)
            self._refit_scores(ind, X, y, n_classes)
            pop.append(ind)

        return pop

    def _random_individual(
        self, specs: list[dict], rng: np.random.Generator
    ) -> _Individual:
        n_rules = int(rng.integers(1, self.max_rules + 1))
        rules = [self._random_rule(specs, rng) for _ in range(n_rules)]
        return _Individual(rules=rules)

    def _random_rule(
        self, specs: list[dict], rng: np.random.Generator
    ) -> _RuleGene:
        n_atoms = int(rng.integers(1, self.max_atoms_per_rule + 1))
        atoms = [self._random_atom(specs, rng) for _ in range(n_atoms)]
        return _RuleGene(atoms=atoms)

    def _random_atom(
        self, specs: list[dict], rng: np.random.Generator
    ) -> _AtomGene:
        if self._atom_pool_ and rng.random() < 0.85:
            fi_keys = list(self._atom_pool_.keys())
            if fi_keys:
                fi = int(fi_keys[int(rng.integers(0, len(fi_keys)))])
                atoms = self._atom_pool_.get(fi, [])
                if atoms:
                    atom = atoms[int(rng.integers(0, len(atoms)))]
                    return _AtomGene(atom.feature_idx, atom.op, atom.value)

        spec = specs[int(rng.integers(0, len(specs)))]
        fi = spec["idx"]
        if spec["kind"] in ("num", "both"):
            thr = spec.get("thresholds", [])
            ivs = spec.get("intervals", [])
            cats = spec.get("categories", []) if spec["kind"] == "both" else []
            ops = []
            if thr:
                ops += ["<=", ">"]
            if ivs:
                ops.append("between")
            if cats:
                ops.append("==")
            if ops:
                op = ops[int(rng.integers(0, len(ops)))]
                if op in ("<=", ">"):
                    return _AtomGene(fi, op, float(thr[int(rng.integers(0, len(thr)))]))
                if op == "between":
                    iv = ivs[int(rng.integers(0, len(ivs)))]
                    return _AtomGene(fi, "between", [float(iv[0]), float(iv[1])])
                if op == "==":
                    k = len(cats)
                    if k >= 3 and rng.random() < 0.4:
                        # Random non-trivial subset (analogous to logicGP SetLiterals)
                        subset_size = int(rng.integers(2, k))
                        idxs = rng.choice(k, size=subset_size, replace=False)
                        chosen = sorted([cats[int(i)] for i in idxs])
                        return _AtomGene(fi, "in", chosen)
                    return _AtomGene(fi, "==", cats[int(rng.integers(0, k))])
        cats = spec.get("categories", [])
        k = len(cats)
        if cats:
            if k >= 3 and rng.random() < 0.4:
                # Random non-trivial subset (analogous to logicGP SetLiterals)
                subset_size = int(rng.integers(2, k))
                idxs = rng.choice(k, size=subset_size, replace=False)
                chosen = sorted([cats[int(i)] for i in idxs])
                return _AtomGene(fi, "in", chosen)
            return _AtomGene(fi, "==", cats[int(rng.integers(0, k))])
        return _AtomGene(fi, ">", 0.0)

    def _build_atom_pool(
        self,
        specs: list[dict],
        X: np.ndarray,
        y: np.ndarray,
        n_classes: int,
    ) -> dict[int, list[_AtomGene]]:
        """Build a feature-indexed atom pool, optionally filtered by min_max_weight.

        The filter mirrors logicGP-RLCW's literal preselection idea: keep atoms
        whose maximal class-balanced weight exceeds the configured threshold.
        """
        class_counts = np.bincount(y, minlength=n_classes).astype(float)
        class_counts = np.maximum(class_counts, 1.0)

        pool: dict[int, list[_AtomGene]] = {}
        top_c2_keys: set[tuple[int, str, str]] | None = None

        if self.atom_preselection_strategy != "none":
            candidates = []
            for sig in iter_native_candidate_signatures(specs):
                atom = _AtomGene(int(sig[0]), str(sig[1]), sig[2])
                candidates.append((signature_key(sig), self._atom_mask(atom, X)))
            selected = select_signatures_by_strategy(
                strategy=self.atom_preselection_strategy,
                candidates=candidates,
                y_idx=y,
                n_classes=n_classes,
                min_samples_leaf=self.min_samples_leaf,
                top_k=int(self.atom_preselection_top_k),
            )
            top_c2_keys = {(int(fi), str(op), str(val)) for fi, op, val in selected}
        for spec in specs:
            fi = int(spec["idx"])
            candidates: list[_AtomGene] = []

            if spec["kind"] in ("num", "both"):
                for thr in spec.get("thresholds", []):
                    candidates.append(_AtomGene(fi, "<=", float(thr)))
                    candidates.append(_AtomGene(fi, ">", float(thr)))
                for lo, hi in spec.get("intervals", []):
                    candidates.append(_AtomGene(fi, "between", [float(lo), float(hi)]))

            if spec["kind"] in ("cat", "both"):
                cats = list(spec.get("categories", []))
                for cat in cats:
                    candidates.append(_AtomGene(fi, "==", cat))
                if len(cats) >= 3:
                    # Add a few deterministic non-trivial subsets.
                    candidates.append(_AtomGene(fi, "in", sorted(cats[:2])))
                    candidates.append(_AtomGene(fi, "in", sorted(cats[-2:])))
                    if len(cats) >= 4:
                        mid = len(cats) // 2
                        candidates.append(_AtomGene(fi, "in", sorted(cats[mid - 1:mid + 1])))

            if not candidates:
                continue

            scored: list[tuple[float, _AtomGene]] = []
            for atom in candidates:
                if top_c2_keys is not None:
                    atom_key = (atom.feature_idx, atom.op, str(atom.value))
                    if atom_key not in top_c2_keys:
                        continue
                mask = self._atom_mask(atom, X)
                support = int(mask.sum())
                if support < self.min_samples_leaf:
                    continue
                counts = np.bincount(y[mask], minlength=n_classes).astype(float)
                balanced = counts / class_counts
                total = float(balanced.sum())
                if total <= 0:
                    continue
                max_weight = float((balanced / total).max())
                if self.min_max_weight > 0 and max_weight <= self.min_max_weight:
                    continue
                scored.append((max_weight, atom))

            # Fallback if filtering was too strict on a feature.
            if not scored and self.min_max_weight > 0:
                for atom in candidates:
                    mask = self._atom_mask(atom, X)
                    support = int(mask.sum())
                    if support < self.min_samples_leaf:
                        continue
                    counts = np.bincount(y[mask], minlength=n_classes).astype(float)
                    balanced = counts / class_counts
                    total = float(balanced.sum())
                    if total <= 0:
                        continue
                    max_weight = float((balanced / total).max())
                    scored.append((max_weight, atom))
                scored.sort(key=lambda x: x[0], reverse=True)
                scored = scored[:2]

            if scored:
                unique: dict[tuple[int, str, str], _AtomGene] = {}
                for _, atom in sorted(scored, key=lambda x: x[0], reverse=True):
                    key = (atom.feature_idx, atom.op, str(atom.value))
                    unique[key] = atom
                pool[fi] = list(unique.values())

        return pool

    def _seed_systematic(
        self,
        specs: list[dict],
        X: np.ndarray,
        y: np.ndarray,
        n_classes: int,
        rng: np.random.Generator,
    ) -> list[_Individual]:
        """One single-rule individual per representative atom (analogous to logicGP's
        per-literal seeding).  For each feature, picks at most 2 atoms with the
        highest discriminative lift and creates a single-atom-rule individual.
        """
        seeds: list[_Individual] = []
        class_counts = np.bincount(y, minlength=n_classes).astype(float)
        class_counts = np.maximum(class_counts, 1.0)

        for spec in specs:
            fi = spec["idx"]
            candidate_atoms: list[_AtomGene] = []

            if spec["kind"] in ("num", "both"):
                thr = spec.get("thresholds", [])
                if thr:
                    # At most 3 evenly spaced thresholds
                    idxs = np.round(np.linspace(0, len(thr) - 1, min(3, len(thr)))).astype(int)
                    for i in idxs:
                        candidate_atoms.append(_AtomGene(fi, "<=", float(thr[i])))
                        candidate_atoms.append(_AtomGene(fi, ">", float(thr[i])))
                ivs = spec.get("intervals", [])
                if ivs:
                    mid = len(ivs) // 2
                    candidate_atoms.append(
                        _AtomGene(fi, "between", [float(ivs[mid][0]), float(ivs[mid][1])])
                    )

            if spec["kind"] in ("cat", "both"):
                cats = spec.get("categories", [])
                for cat in cats[:4]:
                    candidate_atoms.append(_AtomGene(fi, "==", cat))

            # Score atoms by discriminative lift (max class balanced-weight above uniform)
            scored: list[tuple[float, _AtomGene]] = []
            for atom in candidate_atoms:
                mask = self._atom_mask(atom, X)
                support = int(mask.sum())
                if support < self.min_samples_leaf:
                    continue
                counts = np.bincount(y[mask], minlength=n_classes).astype(float)
                balanced = counts / class_counts
                lift = float(balanced.max() - 1.0 / n_classes)
                scored.append((lift, atom))

            scored.sort(key=lambda x: x[0], reverse=True)
            for _, atom in scored[:2]:  # top-2 per feature
                ind = _Individual(rules=[_RuleGene(atoms=[atom])])
                self._refit_scores(ind, X, y, n_classes)
                seeds.append(ind)

        return seeds

    def _seed_class_discriminative(
        self,
        specs: list[dict],
        X: np.ndarray,
        y: np.ndarray,
        n_classes: int,
        rng: np.random.Generator,
    ) -> _Individual | None:
        """Create an individual with one rule per class, using the most
        discriminative atom for each class."""
        target_class = int(rng.integers(0, n_classes))
        mask_cls = y == target_class
        if mask_cls.sum() < self.min_samples_leaf:
            return None

        rules: list[_RuleGene] = []
        for _ in range(min(self.max_rules, n_classes)):
            cls = int(rng.integers(0, n_classes))
            cls_mask = y == cls
            best_atom = None
            best_lift = -1.0
            # Try a few random atoms, pick the one with best class lift
            for _ in range(min(20, len(specs) * 3)):
                atom = self._random_atom(specs, rng)
                atom_mask = self._atom_mask(atom, X)
                support = int(atom_mask.sum())
                if support < self.min_samples_leaf:
                    continue
                cls_in = int((atom_mask & cls_mask).sum())
                lift = cls_in / support - cls_mask.sum() / len(y)
                if lift > best_lift:
                    best_lift = lift
                    best_atom = atom
            if best_atom is not None:
                rules.append(_RuleGene(atoms=[best_atom]))

        if not rules:
            return None
        return _Individual(rules=rules)

    def _seed_greedy_cover(
        self,
        specs: list[dict],
        X: np.ndarray,
        y: np.ndarray,
        n_classes: int,
        rng: np.random.Generator,
    ) -> _Individual:
        """Greedy sequential covering: iteratively find rules that cover
        misclassified samples, remove covered samples, repeat."""
        rules: list[_RuleGene] = []
        remaining = np.arange(len(y))

        for _ in range(self.max_rules):
            if len(remaining) < self.min_samples_leaf:
                break
            X_sub = X[remaining]
            y_sub = y[remaining]

            # Find the best 1-atom rule among random candidates
            best_rule = None
            best_score = -1.0
            for _ in range(30):
                atom = self._random_atom(specs, rng)
                mask = self._atom_mask(atom, X_sub)
                support = int(mask.sum())
                if support < self.min_samples_leaf:
                    continue
                counts = np.bincount(y_sub[mask], minlength=n_classes).astype(float)
                purity = float(counts.max() / max(counts.sum(), 1))
                score = purity * (support / len(y_sub))
                if score > best_score:
                    best_score = score
                    best_rule = _RuleGene(atoms=[atom])

            if best_rule is None:
                break
            rules.append(best_rule)

            # Remove correctly covered samples
            mask = self._rule_mask(best_rule, X_sub)
            counts = np.bincount(y_sub[mask], minlength=n_classes).astype(float)
            dominant_class = int(np.argmax(counts))
            correct = mask & (y_sub == dominant_class)
            remaining = remaining[~correct]

        if not rules:
            rules = [self._random_rule(specs, rng)]
        return _Individual(rules=rules)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        ind: _Individual,
        X: np.ndarray,
        y: np.ndarray,
        n_classes: int,
    ) -> None:
        """Compute fitness: F1-macro, class-aware objectives, and complexity."""
        preds = self._predict_individual(ind, X, n_classes)
        try:
            ind.fitness_f1 = float(f1_score(y, preds, average="macro", zero_division=0))
            class_f1 = f1_score(
                y, preds, average=None, zero_division=0, labels=list(range(n_classes))
            )
            ind.fitness_class_f1 = np.asarray(class_f1, dtype=float)
            ind.fitness_best_class = int(np.argmax(class_f1)) if len(class_f1) > 0 else 0
            if len(class_f1) > 1:
                other = np.delete(ind.fitness_class_f1, ind.fitness_best_class)
                ind.fitness_other_class_f1 = float(np.mean(other)) if other.size else 0.0
            else:
                ind.fitness_other_class_f1 = 0.0
        except Exception:
            ind.fitness_f1 = 0.0
            ind.fitness_class_f1 = np.zeros(n_classes, dtype=float)
            ind.fitness_best_class = 0
            ind.fitness_other_class_f1 = 0.0
        ind.fitness_complexity = ind.total_atoms

    def _predict_individual(
        self,
        ind: _Individual,
        X: np.ndarray,
        n_classes: int,
    ) -> np.ndarray:
        """Predict class labels for X using individual's rules (argmax-sum)."""
        n = X.shape[0]
        scores = np.zeros((n, n_classes), dtype=float)
        any_fired = np.zeros(n, dtype=bool)

        for rule in ind.rules:
            mask = self._rule_mask(rule, X)
            if not mask.any():
                continue
            any_fired |= mask
            rule_scores = getattr(rule, "_scores", None)
            if rule_scores is not None:
                scores[mask] += rule_scores

        # Default: uniform prior for samples where no rule fires
        if not any_fired.all():
            default_scores = getattr(ind, "_default_scores", None)
            if default_scores is not None:
                scores[~any_fired] = default_scores
            else:
                scores[~any_fired] = 1.0 / max(n_classes, 1)

        return np.argmax(scores, axis=1)

    def _refit_scores(
        self,
        ind: _Individual,
        X: np.ndarray,
        y: np.ndarray,
        n_classes: int,
    ) -> None:
        """Recompute class-balanced FLCW-style scores for each rule and the default."""
        any_fired = np.zeros(X.shape[0], dtype=bool)
        class_counts = np.bincount(y, minlength=n_classes).astype(float)
        class_counts = np.maximum(class_counts, 1.0)

        for rule in ind.rules:
            mask = self._rule_mask(rule, X)
            any_fired |= mask
            if mask.sum() > 0:
                counts = np.bincount(y[mask], minlength=n_classes).astype(float)
                balanced = counts / class_counts
                bal_total = balanced.sum()
                rule._scores = balanced / bal_total if bal_total > 0 else np.ones(n_classes) / n_classes
            else:
                rule._scores = np.ones(n_classes, dtype=float) / n_classes

        # Default scores: from non-covered samples
        no_fire = ~any_fired
        if no_fire.any():
            counts = np.bincount(y[no_fire], minlength=n_classes).astype(float)
            balanced = counts / class_counts
            bal_total = balanced.sum()
            ind._default_scores = balanced / bal_total if bal_total > 0 else np.ones(n_classes) / n_classes
        else:
            ind._default_scores = np.ones(n_classes, dtype=float) / n_classes

    # ------------------------------------------------------------------
    # Genetic operators
    # ------------------------------------------------------------------

    def _tournament(
        self, pop: list[_Individual], rng: np.random.Generator
    ) -> _Individual:
        k = min(self.tournament_size, len(pop))
        idx = rng.choice(len(pop), size=k, replace=False)
        # Prefer: higher F1, then stronger non-best-class performance, then smaller models.
        return min(
            (pop[i] for i in idx),
            key=lambda ind: (
                -ind.fitness_f1,
                -ind.fitness_other_class_f1,
                ind.fitness_complexity,
            ),
        )

    def _crossover(
        self,
        p1: _Individual,
        p2: _Individual,
        rng: np.random.Generator,
    ) -> _Individual:
        if rng.random() >= self.crossover_rate or not p1.rules or not p2.rules:
            return p1.clone()

        # Rule-exchange crossover: take rules from both parents
        cut1 = int(rng.integers(0, len(p1.rules) + 1))
        cut2 = int(rng.integers(0, len(p2.rules) + 1))
        child_rules = (
            [_RuleGene(atoms=list(r.atoms)) for r in p1.rules[:cut1]]
            + [_RuleGene(atoms=list(r.atoms)) for r in p2.rules[cut2:]]
        )
        # Enforce max_rules
        if len(child_rules) > self.max_rules:
            child_rules = list(rng.choice(child_rules, self.max_rules, replace=False))
        if not child_rules:
            child_rules = [_RuleGene(atoms=list(p1.rules[0].atoms))]

        return _Individual(rules=child_rules)

    def _mutate(
        self,
        ind: _Individual,
        specs: list[dict],
        rng: np.random.Generator,
    ) -> _Individual:
        if rng.random() >= self.mutation_rate:
            return ind

        ind = ind.clone()
        op = rng.random()

        if op < 0.15 and len(ind.rules) > 1:
            # Delete a random rule
            idx = int(rng.integers(0, len(ind.rules)))
            ind.rules.pop(idx)

        elif op < 0.30 and len(ind.rules) < self.max_rules:
            # Insert a new random rule
            ind.rules.append(self._random_rule(specs, rng))

        elif op < 0.50:
            # Replace an atom in a random rule
            ri = int(rng.integers(0, len(ind.rules)))
            rule = ind.rules[ri]
            if rule.atoms:
                ai = int(rng.integers(0, len(rule.atoms)))
                rule.atoms[ai] = self._random_atom(specs, rng)

        elif op < 0.65:
            # Add an atom to a random rule
            ri = int(rng.integers(0, len(ind.rules)))
            rule = ind.rules[ri]
            if len(rule.atoms) < self.max_atoms_per_rule:
                rule.atoms.append(self._random_atom(specs, rng))

        elif op < 0.80:
            # Delete an atom from a random rule
            ri = int(rng.integers(0, len(ind.rules)))
            rule = ind.rules[ri]
            if len(rule.atoms) > 1:
                ai = int(rng.integers(0, len(rule.atoms)))
                rule.atoms.pop(ai)

        else:
            # Swap a rule between two positions (structural diversity)
            if len(ind.rules) >= 2:
                i, j = rng.choice(len(ind.rules), size=2, replace=False)
                ind.rules[i], ind.rules[j] = ind.rules[j], ind.rules[i]

        return ind

    # ------------------------------------------------------------------
    # Post-hoc compaction
    # ------------------------------------------------------------------

    def _compact(
        self,
        ind: _Individual,
        X: np.ndarray,
        y: np.ndarray,
        n_classes: int,
    ) -> _Individual:
        """Backward elimination of rules and atoms."""
        ind = ind.clone()
        self._refit_scores(ind, X, y, n_classes)
        self._evaluate(ind, X, y, n_classes)
        base_f1 = ind.fitness_f1

        # Phase 1: Remove entire rules
        changed = True
        while changed and len(ind.rules) > 1:
            changed = False
            for ri in range(len(ind.rules) - 1, -1, -1):
                candidate = ind.clone()
                candidate.rules.pop(ri)
                self._refit_scores(candidate, X, y, n_classes)
                self._evaluate(candidate, X, y, n_classes)
                if candidate.fitness_f1 >= base_f1 - 1e-6:
                    ind = candidate
                    base_f1 = candidate.fitness_f1
                    changed = True
                    break

        # Phase 2: Remove individual atoms from rules
        changed = True
        while changed:
            changed = False
            for ri in range(len(ind.rules)):
                rule = ind.rules[ri]
                if len(rule.atoms) <= 1:
                    continue
                for ai in range(len(rule.atoms) - 1, -1, -1):
                    candidate = ind.clone()
                    candidate.rules[ri].atoms.pop(ai)
                    self._refit_scores(candidate, X, y, n_classes)
                    self._evaluate(candidate, X, y, n_classes)
                    if candidate.fitness_f1 >= base_f1 - 1e-6:
                        ind = candidate
                        base_f1 = candidate.fitness_f1
                        changed = True
                        break
                if changed:
                    break

        return ind

    # ------------------------------------------------------------------
    # Model selection from final population
    # ------------------------------------------------------------------

    def _select_best(self, pop: list[_Individual]) -> _Individual:
        """Select the best individual: highest F1, ties broken by smallest model."""
        fronts = _fast_nondominated_sort(pop)
        front0 = [pop[i] for i in fronts[0]]

        # Among the Pareto front: pick the one with best F1
        # If multiple have (near-)identical F1, prefer fewer atoms.
        return max(
            front0,
            key=lambda ind: (
                ind.fitness_f1,
                ind.fitness_other_class_f1,
                -ind.fitness_complexity,
            ),
        )

    # ------------------------------------------------------------------
    # Mask computation
    # ------------------------------------------------------------------

    def _atom_mask(self, atom: _AtomGene, X: np.ndarray) -> np.ndarray:
        col = X[:, atom.feature_idx]
        if atom.op == "<=":
            return np.asarray(col, dtype=float) <= float(atom.value)
        elif atom.op == ">":
            return np.asarray(col, dtype=float) > float(atom.value)
        elif atom.op == "between":
            lo, hi = atom.value
            c = np.asarray(col, dtype=float)
            return (c >= float(lo)) & (c <= float(hi))
        elif atom.op == "==":
            # For numeric low-cardinality features, compare as float;
            # for truly categorical features, compare as object.
            try:
                return np.asarray(col, dtype=float) == float(atom.value)
            except (ValueError, TypeError):
                return np.asarray(col, dtype=object) == atom.value
        elif atom.op == "in":
            return np.isin(np.asarray(col, dtype=object), list(atom.value))
        return np.ones(X.shape[0], dtype=bool)

    def _rule_mask(self, rule: _RuleGene, X: np.ndarray) -> np.ndarray:
        mask = np.ones(X.shape[0], dtype=bool)
        for atom in rule.atoms:
            mask &= self._atom_mask(atom, X)
        return mask

    # ------------------------------------------------------------------
    # Train / val split
    # ------------------------------------------------------------------

    def _stratified_split(
        self, y_idx: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray | None]:
        n = y_idx.shape[0]
        if n < 10 or self.validation_fraction <= 0:
            return np.arange(n), None

        train_parts, val_parts = [], []
        for c in np.unique(y_idx):
            ci = np.where(y_idx == c)[0]
            if ci.size <= 1:
                train_parts.append(ci)
                continue
            perm = ci[rng.permutation(ci.size)]
            nv = max(1, min(ci.size - 1, int(round(ci.size * self.validation_fraction))))
            val_parts.append(perm[:nv])
            train_parts.append(perm[nv:])

        tr = np.concatenate(train_parts) if train_parts else np.arange(n)
        if not val_parts:
            return tr, None
        va = np.concatenate(val_parts)
        return (tr, va) if va.size > 0 else (tr, None)

    # ------------------------------------------------------------------
    # Convert to ScoredRuleSet
    # ------------------------------------------------------------------

    def _to_ruleset(
        self,
        ind: _Individual,
        X: np.ndarray,
        y: np.ndarray,
        n_classes: int,
        generations_ran: int,
    ) -> ScoredRuleSet:
        rules: list[Rule] = []
        class_counts = np.bincount(y, minlength=n_classes).astype(float)
        class_counts = np.maximum(class_counts, 1.0)

        for ri, gene in enumerate(ind.rules):
            mask = self._rule_mask(gene, X)
            if mask.sum() < 1:
                continue
            counts = np.bincount(y[mask], minlength=n_classes).astype(float)
            balanced = counts / class_counts
            bal_total = balanced.sum()
            scores = (balanced / bal_total).tolist() if bal_total > 0 else [1.0 / n_classes] * n_classes
            atoms = [
                Atom(
                    feature=str(self.feature_names_in_[a.feature_idx]),
                    op=a.op,
                    value=self._serialize_atom_value(a),
                )
                for a in gene.atoms
            ]
            rules.append(Rule(
                atoms=atoms,
                scores=scores,
                rule_id=f"rulensga2_rule_{ri}",
                metadata={"source": "rulensga2", "support": int(mask.sum())},
            ))

        # Default rule
        any_fired = np.zeros(X.shape[0], dtype=bool)
        for gene in ind.rules:
            any_fired |= self._rule_mask(gene, X)
        no_fire = ~any_fired
        if no_fire.any():
            counts = np.bincount(y[no_fire], minlength=n_classes).astype(float)
            balanced = counts / class_counts
            bal_total = balanced.sum()
            default_scores = (balanced / bal_total).tolist() if bal_total > 0 else [1.0 / n_classes] * n_classes
        else:
            default_scores = [1.0 / n_classes] * n_classes
        rules.append(Rule(
            atoms=[],
            scores=default_scores,
            rule_id="rulensga2_default",
            metadata={"source": "rulensga2", "kind": "default"},
        ))

        return ScoredRuleSet(
            class_labels=self.classes_.tolist(),
            feature_names=self.feature_names_in_.tolist(),
            aggregation=AggregationSpec(type="argmax_sum", temperature=1.0),
            rules=rules,
            metadata={
                "source": "rulensga2",
                "model_type": "genetic_programming_ruleset",
                "population_size": self.population_size,
                "generations": self.generations,
                "generations_ran": generations_ran,
                "enable_compaction": self.enable_compaction,
                "min_max_weight": self.min_max_weight,
                "atom_space_strategy": self.atom_space_strategy,
                "atom_preselection_strategy": self.atom_preselection_strategy,
                "atom_preselection_top_k": self.atom_preselection_top_k,
                "max_rules": self.max_rules,
                "max_atoms_per_rule": self.max_atoms_per_rule,
            },
        )

    @staticmethod
    def _serialize_atom_value(atom: _AtomGene) -> object:
        if atom.op == "between":
            return [float(atom.value[0]), float(atom.value[1])]
        if atom.op == "in":
            return list(atom.value)
        if isinstance(atom.value, (int, float, np.integer, np.floating)):
            return float(atom.value)
        return atom.value


