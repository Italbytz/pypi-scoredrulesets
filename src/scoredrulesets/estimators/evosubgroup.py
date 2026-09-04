"""
RuleEvoSubgroup - Evolutionary Subgroup Discovery and Exceptional Model Mining
==============================================================================

Evolves sets of rules using NSGA-II to explicitly explore the Pareto frontier
between subgroup quality (coverage * effect size) and model complexity (total atom count).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted

from ..schema import AggregationSpec, Atom, Rule, ScoredRuleSet
from ._time_budget import (
    FitBudgetExceededError,
    deadline_reached,
    resolve_deadline,
)
from .atom_space import (
    ContinuousThresholdStrategy,
    NativeAtomSpaceStrategy,
    build_native_feature_specs,
)
from .base import BaseRuleSetEstimator
from .rulegp_regressor import _AtomGeneReg as _AtomGene, _RuleReg as _Rule


@dataclass
class _RuleSetSubgroup:
    rules: list[_Rule]
    
    def clone(self) -> "_RuleSetSubgroup":
        return _RuleSetSubgroup(
            rules=[_Rule(atoms=list(r.atoms)) for r in self.rules]
        )

@dataclass
class _FitnessSubgroup:
    quality: float
    size: int

    def dominates(self, other: "_FitnessSubgroup") -> bool:
        # Maximize quality, minimize size
        q_ge = self.quality >= other.quality - 1e-6
        sz_le = self.size <= other.size
        q_gt = self.quality > other.quality + 1e-6
        sz_lt = self.size < other.size
        return q_ge and sz_le and (q_gt or sz_lt)


@dataclass
class _NSGA2IndividualSubgroup:
    ruleset: _RuleSetSubgroup
    fitness: _FitnessSubgroup = field(default_factory=lambda: _FitnessSubgroup(quality=-999999.0, size=999999))
    rank: int = 0
    crowding_dist: float = 0.0


def _atom_mask(atom: _AtomGene, X: np.ndarray) -> np.ndarray:
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
        try:
            return np.asarray(col, dtype=float) == float(atom.value)
        except (ValueError, TypeError):
            return np.asarray(col, dtype=object) == atom.value
    elif atom.op == "in":
        return np.isin(np.asarray(col, dtype=object), list(atom.value))
    return np.ones(X.shape[0], dtype=bool)


def _rule_mask(rule: _Rule, X: np.ndarray) -> np.ndarray:
    mask = np.ones(X.shape[0], dtype=bool)
    for atom in rule.atoms:
        mask &= _atom_mask(atom, X)
    return mask


def _evaluate_fitness_subgroup(
    rs: _RuleSetSubgroup, 
    X: np.ndarray, 
    y: np.ndarray, 
    global_mean: float, 
    gamma: float
) -> _FitnessSubgroup:
    n = X.shape[0]
    total_quality = 0.0
    
    # Track overlapping coverage
    coverage_counts = np.zeros(n, dtype=int)
    
    for r in rs.rules:
        mask = _rule_mask(r, X)
        coverage_counts[mask] += 1
        n_r = mask.sum()
        
        if n_r > 0:
            local_mean = float(np.mean(y[mask]))
            # Subgroup Quality: (coverage^gamma) * absolute deviation from global mean
            c_r = n_r / n
            q_r = (c_r ** gamma) * abs(local_mean - global_mean)
            total_quality += q_r
            r.weight = local_mean # store local mean in weight for later export
            r.quality = q_r # store quality
        else:
            r.weight = global_mean
            r.quality = 0.0
            
    # Overlap penalty: heavily penalize rules covering the same instances
    # If a sample is covered by C rules, we penalize C-1
    overlap_sum = np.maximum(0, coverage_counts - 1).sum()
    overlap_penalty = (overlap_sum / n) * abs(global_mean) * 0.5 # scale penalty relative to data
    
    final_quality = total_quality - overlap_penalty
    
    size = sum(len(r.atoms) for r in rs.rules)
    return _FitnessSubgroup(quality=float(final_quality), size=size)


def _non_dominated_sort_subgroup(pop: list[_NSGA2IndividualSubgroup]) -> list[list[_NSGA2IndividualSubgroup]]:
    fronts: list[list[_NSGA2IndividualSubgroup]] = [[]]
    domination_counts = [0] * len(pop)
    dominated_sets: list[list[int]] = [[] for _ in range(len(pop))]

    for i, p in enumerate(pop):
        for j, q in enumerate(pop):
            if i == j:
                continue
            if p.fitness.dominates(q.fitness):
                dominated_sets[i].append(j)
            elif q.fitness.dominates(p.fitness):
                domination_counts[i] += 1

        if domination_counts[i] == 0:
            p.rank = 0
            fronts[0].append(p)

    curr_rank = 0
    while len(fronts[curr_rank]) > 0:
        next_front: list[_NSGA2IndividualSubgroup] = []
        for p in fronts[curr_rank]:
            p_idx = pop.index(p)
            for q_idx in dominated_sets[p_idx]:
                domination_counts[q_idx] -= 1
                if domination_counts[q_idx] == 0:
                    pop[q_idx].rank = curr_rank + 1
                    next_front.append(pop[q_idx])
        curr_rank += 1
        fronts.append(next_front)

    if not fronts[-1]:
        fronts.pop()
    return fronts


def _assign_crowding_distance_subgroup(front: list[_NSGA2IndividualSubgroup]) -> None:
    l = len(front)
    if l == 0:
        return
    for ind in front:
        ind.crowding_dist = 0.0

    if l <= 2:
        for ind in front:
            ind.crowding_dist = float("inf")
        return

    # Objective 1: Quality (Maximize -> sort ascending)
    front.sort(key=lambda ind: ind.fitness.quality)
    front[0].crowding_dist = float("inf")
    front[-1].crowding_dist = float("inf")
    q_range = front[-1].fitness.quality - front[0].fitness.quality
    if q_range > 1e-12:
        for i in range(1, l - 1):
            front[i].crowding_dist += (front[i + 1].fitness.quality - front[i - 1].fitness.quality) / q_range

    # Objective 2: Size (Minimize -> sort ascending)
    front.sort(key=lambda ind: ind.fitness.size)
    front[0].crowding_dist = float("inf")
    front[-1].crowding_dist = float("inf")
    s_range = float(front[-1].fitness.size - front[0].fitness.size)
    if s_range > 1e-12:
        for i in range(1, l - 1):
            front[i].crowding_dist += (front[i + 1].fitness.size - front[i - 1].fitness.size) / s_range


class RuleNSGA2Subgroup(BaseRuleSetEstimator):
    """Multi-Objective NSGA-II Genetic Programming for Subgroup Discovery / Exceptional Model Mining.
    
    Extracts k distinct subgroups that maximize the Pareto trade-off between subgroup quality (coverage * effect size)
    and model complexity (total atom count).
    """

    def __init__(
        self,
        n_subgroups: int = 5,
        gamma: float = 0.5,
        max_generations: int = 50,
        population_size: int = 50,
        n_offspring: int = 30,
        tournament_size: int = 3,
        max_atoms_per_rule: int | None = 4,
        atom_space_strategy: NativeAtomSpaceStrategy = "hybrid",
        continuous_threshold_strategy: ContinuousThresholdStrategy = "quantile_midpoint",
        max_fit_seconds: float | None = None,
        feature_names: list[str] | None = None,
        random_state: int | None = None,
    ):
        self.n_subgroups = n_subgroups
        self.gamma = gamma
        self.max_generations = max_generations
        self.population_size = population_size
        self.n_offspring = n_offspring
        self.tournament_size = tournament_size
        self.max_atoms_per_rule = max_atoms_per_rule
        self.atom_space_strategy = atom_space_strategy
        self.continuous_threshold_strategy = continuous_threshold_strategy
        self.max_fit_seconds = max_fit_seconds
        self.feature_names = feature_names
        self.random_state = random_state

    def fit(self, X, y):
        X_arr = np.asarray(check_array(X, dtype=None))
        y_arr = np.asarray(y, dtype=float).ravel()
        
        self.n_features_in_ = X_arr.shape[1]
        self.feature_names_in_ = self.feature_names or [f"f{i}" for i in range(self.n_features_in_)]
        
        global_mean = float(np.mean(y_arr))
        self.global_mean_ = global_mean

        rng = np.random.default_rng(self.random_state)
        deadline = resolve_deadline(self.max_fit_seconds)

        # 1. Atom Space
        specs = build_native_feature_specs(
            X_arr,
            strategy=self.atom_space_strategy,
            continuous_threshold_strategy=self.continuous_threshold_strategy,
        )

        available_atoms: list[_AtomGene] = []
        for feat_idx, spec in enumerate(specs):
            for op in spec.get("operators", ["<=", ">"]):
                for val in spec.get("thresholds", []):
                    available_atoms.append(_AtomGene(feature_idx=feat_idx, op=op, value=val))
            for val in spec.get("categories", []):
                available_atoms.append(_AtomGene(feature_idx=feat_idx, op="==", value=val))

        if not available_atoms:
            available_atoms.append(_AtomGene(feature_idx=0, op="<=", value=0.0))

        # 2. Initial Population
        pop_size = max(int(self.population_size), 10)
        pop: list[_NSGA2IndividualSubgroup] = []

        for _ in range(pop_size):
            rules = []
            for _ in range(self.n_subgroups):
                n_a = rng.integers(1, min(3, len(available_atoms)) + 1)
                chosen = [rng.choice(available_atoms) for _ in range(n_a)]
                rules.append(_Rule(atoms=chosen))
            rs = _RuleSetSubgroup(rules=rules)
            fit = _evaluate_fitness_subgroup(rs, X_arr, y_arr, global_mean, self.gamma)
            pop.append(_NSGA2IndividualSubgroup(ruleset=rs, fitness=fit))

        # 3. Evolution Loop
        for gen in range(self.max_generations):
            if deadline_reached(deadline):
                break

            # Create offspring
            offspring: list[_NSGA2IndividualSubgroup] = []
            for _ in range(self.n_offspring):
                p1 = self._tournament_select(pop, rng)
                child_rs = p1.ruleset.clone()

                # Mutation
                mut_type = rng.choice(["add_atom", "drop_atom", "modify_atom"])
                max_a = self.max_atoms_per_rule or 4

                if mut_type == "add_atom" and child_rs.rules:
                    r_idx = rng.integers(0, len(child_rs.rules))
                    if len(child_rs.rules[r_idx].atoms) < max_a:
                        new_atom = rng.choice(available_atoms)
                        child_rs.rules[r_idx].atoms.append(new_atom)
                elif mut_type == "drop_atom" and child_rs.rules:
                    r_idx = rng.integers(0, len(child_rs.rules))
                    if len(child_rs.rules[r_idx].atoms) > 1:
                        a_idx = rng.integers(0, len(child_rs.rules[r_idx].atoms))
                        child_rs.rules[r_idx].atoms.pop(a_idx)
                elif mut_type == "modify_atom" and child_rs.rules:
                    r_idx = rng.integers(0, len(child_rs.rules))
                    if child_rs.rules[r_idx].atoms:
                        a_idx = rng.integers(0, len(child_rs.rules[r_idx].atoms))
                        child_rs.rules[r_idx].atoms[a_idx] = rng.choice(available_atoms)

                c_fit = _evaluate_fitness_subgroup(child_rs, X_arr, y_arr, global_mean, self.gamma)
                offspring.append(_NSGA2IndividualSubgroup(ruleset=child_rs, fitness=c_fit))

            # Combine and NSGA-II Select
            combined = pop + offspring
            fronts = _non_dominated_sort_subgroup(combined)
            new_pop: list[_NSGA2IndividualSubgroup] = []

            for front in fronts:
                _assign_crowding_distance_subgroup(front)
                if len(new_pop) + len(front) <= pop_size:
                    new_pop.extend(front)
                else:
                    front.sort(key=lambda ind: ind.crowding_dist, reverse=True)
                    needed = pop_size - len(new_pop)
                    new_pop.extend(front[:needed])
                    break
            pop = new_pop

        # Select best balanced knee-point solution on Pareto rank 0
        front_0 = [ind for ind in pop if ind.rank == 0]
        if not front_0:
            front_0 = pop
            
        # Select individual with best combination of Quality and compactness
        best_ind = max(front_0, key=lambda ind: ind.fitness.quality - 0.05 * ind.fitness.size)
        
        self.ruleset_internal_ = best_ind.ruleset
        self.ruleset_ = self._to_schema_ruleset(self.ruleset_internal_, X_arr)
        return self

    def _tournament_select(self, pop: list[_NSGA2IndividualSubgroup], rng: np.random.Generator) -> _NSGA2IndividualSubgroup:
        k = min(self.tournament_size, len(pop))
        cand_indices = rng.choice(len(pop), size=k, replace=False)
        candidates = [pop[i] for i in cand_indices]
        
        winner = candidates[0]
        for c in candidates[1:]:
            if c.rank < winner.rank:
                winner = c
            elif c.rank == winner.rank and c.crowding_dist > winner.crowding_dist:
                winner = c
        return winner

    def _to_schema_ruleset(self, rs: _RuleSetSubgroup, X: np.ndarray) -> ScoredRuleSet:
        schema_rules: list[Rule] = []
        n = X.shape[0]
        
        # Sort rules by quality descending
        sorted_rules = sorted(rs.rules, key=lambda r: getattr(r, "quality", 0.0), reverse=True)
        
        for idx, rule in enumerate(sorted_rules):
            schema_atoms: list[Atom] = []
            for a in rule.atoms:
                feat_name = self.feature_names_in_[a.feature_idx]
                val = float(a.value) if isinstance(a.value, (int, float, np.number)) else a.value
                schema_atoms.append(Atom(feature=feat_name, op=a.op, value=val))
                
            local_mean = getattr(rule, "weight", 0.0)
            quality = getattr(rule, "quality", 0.0)
            
            mask = _rule_mask(rule, X)
            coverage = mask.sum() / n if n > 0 else 0.0
            
            schema_rules.append(
                Rule(
                    atoms=schema_atoms, 
                    scores=[float(local_mean)], 
                    rule_id=f"subgroup_{idx+1}",
                    metadata={
                        "quality": float(quality), 
                        "coverage": float(coverage),
                        "local_mean": float(local_mean),
                        "global_mean": float(self.global_mean_)
                    }
                )
            )

        ruleset = ScoredRuleSet(
            class_labels=[],
            task_type="regression", 
            feature_names=list(self.feature_names_in_),
            rules=schema_rules,
            aggregation=AggregationSpec(type="independent"),
            metadata={"estimator": "RuleNSGA2Subgroup", "rules_count": len(schema_rules)},
        )
        ruleset.validate()
        return ruleset

    def predict(self, X):
        raise NotImplementedError("RuleNSGA2Subgroup is an exploratory model for subgroup discovery, not a predictive model. Inspect .ruleset_ instead.")

    def to_ruleset(self) -> ScoredRuleSet:
        check_is_fitted(self, "ruleset_")
        return self.ruleset_


# Backward compatibility alias
RuleEvoSubgroup = RuleNSGA2Subgroup
