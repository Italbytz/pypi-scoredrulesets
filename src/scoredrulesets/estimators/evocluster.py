"""
RuleEvoCluster - Unsupervised Evolutionary Algorithm for Scored Rule Sets
=========================================================================

Evolves complete scored rule sets using NSGA-II to explicitly explore the Pareto frontier
between clustering quality (Silhouette Score) and model complexity (total atom count).
This provides a native, interpretable-by-design clustering approach.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import ClusterMixin
from sklearn.metrics import silhouette_score
from sklearn.utils.validation import check_array, check_is_fitted

from ..runtime import predict as predict_from_ruleset
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
class _RuleSetCluster:
    rules: list[_Rule]
    
    def clone(self) -> "_RuleSetCluster":
        return _RuleSetCluster(
            rules=[_Rule(atoms=list(r.atoms)) for r in self.rules]
        )

@dataclass
class _FitnessCluster:
    silhouette: float
    size: int

    def dominates(self, other: "_FitnessCluster") -> bool:
        sil_ge = self.silhouette >= other.silhouette - 1e-6
        sz_le = self.size <= other.size
        sil_gt = self.silhouette > other.silhouette + 1e-6
        sz_lt = self.size < other.size
        return sil_ge and sz_le and (sil_gt or sz_lt)


@dataclass
class _NSGA2IndividualCluster:
    ruleset: _RuleSetCluster
    fitness: _FitnessCluster = field(default_factory=lambda: _FitnessCluster(silhouette=-1.0, size=999999))
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


def _evaluate_fitness_cluster(rs: _RuleSetCluster, X: np.ndarray, n_clusters: int) -> _FitnessCluster:
    n = X.shape[0]
    scores = np.zeros((n, n_clusters))
    
    # We assign each rule to a specific cluster index
    for i, r in enumerate(rs.rules):
        cluster_idx = i % n_clusters
        mask = _rule_mask(r, X)
        scores[mask, cluster_idx] += 1.0
        
    # Uncovered instances get assigned to a default cluster (0)
    uncovered = scores.sum(axis=1) == 0
    if uncovered.any():
        scores[uncovered, 0] += 0.1  # slightly bias uncovered to 0 to avoid random argmax
        
    preds = np.argmax(scores, axis=1)
    
    n_labels = len(np.unique(preds))
    if 1 < n_labels < n:
        if n > 2000:
            sil = float(silhouette_score(X, preds, sample_size=2000, random_state=42))
        else:
            sil = float(silhouette_score(X, preds))
    else:
        sil = -1.0
        
    size = sum(len(r.atoms) for r in rs.rules)
    return _FitnessCluster(silhouette=sil, size=size)


def _non_dominated_sort_cluster(pop: list[_NSGA2IndividualCluster]) -> list[list[_NSGA2IndividualCluster]]:
    fronts: list[list[_NSGA2IndividualCluster]] = [[]]
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
        next_front: list[_NSGA2IndividualCluster] = []
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


def _assign_crowding_distance_cluster(front: list[_NSGA2IndividualCluster]) -> None:
    l = len(front)
    if l == 0:
        return
    for ind in front:
        ind.crowding_dist = 0.0

    if l <= 2:
        for ind in front:
            ind.crowding_dist = float("inf")
        return

    # Objective 1: Silhouette (Maximize -> sort ascending)
    front.sort(key=lambda ind: ind.fitness.silhouette)
    front[0].crowding_dist = float("inf")
    front[-1].crowding_dist = float("inf")
    sil_range = front[-1].fitness.silhouette - front[0].fitness.silhouette
    if sil_range > 1e-12:
        for i in range(1, l - 1):
            front[i].crowding_dist += (front[i + 1].fitness.silhouette - front[i - 1].fitness.silhouette) / sil_range

    # Objective 2: Size (Minimize -> sort ascending)
    front.sort(key=lambda ind: ind.fitness.size)
    front[0].crowding_dist = float("inf")
    front[-1].crowding_dist = float("inf")
    s_range = float(front[-1].fitness.size - front[0].fitness.size)
    if s_range > 1e-12:
        for i in range(1, l - 1):
            front[i].crowding_dist += (front[i + 1].fitness.size - front[i - 1].fitness.size) / s_range


class RuleEvoCluster(BaseRuleSetEstimator, ClusterMixin):
    """Multi-Objective NSGA-II Genetic Programming for Unsupervised Rule-Based Clustering."""

    def __init__(
        self,
        n_clusters: int = 3,
        max_generations: int = 100,
        population_size: int = 50,
        n_offspring: int = 30,
        tournament_size: int = 3,
        max_rules: int | None = None,
        max_atoms_per_rule: int | None = 3,
        atom_space_strategy: NativeAtomSpaceStrategy = "hybrid",
        continuous_threshold_strategy: ContinuousThresholdStrategy = "quantile_midpoint",
        max_fit_seconds: float | None = None,
        feature_names: list[str] | None = None,
        random_state: int | None = None,
    ):
        self.n_clusters = n_clusters
        self.max_generations = max_generations
        self.population_size = population_size
        self.n_offspring = n_offspring
        self.tournament_size = tournament_size
        # By default, use one rule per cluster if not specified
        self.max_rules = max_rules if max_rules is not None else n_clusters
        self.max_atoms_per_rule = max_atoms_per_rule
        self.atom_space_strategy = atom_space_strategy
        self.continuous_threshold_strategy = continuous_threshold_strategy
        self.max_fit_seconds = max_fit_seconds
        self.feature_names = feature_names
        self.random_state = random_state

    def fit(self, X, y=None):
        X_arr = np.asarray(check_array(X, dtype=None))
        self.n_features_in_ = X_arr.shape[1]
        self.feature_names_in_ = self.feature_names or [f"f{i}" for i in range(self.n_features_in_)]

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
        pop: list[_NSGA2IndividualCluster] = []

        for _ in range(pop_size):
            n_r = self.max_rules
            rules = []
            for _ in range(n_r):
                n_a = rng.integers(1, min(3, len(available_atoms)) + 1)
                chosen = [rng.choice(available_atoms) for _ in range(n_a)]
                rules.append(_Rule(atoms=chosen))
            rs = _RuleSetCluster(rules=rules)
            fit = _evaluate_fitness_cluster(rs, X_arr, self.n_clusters)
            pop.append(_NSGA2IndividualCluster(ruleset=rs, fitness=fit))

        # 3. Evolution Loop
        for gen in range(self.max_generations):
            if deadline_reached(deadline):
                break

            # Create offspring
            offspring: list[_NSGA2IndividualCluster] = []
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

                c_fit = _evaluate_fitness_cluster(child_rs, X_arr, self.n_clusters)
                offspring.append(_NSGA2IndividualCluster(ruleset=child_rs, fitness=c_fit))

            # Combine and NSGA-II Select
            combined = pop + offspring
            fronts = _non_dominated_sort_cluster(combined)
            new_pop: list[_NSGA2IndividualCluster] = []

            for front in fronts:
                _assign_crowding_distance_cluster(front)
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
            
        # Select individual with best combination of Silhouette and compactness
        # Higher silhouette is better, lower size is better
        # We maximize: silhouette - lambda * size
        best_ind = max(front_0, key=lambda ind: ind.fitness.silhouette - 0.01 * ind.fitness.size)
        
        self.ruleset_internal_ = best_ind.ruleset
        self.ruleset_ = self._to_schema_ruleset(self.ruleset_internal_)
        
        # Set cluster labels like a standard sklearn Clusterer
        self.labels_ = self.predict(X_arr)
        return self

    def _tournament_select(self, pop: list[_NSGA2IndividualCluster], rng: np.random.Generator) -> _NSGA2IndividualCluster:
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

    def _to_schema_ruleset(self, rs: _RuleSetCluster) -> ScoredRuleSet:
        schema_rules: list[Rule] = []
        for idx, rule in enumerate(rs.rules):
            schema_atoms: list[Atom] = []
            for a in rule.atoms:
                feat_name = self.feature_names_in_[a.feature_idx]
                val = float(a.value) if isinstance(a.value, (int, float, np.number)) else a.value
                schema_atoms.append(Atom(feature=feat_name, op=a.op, value=val))
                
            # Score vector: one-hot for the cluster this rule represents
            cluster_idx = idx % self.n_clusters
            scores = [0.0] * self.n_clusters
            scores[cluster_idx] = 1.0
            
            schema_rules.append(
                Rule(atoms=schema_atoms, scores=scores, rule_id=f"cluster_{cluster_idx}_rule_{idx}")
            )

        # Default rule
        default_scores = [1.0 / self.n_clusters] * self.n_clusters
        schema_rules.append(
            Rule(atoms=[], scores=default_scores, rule_id="default_cluster", metadata={"kind": "default"})
        )

        class_labels = [f"Cluster_{i}" for i in range(self.n_clusters)]
        ruleset = ScoredRuleSet(
            class_labels=class_labels,
            task_type="classification", # We use classification task type to output discrete labels
            feature_names=list(self.feature_names_in_),
            rules=schema_rules,
            aggregation=AggregationSpec(type="argmax_sum"),
            metadata={"estimator": "RuleEvoCluster", "rules_count": len(schema_rules)},
        )
        ruleset.validate()
        return ruleset

    def predict(self, X):
        check_is_fitted(self, "ruleset_")
        X_arr = check_array(X, dtype=None)
        # We output discrete indices from 0 to n_clusters-1
        # The runtime predict returns string labels if class_labels are set,
        # but for clusterers we usually want integer indices.
        # Let's get probas and argmax them to get indices.
        from ..runtime import predict_proba
        probas = predict_proba(self.ruleset_, X_arr)
        return np.argmax(probas, axis=1)

    def to_ruleset(self) -> ScoredRuleSet:
        check_is_fitted(self, "ruleset_")
        return self.ruleset_
