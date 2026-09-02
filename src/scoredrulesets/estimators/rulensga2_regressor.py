"""
RuleNSGA2Regressor - Multi-Objective Evolutionary Algorithm for Scored Rule Sets in Regression
==============================================================================================

Evolves complete scored rule sets using NSGA-II to explicitly explore the Pareto frontier
between prediction error (RMSE) and model complexity (total atom count).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ..runtime import predict_regression as predict_regression_from_ruleset
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
from .rulegp_regressor import (
    _AtomGeneReg,
    _RuleReg,
    _RuleSetReg,
    _compute_regression_weights,
    _evaluate_fitness_reg,
    _FitnessReg,
)


@dataclass(eq=False)
class _NSGA2IndividualReg:
    ruleset: _RuleSetReg
    fitness: _FitnessReg = field(default_factory=lambda: _FitnessReg(rmse=999999.0, size=999999, r2=-999.0))
    rank: int = 0
    crowding_dist: float = 0.0


def _non_dominated_sort_reg(pop: list[_NSGA2IndividualReg]) -> list[list[_NSGA2IndividualReg]]:
    fronts: list[list[_NSGA2IndividualReg]] = [[]]
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
        next_front: list[_NSGA2IndividualReg] = []
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


def _assign_crowding_distance_reg(front: list[_NSGA2IndividualReg]) -> None:
    l = len(front)
    if l == 0:
        return
    for ind in front:
        ind.crowding_dist = 0.0

    if l <= 2:
        for ind in front:
            ind.crowding_dist = float("inf")
        return

    # Objective 1: RMSE
    front.sort(key=lambda ind: ind.fitness.rmse)
    front[0].crowding_dist = float("inf")
    front[-1].crowding_dist = float("inf")
    r_range = front[-1].fitness.rmse - front[0].fitness.rmse
    if r_range > 1e-12:
        for i in range(1, l - 1):
            front[i].crowding_dist += (front[i + 1].fitness.rmse - front[i - 1].fitness.rmse) / r_range

    # Objective 2: Size
    front.sort(key=lambda ind: ind.fitness.size)
    front[0].crowding_dist = float("inf")
    front[-1].crowding_dist = float("inf")
    s_range = float(front[-1].fitness.size - front[0].fitness.size)
    if s_range > 1e-12:
        for i in range(1, l - 1):
            front[i].crowding_dist += (front[i + 1].fitness.size - front[i - 1].fitness.size) / s_range


class RuleNSGA2Regressor(BaseRuleSetEstimator, RegressorMixin):
    """Multi-Objective NSGA-II Genetic Programming for Regression Scored Rule Sets."""

    def __init__(
        self,
        max_generations: int = 150,
        population_size: int = 80,
        n_offspring: int = 40,
        tournament_size: int = 3,
        max_rules: int | None = 10,
        max_atoms_per_rule: int | None = 4,
        atom_space_strategy: NativeAtomSpaceStrategy = "hybrid",
        continuous_threshold_strategy: ContinuousThresholdStrategy = "quantile_midpoint",
        prediction_type: Literal["constant", "linear"] = "constant",
        max_fit_seconds: float | None = None,
        feature_names: list[str] | None = None,
        random_state: int | None = None,
    ):
        self.max_generations = max_generations
        self.population_size = population_size
        self.n_offspring = n_offspring
        self.tournament_size = tournament_size
        self.max_rules = max_rules
        self.max_atoms_per_rule = max_atoms_per_rule
        self.atom_space_strategy = atom_space_strategy
        self.continuous_threshold_strategy = continuous_threshold_strategy
        self.prediction_type = prediction_type
        self.max_fit_seconds = max_fit_seconds
        self.feature_names = feature_names
        self.random_state = random_state

    def fit(self, X, y):
        X_arr, y_arr = check_X_y(X, y, dtype=None, y_numeric=True)
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

        available_atoms: list[_AtomGeneReg] = []
        for feat_idx, spec in enumerate(specs):
            for op in spec.get("operators", ["<=", ">"]):
                for val in spec.get("thresholds", []):
                    available_atoms.append(_AtomGeneReg(feature_idx=feat_idx, op=op, value=val))
            for val in spec.get("categories", []):
                available_atoms.append(_AtomGeneReg(feature_idx=feat_idx, op="==", value=val))

        if not available_atoms:
            available_atoms.append(_AtomGeneReg(feature_idx=0, op="<=", value=0.0))

        # 2. Initial Population
        pop_size = max(int(self.population_size), 10)
        pop: list[_NSGA2IndividualReg] = []

        for _ in range(pop_size):
            n_r = rng.integers(1, 4)
            rules = []
            for _ in range(n_r):
                n_a = rng.integers(1, min(3, len(available_atoms)) + 1)
                chosen = [rng.choice(available_atoms) for _ in range(n_a)]
                rules.append(_RuleReg(atoms=chosen))
            rs = _RuleSetReg(rules=rules, prediction_type=self.prediction_type)
            _compute_regression_weights(rs, X_arr, y_arr)
            fit = _evaluate_fitness_reg(rs, X_arr, y_arr)
            pop.append(_NSGA2IndividualReg(ruleset=rs, fitness=fit))

        # 3. Evolution Loop
        for gen in range(self.max_generations):
            if deadline_reached(deadline):
                break

            # Create offspring
            offspring: list[_NSGA2IndividualReg] = []
            for _ in range(self.n_offspring):
                p1 = self._tournament_select(pop, rng)
                child_rs = p1.ruleset.clone()

                # Mutation
                mut_type = rng.choice(["add_rule", "drop_rule", "add_atom", "drop_atom", "modify_atom"])
                max_r = self.max_rules or 10
                max_a = self.max_atoms_per_rule or 4

                if mut_type == "add_rule" and len(child_rs.rules) < max_r:
                    new_atom = rng.choice(available_atoms)
                    child_rs.rules.append(_RuleReg(atoms=[new_atom]))
                elif mut_type == "drop_rule" and len(child_rs.rules) > 1:
                    r_idx = rng.integers(0, len(child_rs.rules))
                    child_rs.rules.pop(r_idx)
                elif mut_type == "add_atom" and child_rs.rules:
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

                _compute_regression_weights(child_rs, X_arr, y_arr)
                c_fit = _evaluate_fitness_reg(child_rs, X_arr, y_arr)
                offspring.append(_NSGA2IndividualReg(ruleset=child_rs, fitness=c_fit))

            # Combine and NSGA-II Select
            combined = pop + offspring
            fronts = _non_dominated_sort_reg(combined)
            new_pop: list[_NSGA2IndividualReg] = []

            for front in fronts:
                _assign_crowding_distance_reg(front)
                if len(new_pop) + len(front) <= pop_size:
                    new_pop.extend(front)
                else:
                    # Sort by crowding distance and take remaining
                    front.sort(key=lambda ind: ind.crowding_dist, reverse=True)
                    needed = pop_size - len(new_pop)
                    new_pop.extend(front[:needed])
                    break
            pop = new_pop

        # Select best balanced knee-point solution on Pareto rank 0
        front_0 = [ind for ind in pop if ind.rank == 0]
        if not front_0:
            front_0 = pop
        # Select individual with best combination of RMSE and compactness
        best_ind = min(front_0, key=lambda ind: ind.fitness.rmse * (1.0 + 0.02 * ind.fitness.size))
        
        self.ruleset_internal_ = best_ind.ruleset
        self.ruleset_ = self._to_schema_ruleset(self.ruleset_internal_)
        return self

    def _tournament_select(self, pop: list[_NSGA2IndividualReg], rng: np.random.Generator) -> _NSGA2IndividualReg:
        k = min(self.tournament_size, len(pop))
        cand_indices = rng.choice(len(pop), size=k, replace=False)
        candidates = [pop[i] for i in cand_indices]
        
        # Binary tournament: lower rank wins; if equal, higher crowding distance wins
        winner = candidates[0]
        for c in candidates[1:]:
            if c.rank < winner.rank:
                winner = c
            elif c.rank == winner.rank and c.crowding_dist > winner.crowding_dist:
                winner = c
        return winner

    def _to_schema_ruleset(self, rs: _RuleSetReg) -> ScoredRuleSet:
        schema_rules: list[Rule] = []
        for idx, rule in enumerate(rs.rules):
            schema_atoms: list[Atom] = []
            for a in rule.atoms:
                feat_name = self.feature_names_in_[a.feature_idx]
                schema_atoms.append(Atom(feature=feat_name, op=a.op, value=a.value))
            if self.prediction_type == "linear":
                b = getattr(rule, "beta", None)
                scores = list(b if b is not None else []) + [float(getattr(rule, "beta0", 0.0))]
            else:
                scores = [float(rule.weight)]

            schema_rules.append(
                Rule(
                    atoms=schema_atoms, 
                    scores=scores, 
                    rule_id=f"r_{idx}"
                )
            )

        if self.prediction_type == "linear":
            db = getattr(rs, "default_beta", None)
            def_scores = list(db if db is not None else []) + [float(getattr(rs, "default_beta0", 0.0))]
        else:
            def_scores = [float(rs.default_weight)]

        schema_rules.append(
            Rule(
                atoms=[], 
                scores=def_scores, 
                rule_id="default_reg"
            )
        )

        ruleset = ScoredRuleSet(
            class_labels=[],
            task_type="regression",
            feature_names=list(self.feature_names_in_),
            rules=schema_rules,
            aggregation=AggregationSpec(type="takagi_sugeno" if self.prediction_type == "linear" else "mean_active"),
            metadata={"estimator": "RuleNSGA2Regressor", "prediction_type": self.prediction_type, "rules_count": len(schema_rules)},
        )
        ruleset.validate()
        return ruleset

    def predict(self, X):
        check_is_fitted(self, "ruleset_")
        X_arr = check_array(X, dtype=None)
        return predict_regression_from_ruleset(self.ruleset_, X_arr)

    def score(self, X, y, sample_weight=None):
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X), sample_weight=sample_weight)

    def to_ruleset(self) -> ScoredRuleSet:
        check_is_fitted(self, "ruleset_")
        return self.ruleset_
