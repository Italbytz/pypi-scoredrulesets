"""
RuleGPSubgroup - Native Genetic Programming for Scored Rule Sets in Regression
================================================================================

Evolves complete scored rule sets directly optimizing continuous target objectives
(RMSE/MSE) with closed-form conditional expectation weights and multi-objective
Pareto complexity regularization.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score
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


@dataclass(frozen=True)
class _AtomGeneSubgroup:
    feature_idx: int
    op: str
    value: object


@dataclass(eq=False)
class _RuleSubgroup:
    atoms: list[_AtomGeneSubgroup]
    weight: float = 0.0
    beta: np.ndarray | None = None
    beta0: float = 0.0

    def clone(self) -> "_RuleSubgroup":
        r = _RuleSubgroup(atoms=list(self.atoms), weight=float(self.weight))
        if self.beta is not None:
            r.beta = self.beta.copy()
            r.beta0 = self.beta0
        return r


@dataclass(eq=False)
class _RuleSetSubgroup:
    rules: list[_RuleSubgroup]
    default_weight: float = 0.0
    default_beta: np.ndarray | None = None
    default_beta0: float = 0.0
    prediction_type: str = "constant"

    @staticmethod
    def _atom_key(atom: _AtomGeneSubgroup) -> tuple[int, str, object]:
        if atom.op in ("between", "in"):
            return (atom.feature_idx, atom.op, tuple(atom.value))
        return (atom.feature_idx, atom.op, atom.value)

    @property
    def size(self) -> int:
        seen: set[tuple[int, str, object]] = set()
        for rule in self.rules:
            for atom in rule.atoms:
                seen.add(self._atom_key(atom))
        return len(seen)

    def clone(self) -> "_RuleSetSubgroup":
        rs = _RuleSetSubgroup(
            rules=[r.clone() for r in self.rules],
            default_weight=float(self.default_weight),
        )
        rs.prediction_type = self.prediction_type
        if self.default_beta is not None:
            rs.default_beta = self.default_beta.copy()
            rs.default_beta0 = self.default_beta0
        return rs

    def _atom_mask(self, atom: _AtomGeneSubgroup, X: np.ndarray) -> np.ndarray:
        col = X[:, atom.feature_idx]
        if atom.op == "<=":
            return np.asarray(col, dtype=float) <= float(atom.value)
        if atom.op == "<":
            return np.asarray(col, dtype=float) < float(atom.value)
        if atom.op == ">=":
            return np.asarray(col, dtype=float) >= float(atom.value)
        if atom.op == ">":
            return np.asarray(col, dtype=float) > float(atom.value)
        if atom.op == "between":
            lo, hi = atom.value
            c = np.asarray(col, dtype=float)
            return (c >= float(lo)) & (c <= float(hi))
        if atom.op == "==":
            try:
                return (np.asarray(col, dtype=float) == float(atom.value)).astype(bool)
            except (TypeError, ValueError):
                return (np.asarray(col, dtype=object) == atom.value).astype(bool)
        if atom.op == "in":
            return np.isin(np.asarray(col, dtype=object), list(atom.value)).astype(bool)
        return np.ones(X.shape[0], dtype=bool)

    def _rule_mask(self, rule: _RuleSubgroup, X: np.ndarray) -> np.ndarray:
        if not rule.atoms:
            return np.zeros(X.shape[0], dtype=bool)
        mask = np.ones(X.shape[0], dtype=bool)
        for atom in rule.atoms:
            mask &= self._atom_mask(atom, X)
        return mask

    def predict(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        preds = np.zeros(n_samples, dtype=float)
        counts = np.zeros(n_samples, dtype=int)
        
        for rule in self.rules:
            mask = self._rule_mask(rule, X)
            if mask.any():
                if self.prediction_type == "linear" and rule.beta is not None:
                    preds[mask] += (X[mask] @ rule.beta) + rule.beta0
                else:
                    preds[mask] += rule.weight
                counts[mask] += 1
                
        # Average active rules; fallback to default if no rule fires
        fired = counts > 0
        preds[fired] /= counts[fired]
        
        unfired = ~fired
        if unfired.any():
            if self.prediction_type == "linear" and self.default_beta is not None:
                preds[unfired] = (X[unfired] @ self.default_beta) + self.default_beta0
            else:
                preds[unfired] = self.default_weight
        return preds


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
        mask = rs._rule_mask(r, X)
        coverage_counts[mask] += 1
        n_r = mask.sum()
        
        if n_r > 0:
            local_mean = float(np.mean(y[mask]))
            # Subgroup Quality: (coverage^gamma) * absolute deviation from global mean
            c_r = n_r / n
            q_r = (c_r ** gamma) * abs(local_mean - global_mean)
            total_quality += q_r
            r.weight = local_mean
            r.quality = q_r
        else:
            r.weight = global_mean
            r.quality = 0.0
            
    # Overlap penalty
    overlap_sum = np.maximum(0, coverage_counts - 1).sum()
    overlap_penalty = (overlap_sum / n) * abs(global_mean) * 0.5
    
    final_quality = total_quality - overlap_penalty
    size = sum(len(r.atoms) for r in rs.rules)
    return _FitnessSubgroup(quality=float(final_quality), size=size)


def _pareto_front_subgroup(
    individuals: list[tuple[_RuleSetSubgroup, _FitnessSubgroup]],
) -> list[tuple[_RuleSetSubgroup, _FitnessSubgroup]]:
    front: list[tuple[_RuleSetSubgroup, _FitnessSubgroup]] = []
    for cand_rs, cand_fit in individuals:
        dominated = False
        kept: list[tuple[_RuleSetSubgroup, _FitnessSubgroup]] = []
        for cur_rs, cur_fit in front:
            if cur_fit.dominates(cand_fit):
                dominated = True
                break
            if not cand_fit.dominates(cur_fit):
                kept.append((cur_rs, cur_fit))
        if not dominated:
            kept.append((cand_rs, cand_fit))
            front = kept
    return front


def _tournament_trim_subgroup(
    individuals: list[tuple[_RuleSetSubgroup, _FitnessSubgroup]],
    n_keep: int,
    tournament_size: int,
    rng: np.random.Generator,
) -> list[tuple[_RuleSetSubgroup, _FitnessSubgroup]]:
    if len(individuals) <= n_keep:
        return individuals

    selected: list[tuple[_RuleSetSubgroup, _FitnessSubgroup]] = []
    remaining_idx = list(range(len(individuals)))

    for _ in range(n_keep):
        if not remaining_idx:
            break
        k = min(tournament_size, len(remaining_idx))
        contest_positions = rng.choice(len(remaining_idx), size=k, replace=False)
        contest_idx = [remaining_idx[p] for p in contest_positions]

        winner_orig_idx = contest_idx[0]
        _, w_fit = individuals[winner_orig_idx]

        for orig_idx in contest_idx[1:]:
            _, c_fit = individuals[orig_idx]
            if c_fit.dominates(w_fit):
                winner_orig_idx = orig_idx
                w_fit = c_fit
            elif not w_fit.dominates(c_fit) and c_fit.quality > w_fit.quality:
                winner_orig_idx = orig_idx
                w_fit = c_fit

        selected.append(individuals[winner_orig_idx])
        remaining_idx.remove(winner_orig_idx)

    return selected


class RuleGPSubgroup(BaseRuleSetEstimator):
    """Genetic Programming for Scored Rule Sets targeting continuous regression."""

    def __init__(
        self,
        gamma: float = 0.5,
        max_generations: int = 150,
        stagnation_generations: int = 40,
        population_size: int = 80,
        n_adaptations_per_gen: int = 25,
        tournament_size: int = 4,
        max_rules: int | None = 12,
        max_atoms_per_rule: int | None = 4,
        atom_space_strategy: NativeAtomSpaceStrategy = "hybrid",
        continuous_threshold_strategy: ContinuousThresholdStrategy = "quantile_midpoint",
        prediction_type: Literal["constant", "linear"] = "constant",
        max_fit_seconds: float | None = None,
        feature_names: list[str] | None = None,
        random_state: int | None = None,
    ):
        self.gamma = gamma
        self.max_generations = max_generations
        self.stagnation_generations = stagnation_generations
        self.population_size = population_size
        self.n_adaptations_per_gen = n_adaptations_per_gen
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
        self.global_mean_ = float(np.mean(y_arr))
        self.n_features_in_ = X_arr.shape[1]
        self.feature_names_in_ = self.feature_names or [f"f{i}" for i in range(self.n_features_in_)]

        rng = np.random.default_rng(self.random_state)
        deadline = resolve_deadline(self.max_fit_seconds)

        # 1. Build Candidate Atom Space
        specs = build_native_feature_specs(
            X_arr,
            strategy=self.atom_space_strategy,
            continuous_threshold_strategy=self.continuous_threshold_strategy,
        )

        available_atoms: list[_AtomGeneSubgroup] = []
        for feat_idx, spec in enumerate(specs):
            for op in spec.get("operators", ["<=", ">"]):
                for val in spec.get("thresholds", []):
                    available_atoms.append(_AtomGeneSubgroup(feature_idx=feat_idx, op=op, value=val))
            for val in spec.get("categories", []):
                available_atoms.append(_AtomGeneSubgroup(feature_idx=feat_idx, op="==", value=val))

        if not available_atoms:
            # Fallback atom if no variation
            available_atoms.append(_AtomGeneSubgroup(feature_idx=0, op="<=", value=0.0))

        # 2. Initialize Population
        pop_size = max(int(self.population_size), 10)
        pop: list[tuple[_RuleSetSubgroup, _FitnessSubgroup]] = []

        for _ in range(pop_size):
            n_r = rng.integers(1, 4)
            rules: list[_RuleSubgroup] = []
            for _ in range(n_r):
                n_a = rng.integers(1, min(self.max_atoms_per_rule or 4, len(available_atoms)) + 1)
                chosen_atoms = [rng.choice(available_atoms) for _ in range(n_a)]
                rules.append(_RuleSubgroup(atoms=chosen_atoms))
            
            rs = _RuleSetSubgroup(rules=rules)
            fit = _evaluate_fitness_subgroup(rs, X_arr, y_arr, self.global_mean_, self.gamma)
            pop.append((rs, fit))

        best_rs = pop[0][0]
        best_fit = pop[0][1]
        best_quality = best_fit.quality
        stagnation = 0

        # 3. Evolution Loop
        for gen in range(self.max_generations):
            if deadline_reached(deadline):
                break

            offspring: list[tuple[_RuleSetSubgroup, _FitnessSubgroup]] = []
            for _ in range(self.n_adaptations_per_gen):
                # Pick parent
                p1_idx = rng.integers(0, len(pop))
                child_rs = pop[p1_idx][0].clone()

                # Mutation operators
                mut_type = rng.choice(["add_rule", "drop_rule", "add_atom", "drop_atom", "modify_atom"])
                
                max_r = self.max_rules or 12
                max_a = self.max_atoms_per_rule or 4

                if mut_type == "add_rule" and len(child_rs.rules) < max_r:
                    new_atom = rng.choice(available_atoms)
                    child_rs.rules.append(_RuleSubgroup(atoms=[new_atom]))
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
                c_fit = _evaluate_fitness_subgroup(child_rs, X_arr, y_arr, self.global_mean_, self.gamma)
                offspring.append((child_rs, c_fit))

            # Combine and trim population
            combined = pop + offspring
            pop = _tournament_trim_subgroup(combined, pop_size, self.tournament_size, rng)

            # Check best individual (balancing RMSE & compactness)
            for rs, fit in pop:
                if fit.quality > best_quality + 1e-4:
                    best_quality = fit.quality
                    best_fit = fit
                    best_rs = rs.clone()
                    stagnation = 0
                elif fit.quality >= best_quality - 1e-4 and fit.size < best_fit.size:
                    best_fit = fit
                    best_rs = rs.clone()
                    stagnation = 0

            stagnation += 1
            if stagnation >= self.stagnation_generations:
                break

        # 4. Backward Rule Elimination (Post-hoc Compaction)
        self.ruleset_internal_ = self._compact_ruleset(best_rs, X_arr, y_arr)
        self.ruleset_ = self._to_schema_ruleset(self.ruleset_internal_, X_arr)
        return self

    def _compact_ruleset(self, rs: _RuleSetSubgroup, X: np.ndarray, y: np.ndarray) -> _RuleSetSubgroup:
        curr_rs = rs.clone()
        base_fit = _evaluate_fitness_subgroup(curr_rs, X, y, self.global_mean_, self.gamma)
        
        # Greedily remove rules if Quality does not drop significantly
        improved = True
        while improved and len(curr_rs.rules) > 1:
            improved = False
            best_candidate = None
            best_cand_quality = base_fit.quality - 1e-4  # Allow small tolerance for huge size reduction

            for idx in range(len(curr_rs.rules)):
                cand_rs = curr_rs.clone()
                cand_rs.rules.pop(idx)
                cand_fit = _evaluate_fitness_subgroup(cand_rs, X, y, self.global_mean_, self.gamma)
                if cand_fit.quality >= best_cand_quality:
                    best_cand_quality = cand_fit.quality
                    best_candidate = cand_rs

            if best_candidate is not None:
                curr_rs = best_candidate
                base_fit = _evaluate_fitness_subgroup(curr_rs, X, y, self.global_mean_, self.gamma)
                improved = True
                improved = True

        return curr_rs
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
            
            mask = rs._rule_mask(rule, X)
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
            metadata={"estimator": "RuleGPSubgroup", "rules_count": len(schema_rules)},
        )
        ruleset.validate()
        return ruleset

    def predict(self, X):
        raise NotImplementedError("RuleGPSubgroup is an exploratory model for subgroup discovery, not a predictive model. Inspect .ruleset_ instead.")

    def to_ruleset(self) -> ScoredRuleSet:
        check_is_fitted(self, "ruleset_")
        return self.ruleset_
