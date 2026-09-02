"""
RuleGPRegressor - Native Genetic Programming for Scored Rule Sets in Regression
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
class _AtomGeneReg:
    feature_idx: int
    op: str
    value: object


@dataclass
class _RuleReg:
    atoms: list[_AtomGeneReg]
    weight: float = 0.0

    def clone(self) -> "_RuleReg":
        return _RuleReg(atoms=list(self.atoms), weight=float(self.weight))


@dataclass
class _RuleSetReg:
    rules: list[_RuleReg]
    default_weight: float = 0.0

    @staticmethod
    def _atom_key(atom: _AtomGeneReg) -> tuple[int, str, object]:
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

    def clone(self) -> "_RuleSetReg":
        return _RuleSetReg(
            rules=[r.clone() for r in self.rules],
            default_weight=float(self.default_weight),
        )

    def _atom_mask(self, atom: _AtomGeneReg, X: np.ndarray) -> np.ndarray:
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

    def _rule_mask(self, rule: _RuleReg, X: np.ndarray) -> np.ndarray:
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
                preds[mask] += rule.weight
                counts[mask] += 1
                
        # Average active rules; fallback to default if no rule fires
        fired = counts > 0
        preds[fired] /= counts[fired]
        preds[~fired] = self.default_weight
        return preds


@dataclass(frozen=True)
class _FitnessReg:
    rmse: float
    size: int
    r2: float

    def dominates(self, other: "_FitnessReg") -> bool:
        ge = (self.rmse <= other.rmse and self.size <= other.size)
        gt = (self.rmse < other.rmse or self.size < other.size)
        return ge and gt


def _compute_regression_weights(rs: _RuleSetReg, X: np.ndarray, y: np.ndarray) -> None:
    global_mean = float(np.mean(y)) if y.size > 0 else 0.0
    any_fired = np.zeros(X.shape[0], dtype=bool)

    for rule in rs.rules:
        mask = rs._rule_mask(rule, X)
        any_fired |= mask
        if mask.any():
            rule.weight = float(np.mean(y[mask]))
        else:
            rule.weight = global_mean

    uncovered = ~any_fired
    if uncovered.any():
        rs.default_weight = float(np.mean(y[uncovered]))
    else:
        rs.default_weight = global_mean


def _evaluate_fitness_reg(rs: _RuleSetReg, X: np.ndarray, y: np.ndarray) -> _FitnessReg:
    preds = rs.predict(X)
    mse = float(np.mean((y - preds) ** 2))
    rmse = float(np.sqrt(mse))
    var_y = float(np.var(y))
    r2 = 1.0 - (mse / var_y) if var_y > 1e-12 else 0.0
    return _FitnessReg(rmse=rmse, size=rs.size, r2=r2)


def _pareto_front_reg(
    individuals: list[tuple[_RuleSetReg, _FitnessReg]],
) -> list[tuple[_RuleSetReg, _FitnessReg]]:
    front: list[tuple[_RuleSetReg, _FitnessReg]] = []
    for cand_rs, cand_fit in individuals:
        dominated = False
        kept: list[tuple[_RuleSetReg, _FitnessReg]] = []
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


def _tournament_trim_reg(
    individuals: list[tuple[_RuleSetReg, _FitnessReg]],
    n_keep: int,
    tournament_size: int,
    rng: np.random.Generator,
) -> list[tuple[_RuleSetReg, _FitnessReg]]:
    if len(individuals) <= n_keep:
        return individuals

    selected: list[tuple[_RuleSetReg, _FitnessReg]] = []
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
            elif not w_fit.dominates(c_fit) and c_fit.rmse < w_fit.rmse:
                winner_orig_idx = orig_idx
                w_fit = c_fit

        selected.append(individuals[winner_orig_idx])
        remaining_idx.remove(winner_orig_idx)

    return selected


class RuleGPRegressor(BaseRuleSetEstimator, RegressorMixin):
    """Genetic Programming for Scored Rule Sets targeting continuous regression."""

    def __init__(
        self,
        max_generations: int = 150,
        stagnation_generations: int = 40,
        population_size: int = 80,
        n_adaptations_per_gen: int = 25,
        tournament_size: int = 4,
        max_rules: int | None = 12,
        max_atoms_per_rule: int | None = 4,
        atom_space_strategy: NativeAtomSpaceStrategy = "hybrid",
        continuous_threshold_strategy: ContinuousThresholdStrategy = "quantile_midpoint",
        max_fit_seconds: float | None = None,
        feature_names: list[str] | None = None,
        random_state: int | None = None,
    ):
        self.max_generations = max_generations
        self.stagnation_generations = stagnation_generations
        self.population_size = population_size
        self.n_adaptations_per_gen = n_adaptations_per_gen
        self.tournament_size = tournament_size
        self.max_rules = max_rules
        self.max_atoms_per_rule = max_atoms_per_rule
        self.atom_space_strategy = atom_space_strategy
        self.continuous_threshold_strategy = continuous_threshold_strategy
        self.max_fit_seconds = max_fit_seconds
        self.feature_names = feature_names
        self.random_state = random_state

    def fit(self, X, y):
        X_arr, y_arr = check_X_y(X, y, dtype=None, y_numeric=True)
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

        available_atoms: list[_AtomGeneReg] = []
        for feat_idx, spec in enumerate(specs):
            for op in spec.get("operators", ["<=", ">"]):
                for val in spec.get("thresholds", []):
                    available_atoms.append(_AtomGeneReg(feature_idx=feat_idx, op=op, value=val))
            for val in spec.get("categories", []):
                available_atoms.append(_AtomGeneReg(feature_idx=feat_idx, op="==", value=val))

        if not available_atoms:
            # Fallback atom if no variation
            available_atoms.append(_AtomGeneReg(feature_idx=0, op="<=", value=0.0))

        # 2. Initialize Population
        pop_size = max(int(self.population_size), 10)
        pop: list[tuple[_RuleSetReg, _FitnessReg]] = []

        for _ in range(pop_size):
            n_r = rng.integers(1, 4)
            rules: list[_RuleReg] = []
            for _ in range(n_r):
                n_a = rng.integers(1, min(3, len(available_atoms)) + 1)
                chosen_atoms = [rng.choice(available_atoms) for _ in range(n_a)]
                rules.append(_RuleReg(atoms=chosen_atoms))
            
            rs = _RuleSetReg(rules=rules)
            _compute_regression_weights(rs, X_arr, y_arr)
            fit = _evaluate_fitness_reg(rs, X_arr, y_arr)
            pop.append((rs, fit))

        best_rs = pop[0][0]
        best_fit = pop[0][1]
        best_rmse = best_fit.rmse
        stagnation = 0

        # 3. Evolution Loop
        for gen in range(self.max_generations):
            if deadline_reached(deadline):
                break

            offspring: list[tuple[_RuleSetReg, _FitnessReg]] = []
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
                offspring.append((child_rs, c_fit))

            # Combine and trim population
            combined = pop + offspring
            pop = _tournament_trim_reg(combined, pop_size, self.tournament_size, rng)

            # Check best individual (balancing RMSE & compactness)
            for rs, fit in pop:
                if fit.rmse < best_rmse - 1e-4:
                    best_rmse = fit.rmse
                    best_fit = fit
                    best_rs = rs.clone()
                    stagnation = 0
                elif fit.rmse <= best_rmse + 1e-4 and fit.size < best_fit.size:
                    best_fit = fit
                    best_rs = rs.clone()
                    stagnation = 0

            stagnation += 1
            if stagnation >= self.stagnation_generations:
                break

        # 4. Backward Rule Elimination (Post-hoc Compaction)
        self.ruleset_internal_ = self._compact_ruleset(best_rs, X_arr, y_arr)
        self.ruleset_ = self._to_schema_ruleset(self.ruleset_internal_)
        return self

    def _compact_ruleset(self, rs: _RuleSetReg, X: np.ndarray, y: np.ndarray) -> _RuleSetReg:
        curr_rs = rs.clone()
        base_fit = _evaluate_fitness_reg(curr_rs, X, y)
        
        # Greedily remove rules if RMSE does not increase significantly
        improved = True
        while improved and len(curr_rs.rules) > 1:
            improved = False
            best_candidate = None
            best_cand_rmse = base_fit.rmse + 1e-3  # Allow small tolerance for huge size reduction

            for idx in range(len(curr_rs.rules)):
                cand_rs = curr_rs.clone()
                cand_rs.rules.pop(idx)
                _compute_regression_weights(cand_rs, X, y)
                cand_fit = _evaluate_fitness_reg(cand_rs, X, y)
                if cand_fit.rmse <= best_cand_rmse:
                    best_cand_rmse = cand_fit.rmse
                    best_candidate = cand_rs

            if best_candidate is not None:
                curr_rs = best_candidate
                base_fit = _evaluate_fitness_reg(curr_rs, X, y)
                improved = True

        return curr_rs

    def _to_schema_ruleset(self, rs: _RuleSetReg) -> ScoredRuleSet:
        schema_rules: list[Rule] = []
        
        # Non-default rules
        for idx, rule in enumerate(rs.rules):
            schema_atoms: list[Atom] = []
            for a in rule.atoms:
                feat_name = self.feature_names_in_[a.feature_idx]
                schema_atoms.append(Atom(feature=feat_name, op=a.op, value=a.value))
            schema_rules.append(
                Rule(atoms=schema_atoms, scores=[float(rule.weight)], rule_id=f"r_{idx}")
            )

        # Default rule
        schema_rules.append(
            Rule(atoms=[], scores=[float(rs.default_weight)], rule_id="default_reg")
        )

        ruleset = ScoredRuleSet(
            class_labels=[],
            task_type="regression",
            feature_names=list(self.feature_names_in_),
            rules=schema_rules,
            aggregation=AggregationSpec(type="mean_active"),
            metadata={"estimator": "RuleGPRegressor", "rules_count": len(schema_rules)},
        )
        ruleset.validate()
        return ruleset

    def predict(self, X):
        check_is_fitted(self, "ruleset_")
        X_arr = check_array(X, dtype=None)
        return predict_regression_from_ruleset(self.ruleset_, X_arr)

    def to_ruleset(self) -> ScoredRuleSet:
        check_is_fitted(self, "ruleset_")
        return self.ruleset_
