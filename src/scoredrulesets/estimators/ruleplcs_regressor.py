"""
RulePLCSRegressor – Iterative Rule Learning / BioHEL-inspired Regression Scored Rule Sets
========================================================================================

Implements Sequential Covering Iterative Rule Learning (IRL) for continuous targets.
Each GA iteration evolves a variable-length hyperrectangle rule that maximizes local variance
reduction and coverage while penalizing theory length (MDL principle).
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
from .atom_space import RulePLCSFeatureTypingStrategy, build_ruleplcs_feature_info
from .base import BaseRuleSetEstimator
from .ruleplcs import _Predicate, _matches_mask


@dataclass
class _IndividualReg:
    """One regression rule candidate."""
    predicates: list[_Predicate]
    weight: float = 0.0
    rmse_local: float = 999999.0
    coverage_count: int = 0
    theory_length: float = 0.0
    fitness: float = 999999.0  # Minimise fitness (error + complexity)


def _compute_theory_length_reg(ind: _IndividualReg, feature_info: list[dict]) -> float:
    if len(feature_info) == 0:
        return 0.0
    total = 0.0
    for p in ind.predicates:
        info = feature_info[p.feature_idx]
        if p.is_numeric:
            domain = info["max"] - info["min"]
            if domain > 0:
                total += 1.0 - (p.hi - p.lo) / domain
        else:
            n_vals = len(info["values"])
            if n_vals > 0:
                total += (n_vals - len(p.allowed)) / n_vals
    return total / len(feature_info)


def _evaluate_reg_individual(
    ind: _IndividualReg,
    X: np.ndarray,
    y: np.ndarray,
    feature_info: list[dict],
    mdl_weight: float = 0.25,
) -> None:
    mask = _matches_mask(ind, X)
    n_cov = int(mask.sum())
    ind.coverage_count = n_cov

    if n_cov < 3:
        ind.weight = float(np.mean(y)) if y.size > 0 else 0.0
        ind.rmse_local = float(np.std(y)) if y.size > 0 else 1.0
        ind.theory_length = 1.0
        ind.fitness = 999999.0
        return

    y_cov = y[mask]
    ind.weight = float(np.mean(y_cov))
    mse_local = float(np.mean((y_cov - ind.weight) ** 2))
    ind.rmse_local = float(np.sqrt(mse_local))
    
    ind.theory_length = _compute_theory_length_reg(ind, feature_info)
    
    # Coverage reward (fraction of current dataset)
    cov_ratio = n_cov / X.shape[0]
    std_global = float(np.std(y)) if np.std(y) > 1e-12 else 1.0
    
    # Normalized error + coverage penalty + MDL theory length
    norm_err = ind.rmse_local / std_global
    cov_penalty = max(0.0, 1.0 - cov_ratio)
    ind.fitness = norm_err + 0.5 * cov_penalty + mdl_weight * ind.theory_length


def _clone_pred(p: _Predicate) -> _Predicate:
    return _Predicate(
        feature_idx=p.feature_idx,
        is_numeric=p.is_numeric,
        lo=p.lo,
        hi=p.hi,
        allowed=set(p.allowed),
    )


def _init_individual_reg(
    rng: np.random.Generator,
    inst: np.ndarray,
    feature_info: list[dict],
    prob_express: float = 0.3,
) -> _IndividualReg:
    predicates: list[_Predicate] = []
    n_features = len(feature_info)

    for fi in range(n_features):
        if rng.random() > prob_express:
            continue
        info = feature_info[fi]
        if info["numeric"]:
            domain = info["max"] - info["min"]
            if domain <= 0:
                continue
            size = rng.uniform(0.25, 0.75) * domain
            center = inst[fi]
            lo = max(info["min"], center - size / 2)
            hi = min(info["max"], center + size / 2)
            predicates.append(_Predicate(
                feature_idx=fi, is_numeric=True, lo=lo, hi=hi,
            ))
        else:
            vals = info["values"]
            allowed: set[int] = {int(round(inst[fi]))}
            for ov in vals:
                if ov != inst[fi] and rng.random() < 0.5:
                    allowed.add(ov)
            if len(allowed) < len(vals):
                predicates.append(_Predicate(
                    feature_idx=fi, is_numeric=False, allowed=allowed,
                ))

    return _IndividualReg(predicates=predicates)


class RulePLCSRegressor(BaseRuleSetEstimator, RegressorMixin):
    """BioHEL/LCS-inspired Iterative Rule Learning for Continuous Regression."""

    def __init__(
        self,
        max_rules: int = 12,
        pop_size: int = 50,
        num_generations: int = 60,
        tournament_size: int = 4,
        coverage_break: float = 0.85,
        mdl_weight: float = 0.2,
        prob_crossover: float = 0.8,
        prob_mutation: float = 0.15,
        max_fit_seconds: float | None = None,
        feature_names: list[str] | None = None,
        random_state: int | None = None,
    ):
        self.max_rules = max_rules
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.tournament_size = tournament_size
        self.coverage_break = coverage_break
        self.mdl_weight = mdl_weight
        self.prob_crossover = prob_crossover
        self.prob_mutation = prob_mutation
        self.max_fit_seconds = max_fit_seconds
        self.feature_names = feature_names
        self.random_state = random_state

    def fit(self, X, y):
        X_arr, y_arr = check_X_y(X, y, dtype=None, y_numeric=True)
        self.n_features_in_ = X_arr.shape[1]
        self.feature_names_in_ = self.feature_names or [f"f{i}" for i in range(self.n_features_in_)]

        rng = np.random.default_rng(self.random_state)
        deadline = resolve_deadline(self.max_fit_seconds)

        feature_info = build_ruleplcs_feature_info(X_arr, low_cardinality_threshold=10, deadline=deadline)
        
        discovered_rules: list[_IndividualReg] = []
        remaining_indices = np.arange(X_arr.shape[0])
        global_mean = float(np.mean(y_arr))

        # Iterative Rule Learning (Sequential Covering)
        for rule_iter in range(self.max_rules):
            if deadline_reached(deadline) or len(remaining_indices) < 5:
                break

            X_rem = X_arr[remaining_indices]
            y_rem = y_arr[remaining_indices]

            # 1. Initialize GA population seeded around remaining instances
            pop: list[_IndividualReg] = []
            for _ in range(self.pop_size):
                seed_idx = rng.choice(len(remaining_indices))
                inst = X_rem[seed_idx]
                ind = _init_individual_reg(rng, inst, feature_info)
                _evaluate_reg_individual(ind, X_rem, y_rem, feature_info, self.mdl_weight)
                pop.append(ind)

            # 2. Evolve one best rule
            for gen in range(self.num_generations):
                if deadline_reached(deadline):
                    break

                offspring: list[_IndividualReg] = []
                for _ in range(self.pop_size):
                    # Tournament
                    cand_pos = rng.choice(len(pop), size=min(self.tournament_size, len(pop)), replace=False)
                    p1 = min([pop[p] for p in cand_pos], key=lambda ind: ind.fitness)
                    
                    child = _IndividualReg(predicates=[_clone_pred(p) for p in p1.predicates])
                    
                    # Mutate child
                    if rng.random() < self.prob_mutation and len(child.predicates) > 0:
                        p_idx = rng.integers(0, len(child.predicates))
                        pred = child.predicates[p_idx]
                        if pred.is_numeric:
                            dom = feature_info[pred.feature_idx]["max"] - feature_info[pred.feature_idx]["min"]
                            shift = rng.uniform(-0.1, 0.1) * dom
                            pred.lo = max(feature_info[pred.feature_idx]["min"], pred.lo + shift)
                            pred.hi = min(feature_info[pred.feature_idx]["max"], pred.hi + shift)
                    
                    _evaluate_reg_individual(child, X_rem, y_rem, feature_info, self.mdl_weight)
                    offspring.append(child)

                combined = pop + offspring
                combined.sort(key=lambda ind: ind.fitness)
                pop = combined[:self.pop_size]

            best_rule = pop[0]
            if best_rule.coverage_count < 3 or best_rule.fitness >= 999999.0:
                break

            # Compute exact weight over full training set
            full_mask = _matches_mask(best_rule, X_arr)
            if full_mask.any():
                best_rule.weight = float(np.mean(y_arr[full_mask]))
            else:
                best_rule.weight = global_mean

            discovered_rules.append(best_rule)

            # Sequential covering update: remove instances explained well by this rule
            rem_mask = _matches_mask(best_rule, X_rem)
            covered_orig_indices = remaining_indices[rem_mask]
            
            # Remove covered indices
            remaining_indices = np.setdiff1d(remaining_indices, covered_orig_indices)

        self.discovered_rules_ = discovered_rules
        self.default_weight_ = global_mean
        self.ruleset_ = self._to_schema_ruleset()
        return self

    def _to_schema_ruleset(self) -> ScoredRuleSet:
        schema_rules: list[Rule] = []

        for idx, ind in enumerate(self.discovered_rules_):
            schema_atoms: list[Atom] = []
            for p in ind.predicates:
                feat_name = self.feature_names_in_[p.feature_idx]
                if p.is_numeric:
                    schema_atoms.append(Atom(feature=feat_name, op="between", value=[float(p.lo), float(p.hi)]))
                else:
                    schema_atoms.append(Atom(feature=feat_name, op="in", value=list(p.allowed)))
            schema_rules.append(
                Rule(atoms=schema_atoms, scores=[float(ind.weight)], rule_id=f"r_{idx}")
            )

        schema_rules.append(
            Rule(atoms=[], scores=[float(self.default_weight_)], rule_id="default_reg")
        )

        ruleset = ScoredRuleSet(
            class_labels=[],
            task_type="regression",
            feature_names=list(self.feature_names_in_),
            rules=schema_rules,
            aggregation=AggregationSpec(type="mean_active"),
            metadata={"estimator": "RulePLCSRegressor", "rules_count": len(schema_rules)},
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
