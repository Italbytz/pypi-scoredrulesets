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
    fitness: float = 999999.0  # Minimise fitness (variance-reduction penalty + complexity)


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
    mdl_weight: float = 0.05,
    min_samples: int = 10,
) -> None:
    mask = _matches_mask(ind, X)
    n_cov = int(mask.sum())
    n_total = X.shape[0]
    ind.coverage_count = n_cov

    if n_cov < min_samples or n_cov > n_total - max(5, min_samples // 2):
        ind.weight = float(np.mean(y)) if y.size > 0 else 0.0
        ind.rmse_local = float(np.std(y)) if y.size > 0 else 1.0
        ind.theory_length = 1.0
        ind.fitness = 999999.0
        return

    y_cov = y[mask]
    y_rem = y[~mask]
    ind.weight = float(np.mean(y_cov))
    mse_local = float(np.mean((y_cov - ind.weight) ** 2))
    ind.rmse_local = float(np.sqrt(mse_local))
    ind.theory_length = _compute_theory_length_reg(ind, feature_info)
    
    # Global variance reduction criterion
    sse_total = float(np.var(y) * n_total)
    sse_cov = float(np.var(y_cov) * n_cov)
    sse_rem = float(np.var(y_rem) * (n_total - n_cov))
    gain = (sse_total - (sse_cov + sse_rem)) / (sse_total + 1e-12)
    
    # Minimizing fitness: negative gain + MDL theory length penalty
    ind.fitness = -gain + mdl_weight * (ind.theory_length + 0.05 * len(ind.predicates))


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
    prob_express: float = 0.35,
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
            size = rng.uniform(0.35, 0.85) * domain
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
        max_rules: int = 8,
        pop_size: int = 60,
        num_generations: int = 50,
        tournament_size: int = 4,
        min_samples_split: int = 12,
        mdl_weight: float = 0.04,
        prob_crossover: float = 0.8,
        prob_mutation: float = 0.25,
        max_fit_seconds: float | None = None,
        feature_names: list[str] | None = None,
        random_state: int | None = None,
    ):
        self.max_rules = max_rules
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.tournament_size = tournament_size
        self.min_samples_split = min_samples_split
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
        self._feature_info_ = feature_info
        
        discovered_rules: list[_IndividualReg] = []
        rule_masks: list[np.ndarray] = []
        global_mean = float(np.mean(y_arr))
        residuals = y_arr - global_mean
        ridge_alpha = 1.0

        # Iterative Rule Learning on Residuals with Joint Ridge Calibration
        for rule_iter in range(self.max_rules):
            if deadline_reached(deadline):
                break

            # Seed GA population preferentially around high-residual instances
            abs_res = np.abs(residuals)
            res_sum = float(np.sum(abs_res))
            if res_sum > 1e-12:
                seed_probs = abs_res / res_sum
            else:
                seed_probs = np.full(X_arr.shape[0], 1.0 / X_arr.shape[0])

            # 1. Initialize GA population
            pop: list[_IndividualReg] = []
            for _ in range(self.pop_size):
                seed_idx = int(rng.choice(X_arr.shape[0], p=seed_probs))
                inst = X_arr[seed_idx]
                ind = _init_individual_reg(rng, inst, feature_info)
                _evaluate_reg_individual(ind, X_arr, residuals, feature_info, self.mdl_weight, min_samples=self.min_samples_split)
                pop.append(ind)

            # 2. Evolve one best rule on current residuals
            for gen in range(self.num_generations):
                if deadline_reached(deadline):
                    break

                offspring: list[_IndividualReg] = []
                for _ in range(self.pop_size):
                    cand_pos = rng.choice(len(pop), size=min(self.tournament_size, len(pop)), replace=False)
                    p1 = min([pop[p] for p in cand_pos], key=lambda ind: ind.fitness)

                    child = _IndividualReg(predicates=[_clone_pred(p) for p in p1.predicates])

                    # Mutate child
                    if rng.random() < self.prob_mutation and len(child.predicates) > 0:
                        p_idx = rng.integers(0, len(child.predicates))
                        pred = child.predicates[p_idx]
                        if pred.is_numeric:
                            dom = feature_info[pred.feature_idx]["max"] - feature_info[pred.feature_idx]["min"]
                            shift = rng.uniform(-0.15, 0.15) * dom
                            pred.lo = max(feature_info[pred.feature_idx]["min"], pred.lo + shift)
                            pred.hi = min(feature_info[pred.feature_idx]["max"], pred.hi + shift)
                    elif rng.random() < 0.25:
                        if len(child.predicates) > 1 and rng.random() < 0.55:
                            child.predicates.pop(rng.integers(0, len(child.predicates)))
                        else:
                            fi = rng.integers(0, self.n_features_in_)
                            info = feature_info[fi]
                            if info["numeric"]:
                                dom = info["max"] - info["min"]
                                sz = rng.uniform(0.35, 0.85) * dom
                                c = X_arr[int(rng.choice(X_arr.shape[0], p=seed_probs)), fi]
                                child.predicates.append(_Predicate(
                                    feature_idx=fi, is_numeric=True,
                                    lo=max(info["min"], c - sz / 2),
                                    hi=min(info["max"], c + sz / 2),
                                ))

                    _evaluate_reg_individual(child, X_arr, residuals, feature_info, self.mdl_weight, min_samples=self.min_samples_split)
                    offspring.append(child)

                combined = pop + offspring
                combined.sort(key=lambda ind: ind.fitness)
                pop = combined[:self.pop_size]

            best_rule = pop[0]
            if best_rule.coverage_count < self.min_samples_split or best_rule.fitness >= 0.0:
                break

            full_mask = _matches_mask(best_rule, X_arr)
            if not full_mask.any() or np.all(full_mask):
                break

            discovered_rules.append(best_rule)
            rule_masks.append(full_mask)

            # Update residuals via closed-form joint Ridge calibration across discovered rules
            M_sub = np.column_stack(rule_masks).astype(float)
            M_ext = np.column_stack([np.ones(X_arr.shape[0]), M_sub])
            K_ext = M_ext.shape[1]
            reg_mat = ridge_alpha * np.eye(K_ext)
            reg_mat[0, 0] = 1e-8  # Do not regularize intercept
            weights = np.linalg.solve(M_ext.T @ M_ext + reg_mat, M_ext.T @ y_arr)
            residuals = y_arr - (M_ext @ weights)

        self.discovered_rules_ = discovered_rules
        self.default_weight_ = global_mean
        self.pareto_archive_: dict[int, ScoredRuleSet] = {}
        self.ruleset_ = self._build_prefix_archive_and_final_ruleset(X_arr, y_arr, rule_masks, ridge_alpha=ridge_alpha)
        return self

    def _build_ruleset_from_prefix(
        self,
        rules_prefix: list[_IndividualReg],
        weights: list[float],
        intercept: float,
    ) -> ScoredRuleSet:
        schema_rules: list[Rule] = []
        for idx, (ind, w) in enumerate(zip(rules_prefix, weights)):
            schema_atoms: list[Atom] = []
            for p in ind.predicates:
                feat_name = self.feature_names_in_[p.feature_idx]
                if p.is_numeric:
                    info = self._feature_info_[p.feature_idx] if hasattr(self, "_feature_info_") else None
                    if info is not None and p.lo <= info["min"] + 1e-9 and p.hi < info["max"] - 1e-9:
                        schema_atoms.append(Atom(feature=feat_name, op="<=", value=float(p.hi)))
                    elif info is not None and p.hi >= info["max"] - 1e-9 and p.lo > info["min"] + 1e-9:
                        schema_atoms.append(Atom(feature=feat_name, op=">", value=float(p.lo)))
                    else:
                        schema_atoms.append(Atom(feature=feat_name, op="between", value=[float(p.lo), float(p.hi)]))
                else:
                    schema_atoms.append(Atom(feature=feat_name, op="in", value=list(p.allowed)))
            schema_rules.append(
                Rule(atoms=schema_atoms, scores=[float(w)], rule_id=f"r_{idx}")
            )

        schema_rules.append(
            Rule(atoms=[], scores=[float(intercept)], rule_id="default_reg", metadata={"is_default": True})
        )
        comp = sum(len(r.atoms) for r in schema_rules if r.atoms)
        ruleset = ScoredRuleSet(
            class_labels=[],
            task_type="regression",
            feature_names=list(self.feature_names_in_),
            rules=schema_rules,
            aggregation=AggregationSpec(type="default_plus_sum"),
            metadata={
                "estimator": "RulePLCSRegressor",
                "rules_count": len(schema_rules),
                "complexity_atoms": int(comp),
            },
        )
        ruleset.validate()
        return ruleset

    def _build_prefix_archive_and_final_ruleset(
        self,
        X_arr: np.ndarray,
        y_arr: np.ndarray,
        rule_masks: list[np.ndarray],
        ridge_alpha: float = 1.0,
    ) -> ScoredRuleSet:
        n_samples = X_arr.shape[0]
        if not self.discovered_rules_:
            rs = self._build_ruleset_from_prefix([], [], self.default_weight_)
            self.pareto_archive_[0] = rs
            return rs

        M_all = np.column_stack(rule_masks).astype(float)
        final_rs: ScoredRuleSet | None = None
        for k in range(1, len(self.discovered_rules_) + 1):
            M_sub = M_all[:, :k]
            M_ext = np.column_stack([np.ones(n_samples), M_sub])
            K_ext = M_ext.shape[1]
            reg_mat = ridge_alpha * np.eye(K_ext)
            reg_mat[0, 0] = 1e-8
            weights = np.linalg.solve(M_ext.T @ M_ext + reg_mat, M_ext.T @ y_arr)
            w0 = float(weights[0])
            rw = [float(w) for w in weights[1:]]
            for idx, w in enumerate(rw):
                self.discovered_rules_[idx].weight = w
            self.default_weight_ = w0
            rs_k = self._build_ruleset_from_prefix(self.discovered_rules_[:k], rw, w0)
            comp = sum(len(r.atoms) for r in rs_k.rules if r.atoms)
            self.pareto_archive_[comp] = rs_k
            final_rs = rs_k

        assert final_rs is not None
        return final_rs

    def _to_schema_ruleset(self) -> ScoredRuleSet:
        return self._build_ruleset_from_prefix(
            self.discovered_rules_,
            [float(ind.weight) for ind in self.discovered_rules_],
            self.default_weight_,
        )

    def predict(self, X):
        check_is_fitted(self, "ruleset_")
        X_arr = check_array(X, dtype=None)
        return predict_regression_from_ruleset(self.ruleset_, X_arr)

    def to_ruleset(self) -> ScoredRuleSet:
        check_is_fitted(self, "ruleset_")
        return self.ruleset_
