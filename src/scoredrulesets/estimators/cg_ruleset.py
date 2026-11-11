"""Column Generation (CG-Rule) for Scored Rule Sets.

Formulates Rule Induction as Dantzig-Wolfe Column Generation:
- Restricted Master Problem (RMP): L1-regularized convex loss minimization
  (logistic loss) over active candidate rules.
- Pricing Subproblem: Branch-and-Bound search with tight upper-bound pruning,
  literal screening, and sub-index intersection over the space of conjunctions.
- Produces a valid, interpretable ScoredRuleSet with exact feature and atom semantics.
"""

from __future__ import annotations

import heapq
import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .atom_space import (
    binarize_with_thresholds,
    compute_nln_thresholds,
)
from ..runtime import predict as predict_from_ruleset
from ..runtime import predict_proba as predict_proba_from_ruleset
from ..runtime import predict_regression as predict_regression_from_ruleset
from ..schema import AggregationSpec, Atom, Rule, ScoredRuleSet


@dataclass(frozen=True)
class LiteralSpec:
    """A single atomic literal: feature index, operator, threshold, and proposition index."""

    feat_idx: int
    op: Literal["<=", ">"]
    threshold: float
    prop_idx: int

    def to_atom(self, feature_name: str | None = None) -> Atom:
        fname = feature_name if feature_name is not None else f"f{self.feat_idx}"
        return Atom(feature=fname, op=self.op, value=self.threshold)


class PricingNode:
    """Search node for Branch-and-Bound rule generation."""

    __slots__ = ("literals", "indices", "width", "last_rank", "ub", "score")

    def __init__(
        self,
        literals: tuple[int, ...],
        indices: np.ndarray,
        width: int,
        last_rank: int,
        ub: float,
        score: float,
    ):
        self.literals = literals
        self.indices = indices
        self.width = width
        self.last_rank = last_rank
        self.ub = ub
        self.score = score

    def __lt__(self, other: PricingNode) -> bool:
        # Max-heap priority queue by upper bound
        return self.ub > other.ub


def _solve_pricing_subproblem(
    P_bool: np.ndarray,
    residuals: np.ndarray,
    meta: list[LiteralSpec],
    max_depth: int = 3,
    min_support: int = 3,
    max_admissible: int = 60,
    max_nodes: int = 1500,
) -> tuple[tuple[int, ...] | None, float]:
    """Solve pricing subproblem via Branch-and-Bound with UB pruning & sub-indexing.

    Finds conjunction R = AND(literals) maximizing score(R) = sum_{i in I(R)} residuals[i].
    Upper bound for any descendant R' subseteq R is:
        UB(R') <= sum_{i in I(R), residuals[i] > 0} residuals[i].
    """
    n_samples, n_literals = P_bool.shape
    u = residuals
    u_pos = np.maximum(u, 0.0)

    total_pos_ub = float(np.sum(u_pos))
    if total_pos_ub <= 1e-7:
        return None, 0.0

    best_score = 0.0
    best_conjunction: tuple[int, ...] | None = None

    # Step 1: Precompute support, score, and UB for all individual literals
    lit_scores = np.zeros(n_literals, dtype=float)
    lit_ubs = np.zeros(n_literals, dtype=float)
    lit_indices: list[np.ndarray | None] = [None] * n_literals

    for lit_idx in range(n_literals):
        col = P_bool[:, lit_idx]
        idx = np.flatnonzero(col)
        supp = len(idx)
        if supp < min_support:
            continue
        sc = float(np.sum(u[idx]))
        ub = float(np.sum(u_pos[idx]))
        lit_scores[lit_idx] = sc
        lit_ubs[lit_idx] = ub
        lit_indices[lit_idx] = idx

        if sc > best_score:
            best_score = sc
            best_conjunction = (lit_idx,)

    if max_depth <= 1:
        return best_conjunction, best_score

    # Step 2: Literal Screening & Ordering
    admissible = [
        lit_idx for lit_idx in range(n_literals)
        if lit_indices[lit_idx] is not None and lit_ubs[lit_idx] > best_score
    ]
    admissible.sort(key=lambda idx: lit_ubs[idx], reverse=True)
    if max_admissible is not None and len(admissible) > max_admissible:
        admissible = admissible[:max_admissible]

    if not admissible:
        return best_conjunction, best_score

    feat_of_lit = [meta[idx].feat_idx for idx in range(n_literals)]

    # Step 3: Priority Queue of candidate nodes
    pq: list[PricingNode] = []
    for rank, lit_idx in enumerate(admissible):
        pq.append(
            PricingNode(
                literals=(lit_idx,),
                indices=lit_indices[lit_idx],
                width=1,
                last_rank=rank,
                ub=lit_ubs[lit_idx],
                score=lit_scores[lit_idx],
            )
        )
    heapq.heapify(pq)

    nodes_explored = 0
    while pq and nodes_explored < max_nodes:
        node = heapq.heappop(pq)
        nodes_explored += 1

        if node.ub <= best_score:
            break

        if node.width >= max_depth:
            continue

        used_features = {feat_of_lit[l] for l in node.literals}

        # Sub-indexing: check subsequent admissible literals on node.indices only!
        for next_rank in range(node.last_rank + 1, len(admissible)):
            next_lit = admissible[next_rank]
            if lit_ubs[next_lit] <= best_score:
                break

            if feat_of_lit[next_lit] in used_features:
                continue

            sub_mask = P_bool[node.indices, next_lit]
            next_idx = node.indices[sub_mask]
            supp = len(next_idx)
            if supp < min_support:
                continue

            next_ub = float(np.sum(u_pos[next_idx]))
            if next_ub <= best_score:
                continue

            next_score = float(np.sum(u[next_idx]))
            new_lits = node.literals + (next_lit,)

            if next_score > best_score:
                best_score = next_score
                best_conjunction = new_lits

            if node.width + 1 < max_depth:
                heapq.heappush(
                    pq,
                    PricingNode(
                        literals=new_lits,
                        indices=next_idx,
                        width=node.width + 1,
                        last_rank=next_rank,
                        ub=next_ub,
                        score=next_score,
                    ),
                )

    return best_conjunction, best_score


class ColumnGenerationClassifier(BaseEstimator, ClassifierMixin):
    """Interpretable Scored Rule Set Classifier trained via Dantzig-Wolfe Column Generation.

    Parameters
    ----------
    max_rules : int, default=20
        Maximum number of rules to generate per binary class problem.
    max_depth : int, default=3
        Maximum number of literals (conjunction width) in a rule.
    min_support : int, default=3
        Minimum number of samples covered by a valid rule.
    n_bins : int, default=5
        Number of quantile bins used for continuous feature binarization.
    max_thresholds_per_feature : int, default=5
        Maximum cutoffs per continuous feature.
    c_reg : float, default=1.0
        Regularization strength for Restricted Master Problem (smaller C = more sparse).
    tol : float, default=1e-3
        Minimum reduced cost improvement threshold to continue column generation.
    pricing_max_nodes : int, default=1500
        Maximum branch-and-bound nodes explored per pricing iteration.
    pricing_max_admissible : int, default=60
        Maximum number of most promising literals to consider during conjunction search.
    random_state : int or None, default=None
    """

    def __init__(
        self,
        max_rules: int = 20,
        max_depth: int = 3,
        min_support: int = 3,
        n_bins: int = 5,
        max_thresholds_per_feature: int = 5,
        c_reg: float = 1.0,
        tol: float = 1e-3,
        pricing_max_nodes: int = 1500,
        pricing_max_admissible: int = 60,
        random_state: int | None = None,
    ):
        self.max_rules = max_rules
        self.max_depth = max_depth
        self.min_support = min_support
        self.n_bins = n_bins
        self.max_thresholds_per_feature = max_thresholds_per_feature
        self.c_reg = c_reg
        self.tol = tol
        self.pricing_max_nodes = pricing_max_nodes
        self.pricing_max_admissible = pricing_max_admissible
        self.random_state = random_state

    def fit(self, X, y, feature_names: list[str] | None = None):
        if hasattr(X, "columns"):
            detected_names = [str(c) for c in X.columns]
        else:
            detected_names = None

        X_arr, y_arr = check_X_y(X, y, accept_sparse=False, dtype="numeric")
        self.classes_ = unique_labels(y_arr)
        n_classes = len(self.classes_)
        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features

        if feature_names is not None:
            self.feature_names_in_ = list(feature_names)
        elif detected_names is not None and len(detected_names) == n_features:
            self.feature_names_in_ = detected_names
        else:
            self.feature_names_in_ = [f"f{j}" for j in range(n_features)]

        # 1. Feature Binarization
        self._thresholds_ = compute_nln_thresholds(
            X_arr,
            n_bins=self.n_bins,
            max_thresholds_per_feature=self.max_thresholds_per_feature,
        )
        P = binarize_with_thresholds(X_arr, self._thresholds_)
        P_bool = P.astype(bool)

        # Build Literal Metadata
        meta: list[LiteralSpec] = []
        prop_idx = 0
        for j, thrs in enumerate(self._thresholds_):
            for t in thrs:
                meta.append(LiteralSpec(feat_idx=j, op="<=", threshold=float(t), prop_idx=prop_idx))
                prop_idx += 1
            for t in thrs:
                meta.append(LiteralSpec(feat_idx=j, op=">", threshold=float(t), prop_idx=prop_idx))
                prop_idx += 1
        self.literal_meta_ = meta

        # 2. Fit Per Class (OvR for multiclass or single binary task)
        all_rules: list[Rule] = []
        rule_counter = 0

        y_idx = np.array([int(np.searchsorted(self.classes_, v)) for v in y_arr], dtype=int)
        classes_to_fit = [1] if n_classes == 2 else list(range(n_classes))
        default_scores = [0.0] * n_classes

        for c_idx in classes_to_fit:
            y_bin = (y_idx == c_idx).astype(float)
            target_class_idx = c_idx if n_classes > 2 else 1

            rules_c, w_0, _ = self._column_generation_binary(
                P_bool, y_bin, meta, target_class_idx, n_classes, rule_counter
            )
            rule_counter += len(rules_c)
            all_rules.extend(rules_c)

            if n_classes == 2:
                default_scores[1] = float(w_0)
            else:
                default_scores[c_idx] = float(w_0)

        # Assemble ScoredRuleSet
        default_rule = Rule(
            atoms=[],
            scores=default_scores,
            rule_id="cg_default",
            metadata={"source": "column_generation", "kind": "default"},
        )

        rs = ScoredRuleSet(
            class_labels=self.classes_.tolist(),
            feature_names=self.feature_names_in_,
            rules=[default_rule] + all_rules,
            aggregation=AggregationSpec(type="argmax_sum", temperature=1.0),
            metadata={
                "backend": "cg",
                "max_rules": self.max_rules,
                "max_depth": self.max_depth,
                "n_rules_generated": len(all_rules),
            },
        )
        rs.validate()
        self.ruleset_ = rs
        return self

    def _column_generation_binary(
        self,
        P_bool: np.ndarray,
        y_bin: np.ndarray,
        meta: list[LiteralSpec],
        target_class_idx: int,
        n_classes: int,
        start_rule_id: int,
    ) -> tuple[list[Rule], float, np.ndarray]:
        """Perform Column Generation for a binary indicator task."""
        n_samples = len(y_bin)

        # Candidate pool: list of literal tuples
        active_rules: list[tuple[int, ...]] = []
        active_masks: list[np.ndarray] = []

        # Initial RMP: only default bias (intercept)
        p_mean = float(np.clip(np.mean(y_bin), 1e-4, 1.0 - 1e-4))
        w_0 = float(np.log(p_mean / (1.0 - p_mean)))
        p_pred = np.full(n_samples, p_mean)

        seen_rules: set[tuple[int, ...]] = set()

        for it in range(self.max_rules):
            residuals = y_bin - p_pred

            # Pricing: Find rule maximizing positive residual
            best_lits_pos, score_pos = _solve_pricing_subproblem(
                P_bool,
                residuals,
                meta,
                max_depth=self.max_depth,
                min_support=self.min_support,
                max_admissible=self.pricing_max_admissible,
                max_nodes=self.pricing_max_nodes,
            )

            # Pricing: Find rule maximizing negative residual
            best_lits_neg, score_neg = _solve_pricing_subproblem(
                P_bool,
                -residuals,
                meta,
                max_depth=self.max_depth,
                min_support=self.min_support,
                max_admissible=self.pricing_max_admissible,
                max_nodes=self.pricing_max_nodes,
            )

            chosen_lits = None
            if score_pos >= score_neg and score_pos > self.tol:
                chosen_lits = best_lits_pos
            elif score_neg > score_pos and score_neg > self.tol:
                chosen_lits = best_lits_neg

            if chosen_lits is None:
                break

            sorted_lits = tuple(sorted(chosen_lits))
            if sorted_lits in seen_rules:
                break

            seen_rules.add(sorted_lits)
            active_rules.append(sorted_lits)

            mask = np.ones(n_samples, dtype=bool)
            for lit in sorted_lits:
                mask &= P_bool[:, lit]
            active_masks.append(mask)

            # Solve Restricted Master Problem (RMP) via L1-Logistic Regression
            A = np.column_stack([m.astype(float) for m in active_masks])
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                rmp_clf = LogisticRegression(
                    penalty="l1",
                    C=self.c_reg,
                    solver="liblinear",
                    fit_intercept=True,
                    random_state=self.random_state,
                    max_iter=500,
                )
                rmp_clf.fit(A, y_bin)

            w_0 = float(rmp_clf.intercept_[0])
            p_pred = rmp_clf.predict_proba(A)[:, 1]

        if not active_rules:
            return [], w_0, np.array([])

        A = np.column_stack([m.astype(float) for m in active_masks])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            rmp_clf = LogisticRegression(
                penalty="l1",
                C=self.c_reg,
                solver="liblinear",
                fit_intercept=True,
                random_state=self.random_state,
                max_iter=1000,
            )
            rmp_clf.fit(A, y_bin)
        w_0 = float(rmp_clf.intercept_[0])
        final_weights = rmp_clf.coef_[0]

        extracted_rules: list[Rule] = []
        for idx, (lits, w) in enumerate(zip(active_rules, final_weights)):
            if abs(w) < 1e-5:
                continue

            atoms = [
                meta[l].to_atom(self.feature_names_in_[meta[l].feat_idx])
                for l in lits
            ]
            scores = [0.0] * n_classes
            scores[target_class_idx] = float(w)

            extracted_rules.append(
                Rule(
                    atoms=atoms,
                    scores=scores,
                    rule_id=f"cg_{target_class_idx}_{start_rule_id + len(extracted_rules)}",
                    metadata={
                        "source": "column_generation",
                        "weight": float(w),
                        "width": len(atoms),
                        "target_class": target_class_idx,
                    },
                )
            )

        return extracted_rules, w_0, final_weights

    def predict(self, X):
        check_is_fitted(self, "ruleset_")
        X_arr = check_array(X, dtype="numeric")
        return predict_from_ruleset(self.ruleset_, X_arr)

    def predict_proba(self, X):
        check_is_fitted(self, "ruleset_")
        X_arr = check_array(X, dtype="numeric")
        return predict_proba_from_ruleset(self.ruleset_, X_arr)

    def to_ruleset(self) -> ScoredRuleSet:
        check_is_fitted(self, "ruleset_")
        return self.ruleset_


class ColumnGenerationRegressor(BaseEstimator, RegressorMixin):
    """Interpretable Scored Rule Set Regressor trained via Dantzig-Wolfe Column Generation.

    Parameters
    ----------
    max_rules : int, default=20
        Maximum number of rules to generate.
    max_depth : int, default=3
        Maximum number of literals (conjunction width) in a rule.
    min_support : int, default=3
        Minimum number of samples covered by a valid rule.
    n_bins : int, default=5
        Number of quantile bins used for continuous feature binarization.
    max_thresholds_per_feature : int, default=5
        Maximum cutoffs per continuous feature.
    alpha : float, default=0.05
        L1 regularization parameter for Lasso in Restricted Master Problem.
    tol : float, default=1e-3
        Minimum residual improvement to continue column generation.
    pricing_max_nodes : int, default=1500
        Maximum branch-and-bound nodes explored per pricing iteration.
    pricing_max_admissible : int, default=60
        Maximum number of most promising literals to consider during conjunction search.
    random_state : int or None, default=None
    """

    def __init__(
        self,
        max_rules: int = 20,
        max_depth: int = 3,
        min_support: int = 3,
        n_bins: int = 5,
        max_thresholds_per_feature: int = 5,
        alpha: float = 0.05,
        tol: float = 1e-3,
        pricing_max_nodes: int = 1500,
        pricing_max_admissible: int = 60,
        random_state: int | None = None,
    ):
        self.max_rules = max_rules
        self.max_depth = max_depth
        self.min_support = min_support
        self.n_bins = n_bins
        self.max_thresholds_per_feature = max_thresholds_per_feature
        self.alpha = alpha
        self.tol = tol
        self.pricing_max_nodes = pricing_max_nodes
        self.pricing_max_admissible = pricing_max_admissible
        self.random_state = random_state

    def fit(self, X, y, feature_names: list[str] | None = None):
        if hasattr(X, "columns"):
            detected_names = [str(c) for c in X.columns]
        else:
            detected_names = None

        X_arr, y_arr = check_X_y(X, y, accept_sparse=False, dtype="numeric")
        y_arr = y_arr.astype(float)
        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features

        if feature_names is not None:
            self.feature_names_in_ = list(feature_names)
        elif detected_names is not None and len(detected_names) == n_features:
            self.feature_names_in_ = detected_names
        else:
            self.feature_names_in_ = [f"f{j}" for j in range(n_features)]

        # 1. Feature Binarization
        self._thresholds_ = compute_nln_thresholds(
            X_arr,
            n_bins=self.n_bins,
            max_thresholds_per_feature=self.max_thresholds_per_feature,
        )
        P = binarize_with_thresholds(X_arr, self._thresholds_)
        P_bool = P.astype(bool)

        meta: list[LiteralSpec] = []
        prop_idx = 0
        for j, thrs in enumerate(self._thresholds_):
            for t in thrs:
                meta.append(LiteralSpec(feat_idx=j, op="<=", threshold=float(t), prop_idx=prop_idx))
                prop_idx += 1
            for t in thrs:
                meta.append(LiteralSpec(feat_idx=j, op=">", threshold=float(t), prop_idx=prop_idx))
                prop_idx += 1
        self.literal_meta_ = meta

        # 2. Column Generation Loop for Regression (Least Squares / Lasso RMP)
        active_rules: list[tuple[int, ...]] = []
        active_masks: list[np.ndarray] = []

        w_0 = float(np.mean(y_arr))
        y_pred = np.full(n_samples, w_0)
        seen_rules: set[tuple[int, ...]] = set()

        for it in range(self.max_rules):
            residuals = y_arr - y_pred

            best_lits_pos, score_pos = _solve_pricing_subproblem(
                P_bool,
                residuals,
                meta,
                max_depth=self.max_depth,
                min_support=self.min_support,
                max_admissible=self.pricing_max_admissible,
                max_nodes=self.pricing_max_nodes,
            )

            best_lits_neg, score_neg = _solve_pricing_subproblem(
                P_bool,
                -residuals,
                meta,
                max_depth=self.max_depth,
                min_support=self.min_support,
                max_admissible=self.pricing_max_admissible,
                max_nodes=self.pricing_max_nodes,
            )

            chosen_lits = None
            if score_pos >= score_neg and score_pos > self.tol:
                chosen_lits = best_lits_pos
            elif score_neg > score_pos and score_neg > self.tol:
                chosen_lits = best_lits_neg

            if chosen_lits is None:
                break

            sorted_lits = tuple(sorted(chosen_lits))
            if sorted_lits in seen_rules:
                break

            seen_rules.add(sorted_lits)
            active_rules.append(sorted_lits)

            mask = np.ones(n_samples, dtype=bool)
            for lit in sorted_lits:
                mask &= P_bool[:, lit]
            active_masks.append(mask)

            # Solve Restricted Master Problem via Lasso
            A = np.column_stack([m.astype(float) for m in active_masks])
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                rmp_reg = Lasso(
                    alpha=self.alpha,
                    fit_intercept=True,
                    random_state=self.random_state,
                    max_iter=1000,
                )
                rmp_reg.fit(A, y_arr)

            w_0 = float(rmp_reg.intercept_)
            y_pred = rmp_reg.predict(A)

        # 3. Final Master Problem Solve
        if not active_rules:
            final_rules = []
        else:
            A = np.column_stack([m.astype(float) for m in active_masks])
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                rmp_reg = Lasso(
                    alpha=self.alpha,
                    fit_intercept=True,
                    random_state=self.random_state,
                    max_iter=2000,
                )
                rmp_reg.fit(A, y_arr)
            w_0 = float(rmp_reg.intercept_)
            final_weights = rmp_reg.coef_

            final_rules: list[Rule] = []
            for idx, (lits, w) in enumerate(zip(active_rules, final_weights)):
                if abs(w) < 1e-6:
                    continue
                atoms = [
                    meta[l].to_atom(self.feature_names_in_[meta[l].feat_idx])
                    for l in lits
                ]
                final_rules.append(
                    Rule(
                        atoms=atoms,
                        scores=[float(w)],
                        rule_id=f"cg_reg_{len(final_rules)}",
                        metadata={
                            "source": "column_generation_regression",
                            "weight": float(w),
                            "width": len(atoms),
                        },
                    )
                )

        default_rule = Rule(
            atoms=[],
            scores=[float(w_0)],
            rule_id="cg_reg_default",
            metadata={"source": "column_generation_regression", "kind": "default"},
        )

        rs = ScoredRuleSet(
            class_labels=[],
            task_type="regression",
            feature_names=self.feature_names_in_,
            rules=[default_rule] + final_rules,
            aggregation=AggregationSpec(type="default_plus_sum"),
            metadata={
                "backend": "cg",
                "task": "regression",
                "max_rules": self.max_rules,
                "max_depth": self.max_depth,
                "n_rules_generated": len(final_rules),
            },
        )
        rs.validate()
        self.ruleset_ = rs
        return self

    def predict(self, X):
        check_is_fitted(self, "ruleset_")
        X_arr = check_array(X, dtype="numeric")
        return predict_regression_from_ruleset(self.ruleset_, X_arr)

    def to_ruleset(self) -> ScoredRuleSet:
        check_is_fitted(self, "ruleset_")
        return self.ruleset_

