"""Native Greedy Pareto Scored Rule Sets estimators.

Provides:
- GreedyParetoClassifier: Vectorized bitmask multi-objective greedy beam search
  constructing a Pareto front of Scored Rule Sets for classification.
- GreedyCascadedRegressor: 2-stage residual boosting greedy rule inducer
  with closed-form ridge calibration for regression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
import warnings

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import f1_score, r2_score
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ..runtime import predict as predict_from_ruleset
from ..runtime import predict_proba as predict_proba_from_ruleset
from ..runtime import predict_regression as predict_regression_from_ruleset
from ..schema import AggregationSpec, Atom, Rule, ScoredRuleSet
from .base import BaseRuleSetEstimator


@dataclass(frozen=True)
class _FastAtom:
    atom_id: int
    feature_idx: int
    op: str  # "<=" or ">"
    threshold: float


class GreedyParetoClassifier(BaseRuleSetEstimator, ClassifierMixin):
    """Deterministic, vectorized multi-objective greedy beam search for Scored Rule Sets.

    Directly constructs a Pareto front trading off structural complexity (atoms)
    against predictive performance (macro-F1) in a single pass without stochastic
    noise or population bloat.

    Parameters
    ----------
    max_complexity : int, default=12
        Maximum number of atomic conditions across the rule set.
    beam_width : int, default=4
        Number of candidate rule sets retained per complexity level.
    n_quantiles : int, default=5
        Number of quantile thresholds evaluated per numeric feature.
    preference : str, default="balanced"
        Selection criterion from the Pareto archive: "compact", "balanced", or "accuracy".
    random_state : int | None, default=None
        Ignored (algorithm is 100% deterministic), included for sklearn API compatibility.
    """

    def __init__(
        self,
        max_complexity: int = 12,
        beam_width: int = 4,
        n_quantiles: int = 5,
        preference: str = "balanced",
        random_state: int | None = None,
    ):
        self.max_complexity = max_complexity
        self.beam_width = beam_width
        self.n_quantiles = n_quantiles
        self.preference = preference
        self.random_state = random_state

    def fit(self, X, y, feature_names: Sequence[str] | None = None):
        X_arr, y_arr = check_X_y(X, y, dtype=None)
        self.n_features_in_ = X_arr.shape[1]

        # Extract or assign feature names
        if feature_names is not None:
            self.feature_names_ = list(feature_names)
        elif hasattr(X, "columns"):
            self.feature_names_ = [str(c) for c in X.columns]
        else:
            self.feature_names_ = [f"x{j}" for j in range(self.n_features_in_)]

        # Class labels mapping
        classes, y_encoded = np.unique(y_arr, return_inverse=True)
        self.classes_ = classes
        n_classes = len(classes)
        n_samples = len(y_encoded)

        class_counts = np.bincount(y_encoded, minlength=n_classes).astype(float)
        class_counts = np.maximum(class_counts, 1.0)

        # 1. Precompute candidate atoms and boolean masks (N, M)
        atoms: list[_FastAtom] = []
        atom_masks_list: list[np.ndarray] = []
        atom_id = 0

        for j in range(self.n_features_in_):
            col = X_arr[:, j].astype(float)
            qs = np.unique(np.quantile(col, np.linspace(0.15, 0.85, self.n_quantiles)))
            for q in qs:
                # <= q
                atoms.append(_FastAtom(atom_id, j, "<=", float(q)))
                atom_masks_list.append(col <= q)
                atom_id += 1
                # > q
                atoms.append(_FastAtom(atom_id, j, ">", float(q)))
                atom_masks_list.append(col > q)
                atom_id += 1

        if not atom_masks_list:
            A_masks = np.zeros((n_samples, 0), dtype=bool)
        else:
            A_masks = np.column_stack(atom_masks_list)
        n_atoms = len(atoms)

        # Helper: evaluate weights and macro F1 for a given set of rule masks
        def _evaluate_rule_masks(rule_masks: list[np.ndarray]):
            any_fired = np.zeros(n_samples, dtype=bool)
            scores = np.zeros((n_samples, n_classes), dtype=float)
            r_weights: list[list[float]] = []

            for r_mask in rule_masks:
                any_fired |= r_mask
                if not r_mask.any():
                    w = np.ones(n_classes, dtype=float) / n_classes
                else:
                    c = np.bincount(y_encoded[r_mask], minlength=n_classes).astype(float)
                    bal = c / class_counts
                    tot = float(bal.sum())
                    w = bal / tot if tot > 0 else np.ones(n_classes, dtype=float) / n_classes
                r_weights.append(w.tolist())
                if r_mask.any():
                    scores[r_mask] += w

            no_fire = ~any_fired
            if no_fire.any():
                c = np.bincount(y_encoded[no_fire], minlength=n_classes).astype(float)
                bal = c / class_counts
                tot = float(bal.sum())
                def_w = bal / tot if tot > 0 else np.ones(n_classes, dtype=float) / n_classes
            else:
                def_w = np.ones(n_classes, dtype=float) / n_classes

            scores[no_fire] += def_w
            preds = np.argmax(scores, axis=1)

            # Fast macro F1 calculation via bincount (no sklearn overhead)
            cm = np.bincount(n_classes * y_encoded + preds, minlength=n_classes * n_classes).reshape((n_classes, n_classes))
            tp = np.diag(cm)
            fp = np.sum(cm, axis=0) - tp
            fn = np.sum(cm, axis=1) - tp
            denom = 2 * tp + fp + fn
            f1s = np.where(denom > 0, (2.0 * tp) / denom, 0.0)
            f1 = float(np.mean(f1s))
            return f1, r_weights, def_w.tolist()

        # Helper: convert internal rule tuples to schema ScoredRuleSet
        def _to_schema_ruleset(rule_atom_ids: list[tuple[int, ...]], r_weights: list[list[float]], def_w: list[float]) -> ScoredRuleSet:
            rules: list[Rule] = []
            for r_idx, atom_tuple in enumerate(rule_atom_ids):
                rule_atoms = [
                    Atom(
                        feature=self.feature_names_[atoms[aid].feature_idx],
                        op=atoms[aid].op,
                        value=atoms[aid].threshold,
                    )
                    for aid in atom_tuple
                ]
                rules.append(Rule(atoms=rule_atoms, scores=r_weights[r_idx]))

            # Add default rule
            rules.append(Rule(atoms=[], scores=def_w))

            return ScoredRuleSet(
                class_labels=list(self.classes_),
                rules=rules,
                task_type="classification",
                feature_names=list(self.feature_names_),
                aggregation=AggregationSpec(type="argmax_sum"),
            )

        # 2. Multi-Objective Beam Search across complexity k = 1 ... max_complexity
        # state: (rule_atom_ids, rule_masks)
        archive: dict[int, tuple[float, list[tuple[int, ...]], list[list[float]], list[float]]] = {}
        beam: list[tuple[float, list[tuple[int, ...]], list[np.ndarray]]] = []

        # Complexity 1: single atoms
        c1_candidates = []
        for aid in range(n_atoms):
            r_mask = A_masks[:, aid]
            f1, r_w, d_w = _evaluate_rule_masks([r_mask])
            c1_candidates.append((f1, [(aid,)], [r_mask], r_w, d_w))

        if c1_candidates:
            c1_candidates.sort(key=lambda x: x[0], reverse=True)
            best_c1 = c1_candidates[0]
            archive[1] = (best_c1[0], best_c1[1], best_c1[3], best_c1[4])
            beam = [(c[0], c[1], c[2]) for c in c1_candidates[:self.beam_width]]

        for k in range(2, self.max_complexity + 1):
            next_candidates = []
            seen_signatures = set()

            for parent_f1, parent_rules, parent_masks in beam:
                parent_used_atoms = {aid for r in parent_rules for aid in r}

                # Op A: Specialization (c_j AND a)
                for r_idx, rule_tuple in enumerate(parent_rules):
                    if len(rule_tuple) >= 4:  # limit single rule depth
                        continue
                    rule_features = {atoms[aid].feature_idx for aid in rule_tuple}

                    for aid in range(n_atoms):
                        if aid in rule_tuple or atoms[aid].feature_idx in rule_features:
                            continue
                        new_mask = parent_masks[r_idx] & A_masks[:, aid]
                        if not new_mask.any():
                            continue

                        new_rules = list(parent_rules)
                        new_rules[r_idx] = tuple(sorted(rule_tuple + (aid,)))
                        sig = tuple(sorted(new_rules))
                        if sig in seen_signatures:
                            continue
                        seen_signatures.add(sig)

                        new_masks = list(parent_masks)
                        new_masks[r_idx] = new_mask
                        f1, r_w, d_w = _evaluate_rule_masks(new_masks)
                        next_candidates.append((f1, new_rules, new_masks, r_w, d_w))

                # Op B: Addition (add new rule)
                if len(parent_rules) < 10:
                    for aid in range(n_atoms):
                        if aid in parent_used_atoms:
                            continue
                        new_mask = A_masks[:, aid]
                        if not new_mask.any():
                            continue

                        # Keep rules and masks paired through the sort: the
                        # weight vector returned by _evaluate_rule_masks is
                        # aligned with the MASK order.  Sorting rules without
                        # permuting masks accordingly (the historical bug)
                        # scrambles rule<->weight pairing for every
                        # multi-rule candidate whose sort order differs from
                        # insertion order, silently corrupting the archive.
                        pairs = list(zip(parent_rules, parent_masks))
                        pairs.append(((aid,), new_mask))
                        pairs.sort(key=lambda p: p[0])
                        new_rules = [p[0] for p in pairs]
                        new_masks = [p[1] for p in pairs]
                        sig = tuple(new_rules)
                        if sig in seen_signatures:
                            continue
                        seen_signatures.add(sig)

                        f1, r_w, d_w = _evaluate_rule_masks(new_masks)
                        next_candidates.append((f1, new_rules, new_masks, r_w, d_w))

            if not next_candidates:
                break

            next_candidates.sort(key=lambda x: x[0], reverse=True)
            best_k = next_candidates[0]
            archive[k] = (best_k[0], best_k[1], best_k[3], best_k[4])
            beam = [(c[0], c[1], c[2]) for c in next_candidates[:self.beam_width]]

        # Store complete Pareto archive
        self.pareto_archive_: dict[int, ScoredRuleSet] = {}
        self.archive_scores_: dict[int, float] = {}

        for comp, (f1_val, r_tuples, r_w, d_w) in archive.items():
            rs = _to_schema_ruleset(r_tuples, r_w, d_w)
            rs.metadata["train_macro_f1"] = float(f1_val)
            rs.metadata["complexity_atoms"] = comp
            self.pareto_archive_[comp] = rs
            self.archive_scores_[comp] = f1_val

        # Select model based on preference
        self.ruleset_ = self._select_model_by_preference(self.preference)
        return self

    def _select_model_by_preference(self, preference: str) -> ScoredRuleSet:
        if not self.pareto_archive_:
            raise RuntimeError("No models found in Pareto archive.")

        comps = sorted(self.pareto_archive_.keys())
        if preference == "compact":
            # Highest score among complexity <= 4 (or minimum available)
            compact_comps = [c for c in comps if c <= 4]
            chosen = max(compact_comps or comps, key=lambda c: self.archive_scores_[c])
        elif preference == "accuracy":
            # Global maximum score
            chosen = max(comps, key=lambda c: self.archive_scores_[c])
        else:  # "balanced" / knee-point
            if len(comps) <= 2:
                chosen = max(comps, key=lambda c: self.archive_scores_[c])
            else:
                # Perpendicular distance to line connecting ends
                c_min, c_max = comps[0], comps[-1]
                s_min = self.archive_scores_[c_min]
                s_max = self.archive_scores_[c_max]
                best_dist = -1.0
                chosen = comps[0]

                denom = np.hypot(c_max - c_min, s_max - s_min)
                if denom < 1e-9:
                    chosen = max(comps, key=lambda c: self.archive_scores_[c])
                else:
                    for c in comps:
                        s = self.archive_scores_[c]
                        # Distance from point (c, s) to line (c_min, s_min)-(c_max, s_max)
                        num = abs((s_max - s_min) * c - (c_max - c_min) * s + c_max * s_min - s_max * c_min)
                        dist = num / denom
                        if dist > best_dist:
                            best_dist = dist
                            chosen = c

        return self.pareto_archive_[chosen]

    def to_ruleset(self) -> ScoredRuleSet:
        check_is_fitted(self, ["ruleset_"])
        return self.ruleset_

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, ["ruleset_"])
        return predict_from_ruleset(self.ruleset_, X)

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, ["ruleset_"])
        return predict_proba_from_ruleset(self.ruleset_, X)


class GreedyParetoRegressor(BaseRuleSetEstimator, RegressorMixin):
    """Deterministic, vectorized multi-objective greedy beam search for Scored Rule Sets regression.

    Directly constructs a Pareto front trading off structural complexity (atoms)
    against predictive performance (R2) via forward beam search with closed-form
    Ridge regression weight calculation.

    Parameters
    ----------
    max_complexity : int, default=12
        Maximum number of atomic conditions across the rule set.
    beam_width : int, default=4
        Number of candidate rule sets retained per complexity level.
    n_quantiles : int, default=5
        Number of quantile thresholds evaluated per numeric feature.
    ridge_alpha : float, default=1e-4
        L2 regularization parameter for closed-form weight solving.
    preference : str, default="balanced"
        Selection criterion from the Pareto archive: "compact", "balanced", or "accuracy".
    max_rules : int, default=6
        Maximum number of rules in any candidate rule set.
    random_state : int | None, default=None
        Ignored (algorithm is 100% deterministic), included for sklearn API compatibility.
    """

    def __init__(
        self,
        max_complexity: int = 12,
        beam_width: int = 4,
        n_quantiles: int = 5,
        ridge_alpha: float = 1e-4,
        preference: str = "balanced",
        max_rules: int = 6,
        random_state: int | None = None,
    ):
        self.max_complexity = max_complexity
        self.beam_width = beam_width
        self.n_quantiles = n_quantiles
        self.ridge_alpha = ridge_alpha
        self.preference = preference
        self.max_rules = max_rules
        self.random_state = random_state

    def fit(self, X, y, feature_names: Sequence[str] | None = None):
        X_arr, y_arr = check_X_y(X, y, dtype=None, y_numeric=True)
        y_arr = y_arr.astype(float)
        self.n_features_in_ = X_arr.shape[1]

        if feature_names is not None:
            self.feature_names_ = list(feature_names)
        elif hasattr(X, "columns"):
            self.feature_names_ = [str(c) for c in X.columns]
        else:
            self.feature_names_ = [f"x{j}" for j in range(self.n_features_in_)]

        n_samples = len(y_arr)
        y_mean = float(np.mean(y_arr))

        # 1. Precompute candidate atoms
        atoms: list[_FastAtom] = []
        atom_masks_list: list[np.ndarray] = []
        atom_id = 0

        for j in range(self.n_features_in_):
            col = X_arr[:, j].astype(float)
            qs = np.unique(np.quantile(col, np.linspace(0.1, 0.9, self.n_quantiles)))
            for q in qs:
                atoms.append(_FastAtom(atom_id, j, "<=", float(q)))
                atom_masks_list.append(col <= q)
                atom_id += 1

                atoms.append(_FastAtom(atom_id, j, ">", float(q)))
                atom_masks_list.append(col > q)
                atom_id += 1

        if not atom_masks_list:
            A_masks = np.zeros((n_samples, 0), dtype=bool)
        else:
            A_masks = np.column_stack(atom_masks_list)
        n_atoms = len(atoms)

        def _evaluate_rule_masks(rule_masks: list[np.ndarray]):
            if not rule_masks:
                preds = np.full(n_samples, y_mean)
                return float(r2_score(y_arr, preds)), [], y_mean

            M = np.column_stack(rule_masks).astype(float)
            M_ext = np.column_stack([np.ones(n_samples), M])
            K_ext = M_ext.shape[1]

            MtM = M_ext.T @ M_ext + self.ridge_alpha * np.eye(K_ext)
            Mty = M_ext.T @ y_arr
            weights = np.linalg.solve(MtM, Mty)

            w0 = float(weights[0])
            rw = [float(w) for w in weights[1:]]
            preds = M_ext @ weights
            r2 = float(r2_score(y_arr, preds))
            return r2, rw, w0

        def _to_schema_ruleset(rule_tuples: list[tuple[int, ...]], r_w: list[float], def_w: float) -> ScoredRuleSet:
            rules_out: list[Rule] = []
            for r_idx, atom_tuple in enumerate(rule_tuples):
                rule_atoms = [
                    Atom(
                        feature=self.feature_names_[atoms[aid].feature_idx],
                        op=atoms[aid].op,
                        value=atoms[aid].threshold,
                    )
                    for aid in atom_tuple
                ]
                rules_out.append(Rule(atoms=rule_atoms, scores=[r_w[r_idx]]))

            default_rule = Rule(atoms=[], scores=[def_w], metadata={"is_default": True})
            rules_out.append(default_rule)

            total_atoms = sum(len(r.atoms) for r in rules_out)
            return ScoredRuleSet(
                class_labels=[],
                rules=rules_out,
                task_type="regression",
                feature_names=list(self.feature_names_),
                aggregation=AggregationSpec(type="default_plus_sum"),
                metadata={
                    "total_atoms": total_atoms,
                    "n_rules": len(rule_tuples),
                },
            )

        # Baseline: empty rule set (mean default only)
        base_r2, _, _ = _evaluate_rule_masks([])
        beam_by_size: dict[int, list[dict]] = {
            0: [{"rules": [], "masks": [], "r2": base_r2, "rw": [], "w0": y_mean, "size": 0}]
        }

        for target_k in range(1, self.max_complexity + 1):
            candidates = []

            for prev_k in range(max(0, target_k - 2), target_k):
                if prev_k not in beam_by_size:
                    continue

                for parent in beam_by_size[prev_k]:
                    p_rules = parent["rules"]
                    p_masks = parent["masks"]

                    # Op 1: Add new 1-atom rule
                    if prev_k + 1 == target_k and len(p_rules) < self.max_rules:
                        existing_aids = {aid for r in p_rules for aid in r}
                        for aid in range(n_atoms):
                            if aid in existing_aids:
                                continue
                            new_mask = A_masks[:, aid]
                            if not new_mask.any() or np.all(new_mask):
                                continue

                            cand_rules = p_rules + [(aid,)]
                            cand_masks = p_masks + [new_mask]
                            cand_size = len({a for r in cand_rules for a in r})

                            if cand_size == target_k:
                                r2, rw, w0 = _evaluate_rule_masks(cand_masks)
                                candidates.append({
                                    "rules": cand_rules,
                                    "masks": cand_masks,
                                    "r2": r2,
                                    "rw": rw,
                                    "w0": w0,
                                    "size": cand_size,
                                })

                    # Op 2: Specialize an existing rule with an atom
                    for r_idx, (r_tuple, r_mask) in enumerate(zip(p_rules, p_masks)):
                        if len(r_tuple) >= 4:
                            continue
                        r_feat_indices = {atoms[aid].feature_idx for aid in r_tuple}
                        for aid in range(n_atoms):
                            if atoms[aid].feature_idx in r_feat_indices:
                                continue
                            ext_mask = r_mask & A_masks[:, aid]
                            if not ext_mask.any() or np.array_equal(ext_mask, r_mask):
                                continue

                            ext_tuple = r_tuple + (aid,)
                            cand_rules = list(p_rules)
                            cand_rules[r_idx] = ext_tuple
                            cand_masks = list(p_masks)
                            cand_masks[r_idx] = ext_mask
                            cand_size = len({a for r in cand_rules for a in r})

                            if cand_size == target_k:
                                r2, rw, w0 = _evaluate_rule_masks(cand_masks)
                                candidates.append({
                                    "rules": cand_rules,
                                    "masks": cand_masks,
                                    "r2": r2,
                                    "rw": rw,
                                    "w0": w0,
                                    "size": cand_size,
                                })

            if not candidates:
                continue

            candidates.sort(key=lambda item: item["r2"], reverse=True)
            beam_by_size[target_k] = candidates[:self.beam_width]

        # Extract Pareto front
        all_models = []
        for k in sorted(beam_by_size.keys()):
            for item in beam_by_size[k]:
                all_models.append((item["size"], item["r2"], item))

        all_models.sort(key=lambda m: (m[0], -m[1]))
        front_items = []
        max_r2 = -1e9
        for s, r2, item in all_models:
            if r2 > max_r2:
                front_items.append(item)
                max_r2 = r2

        self.pareto_archive_: dict[int, ScoredRuleSet] = {}
        self.archive_scores_: dict[int, float] = {}

        for item in front_items:
            s_val = item["size"]
            rs = _to_schema_ruleset(item["rules"], item["rw"], item["w0"])
            rs.metadata["train_r2"] = float(item["r2"])
            rs.metadata["complexity_atoms"] = s_val
            self.pareto_archive_[s_val] = rs
            self.archive_scores_[s_val] = item["r2"]

        # Select model based on preference
        self.ruleset_ = self._select_model_by_preference(self.preference)
        return self

    def _select_model_by_preference(self, preference: str) -> ScoredRuleSet:
        if not self.pareto_archive_:
            raise RuntimeError("No models found in Pareto archive.")

        comps = sorted(self.pareto_archive_.keys())
        if preference == "compact":
            compact_comps = [c for c in comps if c <= 4]
            chosen = max(compact_comps or comps, key=lambda c: self.archive_scores_[c])
        elif preference == "accuracy":
            chosen = max(comps, key=lambda c: self.archive_scores_[c])
        else:  # "balanced" / knee-point
            if len(comps) <= 2:
                chosen = max(comps, key=lambda c: self.archive_scores_[c])
            else:
                c_min, c_max = comps[0], comps[-1]
                s_min = self.archive_scores_[c_min]
                s_max = self.archive_scores_[c_max]
                best_dist = -1.0
                chosen = comps[0]

                denom = np.hypot(c_max - c_min, s_max - s_min)
                if denom < 1e-9:
                    chosen = max(comps, key=lambda c: self.archive_scores_[c])
                else:
                    for c in comps:
                        s = self.archive_scores_[c]
                        num = abs((s_max - s_min) * c - (c_max - c_min) * s + c_max * s_min - s_max * c_min)
                        dist = num / denom
                        if dist > best_dist:
                            best_dist = dist
                            chosen = c

        return self.pareto_archive_[chosen]

    def to_ruleset(self) -> ScoredRuleSet:
        check_is_fitted(self, ["ruleset_"])
        return self.ruleset_

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, ["ruleset_"])
        return predict_regression_from_ruleset(self.ruleset_, X)


class GreedyCascadedRegressor(BaseRuleSetEstimator, RegressorMixin):
    """Two-stage residual boosting greedy rule inducer with closed-form ridge calibration.

    Implements a 2-stage cascaded architecture matching EuroGP 2027:
    - Stage 1: Greedy beam search for macro-trend rule set on target y.
    - Stage 2: Greedy beam search for residual correction on error e = y - y_stage1.
    - Joint Fusion: Optimal closed-form Ridge recalibration across all Stage 1 + Stage 2 rules.

    Parameters
    ----------
    k_stage1 : int, default=3
        Max rules in Stage 1.
    k_stage2 : int, default=3
        Max rules in Stage 2.
    max_complexity_stage1 : int | None, default=None
        Max atoms in Stage 1 (defaults to k_stage1 * 2).
    max_complexity_stage2 : int | None, default=None
        Max atoms in Stage 2 (defaults to k_stage2 * 2).
    beam_width : int, default=4
        Beam width for greedy rule exploration.
    n_quantiles : int, default=5
        Number of quantile thresholds evaluated per feature.
    ridge_alpha : float, default=1e-4
        Regularization parameter for closed-form weight solving.
    random_state : int | None, default=None
        Ignored (100% deterministic), included for sklearn API compatibility.
    """

    def __init__(
        self,
        k_stage1: int = 3,
        k_stage2: int = 3,
        max_rules_stage1: int | None = None,
        max_rules_stage2: int | None = None,
        max_complexity_stage1: int | None = None,
        max_complexity_stage2: int | None = None,
        beam_width: int = 4,
        n_quantiles: int = 5,
        ridge_alpha: float = 1e-4,
        random_state: int | None = None,
    ):
        self.k_stage1 = k_stage1
        self.k_stage2 = k_stage2
        self.max_rules_stage1 = max_rules_stage1
        self.max_rules_stage2 = max_rules_stage2
        self.max_complexity_stage1 = max_complexity_stage1
        self.max_complexity_stage2 = max_complexity_stage2
        self.beam_width = beam_width
        self.n_quantiles = n_quantiles
        self.ridge_alpha = ridge_alpha
        self.random_state = random_state

    def fit(self, X, y, feature_names: Sequence[str] | None = None):
        X_arr, y_arr = check_X_y(X, y, dtype=None, y_numeric=True)
        y_arr = y_arr.astype(float)
        self.n_features_in_ = X_arr.shape[1]

        if feature_names is not None:
            self.feature_names_ = list(feature_names)
        elif hasattr(X, "columns"):
            self.feature_names_ = [str(c) for c in X.columns]
        else:
            self.feature_names_ = [f"x{j}" for j in range(self.n_features_in_)]

        n_samples = len(y_arr)
        max_r1 = self.max_rules_stage1 if self.max_rules_stage1 is not None else (self.k_stage1 * 2)
        max_r2 = self.max_rules_stage2 if self.max_rules_stage2 is not None else (self.k_stage2 * 2)
        comp1 = self.max_complexity_stage1 if self.max_complexity_stage1 is not None else (self.k_stage1 * 2)
        comp2 = self.max_complexity_stage2 if self.max_complexity_stage2 is not None else (self.k_stage2 * 2)

        # Stage 1: Macro trends on y
        self.stage1_ = GreedyParetoRegressor(
            max_complexity=comp1,
            max_rules=max_r1,
            beam_width=self.beam_width,
            n_quantiles=self.n_quantiles,
            ridge_alpha=self.ridge_alpha,
            preference="accuracy",
        )
        self.stage1_.fit(X_arr, y_arr, feature_names=self.feature_names_)
        preds1 = self.stage1_.predict(X_arr)

        # Stage 2: Residuals e = y - preds1
        res1 = y_arr - preds1
        self.stage2_ = GreedyParetoRegressor(
            max_complexity=comp2,
            max_rules=max_r2,
            beam_width=self.beam_width,
            n_quantiles=self.n_quantiles,
            ridge_alpha=self.ridge_alpha,
            preference="accuracy",
        )
        self.stage2_.fit(X_arr, res1, feature_names=self.feature_names_)

        # Extract non-default rules from Stage 1 and Stage 2
        rules1 = [r for r in self.stage1_.ruleset_.rules if getattr(r, "atoms", None)]
        rules2 = [r for r in self.stage2_.ruleset_.rules if getattr(r, "atoms", None)]

        tagged_rules: list[Rule] = []
        for r in rules1:
            meta = dict(getattr(r, "metadata", {}))
            meta["stage"] = 1
            tagged_rules.append(Rule(atoms=list(r.atoms), scores=list(r.scores), metadata=meta))

        for r in rules2:
            meta = dict(getattr(r, "metadata", {}))
            meta["stage"] = 2
            tagged_rules.append(Rule(atoms=list(r.atoms), scores=list(r.scores), metadata=meta))

        # Joint Ridge recalibration across all rules simultaneously
        if tagged_rules:
            feature_idx_map = {name: i for i, name in enumerate(self.feature_names_)}
            masks = []
            for r in tagged_rules:
                mask = np.ones(n_samples, dtype=bool)
                for atom in r.atoms:
                    idx = feature_idx_map[atom.feature]
                    val = X_arr[:, idx]
                    if atom.op == "<=":
                        mask &= (val <= atom.value)
                    elif atom.op == ">":
                        mask &= (val > atom.value)
                    elif atom.op == "==":
                        mask &= (val == atom.value)
                    elif atom.op == "!=":
                        mask &= (val != atom.value)
                masks.append(mask)

            M = np.column_stack(masks).astype(float)
            M_ext = np.column_stack([np.ones(n_samples), M])
            K_ext = M_ext.shape[1]

            MtM = M_ext.T @ M_ext + self.ridge_alpha * np.eye(K_ext)
            Mty = M_ext.T @ y_arr
            weights = np.linalg.solve(MtM, Mty)

            w0 = float(weights[0])
            rw = [float(w) for w in weights[1:]]

            # Update scores with optimal joint weights
            for r_idx, r in enumerate(tagged_rules):
                r.scores = [rw[r_idx]]

            combined_rules = tagged_rules + [Rule(atoms=[], scores=[w0], metadata={"is_default": True})]
        else:
            w0 = float(np.mean(y_arr))
            combined_rules = [Rule(atoms=[], scores=[w0], metadata={"is_default": True})]

        total_atoms = sum(len(r.atoms) for r in combined_rules)

        self.ruleset_ = ScoredRuleSet(
            class_labels=[],
            rules=combined_rules,
            task_type="regression",
            feature_names=list(self.feature_names_),
            aggregation=AggregationSpec(type="default_plus_sum"),
            metadata={
                "total_atoms": total_atoms,
                "n_rules_stage1": len(rules1),
                "n_rules_stage2": len(rules2),
            },
        )

        self.reg1_ = self.stage1_
        self.reg2_ = self.stage2_

        # Expose internal Pareto front combining Stage 1 models and the final cascaded model
        self.pareto_archive_: dict[int, ScoredRuleSet] = dict(getattr(self.stage1_, "pareto_archive_", {}))
        self.pareto_archive_[total_atoms] = self.ruleset_

        return self

    def get_pareto_spectrum(self) -> list[dict[str, Any]]:
        check_is_fitted(self, ["pareto_archive_"])
        spectrum = []
        for c in sorted(self.pareto_archive_.keys()):
            rs = self.pareto_archive_[c]
            spectrum.append({
                "complexity": c,
                "rules": len(rs.rules),
                "ruleset": rs,
            })
        return spectrum

    def select_ruleset(self, preference: str = "balanced") -> ScoredRuleSet:
        check_is_fitted(self, ["pareto_archive_"])
        if preference == "accuracy":
            return self.ruleset_
        if preference == "compact":
            min_c = min(self.pareto_archive_.keys())
            return self.pareto_archive_[min_c]
        return self.ruleset_

    def to_ruleset(self) -> ScoredRuleSet:
        check_is_fitted(self, ["ruleset_"])
        return self.ruleset_

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, ["ruleset_"])
        return predict_regression_from_ruleset(self.ruleset_, X)

