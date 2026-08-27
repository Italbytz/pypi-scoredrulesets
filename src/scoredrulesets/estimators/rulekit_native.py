"""
Pure-Python re-implementation of the RuleKit classification algorithm.

RuleKit (https://github.com/adaa-polsl/RuleKit) uses a *Separate-and-Conquer*
(sequential covering) strategy to learn an ordered list of classification rules.
Each rule is a conjunction of elementary conditions (atoms) on individual
features.  The algorithm works as follows:

1. **Sequential Covering** – iteratively learn rules, remove covered examples.
2. **Growing** – greedily add the best condition (highest quality gain) to the
   current rule.  Conditions are of the form ``feature op threshold`` where
   *op* ∈ {<=, >, ==} for numeric features and ``== category`` for nominal
   features.  RuleKit uses the C2 quality measure: ``(p - n) / (P + N)``
   where *p*/*n* are covered positives/negatives and *P*/*N* are total
   positives/negatives in the full dataset.
3. **Pruning** – after growing, conditions are removed one at a time (back to
   front) if removal does not decrease rule quality on a held-out pruning
   set (Reduced Error Pruning, REP).

For each round the algorithm grows a candidate rule for **every** class and
selects the one with the highest quality.  After selection all examples
of the rule's target class that are covered by the rule are removed from
the uncovered set.

Each induced rule receives a **one-hot score** weighted by its precision
(confidence) on the covered examples — this matches the Java-RuleKit
voting scheme.

The existing Java-based backend (``backend="rulekit"``) is **not** changed –
this module provides an independent ``backend="rulekit_native"`` alternative
that requires no JVM.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ..runtime import predict as predict_from_ruleset
from ..runtime import predict_proba as predict_proba_from_ruleset
from ..schema import AggregationSpec, Atom, Rule, ScoredRuleSet
from .atom_selection import (
    available_atom_selection_strategies,
    is_atom_selection_strategy_available,
    select_signatures_by_strategy,
)
from .base import BaseRuleSetEstimator
from ._split_utils import distribution_to_scores
from ._time_budget import FitBudgetExceededError


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class _ElementaryCondition:
    """A single condition ``feature op value``.

    For interval conditions (like Java RuleKit), *op* is ``"in"`` and *value*
    is a tuple ``(lower, upper)`` representing ``lower <= feature < upper``.
    """
    feature_idx: int
    feature_name: str
    op: str          # one of <=, >, <, >=, ==, in
    value: Any

    def matches_array(self, X: np.ndarray) -> np.ndarray:
        """Return boolean mask of shape (n_samples,)."""
        col = X[:, self.feature_idx]
        if self.op == "<=":
            return col <= self.value
        if self.op == ">":
            return col > self.value
        if self.op == "<":
            return col < self.value
        if self.op == ">=":
            return col >= self.value
        if self.op == "==":
            return np.asarray(col == self.value, dtype=bool)
        if self.op == "in":
            lo, hi = self.value
            return (col >= lo) & (col < hi)
        raise ValueError(f"Unsupported op: {self.op}")


@dataclass
class _InducedRule:
    """An induced rule: conjunction of elementary conditions + predicted class."""
    conditions: list[_ElementaryCondition]
    class_idx: int
    quality: float       # C2 quality
    coverage: int        # number of examples covered
    precision: float     # p / (p + n) on covered examples
    preselected_atoms: int | None = None


# ---------------------------------------------------------------------------
# Quality measure – RuleKit C2
# ---------------------------------------------------------------------------

def _c2_quality(
    p: float,
    n: float,
    P: float,
    N: float,
) -> float:
    """RuleKit C2 quality measure: (p - n) / (P + N).

    Parameters
    ----------
    p : covered positives (target class)
    n : covered negatives (other classes)
    P : total positives in full dataset
    N : total negatives in full dataset
    """
    denom = P + N
    if denom <= 0:
        return 0.0
    return float((p - n) / denom)


# ---------------------------------------------------------------------------
# Growing: greedy condition addition
# ---------------------------------------------------------------------------

def _compute_covered_pn(
    y_idx: np.ndarray,
    mask: np.ndarray,
    class_idx: int,
    n_classes: int,
) -> tuple[float, float]:
    """Return (p, n) for the subset of y_idx selected by *mask*."""
    counts = np.bincount(y_idx[mask], minlength=n_classes).astype(float)
    p = counts[class_idx]
    n = float(np.sum(counts)) - p
    return p, n


def _condition_signature(cond: _ElementaryCondition) -> tuple[int, str, Any]:
    """Create a hashable signature for a condition."""
    value: Any
    if cond.op == "in":
        lo, hi = cond.value
        value = (float(lo), float(hi))
    else:
        value = cond.value.item() if isinstance(cond.value, np.generic) else cond.value
    return (cond.feature_idx, cond.op, value)


def _iter_candidate_conditions(
    X: np.ndarray,
    active_mask: np.ndarray,
    feature_names: list[str],
    max_thresholds: int | None,
    enable_intervals: bool,
    feature_indices: np.ndarray | None,
):
    """Yield candidate conditions and their boolean masks on X."""
    if feature_indices is None:
        feature_iter = range(X.shape[1])
    else:
        feature_iter = (int(i) for i in feature_indices.tolist())

    for feat_idx in feature_iter:
        col = X[:, feat_idx]
        is_numeric = np.issubdtype(col.dtype, np.number)

        if is_numeric:
            vals = col.astype(float)
            active_vals = vals[active_mask]
            unique_vals = np.unique(active_vals)
            if unique_vals.size < 2:
                continue
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
            if max_thresholds is not None and len(thresholds) > max_thresholds:
                idx = np.round(
                    np.linspace(0, len(thresholds) - 1, max_thresholds)
                ).astype(int)
                thresholds = thresholds[idx]

            for thr in thresholds:
                for op in ("<=", ">"):
                    cand_mask = (vals <= thr) if op == "<=" else (vals > thr)
                    yield _ElementaryCondition(
                        feat_idx,
                        feature_names[feat_idx],
                        op,
                        float(thr),
                    ), cand_mask

            if enable_intervals and len(thresholds) >= 2:
                if len(thresholds) > 30:
                    iv_idx = np.round(
                        np.linspace(0, len(thresholds) - 1, 30)
                    ).astype(int)
                    iv_thresholds = thresholds[iv_idx]
                else:
                    iv_thresholds = thresholds

                for i_lo in range(len(iv_thresholds)):
                    lo = float(iv_thresholds[i_lo])
                    for i_hi in range(i_lo + 1, len(iv_thresholds)):
                        hi = float(iv_thresholds[i_hi])
                        cand_mask = (vals >= lo) & (vals < hi)
                        yield _ElementaryCondition(
                            feat_idx,
                            feature_names[feat_idx],
                            "in",
                            (lo, hi),
                        ), cand_mask
        else:
            active_col = col[active_mask]
            categories = np.unique(active_col)
            if categories.size < 2:
                continue
            for cat in categories:
                cand_mask = np.asarray(col == cat, dtype=bool)
                value = cat.item() if isinstance(cat, np.generic) else cat
                yield _ElementaryCondition(
                    feat_idx,
                    feature_names[feat_idx],
                    "==",
                    value,
                ), cand_mask


def _preselect_atom_signatures(
    X: np.ndarray,
    y_idx: np.ndarray,
    n_classes: int,
    class_idx: int,
    feature_names: list[str],
    min_samples_leaf: int,
    max_thresholds: int | None,
    P: float,
    N: float,
    enable_intervals: bool,
    strategy: str,
    top_k: int | None,
) -> set[tuple[int, str, Any]] | None:
    """Select candidate atoms once per grown rule and return their signatures."""
    if strategy == "none":
        return None

    if not is_atom_selection_strategy_available(strategy):
        available = ", ".join(available_atom_selection_strategies())
        raise ValueError(
            "atom_preselection_strategy must be 'none' or a registered "
            f"atom-selection strategy. Available registered strategies: [{available}]"
        )
    if top_k is None or int(top_k) <= 0:
        raise ValueError(
            "atom_preselection_top_k must be a positive integer when "
            "atom_preselection_strategy != 'none'."
        )

    active_mask = np.ones(X.shape[0], dtype=bool)
    candidates: list[tuple[tuple[int, str, Any], np.ndarray]] = []
    for cond, cand_mask in _iter_candidate_conditions(
        X,
        active_mask,
        feature_names,
        max_thresholds,
        enable_intervals,
        feature_indices=None,
    ):
        candidates.append((_condition_signature(cond), cand_mask))

    if not candidates:
        return None

    return select_signatures_by_strategy(
        strategy=strategy,
        candidates=candidates,
        y_idx=y_idx,
        n_classes=n_classes,
        min_samples_leaf=min_samples_leaf,
        top_k=int(top_k),
    )


def _grow_rule(
    X: np.ndarray,
    y_idx: np.ndarray,
    n_classes: int,
    class_idx: int,
    feature_names: list[str],
    min_samples_leaf: int,
    max_conditions: int,
    max_thresholds: int | None,
    P: float,
    N: float,
    enable_intervals: bool = True,
    atom_preselection_strategy: str = "none",
    atom_preselection_top_k: int | None = None,
    allowed_condition_signatures: set[tuple[int, str, Any]] | None = None,
    fit_deadline: float | None = None,
) -> _InducedRule | None:
    """Grow a single rule by greedy condition addition.

    At each step we evaluate *all* candidate elementary conditions on the
    examples currently covered by the partial rule and add the one that
    maximises the C2 quality measure.  Growing stops when:
      * the rule is pure (no negatives covered),
      * no condition improves quality,
      * the minimum-leaf constraint cannot be met, or
      * the maximum number of conditions is reached.

    When *enable_intervals* is True (the default), interval conditions of the
    form ``lower <= feature < upper`` are evaluated as single candidates –
    matching the Java RuleKit approach where each elementary condition is an
    interval.  This allows the algorithm to learn intervals in a single growing
    step rather than requiring two steps.

    ``P`` and ``N`` are the *full-dataset* class counts so that the quality
    measure is comparable across growing steps and across rules for different
    classes.
    """
    n_samples = X.shape[0]
    conditions: list[_ElementaryCondition] = []
    active_mask = np.ones(n_samples, dtype=bool)

    p_cur, n_cur = _compute_covered_pn(y_idx, active_mask, class_idx, n_classes)
    current_quality = -np.inf
    if allowed_condition_signatures is not None:
        allowed_signatures = allowed_condition_signatures
    else:
        allowed_signatures = _preselect_atom_signatures(
            X,
            y_idx,
            n_classes,
            class_idx,
            feature_names,
            min_samples_leaf,
            max_thresholds,
            P,
            N,
            enable_intervals,
            atom_preselection_strategy,
            atom_preselection_top_k,
        )

    for step in range(max_conditions):
        if fit_deadline is not None and time.monotonic() >= fit_deadline:
            break
        if active_mask.sum() < min_samples_leaf:
            break
        if n_cur <= 0:
            break

        best_cond: _ElementaryCondition | None = None
        best_q = current_quality  # must *strictly* improve
        best_cov: int = 0         # tiebreaker: prefer more coverage

        def _evaluate_candidate(
            cand: _ElementaryCondition,
            new_mask: np.ndarray,
        ) -> None:
            """Check if *cand* is better than the current best."""
            nonlocal best_cond, best_q, best_cov
            n_cov = int(new_mask.sum())
            if n_cov < min_samples_leaf:
                return
            p_c, n_c = _compute_covered_pn(y_idx, new_mask, class_idx, n_classes)
            q = _c2_quality(p_c, n_c, P, N)
            # Strict improvement OR same quality but more coverage (broader rule)
            if q > best_q + 1e-12 or (abs(q - best_q) <= 1e-12 and n_cov > best_cov):
                best_q = q
                best_cov = n_cov
                best_cond = cand

        for cond, cand_mask in _iter_candidate_conditions(
            X,
            active_mask,
            feature_names,
            max_thresholds,
            enable_intervals,
            feature_indices=None,
        ):
            if allowed_signatures is not None:
                if _condition_signature(cond) not in allowed_signatures:
                    continue
            new_mask = active_mask & cand_mask
            _evaluate_candidate(cond, new_mask)

        if best_cond is None:
            break

        conditions.append(best_cond)
        active_mask = active_mask & best_cond.matches_array(X)
        p_cur, n_cur = _compute_covered_pn(y_idx, active_mask, class_idx, n_classes)
        current_quality = best_q

    if not conditions:
        return None

    coverage = int(active_mask.sum())
    precision = float(p_cur / max(p_cur + n_cur, 1e-12))

    return _InducedRule(
        conditions=conditions,
        class_idx=class_idx,
        quality=current_quality,
        coverage=coverage,
        precision=precision,
        preselected_atoms=(
            len(allowed_signatures) if allowed_signatures is not None else None
        ),
    )


# ---------------------------------------------------------------------------
# Pruning: reduced error pruning (REP)
# ---------------------------------------------------------------------------

def _prune_rule(
    rule: _InducedRule,
    X_prune: np.ndarray,
    y_prune: np.ndarray,
    n_classes: int,
    P: float,
    N: float,
) -> _InducedRule:
    """Remove conditions from the rule (back to front) if quality does not drop.

    This implements RuleKit's post-growing pruning phase.
    """
    if len(rule.conditions) <= 1:
        return rule

    def _quality_of(conditions: list[_ElementaryCondition]) -> float:
        mask = np.ones(X_prune.shape[0], dtype=bool)
        for c in conditions:
            mask &= c.matches_array(X_prune)
        if mask.sum() == 0:
            return -np.inf
        p, n = _compute_covered_pn(y_prune, mask, rule.class_idx, n_classes)
        return _c2_quality(p, n, P, N)

    current_conditions = list(rule.conditions)
    current_quality = _quality_of(current_conditions)

    # Iterate backwards; restart scan after each removal
    changed = True
    while changed and len(current_conditions) > 1:
        changed = False
        for i in range(len(current_conditions) - 1, -1, -1):
            reduced = current_conditions[:i] + current_conditions[i + 1:]
            if not reduced:
                continue
            q = _quality_of(reduced)
            if q >= current_quality - 1e-9:
                current_conditions = reduced
                current_quality = q
                changed = True
                break

    # Recompute coverage and precision on prune set
    mask = np.ones(X_prune.shape[0], dtype=bool)
    for c in current_conditions:
        mask &= c.matches_array(X_prune)
    coverage = int(mask.sum())
    p, n = _compute_covered_pn(y_prune, mask, rule.class_idx, n_classes)
    precision = float(p / max(p + n, 1e-12))

    return _InducedRule(
        conditions=current_conditions,
        class_idx=rule.class_idx,
        quality=current_quality,
        coverage=coverage,
        precision=precision,
        preselected_atoms=rule.preselected_atoms,
    )


# ---------------------------------------------------------------------------
# Conversion to ScoredRuleSet Rules
# ---------------------------------------------------------------------------

def _induced_rule_to_scored_rule(
    rule: _InducedRule,
    n_classes: int,
    rule_idx: int,
) -> Rule:
    """Convert an _InducedRule to a ScoredRuleSet Rule.

    Uses **one-hot scoring** weighted by the rule's precision (confidence),
    matching the Java-RuleKit voting scheme where each rule votes for its
    predicted class with a strength proportional to its confidence.

    Interval conditions (``op="in"``, ``value=(lo, hi)``) are expanded into
    two atoms: ``feature >= lo`` and ``feature < hi``.
    """
    atoms = []
    for c in rule.conditions:
        if c.op == "in":
            lo, hi = c.value
            atoms.append(Atom(feature=c.feature_name, op=">=", value=lo))
            atoms.append(Atom(feature=c.feature_name, op="<", value=hi))
        else:
            atoms.append(Atom(feature=c.feature_name, op=c.op, value=c.value))

    # One-hot score: only the target class gets a non-zero score
    scores = [0.0] * n_classes
    scores[rule.class_idx] = max(rule.precision, 1e-6)

    return Rule(
        atoms=atoms,
        scores=scores,
        rule_id=f"rulekit_native_{rule_idx}",
        metadata={
            "source": "rulekit_native",
            "class_idx": rule.class_idx,
            "quality": rule.quality,
            "precision": rule.precision,
            "coverage": rule.coverage,
            "n_conditions": len(rule.conditions),
            "preselected_atoms": rule.preselected_atoms,
        },
    )


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

class RuleKitNativeClassifier(BaseRuleSetEstimator):
    """Pure-Python re-implementation of the RuleKit classification algorithm.

    This classifier learns an ordered rule list using the Separate-and-Conquer
    (sequential covering) strategy with greedy condition growing and optional
    Reduced Error Pruning (REP).

    Parameters
    ----------
    max_rules : int
        Maximum number of rules to learn (excluding the default rule).
    max_conditions : int
        Maximum number of elementary conditions per rule (conjunction depth).
    min_samples_leaf : int
        Minimum number of examples that must be covered by a rule.
    min_rule_covered : int
        Stop learning rules when fewer than this many uncovered examples remain.
    enable_pruning : bool
        Whether to apply Reduced Error Pruning after growing each rule.
    pruning_mode : str
        How to split data for pruning.  ``"same"`` (default) grows and prunes
        on the same data – matching the Java RuleKit behaviour.  ``"split"``
        reserves a fraction (see *pruning_fraction*) of each uncovered set
        as a held-out pruning set.
    pruning_fraction : float
        Fraction of training data reserved for pruning (only used when
        *pruning_mode* is ``"split"``).
    enable_intervals : bool
        If True (default), interval conditions ``[lower, upper)`` are
        evaluated as single candidates during growing – matching Java
        RuleKit's approach.
    validation_fraction : float
        Fraction of training data used as validation set for overall stopping.
    max_thresholds_per_feature : int | None
        Cap the number of candidate thresholds per numeric feature.
    atom_preselection_strategy : str
        Candidate-atom preselection strategy. ``"none"`` evaluates all atoms,
        otherwise a registered plugin strategy preselects atoms before growing.
    atom_preselection_top_k : int | None
        Number of atoms kept when a preselection strategy is used.
    aggregation : str
        Aggregation type for the ScoredRuleSet (``"argmax_sum"`` or
        ``"softmax_sum"``).
    temperature : float
        Temperature for softmax aggregation.
    max_fit_seconds : float | None
        Maximum wall-clock runtime for the complete fit in seconds. Once at
        least one rule has been induced, sequential covering stops cleanly when
        the budget is exhausted and the induced rules so far are returned. If
        the budget is exhausted during setup, before a single rule is induced,
        a :class:`~scoredrulesets.FitBudgetExceededError` is raised instead,
        since no meaningful model exists yet.
    random_state : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        max_rules: int = 20,
        max_conditions: int = 5,
        min_samples_leaf: int = 5,
        min_rule_covered: int = 5,
        enable_pruning: bool = True,
        pruning_mode: str = "same",
        pruning_fraction: float = 0.33,
        enable_intervals: bool = True,
        validation_fraction: float = 0.0,
        max_thresholds_per_feature: int | None = None,
        atom_preselection_strategy: str = "none",
        atom_preselection_top_k: int | None = None,
        aggregation: str = "argmax_sum",
        temperature: float = 1.0,
        max_fit_seconds: float | None = None,
        random_state: int | None = None,
    ):
        self.max_rules = max_rules
        self.max_conditions = max_conditions
        self.min_samples_leaf = min_samples_leaf
        self.min_rule_covered = min_rule_covered
        self.enable_pruning = enable_pruning
        self.pruning_mode = pruning_mode
        self.pruning_fraction = pruning_fraction
        self.enable_intervals = enable_intervals
        self.validation_fraction = validation_fraction
        self.max_thresholds_per_feature = max_thresholds_per_feature
        self.atom_preselection_strategy = atom_preselection_strategy
        self.atom_preselection_top_k = atom_preselection_top_k
        self.aggregation = aggregation
        self.temperature = temperature
        self.max_fit_seconds = max_fit_seconds
        self.random_state = random_state

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
    # fit
    # ------------------------------------------------------------------

    def fit(self, X, y):
        X_valid, y_valid = check_X_y(X, y, dtype=None)
        self.n_features_in_ = X_valid.shape[1]
        self.feature_names_in_ = np.asarray(
            [f"f{i}" for i in range(self.n_features_in_)], dtype=object,
        )
        self.classes_ = unique_labels(y_valid)
        self._rng_ = np.random.default_rng(self.random_state)

        class_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
        y_idx = np.asarray([class_to_idx[v] for v in y_valid], dtype=int)
        n_classes = len(self.classes_)
        feature_names = self.feature_names_in_.tolist()

        # Optional validation split
        if self.validation_fraction > 0:
            _, counts = np.unique(y_idx, return_counts=True)
            if np.all(counts >= 2):
                train_idx, _val_idx = train_test_split(
                    np.arange(len(y_idx)),
                    test_size=float(self.validation_fraction),
                    random_state=self.random_state,
                    stratify=y_idx,
                )
                X_work = X_valid[train_idx]
                y_work = y_idx[train_idx]
            else:
                X_work, y_work = X_valid, y_idx
        else:
            X_work, y_work = X_valid, y_idx

        # Full-dataset class counts (constant for C2 quality measure)
        full_counts = np.bincount(y_work, minlength=n_classes).astype(float)

        preselected_signatures: set[tuple[int, str, Any]] | None = None
        if self.atom_preselection_strategy != "none":
            candidates: list[tuple[tuple[int, str, Any], np.ndarray]] = []
            all_active = np.ones(X_work.shape[0], dtype=bool)
            for cond, cand_mask in _iter_candidate_conditions(
                X_work,
                all_active,
                feature_names,
                self.max_thresholds_per_feature,
                self.enable_intervals,
                feature_indices=None,
            ):
                candidates.append((_condition_signature(cond), cand_mask))

            preselected_signatures = select_signatures_by_strategy(
                strategy=self.atom_preselection_strategy,
                candidates=candidates,
                y_idx=y_work,
                n_classes=n_classes,
                min_samples_leaf=self.min_samples_leaf,
                top_k=int(self.atom_preselection_top_k),
            )

        # ---- Sequential Covering (class-first, like Java RuleKit) ----
        # For each class c, learn rules until no uncovered positives remain
        # or no improving rule can be found.  Then move to the next class.
        induced_rules: list[_InducedRule] = []
        uncovered_masks: list[np.ndarray] = [
            np.ones(X_work.shape[0], dtype=bool) for _ in range(n_classes)
        ]
        fit_deadline = None
        if self.max_fit_seconds is not None:
            fit_deadline = time.monotonic() + float(self.max_fit_seconds)

        for c_idx in range(n_classes):
            if fit_deadline is not None and time.monotonic() >= fit_deadline:
                break

            P = full_counts[c_idx]
            N = float(np.sum(full_counts)) - P
            if P <= 0:
                continue

            rules_for_class = 0
            max_rules_per_class = max(1, self.max_rules)

            while rules_for_class < max_rules_per_class:
                if fit_deadline is not None and time.monotonic() >= fit_deadline:
                    break

                # Count uncovered positives for this class
                uncov = uncovered_masks[c_idx]
                n_uncov_pos = int(np.sum(
                    (y_work == c_idx) & uncov
                ))
                if n_uncov_pos < self.min_rule_covered:
                    break

                # Working set: all uncovered examples for this class
                X_uncov = X_work[uncov]
                y_uncov = y_work[uncov]

                if X_uncov.shape[0] < self.min_samples_leaf:
                    break

                # ---- Grow/Prune split ----
                if (self.enable_pruning
                        and self.pruning_mode == "split"
                        and self.pruning_fraction > 0
                        and X_uncov.shape[0] >= 2 * self.min_samples_leaf):
                    _, uc_counts = np.unique(y_uncov, return_counts=True)
                    if np.all(uc_counts >= 2):
                        grow_idx, prune_idx = train_test_split(
                            np.arange(X_uncov.shape[0]),
                            test_size=float(self.pruning_fraction),
                            random_state=self.random_state,
                            stratify=y_uncov,
                        )
                        X_grow, y_grow = X_uncov[grow_idx], y_uncov[grow_idx]
                        X_prune, y_prune = X_uncov[prune_idx], y_uncov[prune_idx]
                    else:
                        X_grow, y_grow = X_uncov, y_uncov
                        X_prune, y_prune = X_uncov, y_uncov
                else:
                    # Default: grow and prune on same data (like Java RuleKit)
                    X_grow, y_grow = X_uncov, y_uncov
                    X_prune, y_prune = X_uncov, y_uncov

                # ---- Grow rule for this class ----
                candidate = _grow_rule(
                    X_grow, y_grow, n_classes, c_idx,
                    feature_names, self.min_samples_leaf,
                    self.max_conditions, self.max_thresholds_per_feature,
                    P, N,
                    enable_intervals=self.enable_intervals,
                    atom_preselection_strategy=self.atom_preselection_strategy,
                    atom_preselection_top_k=self.atom_preselection_top_k,
                    allowed_condition_signatures=preselected_signatures,
                    fit_deadline=fit_deadline,
                )
                if candidate is None:
                    break

                # ---- Prune ----
                if self.enable_pruning:
                    candidate = _prune_rule(
                        candidate, X_prune, y_prune, n_classes, P, N,
                    )

                # Quality gate
                if candidate.quality <= 0 or candidate.precision <= 0:
                    break

                # Duplicate check
                cond_sig = tuple(
                    (c.feature_idx, c.op, c.value) for c in candidate.conditions
                )
                existing_sigs = {
                    tuple((c.feature_idx, c.op, c.value) for c in r.conditions)
                    for r in induced_rules
                }
                if cond_sig in existing_sigs:
                    break

                induced_rules.append(candidate)
                rules_for_class += 1

                # ---- Remove covered positives of this class ----
                indices = np.where(uncov)[0]
                X_active = X_work[uncov]
                y_active = y_work[uncov]
                rule_mask = np.ones(X_active.shape[0], dtype=bool)
                for cond in candidate.conditions:
                    rule_mask &= cond.matches_array(X_active)
                positive_covered = rule_mask & (y_active == c_idx)
                cover_indices = indices[positive_covered]
                uncovered_masks[c_idx][cover_indices] = False

                # If this rule didn't cover any new positives, stop
                if not positive_covered.any():
                    break

        # If the budget was exhausted during setup (feature discretization and
        # atom preselection) before a single rule could be induced, no
        # meaningful model exists.  Surface a clear timeout instead of returning
        # a degenerate default-only rule set.
        if (
            not induced_rules
            and fit_deadline is not None
            and time.monotonic() >= fit_deadline
        ):
            raise FitBudgetExceededError(
                "max_fit_seconds exhausted during setup before the first rule "
                "was induced; the estimator is not viable within this budget."
            )

        # ---- Build ScoredRuleSet ----
        rules: list[Rule] = []

        # Default rule from class prior (weak uniform fallback)
        prior_counts = full_counts.copy()
        default_scores = distribution_to_scores(prior_counts, self.aggregation)
        rules.append(Rule(
            atoms=[],
            scores=default_scores,
            rule_id="rulekit_native_default",
            metadata={"source": "rulekit_native", "kind": "class_prior"},
        ))

        for i, irule in enumerate(induced_rules):
            rules.append(_induced_rule_to_scored_rule(irule, n_classes, i))

        self.ruleset_ = ScoredRuleSet(
            class_labels=self.classes_.tolist(),
            feature_names=feature_names,
            aggregation=AggregationSpec(
                type=self.aggregation, temperature=self.temperature,
            ),
            rules=rules,
            metadata={
                "source": "rulekit_native",
                "model_type": "separate_and_conquer",
                "n_rules_induced": len(induced_rules),
                "enable_pruning": self.enable_pruning,
                "pruning_mode": self.pruning_mode,
                "enable_intervals": self.enable_intervals,
                "atom_preselection_strategy": self.atom_preselection_strategy,
                "atom_preselection_top_k": self.atom_preselection_top_k,
                "max_rules": self.max_rules,
                "max_conditions": self.max_conditions,
                "max_fit_seconds": self.max_fit_seconds,
            },
        )
        self.ruleset_.validate()
        return self

    # ------------------------------------------------------------------
    # predict / predict_proba / to_ruleset
    # ------------------------------------------------------------------

    def predict(self, X):
        check_is_fitted(self, "ruleset_")
        X_valid = np.asarray(check_array(X, dtype=None))
        return predict_from_ruleset(self.ruleset_, X_valid)

    def predict_proba(self, X):
        check_is_fitted(self, "ruleset_")
        X_valid = np.asarray(check_array(X, dtype=None))
        return predict_proba_from_ruleset(self.ruleset_, X_valid)

    def to_ruleset(self) -> ScoredRuleSet:
        check_is_fitted(self, "ruleset_")
        return self.ruleset_

