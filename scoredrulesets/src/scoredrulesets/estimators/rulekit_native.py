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
   features.  RuleKit uses an entropy- or C2-based quality measure; we
   approximate this with Gini-based information gain to reuse the shared
   ``_split_utils`` infrastructure.
3. **Pruning** – after growing, conditions are removed one at a time (back to
   front) if removal does not decrease rule quality on a held-out validation
   set (Reduced Error Pruning, REP).

The resulting classifier exposes the standard ``BaseRuleSetEstimator`` API and
produces a ``ScoredRuleSet`` that can be evaluated by the common runtime.

The existing Java-based backend (``backend="rulekit"``) is **not** changed –
this module provides an independent ``backend="rulekit_native"`` alternative
that requires no JVM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ..runtime import predict as predict_from_ruleset
from ..runtime import predict_proba as predict_proba_from_ruleset
from ..schema import AggregationSpec, Atom, Rule, ScoredRuleSet
from .base import BaseRuleSetEstimator
from ._split_utils import (
    best_numeric_split,
    categorical_splits,
    distribution_to_scores,
    gini,
)


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class _ElementaryCondition:
    """A single condition ``feature op value``."""
    feature_idx: int
    feature_name: str
    op: str  # one of <=, >, <, >=, ==
    value: Any

    def matches(self, row: np.ndarray) -> bool:
        v = row[self.feature_idx]
        if self.op == "<=":
            return v <= self.value
        if self.op == ">":
            return v > self.value
        if self.op == "<":
            return v < self.value
        if self.op == ">=":
            return v >= self.value
        if self.op == "==":
            return v == self.value
        raise ValueError(f"Unsupported op: {self.op}")

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
        raise ValueError(f"Unsupported op: {self.op}")


@dataclass
class _InducedRule:
    """An induced rule: conjunction of elementary conditions + predicted class."""
    conditions: list[_ElementaryCondition]
    class_idx: int
    quality: float  # rule quality on training or pruning data
    coverage: int  # number of examples covered


# ---------------------------------------------------------------------------
# Quality measure (approximation of RuleKit's C2 / entropy gain)
# ---------------------------------------------------------------------------

def _rule_quality(
    covered_counts: np.ndarray,
    total_counts: np.ndarray,
    class_idx: int,
) -> float:
    """Compute rule quality for the target class.

    We use a precision × coverage heuristic similar to RuleKit's default C2
    measure:  quality = (p - n) / (P + N)  where
      p = covered positives,  n = covered negatives,
      P = total positives,   N = total negatives.
    """
    p = covered_counts[class_idx]
    n = float(np.sum(covered_counts)) - p
    P = total_counts[class_idx]
    N = float(np.sum(total_counts)) - P
    denom = P + N
    if denom <= 0:
        return 0.0
    return float((p - n) / denom)


def _rule_precision(covered_counts: np.ndarray, class_idx: int) -> float:
    total = float(np.sum(covered_counts))
    if total <= 0:
        return 0.0
    return float(covered_counts[class_idx] / total)


# ---------------------------------------------------------------------------
# Growing: greedy condition addition
# ---------------------------------------------------------------------------

def _best_condition_for_feature(
    col: np.ndarray,
    y_idx: np.ndarray,
    n_classes: int,
    class_idx: int,
    total_counts: np.ndarray,
    min_samples_leaf: int,
    max_thresholds: int | None,
) -> tuple[_ElementaryCondition, float] | None:
    """Find the best elementary condition on a single feature column.

    Returns ``(condition, quality)`` or ``None``.
    """
    is_numeric = np.issubdtype(col.dtype, np.number)

    best: tuple[_ElementaryCondition, float] | None = None

    if is_numeric:
        values = col.astype(float)
        unique_vals = np.unique(values)
        if unique_vals.size < 2:
            return None

        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
        if max_thresholds is not None and len(thresholds) > max_thresholds:
            idx = np.round(np.linspace(0, len(thresholds) - 1, max_thresholds)).astype(int)
            thresholds = thresholds[idx]

        for threshold in thresholds:
            for op in ("<=", ">"):
                if op == "<=":
                    mask = values <= threshold
                else:
                    mask = values > threshold
                if mask.sum() < min_samples_leaf or (~mask).sum() < min_samples_leaf:
                    continue
                covered_counts = np.bincount(y_idx[mask], minlength=n_classes).astype(float)
                q = _rule_quality(covered_counts, total_counts, class_idx)
                if best is None or q > best[1]:
                    cond = _ElementaryCondition(
                        feature_idx=-1,  # will be set by caller
                        feature_name="",
                        op=op,
                        value=float(threshold),
                    )
                    best = (cond, q)
    else:
        # Categorical feature
        categories = np.unique(col)
        if categories.size < 2:
            return None
        for cat in categories:
            mask = np.asarray(col == cat, dtype=bool)
            if mask.sum() < min_samples_leaf or (~mask).sum() < min_samples_leaf:
                continue
            covered_counts = np.bincount(y_idx[mask], minlength=n_classes).astype(float)
            q = _rule_quality(covered_counts, total_counts, class_idx)
            if best is None or q > best[1]:
                cond = _ElementaryCondition(
                    feature_idx=-1,
                    feature_name="",
                    op="==",
                    value=cat,
                )
                best = (cond, q)

    return best


def _grow_rule(
    X: np.ndarray,
    y_idx: np.ndarray,
    n_classes: int,
    class_idx: int,
    feature_names: list[str],
    min_samples_leaf: int,
    max_conditions: int,
    max_thresholds: int | None,
) -> _InducedRule | None:
    """Grow a single rule by greedy condition addition (RuleKit growing phase).

    Starting from the full dataset the algorithm repeatedly adds the
    elementary condition that maximises the C2-like quality measure until
    no improvement is possible or the maximum number of conditions is reached.
    """
    n_samples = X.shape[0]
    total_counts = np.bincount(y_idx, minlength=n_classes).astype(float)

    conditions: list[_ElementaryCondition] = []
    active_mask = np.ones(n_samples, dtype=bool)

    for _ in range(max_conditions):
        if active_mask.sum() < min_samples_leaf:
            break

        active_X = X[active_mask]
        active_y = y_idx[active_mask]
        active_total = np.bincount(active_y, minlength=n_classes).astype(float)

        best_cond: _ElementaryCondition | None = None
        best_quality = -np.inf

        for feat_idx in range(X.shape[1]):
            col = active_X[:, feat_idx]
            result = _best_condition_for_feature(
                col, active_y, n_classes, class_idx,
                active_total, min_samples_leaf, max_thresholds,
            )
            if result is None:
                continue
            cond, quality = result
            if quality > best_quality:
                best_quality = quality
                best_cond = _ElementaryCondition(
                    feature_idx=feat_idx,
                    feature_name=feature_names[feat_idx],
                    op=cond.op,
                    value=cond.value,
                )

        if best_cond is None:
            break

        # Check that adding the condition actually improves quality
        new_mask = active_mask & best_cond.matches_array(X)
        if new_mask.sum() < min_samples_leaf:
            break

        # Check improvement over current quality
        current_covered = np.bincount(y_idx[active_mask], minlength=n_classes).astype(float)
        current_q = _rule_quality(current_covered, total_counts, class_idx)
        new_covered = np.bincount(y_idx[new_mask], minlength=n_classes).astype(float)
        new_q = _rule_quality(new_covered, total_counts, class_idx)

        if new_q <= current_q and len(conditions) > 0:
            break

        conditions.append(best_cond)
        active_mask = new_mask

    if not conditions:
        return None

    covered_counts = np.bincount(y_idx[active_mask], minlength=n_classes).astype(float)
    quality = _rule_quality(covered_counts, total_counts, class_idx)

    return _InducedRule(
        conditions=conditions,
        class_idx=class_idx,
        quality=quality,
        coverage=int(active_mask.sum()),
    )


# ---------------------------------------------------------------------------
# Pruning: reduced error pruning (REP)
# ---------------------------------------------------------------------------

def _prune_rule(
    rule: _InducedRule,
    X_prune: np.ndarray,
    y_prune: np.ndarray,
    n_classes: int,
) -> _InducedRule:
    """Remove conditions from the rule (back to front) if quality does not drop.

    This implements RuleKit's post-growing pruning phase: conditions are
    tentatively removed and the rule is kept shorter if quality on the
    pruning set stays the same or improves.
    """
    if len(rule.conditions) <= 1:
        return rule

    total_counts = np.bincount(y_prune, minlength=n_classes).astype(float)

    def _quality_of(conditions: list[_ElementaryCondition]) -> float:
        mask = np.ones(X_prune.shape[0], dtype=bool)
        for c in conditions:
            mask &= c.matches_array(X_prune)
        if mask.sum() == 0:
            return -np.inf
        covered = np.bincount(y_prune[mask], minlength=n_classes).astype(float)
        return _rule_quality(covered, total_counts, rule.class_idx)

    current_conditions = list(rule.conditions)
    current_quality = _quality_of(current_conditions)

    # Iterate backwards (most recently added condition first)
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
                break  # restart backward scan

    mask = np.ones(X_prune.shape[0], dtype=bool)
    for c in current_conditions:
        mask &= c.matches_array(X_prune)
    coverage = int(mask.sum())

    return _InducedRule(
        conditions=current_conditions,
        class_idx=rule.class_idx,
        quality=current_quality,
        coverage=coverage,
    )


# ---------------------------------------------------------------------------
# Conflict resolution / voting scheme
# ---------------------------------------------------------------------------

def _induced_rule_to_scored_rule(
    rule: _InducedRule,
    n_classes: int,
    rule_idx: int,
    X: np.ndarray,
    y_idx: np.ndarray,
    aggregation: str,
) -> Rule:
    """Convert an _InducedRule into a ScoredRuleSet Rule."""
    atoms: list[Atom] = []
    for cond in rule.conditions:
        atoms.append(Atom(
            feature=cond.feature_name,
            op=cond.op,
            value=cond.value,
        ))

    # Compute class distribution of covered examples
    mask = np.ones(X.shape[0], dtype=bool)
    for cond in rule.conditions:
        mask &= cond.matches_array(X)

    if mask.sum() > 0:
        covered_counts = np.bincount(y_idx[mask], minlength=n_classes).astype(float)
    else:
        covered_counts = np.zeros(n_classes, dtype=float)

    scores = distribution_to_scores(covered_counts, aggregation)

    return Rule(
        atoms=atoms,
        scores=scores,
        rule_id=f"rulekit_native_{rule_idx}",
        metadata={
            "source": "rulekit_native",
            "class_idx": rule.class_idx,
            "quality": rule.quality,
            "coverage": rule.coverage,
            "n_conditions": len(rule.conditions),
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
    pruning_fraction : float
        Fraction of training data reserved for pruning (only if enable_pruning).
    validation_fraction : float
        Fraction of training data used as validation set for overall stopping.
    max_thresholds_per_feature : int | None
        Cap the number of candidate thresholds per numeric feature.
    aggregation : str
        Aggregation type for the ScoredRuleSet (``"argmax_sum"`` or
        ``"softmax_sum"``).
    temperature : float
        Temperature for softmax aggregation.
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
        pruning_fraction: float = 0.33,
        validation_fraction: float = 0.0,
        max_thresholds_per_feature: int | None = None,
        aggregation: str = "argmax_sum",
        temperature: float = 1.0,
        random_state: int | None = None,
    ):
        self.max_rules = max_rules
        self.max_conditions = max_conditions
        self.min_samples_leaf = min_samples_leaf
        self.min_rule_covered = min_rule_covered
        self.enable_pruning = enable_pruning
        self.pruning_fraction = pruning_fraction
        self.validation_fraction = validation_fraction
        self.max_thresholds_per_feature = max_thresholds_per_feature
        self.aggregation = aggregation
        self.temperature = temperature
        self.random_state = random_state

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

        # ---- Sequential Covering ----
        induced_rules: list[_InducedRule] = []
        uncovered_mask = np.ones(X_work.shape[0], dtype=bool)

        for _round in range(self.max_rules):
            if uncovered_mask.sum() < self.min_rule_covered:
                break

            X_uncovered = X_work[uncovered_mask]
            y_uncovered = y_work[uncovered_mask]

            # Determine target class for this round (class with most
            # remaining uncovered examples, cycling through classes)
            remaining_counts = np.bincount(y_uncovered, minlength=n_classes)
            if remaining_counts.sum() == 0:
                break
            target_class = int(np.argmax(remaining_counts))

            # Split uncovered data into grow/prune sets
            if self.enable_pruning and self.pruning_fraction > 0 and X_uncovered.shape[0] >= 2 * self.min_samples_leaf:
                _, uc_counts = np.unique(y_uncovered, return_counts=True)
                if np.all(uc_counts >= 2):
                    grow_idx, prune_idx = train_test_split(
                        np.arange(X_uncovered.shape[0]),
                        test_size=float(self.pruning_fraction),
                        random_state=self.random_state,
                        stratify=y_uncovered,
                    )
                    X_grow = X_uncovered[grow_idx]
                    y_grow = y_uncovered[grow_idx]
                    X_prune = X_uncovered[prune_idx]
                    y_prune = y_uncovered[prune_idx]
                else:
                    X_grow, y_grow = X_uncovered, y_uncovered
                    X_prune, y_prune = X_uncovered, y_uncovered
            else:
                X_grow, y_grow = X_uncovered, y_uncovered
                X_prune, y_prune = X_uncovered, y_uncovered

            # Growing phase
            rule = _grow_rule(
                X_grow, y_grow, n_classes, target_class,
                feature_names, self.min_samples_leaf,
                self.max_conditions, self.max_thresholds_per_feature,
            )
            if rule is None:
                break

            # Pruning phase
            if self.enable_pruning:
                rule = _prune_rule(rule, X_prune, y_prune, n_classes)

            induced_rules.append(rule)

            # Remove covered examples (correctly classified ones)
            indices = np.where(uncovered_mask)[0]
            X_active = X_work[uncovered_mask]
            y_active = y_work[uncovered_mask]
            rule_mask = np.ones(X_active.shape[0], dtype=bool)
            for cond in rule.conditions:
                rule_mask &= cond.matches_array(X_active)
            correctly_covered = rule_mask & (y_active == rule.class_idx)

            # Mark correctly covered examples as covered
            cover_indices = indices[correctly_covered]
            uncovered_mask[cover_indices] = False

            if uncovered_mask.sum() == 0:
                break

        # ---- Build ScoredRuleSet ----
        rules: list[Rule] = []

        # Default rule from class prior
        prior_counts = np.bincount(y_work, minlength=n_classes).astype(float)
        default_scores = distribution_to_scores(prior_counts, self.aggregation)
        rules.append(Rule(
            atoms=[],
            scores=default_scores,
            rule_id="rulekit_native_default",
            metadata={"source": "rulekit_native", "kind": "class_prior"},
        ))

        for i, irule in enumerate(induced_rules):
            rules.append(_induced_rule_to_scored_rule(
                irule, n_classes, i, X_work, y_work, self.aggregation,
            ))

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
                "max_rules": self.max_rules,
                "max_conditions": self.max_conditions,
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

