from __future__ import annotations

import numpy as np
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ..runtime import predict as predict_from_ruleset
from ..runtime import predict_proba as predict_proba_from_ruleset
from ..schema import AggregationSpec, Atom, Rule, ScoredRuleSet
from .base import BaseRuleSetEstimator


class NativeScoredRuleSetClassifier(BaseRuleSetEstimator):
    """Einfacher nativer Schaetzer mit univariaten Split-Regeln plus Prior-Rule."""

    def __init__(
        self,
        aggregation: str = "softmax_sum",
        temperature: float = 1.0,
        include_default_rule: bool = True,
        enable_categorical_rules: bool = True,
        max_rules: int = 6,
        min_samples_leaf: int = 5,
        min_gain: float = 1e-9,
    ):
        self.aggregation = aggregation
        self.temperature = temperature
        self.include_default_rule = include_default_rule
        self.enable_categorical_rules = enable_categorical_rules
        self.max_rules = max_rules
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain

    def fit(self, X, y):
        X_valid, y_valid = check_X_y(X, y, dtype=None)
        self.n_features_in_ = X_valid.shape[1]
        self.feature_names_in_ = np.asarray([f"f{i}" for i in range(self.n_features_in_)], dtype=object)
        self.classes_ = unique_labels(y_valid)

        class_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
        y_idx = np.asarray([class_to_idx[v] for v in y_valid], dtype=int)
        n_classes = len(self.classes_)

        prior_counts = np.bincount(y_idx, minlength=n_classes).astype(float)
        default_scores = self._distribution_to_scores(prior_counts)

        candidate_rules: list[tuple[float, Rule]] = []
        for feature_idx in range(self.n_features_in_):
            column = X_valid[:, feature_idx]
            feature_name = str(self.feature_names_in_[feature_idx])
            split = self._best_numeric_split(column, y_idx, n_classes)
            if split is not None:
                threshold, gain, left_counts, right_counts = split
                if gain >= self.min_gain:
                    candidate_rules.append(
                        (
                            gain,
                            Rule(
                                atoms=[Atom(feature=feature_name, op="<=", value=float(threshold))],
                                scores=self._distribution_to_scores(left_counts),
                                rule_id=f"native_f{feature_idx}_le",
                                metadata={"source": "native", "gain": float(gain)},
                            ),
                        )
                    )
                    candidate_rules.append(
                        (
                            gain,
                            Rule(
                                atoms=[Atom(feature=feature_name, op=">", value=float(threshold))],
                                scores=self._distribution_to_scores(right_counts),
                                rule_id=f"native_f{feature_idx}_gt",
                                metadata={"source": "native", "gain": float(gain)},
                            ),
                        )
                    )
                continue

            if self.enable_categorical_rules:
                for category_idx, (gain, category, match_counts) in enumerate(
                    self._categorical_splits(column, y_idx, n_classes)
                ):
                    if gain < self.min_gain:
                        continue
                    candidate_rules.append(
                        (
                            gain,
                            Rule(
                                atoms=[Atom(feature=feature_name, op="==", value=category)],
                                scores=self._distribution_to_scores(match_counts),
                                rule_id=f"native_f{feature_idx}_eq_{category_idx}",
                                metadata={"source": "native", "gain": float(gain), "category": category},
                            ),
                        )
                    )

        candidate_rules.sort(key=lambda item: item[0], reverse=True)
        chosen_rules = [rule for _, rule in candidate_rules[: max(0, int(self.max_rules))]]

        rules: list[Rule] = list(chosen_rules)
        if self.include_default_rule:
            rules.append(
                Rule(
                    atoms=[],
                    scores=default_scores,
                    rule_id="native_default_prior",
                    metadata={"source": "native", "kind": "class_prior"},
                )
            )

        self.ruleset_ = ScoredRuleSet(
            class_labels=self.classes_.tolist(),
            feature_names=self.feature_names_in_.tolist(),
            aggregation=AggregationSpec(type=self.aggregation, temperature=self.temperature),
            rules=rules,
            metadata={
                "source": "native_baseline",
                "model_type": "univariate_split_rules" if chosen_rules else "prior_only",
                "n_split_rules": len(chosen_rules),
            },
        )
        self.ruleset_.validate()
        return self

    def predict(self, X):
        check_is_fitted(self, "ruleset_")
        X_valid = check_array(X, dtype=None)
        return predict_from_ruleset(self.ruleset_, X_valid)

    def predict_proba(self, X):
        check_is_fitted(self, "ruleset_")
        X_valid = check_array(X, dtype=None)
        return predict_proba_from_ruleset(self.ruleset_, X_valid)

    def to_ruleset(self) -> ScoredRuleSet:
        check_is_fitted(self, "ruleset_")
        return self.ruleset_

    def _distribution_to_scores(self, counts: np.ndarray) -> list[float]:
        probs = counts / max(float(np.sum(counts)), 1.0)
        if self.aggregation == "softmax_sum":
            return np.log(np.maximum(probs, 1e-12)).tolist()
        return probs.tolist()

    def _best_numeric_split(self, feature_values, y_idx: np.ndarray, n_classes: int):
        values = np.asarray(feature_values)
        if not np.issubdtype(values.dtype, np.number):
            return None

        values = values.astype(float)
        unique = np.unique(values)
        if unique.size < 2:
            return None

        parent_counts = np.bincount(y_idx, minlength=n_classes).astype(float)
        parent_impurity = self._gini(parent_counts)

        best: tuple[float, float, np.ndarray, np.ndarray] | None = None
        for threshold in (unique[:-1] + unique[1:]) / 2.0:
            left_mask = values <= threshold
            right_mask = ~left_mask
            if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                continue

            left_counts = np.bincount(y_idx[left_mask], minlength=n_classes).astype(float)
            right_counts = np.bincount(y_idx[right_mask], minlength=n_classes).astype(float)

            left_weight = float(left_mask.mean())
            right_weight = 1.0 - left_weight
            child_impurity = (
                left_weight * self._gini(left_counts)
                + right_weight * self._gini(right_counts)
            )
            gain = parent_impurity - child_impurity

            if best is None or gain > best[1]:
                best = (float(threshold), float(gain), left_counts, right_counts)

        return best

    def _categorical_splits(self, feature_values, y_idx: np.ndarray, n_classes: int):
        values = np.asarray(feature_values, dtype=object)
        unique = np.unique(values)
        if unique.size < 2:
            return []

        parent_counts = np.bincount(y_idx, minlength=n_classes).astype(float)
        parent_impurity = self._gini(parent_counts)
        candidates: list[tuple[float, object, np.ndarray]] = []

        for category in unique.tolist():
            match_mask = values == category
            non_match_mask = ~match_mask
            if (
                match_mask.sum() < self.min_samples_leaf
                or non_match_mask.sum() < self.min_samples_leaf
            ):
                continue

            match_counts = np.bincount(y_idx[match_mask], minlength=n_classes).astype(float)
            non_match_counts = np.bincount(y_idx[non_match_mask], minlength=n_classes).astype(float)

            match_weight = float(match_mask.mean())
            child_impurity = (
                match_weight * self._gini(match_counts)
                + (1.0 - match_weight) * self._gini(non_match_counts)
            )
            gain = parent_impurity - child_impurity
            candidates.append((float(gain), category, match_counts))

        return candidates

    @staticmethod
    def _gini(counts: np.ndarray) -> float:
        total = float(np.sum(counts))
        if total <= 0:
            return 0.0
        probs = counts / total
        return float(1.0 - np.sum(probs**2))

