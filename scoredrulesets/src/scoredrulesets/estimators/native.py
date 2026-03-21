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
                for interval_idx, (interval_gain, low, high, interval_counts) in enumerate(
                    self._numeric_interval_splits(column, y_idx, n_classes)
                ):
                    if interval_gain < self.min_gain:
                        continue
                    candidate_rules.append(
                        (
                            interval_gain,
                            Rule(
                                atoms=[Atom(feature=feature_name, op="between", value=[float(low), float(high)])],
                                scores=self._distribution_to_scores(interval_counts),
                                rule_id=f"native_f{feature_idx}_between_{interval_idx}",
                                metadata={"source": "native", "gain": float(interval_gain)},
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
                for group_idx, (group_gain, group_values, group_counts) in enumerate(
                    self._categorical_group_splits(column, y_idx, n_classes)
                ):
                    if group_gain < self.min_gain:
                        continue
                    candidate_rules.append(
                        (
                            group_gain,
                            Rule(
                                atoms=[Atom(feature=feature_name, op="in", value=group_values)],
                                scores=self._distribution_to_scores(group_counts),
                                rule_id=f"native_f{feature_idx}_in_{group_idx}",
                                metadata={"source": "native", "gain": float(group_gain), "group": group_values},
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

    def _numeric_interval_splits(self, feature_values, y_idx: np.ndarray, n_classes: int):
        values = np.asarray(feature_values)
        if not np.issubdtype(values.dtype, np.number):
            return []

        values = values.astype(float)
        if np.unique(values).size < 3:
            return []

        q_points = np.unique(np.quantile(values, [0.1, 0.25, 0.4, 0.6, 0.75, 0.9]))
        if q_points.size < 2:
            return []

        parent_counts = np.bincount(y_idx, minlength=n_classes).astype(float)
        parent_impurity = self._gini(parent_counts)
        candidates: list[tuple[float, float, float, np.ndarray]] = []

        for i in range(len(q_points) - 1):
            for j in range(i + 1, len(q_points)):
                low = float(q_points[i])
                high = float(q_points[j])
                if not low < high:
                    continue

                in_mask = (values >= low) & (values <= high)
                out_mask = ~in_mask
                if (
                    in_mask.sum() < self.min_samples_leaf
                    or out_mask.sum() < self.min_samples_leaf
                ):
                    continue

                in_counts = np.bincount(y_idx[in_mask], minlength=n_classes).astype(float)
                out_counts = np.bincount(y_idx[out_mask], minlength=n_classes).astype(float)
                in_weight = float(in_mask.mean())
                child_impurity = (
                    in_weight * self._gini(in_counts)
                    + (1.0 - in_weight) * self._gini(out_counts)
                )
                gain = parent_impurity - child_impurity
                candidates.append((float(gain), low, high, in_counts))

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[:2]

    def _categorical_group_splits(self, feature_values, y_idx: np.ndarray, n_classes: int):
        values = np.asarray(feature_values, dtype=object)
        unique = np.unique(values)
        if unique.size < 3:
            return []

        parent_counts = np.bincount(y_idx, minlength=n_classes).astype(float)
        parent_impurity = self._gini(parent_counts)
        candidates: list[tuple[float, list[object], np.ndarray]] = []
        seen_groups: set[tuple[str, ...]] = set()

        for class_idx in range(n_classes):
            class_mask = y_idx == class_idx
            if class_mask.sum() == 0:
                continue

            class_values = values[class_mask]
            cats, counts = np.unique(class_values, return_counts=True)
            order = np.argsort(-counts)
            ranked_cats = [cats[i] for i in order]

            for group_size in range(2, min(3, len(ranked_cats)) + 1):
                group = ranked_cats[:group_size]
                key = tuple(sorted(str(v) for v in group))
                if key in seen_groups:
                    continue
                seen_groups.add(key)

                in_mask = np.isin(values, group)
                out_mask = ~in_mask
                if (
                    in_mask.sum() < self.min_samples_leaf
                    or out_mask.sum() < self.min_samples_leaf
                ):
                    continue

                in_counts = np.bincount(y_idx[in_mask], minlength=n_classes).astype(float)
                out_counts = np.bincount(y_idx[out_mask], minlength=n_classes).astype(float)
                in_weight = float(in_mask.mean())
                child_impurity = (
                    in_weight * self._gini(in_counts)
                    + (1.0 - in_weight) * self._gini(out_counts)
                )
                gain = parent_impurity - child_impurity
                candidates.append((float(gain), list(group), in_counts))

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[:3]

    @staticmethod
    def _gini(counts: np.ndarray) -> float:
        total = float(np.sum(counts))
        if total <= 0:
            return 0.0
        probs = counts / total
        return float(1.0 - np.sum(probs**2))

