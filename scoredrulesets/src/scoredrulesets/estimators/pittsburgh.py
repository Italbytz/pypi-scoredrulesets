from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ..runtime import predict as predict_from_ruleset
from ..runtime import predict_proba as predict_proba_from_ruleset
from ..schema import AggregationSpec, Atom, Rule, ScoredRuleSet
from .base import BaseRuleSetEstimator


@dataclass(frozen=True)
class _CandidateRule:
    gain: float
    rule: Rule
    coverage: int


class PittsburghRuleSetClassifier(BaseRuleSetEstimator):
    """Simple Pittsburgh-style rule-set learner with beam search over rule subsets."""

    def __init__(
        self,
        aggregation: str = "argmax_sum",
        temperature: float = 1.0,
        include_default_rule: bool = True,
        enable_categorical_rules: bool = True,
        max_rules: int = 5,
        min_samples_leaf: int = 5,
        min_gain: float = 1e-9,
        candidate_pool_size: int = 24,
        beam_width: int = 8,
        max_iterations: int = 16,
        validation_fraction: float = 0.2,
        complexity_penalty: float = 0.01,
        random_state: int | None = None,
    ):
        self.aggregation = aggregation
        self.temperature = temperature
        self.include_default_rule = include_default_rule
        self.enable_categorical_rules = enable_categorical_rules
        self.max_rules = max_rules
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.candidate_pool_size = candidate_pool_size
        self.beam_width = beam_width
        self.max_iterations = max_iterations
        self.validation_fraction = validation_fraction
        self.complexity_penalty = complexity_penalty
        self.random_state = random_state

    def fit(self, X, y):
        X_valid, y_valid = check_X_y(X, y, dtype=None)
        self.n_features_in_ = X_valid.shape[1]
        self.feature_names_in_ = np.asarray([f"f{i}" for i in range(self.n_features_in_)], dtype=object)
        self.classes_ = unique_labels(y_valid)
        self._rng_ = np.random.default_rng(self.random_state)

        class_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
        y_idx = np.asarray([class_to_idx[v] for v in y_valid], dtype=int)
        n_classes = len(self.classes_)
        train_idx, val_idx = self._train_val_indices(y_idx)

        train_X = X_valid[train_idx]
        train_y = y_idx[train_idx]
        prior_counts = np.bincount(train_y, minlength=n_classes).astype(float)
        self._default_scores_ = self._distribution_to_scores(prior_counts)

        candidates = self._build_candidate_rules(train_X, train_y, n_classes)
        self._candidate_count_ = len(candidates)

        if not candidates:
            self.ruleset_ = self._build_ruleset([])
            return self

        initial_states = {tuple()}
        initial_states.update((idx,) for idx in range(min(len(candidates), self.beam_width)))
        frontier = sorted(initial_states)
        best_state = tuple()
        best_score = self._state_score(best_state, candidates, X_valid, y_idx, train_idx, val_idx)
        no_improvement_rounds = 0
        iterations_ran = 0

        for iteration in range(max(1, int(self.max_iterations))):
            iterations_ran = iteration + 1
            improved = False
            scored_frontier = []
            seen: set[tuple[int, ...]] = set()
            for state in frontier:
                if state in seen:
                    continue
                seen.add(state)
                score = self._state_score(state, candidates, X_valid, y_idx, train_idx, val_idx)
                scored_frontier.append((score, state))
                if score > best_score + 1e-12:
                    best_score = score
                    best_state = state
                    improved = True

                for neighbor in self._neighbors(state, len(candidates)):
                    if neighbor in seen:
                        continue
                    seen.add(neighbor)
                    neighbor_score = self._state_score(
                        neighbor,
                        candidates,
                        X_valid,
                        y_idx,
                        train_idx,
                        val_idx,
                    )
                    scored_frontier.append((neighbor_score, neighbor))
                    if neighbor_score > best_score + 1e-12:
                        best_score = neighbor_score
                        best_state = neighbor
                        improved = True

            ranked = sorted(
                scored_frontier,
                key=lambda item: (
                    item[0],
                    -len(item[1]),
                    -self._state_atom_count(item[1], candidates),
                    tuple(-i for i in item[1]),
                ),
                reverse=True,
            )
            frontier = [state for _, state in ranked[: max(1, int(self.beam_width))]]

            if best_state not in frontier:
                frontier.insert(0, best_state)
                frontier = frontier[: max(1, int(self.beam_width))]

            if improved:
                no_improvement_rounds = 0
            else:
                no_improvement_rounds += 1
            if no_improvement_rounds >= 4:
                break

        selected_rules = [candidates[idx].rule for idx in best_state]
        ruleset = self._build_ruleset(selected_rules)
        ruleset.metadata.update(
            {
                "source": "pittsburgh",
                "model_type": "pittsburgh_rule_set_search",
                "candidate_pool_size": int(self._candidate_count_),
                "beam_width": int(self.beam_width),
                "max_iterations": int(self.max_iterations),
                "iterations_ran": int(iterations_ran),
                "selected_rule_count": int(len(selected_rules)),
                "used_validation": bool(val_idx is not None),
                "validation_fraction": float(self.validation_fraction),
                "best_state_score": float(best_score),
            }
        )
        ruleset.validate()
        self.ruleset_ = ruleset
        return self

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

    def _build_ruleset(self, selected_rules: list[Rule]) -> ScoredRuleSet:
        rules = list(selected_rules)
        if self.include_default_rule:
            rules.append(
                Rule(
                    atoms=[],
                    scores=list(self._default_scores_),
                    rule_id="pittsburgh_default_prior",
                    metadata={"source": "pittsburgh", "kind": "class_prior"},
                )
            )
        return ScoredRuleSet(
            class_labels=self.classes_.tolist(),
            feature_names=self.feature_names_in_.tolist(),
            aggregation=AggregationSpec(type=self.aggregation, temperature=self.temperature),
            rules=rules,
            metadata={},
        )

    def _train_val_indices(self, y_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if self.validation_fraction <= 0.0:
            return np.arange(len(y_idx)), None
        _, counts = np.unique(y_idx, return_counts=True)
        if np.any(counts < 2):
            return np.arange(len(y_idx)), None
        train_idx, val_idx = train_test_split(
            np.arange(len(y_idx)),
            test_size=float(self.validation_fraction),
            random_state=self.random_state,
            stratify=y_idx,
        )
        return np.asarray(train_idx), np.asarray(val_idx)

    def _build_candidate_rules(
        self,
        X: np.ndarray,
        y_idx: np.ndarray,
        n_classes: int,
    ) -> list[_CandidateRule]:
        candidates: list[_CandidateRule] = []
        seen_signatures: set[tuple[tuple[str, str, str], ...]] = set()

        for feature_idx in range(X.shape[1]):
            column = X[:, feature_idx]
            feature_name = str(self.feature_names_in_[feature_idx])
            split = self._best_numeric_split(column, y_idx, n_classes)
            if split is not None:
                threshold, gain, left_counts, right_counts, left_cov, right_cov = split
                if gain >= self.min_gain:
                    candidates.extend(
                        [
                            self._candidate_from_rule(
                                Rule(
                                    atoms=[Atom(feature=feature_name, op="<=", value=float(threshold))],
                                    scores=self._distribution_to_scores(left_counts),
                                    rule_id=f"pittsburgh_rule_f{feature_idx}_le",
                                    metadata={"source": "pittsburgh", "gain": float(gain)},
                                ),
                                gain,
                                left_cov,
                                seen_signatures,
                            ),
                            self._candidate_from_rule(
                                Rule(
                                    atoms=[Atom(feature=feature_name, op=">", value=float(threshold))],
                                    scores=self._distribution_to_scores(right_counts),
                                    rule_id=f"pittsburgh_rule_f{feature_idx}_gt",
                                    metadata={"source": "pittsburgh", "gain": float(gain)},
                                ),
                                gain,
                                right_cov,
                                seen_signatures,
                            ),
                        ]
                    )

                for interval_idx, (interval_gain, low, high, counts, coverage) in enumerate(
                    self._numeric_interval_splits(column, y_idx, n_classes)
                ):
                    if interval_gain < self.min_gain:
                        continue
                    candidates.append(
                        self._candidate_from_rule(
                            Rule(
                                atoms=[Atom(feature=feature_name, op="between", value=[float(low), float(high)])],
                                scores=self._distribution_to_scores(counts),
                                rule_id=f"pittsburgh_rule_f{feature_idx}_between_{interval_idx}",
                                metadata={"source": "pittsburgh", "gain": float(interval_gain)},
                            ),
                            interval_gain,
                            coverage,
                            seen_signatures,
                        )
                    )
                continue

            if self.enable_categorical_rules:
                for category_idx, (gain, category, match_counts, coverage) in enumerate(
                    self._categorical_splits(column, y_idx, n_classes)
                ):
                    if gain < self.min_gain:
                        continue
                    candidates.append(
                        self._candidate_from_rule(
                            Rule(
                                atoms=[Atom(feature=feature_name, op="==", value=category)],
                                scores=self._distribution_to_scores(match_counts),
                                rule_id=f"pittsburgh_rule_f{feature_idx}_eq_{category_idx}",
                                metadata={"source": "pittsburgh", "gain": float(gain), "category": category},
                            ),
                            gain,
                            coverage,
                            seen_signatures,
                        )
                    )
                for group_idx, (gain, group_values, group_counts, coverage) in enumerate(
                    self._categorical_group_splits(column, y_idx, n_classes)
                ):
                    if gain < self.min_gain:
                        continue
                    candidates.append(
                        self._candidate_from_rule(
                            Rule(
                                atoms=[Atom(feature=feature_name, op="in", value=group_values)],
                                scores=self._distribution_to_scores(group_counts),
                                rule_id=f"pittsburgh_rule_f{feature_idx}_in_{group_idx}",
                                metadata={"source": "pittsburgh", "gain": float(gain), "group": group_values},
                            ),
                            gain,
                            coverage,
                            seen_signatures,
                        )
                    )

        candidates = [c for c in candidates if c is not None]
        candidates.sort(key=lambda c: (c.gain, c.coverage, -len(c.rule.atoms)), reverse=True)
        return candidates[: max(1, int(self.candidate_pool_size))]

    def _candidate_from_rule(
        self,
        rule: Rule,
        gain: float,
        coverage: int,
        seen_signatures: set[tuple[tuple[str, str, str], ...]],
    ) -> _CandidateRule | None:
        signature = tuple(
            (str(atom.feature), atom.op, repr(atom.value))
            for atom in rule.atoms
        )
        if signature in seen_signatures:
            return None
        seen_signatures.add(signature)
        return _CandidateRule(gain=float(gain), rule=rule, coverage=int(coverage))

    def _neighbors(self, state: tuple[int, ...], n_candidates: int) -> list[tuple[int, ...]]:
        neighbors: set[tuple[int, ...]] = set()
        state_set = set(state)
        candidate_order = list(range(n_candidates))

        if len(state) < max(1, int(self.max_rules)):
            for idx in candidate_order[: max(4, int(self.beam_width))]:
                if idx not in state_set:
                    neighbors.add(tuple(sorted((*state, idx))))

        for idx in state:
            reduced = tuple(v for v in state if v != idx)
            neighbors.add(reduced)

        if state:
            absent = [idx for idx in candidate_order if idx not in state_set][: max(2, int(self.beam_width // 2) + 1)]
            for drop_idx in state[: max(1, len(state))]:
                for add_idx in absent:
                    swapped = tuple(sorted([v for v in state if v != drop_idx] + [add_idx]))
                    if len(swapped) <= max(1, int(self.max_rules)):
                        neighbors.add(swapped)

        return [neighbor for neighbor in neighbors if len(neighbor) <= max(1, int(self.max_rules))]

    def _state_score(
        self,
        state: tuple[int, ...],
        candidates: list[_CandidateRule],
        X: np.ndarray,
        y_idx: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray | None,
    ) -> float:
        selected_rules = [candidates[idx].rule for idx in state]
        ruleset = self._build_ruleset(selected_rules)
        eval_idx = val_idx if val_idx is not None else train_idx
        y_pred = predict_from_ruleset(ruleset, X[eval_idx])
        y_true = np.asarray(self.classes_[y_idx[eval_idx]], dtype=str)
        y_pred = np.asarray(y_pred, dtype=str)
        f1 = float(f1_score(y_true, y_pred, average="macro"))
        n_rules = len(selected_rules)
        n_atoms = sum(len(rule.atoms) for rule in selected_rules)
        complexity = self.complexity_penalty * (
            n_rules + n_atoms / max(1.0, float(self.max_rules))
        )
        coverage_bonus = 0.001 * sum(candidates[idx].coverage for idx in state) / max(1, len(train_idx))
        return f1 - complexity + coverage_bonus

    def _state_atom_count(self, state: tuple[int, ...], candidates: list[_CandidateRule]) -> int:
        return sum(len(candidates[idx].rule.atoms) for idx in state)

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
        best = None
        for threshold in (unique[:-1] + unique[1:]) / 2.0:
            left_mask = np.asarray(values <= threshold, dtype=bool)
            right_mask = np.asarray(~left_mask, dtype=bool)
            if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                continue

            left_counts = np.bincount(y_idx[left_mask], minlength=n_classes).astype(float)
            right_counts = np.bincount(y_idx[right_mask], minlength=n_classes).astype(float)
            left_weight = float(left_mask.mean())
            child_impurity = left_weight * self._gini(left_counts) + (1.0 - left_weight) * self._gini(right_counts)
            gain = parent_impurity - child_impurity
            candidate = (
                float(threshold),
                float(gain),
                left_counts,
                right_counts,
                int(left_mask.sum()),
                int(right_mask.sum()),
            )
            if best is None or gain > best[1]:
                best = candidate
        return best

    def _categorical_splits(self, feature_values, y_idx: np.ndarray, n_classes: int):
        values = np.asarray(feature_values, dtype=object)
        unique = np.unique(values)
        if unique.size < 2:
            return []

        parent_counts = np.bincount(y_idx, minlength=n_classes).astype(float)
        parent_impurity = self._gini(parent_counts)
        candidates = []
        for category in unique.tolist():
            match_mask = np.asarray(values == category, dtype=bool)
            non_match_mask = np.asarray(~match_mask, dtype=bool)
            if match_mask.sum() < self.min_samples_leaf or non_match_mask.sum() < self.min_samples_leaf:
                continue
            match_counts = np.bincount(y_idx[match_mask], minlength=n_classes).astype(float)
            non_match_counts = np.bincount(y_idx[non_match_mask], minlength=n_classes).astype(float)
            match_weight = float(match_mask.mean())
            child_impurity = match_weight * self._gini(match_counts) + (1.0 - match_weight) * self._gini(non_match_counts)
            gain = parent_impurity - child_impurity
            candidates.append((float(gain), category, match_counts, int(match_mask.sum())))
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
        candidates = []
        for i in range(len(q_points) - 1):
            for j in range(i + 1, len(q_points)):
                low = float(q_points[i])
                high = float(q_points[j])
                if not low < high:
                    continue
                in_mask = (values >= low) & (values <= high)
                in_mask = np.asarray(in_mask, dtype=bool)
                out_mask = np.asarray(~in_mask, dtype=bool)
                if in_mask.sum() < self.min_samples_leaf or out_mask.sum() < self.min_samples_leaf:
                    continue
                in_counts = np.bincount(y_idx[in_mask], minlength=n_classes).astype(float)
                out_counts = np.bincount(y_idx[out_mask], minlength=n_classes).astype(float)
                in_weight = float(in_mask.mean())
                child_impurity = in_weight * self._gini(in_counts) + (1.0 - in_weight) * self._gini(out_counts)
                gain = parent_impurity - child_impurity
                candidates.append((float(gain), low, high, in_counts, int(in_mask.sum())))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[:2]

    def _categorical_group_splits(self, feature_values, y_idx: np.ndarray, n_classes: int):
        values = np.asarray(feature_values, dtype=object)
        unique = np.unique(values)
        if unique.size < 3:
            return []

        parent_counts = np.bincount(y_idx, minlength=n_classes).astype(float)
        parent_impurity = self._gini(parent_counts)
        candidates = []
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
                in_mask = np.asarray(in_mask, dtype=bool)
                out_mask = np.asarray(~in_mask, dtype=bool)
                if in_mask.sum() < self.min_samples_leaf or out_mask.sum() < self.min_samples_leaf:
                    continue
                in_counts = np.bincount(y_idx[in_mask], minlength=n_classes).astype(float)
                out_counts = np.bincount(y_idx[out_mask], minlength=n_classes).astype(float)
                in_weight = float(in_mask.mean())
                child_impurity = in_weight * self._gini(in_counts) + (1.0 - in_weight) * self._gini(out_counts)
                gain = parent_impurity - child_impurity
                candidates.append((float(gain), list(group), in_counts, int(in_mask.sum())))

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[:2]

    @staticmethod
    def _gini(counts: np.ndarray) -> float:
        total = float(np.sum(counts))
        if total <= 0.0:
            return 0.0
        probs = counts / total
        return float(1.0 - np.sum(probs ** 2))



