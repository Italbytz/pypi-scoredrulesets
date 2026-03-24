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
from ._split_utils import (
    best_numeric_split,
    categorical_group_splits,
    categorical_splits,
    distribution_to_scores,
    gini,
    numeric_interval_splits,
)


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
        max_thresholds_per_feature: int | None = None,
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
        self.max_thresholds_per_feature = max_thresholds_per_feature

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
        return distribution_to_scores(counts, self.aggregation)

    def _best_numeric_split(self, feature_values, y_idx: np.ndarray, n_classes: int):
        return best_numeric_split(
            feature_values, y_idx, n_classes,
            min_samples_leaf=self.min_samples_leaf,
            max_thresholds_per_feature=self.max_thresholds_per_feature,
        )

    def _categorical_splits(self, feature_values, y_idx: np.ndarray, n_classes: int):
        return categorical_splits(
            feature_values, y_idx, n_classes,
            min_samples_leaf=self.min_samples_leaf,
        )

    def _numeric_interval_splits(self, feature_values, y_idx: np.ndarray, n_classes: int):
        return numeric_interval_splits(
            feature_values, y_idx, n_classes,
            min_samples_leaf=self.min_samples_leaf,
            max_thresholds_per_feature=self.max_thresholds_per_feature,
            max_results=2,
        )

    def _categorical_group_splits(self, feature_values, y_idx: np.ndarray, n_classes: int):
        return categorical_group_splits(
            feature_values, y_idx, n_classes,
            min_samples_leaf=self.min_samples_leaf,
            max_results=2,
        )

    @staticmethod
    def _gini(counts: np.ndarray) -> float:
        return gini(counts)



