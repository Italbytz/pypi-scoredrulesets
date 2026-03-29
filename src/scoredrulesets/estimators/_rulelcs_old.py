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


class RulePLCSClassifier(BaseRuleSetEstimator):
    """Pittsburgh-style rule-set learner with beam search over rule subsets.

    Supports several BioHEL-inspired enhancements beyond a simple beam search:

    * **Sequential Covering (IRL)** – when ``sequential_covering=True``, rules
      are learned iteratively: after each rule is selected, the correctly covered
      examples are removed and candidates are rebuilt on the residual dataset.
    * **Token Competition** – ``token_competition_weight`` adds an overlap
      penalty to the fitness so that rules covering the same examples as already
      selected rules are penalized, promoting specialization.
    * **Post-hoc Rule Compaction** – ``enable_compaction=True`` runs a backward
      elimination step after fitting: rules are greedily removed if removal does
      not decrease macro-F1 on the evaluation set.  ``compaction_min_gain``
      (default ``0.0``) raises the bar: a rule is only removed when its absence
      *improves* F1 by at least that value.  Set to e.g. ``0.005`` to keep rules
      that are likely to generalise even if they add no measurable gain on a
      small val set; set to a negative value for more aggressive pruning.
    * **Windowing** – ``window_fraction`` (0–1] controls stochastic subset
      evaluation.  When set to less than 1.0, each beam-search iteration
      evaluates candidates on a random subset of the evaluation indices,
      significantly speeding up large datasets.
    * **Low-Cardinality Detection** – ``low_cardinality_threshold`` (default 10)
      causes numeric features with at most that many unique values to be
      additionally treated as categorical: equality splits (``== value``) are
      generated alongside the standard threshold splits (``<=`` / ``>``).  This
      is essential for datasets where categorical features are integer-encoded
      (e.g. MONK, car_evaluation).
    """

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
        complexity_penalty: float = 0.003,
        random_state: int | None = None,
        max_thresholds_per_feature: int | None = None,
        # -- BioHEL-inspired enhancements --
        sequential_covering: bool = False,
        token_competition_weight: float = 0.0,
        enable_compaction: bool = False,
        compaction_min_gain: float = 0.0,
        window_fraction: float = 1.0,
        # -- Multi-class strategy --
        multiclass_strategy: str = "auto",
        # -- Low-cardinality detection --
        low_cardinality_threshold: int = 10,
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
        self.sequential_covering = sequential_covering
        self.token_competition_weight = token_competition_weight
        self.enable_compaction = enable_compaction
        self.compaction_min_gain = compaction_min_gain
        self.window_fraction = window_fraction
        self.multiclass_strategy = multiclass_strategy
        self.low_cardinality_threshold = low_cardinality_threshold
        self._ovr_binary_f1_ = False

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

        if self.multiclass_strategy not in ("direct", "ovr", "auto"):
            raise ValueError(
                f"multiclass_strategy must be 'direct', 'ovr' or 'auto', got {self.multiclass_strategy!r}"
            )

        # 'auto' → OvR for >=3 classes, direct for binary
        effective_strategy = self.multiclass_strategy
        if effective_strategy == "auto":
            effective_strategy = "ovr" if n_classes >= 3 else "direct"

        use_ovr = effective_strategy == "ovr" and n_classes > 2

        if use_ovr:
            selected_rules, iterations_ran = self._fit_ovr(
                X_valid, y_idx, n_classes, train_idx, val_idx,
            )
        elif self.sequential_covering:
            selected_rules, iterations_ran = self._fit_sequential_covering(
                X_valid, y_idx, n_classes, train_idx, val_idx,
            )
        else:
            selected_rules, iterations_ran = self._fit_beam_search(
                X_valid, y_idx, n_classes, train_idx, val_idx,
            )

        # -- Post-hoc Rule Compaction (BioHEL-inspired) --
        if self.enable_compaction and len(selected_rules) > 1:
            selected_rules = self._compact_rules(selected_rules, X_valid, y_idx, train_idx, val_idx)

        ruleset = self._build_ruleset(selected_rules)
        ruleset.metadata.update(
            {
                "source": "ruleplcs",
                "model_type": "rulelcs_rule_set_search",
                "candidate_pool_size": int(self._candidate_count_),
                "beam_width": int(self.beam_width),
                "max_iterations": int(self.max_iterations),
                "iterations_ran": int(iterations_ran),
                "selected_rule_count": int(len(selected_rules)),
                "used_validation": bool(val_idx is not None),
                "validation_fraction": float(self.validation_fraction),
                "sequential_covering": bool(self.sequential_covering),
                "token_competition_weight": float(self.token_competition_weight),
                "compaction_enabled": bool(self.enable_compaction),
                "window_fraction": float(self.window_fraction),
                "multiclass_strategy": str(self.multiclass_strategy),
            }
        )
        ruleset.validate()
        self.ruleset_ = ruleset
        return self

    # ------------------------------------------------------------------
    # Standard beam search (original approach)
    # ------------------------------------------------------------------

    def _fit_beam_search(
        self,
        X_valid: np.ndarray,
        y_idx: np.ndarray,
        n_classes: int,
        train_idx: np.ndarray,
        val_idx: np.ndarray | None,
    ) -> tuple[list[Rule], int]:
        """Run the classic beam-search over rule subsets."""
        train_X = X_valid[train_idx]
        train_y = y_idx[train_idx]
        candidates = self._build_candidate_rules(train_X, train_y, n_classes)
        self._candidate_count_ = len(candidates)

        if not candidates:
            return [], 0

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
            if no_improvement_rounds >= 6:
                break

        selected_rules = [candidates[idx].rule for idx in best_state]
        return selected_rules, iterations_ran

    # ------------------------------------------------------------------
    # Sequential Covering / Iterative Rule Learning (BioHEL-inspired)
    # ------------------------------------------------------------------

    def _fit_sequential_covering(
        self,
        X_valid: np.ndarray,
        y_idx: np.ndarray,
        n_classes: int,
        train_idx: np.ndarray,
        val_idx: np.ndarray | None,
    ) -> tuple[list[Rule], int]:
        """Learn rules one at a time, removing covered examples after each round.

        This is the *Iterative Rule Learning* (IRL) strategy used in BioHEL.
        After each rule is selected the correctly covered training examples are
        removed so the next round focuses on the residual (uncovered) data.
        """
        max_rules = max(1, int(self.max_rules))
        remaining_train_idx = train_idx.copy()
        selected_rules: list[Rule] = []
        total_iterations = 0
        self._candidate_count_ = 0

        for _round in range(max_rules):
            if remaining_train_idx.size < max(2, self.min_samples_leaf):
                break

            residual_X = X_valid[remaining_train_idx]
            residual_y = y_idx[remaining_train_idx]
            candidates = self._build_candidate_rules(residual_X, residual_y, n_classes)
            self._candidate_count_ += len(candidates)

            if not candidates:
                break

            # Score each candidate in combination with already-selected rules
            # (not in isolation) so that rules 2+ are evaluated in the context
            # of earlier rules and don't appear weak just because they specialise
            # on residual cases.  We always add the best-scoring rule (original
            # IRL behaviour); the compaction pass prunes redundant rules later.
            eval_idx_comb = val_idx if val_idx is not None else train_idx
            best_rule, best_combined_f1 = None, -np.inf
            for cand in candidates:
                combo_rules = list(selected_rules) + [cand.rule]
                score = self._evaluate_rule_list_f1(
                    combo_rules, X_valid, y_idx, eval_idx_comb
                )
                if score > best_combined_f1:
                    best_combined_f1 = score
                    best_rule = cand.rule
            total_iterations += 1

            if best_rule is None:
                break

            selected_rules.append(best_rule)

            # Remove correctly covered training examples (IRL core step)
            ruleset_so_far = self._build_ruleset(selected_rules)
            preds = predict_from_ruleset(ruleset_so_far, X_valid[remaining_train_idx])
            y_true_str = np.asarray(self.classes_[y_idx[remaining_train_idx]], dtype=str)
            preds_str = np.asarray(preds, dtype=str)
            correctly_covered = preds_str == y_true_str
            # Keep only incorrectly classified examples for the next round
            remaining_train_idx = remaining_train_idx[~correctly_covered]

            if remaining_train_idx.size == 0:
                break

        # NOTE: _default_scores_ is deliberately NOT recomputed from the
        # residual here.  The default rule has empty atoms and fires as an
        # additive baseline for *all* examples via argmax_sum, not just
        # uncovered ones.  Overwriting it with the (heavily skewed) residual
        # distribution would corrupt the global prior and cause the model to
        # predict only the residual's majority class on every sample.  The
        # full-training-set prior set at the start of fit() is the correct
        # baseline to keep.

        return selected_rules, total_iterations

    # ------------------------------------------------------------------
    # One-vs-Rest (OvR) Multi-class Strategy
    # ------------------------------------------------------------------

    def _fit_ovr(
        self,
        X_valid: np.ndarray,
        y_idx: np.ndarray,
        n_classes: int,
        train_idx: np.ndarray,
        val_idx: np.ndarray | None,
    ) -> tuple[list[Rule], int]:
        """Learn rules via One-vs-Rest decomposition.

        For each class *c* the labels are binarised (class *c* → 1, rest → 0)
        and either beam search or sequential covering is run on the resulting
        2-class problem.  The binary score vectors are then expanded back to
        the full *n_classes* dimensions so that all per-class rules can be
        combined into a single :class:`ScoredRuleSet`.
        """
        all_rules: list[Rule] = []
        total_iterations = 0
        self._candidate_count_ = 0

        # Save originals – will be restored after all OvR rounds
        saved_default_scores = list(self._default_scores_)
        saved_classes = self.classes_.copy()
        saved_include_default = self.include_default_rule

        max_rules_per_class = max(1, int(self.max_rules))
        binary_classes = np.array([0, 1])

        for class_idx in range(n_classes):
            # Binarise: class_idx → 1, everything else → 0
            y_bin = (y_idx == class_idx).astype(int)

            # Temporarily set up binary environment so _build_ruleset
            # and the runtime create consistent 2-class rulesets.
            self.classes_ = binary_classes
            # Use binary F1 (for the positive class) during OvR subproblems
            # so that rules for minority classes are properly rewarded.
            self._ovr_binary_f1_ = True
            # Disable default rule during OvR search so that rules can
            # actually shift predictions toward the positive class
            # (with argmax_sum the default prior overwhelms minority rules).
            self.include_default_rule = False

            # Temporary binary defaults (still needed for _distribution_to_scores)
            bin_train_y = y_bin[train_idx]
            bin_counts = np.bincount(bin_train_y, minlength=2).astype(float)
            # Store base rate for contrast scoring
            self._ovr_base_rate_ = float(bin_counts[1]) / max(1.0, float(bin_counts.sum()))
            self._default_scores_ = self._distribution_to_scores(bin_counts)

            if self.sequential_covering:
                class_rules, iters = self._fit_sequential_covering(
                    X_valid, y_bin, 2, train_idx, val_idx,
                )
            else:
                class_rules, iters = self._fit_beam_search(
                    X_valid, y_bin, 2, train_idx, val_idx,
                )

            total_iterations += iters

            # Expand binary scores to full n_classes vector and relabel rules
            for rule in class_rules:
                expanded = self._expand_binary_scores_to_multiclass(
                    rule, class_idx, n_classes,
                )
                all_rules.append(expanded)

        # Restore originals
        self.classes_ = saved_classes
        self._default_scores_ = saved_default_scores
        self._ovr_binary_f1_ = False
        self.include_default_rule = saved_include_default

        # Limit total rules: keep at most max_rules (round-robin across classes)
        if len(all_rules) > max_rules_per_class * n_classes:
            all_rules = all_rules[: max_rules_per_class * n_classes]

        # -- Post-hoc score rebalancing: scale per-class scores to maximise
        #    macro-F1 on the evaluation set.  Lift scoring amplifies minority
        #    classes, so we search for per-class scale factors in (0, 1] that
        #    improve the combined model.
        eval_idx = val_idx if val_idx is not None else train_idx
        all_rules = self._rebalance_ovr_scores(
            all_rules, X_valid, y_idx, eval_idx, n_classes,
        )

        return all_rules, total_iterations

    def _rebalance_ovr_scores(
        self,
        rules: list[Rule],
        X: np.ndarray,
        y_idx: np.ndarray,
        eval_idx: np.ndarray,
        n_classes: int,
    ) -> list[Rule]:
        """Scale per-class scores to maximise macro-F1 on the evaluation set.

        After OvR, minority-class rules often have inflated lift scores that
        cause over-prediction.  We search for per-class scale factors that
        improve the combined macro-F1.
        """
        if not rules:
            return rules

        # Group rules by OvR class index
        class_rule_indices: dict[int, list[int]] = {}
        for i, rule in enumerate(rules):
            ci = rule.metadata.get("ovr_class_index")
            if ci is not None:
                class_rule_indices.setdefault(int(ci), []).append(i)

        if not class_rule_indices:
            return rules

        # Evaluate baseline macro-F1
        def _eval_f1() -> float:
            rs = self._build_ruleset(rules)
            y_pred = predict_from_ruleset(rs, X[eval_idx])
            y_true = np.asarray(self.classes_[y_idx[eval_idx]], dtype=str)
            y_pred = np.asarray(y_pred, dtype=str)
            return float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        best_f1 = _eval_f1()

        # Coordinate descent: for each class, search for best scale factor
        scale_grid = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
        best_scales = {ci: 1.0 for ci in class_rule_indices}

        for _pass in range(3):  # 3 passes of coordinate descent
            improved = False
            for ci in sorted(class_rule_indices.keys()):
                cur_scale = best_scales[ci]
                best_ci_scale = cur_scale
                # Save current scores
                saved = {idx: list(rules[idx].scores) for idx in class_rule_indices[ci]}

                for scale in scale_grid:
                    if abs(scale - cur_scale) < 1e-9:
                        continue
                    # Apply new scale
                    ratio = scale / cur_scale
                    for idx in class_rule_indices[ci]:
                        rules[idx].scores = [s * ratio for s in saved[idx]]
                    trial_f1 = _eval_f1()
                    # Restore
                    for idx in class_rule_indices[ci]:
                        rules[idx].scores = list(saved[idx])

                    if trial_f1 > best_f1 + 1e-6:
                        best_f1 = trial_f1
                        best_ci_scale = scale
                        improved = True

                # Apply best scale permanently
                if best_ci_scale != cur_scale:
                    ratio = best_ci_scale / cur_scale
                    for idx in class_rule_indices[ci]:
                        rules[idx].scores = [s * ratio for s in rules[idx].scores]
                    best_scales[ci] = best_ci_scale

            if not improved:
                break

        return rules

    @staticmethod
    def _expand_binary_scores_to_multiclass(
        rule: Rule,
        class_idx: int,
        n_classes: int,
    ) -> Rule:
        """Convert a 2-class score vector to an *n_classes* vector.

        The positive-class score (``scores[1]``) is placed at position
        *class_idx*; all other positions receive 0.0.
        """
        binary_scores = rule.scores
        # The "positive" score in the binary sub-problem
        pos_score = binary_scores[1] if len(binary_scores) >= 2 else binary_scores[0]

        full_scores = [0.0] * n_classes
        full_scores[class_idx] = float(pos_score)

        new_metadata = dict(rule.metadata) if rule.metadata else {}
        new_metadata["ovr_class_index"] = class_idx

        return Rule(
            atoms=rule.atoms,
            scores=full_scores,
            rule_id=f"rulelcs_ovr_c{class_idx}_{rule.rule_id or 'rule'}",
            metadata=new_metadata,
        )

    def _single_rule_score(
        self,
        rule: Rule,
        X: np.ndarray,
        y_idx: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray | None,
    ) -> float:
        """Evaluate a single rule wrapped in a ruleset."""
        ruleset = self._build_ruleset([rule])
        eval_idx = val_idx if val_idx is not None else train_idx
        y_pred = predict_from_ruleset(ruleset, X[eval_idx])
        y_true = np.asarray(self.classes_[y_idx[eval_idx]], dtype=str)
        y_pred = np.asarray(y_pred, dtype=str)
        return float(f1_score(y_true, y_pred, **self._f1_kwargs))

    # ------------------------------------------------------------------
    # Post-hoc Rule Compaction (BioHEL-inspired)
    # ------------------------------------------------------------------

    def _compact_rules(
        self,
        rules: list[Rule],
        X: np.ndarray,
        y_idx: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray | None,
    ) -> list[Rule]:
        """Backward elimination: greedily remove rules that don't help F1."""
        eval_idx = val_idx if val_idx is not None else train_idx
        current_rules = list(rules)
        current_f1 = self._evaluate_rule_list_f1(current_rules, X, y_idx, eval_idx)

        improved = True
        while improved and len(current_rules) > 1:
            improved = False
            best_drop_idx = None
            best_drop_f1 = -np.inf
            for i in range(len(current_rules)):
                reduced = current_rules[:i] + current_rules[i + 1:]
                reduced_f1 = self._evaluate_rule_list_f1(reduced, X, y_idx, eval_idx)
                # A rule is removed only when its *absence* improves (or at
                # worst barely changes) val F1 by at least compaction_min_gain.
                # Setting compaction_min_gain > 0 makes compaction conservative
                # so rules with marginal-but-real contributions are kept.
                remove_threshold = float(self.compaction_min_gain) - 1e-9
                if reduced_f1 >= current_f1 + remove_threshold and reduced_f1 > best_drop_f1:
                    best_drop_f1 = reduced_f1
                    best_drop_idx = i

            if best_drop_idx is not None:
                current_rules = current_rules[:best_drop_idx] + current_rules[best_drop_idx + 1:]
                current_f1 = best_drop_f1
                improved = True

        return current_rules

    @property
    def _f1_kwargs(self) -> dict:
        """Return f1_score kwargs: 'binary' with pos_label='1' during OvR, 'macro' otherwise."""
        if getattr(self, "_ovr_binary_f1_", False):
            return {"average": "binary", "pos_label": "1", "zero_division": 0}
        return {"average": "macro", "zero_division": 0}

    def _evaluate_rule_list_f1(
        self,
        rules: list[Rule],
        X: np.ndarray,
        y_idx: np.ndarray,
        eval_idx: np.ndarray,
    ) -> float:
        ruleset = self._build_ruleset(rules)
        y_pred = predict_from_ruleset(ruleset, X[eval_idx])
        y_true = np.asarray(self.classes_[y_idx[eval_idx]], dtype=str)
        y_pred = np.asarray(y_pred, dtype=str)
        return float(f1_score(y_true, y_pred, **self._f1_kwargs))

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
                    rule_id="rulelcs_default_prior",
                    metadata={"source": "ruleplcs", "kind": "class_prior"},
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

            # -- Low-cardinality detection: numeric features with few
            #    unique values are *also* treated as categorical so that
            #    equality splits (== value) are generated in addition to
            #    the standard threshold splits (<= / >).
            col_arr = np.asarray(column)
            is_numeric = np.issubdtype(col_arr.dtype, np.number)
            n_unique = len(np.unique(col_arr))
            is_low_cardinality = (
                is_numeric and n_unique <= self.low_cardinality_threshold
            )

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
                                    rule_id=f"rulelcs_rule_f{feature_idx}_le",
                                    metadata={"source": "ruleplcs", "gain": float(gain)},
                                ),
                                gain,
                                left_cov,
                                seen_signatures,
                            ),
                            self._candidate_from_rule(
                                Rule(
                                    atoms=[Atom(feature=feature_name, op=">", value=float(threshold))],
                                    scores=self._distribution_to_scores(right_counts),
                                    rule_id=f"rulelcs_rule_f{feature_idx}_gt",
                                    metadata={"source": "ruleplcs", "gain": float(gain)},
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
                                rule_id=f"rulelcs_rule_f{feature_idx}_between_{interval_idx}",
                                metadata={"source": "ruleplcs", "gain": float(interval_gain)},
                            ),
                            interval_gain,
                            coverage,
                            seen_signatures,
                        )
                    )

                # For high-cardinality numeric features we are done;
                # for low-cardinality ones fall through to the categorical block.
                if not is_low_cardinality:
                    continue

            # -- Categorical / low-cardinality equality splits --
            if self.enable_categorical_rules or is_low_cardinality:
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
                                rule_id=f"rulelcs_rule_f{feature_idx}_eq_{category_idx}",
                                metadata={"source": "ruleplcs", "gain": float(gain), "category": category},
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
                                rule_id=f"rulelcs_rule_f{feature_idx}_in_{group_idx}",
                                metadata={"source": "ruleplcs", "gain": float(gain), "group": group_values},
                            ),
                            gain,
                            coverage,
                            seen_signatures,
                        )
                    )

        candidates = [c for c in candidates if c is not None]
        candidates.sort(key=lambda c: (c.gain, c.coverage, -len(c.rule.atoms)), reverse=True)

        feat_name_to_idx = {str(fn): idx for idx, fn in enumerate(self.feature_names_in_)}

        # -- Conjunctive candidates: combine top single-atom rules from
        #    different features into 2-atom conjunctions.  This is essential
        #    for datasets where individual features are weak (e.g. categorical
        #    multi-class datasets like car_evaluation).
        single_atom = [c for c in candidates if len(c.rule.atoms) == 1]
        n_conj_base = min(len(single_atom), max(6, int(self.candidate_pool_size // 4)))
        top_singles = single_atom[:n_conj_base]
        conj2_candidates: list[_CandidateRule] = []
        for i in range(len(top_singles)):
            for j in range(i + 1, len(top_singles)):
                a_i = top_singles[i].rule.atoms[0]
                a_j = top_singles[j].rule.atoms[0]
                if a_i.feature == a_j.feature:
                    continue
                c = self._make_conjunction_candidate(
                    [a_i, a_j], X, y_idx, n_classes, feat_name_to_idx,
                    (top_singles[i].gain + top_singles[j].gain) / 2.0,
                    seen_signatures,
                )
                if c is not None:
                    conj2_candidates.append(c)

        candidates.extend(conj2_candidates)

        # -- 3-atom conjunctions: extend top 2-atom rules with an extra
        #    single atom from a different feature.  This is critical for
        #    minority classes on datasets like car_evaluation where 3–4
        #    feature interactions determine class membership.
        conj3_candidates: list[_CandidateRule] = []
        top_conj2 = sorted(conj2_candidates, key=lambda c: c.gain, reverse=True)[
            : max(4, int(self.candidate_pool_size // 6))
        ]
        for c2 in top_conj2:
            used_features = {str(a.feature) for a in c2.rule.atoms}
            for s in top_singles:
                a_s = s.rule.atoms[0]
                if str(a_s.feature) in used_features:
                    continue
                atoms3 = list(c2.rule.atoms) + [a_s]
                c3 = self._make_conjunction_candidate(
                    atoms3, X, y_idx, n_classes, feat_name_to_idx,
                    c2.gain * 0.9,  # slight discount for deeper rules
                    seen_signatures,
                )
                if c3 is not None:
                    conj3_candidates.append(c3)

        candidates.extend(conj3_candidates)
        candidates.sort(key=lambda c: (c.gain, c.coverage, -len(c.rule.atoms)), reverse=True)
        return candidates[: max(1, int(self.candidate_pool_size))]

    def _make_conjunction_candidate(
        self,
        atoms: list[Atom],
        X: np.ndarray,
        y_idx: np.ndarray,
        n_classes: int,
        feat_name_to_idx: dict[str, int],
        base_gain: float,
        seen_signatures: set[tuple[tuple[str, str, str], ...]],
    ) -> _CandidateRule | None:
        """Create a conjunctive candidate rule from a list of atoms."""
        mask = np.ones(X.shape[0], dtype=bool)
        for atom in atoms:
            feat_idx = feat_name_to_idx.get(str(atom.feature))
            if feat_idx is None:
                continue
            col = X[:, feat_idx]
            if atom.op == "==":
                mask &= col == atom.value
            elif atom.op == "<=":
                mask &= col <= atom.value
            elif atom.op == ">":
                mask &= col > atom.value
            elif atom.op == "in":
                vals = atom.value if isinstance(atom.value, (list, tuple)) else [atom.value]
                mask &= np.isin(col, vals)
            elif atom.op == "between":
                lo, hi = atom.value[0], atom.value[1]
                mask &= (col >= lo) & (col <= hi)

        coverage = int(mask.sum())
        if coverage < max(1, self.min_samples_leaf):
            return None

        conj_y = y_idx[mask]
        counts = np.bincount(conj_y, minlength=n_classes).astype(float)

        feat_tag = "_".join(str(a.feature) for a in atoms)
        rule = Rule(
            atoms=atoms,
            scores=self._distribution_to_scores(counts),
            rule_id=f"rulelcs_conj_{feat_tag}",
            metadata={"source": "ruleplcs", "gain": float(base_gain), "kind": "conjunction"},
        )
        return self._candidate_from_rule(rule, base_gain, coverage, seen_signatures)

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
            # Consider a wider range so mid-tier candidates can be combined
            n_add = min(n_candidates, max(4 * max(1, int(self.max_rules)), int(self.beam_width) * 2))
            for idx in candidate_order[:n_add]:
                if idx not in state_set:
                    neighbors.add(tuple(sorted((*state, idx))))

        for idx in state:
            reduced = tuple(v for v in state if v != idx)
            neighbors.add(reduced)

        if state:
            absent = [idx for idx in candidate_order if idx not in state_set][: max(4, int(self.beam_width))]
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

        # -- Windowing (BioHEL-inspired): stochastic subset evaluation --
        wf = float(self.window_fraction)
        if 0.0 < wf < 1.0 and eval_idx.size > 10:
            window_size = max(5, int(round(eval_idx.size * wf)))
            window_idx = self._rng_.choice(eval_idx, size=window_size, replace=False)
        else:
            window_idx = eval_idx

        y_pred = predict_from_ruleset(ruleset, X[window_idx])
        y_true = np.asarray(self.classes_[y_idx[window_idx]], dtype=str)
        y_pred = np.asarray(y_pred, dtype=str)
        f1 = float(f1_score(y_true, y_pred, **self._f1_kwargs))
        n_rules = len(selected_rules)
        n_atoms = sum(len(rule.atoms) for rule in selected_rules)
        complexity = self.complexity_penalty * (
            n_rules + n_atoms / max(1.0, float(self.max_rules))
        )
        coverage_bonus = 0.001 * sum(candidates[idx].coverage for idx in state) / max(1, len(train_idx))

        # -- Token Competition (BioHEL-inspired): penalize rule overlap --
        token_penalty = 0.0
        tc_weight = float(self.token_competition_weight)
        if tc_weight > 0.0 and len(state) >= 2:
            # Count how many rules fire per example; penalize multi-coverage
            fire_count = np.zeros(window_idx.size, dtype=int)
            for idx in state:
                rule = candidates[idx].rule
                # Build a minimal single-rule ruleset to get firing mask
                single_rs = self._build_ruleset([rule])
                single_pred = predict_from_ruleset(single_rs, X[window_idx])
                # A rule "fires" if it changes the prediction from default
                default_rs = self._build_ruleset([])
                default_pred = predict_from_ruleset(default_rs, X[window_idx])
                fires = np.asarray(single_pred, dtype=str) != np.asarray(default_pred, dtype=str)
                fire_count += fires.astype(int)
            # Fraction of examples covered by >1 rule
            overlap_fraction = float((fire_count > 1).mean())
            token_penalty = tc_weight * overlap_fraction

        return f1 - complexity + coverage_bonus - token_penalty

    def _state_atom_count(self, state: tuple[int, ...], candidates: list[_CandidateRule]) -> int:
        return sum(len(candidates[idx].rule.atoms) for idx in state)

    def _distribution_to_scores(self, counts: np.ndarray) -> list[float]:
        if getattr(self, "_ovr_binary_f1_", False) and len(counts) == 2:
            # OvR contrast scoring: use lift-based scores so that rules
            # covering regions enriched in the positive class actually
            # predict class 1 under argmax_sum aggregation.
            total = float(counts.sum())
            if total == 0:
                return [0.0, 0.0]
            pos_rate = float(counts[1]) / total
            base_rate = getattr(self, "_ovr_base_rate_", 0.5)
            if pos_rate > base_rate:
                # Positive-enriched region → favor class 1
                lift = min(pos_rate / max(base_rate, 1e-6), 10.0)
                return [0.0, float(lift)]
            else:
                # Negative-enriched region → favor class 0
                lift = min((1.0 - pos_rate) / max(1.0 - base_rate, 1e-6), 10.0)
                return [float(lift), 0.0]
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



