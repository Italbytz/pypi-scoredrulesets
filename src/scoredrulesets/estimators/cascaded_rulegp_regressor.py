"""
CascadedRuleGPRegressor - Two-stage residual boosting using Scored Rule Sets.
=============================================================================

Implements the two-stage cascaded regression framework proposed for EvoStar (EuroGP).
- Stage 1: Fits a compact primary rule set (R1) on y to capture the global trend.
- Stage 2: Fits a compact secondary rule set (R2) on the training residuals e = y - ŷ1.
- Joint Model: ŷ(x) = R1(x) + R2(x)
"""

from __future__ import annotations

from typing import Literal
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .base import BaseRuleSetEstimator
from .rulegp_regressor import RuleGPRegressor
from ..schema import AggregationSpec, Rule, ScoredRuleSet
from ..runtime import predict_regression as predict_regression_from_ruleset


class CascadedRuleGPRegressor(BaseRuleSetEstimator, RegressorMixin):
    """
    Two-stage cascaded genetic programming regressor.

    Parameters
    ----------
    max_rules_stage1 : int, default=3
        Maximum rules allocated to the global trend stage.
    max_rules_stage2 : int, default=3
        Maximum rules allocated to the residual correction stage.
    prediction_type : {"constant", "linear"}, default="constant"
        Whether rules use constant means or local linear models.
    max_generations : int, default=50
        Max generations per stage.
    population_size : int, default=50
        Population size per stage.
    stagnation_generations : int, default=20
        Generations without improvement before early stopping.
    random_state : int or None, default=None
        Random state for reproducibility.
    """

    def __init__(
        self,
        max_rules_stage1: int = 3,
        max_rules_stage2: int = 3,
        prediction_type: Literal["constant", "linear"] = "constant",
        max_generations: int = 50,
        population_size: int = 50,
        stagnation_generations: int = 20,
        max_fit_seconds: float | None = None,
        feature_names: list[str] | None = None,
        random_state: int | None = None,
    ):
        self.max_rules_stage1 = max_rules_stage1
        self.max_rules_stage2 = max_rules_stage2
        self.prediction_type = prediction_type
        self.max_generations = max_generations
        self.population_size = population_size
        self.stagnation_generations = stagnation_generations
        self.max_fit_seconds = max_fit_seconds
        self.feature_names = feature_names
        self.random_state = random_state

    def fit(self, X, y):
        X_arr, y_arr = check_X_y(X, y, dtype=None, y_numeric=True)
        self.n_features_in_ = X_arr.shape[1]
        self.feature_names_in_ = self.feature_names or [f"f{i}" for i in range(self.n_features_in_)]

        rng = np.random.default_rng(self.random_state)
        seed1 = int(rng.integers(0, 1_000_000))
        seed2 = int(rng.integers(0, 1_000_000))

        # Stage 1: Fit primary model on target y
        self.stage1_ = RuleGPRegressor(
            max_rules=self.max_rules_stage1,
            prediction_type=self.prediction_type,
            max_generations=self.max_generations,
            population_size=self.population_size,
            stagnation_generations=self.stagnation_generations,
            max_fit_seconds=self.max_fit_seconds,
            feature_names=self.feature_names_in_,
            random_state=seed1,
        )
        self.stage1_.fit(X_arr, y_arr)
        pred_stage1_train = self.stage1_.predict(X_arr)

        # Compute residuals e = y - ŷ1
        residuals_train = y_arr - pred_stage1_train

        # Stage 2: Fit secondary model on residuals e
        self.stage2_ = RuleGPRegressor(
            max_rules=self.max_rules_stage2,
            prediction_type=self.prediction_type,
            max_generations=self.max_generations,
            population_size=self.population_size,
            stagnation_generations=self.stagnation_generations,
            max_fit_seconds=self.max_fit_seconds,
            feature_names=self.feature_names_in_,
            random_state=seed2,
        )
        self.stage2_.fit(X_arr, residuals_train)

        # Fuse both stages into a unified ScoredRuleSet
        self.ruleset_ = self._fuse_rulesets(self.stage1_.ruleset_, self.stage2_.ruleset_)
        return self

    def _fuse_rulesets(self, rs1: ScoredRuleSet, rs2: ScoredRuleSet) -> ScoredRuleSet:
        """Merges rules from Stage 1 and Stage 2 with stage metadata."""
        fused_rules: list[Rule] = []

        # Tag and append stage 1 rules
        for idx, r in enumerate(rs1.rules):
            meta = dict(getattr(r, "metadata", {}) or {})
            meta["stage"] = 1
            meta["original_rule_id"] = r.rule_id
            fused_rules.append(
                Rule(
                    atoms=list(r.atoms),
                    scores=list(r.scores),
                    rule_id=f"stage1_{r.rule_id}",
                    metadata=meta,
                )
            )

        # Tag and append stage 2 rules
        for idx, r in enumerate(rs2.rules):
            meta = dict(getattr(r, "metadata", {}) or {})
            meta["stage"] = 2
            meta["original_rule_id"] = r.rule_id
            fused_rules.append(
                Rule(
                    atoms=list(r.atoms),
                    scores=list(r.scores),
                    rule_id=f"stage2_{r.rule_id}",
                    metadata=meta,
                )
            )

        ruleset = ScoredRuleSet(
            class_labels=[],
            task_type="regression",
            feature_names=list(self.feature_names_in_),
            rules=fused_rules,
            aggregation=AggregationSpec(type="cascaded_sum"),
            metadata={
                "estimator": "CascadedRuleGPRegressor",
                "prediction_type": self.prediction_type,
                "stage1_rules_count": len(rs1.rules),
                "stage2_rules_count": len(rs2.rules),
                "total_rules_count": len(fused_rules),
            },
        )
        ruleset.validate()
        return ruleset

    def predict(self, X):
        check_is_fitted(self, "ruleset_")
        X_arr = check_array(X, dtype=None)
        # Prediction is the sum of stage 1 and stage 2 predictions
        pred1 = self.stage1_.predict(X_arr)
        pred2 = self.stage2_.predict(X_arr)
        return pred1 + pred2

    def score(self, X, y, sample_weight=None):
        return r2_score(y, self.predict(X), sample_weight=sample_weight)

    def to_ruleset(self) -> ScoredRuleSet:
        check_is_fitted(self, "ruleset_")
        return self.ruleset_
