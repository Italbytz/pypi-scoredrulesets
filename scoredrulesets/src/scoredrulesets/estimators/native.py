from __future__ import annotations

import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ..runtime import predict as predict_from_ruleset
from ..runtime import predict_proba as predict_proba_from_ruleset
from ..schema import AggregationSpec, Rule, ScoredRuleSet
from .base import BaseRuleSetEstimator


class NativeScoredRuleSetClassifier(BaseRuleSetEstimator):
    """Minimaler nativer Schaetzer, der ein priorbasiertes Scored Rule Set lernt."""

    def __init__(
        self,
        aggregation: str = "softmax_sum",
        temperature: float = 1.0,
        include_default_rule: bool = True,
    ):
        self.aggregation = aggregation
        self.temperature = temperature
        self.include_default_rule = include_default_rule

    def fit(self, X, y):
        X_valid, y_valid = check_X_y(X, y, dtype=None)
        self.feature_names_in_ = [f"f{i}" for i in range(X_valid.shape[1])]

        classes, counts = np.unique(y_valid, return_counts=True)
        self.classes_ = classes

        priors = counts / np.sum(counts)
        if self.aggregation == "softmax_sum":
            # Mit log-priors wird softmax wieder zu den empirischen Priors.
            default_scores = np.log(np.maximum(priors, 1e-12)).tolist()
        else:
            default_scores = priors.tolist()

        rules = []
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
            feature_names=self.feature_names_in_,
            aggregation=AggregationSpec(type=self.aggregation, temperature=self.temperature),
            rules=rules,
            metadata={"source": "native_baseline", "model_type": "prior_only"},
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

