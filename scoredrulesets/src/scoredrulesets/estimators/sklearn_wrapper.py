from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ..io import dump_ruleset_json, load_ruleset_json
from ..runtime import predict as predict_from_ruleset
from ..runtime import predict_proba as predict_proba_from_ruleset
from ..schema import ScoredRuleSet
from .backends import build_backend_estimator
from .base import BaseRuleSetEstimator
from .tree_transform import TreeTransformParams, estimator_to_scored_ruleset


class ScoredRuleSetClassifier(BaseRuleSetEstimator):
    """Sklearn-kompatibler Wrapper mit post-hoc Transformation in Scored Rule Sets."""

    def __init__(
        self,
        backend: str = "hs",
        backend_params: dict[str, Any] | None = None,
        transform_params: dict[str, Any] | None = None,
        estimator: Any | None = None,
        random_state: int | None = None,
    ):
        self.backend = backend
        self.backend_params = backend_params
        self.transform_params = transform_params
        self.estimator = estimator
        self.random_state = random_state

    def fit(self, X, y):
        X_valid, y_valid = check_X_y(X, y, dtype=None)
        self.feature_names_in_ = self._infer_feature_names(X_valid)

        if self.estimator is not None:
            self.estimator_ = clone(self.estimator)
        else:
            self.estimator_ = build_backend_estimator(
                backend=self.backend,
                backend_params=self.backend_params,
                random_state=self.random_state,
            )

        self.estimator_.fit(X_valid, y_valid)
        self.classes_ = np.asarray(getattr(self.estimator_, "classes_", np.unique(y_valid)))

        transform_cfg = TreeTransformParams(**(self.transform_params or {}))
        self.ruleset_ = estimator_to_scored_ruleset(
            estimator=self.estimator_,
            class_labels=self.classes_.tolist(),
            feature_names=self.feature_names_in_,
            params=transform_cfg,
        )
        self.is_ruleset_mode_ = False
        return self

    def predict(self, X):
        check_is_fitted(self, "ruleset_")
        X_valid = check_array(X, dtype=None)

        if self.is_ruleset_mode_:
            return predict_from_ruleset(self.ruleset_, X_valid)

        return self.estimator_.predict(X_valid)

    def predict_proba(self, X):
        check_is_fitted(self, "ruleset_")
        X_valid = check_array(X, dtype=None)

        if self.is_ruleset_mode_:
            return predict_proba_from_ruleset(self.ruleset_, X_valid)

        if hasattr(self.estimator_, "predict_proba"):
            return self.estimator_.predict_proba(X_valid)
        return predict_proba_from_ruleset(self.ruleset_, X_valid)

    def to_ruleset(self) -> ScoredRuleSet:
        check_is_fitted(self, "ruleset_")
        return self.ruleset_

    def save_ruleset(self, path: str | Path) -> None:
        check_is_fitted(self, "ruleset_")
        dump_ruleset_json(self.ruleset_, path)

    @classmethod
    def from_ruleset_json(cls, path: str | Path) -> "ScoredRuleSetClassifier":
        model = cls(backend="hs")
        model.ruleset_ = load_ruleset_json(path)
        model.classes_ = np.asarray(model.ruleset_.class_labels)
        model.feature_names_in_ = model.ruleset_.feature_names
        model.transform_params = asdict(TreeTransformParams())
        model.estimator_ = None
        model.is_ruleset_mode_ = True
        return model

    @staticmethod
    def _infer_feature_names(X: np.ndarray) -> list[str]:
        return [f"f{i}" for i in range(X.shape[1])]

