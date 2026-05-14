from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, clone
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_array, check_is_fitted

from ..io import dump_ruleset_json
from ..schema import ScoredRuleSet
from .sklearn_wrapper import ScoredRuleSetClassifier


class ScoredRuleSetClusterer(ClusterMixin, BaseEstimator):
    """Approximate external clustering assignments with a ScoredRuleSet.

    The workflow is intentionally two-stage:
    1) Fit an external clusterer (default: KMeans) and obtain cluster labels.
    2) Fit a ruleset classifier to reproduce these labels.

    This yields a single, explainable ruleset that can be used uniformly via
    ``predict`` and serialized with ``save_ruleset``.
    """

    def __init__(
        self,
        clusterer: Any | None = None,
        n_clusters: int = 3,
        approximation_backend: str = "cart",
        approximation_backend_params: dict[str, Any] | None = None,
        transform_params: dict[str, Any] | None = None,
        random_state: int | None = None,
    ):
        self.clusterer = clusterer
        self.n_clusters = n_clusters
        self.approximation_backend = approximation_backend
        self.approximation_backend_params = approximation_backend_params
        self.transform_params = transform_params
        self.random_state = random_state

    def fit(self, X, y=None):
        X_valid: np.ndarray = np.asarray(check_array(X, dtype=None))
        self.n_features_in_ = X_valid.shape[1]
        self.feature_names_in_ = ScoredRuleSetClassifier._infer_feature_names(X_valid)

        if self.clusterer is not None:
            self.clusterer_ = clone(self.clusterer)
        else:
            self.clusterer_ = KMeans(
                n_clusters=int(self.n_clusters),
                random_state=self.random_state,
                n_init="auto",
            )

        if hasattr(self.clusterer_, "fit_predict"):
            cluster_labels = np.asarray(self.clusterer_.fit_predict(X_valid))
        else:
            self.clusterer_.fit(X_valid)
            inferred = getattr(self.clusterer_, "labels_", None)
            if inferred is None:
                raise ValueError(
                    "Clusterer must provide fit_predict(X) or labels_ after fit(X)."
                )
            cluster_labels = np.asarray(inferred)

        self.cluster_labels_ = cluster_labels
        self.n_clusters_ = int(np.unique(cluster_labels).size)

        self.approximator_ = ScoredRuleSetClassifier(
            backend=self.approximation_backend,
            backend_params=self.approximation_backend_params,
            transform_params=self.transform_params,
            random_state=self.random_state,
        )
        self.approximator_.fit(X_valid, cluster_labels)
        self.ruleset_ = self.approximator_.to_ruleset()

        approx_train = np.asarray(self.approximator_.predict(X_valid))
        self.train_label_fidelity_ = float(np.mean(approx_train == cluster_labels))

        if hasattr(self.clusterer_, "predict"):
            cluster_pred = np.asarray(self.clusterer_.predict(X_valid))
            self.clusterer_predict_consistency_ = float(np.mean(cluster_pred == cluster_labels))

        return self

    def predict(self, X):
        check_is_fitted(self, "ruleset_")
        X_valid: np.ndarray = np.asarray(check_array(X, dtype=None))
        if self.n_features_in_ is not None and X_valid.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_valid.shape[1]} features, but {self.__class__.__name__} "
                f"is expecting {self.n_features_in_} features as input"
            )
        return self.approximator_.predict(X_valid)

    def predict_proba(self, X):
        check_is_fitted(self, "ruleset_")
        return self.approximator_.predict_proba(X)

    def to_ruleset(self) -> ScoredRuleSet:
        check_is_fitted(self, "ruleset_")
        return self.ruleset_

    def save_ruleset(self, path: str | Path) -> None:
        check_is_fitted(self, "ruleset_")
        dump_ruleset_json(self.ruleset_, path)

    @classmethod
    def from_ruleset_json(cls, path: str | Path) -> "ScoredRuleSetClusterer":
        model = cls()
        model.approximator_ = ScoredRuleSetClassifier.from_ruleset_json(path)
        model.ruleset_ = model.approximator_.to_ruleset()
        model.feature_names_in_ = model.approximator_.feature_names_in_
        model.n_features_in_ = model.approximator_.n_features_in_
        model.n_clusters_ = len(model.ruleset_.class_labels)
        return model

    def fidelity_against_clusterer(self, X) -> float:
        """Agreement rate between ruleset prediction and underlying clusterer prediction."""
        check_is_fitted(self, "ruleset_")
        if not hasattr(self, "clusterer_") or not hasattr(self.clusterer_, "predict"):
            raise ValueError("No fitted clusterer with predict() available for fidelity comparison.")
        X_valid: np.ndarray = np.asarray(check_array(X, dtype=None))
        cluster_labels = np.asarray(self.clusterer_.predict(X_valid))
        ruleset_labels = np.asarray(self.predict(X_valid))
        return float(np.mean(cluster_labels == ruleset_labels))
