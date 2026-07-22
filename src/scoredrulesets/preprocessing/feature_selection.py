from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone


class BorutaSelector(BaseEstimator, TransformerMixin):
    """Sklearn-compatible Boruta wrapper.

    The underlying BorutaPy dependency is optional and imported lazily in fit().
    """

    def __init__(self, estimator: Any | None = None, random_state: int | None = 0, **kwargs):
        self.estimator = estimator
        self.random_state = random_state
        self.kwargs = kwargs

    def fit(self, X, y):
        try:
            from boruta import BorutaPy
        except ImportError as exc:
            raise ImportError(
                "BorutaPy is not installed. Install with: pip install boruta"
            ) from exc

        if self.estimator is None:
            from sklearn.ensemble import RandomForestClassifier

            estimator = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=self.random_state)
        else:
            estimator = clone(self.estimator)

        self._boruta_ = BorutaPy(
            estimator,
            n_estimators="auto",
            verbose=0,
            random_state=self.random_state,
            **self.kwargs,
        )
        self._boruta_.fit(X, y)
        self.support_ = np.asarray(self._boruta_.support_, dtype=bool)
        return self

    def transform(self, X):
        if not hasattr(self, "support_"):
            raise RuntimeError("BorutaSelector is not fitted yet.")
        X_arr = np.asarray(X)
        return X_arr[:, self.support_]

    def get_support(self):
        if not hasattr(self, "support_"):
            raise RuntimeError("BorutaSelector is not fitted yet.")
        return self.support_


def build_feature_selector(
    *,
    method: str,
    k: int,
    score_func: Any | None = None,
    estimator: Any | None = None,
    random_state: int | None = None,
    params: dict[str, Any] | None = None,
):
    """Create an sklearn-compatible feature selector by strategy name."""
    method_key = str(method).lower()
    cfg = dict(params or {})

    if method_key == "kbest":
        from sklearn.feature_selection import SelectKBest, mutual_info_classif

        score_fn = score_func or cfg.pop("score_func", None) or mutual_info_classif
        return SelectKBest(score_func=score_fn, k=int(k))

    if method_key == "rfe":
        from sklearn.feature_selection import RFE

        base_est = estimator or cfg.pop("estimator", None)
        if base_est is None:
            from sklearn.ensemble import RandomForestClassifier

            base_est = RandomForestClassifier(
                n_estimators=50,
                n_jobs=-1,
                random_state=random_state,
            )
        return RFE(base_est, n_features_to_select=int(k), step=cfg.pop("step", 1), **cfg)

    if method_key == "boruta":
        base_est = estimator or cfg.pop("estimator", None)
        return BorutaSelector(estimator=base_est, random_state=random_state, **cfg)

    raise ValueError(f"Unknown feature selection method: {method}")


def get_selected_feature_names(
    selector: Any,
    original_feature_names: list[str],
    transformed_width: int,
) -> tuple[list[str], np.ndarray | None]:
    """Infer output feature names and optional index mask from fitted selector."""
    names: list[str] | None = None
    indices: np.ndarray | None = None

    if hasattr(selector, "get_feature_names_out"):
        try:
            names_out = selector.get_feature_names_out(original_feature_names)
            names = [str(name) for name in names_out]
        except Exception:
            names = None

    if hasattr(selector, "get_support"):
        try:
            support = np.asarray(selector.get_support(), dtype=bool)
            if support.shape[0] == len(original_feature_names):
                indices = np.where(support)[0]
                if names is None:
                    names = [name for name, keep in zip(original_feature_names, support) if keep]
        except Exception:
            pass

    if names is None:
        if transformed_width == len(original_feature_names):
            names = list(original_feature_names)
        else:
            names = [f"f{i}" for i in range(transformed_width)]

    return names, indices
