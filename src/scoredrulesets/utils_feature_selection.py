"""
Utility functions for feature selection before rule-based estimators.

Example usage:

from scoredrulesets.utils.feature_selection import select_features
X_new, feature_names = select_features(X, y, method="kbest", k=20)

Supported methods:
- kbest: SelectKBest (e.g. mutual_info_classif)
- rfe: Recursive Feature Elimination
- boruta: BorutaPy (if installed)

"""

from typing import Tuple, List, Optional, Any
import numpy as np

from .preprocessing.feature_selection import build_feature_selector, get_selected_feature_names


def select_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    method: str = "kbest",
    k: int = 20,
    score_func: Any = None,
    estimator: Any = None,
    **kwargs,
) -> Tuple[np.ndarray, List[str]]:
    """
    Select the most relevant features using the chosen method.
    Returns transformed data and selected feature names.
    """
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]

    selector = build_feature_selector(
        method=method,
        k=int(k),
        score_func=score_func,
        estimator=estimator,
        params=kwargs,
    )
    X_new = selector.fit_transform(X, y)
    selected_names, _ = get_selected_feature_names(
        selector,
        list(feature_names),
        transformed_width=int(np.asarray(X_new).shape[1]),
    )
    return np.asarray(X_new), selected_names

