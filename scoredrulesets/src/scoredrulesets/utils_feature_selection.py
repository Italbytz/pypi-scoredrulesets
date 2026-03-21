"""
Utility-Funktionen für Feature-Selektion vor regelbasierten Schätzern.

Beispielnutzung:

from scoredrulesets.utils.feature_selection import select_features
X_new, feature_names = select_features(X, y, method="kbest", k=20)

Unterstützte Methoden:
- kbest: SelectKBest (z. B. mutual_info_classif)
- rfe: Recursive Feature Elimination
- boruta: BorutaPy (falls installiert)

"""

from typing import Tuple, List, Optional, Any
import numpy as np


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
    Selektiert die wichtigsten Features mit dem angegebenen Verfahren.
    Gibt die transformierten Daten und die Namen der selektierten Features zurück.
    """
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]

    if method == "kbest":
        from sklearn.feature_selection import SelectKBest, mutual_info_classif
        if score_func is None:
            score_func = mutual_info_classif
        selector = SelectKBest(score_func, k=k)
        X_new = selector.fit_transform(X, y)
        mask = selector.get_support()
        selected_names = [name for name, keep in zip(feature_names, mask) if keep]
        return X_new, selected_names

    elif method == "rfe":
        from sklearn.feature_selection import RFE
        if estimator is None:
            from sklearn.ensemble import RandomForestClassifier
            estimator = RandomForestClassifier(n_estimators=50, n_jobs=-1)
        selector = RFE(estimator, n_features_to_select=k, step=kwargs.get("step", 1))
        X_new = selector.fit_transform(X, y)
        mask = selector.get_support()
        selected_names = [name for name, keep in zip(feature_names, mask) if keep]
        return X_new, selected_names

    elif method == "boruta":
        try:
            from boruta import BorutaPy
        except ImportError:
            raise ImportError("BorutaPy ist nicht installiert. Installiere mit: pip install boruta")
        if estimator is None:
            from sklearn.ensemble import RandomForestClassifier
            estimator = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        selector = BorutaPy(estimator, n_estimators="auto", verbose=0, random_state=0, **kwargs)
        selector.fit(X, y)
        mask = selector.support_
        X_new = X[:, mask]
        selected_names = [name for name, keep in zip(feature_names, mask) if keep]
        return X_new, selected_names

    else:
        raise ValueError(f"Unbekannte Feature-Selection-Methode: {method}")

