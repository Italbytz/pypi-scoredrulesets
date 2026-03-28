"""AutoML meta-estimator that selects the best backend for Scored Rule Sets.

Usage::

    from scoredrulesets import AutoScoredRuleSetClassifier

    clf = AutoScoredRuleSetClassifier(
        candidate_backends=["cart", "hs", "rulelcs", "gp"],
        cv=5,
        scoring="f1_weighted",
    )
    clf.fit(X_train, y_train)
    print(clf.best_backend_)       # e.g. "hs"
    print(clf.cv_results_)         # dict[str, float]
    predictions = clf.predict(X_test)
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ..schema import ScoredRuleSet
from .base import BaseRuleSetEstimator
from .sklearn_wrapper import ScoredRuleSetClassifier


_DEFAULT_BACKENDS = ["cart", "hs", "rulelcs"]


class AutoScoredRuleSetClassifier(BaseRuleSetEstimator):
    """Select the best backend via cross-validation, then refit on the full data.

    Parameters
    ----------
    candidate_backends : list[str] | None
        List of backend names to evaluate (default: ``["cart", "hs",
        "rulelcs"]``).  Any backend supported by
        :class:`ScoredRuleSetClassifier` can be listed.
    backend_params : dict[str, dict] | None
        Per-backend constructor parameters.  Keys are backend names, values
        are dicts forwarded as *backend_params*.  Backends not listed here
        use their default parameters.
    cv : int
        Number of cross-validation folds (default 5).
    scoring : str
        Sklearn scoring metric (default ``"f1_weighted"``).
    preprocessing : dict | None
        Preprocessing dict forwarded to each candidate
        :class:`ScoredRuleSetClassifier` (feature selection, threshold budget).
    timeout_per_backend : float | None
        Maximum seconds for a single backend's CV loop.  If exceeded the
        backend is skipped with a warning (default ``None`` — no limit).
    random_state : int | None
        Random seed.
    """

    def __init__(
        self,
        candidate_backends: list[str] | None = None,
        backend_params: dict[str, dict[str, Any]] | None = None,
        cv: int = 5,
        scoring: str = "f1_weighted",
        preprocessing: dict[str, Any] | None = None,
        timeout_per_backend: float | None = None,
        random_state: int | None = None,
    ):
        self.candidate_backends = candidate_backends
        self.backend_params = backend_params
        self.cv = cv
        self.scoring = scoring
        self.preprocessing = preprocessing
        self.timeout_per_backend = timeout_per_backend
        self.random_state = random_state

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X, y):
        X_valid, y_valid = check_X_y(X, y, dtype=None)
        self.n_features_in_ = X_valid.shape[1]

        backends = list(self.candidate_backends or _DEFAULT_BACKENDS)
        per_backend_params = dict(self.backend_params or {})

        cv_results: dict[str, float] = {}
        best_backend: str | None = None
        best_score = -np.inf

        cv_splitter = StratifiedKFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )

        for backend in backends:
            bp = per_backend_params.get(backend)
            clf = ScoredRuleSetClassifier(
                backend=backend,
                backend_params=bp,
                preprocessing=self.preprocessing,
                random_state=self.random_state,
            )
            t0 = time.monotonic()
            try:
                scores = cross_val_score(
                    clf, X_valid, y_valid,
                    cv=cv_splitter,
                    scoring=self.scoring,
                    error_score="raise",
                )
                elapsed = time.monotonic() - t0
                if (
                    self.timeout_per_backend is not None
                    and elapsed > self.timeout_per_backend
                ):
                    warnings.warn(
                        f"AutoScoredRuleSet: backend '{backend}' exceeded "
                        f"timeout ({elapsed:.1f}s > {self.timeout_per_backend:.1f}s). "
                        f"Result included but consider removing slow backends.",
                        UserWarning,
                    )
                mean_score = float(scores.mean())
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"AutoScoredRuleSet: backend '{backend}' failed during CV: {exc}",
                    UserWarning,
                )
                mean_score = float("-inf")

            cv_results[backend] = mean_score
            if mean_score > best_score:
                best_score = mean_score
                best_backend = backend

        if best_backend is None:
            raise RuntimeError(
                "AutoScoredRuleSet: all candidate backends failed during "
                "cross-validation. Check warnings for details."
            )

        # Re-fit the winner on the full training data
        bp = per_backend_params.get(best_backend)
        winner = ScoredRuleSetClassifier(
            backend=best_backend,
            backend_params=bp,
            preprocessing=self.preprocessing,
            random_state=self.random_state,
        )
        winner.fit(X_valid, y_valid)

        # Expose results
        self.best_backend_ = best_backend
        self.best_score_ = best_score
        self.cv_results_ = cv_results
        self.best_estimator_ = winner
        self.classes_ = winner.classes_
        self.ruleset_ = winner.ruleset_
        self.feature_names_in_ = winner.feature_names_in_
        return self

    # ------------------------------------------------------------------
    # predict / predict_proba / to_ruleset
    # ------------------------------------------------------------------

    def predict(self, X):
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.predict_proba(X)

    def to_ruleset(self) -> ScoredRuleSet:
        check_is_fitted(self, "ruleset_")
        return self.ruleset_



