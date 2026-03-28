from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ..io import dump_ruleset_json, load_ruleset_json
from ..runtime import predict as predict_from_ruleset
from ..runtime import predict_proba as predict_proba_from_ruleset
from ..schema import ScoredRuleSet
from .backends import build_backend_estimator
from .base import BaseRuleSetEstimator
from .tree_transform import TreeTransformParams, estimator_to_scored_ruleset


class ScoredRuleSetClassifier(BaseRuleSetEstimator):
    """Sklearn-compatible wrapper with post-hoc transformation into Scored Rule Sets.

    Parameters
    ----------
    backend : str
        Backend estimator to use (e.g. 'cart', 'hs', 'rulekit', 'exstracs',
        'logicgp', 'rulelcs', 'rulenln', 'rulegp', 'rulegp2').
    backend_params : dict, optional
        Parameters forwarded to the backend estimator constructor.
    transform_params : dict, optional
        Parameters for the tree-to-ruleset transformation (CART/HS).
    exstracs_params : dict, optional
        Parameters for ExSTraCS shrinking.
    preprocessing : dict, optional
        Preprocessing configuration applied *before* the backend sees the data.
        Supported keys:

        - ``"feature_selection"`` (str): method name for :func:`select_features`
          (``"kbest"``, ``"rfe"``, ``"boruta"``).
        - ``"k"`` (int): number of features to keep (default 20).
        - ``"max_thresholds_per_feature"`` (int): cap on numeric thresholds
          for native backends that build their own atom candidates
          (gp, rulelcs, rulenln, logicgp).
    estimator : object, optional
        A pre-built sklearn-compatible estimator; overrides *backend*.
    random_state : int, optional
        Random seed for reproducibility.
    """

    # Backends whose estimator exposes `max_thresholds_per_feature` attribute
    _NATIVE_THRESHOLD_BACKENDS = frozenset({"gp", "rulelcs", "rulenln", "logicgp", "rulegp", "rulegp2"})

    def __init__(
        self,
        backend: str = "hs",
        backend_params: dict[str, Any] | None = None,
        transform_params: dict[str, Any] | None = None,
        exstracs_params: dict[str, Any] | None = None,
        preprocessing: dict[str, Any] | None = None,
        estimator: Any | None = None,
        random_state: int | None = None,
    ):
        self.backend = backend
        self.backend_params = backend_params
        self.transform_params = transform_params
        self.exstracs_params = exstracs_params
        self.preprocessing = preprocessing
        self.estimator = estimator
        self.random_state = random_state

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X, y):
        X_valid, y_valid = check_X_y(X, y, dtype=None)
        self.n_features_in_ = X_valid.shape[1]
        self.feature_names_in_ = self._infer_feature_names(X_valid)

        # ----- Preprocessing: feature selection (Step 1) -----
        self.selected_feature_indices_: np.ndarray | None = None
        preproc = self.preprocessing or {}
        fs_method = preproc.get("feature_selection")
        if fs_method is not None:
            from ..utils_feature_selection import select_features

            k = preproc.get("k", min(20, X_valid.shape[1]))
            X_reduced, selected_names = select_features(
                X_valid,
                y_valid,
                feature_names=list(self.feature_names_in_),
                method=fs_method,
                k=int(k),
            )
            # Store boolean mask for predict-time slicing
            mask = np.isin(self.feature_names_in_, selected_names)
            self.selected_feature_indices_ = np.where(mask)[0]
            # Update working data – but keep n_features_in_ as the ORIGINAL
            # width so sklearn checks pass (predict validates against it).
            X_valid = X_reduced
            self.feature_names_in_ = list(selected_names)

        # ----- Build backend estimator -----
        if self.estimator is not None:
            self.estimator_ = clone(self.estimator)
        else:
            self.estimator_ = build_backend_estimator(
                backend=self.backend,
                backend_params=self.backend_params,
                random_state=self.random_state,
            )

        # ----- Preprocessing: threshold budget (Step 2) -----
        max_thr = preproc.get("max_thresholds_per_feature")
        if max_thr is not None and self.backend.lower() in self._NATIVE_THRESHOLD_BACKENDS:
            if hasattr(self.estimator_, "max_thresholds_per_feature"):
                self.estimator_.max_thresholds_per_feature = int(max_thr)
            else:
                # For backends that accept it as a constructor param
                try:
                    self.estimator_.set_params(max_thresholds_per_feature=int(max_thr))
                except (ValueError, TypeError):
                    pass  # Backend does not support this parameter – skip silently

        self.estimator_.fit(X_valid, y_valid)
        self.classes_ = np.asarray(getattr(self.estimator_, "classes_", np.unique(y_valid)))

        # Select the appropriate transformation based on the backend.
        backend_lower = self.backend.lower()
        
        if backend_lower == "rulekit":
            from .ruleset_transform import rulekit_to_scored_ruleset

            self.ruleset_ = rulekit_to_scored_ruleset(
                self.estimator_,
                self.classes_.tolist(),
                self.feature_names_in_,
                y_valid,
            )
        elif backend_lower == "exstracs":
            from .ruleset_transform import exstracs_to_scored_ruleset

            self.ruleset_ = exstracs_to_scored_ruleset(
                estimator=self.estimator_,
                class_labels=self.classes_.tolist(),
                feature_names=self.feature_names_in_,
            )

            # Apply ExSTraCS shrinking (if configured).
            if self.exstracs_params:
                self.ruleset_ = self._apply_exstracs_shrinking(
                    self.ruleset_,
                    X_valid,
                    y_valid,
                )
        elif backend_lower == "logicgp":
            # logicGP manages its ScoredRuleSet internally; use it directly.
            if hasattr(self.estimator_, "ruleset_"):
                self.ruleset_ = self.estimator_.ruleset_
            else:
                raise RuntimeError(
                    "LogicGPClassifier has no 'ruleset_' after fit(). "
                    "Please check logicgp.py for errors."
                )

        elif backend_lower == "rulenln":
            if hasattr(self.estimator_, "ruleset_"):
                self.ruleset_ = self.estimator_.ruleset_
            else:
                raise RuntimeError(
                    "RuleNLNClassifier has no 'ruleset_' after fit(). "
                    "Please check rulenln.py for errors."
                )
        elif backend_lower == "rulekit_native":
            if hasattr(self.estimator_, "ruleset_"):
                self.ruleset_ = self.estimator_.ruleset_
            else:
                raise RuntimeError(
                    "RuleKitNativeClassifier has no 'ruleset_' after fit(). "
                    "Please check rulekit_native.py for errors."
                )
        elif backend_lower == "rulegp":
            if hasattr(self.estimator_, "ruleset_"):
                self.ruleset_ = self.estimator_.ruleset_
            else:
                raise RuntimeError(
                    "RuleGPClassifier has no 'ruleset_' after fit(). "
                    "Please check rulegp.py for errors."
                )
        elif backend_lower == "rulegp2":
            if hasattr(self.estimator_, "ruleset_"):
                self.ruleset_ = self.estimator_.ruleset_
            else:
                raise RuntimeError(
                    "RuleGP2Classifier has no 'ruleset_' after fit(). "
                    "Please check rulegp2.py for errors."
                )
        elif backend_lower == "rulelcs":
            if hasattr(self.estimator_, "ruleset_"):
                self.ruleset_ = self.estimator_.ruleset_
            else:
                raise RuntimeError(
                    "RuleLCSClassifier has no 'ruleset_' after fit(). "
                    "Please check rulelcs.py for errors."
                )
        else:
            # Tree-basierte Transformation (CART, HS)
            transform_cfg = TreeTransformParams(**(self.transform_params or {}))
            self.ruleset_ = estimator_to_scored_ruleset(
                estimator=self.estimator_,
                class_labels=self.classes_.tolist(),
                feature_names=self.feature_names_in_,
                params=transform_cfg,
                X_ref=X_valid,
            )

        # Route predictions through the ScoredRuleSet for all backends,
        # so benchmarks evaluate the actual transformed representation.
        self.is_ruleset_mode_ = True
        self.transformation_lossy_ = False
        return self

    def _apply_exstracs_shrinking(self, ruleset: ScoredRuleSet, X: np.ndarray, y: np.ndarray) -> ScoredRuleSet:
        """Apply ExSTraCS shrinking parameters."""
        from .exstracs_shrinking import ExSTraCSPruningParams, exstracs_apply_all_shrinking

        sanitized = self._sanitize_exstracs_params(self.exstracs_params)
        params = ExSTraCSPruningParams(**sanitized)

        # For aggressive pruning: split training data.
        X_train_split = X
        y_train_split = y
        # Default: use training data as reference for conservative pruning.
        X_val_split = X
        y_val_split = y

        if params.aggressive_prune:
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, stratify=y
            )

        return exstracs_apply_all_shrinking(
            ruleset,
            X_val=X_val_split,
            y_val=y_val_split,
            params=params,
        )

    @staticmethod
    def _sanitize_exstracs_params(exstracs_params: dict[str, Any] | None) -> dict[str, Any]:
        """Filter unknown ExSTraCS keys and warn about likely typos."""
        from .exstracs_shrinking import ExSTraCSPruningParams

        if not exstracs_params:
            return {}

        allowed = set(ExSTraCSPruningParams.__dataclass_fields__.keys())
        sanitized = {k: v for k, v in exstracs_params.items() if k in allowed}
        unknown = sorted(k for k in exstracs_params if k not in allowed)
        if unknown:
            warnings.warn(
                "Ignoring unknown exstracs_params keys: " + ", ".join(unknown),
                UserWarning,
            )
        return sanitized

    def _prepare_X_for_prediction(self, X_valid: np.ndarray) -> np.ndarray:
        """Prepare X for ScoredRuleSet prediction.
        
        LogicGP requires discretized data because atoms in the ScoredRuleSet
        are defined over bin indices rather than raw values.
        """
        backend_lower = self.backend.lower()
        if backend_lower == "logicgp" and hasattr(self.estimator_, "_binners_"):
            from .logicgp import _discretize_features
            X_disc, _, _ = _discretize_features(
                X_valid,
                fitted_binners=self.estimator_._binners_,
                cat_masks=self.estimator_._cat_masks_,
            )
            return X_disc
        return X_valid

    def _apply_feature_selection(self, X: np.ndarray) -> np.ndarray:
        """Slice X to the features selected during fit(), if applicable."""
        indices = getattr(self, "selected_feature_indices_", None)
        if indices is not None:
            return X[:, indices]
        return X

    def predict(self, X):
        check_is_fitted(self, "ruleset_")
        X_valid: np.ndarray = np.asarray(check_array(X, dtype=None))
        if self.n_features_in_ is not None and X_valid.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_valid.shape[1]} features, but {self.__class__.__name__} "
                f"is expecting {self.n_features_in_} features as input"
            )

        # Apply feature selection if it was used during fit()
        X_valid = self._apply_feature_selection(X_valid)

        if self.is_ruleset_mode_:
            X_prepared = self._prepare_X_for_prediction(X_valid)
            return predict_from_ruleset(self.ruleset_, X_prepared)

        estimator: Any = self.estimator_
        return estimator.predict(X_valid)

    def predict_proba(self, X):
        check_is_fitted(self, "ruleset_")
        X_valid: np.ndarray = np.asarray(check_array(X, dtype=None))
        if self.n_features_in_ is not None and X_valid.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_valid.shape[1]} features, but {self.__class__.__name__} "
                f"is expecting {self.n_features_in_} features as input"
            )

        # Apply feature selection if it was used during fit()
        X_valid = self._apply_feature_selection(X_valid)

        if self.is_ruleset_mode_:
            X_prepared = self._prepare_X_for_prediction(X_valid)
            return predict_proba_from_ruleset(self.ruleset_, X_prepared)

        estimator: Any = self.estimator_
        if hasattr(estimator, "predict_proba"):
            return estimator.predict_proba(X_valid)
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
        model.n_features_in_ = cls._infer_ruleset_n_features(model.ruleset_)
        model.transform_params = asdict(TreeTransformParams())
        model.estimator_ = None
        model.is_ruleset_mode_ = True
        return model

    @staticmethod
    def _infer_feature_names(X) -> list[str]:
        # Wenn DataFrame, verwende echte Spaltennamen
        try:
            import pandas as pd
            if isinstance(X, pd.DataFrame):
                return list(X.columns)
        except ImportError:
            pass
        # Fallback: NumPy-Array
        return [f"f{i}" for i in range(X.shape[1])]

    @staticmethod
    def _infer_ruleset_n_features(ruleset: ScoredRuleSet) -> int | None:
        if ruleset.feature_names:
            return len(ruleset.feature_names)
        max_idx = -1
        for rule in ruleset.rules:
            for atom in rule.atoms:
                if isinstance(atom.feature, int):
                    max_idx = max(max_idx, atom.feature)
        if max_idx >= 0:
            return max_idx + 1
        return None

