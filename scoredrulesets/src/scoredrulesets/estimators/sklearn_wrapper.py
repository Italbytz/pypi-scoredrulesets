from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

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
from .ruleset_transform import rulekit_to_scored_ruleset, exstracs_to_scored_ruleset
from .exstracs_shrinking import ExSTraCSPruningParams, exstracs_apply_all_shrinking


class ScoredRuleSetClassifier(BaseRuleSetEstimator):
    """Sklearn-kompatibler Wrapper mit post-hoc Transformation in Scored Rule Sets."""

    def __init__(
        self,
        backend: str = "hs",
        backend_params: dict[str, Any] | None = None,
        transform_params: dict[str, Any] | None = None,
        exstracs_params: dict[str, Any] | None = None,
        estimator: Any | None = None,
        random_state: int | None = None,
    ):
        self.backend = backend
        self.backend_params = backend_params
        self.transform_params = transform_params
        self.exstracs_params = exstracs_params
        self.estimator = estimator
        self.random_state = random_state

    def fit(self, X, y):
        X_valid, y_valid = check_X_y(X, y, dtype=None)
        self.n_features_in_ = X_valid.shape[1]
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

        # Wähle passende Transformation basierend auf Backend
        backend_lower = self.backend.lower()
        
        if backend_lower == "rulekit":
            self.ruleset_ = rulekit_to_scored_ruleset(
                estimator=self.estimator_,
                class_labels=self.classes_.tolist(),
                feature_names=self.feature_names_in_,
            )
        elif backend_lower == "rulefit":
            from .ruleset_transform import rulefit_to_scored_ruleset
            self.ruleset_ = rulefit_to_scored_ruleset(
                estimator=self.estimator_,
                class_labels=self.classes_.tolist(),
                feature_names=self.feature_names_in_,
            )
        elif backend_lower == "exstracs":
            self.ruleset_ = exstracs_to_scored_ruleset(
                estimator=self.estimator_,
                class_labels=self.classes_.tolist(),
                feature_names=self.feature_names_in_,
            )

            # Debug-Ausgabe: Default-Regel und Score-Verteilung
            default_rules = [r for r in self.ruleset_.rules if not getattr(r, 'atoms', None)]
            print(f"[DEBUG ExSTraCS->ScoredRuleSet] n_rules={len(self.ruleset_.rules)} | n_default_rules={len(default_rules)}")
            if default_rules:
                print(f"[DEBUG ExSTraCS->ScoredRuleSet] Default-Regel scores: {default_rules[0].scores}")
            # Score-Verteilung der ersten 5 Regeln
            for i, r in enumerate(self.ruleset_.rules[:5]):
                print(f"[DEBUG ExSTraCS->ScoredRuleSet] Regel {i}: Atome={len(r.atoms)} Scores={r.scores}")

            # Wende ExSTraCS Shrinking an (falls konfiguriert), aber deaktiviere Atom-Pruning explizit
            if self.exstracs_params:
                # Kopiere exstracs_params, aber setze aggressive_prune und prune_atoms auf False
                exstracs_params_no_prune = dict(self.exstracs_params)
                exstracs_params_no_prune["aggressive_prune"] = False
                exstracs_params_no_prune["prune_atoms"] = False
                self.exstracs_params = exstracs_params_no_prune
                self.ruleset_ = self._apply_exstracs_shrinking(
                    self.ruleset_,
                    X_valid,
                    y_valid,
                )
        elif backend_lower == "logicgp":
            # logicGP verwaltet sein ScoredRuleSet intern – direkt uebernehmen.
            if hasattr(self.estimator_, "ruleset_"):
                self.ruleset_ = self.estimator_.ruleset_
            else:
                raise RuntimeError(
                    "LogicGPClassifier hat kein 'ruleset_' nach fit(). "
                    "Bitte logicgp.py auf Fehler pruefen."
                )
        elif backend_lower == "pittsburgh":
            if hasattr(self.estimator_, "ruleset_"):
                self.ruleset_ = self.estimator_.ruleset_
            else:
                raise RuntimeError(
                    "PittsburghRuleSetClassifier has no 'ruleset_' after fit(). "
                    "Please check pittsburgh.py for errors."
                )
        elif backend_lower == "michigan":
            if hasattr(self.estimator_, "ruleset_"):
                self.ruleset_ = self.estimator_.ruleset_
            else:
                raise RuntimeError(
                    "MichiganRuleSetClassifier has no 'ruleset_' after fit(). "
                    "Please check michigan.py for errors."
                )
        else:
            # Tree-basierte Transformation (CART, HS)
            transform_cfg = TreeTransformParams(**(self.transform_params or {}))
            self.ruleset_ = estimator_to_scored_ruleset(
                estimator=self.estimator_,
                class_labels=self.classes_.tolist(),
                feature_names=self.feature_names_in_,
                params=transform_cfg,
            )

        # rulekit/exstracs/logicgp: Prediction immer ueber ScoredRuleSet routen
        self.is_ruleset_mode_ = backend_lower in (
            "rulekit",
            "exstracs",
            "logicgp",
            "pittsburgh",
            "michigan",
        )
        return self

    def _apply_exstracs_shrinking(self, ruleset: ScoredRuleSet, X: np.ndarray, y: np.ndarray) -> ScoredRuleSet:
        """Wende ExSTraCS Shrinking-Parameter an"""
        params = ExSTraCSPruningParams(**(self.exstracs_params or {}))
        
        # Für aggressive Pruning: Split Trainings-Daten
        X_train_split = X
        y_train_split = y
        X_val_split = None
        y_val_split = None
        
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

    def _prepare_X_for_prediction(self, X_valid: np.ndarray) -> np.ndarray:
        """Bereite X für die ScoredRuleSet-Prediction vor.
        
        LogicGP benötigt diskretisierte Daten, da die Atome im ScoredRuleSet
        auf Bin-Indizes basieren, nicht auf Rohwerten.
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

    def predict(self, X):
        check_is_fitted(self, "ruleset_")
        X_valid: np.ndarray = np.asarray(check_array(X, dtype=None))
        if self.n_features_in_ is not None and X_valid.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_valid.shape[1]} features, but {self.__class__.__name__} "
                f"is expecting {self.n_features_in_} features as input"
            )

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

