"""Shared pre-fit atom-space helpers.

This module centralizes feature-to-atom preparation that is used by multiple
atom-based estimators. The goal is to keep one architecture with pluggable
strategies while preserving estimator-specific behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer


FeatureKind = Literal["num", "cat", "both"]
LogicGPEncodingStrategy = Literal["auto_low_cardinality", "force_numeric_bins"]
NativeAtomSpaceStrategy = Literal["hybrid", "numeric_only", "categorical_low_cardinality_only"]
NLNThresholdStrategy = Literal["quantile_midpoint", "quantile_only", "midpoint_only"]
RuleLCSFeatureTypingStrategy = Literal[
    "auto_low_cardinality",
    "all_numeric",
    "all_integer_categorical",
]


@dataclass(frozen=True)
class AtomFeatureSpec:
    """Feature description used during atom generation."""

    idx: int
    kind: FeatureKind
    thresholds: list[float] | None = None
    intervals: list[tuple[float, float]] | None = None
    categories: list[object] | None = None


class FeatureEncoder(Protocol):
    """Pluggable pre-fit feature encoder for atom-based estimators."""

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        ...

    def transform(self, X: np.ndarray) -> np.ndarray:
        ...


class LogicGPDiscretizingEncoder:
    """LogicGP-compatible discretization encoder."""

    def __init__(self, n_bins: int = 5):
        self.n_bins = n_bins
        self._binners: list[Any] | None = None
        self._cat_masks: np.ndarray | None = None

    @property
    def fitted_binners_(self) -> list[Any] | None:
        return self._binners

    @property
    def cat_masks_(self) -> np.ndarray | None:
        return self._cat_masks

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X_disc, self._binners, self._cat_masks = discretize_logicgp_features(
            X,
            n_bins=self.n_bins,
            fitted_binners=None,
            cat_masks=None,
        )
        return X_disc

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._binners is None or self._cat_masks is None:
            raise ValueError("Encoder must be fitted before calling transform().")
        X_disc, _, _ = discretize_logicgp_features(
            X,
            n_bins=self.n_bins,
            fitted_binners=self._binners,
            cat_masks=self._cat_masks,
        )
        return X_disc


def discretize_logicgp_features(
    X: np.ndarray,
    n_bins: int = 5,
    fitted_binners: list[Any] | None = None,
    cat_masks: np.ndarray | None = None,
    strategy: LogicGPEncodingStrategy = "auto_low_cardinality",
) -> tuple[np.ndarray, list[Any], np.ndarray]:
    """Discretize continuous features into bin indices.

    Categorical features (non-numeric or <= n_bins unique values) stay
    unchanged.
    """
    if strategy not in ("auto_low_cardinality", "force_numeric_bins"):
        raise ValueError(
            "Unknown LogicGP encoding strategy "
            f"'{strategy}'. Choose 'auto_low_cardinality' or 'force_numeric_bins'."
        )

    n_samples, n_features = X.shape
    X_disc = np.empty((n_samples, n_features), dtype=object)

    is_fit = fitted_binners is not None

    if not is_fit:
        fitted_binners = []
        cat_masks_list = []
    else:
        cat_masks_list = None

    for f in range(n_features):
        col = X[:, f]
        arr = np.asarray(col, dtype=object)

        is_numeric = False
        float_col = None
        try:
            float_col = arr.astype(float)
            is_numeric = True
        except (ValueError, TypeError):
            pass

        if is_numeric and float_col is not None:
            unique_vals = np.unique(float_col)
            n_unique = len(unique_vals)
            if strategy == "force_numeric_bins":
                is_cat = False
            else:
                is_cat = n_unique <= n_bins
        else:
            is_cat = True
            float_col = None
            n_unique = len(np.unique(arr))

        if not is_fit:
            cat_masks_list.append(is_cat)

        if is_cat or not is_numeric:
            X_disc[:, f] = arr
            if not is_fit:
                fitted_binners.append(None)
        else:
            actual_bins = min(n_bins, n_unique)
            if not is_fit:
                binner = KBinsDiscretizer(
                    n_bins=actual_bins,
                    encode="ordinal",
                    strategy="quantile",
                    subsample=None,
                )
                binner.fit(float_col.reshape(-1, 1))
                fitted_binners.append(binner)
            else:
                binner = fitted_binners[f]

            if binner is not None:
                binned = binner.transform(float_col.reshape(-1, 1)).ravel().astype(int)
                X_disc[:, f] = binned
            else:
                X_disc[:, f] = arr

    if not is_fit:
        cat_masks_arr = np.array(cat_masks_list, dtype=bool)
    else:
        if cat_masks is None:
            raise ValueError("cat_masks must be provided with fitted_binners.")
        cat_masks_arr = cat_masks

    return X_disc, fitted_binners, cat_masks_arr


def build_native_feature_specs(
    X: np.ndarray,
    max_thresholds: int | None = None,
    low_cardinality_threshold: int = 10,
    strategy: NativeAtomSpaceStrategy = "hybrid",
) -> list[dict[str, Any]]:
    """Build per-feature specs for native atom generation."""
    if strategy not in ("hybrid", "numeric_only", "categorical_low_cardinality_only"):
        raise ValueError(
            "Unknown native atom-space strategy "
            f"'{strategy}'. Choose 'hybrid', 'numeric_only', or "
            "'categorical_low_cardinality_only'."
        )

    specs: list[dict[str, Any]] = []
    for fi in range(X.shape[1]):
        col = X[:, fi]
        arr = np.asarray(col)
        if np.issubdtype(arr.dtype, np.number):
            vals = np.unique(arr.astype(float))
            if vals.size >= 2:
                if vals.size <= 20:
                    thr = ((vals[:-1] + vals[1:]) / 2.0).tolist()
                else:
                    q = np.unique(np.quantile(vals, np.linspace(0.05, 0.95, 10)))
                    thr = q.astype(float).tolist()
            else:
                thr = []
            if max_thresholds and len(thr) > max_thresholds:
                idx = np.round(np.linspace(0, len(thr) - 1, max_thresholds)).astype(int)
                thr = [thr[i] for i in idx]

            intervals: list[tuple[float, float]] = []
            if vals.size >= 3:
                qp = np.unique(np.quantile(vals, [0.15, 0.35, 0.5, 0.65, 0.85]))
                for i in range(len(qp) - 1):
                    if qp[i] < qp[i + 1]:
                        intervals.append((float(qp[i]), float(qp[i + 1])))

            if strategy == "numeric_only":
                specs.append(
                    {
                        "idx": fi,
                        "kind": "num",
                        "thresholds": thr,
                        "intervals": intervals,
                    }
                )
            elif vals.size <= low_cardinality_threshold and strategy == "hybrid":
                specs.append(
                    {
                        "idx": fi,
                        "kind": "both",
                        "thresholds": thr,
                        "intervals": intervals,
                        "categories": vals.tolist(),
                    }
                )
            elif vals.size <= low_cardinality_threshold and strategy == "categorical_low_cardinality_only":
                specs.append(
                    {
                        "idx": fi,
                        "kind": "cat",
                        "categories": vals.tolist(),
                    }
                )
            else:
                specs.append(
                    {
                        "idx": fi,
                        "kind": "num",
                        "thresholds": thr,
                        "intervals": intervals,
                    }
                )
        else:
            cats = np.unique(np.asarray(col, dtype=object)).tolist()
            specs.append({"idx": fi, "kind": "cat", "categories": cats})
    return specs


def compute_nln_thresholds(
    X: np.ndarray,
    n_bins: int,
    max_thresholds_per_feature: int | None = None,
    strategy: NLNThresholdStrategy = "quantile_midpoint",
) -> list[np.ndarray]:
    """Compute RuleNLN-style quantile and midpoint thresholds per feature."""
    if strategy not in ("quantile_midpoint", "quantile_only", "midpoint_only"):
        raise ValueError(
            "Unknown NLN threshold strategy "
            f"'{strategy}'. Choose 'quantile_midpoint', 'quantile_only', or 'midpoint_only'."
        )

    thresholds: list[np.ndarray] = []
    quantiles = np.linspace(0, 1, n_bins + 2)[1:-1]

    for j in range(X.shape[1]):
        col = X[:, j]
        uniq = np.unique(col)

        use_midpoints = strategy == "midpoint_only" or (
            strategy == "quantile_midpoint" and len(uniq) <= max(n_bins, 2)
        )

        if use_midpoints:
            if len(uniq) <= 1:
                thr = np.array([float(uniq[0])]) if len(uniq) == 1 else np.array([0.0])
            else:
                thr = (uniq[:-1] + uniq[1:]) / 2.0
        else:
            thr = np.unique(np.quantile(col, quantiles))
            if len(thr) == 0:
                thr = np.array([float(col[0])]) if len(col) > 0 else np.array([0.0])

        col_min, col_max = float(col.min()), float(col.max())
        thr = np.array([t for t in thr if col_min <= t < col_max], dtype=float)
        if len(thr) == 0:
            thr = np.array([(col_min + col_max) / 2.0])

        if max_thresholds_per_feature is not None and len(thr) > max_thresholds_per_feature:
            idx = np.round(
                np.linspace(0, len(thr) - 1, max_thresholds_per_feature)
            ).astype(int)
            thr = thr[idx]

        thresholds.append(thr)

    return thresholds


def binarize_with_thresholds(X: np.ndarray, thresholds: list[np.ndarray]) -> np.ndarray:
    """Build proposition matrix with <= and > atoms for each threshold."""
    parts: list[np.ndarray] = []
    for j, thr in enumerate(thresholds):
        col = X[:, j : j + 1]
        leq = (col <= thr[np.newaxis, :]).astype(float)
        gt = (col > thr[np.newaxis, :]).astype(float)
        parts.append(leq)
        parts.append(gt)
    return np.hstack(parts)


def build_rulelcs_feature_info(
    X: np.ndarray,
    low_cardinality_threshold: int,
    strategy: RuleLCSFeatureTypingStrategy = "auto_low_cardinality",
) -> list[dict[str, Any]]:
    """Build RuleLCS feature metadata with low-cardinality category heuristic."""
    if strategy not in (
        "auto_low_cardinality",
        "all_numeric",
        "all_integer_categorical",
    ):
        raise ValueError(
            "Unknown RuleLCS feature typing strategy "
            f"'{strategy}'. Choose 'auto_low_cardinality', 'all_numeric', "
            "or 'all_integer_categorical'."
        )

    info: list[dict[str, Any]] = []
    for fi in range(X.shape[1]):
        col = X[:, fi]
        unique_vals = np.unique(col)
        n_unique = len(unique_vals)
        if strategy == "all_numeric":
            is_cat = False
        elif strategy == "all_integer_categorical":
            is_cat = bool(np.all(col == np.round(col)))
        else:
            is_cat = (
                n_unique <= low_cardinality_threshold and np.all(col == np.round(col))
            )
        if is_cat:
            info.append(
                {
                    "numeric": False,
                    "values": set(int(v) for v in unique_vals),
                    "min": float(col.min()),
                    "max": float(col.max()),
                }
            )
        else:
            info.append(
                {
                    "numeric": True,
                    "values": set(),
                    "min": float(col.min()),
                    "max": float(col.max()),
                }
            )
    return info
