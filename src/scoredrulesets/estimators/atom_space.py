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
NativeAtomSpaceStrategy = Literal[
    "hybrid",
    "numeric_only",
    "categorical_low_cardinality_only",
    "genotype_aware",
]
ContinuousThresholdStrategy = Literal["quantile_midpoint", "supervised_mdl"]
NLNThresholdStrategy = Literal["quantile_midpoint", "quantile_only", "midpoint_only"]
RulePLCSFeatureTypingStrategy = Literal[
    "auto_low_cardinality",
    "all_numeric",
    "all_integer_categorical",
    "genotype_aware",
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


def genotype_levels(
    col: np.ndarray,
    max_levels: int = 10,
) -> list[float] | None:
    """Return sorted integer levels if ``col`` is an ordinal count/genotype feature.

    A feature qualifies when it is numeric, non-negative, integer-valued, and has
    between two and ``max_levels`` distinct levels. The canonical case is the SNP
    genotype encoding with levels ``{0, 1, 2}``; more generally this covers small
    ordinal count features. Returns ``None`` when the feature does not qualify.
    """
    arr = np.asarray(col)
    try:
        float_col = arr.astype(float)
    except (ValueError, TypeError):
        return None
    if float_col.size == 0 or not np.all(np.isfinite(float_col)):
        return None
    if np.any(float_col < 0):
        return None
    if not np.all(float_col == np.round(float_col)):
        return None
    levels = np.unique(float_col)
    if levels.size < 2 or levels.size > max_levels:
        return None
    return levels.tolist()


def genotype_feature_spec(fi: int, levels: list[float]) -> dict[str, Any]:
    """Build a compact genetic-model atom spec for an ordinal genotype feature.

    For sorted integer ``levels`` the thresholds are the half-integer midpoints
    between consecutive levels. For the canonical genotype encoding ``{0, 1, 2}``
    this yields thresholds ``[0.5, 1.5]``, so the expanded atom pool contains
    ``> 0.5`` (dominant model, carrier of the minor allele, ``x >= 1``) and
    ``> 1.5`` (recessive model, homozygous minor, ``x == 2``) together with their
    complements, capturing the additive structure. No interval or subset atoms
    are emitted, keeping the per-feature pool minimal.
    """
    thresholds = [
        (levels[i] + levels[i + 1]) / 2.0 for i in range(len(levels) - 1)
    ]
    return {
        "idx": fi,
        "kind": "num",
        "thresholds": thresholds,
        "intervals": [],
    }


def _entropy(class_counts: np.ndarray) -> float:
    """Shannon entropy (in bits) of a class-count vector."""
    total = float(class_counts.sum())
    if total <= 0.0:
        return 0.0
    p = class_counts[class_counts > 0] / total
    return float(-np.sum(p * np.log2(p)))


def fayyad_irani_cut_points(
    x: np.ndarray,
    y: np.ndarray,
    max_cut_points: int | None = None,
) -> list[float]:
    """Supervised multi-interval discretization cut points (Fayyad & Irani, 1993).

    Recursively splits the range of a continuous feature at the boundary that
    minimizes class-information entropy, accepting a split only when it passes
    the MDLP (minimum description length principle) stopping criterion. Returns
    the sorted accepted cut points (empty when the feature is uninformative).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    if x.size < 2:
        return []
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    _, y_idx = np.unique(y[order], return_inverse=True)
    n_classes = int(y_idx.max()) + 1 if y_idx.size else 0
    if n_classes < 2:
        return []

    cut_points: list[float] = []

    def recurse(lo: int, hi: int) -> None:
        n = hi - lo
        if n < 2:
            return
        seg_x = xs[lo:hi]
        seg_y = y_idx[lo:hi]
        total_counts = np.bincount(seg_y, minlength=n_classes).astype(float)
        k = int(np.count_nonzero(total_counts))
        if k < 2:
            return
        ent_s = _entropy(total_counts)

        best_gain = -1.0
        best_pos = -1
        best_left: np.ndarray | None = None
        best_right: np.ndarray | None = None
        left_counts = np.zeros(n_classes, dtype=float)
        for i in range(n - 1):
            left_counts[seg_y[i]] += 1.0
            if seg_x[i] == seg_x[i + 1]:
                continue
            right_counts = total_counts - left_counts
            n_left = i + 1
            n_right = n - n_left
            e = (n_left / n) * _entropy(left_counts) + (
                n_right / n
            ) * _entropy(right_counts)
            gain = ent_s - e
            if gain > best_gain:
                best_gain = gain
                best_pos = i
                best_left = left_counts.copy()
                best_right = right_counts.copy()

        if best_pos < 0 or best_left is None or best_right is None:
            return

        k1 = int(np.count_nonzero(best_left))
        k2 = int(np.count_nonzero(best_right))
        ent_s1 = _entropy(best_left)
        ent_s2 = _entropy(best_right)
        delta = np.log2(3.0 ** k - 2.0) - (
            k * ent_s - k1 * ent_s1 - k2 * ent_s2
        )
        threshold = (np.log2(n - 1) + delta) / n
        if best_gain <= threshold:
            return

        cut = (seg_x[best_pos] + seg_x[best_pos + 1]) / 2.0
        cut_points.append(float(cut))
        split = lo + best_pos + 1
        recurse(lo, split)
        recurse(split, hi)

    recurse(0, xs.size)
    cuts = sorted(set(cut_points))
    if max_cut_points is not None and len(cuts) > max_cut_points:
        idx = np.round(np.linspace(0, len(cuts) - 1, max_cut_points)).astype(int)
        cuts = [cuts[i] for i in sorted(set(idx.tolist()))]
    return cuts


def build_native_feature_specs(
    X: np.ndarray,
    max_thresholds: int | None = None,
    low_cardinality_threshold: int = 10,
    strategy: NativeAtomSpaceStrategy = "hybrid",
    y: np.ndarray | None = None,
    continuous_threshold_strategy: ContinuousThresholdStrategy = "quantile_midpoint",
) -> list[dict[str, Any]]:
    """Build per-feature specs for native atom generation."""
    if strategy not in (
        "hybrid",
        "numeric_only",
        "categorical_low_cardinality_only",
        "genotype_aware",
    ):
        raise ValueError(
            "Unknown native atom-space strategy "
            f"'{strategy}'. Choose 'hybrid', 'numeric_only', "
            "'categorical_low_cardinality_only', or 'genotype_aware'."
        )
    if continuous_threshold_strategy not in ("quantile_midpoint", "supervised_mdl"):
        raise ValueError(
            "Unknown continuous threshold strategy "
            f"'{continuous_threshold_strategy}'. Choose 'quantile_midpoint' or "
            "'supervised_mdl'."
        )
    supervised = continuous_threshold_strategy == "supervised_mdl" and y is not None

    specs: list[dict[str, Any]] = []
    for fi in range(X.shape[1]):
        col = X[:, fi]
        arr = np.asarray(col)
        if strategy == "genotype_aware":
            levels = genotype_levels(col, max_levels=low_cardinality_threshold)
            if levels is not None:
                specs.append(genotype_feature_spec(fi, levels))
                continue
        if np.issubdtype(arr.dtype, np.number):
            float_full = arr.astype(float)
            vals = np.unique(float_full)
            if supervised and vals.size >= 2:
                thr = fayyad_irani_cut_points(
                    float_full, y, max_cut_points=max_thresholds
                )
            elif vals.size >= 2:
                if vals.size <= 20:
                    thr = ((vals[:-1] + vals[1:]) / 2.0).tolist()
                else:
                    q = np.unique(np.quantile(vals, np.linspace(0.05, 0.95, 10)))
                    thr = q.astype(float).tolist()
            else:
                thr = []
            if not supervised and max_thresholds and len(thr) > max_thresholds:
                idx = np.round(np.linspace(0, len(thr) - 1, max_thresholds)).astype(int)
                thr = [thr[i] for i in idx]

            intervals: list[tuple[float, float]] = []
            if supervised and len(thr) >= 1:
                edges = sorted(
                    set(
                        [float(vals.min())]
                        + [float(t) for t in thr]
                        + [float(vals.max())]
                    )
                )
                for i in range(len(edges) - 1):
                    if edges[i] < edges[i + 1]:
                        intervals.append((edges[i], edges[i + 1]))
            elif not supervised and vals.size >= 3:
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


def build_ruleplcs_feature_info(
    X: np.ndarray,
    low_cardinality_threshold: int,
    strategy: RulePLCSFeatureTypingStrategy = "auto_low_cardinality",
    deadline: float | None = None,
) -> list[dict[str, Any]]:
    """Build RulePLCS feature metadata with low-cardinality category heuristic.

    When ``deadline`` (an absolute monotonic timestamp) is supplied and it is
    reached before all features have been analysed, a
    :class:`FitBudgetExceededError` is raised: the feature-space setup could not
    finish within the fit-time budget, so no viable model can be built.
    """
    from ._time_budget import FitBudgetExceededError, deadline_reached

    if strategy not in (
        "auto_low_cardinality",
        "all_numeric",
        "all_integer_categorical",
        "genotype_aware",
    ):
        raise ValueError(
            "Unknown RulePLCS feature typing strategy "
            f"'{strategy}'. Choose 'auto_low_cardinality', 'all_numeric', "
            "'all_integer_categorical', or 'genotype_aware'."
        )

    info: list[dict[str, Any]] = []
    for fi in range(X.shape[1]):
        if deadline is not None and (fi & 0xFF) == 0 and deadline_reached(deadline):
            raise FitBudgetExceededError(
                "max_fit_seconds exhausted while building the RulePLCS feature "
                f"space (analysed {fi} of {X.shape[1]} features); the estimator "
                "is not viable within this budget."
            )
        col = X[:, fi]
        unique_vals = np.unique(col)
        n_unique = len(unique_vals)
        if strategy == "all_numeric":
            is_cat = False
        elif strategy == "all_integer_categorical":
            is_cat = bool(np.all(col == np.round(col)))
        elif strategy == "genotype_aware":
            is_cat = genotype_levels(col, max_levels=low_cardinality_threshold) is not None
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
