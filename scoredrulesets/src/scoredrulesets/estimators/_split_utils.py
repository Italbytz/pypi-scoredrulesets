"""
Gemeinsame Gini-basierte Split-Funktionen fuer regelbasierte Schaetzer.

Dieses Modul buendelt die Logik zum Finden von numerischen und
kategorischen Splits (bester Schwellenwert, Intervall-Splits,
Einzel-Kategorie-Splits und Gruppen-Splits).  Die Funktionen werden
von ``PittsburghRuleSetClassifier`` (und ggf. weiteren Schätzern) genutzt.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Gini-Impurity
# ---------------------------------------------------------------------------

def gini(counts: np.ndarray) -> float:
    """Gini-Impurity aus Klassen-Counts."""
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0
    probs = counts / total
    return float(1.0 - np.sum(probs ** 2))


# ---------------------------------------------------------------------------
# Score-Berechnung
# ---------------------------------------------------------------------------

def distribution_to_scores(counts: np.ndarray, aggregation: str) -> list[float]:
    """Konvertiert Klassen-Counts in Score-Vektoren.

    Bei ``aggregation="softmax_sum"`` werden Log-Wahrscheinlichkeiten
    zurueckgegeben, sonst Wahrscheinlichkeiten.
    """
    probs = counts / max(float(np.sum(counts)), 1.0)
    if aggregation == "softmax_sum":
        return np.log(np.maximum(probs, 1e-12)).tolist()
    return probs.tolist()


# ---------------------------------------------------------------------------
# Bester numerischer Split
# ---------------------------------------------------------------------------

def best_numeric_split(
    feature_values,
    y_idx: np.ndarray,
    n_classes: int,
    min_samples_leaf: int,
    max_thresholds_per_feature: int | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray, int, int] | None:
    """Findet den besten binaeren Gini-Split fuer ein numerisches Feature.

    Returns
    -------
    tuple or None
        ``(threshold, gain, left_counts, right_counts, left_coverage,
        right_coverage)`` oder ``None`` wenn kein Split moeglich ist.
    """
    values = np.asarray(feature_values)
    if not np.issubdtype(values.dtype, np.number):
        return None

    values = values.astype(float)
    unique = np.unique(values)
    if unique.size < 2:
        return None

    parent_counts = np.bincount(y_idx, minlength=n_classes).astype(float)
    parent_impurity = gini(parent_counts)

    thresholds = (unique[:-1] + unique[1:]) / 2.0
    if max_thresholds_per_feature is not None and len(thresholds) > max_thresholds_per_feature:
        idx = np.round(np.linspace(0, len(thresholds) - 1, max_thresholds_per_feature)).astype(int)
        thresholds = thresholds[idx]

    best = None
    for threshold in thresholds:
        left_mask = np.asarray(values <= threshold, dtype=bool)
        right_mask = np.asarray(~left_mask, dtype=bool)
        if left_mask.sum() < min_samples_leaf or right_mask.sum() < min_samples_leaf:
            continue

        left_counts = np.bincount(y_idx[left_mask], minlength=n_classes).astype(float)
        right_counts = np.bincount(y_idx[right_mask], minlength=n_classes).astype(float)
        left_weight = float(left_mask.mean())
        child_impurity = left_weight * gini(left_counts) + (1.0 - left_weight) * gini(right_counts)
        gain = parent_impurity - child_impurity

        candidate = (
            float(threshold),
            float(gain),
            left_counts,
            right_counts,
            int(left_mask.sum()),
            int(right_mask.sum()),
        )
        if best is None or gain > best[1]:
            best = candidate

    return best


# ---------------------------------------------------------------------------
# Kategorische Einzel-Splits
# ---------------------------------------------------------------------------

def categorical_splits(
    feature_values,
    y_idx: np.ndarray,
    n_classes: int,
    min_samples_leaf: int,
) -> list[tuple[float, object, np.ndarray, int]]:
    """Berechnet Gini-Gain fuer jeden einzelnen Kategoriewert.

    Returns
    -------
    list of (gain, category, match_counts, coverage)
    """
    values = np.asarray(feature_values, dtype=object)
    unique = np.unique(values)
    if unique.size < 2:
        return []

    parent_counts = np.bincount(y_idx, minlength=n_classes).astype(float)
    parent_impurity = gini(parent_counts)
    candidates: list[tuple[float, object, np.ndarray, int]] = []

    for category in unique.tolist():
        match_mask = np.asarray(values == category, dtype=bool)
        non_match_mask = np.asarray(~match_mask, dtype=bool)
        if match_mask.sum() < min_samples_leaf or non_match_mask.sum() < min_samples_leaf:
            continue

        match_counts = np.bincount(y_idx[match_mask], minlength=n_classes).astype(float)
        non_match_counts = np.bincount(y_idx[non_match_mask], minlength=n_classes).astype(float)
        match_weight = float(match_mask.mean())
        child_impurity = match_weight * gini(match_counts) + (1.0 - match_weight) * gini(non_match_counts)
        gain_val = parent_impurity - child_impurity

        candidates.append((float(gain_val), category, match_counts, int(match_mask.sum())))

    return candidates


# ---------------------------------------------------------------------------
# Numerische Intervall-Splits
# ---------------------------------------------------------------------------

def numeric_interval_splits(
    feature_values,
    y_idx: np.ndarray,
    n_classes: int,
    min_samples_leaf: int,
    max_thresholds_per_feature: int | None = None,
    max_results: int = 2,
) -> list[tuple[float, float, float, np.ndarray, int]]:
    """Findet die besten Intervall-Splits ``[low, high]`` fuer ein numerisches Feature.

    Returns
    -------
    list of (gain, low, high, in_counts, coverage)
        Sortiert nach absteigendem Gain, maximal *max_results* Eintraege.
    """
    values = np.asarray(feature_values)
    if not np.issubdtype(values.dtype, np.number):
        return []
    values = values.astype(float)
    if np.unique(values).size < 3:
        return []

    q_points = np.unique(np.quantile(values, [0.1, 0.25, 0.4, 0.6, 0.75, 0.9]))
    if q_points.size < 2:
        return []
    if max_thresholds_per_feature is not None and len(q_points) > max_thresholds_per_feature:
        idx = np.round(np.linspace(0, len(q_points) - 1, max_thresholds_per_feature)).astype(int)
        q_points = q_points[idx]

    parent_counts = np.bincount(y_idx, minlength=n_classes).astype(float)
    parent_impurity = gini(parent_counts)
    candidates: list[tuple[float, float, float, np.ndarray, int]] = []

    for i in range(len(q_points) - 1):
        for j in range(i + 1, len(q_points)):
            low = float(q_points[i])
            high = float(q_points[j])
            if not low < high:
                continue
            in_mask = np.asarray((values >= low) & (values <= high), dtype=bool)
            out_mask = np.asarray(~in_mask, dtype=bool)
            if in_mask.sum() < min_samples_leaf or out_mask.sum() < min_samples_leaf:
                continue
            in_counts = np.bincount(y_idx[in_mask], minlength=n_classes).astype(float)
            out_counts = np.bincount(y_idx[out_mask], minlength=n_classes).astype(float)
            in_weight = float(in_mask.mean())
            child_impurity = in_weight * gini(in_counts) + (1.0 - in_weight) * gini(out_counts)
            gain_val = parent_impurity - child_impurity
            candidates.append((float(gain_val), low, high, in_counts, int(in_mask.sum())))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[:max_results]


# ---------------------------------------------------------------------------
# Kategorische Gruppen-Splits
# ---------------------------------------------------------------------------

def categorical_group_splits(
    feature_values,
    y_idx: np.ndarray,
    n_classes: int,
    min_samples_leaf: int,
    max_results: int = 2,
) -> list[tuple[float, list[object], np.ndarray, int]]:
    """Findet die besten Kategorie-Gruppen-Splits (2–3 Kategorien zusammen).

    Returns
    -------
    list of (gain, group_values, in_counts, coverage)
        Sortiert nach absteigendem Gain, maximal *max_results* Eintraege.
    """
    values = np.asarray(feature_values, dtype=object)
    unique = np.unique(values)
    if unique.size < 3:
        return []

    parent_counts = np.bincount(y_idx, minlength=n_classes).astype(float)
    parent_impurity = gini(parent_counts)
    candidates: list[tuple[float, list[object], np.ndarray, int]] = []
    seen_groups: set[tuple[str, ...]] = set()

    for class_idx in range(n_classes):
        class_mask = y_idx == class_idx
        if class_mask.sum() == 0:
            continue
        class_values = values[class_mask]
        cats, counts = np.unique(class_values, return_counts=True)
        order = np.argsort(-counts)
        ranked_cats = [cats[i] for i in order]

        for group_size in range(2, min(3, len(ranked_cats)) + 1):
            group = ranked_cats[:group_size]
            key = tuple(sorted(str(v) for v in group))
            if key in seen_groups:
                continue
            seen_groups.add(key)
            in_mask = np.asarray(np.isin(values, group), dtype=bool)
            out_mask = np.asarray(~in_mask, dtype=bool)
            if in_mask.sum() < min_samples_leaf or out_mask.sum() < min_samples_leaf:
                continue
            in_counts = np.bincount(y_idx[in_mask], minlength=n_classes).astype(float)
            out_counts = np.bincount(y_idx[out_mask], minlength=n_classes).astype(float)
            in_weight = float(in_mask.mean())
            child_impurity = in_weight * gini(in_counts) + (1.0 - in_weight) * gini(out_counts)
            gain_val = parent_impurity - child_impurity
            candidates.append((float(gain_val), list(group), in_counts, int(in_mask.sum())))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[:max_results]

