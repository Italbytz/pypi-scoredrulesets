from __future__ import annotations

from typing import Any

import numpy as np

from .schema import Atom, Rule, ScoredRuleSet


def _resolve_feature_index(atom: Atom, feature_names: list[str]) -> int:
    if isinstance(atom.feature, int):
        return atom.feature
    if atom.feature in feature_names:
        return feature_names.index(atom.feature)
    raise KeyError(f"Feature '{atom.feature}' not found in feature_names")


def _atom_matches(value: Any, atom: Atom) -> bool:
    op = atom.op
    ref = atom.value
    if op == "==":
        return value == ref
    if op == "!=":
        return value != ref
    if op == "<=":
        return value <= ref
    if op == "<":
        return value < ref
    if op == ">":
        return value > ref
    if op == ">=":
        return value >= ref
    if op == "in":
        return value in ref
    if op == "not_in":
        return value not in ref
    if op == "between":
        low, high = ref
        return low <= value <= high
    raise ValueError(f"Unsupported atom operator: {op}")


def _rule_fires(row: np.ndarray, rule: Rule, feature_names: list[str]) -> bool:
    for atom in rule.atoms:
        index = _resolve_feature_index(atom, feature_names)
        if not _atom_matches(row[index], atom):
            return False
    return True


def decision_function(ruleset: ScoredRuleSet, X: np.ndarray) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    n_classes = len(ruleset.class_labels)
    scores = np.zeros((X.shape[0], n_classes), dtype=float)
    feature_names = ruleset.feature_names or [f"f{i}" for i in range(X.shape[1])]

    for i, row in enumerate(X):
        for rule in ruleset.rules:
            if _rule_fires(row, rule, feature_names):
                scores[i] += np.asarray(rule.scores, dtype=float)
    return scores


def predict_proba(ruleset: ScoredRuleSet, X: np.ndarray) -> np.ndarray:
    scores = decision_function(ruleset, X)
    agg = ruleset.aggregation.type
    if agg == "argmax_sum":
        proba = np.zeros_like(scores)
        best_idx = np.argmax(scores, axis=1)
        proba[np.arange(scores.shape[0]), best_idx] = 1.0
        return proba
    if agg == "softmax_sum":
        temperature = max(float(ruleset.aggregation.temperature), 1e-12)
        scaled = scores / temperature
        shifted = scaled - np.max(scaled, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=1, keepdims=True)
    raise ValueError(f"Unsupported aggregation type: {agg}")


def predict(ruleset: ScoredRuleSet, X: np.ndarray) -> np.ndarray:
    proba = predict_proba(ruleset, X)
    indices = np.argmax(proba, axis=1)
    labels = np.asarray(ruleset.class_labels, dtype=object)
    return labels[indices]

