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
        # ExSTraCS may provide intervals as dicts with 'lower' and 'upper'.
        if isinstance(ref, dict) and "lower" in ref and "upper" in ref:
            lower = ref["lower"]
            upper = ref["upper"]
            # Ensure correct ordering in case lower > upper.
            if lower > upper:
                lower, upper = upper, lower
            return lower <= value <= upper
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


def _ensure_task_type(ruleset: ScoredRuleSet, expected: str) -> None:
    if ruleset.task_type != expected:
        raise ValueError(
            f"Expected task_type='{expected}', got '{ruleset.task_type}'"
        )


def decision_function(ruleset: ScoredRuleSet, X: np.ndarray, debug: bool = False) -> np.ndarray:
    _ensure_task_type(ruleset, "classification")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    n_classes = len(ruleset.class_labels)
    scores = np.zeros((X.shape[0], n_classes), dtype=float)
    feature_names = ruleset.feature_names or [f"f{i}" for i in range(X.shape[1])]

    # Identify default rule(s) (rules without conditions).
    default_scores = np.zeros(n_classes, dtype=float)
    has_default = False
    non_default_rules = []
    for rule in ruleset.rules:
        if not getattr(rule, 'atoms', None):
            default_scores += np.asarray(rule.scores, dtype=float)
            has_default = True
        else:
            non_default_rules.append(rule)

    for i, row in enumerate(X):
        fired_rules = []
        # Check whether at least one non-default rule fires.
        non_default_fired = []
        for rule in non_default_rules:
            if _rule_fires(row, rule, feature_names):
                scores[i] += np.asarray(rule.scores, dtype=float)
                fired_rules.append(rule)
                non_default_fired.append(rule)
        # If no non-default rule fires, apply default scores if available.
        if not non_default_fired and has_default:
            scores[i] += default_scores
            fired_rules.append(("default_combined", default_scores.tolist()))
        if debug:
            debug_rules = []
            for r in fired_rules:
                if isinstance(r, tuple):
                    debug_rules.append(r)
                else:
                    debug_rules.append((getattr(r, 'rule_id', None), getattr(r, 'scores', None)))
            print(f"[DEBUG decision_function] Sample {i}: fired_rules={debug_rules}")
            print(f"[DEBUG decision_function] Sample {i}: scores={scores[i]}")
    return scores


def decision_function_regression(ruleset: ScoredRuleSet, X: np.ndarray, debug: bool = False) -> np.ndarray:
    _ensure_task_type(ruleset, "regression")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    agg = ruleset.aggregation.type
    d = 1
    if ruleset.rules and agg != "takagi_sugeno":
        d = len(ruleset.rules[0].scores)

    if d > 1:
        predictions = np.zeros((X.shape[0], d), dtype=float)
        default_value = np.zeros(d, dtype=float)
    else:
        predictions = np.zeros(X.shape[0], dtype=float)
        default_value = 0.0

    feature_names = ruleset.feature_names or [f"f{i}" for i in range(X.shape[1])]
    has_default = False
    non_default_rules = []
    
    for rule in ruleset.rules:
        if not getattr(rule, "atoms", None):
            if d > 1:
                default_value += np.asarray(rule.scores, dtype=float)
            else:
                default_value += float(rule.scores[0])
            has_default = True
        else:
            non_default_rules.append(rule)

    for i, row in enumerate(X):
        if agg == "takagi_sugeno":
            # TS is strictly 1D for now
            active_ts_values = []
            for rule in non_default_rules:
                if _rule_fires(row, rule, feature_names):
                    beta = rule.scores[:-1]
                    beta0 = rule.scores[-1]
                    active_ts_values.append(np.dot(row, beta) + beta0)
            if active_ts_values:
                predictions[i] = float(np.mean(active_ts_values))
            elif has_default:
                default_beta = ruleset.rules[-1].scores[:-1]
                default_beta0 = ruleset.rules[-1].scores[-1]
                predictions[i] = float(np.dot(row, default_beta) + default_beta0)
            else:
                predictions[i] = 0.0
            continue

        active_values = []
        for rule in non_default_rules:
            if _rule_fires(row, rule, feature_names):
                if d > 1:
                    active_values.append(np.asarray(rule.scores, dtype=float))
                else:
                    active_values.append(float(rule.scores[0]))

        if agg == "weighted_sum":
            if active_values:
                predictions[i] = np.sum(active_values, axis=0)
            else:
                predictions[i] = default_value if has_default else (np.zeros(d) if d > 1 else 0.0)
        elif agg == "mean_active":
            if active_values:
                predictions[i] = np.mean(active_values, axis=0)
            else:
                predictions[i] = default_value if has_default else (np.zeros(d) if d > 1 else 0.0)
        elif agg == "default_plus_sum":
            if active_values:
                predictions[i] = default_value + np.sum(active_values, axis=0)
            else:
                predictions[i] = default_value
        else:
            raise ValueError(f"Unsupported regression aggregation type: {agg}")

        if debug:
            print(f"[DEBUG regression] Sample {i}: prediction={predictions[i]}")

    return predictions


def predict_proba(ruleset: ScoredRuleSet, X: np.ndarray, debug: bool = False) -> np.ndarray:
    _ensure_task_type(ruleset, "classification")
    scores = decision_function(ruleset, X, debug=debug)
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


def predict(ruleset: ScoredRuleSet, X: np.ndarray, debug: bool = False) -> np.ndarray:
    _ensure_task_type(ruleset, "classification")
    proba = predict_proba(ruleset, X, debug=debug)
    indices = np.argmax(proba, axis=1)
    # Infer a sensible dtype: let numpy decide (int, float, str, …)
    # instead of forcing dtype=object which confuses sklearn metrics.
    labels = np.asarray(ruleset.class_labels)
    return labels[indices]


def predict_regression(ruleset: ScoredRuleSet, X: np.ndarray, debug: bool = False) -> np.ndarray:
    return decision_function_regression(ruleset, X, debug=debug)


def consolidate_default_rules(ruleset: ScoredRuleSet) -> ScoredRuleSet:
    """
    Ensure that the ruleset contains at most one default rule (atoms=[]).
    If multiple defaults exist, their scores are summed into a single rule.
    All non-default rules remain unchanged.
    """
    default_rules = [r for r in ruleset.rules if not getattr(r, 'atoms', None)]
    non_default_rules = [r for r in ruleset.rules if getattr(r, 'atoms', None)]
    if not default_rules:
        return ruleset
    # Sum scores of all default rules.
    summed_scores = np.sum([np.asarray(r.scores, dtype=float) for r in default_rules], axis=0)
    # Merge metadata.
    merged_metadata = {"merged_default_rules": len(default_rules)}
    merged_metadata.update(getattr(default_rules[0], 'metadata', {}))
    merged_default_rule = Rule(
        atoms=[],
        scores=summed_scores.tolist(),
        rule_id="merged_default",
        metadata=merged_metadata,
    )
    new_rules = non_default_rules + [merged_default_rule]
    return ScoredRuleSet(
        class_labels=ruleset.class_labels,
        task_type=ruleset.task_type,
        feature_names=ruleset.feature_names,
        rules=new_rules,
        aggregation=ruleset.aggregation,
        metadata=ruleset.metadata,
    )
