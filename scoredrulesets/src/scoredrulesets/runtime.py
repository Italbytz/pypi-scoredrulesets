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


def decision_function(ruleset: ScoredRuleSet, X: np.ndarray, debug: bool = False) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    n_classes = len(ruleset.class_labels)
    scores = np.zeros((X.shape[0], n_classes), dtype=float)
    feature_names = ruleset.feature_names or [f"f{i}" for i in range(X.shape[1])]

    # Identifiziere Default-Regel (Regel ohne Bedingungen)
    default_rule = None
    non_default_rules = []
    for rule in ruleset.rules:
        if not getattr(rule, 'atoms', None):
            default_rule = rule
        else:
            non_default_rules.append(rule)

    for i, row in enumerate(X):
        fired_rules = []
        # Prüfe, ob mindestens eine Nicht-Default-Regel feuert
        non_default_fired = []
        for rule in non_default_rules:
            if _rule_fires(row, rule, feature_names):
                scores[i] += np.asarray(rule.scores, dtype=float)
                fired_rules.append(rule)
                non_default_fired.append(rule)
        # Falls keine Nicht-Default-Regel feuert, feuert die Default-Regel (falls vorhanden)
        if not non_default_fired and default_rule is not None:
            scores[i] += np.asarray(default_rule.scores, dtype=float)
            fired_rules.append(default_rule)
        if debug:
            print(f"[DEBUG decision_function] Sample {i}: fired_rules={[ (getattr(r, 'rule_id', None), getattr(r, 'scores', None)) for r in fired_rules ]}")
            print(f"[DEBUG decision_function] Sample {i}: scores={scores[i]}")
    return scores


def predict_proba(ruleset: ScoredRuleSet, X: np.ndarray, debug: bool = False) -> np.ndarray:
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
    proba = predict_proba(ruleset, X, debug=debug)
    indices = np.argmax(proba, axis=1)
    labels = np.asarray(ruleset.class_labels, dtype=object)
    return labels[indices]


def consolidate_default_rules(ruleset: ScoredRuleSet) -> ScoredRuleSet:
    """
    Sorgt dafür, dass im Ruleset maximal eine Default-Regel (atoms=[]) enthalten ist.
    Falls mehrere vorhanden sind, werden deren Scores aufsummiert und zu einer Regel zusammengefasst.
    Alle anderen Regeln bleiben unverändert.
    """
    default_rules = [r for r in ruleset.rules if not getattr(r, 'atoms', None)]
    non_default_rules = [r for r in ruleset.rules if getattr(r, 'atoms', None)]
    if not default_rules:
        return ruleset
    # Scores aufsummieren
    summed_scores = np.sum([np.asarray(r.scores, dtype=float) for r in default_rules], axis=0)
    # Metadaten zusammenfassen
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
        feature_names=ruleset.feature_names,
        rules=new_rules,
        aggregation=ruleset.aggregation,
        metadata=ruleset.metadata,
    )
