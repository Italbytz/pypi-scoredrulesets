from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..schema import AggregationSpec, Atom, Rule, ScoredRuleSet


# Standard-Kandidaten fuer die automatische Lambda-Suche beim Atom-Pruning.
_DEFAULT_LAMBDA_CANDIDATES: list[float] = [1.1, 1.5, 2.0, 3.0, 5.0, 10.0]


@dataclass
class TreeTransformParams:
    depth_decay_lambda: float = 2.0
    include_default_rule: bool = False
    default_rule_strength: float = 0.0
    aggressive_prune: bool = False
    prune_atoms: bool = False  # Aktiviert Atom-Pruning mit Äquivalenzvalidierung
    prune_lambda: float | None = None  # (veraltet) einzelner Lambda-Wert; bei None wird auto-search genutzt
    prune_lambda_candidates: list[float] = field(default_factory=lambda: list(_DEFAULT_LAMBDA_CANDIDATES))


def estimator_to_scored_ruleset(
    estimator: Any,
    class_labels: list[Any],
    feature_names: list[str],
    params: TreeTransformParams,
    X_ref: np.ndarray | None = None,
) -> ScoredRuleSet:
    tree_estimator = _unwrap_tree_estimator(estimator)
    if not hasattr(tree_estimator, "tree_"):
        raise TypeError("Estimator kann nicht in tree_ aufgeloest werden")

    if params.prune_atoms:
        return _build_best_pruned_ruleset(
            tree_estimator, class_labels, feature_names, params, X_ref,
        )

    # Kein Pruning – einmalig bauen
    rules = _build_tree_rules(tree_estimator, class_labels, feature_names, params.depth_decay_lambda)

    if params.include_default_rule:
        default_scores = [float(params.default_rule_strength) for _ in class_labels]
        rules.append(Rule(atoms=[], scores=default_scores, rule_id="default"))

    if params.aggressive_prune:
        rules = _deduplicate_atoms(rules)

    return _assemble_ruleset(rules, class_labels, feature_names, params.depth_decay_lambda)


def _build_tree_rules(
    tree_estimator: Any,
    class_labels: list[Any],
    feature_names: list[str],
    depth_decay_lambda: float,
) -> list[Rule]:
    """Extrahiere Regeln aus einem sklearn-Tree mit gegebenem depth_decay_lambda."""
    tree = tree_estimator.tree_
    children_left = tree.children_left
    children_right = tree.children_right
    feature_arr = tree.feature
    threshold = tree.threshold
    leaf_values = tree.value.squeeze(axis=1) if tree.value.ndim == 3 else tree.value

    rules: list[Rule] = []

    def visit(node_id: int, path_atoms: list[Atom], depth: int) -> None:
        is_leaf = children_left[node_id] == children_right[node_id]
        if is_leaf:
            class_idx = int(np.argmax(leaf_values[node_id]))
            weight = float(depth_decay_lambda ** (-max(depth, 1)))
            scores = [0.0 for _ in class_labels]
            scores[class_idx] = weight
            rules.append(
                Rule(
                    atoms=list(path_atoms),
                    scores=scores,
                    rule_id=f"leaf_{node_id}",
                    metadata={"depth": depth, "source": "tree_path"},
                )
            )
            return

        split_feature_idx = int(feature_arr[node_id])
        split_name = feature_names[split_feature_idx]
        split_threshold = float(threshold[node_id])

        visit(
            children_left[node_id],
            path_atoms + [Atom(feature=split_name, op="<=", value=split_threshold)],
            depth + 1,
        )
        visit(
            children_right[node_id],
            path_atoms + [Atom(feature=split_name, op=">", value=split_threshold)],
            depth + 1,
        )

    visit(0, [], 0)
    return rules


def _assemble_ruleset(
    rules: list[Rule],
    class_labels: list[Any],
    feature_names: list[str],
    depth_decay_lambda: float,
) -> ScoredRuleSet:
    rs = ScoredRuleSet(
        class_labels=class_labels,
        feature_names=feature_names,
        rules=rules,
        aggregation=AggregationSpec(type="argmax_sum", temperature=1.0),
        metadata={"transform": "tree_to_scored_ruleset", "depth_decay_lambda": depth_decay_lambda},
    )
    rs.validate()
    return rs


def _build_best_pruned_ruleset(
    tree_estimator: Any,
    class_labels: list[Any],
    feature_names: list[str],
    params: TreeTransformParams,
    X_ref: np.ndarray | None,
) -> ScoredRuleSet:
    """Probiere mehrere depth_decay_lambda-Werte, prune jeweils, waehle kleinstes Ergebnis."""
    # Kandidaten bestimmen
    if params.prune_lambda is not None:
        # Expliziter einzelner Lambda (Rueckwaertskompatibilitaet)
        candidates = [params.prune_lambda]
    else:
        candidates = list(params.prune_lambda_candidates)

    best_ruleset: ScoredRuleSet | None = None
    best_n_atoms = float("inf")

    for lam in candidates:
        rules = _build_tree_rules(tree_estimator, class_labels, feature_names, lam)

        if params.aggressive_prune:
            rules = _deduplicate_atoms(rules)

        rules = _aggressive_atom_pruning(
            rules,
            prune_lambda=lam,
            class_labels=class_labels,
            feature_names=feature_names,
            X_ref=X_ref,
        )

        n_atoms = sum(len(r.atoms) for r in rules)
        if n_atoms < best_n_atoms:
            best_n_atoms = n_atoms
            best_ruleset = _assemble_ruleset(rules, class_labels, feature_names, lam)

    assert best_ruleset is not None
    return best_ruleset


def _unwrap_tree_estimator(estimator: Any) -> Any:
    if hasattr(estimator, "tree_"):
        return estimator

    for attr in ("estimator_", "model_", "best_estimator_", "tree_estimator_"):
        inner = getattr(estimator, attr, None)
        if inner is not None and inner is not estimator:
            if hasattr(inner, "tree_"):
                return inner

    return estimator


def _deduplicate_atoms(rules: list[Rule]) -> list[Rule]:
    compact_rules: list[Rule] = []
    for rule in rules:
        seen: set[tuple[str, str, str]] = set()
        deduped = []
        for atom in rule.atoms:
            key = (str(atom.feature), atom.op, str(atom.value))
            if key not in seen:
                seen.add(key)
                deduped.append(atom)
        compact_rules.append(
            Rule(
                atoms=deduped,
                scores=rule.scores,
                rule_id=rule.rule_id,
                metadata=rule.metadata,
            )
        )
    return compact_rules


def _aggressive_atom_pruning(
    rules: list[Rule],
    prune_lambda: float,
    class_labels: list[Any] | None = None,
    feature_names: list[str] | None = None,
    X_ref: np.ndarray | None = None,
) -> list[Rule]:
    """
    Atom-Pruning-Algorithmus mit Vorhersage-Äquivalenzprüfung.

    Entfernt iterativ Atome, sofern die argmax-Vorhersagen auf *X_ref*
    identisch bleiben.  Ohne *X_ref* wird nur die strukturelle
    Sicherheitsprüfung verwendet (riskant – kann Vorhersagen verändern).
    """
    from ..runtime import predict as _predict_from_ruleset

    if prune_lambda <= 1.0:
        raise ValueError(f"prune_lambda muss > 1 sein, erhalten: {prune_lambda}")

    # Erstelle Kopie zum Modifizieren
    working_rules = [
        Rule(
            atoms=list(rule.atoms),
            scores=list(rule.scores),
            rule_id=rule.rule_id,
            metadata=dict(rule.metadata) if rule.metadata else {},
        )
        for rule in rules
    ]

    # Baseline-Vorhersagen berechnen (falls Referenzdaten vorhanden)
    baseline_preds: np.ndarray | None = None
    if X_ref is not None and class_labels is not None and feature_names is not None:
        baseline_ruleset = ScoredRuleSet(
            class_labels=class_labels,
            feature_names=feature_names,
            rules=working_rules,
            aggregation=AggregationSpec(type="argmax_sum", temperature=1.0),
            metadata={},
        )
        baseline_preds = _predict_from_ruleset(baseline_ruleset, X_ref)

    changed = True
    iteration = 0
    atoms_removed_total = 0

    while changed:
        changed = False
        iteration += 1

        for rule_idx, rule in enumerate(working_rules):
            if not rule.atoms:
                continue

            for atom_idx in range(len(rule.atoms) - 1, -1, -1):
                candidate_atoms = rule.atoms[:atom_idx] + rule.atoms[atom_idx + 1:]

                # Strukturelle Sicherheit: nicht leer werden, Scores positiv
                if not candidate_atoms:
                    continue
                if all(s == 0.0 for s in rule.scores):
                    continue

                candidate_rule = Rule(
                    atoms=candidate_atoms,
                    scores=rule.scores,
                    rule_id=rule.rule_id,
                    metadata=rule.metadata,
                )

                # Mit Referenzdaten: Vorhersage-Äquivalenz prüfen
                if baseline_preds is not None:
                    saved = working_rules[rule_idx]
                    working_rules[rule_idx] = candidate_rule
                    candidate_ruleset = ScoredRuleSet(
                        class_labels=class_labels,
                        feature_names=feature_names,
                        rules=working_rules,
                        aggregation=AggregationSpec(type="argmax_sum", temperature=1.0),
                        metadata={},
                    )
                    candidate_preds = _predict_from_ruleset(candidate_ruleset, X_ref)
                    if np.array_equal(candidate_preds, baseline_preds):
                        # Äquivalenz bestätigt – Atom dauerhaft entfernen
                        atoms_removed_total += 1
                        changed = True
                        break
                    else:
                        # Vorhersage geändert – Atom zurücksetzen
                        working_rules[rule_idx] = saved
                        continue
                else:
                    # Ohne Referenzdaten: rein strukturell (Fallback)
                    working_rules[rule_idx] = candidate_rule
                    atoms_removed_total += 1
                    changed = True
                    break

    return working_rules


