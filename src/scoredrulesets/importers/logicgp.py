from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..schema import AggregationSpec, Atom, Rule, ScoredRuleSet


def _decode_logicgp_categories(
    feature_value_mappings: list[dict[str, int]] | None,
    feature_idx: int,
    categories: list[int],
) -> list[Any]:
    if not feature_value_mappings or feature_idx >= len(feature_value_mappings):
        return categories

    mapping = feature_value_mappings[feature_idx]
    inverse: dict[int, Any] = {}
    for raw, cat in mapping.items():
        inverse[int(cat)] = raw
    return [inverse.get(cat, cat) for cat in categories]


def import_logicgp_json(
    path: str | Path,
    decode_categories: bool = False,
    include_default_rule: bool = True,
) -> ScoredRuleSet:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    polynomial = payload["Model"]["Genotype"]["Polynomial"]
    raw_monomials = polynomial.get("Monomials", [])
    default_scores = [float(v) for v in polynomial.get("Weights", [])]

    reverse = payload.get("_reverseLabelMapping", {})
    class_labels = [reverse[str(i)] if str(i) in reverse else i for i in range(len(default_scores))]

    feature_value_mappings = payload.get("_featureValueMappings")

    rules: list[Rule] = []
    if include_default_rule and any(abs(v) > 0.0 for v in default_scores):
        rules.append(Rule(atoms=[], scores=default_scores, rule_id="logicgp_default"))

    for idx, monomial in enumerate(raw_monomials):
        atoms: list[Atom] = []
        for lit_idx, literal in enumerate(monomial.get("Literals", [])):
            feature_idx = int(literal["Feature"])
            categories = [int(c) for c in literal.get("Categories", [])]
            categories_value: list[Any] = categories
            if decode_categories:
                categories_value = _decode_logicgp_categories(
                    feature_value_mappings=feature_value_mappings,
                    feature_idx=feature_idx,
                    categories=categories,
                )
            atoms.append(
                Atom(
                    feature=feature_idx,
                    op="in",
                    value=categories_value,
                    metadata={
                        "logicgp_literal_type": literal.get("LiteralType"),
                        "logicgp_set": literal.get("Set"),
                        "logicgp_category_ids": categories,
                        "source": "logicgp",
                        "literal_index": lit_idx,
                    },
                )
            )
        scores = [float(v) for v in monomial.get("Weights", [])]
        rules.append(Rule(atoms=atoms, scores=scores, rule_id=f"logicgp_monomial_{idx}"))

    ruleset = ScoredRuleSet(
        class_labels=class_labels,
        rules=rules,
        feature_names=[f"f{i}" for i in range(len(payload.get("_featureValueMappings", [])))],
        aggregation=AggregationSpec(type="softmax_sum", temperature=1.0),
        metadata={"source": "logicgp", "logicgp_prediction_strategy": payload["Model"]["Genotype"].get("PredictionStrategy")},
    )
    ruleset.validate()
    return ruleset

