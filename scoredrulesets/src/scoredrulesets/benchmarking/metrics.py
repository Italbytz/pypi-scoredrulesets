from __future__ import annotations

import json

from ..schema import ScoredRuleSet


def model_size_metrics(ruleset: ScoredRuleSet) -> tuple[int, int, int]:
    n_rules = len(ruleset.rules)
    n_atoms = sum(len(rule.atoms) for rule in ruleset.rules)
    json_bytes = len(json.dumps(ruleset.to_dict(), separators=(",", ":")).encode("utf-8"))
    return n_rules, n_atoms, json_bytes

