from __future__ import annotations

import numpy as np

from scoredrulesets.estimators.exstracs_shrinking import exstracs_prune_conservative
from scoredrulesets.runtime import decision_function
from scoredrulesets.schema import AggregationSpec, Atom, Rule, ScoredRuleSet


def _toy_ruleset_with_defaults() -> ScoredRuleSet:
    return ScoredRuleSet(
        class_labels=[0, 1, 2],
        feature_names=["f0", "f1"],
        rules=[
            Rule(atoms=[], scores=[0.1, 0.0, 0.0], rule_id="default_a"),
            Rule(atoms=[], scores=[0.0, 0.2, 0.0], rule_id="default_b"),
            Rule(
                atoms=[Atom(feature="f0", op=">", value=0.5)],
                scores=[0.0, 0.0, 1.0],
                rule_id="r1",
            ),
        ],
        aggregation=AggregationSpec(type="argmax_sum", temperature=1.0),
        metadata={"source": "test"},
    )


def test_conservative_pruning_does_not_create_empty_non_default_rules():
    rs = ScoredRuleSet(
        class_labels=[0, 1],
        feature_names=["f0"],
        rules=[
            Rule(atoms=[], scores=[0.5, 0.5], rule_id="default"),
            Rule(atoms=[Atom(feature="f0", op=">", value=0.1)], scores=[1.0, 0.0], rule_id="r_pos"),
            Rule(atoms=[Atom(feature="f0", op="<=", value=0.1)], scores=[0.0, 1.0], rule_id="r_neg"),
        ],
        aggregation=AggregationSpec(type="argmax_sum", temperature=1.0),
        metadata={"source": "test"},
    )

    pruned = exstracs_prune_conservative(rs)

    non_default = [r for r in pruned.rules if r.atoms]
    assert len(non_default) == 2
    assert all(len(r.atoms) > 0 for r in non_default)


def test_multiple_default_rules_are_consolidated_after_pruning():
    rs = _toy_ruleset_with_defaults()
    pruned = exstracs_prune_conservative(rs)

    defaults = [r for r in pruned.rules if not r.atoms]
    assert len(defaults) == 1
    # default_a + default_b
    assert defaults[0].scores == [0.1, 0.2, 0.0]


def test_runtime_combines_multiple_default_rules_on_fallback():
    rs = ScoredRuleSet(
        class_labels=[0, 1],
        feature_names=["f0"],
        rules=[
            Rule(atoms=[], scores=[0.2, 0.0], rule_id="default_a"),
            Rule(atoms=[], scores=[0.0, 0.3], rule_id="default_b"),
        ],
        aggregation=AggregationSpec(type="argmax_sum", temperature=1.0),
        metadata={"source": "test"},
    )

    scores = decision_function(rs, X=np.asarray([[0.0]], dtype=float))
    assert scores.shape == (1, 2)
    assert scores[0, 0] == 0.2
    assert scores[0, 1] == 0.3

