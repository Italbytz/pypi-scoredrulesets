from __future__ import annotations

import numpy as np

from scoredrulesets.estimators.exstracs_shrinking import exstracs_prune_conservative
from scoredrulesets.estimators.exstracs_shrinking import (
    exstracs_merge_intervals,
    _interval_iou,
    _extract_feature_intervals,
    _rule_feature_schema,
)
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


# ---------------------------------------------------------------------------
# Interval-Merge Tests
# ---------------------------------------------------------------------------


def _make_binary_ruleset(rules: list[Rule]) -> ScoredRuleSet:
    """Helper function: binary ruleset with default rule."""
    return ScoredRuleSet(
        class_labels=[0, 1],
        feature_names=["f0", "f1", "f2"],
        rules=[Rule(atoms=[], scores=[0.1, 0.1], rule_id="default")] + rules,
        aggregation=AggregationSpec(type="argmax_sum", temperature=1.0),
        metadata={"source": "test"},
    )


class TestIntervalIou:
    def test_identical_intervals(self):
        assert _interval_iou((1.0, 3.0), (1.0, 3.0)) == 1.0

    def test_disjoint_intervals(self):
        assert _interval_iou((0.0, 1.0), (2.0, 3.0)) == 0.0

    def test_partial_overlap(self):
        iou = _interval_iou((0.0, 2.0), (1.0, 3.0))
        # Intersection = [1,2] = 1.0; Union = [0,3] = 3.0 → IoU = 1/3
        assert abs(iou - 1.0 / 3.0) < 1e-9

    def test_contained_interval(self):
        iou = _interval_iou((0.0, 4.0), (1.0, 3.0))
        # Intersection = [1,3] = 2.0; Union = [0,4] = 4.0 → IoU = 0.5
        assert abs(iou - 0.5) < 1e-9

    def test_point_intervals_same(self):
        assert _interval_iou((2.0, 2.0), (2.0, 2.0)) == 1.0

    def test_point_intervals_different(self):
        assert _interval_iou((1.0, 1.0), (2.0, 2.0)) == 0.0


class TestExtractFeatureIntervals:
    def test_interval_from_gt_lt(self):
        rule = Rule(
            atoms=[
                Atom(feature="f0", op=">", value=1.0),
                Atom(feature="f0", op="<", value=5.0),
            ],
            scores=[1.0, 0.0],
        )
        ivs = _extract_feature_intervals(rule)
        assert "f0" in ivs
        assert ivs["f0"] == (1.0, 5.0)

    def test_equality_atom(self):
        rule = Rule(
            atoms=[Atom(feature="f1", op="==", value=3.0)],
            scores=[0.0, 1.0],
        )
        ivs = _extract_feature_intervals(rule)
        assert ivs["f1"] == (3.0, 3.0)

    def test_multiple_features(self):
        rule = Rule(
            atoms=[
                Atom(feature="f0", op=">", value=0.0),
                Atom(feature="f0", op="<", value=2.0),
                Atom(feature="f1", op=">=", value=5.0),
            ],
            scores=[1.0, 0.0],
        )
        ivs = _extract_feature_intervals(rule)
        assert ivs["f0"] == (0.0, 2.0)
        assert ivs["f1"] == (5.0, np.inf)


class TestRuleFeatureSchema:
    def test_basic(self):
        rule = Rule(
            atoms=[
                Atom(feature="f0", op=">", value=1.0),
                Atom(feature="f0", op="<", value=5.0),
                Atom(feature="f1", op=">", value=0.0),
            ],
            scores=[1.0, 0.0],
        )
        assert _rule_feature_schema(rule) == frozenset({"f0", "f1"})


class TestIntervalMerge:
    def test_same_class_same_schema_merges(self):
        """Two rules with same class and schema merge into one."""
        rules = [
            Rule(
                atoms=[
                    Atom(feature="f0", op=">", value=1.0),
                    Atom(feature="f0", op="<", value=3.0),
                ],
                scores=[0.0, 2.0],
                rule_id="r0",
                metadata={"numerosity": 5},
            ),
            Rule(
                atoms=[
                    Atom(feature="f0", op=">", value=2.0),
                    Atom(feature="f0", op="<", value=4.0),
                ],
                scores=[0.0, 3.0],
                rule_id="r1",
                metadata={"numerosity": 3},
            ),
        ]
        rs = _make_binary_ruleset(rules)
        merged = exstracs_merge_intervals(rs, iou_threshold=0.1)

        non_default = [r for r in merged.rules if r.atoms]
        assert len(non_default) == 1, f"Expected 1 merged rule, got {len(non_default)}"

        # Scores are summed
        m = non_default[0]
        assert m.scores[0] == 0.0
        assert m.scores[1] == 5.0  # 2 + 3

        # Interval is the union [1, 4]
        ivs = _extract_feature_intervals(m)
        assert ivs["f0"][0] == 1.0  # min lower bound
        assert ivs["f0"][1] == 4.0  # max upper bound

    def test_different_class_not_merged(self):
        """Rules from different classes are not merged."""
        rules = [
            Rule(
                atoms=[
                    Atom(feature="f0", op=">", value=1.0),
                    Atom(feature="f0", op="<", value=3.0),
                ],
                scores=[2.0, 0.0],  # class 0
                rule_id="r0",
            ),
            Rule(
                atoms=[
                    Atom(feature="f0", op=">", value=1.5),
                    Atom(feature="f0", op="<", value=3.5),
                ],
                scores=[0.0, 3.0],  # class 1
                rule_id="r1",
            ),
        ]
        rs = _make_binary_ruleset(rules)
        merged = exstracs_merge_intervals(rs, iou_threshold=0.1)

        non_default = [r for r in merged.rules if r.atoms]
        assert len(non_default) == 2, "Different classes should not be merged"

    def test_different_schema_not_merged(self):
        """Rules with different feature sets are not merged."""
        rules = [
            Rule(
                atoms=[
                    Atom(feature="f0", op=">", value=1.0),
                    Atom(feature="f0", op="<", value=3.0),
                ],
                scores=[0.0, 2.0],
                rule_id="r0",
            ),
            Rule(
                atoms=[
                    Atom(feature="f1", op=">", value=1.0),
                    Atom(feature="f1", op="<", value=3.0),
                ],
                scores=[0.0, 3.0],
                rule_id="r1",
            ),
        ]
        rs = _make_binary_ruleset(rules)
        merged = exstracs_merge_intervals(rs, iou_threshold=0.1)

        non_default = [r for r in merged.rules if r.atoms]
        assert len(non_default) == 2, "Different feature schemas should not be merged"

    def test_disjoint_intervals_high_threshold_not_merged(self):
        """Disjoint intervals + high threshold -> no merge."""
        rules = [
            Rule(
                atoms=[
                    Atom(feature="f0", op=">", value=0.0),
                    Atom(feature="f0", op="<", value=1.0),
                ],
                scores=[0.0, 2.0],
                rule_id="r0",
            ),
            Rule(
                atoms=[
                    Atom(feature="f0", op=">", value=10.0),
                    Atom(feature="f0", op="<", value=11.0),
                ],
                scores=[0.0, 3.0],
                rule_id="r1",
            ),
        ]
        rs = _make_binary_ruleset(rules)
        merged = exstracs_merge_intervals(rs, iou_threshold=0.5)

        non_default = [r for r in merged.rules if r.atoms]
        assert len(non_default) == 2, "Disjoint intervals with high IoU threshold should not be merged"

    def test_preserves_default_rule(self):
        """Default rule is preserved."""
        rules = [
            Rule(
                atoms=[
                    Atom(feature="f0", op=">", value=1.0),
                    Atom(feature="f0", op="<", value=3.0),
                ],
                scores=[0.0, 2.0],
                rule_id="r0",
            ),
        ]
        rs = _make_binary_ruleset(rules)
        merged = exstracs_merge_intervals(rs, iou_threshold=0.3)

        defaults = [r for r in merged.rules if not r.atoms]
        assert len(defaults) == 1
        assert defaults[0].scores == [0.1, 0.1]

    def test_discrete_atoms_same_value_merged(self):
        """Discrete atoms (==) with the same value are merged."""
        rules = [
            Rule(
                atoms=[Atom(feature="f0", op="==", value=1.0)],
                scores=[0.0, 2.0],
                rule_id="r0",
                metadata={"numerosity": 3},
            ),
            Rule(
                atoms=[Atom(feature="f0", op="==", value=1.0)],
                scores=[0.0, 4.0],
                rule_id="r1",
                metadata={"numerosity": 7},
            ),
        ]
        rs = _make_binary_ruleset(rules)
        merged = exstracs_merge_intervals(rs, iou_threshold=0.3)

        non_default = [r for r in merged.rules if r.atoms]
        assert len(non_default) == 1
        assert non_default[0].scores[1] == 6.0  # 2 + 4

    def test_metadata_contains_merged_count(self):
        """Merge metadata contains merged_count and total_numerosity."""
        rules = [
            Rule(
                atoms=[Atom(feature="f0", op=">", value=1.0), Atom(feature="f0", op="<", value=5.0)],
                scores=[0.0, 2.0],
                rule_id="r0",
                metadata={"numerosity": 4},
            ),
            Rule(
                atoms=[Atom(feature="f0", op=">", value=2.0), Atom(feature="f0", op="<", value=6.0)],
                scores=[0.0, 3.0],
                rule_id="r1",
                metadata={"numerosity": 6},
            ),
        ]
        rs = _make_binary_ruleset(rules)
        merged = exstracs_merge_intervals(rs, iou_threshold=0.1)

        non_default = [r for r in merged.rules if r.atoms]
        assert len(non_default) == 1
        m = non_default[0]
        assert m.metadata["merged_count"] == 2
        assert m.metadata["total_numerosity"] == 10.0

    def test_massive_reduction(self):
        """Simulate a typical ExSTraCS scenario: many similar rules -> few."""
        rng = np.random.default_rng(42)
        many_rules = []
        for i in range(100):
            center = rng.uniform(0, 5)
            width = rng.uniform(0.5, 2.0)
            score = rng.uniform(0.1, 1.0)
            many_rules.append(
                Rule(
                    atoms=[
                        Atom(feature="f0", op=">", value=center - width / 2),
                        Atom(feature="f0", op="<", value=center + width / 2),
                    ],
                    scores=[0.0, score],
                    rule_id=f"r{i}",
                    metadata={"numerosity": int(rng.integers(1, 10))},
                )
            )
        rs = _make_binary_ruleset(many_rules)
        merged = exstracs_merge_intervals(rs, iou_threshold=0.2)

        non_default_before = len(many_rules)
        non_default_after = len([r for r in merged.rules if r.atoms])
        reduction = 1.0 - non_default_after / non_default_before

        assert non_default_after < non_default_before, "Interval merge should reduce rule count"
        assert reduction > 0.5, f"Expected >50% reduction, got {reduction:.0%}"

    def test_score_sum_preserves_total_weight(self):
        """The sum of all scores is preserved after merge."""
        rules = [
            Rule(
                atoms=[Atom(feature="f0", op=">", value=i * 0.1), Atom(feature="f0", op="<", value=i * 0.1 + 2.0)],
                scores=[0.0, float(i + 1)],
                rule_id=f"r{i}",
            )
            for i in range(10)
        ]
        rs = _make_binary_ruleset(rules)
        total_before = sum(s for r in rs.rules for s in r.scores)

        merged = exstracs_merge_intervals(rs, iou_threshold=0.1)
        total_after = sum(s for r in merged.rules for s in r.scores)

        assert abs(total_before - total_after) < 1e-9, (
            f"Total score weight should be preserved: {total_before} vs {total_after}"
        )

