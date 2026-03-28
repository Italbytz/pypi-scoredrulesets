from pathlib import Path

from scoredrulesets import AggregationSpec, Atom, Rule, ScoredRuleSet, dump_ruleset_json, load_ruleset_json


def test_json_roundtrip(tmp_path: Path):
    ruleset = ScoredRuleSet(
        class_labels=["a", "b"],
        feature_names=["f0"],
        aggregation=AggregationSpec(type="argmax_sum"),
        rules=[
            Rule(atoms=[Atom(feature="f0", op=">", value=0.5)], scores=[1.0, 0.0]),
            Rule(atoms=[], scores=[0.0, 0.1]),
        ],
        metadata={"source": "unit_test"},
    )

    target = tmp_path / "model.json"
    dump_ruleset_json(ruleset, target)

    loaded = load_ruleset_json(target)
    assert loaded.class_labels == ["a", "b"]
    assert len(loaded.rules) == 2
    assert loaded.rules[0].atoms[0].op == ">"

