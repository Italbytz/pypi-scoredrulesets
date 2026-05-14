from pathlib import Path

from scoredrulesets import (
    AggregationSpec,
    Atom,
    Rule,
    ScoredRuleSet,
    dump_ruleset_json,
    format_ruleset_latex,
    format_ruleset_markdown,
    format_ruleset_table,
    load_ruleset_json,
)


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


def test_ruleset_table_exports_use_separate_score_columns():
    ruleset = ScoredRuleSet(
        class_labels=["setosa", "versicolor"],
        feature_names=["petal length"],
        rules=[
            Rule(
                atoms=[Atom(feature="petal length", op="between", value=[1.4, 1.6])],
                scores=[1.0, 0.0],
                rule_id="r0",
            ),
            Rule(atoms=[], scores=[0.0, 1.0], rule_id="default"),
        ],
    )

    ascii_table = format_ruleset_table(ruleset)
    markdown_table = format_ruleset_markdown(ruleset)
    latex_table = format_ruleset_latex(ruleset)

    for output in [ascii_table, markdown_table]:
        assert "setosa" in output
        assert "versicolor" in output
        assert "S_r" in output
        assert "1.00" in output
        assert "0.00" in output

    assert "S\\_r" in latex_table

    assert "petal length in [1.40, 1.60]" in ascii_table
    assert "petal length in [1.40, 1.60]" in markdown_table
    assert "petal length in [1.40, 1.60]" in latex_table
    assert "emptyset" in ascii_table
    assert "emptyset" in markdown_table
    assert "emptyset" in latex_table
    assert "\\begin{tabular}" in latex_table


def test_ruleset_table_uses_yes_no_for_binary_zero_one_labels_only():
    binary_ruleset = ScoredRuleSet(
        class_labels=[0, 1],
        feature_names=["f0"],
        rules=[
            Rule(atoms=[Atom(feature="f0", op=">", value=0.5)], scores=[1.0, 0.0], rule_id="r0"),
            Rule(atoms=[], scores=[0.0, 1.0], rule_id="default"),
        ],
    )
    binary_table = format_ruleset_markdown(binary_ruleset)
    assert "| No | Yes | S_r |" in binary_table

    multiclass_ruleset = ScoredRuleSet(
        class_labels=[0, 1, 2],
        feature_names=["f0"],
        rules=[Rule(atoms=[], scores=[0.2, 0.3, 0.5], rule_id="default")],
    )
    multiclass_table = format_ruleset_markdown(multiclass_ruleset)
    assert "| 0 | 1 | 2 | S_r |" in multiclass_table


def test_json_roundtrip_regression_ruleset(tmp_path: Path):
    ruleset = ScoredRuleSet(
        class_labels=[],
        task_type="regression",
        feature_names=["x"],
        aggregation=AggregationSpec(type="default_plus_sum"),
        rules=[
            Rule(atoms=[Atom(feature="x", op=">", value=0.5)], scores=[0.2]),
            Rule(atoms=[], scores=[1.5]),
        ],
        metadata={"source": "unit_test_regression"},
    )

    target = tmp_path / "regression_model.json"
    dump_ruleset_json(ruleset, target)

    loaded = load_ruleset_json(target)
    assert loaded.task_type == "regression"
    assert loaded.class_labels == []
    assert len(loaded.rules) == 2
    assert loaded.rules[0].scores == [0.2]

