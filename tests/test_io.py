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


def _build_synthetic_archive():
    from scoredrulesets.estimators.auto import MasterParetoArchive, ParetoCandidate

    def _make_rs(n_rules: int, atoms_per_rule: int, backend: str, tree_splits: int | None = None):
        rules = []
        for r in range(n_rules):
            atoms = [
                Atom(feature=f"f{a % 2}", op=">", value=float(r + a) * 0.1)
                for a in range(atoms_per_rule)
            ]
            rules.append(Rule(atoms=atoms, scores=[1.0 if r % 2 == 0 else 0.0, 0.5], rule_id=f"r{r}"))
        rules.append(Rule(atoms=[], scores=[0.1, 0.9], rule_id="default"))
        meta = {"source": backend}
        if tree_splits is not None:
            meta["tree_n_splits"] = tree_splits
        return ScoredRuleSet(
            class_labels=[0, 1],
            feature_names=["f0", "f1"],
            aggregation=AggregationSpec(type="argmax_sum"),
            rules=rules,
            metadata=meta,
        )

    c1 = ParetoCandidate(
        backend="greedy_pareto",
        atoms=2,
        rules=2,
        score=0.72,
        ruleset=_make_rs(1, 2, "greedy_pareto"),
    )
    c2 = ParetoCandidate(
        backend="cart",
        atoms=5,
        rules=4,
        score=0.81,
        ruleset=_make_rs(3, 1, "cart", tree_splits=3),
    )
    c3 = ParetoCandidate(
        backend="rulekit_native",
        atoms=9,
        rules=5,
        score=0.87,
        ruleset=_make_rs(4, 2, "rulekit_native"),
    )
    c4 = ParetoCandidate(
        backend="ebm",
        atoms=18,
        rules=10,
        score=0.90,
        ruleset=_make_rs(9, 2, "ebm"),
    )
    dominated = ParetoCandidate(
        backend="cart",
        atoms=7,
        rules=5,
        score=0.79,
        ruleset=_make_rs(4, 1, "cart", tree_splits=4),
    )

    arch = MasterParetoArchive(
        candidates_=[c1, c2, c3, c4],
        candidate_pool_=[c1, c2, c3, c4, dominated],
        metadata_={"dataset": "synthetic_demo", "n_samples": 120},
        metric_name="balanced_accuracy",
    )
    return arch


def test_archive_zip_bundle_structure_and_roundtrip(tmp_path: Path):
    import zipfile
    import numpy as np
    from scoredrulesets import MasterParetoArchive, dump_archive_zip, load_archive_zip

    arch = _build_synthetic_archive()
    zip_path = tmp_path / "master_archive.zip"

    saved_path = dump_archive_zip(arch, zip_path)
    assert saved_path == zip_path
    assert zip_path.exists()

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        assert "manifest.json" in names
        assert "archive.json" in names
        assert "candidate_pool.json" in names
        assert "SUMMARY.md" in names
        candidate_jsons = [n for n in names if n.startswith("candidates/") and n.endswith(".json")]
        candidate_mds = [n for n in names if n.startswith("candidates/") and n.endswith(".md")]
        assert len(candidate_jsons) == 4
        assert len(candidate_mds) == 4

    loaded = load_archive_zip(zip_path)
    assert isinstance(loaded, MasterParetoArchive)
    assert loaded.metric_name == "balanced_accuracy"
    assert loaded.metadata_["dataset"] == "synthetic_demo"
    assert len(loaded.candidates_) == 4
    assert len(loaded.candidate_pool_) == 5

    # Also test MasterParetoArchive.save / MasterParetoArchive.load dispatch for .zip and .json
    auto_zip = arch.save(tmp_path / "via_save.zip")
    reloaded_zip = MasterParetoArchive.load(auto_zip)
    assert len(reloaded_zip.candidates_) == 4
    assert len(reloaded_zip.candidate_pool_) == 5

    auto_json = arch.save(tmp_path / "via_save.json")
    reloaded_json = MasterParetoArchive.load(auto_json)
    assert len(reloaded_json.candidates_) == 4

    X = np.array([[0.5, 0.5], [0.0, 0.0], [1.2, 0.8]])
    for pref in ("compact", "balanced", "performance", "expressive", "accuracy"):
        c_orig = arch.select_candidate(pref)
        c_load = loaded.select_candidate(pref)
        assert c_orig.atoms == c_load.atoms
        assert c_orig.backend == c_load.backend
        np.testing.assert_array_equal(
            arch.predict(X, preference=pref),
            loaded.predict(X, preference=pref),
        )
        np.testing.assert_allclose(
            arch.predict_proba(X, preference=pref),
            loaded.predict_proba(X, preference=pref),
        )


def test_archive_post_hoc_selection_by_strategy_and_fixed_size(tmp_path: Path):
    import pytest
    from scoredrulesets import load_archive_zip

    arch = _build_synthetic_archive()
    zip_path = arch.save_zip(tmp_path / "post_hoc.zip")
    loaded = load_archive_zip(zip_path)

    # 1. Operational intent strategies (monotonic complexity ordering)
    c_comp = loaded.select_candidate("compact")
    c_bal = loaded.select_candidate("balanced")
    c_perf = loaded.select_candidate("performance")
    c_expr = loaded.select_candidate("expressive")
    c_acc = loaded.select_candidate("accuracy")
    assert c_comp.atoms <= c_bal.atoms <= c_perf.atoms <= c_acc.atoms
    assert c_bal.atoms <= c_expr.atoms <= c_acc.atoms
    assert c_acc.atoms == 18
    profile_summary = loaded.get_profile_summary()
    assert set(profile_summary.keys()) == {"compact", "balanced", "performance", "expressive", "accuracy"}
    md_summary = loaded.format_markdown_summary()
    assert "Post-Hoc Operational Profile Menu" in md_summary
    assert "Non-Dominated Master Pareto Frontier" in md_summary

    # 2. Closest size selection (default mode="closest")
    c_atoms_6 = loaded.select_by_size(atoms=6, mode="closest", return_candidate=True)
    assert c_atoms_6.atoms == 5  # closest to 6 among {2, 5, 9, 18}

    c_rules_4 = loaded.select_by_size(rules=4, mode="closest", return_candidate=True)
    assert c_rules_4.rules == 4

    c_eff_3 = loaded.select_by_size(effective_nodes=3, mode="closest", return_candidate=True)
    assert c_eff_3.effective_nodes == 3

    # 3. Exact size selection (mode="exact")
    c_exact = loaded.select_by_size(atoms=9, mode="exact", return_candidate=True)
    assert c_exact.atoms == 9
    with pytest.raises(ValueError, match="matches exact size"):
        loaded.select_by_size(atoms=7, mode="exact")

    # 4. Bounded size selection (mode="max" or max_atoms / max_rules / max_effective_nodes)
    c_max_10 = loaded.select_by_size(atoms=10, mode="max", preference="accuracy", return_candidate=True)
    assert c_max_10.atoms == 9
    c_bounded = loaded.select_candidate("accuracy", max_atoms=6)
    assert c_bounded.atoms == 5


def test_archive_post_hoc_visualization(tmp_path: Path):
    from scoredrulesets import load_archive_zip

    arch = _build_synthetic_archive()
    zip_path = arch.save_zip(tmp_path / "viz_archive.zip")
    loaded = load_archive_zip(zip_path)

    for metric in ("atoms", "effective_nodes", "rules"):
        fig_path = tmp_path / f"pareto_{metric}.png"
        fig = loaded.plot_pareto_front(
            output_path=fig_path,
            complexity_metric=metric,
            show_backend_fronts=True,
            highlight_size=5,
            title=f"Post-Hoc Pareto Front ({metric})",
        )
        assert fig is not None
        assert fig_path.exists()
        assert fig_path.stat().st_size > 1000


