"""Tests for Pareto front and Rashomon interaction analysis."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from scoredrulesets.analysis import (
    ParetoFrontSummary,
    ParetoInteractionEdge,
    extract_pareto_interactions,
)
from scoredrulesets.schema import Atom, Rule, ScoredRuleSet


def _make_sample_ruleset(rules_atoms: list[list[str]], feature_names: list[str]) -> ScoredRuleSet:
    rules = []
    for atom_feats in rules_atoms:
        atoms = [Atom(feature=f, op="==", value=1) for f in atom_feats]
        rules.append(Rule(atoms=atoms, scores=[1.0, 0.0]))
    return ScoredRuleSet(
        class_labels=[0, 1],
        rules=rules,
        feature_names=feature_names,
    )


def test_extract_pareto_interactions_from_rulesets():
    fn = ["age", "bmi", "smoking", "gene_a", "gene_b"]

    # Model 1: (age & bmi), (gene_a & gene_b)
    m1 = _make_sample_ruleset([["age", "bmi"], ["gene_a", "gene_b"]], fn)
    # Model 2: (age & bmi & smoking), (gene_a & gene_b)
    m2 = _make_sample_ruleset([["age", "bmi", "smoking"], ["gene_a", "gene_b"]], fn)
    # Model 3: (age & smoking), (gene_a)
    m3 = _make_sample_ruleset([["age", "smoking"], ["gene_a"]], fn)

    summary = extract_pareto_interactions(
        [m1, m2, m3],
        backbone_threshold=0.6,
        min_occurrences=1,
        min_ratio=0.0,
    )

    assert summary.n_models == 3
    # age appears in all 3 models -> frequency = 1.0 (backbone)
    # gene_a appears in all 3 models -> frequency = 1.0 (backbone)
    # bmi appears in 2 of 3 -> frequency = 0.667 (backbone)
    # smoking appears in 2 of 3 -> frequency = 0.667 (backbone)
    # gene_b appears in 2 of 3 -> frequency = 0.667 (backbone)
    assert "age" in summary.backbone_features
    assert "gene_a" in summary.backbone_features

    # Check interactions:
    # age & bmi co-occur in m1 and m2 (count=2)
    # gene_a & gene_b co-occur in m1 and m2 (count=2)
    edge_map = {(e.feature_a, e.feature_b): e.count for e in summary.edges}
    assert edge_map.get(("age", "bmi")) == 2
    assert edge_map.get(("gene_a", "gene_b")) == 2
    assert edge_map.get(("bmi", "smoking")) == 1


def test_pareto_summary_markdown_and_dot():
    fn = ["f0", "f1", "f2"]
    m1 = _make_sample_ruleset([["f0", "f1"], ["f0", "f2"]], fn)
    summary = extract_pareto_interactions([m1])

    dot_str = summary.to_dot(title="Test Front")
    assert 'graph "Test Front"' in dot_str
    assert '"f0"' in dot_str
    assert '"f1"' in dot_str
    assert "--" in dot_str

    md_str = summary.to_markdown()
    assert "Pareto Front Interaction Summary" in md_str
    assert "| Feature |" in md_str
    assert "| `f0` |" in md_str

    d = summary.to_dict()
    assert d["n_models"] == 1
    assert "edges" in d


def test_file_exports():
    fn = ["x1", "x2"]
    m1 = _make_sample_ruleset([["x1", "x2"]], fn)

    with tempfile.TemporaryDirectory() as tmpdir:
        dot_path = Path(tmpdir) / "interactions.dot"
        csv_path = Path(tmpdir) / "interactions.csv"

        summary = extract_pareto_interactions(
            [m1],
            out_dot=dot_path,
            out_csv=csv_path,
        )

        assert dot_path.exists()
        assert csv_path.exists()
        assert len(dot_path.read_text(encoding="utf-8")) > 0
        assert "feature_a,feature_b" in csv_path.read_text(encoding="utf-8")


def test_rulegp_extract_pareto_interactions():
    from scoredrulesets import RuleGPClassifier

    rng = np.random.RandomState(42)
    X = rng.randn(60, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    clf = RuleGPClassifier(
        max_generations=5,
        record_population=True,
        random_state=42,
    )
    clf.fit(X, y)

    # Call delegate method directly on the estimator
    summary = clf.extract_pareto_interactions(min_occurrences=1, min_ratio=0.0)
    assert isinstance(summary, ParetoFrontSummary)
    assert summary.n_models > 0
    assert len(summary.feature_counts) > 0


def test_logicgp_extract_pareto_interactions():
    from scoredrulesets import LogicGPClassifier

    rng = np.random.default_rng(42)
    X = rng.integers(0, 3, size=(60, 4)).astype(object)
    y = np.array([0] * 30 + [1] * 30)

    clf = LogicGPClassifier(
        trainer="rlcw",
        max_generations=10,
        stagnation_generations=5,
        n_bins=3,
        random_state=42,
    )
    clf.fit(X, y)

    summary = clf.extract_pareto_interactions(min_occurrences=1, min_ratio=0.0)
    assert isinstance(summary, ParetoFrontSummary)
    assert summary.n_models > 0


def test_extract_ensemble_interactions_ruleset():
    """Test Mode B (Intra-Ensemble Distillation) with a single ScoredRuleSet."""
    fn = ["SNP1", "SNP2", "SNP3", "SNP4", "SNP5"]
    rules_atoms = [
        ["SNP1", "SNP2"],
        ["SNP1", "SNP2"],
        ["SNP1", "SNP2"],
        ["SNP1", "SNP3"],
        ["SNP4", "SNP5"],
    ]
    srs = _make_sample_ruleset(rules_atoms, fn)

    summary = extract_pareto_interactions(
        srs,
        mode="ensemble",
        backbone_threshold=0.5,
        min_occurrences=1,
        min_ratio=0.0,
    )

    assert summary.mode == "ensemble"
    assert summary.n_rules == 5
    # SNP1 appears in 4/5 rules (80%) -> backbone
    # SNP2 appears in 3/5 rules (60%) -> backbone
    # SNP3 appears in 1/5 rules (20%) -> peripheral
    assert "SNP1" in summary.backbone_features
    assert "SNP2" in summary.backbone_features
    assert "SNP3" in summary.peripheral_features

    # Check edge between SNP1 and SNP2
    edges = {(e.feature_a, e.feature_b): e for e in summary.edges}
    assert ("SNP1", "SNP2") in edges
    assert edges[("SNP1", "SNP2")].count == 3
    assert edges[("SNP1", "SNP2")].weight > 0

    # Test Markdown and DOT generation for ensemble mode
    md = summary.to_markdown()
    assert "Intra-Ensemble Interaction Summary" in md
    assert "Rule Frequency" in md
    dot = summary.to_dot()
    assert "Intra-Ensemble Interaction Graph" in dot
    assert '"SNP1"' in dot


def test_extract_ensemble_interactions_backbone_criterion():
    """Test backbone_criterion='weight' vs 'frequency'."""
    fn = ["A", "B", "C"]
    rules = [
        Rule(atoms=[Atom(feature="A", op="==", value=1)], scores=[10.0, 0.0]),
        Rule(atoms=[Atom(feature="B", op="==", value=1)], scores=[1.0, 0.0]),
        Rule(atoms=[Atom(feature="B", op="==", value=1)], scores=[1.0, 0.0]),
        Rule(atoms=[Atom(feature="B", op="==", value=1)], scores=[1.0, 0.0]),
    ]
    srs = ScoredRuleSet(class_labels=[0, 1], rules=rules, feature_names=fn)

    # Frequency-based: B has frequency 3/4 = 75%, A has 1/4 = 25%
    sum_freq = extract_pareto_interactions(
        srs, mode="ensemble", backbone_threshold=0.5, backbone_criterion="frequency"
    )
    assert "B" in sum_freq.backbone_features
    assert "A" not in sum_freq.backbone_features

    # Weight-based: A has weight 10.0 (100% of max), B has weight 3.0 (30% of max)
    sum_wt = extract_pareto_interactions(
        srs, mode="ensemble", backbone_threshold=0.5, backbone_criterion="weight"
    )
    assert "A" in sum_wt.backbone_features
    assert "B" not in sum_wt.backbone_features


def test_extract_ensemble_rulefit_mock():
    """Test RuleFit extraction via mock DataFrame from visualize()."""
    import pandas as pd

    class MockRuleFit:
        feature_names_in_ = ["SNP1", "SNP2", "SNP3"]

        def visualize(self):
            return pd.DataFrame(
                [
                    {"rule": "SNP1 > 0.5 and SNP2 <= 1.0", "coef": 5.2},
                    {"rule": "SNP1 > 0.5 and SNP2 > 1.0", "coef": 3.8},
                    {"rule": "SNP3 <= 0.0", "coef": 0.4},
                    {"rule": "SNP1 > 0.0", "coef": 0.00001},  # pruned by threshold
                ]
            )

    rf = MockRuleFit()
    summary = extract_pareto_interactions(rf, mode="auto", backbone_threshold=0.5)

    assert summary.mode == "ensemble"
    assert summary.n_rules == 3  # zero/tiny pruned
    assert "SNP1" in summary.backbone_features
    assert "SNP2" in summary.backbone_features
    assert summary.feature_weights["SNP1"] == pytest.approx(9.0, abs=0.01)

    edges = {(e.feature_a, e.feature_b): e for e in summary.edges}
    assert ("SNP1", "SNP2") in edges
    assert edges[("SNP1", "SNP2")].count == 2
    assert edges[("SNP1", "SNP2")].weight == pytest.approx(9.0, abs=0.01)


def test_extract_ensemble_rulefit_real():
    """Test with real fitted RuleFitClassifier from imodels if available."""
    try:
        from imodels import RuleFitClassifier
    except ImportError:
        pytest.skip("imodels not installed")

    rng = np.random.RandomState(42)
    X = rng.randn(60, 4)
    y = ((X[:, 0] > 0) & (X[:, 1] > 0)).astype(int)
    fn = ["F0", "F1", "F2", "F3"]

    rf = RuleFitClassifier(max_rules=15, random_state=42)
    rf.fit(X, y, feature_names=fn)

    summary = extract_pareto_interactions(
        rf, mode="auto", backbone_threshold=0.2, min_occurrences=1, min_ratio=0.0
    )
    assert summary.mode == "ensemble"
    assert summary.n_rules > 0
    assert len(summary.feature_counts) > 0
    assert "F0" in summary.feature_names


def test_invalid_mode_raises():
    fn = ["x1", "x2"]
    m1 = _make_sample_ruleset([["x1", "x2"]], fn)
    with pytest.raises(ValueError, match="Invalid mode"):
        extract_pareto_interactions([m1], mode="nonexistent")


def test_extract_pareto_from_master_archive():
    from scoredrulesets.estimators.auto import MasterParetoArchive

    fn = ["SNP1", "SNP2", "SNP3"]
    m1 = _make_sample_ruleset([["x0", "x1"]], fn)
    m2 = _make_sample_ruleset([["f0", "f1", "f2"]], fn)

    archive = MasterParetoArchive(higher_is_better=True)
    archive.add(m1, score=0.8, backend="greedy")
    archive.add(m2, score=0.85, backend="cart")

    summary = extract_pareto_interactions(archive, feature_names=fn, backbone_threshold=0.5)
    assert summary.n_models == 2
    assert "SNP1" in summary.backbone_features
    assert "SNP2" in summary.backbone_features
    edges = {(e.feature_a, e.feature_b): e for e in summary.edges}
    assert ("SNP1", "SNP2") in edges
    assert edges[("SNP1", "SNP2")].count == 2
    assert "cart" in edges[("SNP1", "SNP2")].backends
    assert "greedy" in edges[("SNP1", "SNP2")].backends

    # Check TikZ export
    tikz_str = summary.to_tikz(standalone=True)
    assert r"\documentclass[tikz,border=5pt]{standalone}" in tikz_str
    assert "SNP1" in tikz_str
    assert "SNP2" in tikz_str
    assert r"\begin{tikzpicture}" in tikz_str


def test_balanced_pool_epsilon_and_equidistant_tiers():
    from scoredrulesets.estimators.auto import MasterParetoArchive

    fn = ["x0", "x1", "x2"]
    archive = MasterParetoArchive(higher_is_better=True)

    # Backend A: wide complexity spectrum inside epsilon=0.05 (atoms 2..12) + 1 outside (score 0.50)
    for atoms, score in [(1, 0.50), (2, 0.86), (4, 0.88), (6, 0.89), (8, 0.90), (10, 0.87), (12, 0.88)]:
        rs = _make_sample_ruleset([["x0", "x1"]] * max(1, atoms // 2), fn)
        archive.add(rs, score=score, backend="cart")
        # add() only puts non-dominated into candidates_; mirror auto.py candidate_pool_
        from scoredrulesets.estimators.auto import ParetoCandidate
        archive.candidate_pool_.append(
            ParetoCandidate(ruleset=rs, score=score, atoms=atoms, rules=len(rs.rules), backend="cart")
        )

    # Collapsed Backend B below competence floor (best score 0.52)
    rs_low = _make_sample_ruleset([["x1", "x2"]], fn)
    from scoredrulesets.estimators.auto import ParetoCandidate
    archive.candidate_pool_.append(
        ParetoCandidate(ruleset=rs_low, score=0.52, atoms=2, rules=1, backend="weak")
    )

    summary = extract_pareto_interactions(
        archive,
        archive_source="balanced_pool",
        max_models_per_backend=3,
        rashomon_epsilon=0.05,
        min_backend_score=0.60,
        backbone_threshold=0.5,
        min_occurrences=1,
        min_ratio=0.0,
    )
    # Collapsed backend excluded, equidistant selection picks 3 tiers from [2,4,6,8,10,12]
    assert summary.n_models == 3
    assert summary.active_backends == ["cart"]
    assert all(sc >= 0.85 for sc in summary.pool_scores)


def test_laplace_and_provenance_suppress_rare_noise_edges():
    fn = ["A", "B", "N1", "N2"]
    # High-support pair (A, B) shared across 2 backends; rare count=1 noise pair (N1, N2) on 1 backend
    m_cart = _make_sample_ruleset([["A", "B"], ["A", "B"], ["A", "N1"]], fn)
    m_cart.metadata["backend"] = "cart"
    m_hs = _make_sample_ruleset([["A", "B"], ["A", "B"], ["N1", "N2"]], fn)
    m_hs.metadata["backend"] = "hs"

    summary = extract_pareto_interactions(
        [m_cart, m_hs],
        laplace_alpha=1.0,
        normalize_per_model=True,
        provenance_weighting=True,
        backbone_threshold=0.5,
        min_occurrences=1,
        min_ratio=0.0,
    )
    top_edge = summary.edges[0]
    assert (top_edge.feature_a, top_edge.feature_b) == ("A", "B")
    assert set(top_edge.backends) == {"cart", "hs"}


def test_model_normalization_prevents_bloated_tree_dominance():
    fn = ["X1", "X2", "N1", "N2"]
    # Bloated tree with 20 noise rules on N1 & N2
    bloated = _make_sample_ruleset([["N1", "N2"]] * 20, fn)
    bloated.metadata["backend"] = "hs"
    # Compact EC model with 1 rule on X1 & X2
    compact = _make_sample_ruleset([["X1", "X2"]], fn)
    compact.metadata["backend"] = "rulegp"

    summary = extract_pareto_interactions(
        [bloated, compact],
        laplace_alpha=1.0,
        normalize_per_model=True,
        provenance_weighting=True,
        backbone_threshold=0.5,
        min_occurrences=1,
        min_ratio=0.0,
    )
    edges = {(e.feature_a, e.feature_b): e for e in summary.edges}
    # Both models carry identical total mass 1.0 -> equal strength despite 20x rule count difference
    assert edges[("X1", "X2")].strength == pytest.approx(edges[("N1", "N2")].strength, abs=1e-6)


def test_three_way_interaction_lift():
    fn = ["A", "B", "C", "D"]
    # 2 models: each has 1 rule on (A,B,C) and 3 distractor rules on D
    m1 = _make_sample_ruleset([["A", "B", "C"], ["D"], ["D"], ["D"]], fn)
    m2 = _make_sample_ruleset([["A", "B", "C"], ["D"], ["D"], ["D"]], fn)
    m1.metadata["backend"] = "cart"
    m2.metadata["backend"] = "rulegp"

    summary = extract_pareto_interactions(
        [m1, m2],
        backbone_threshold=0.5,
        min_occurrences=1,
        min_ratio=0.0,
    )
    assert len(summary.triplets) == 1
    tr = summary.triplets[0]
    assert tr.features == ("A", "B", "C")
    assert tr.count == 2
    assert tr.lift > 1.0
    assert set(tr.backends) == {"cart", "rulegp"}


def test_mode_b_ochiai_strength_and_rulefit_support():
    import pandas as pd

    class MockRuleFitWithSupport:
        feature_names_in_ = ["X1", "X2", "X3"]

        def _get_rules(self):
            return pd.DataFrame(
                [
                    {"rule": "X1 > 0.5 and X2 > 0.5", "type": "rule", "coef": 4.0, "support": 0.5},
                    {"rule": "X1 > 0.5 and X3 <= 0.0", "type": "rule", "coef": 2.0, "support": 0.25},
                ]
            )

    summary = extract_pareto_interactions(
        MockRuleFitWithSupport(),
        mode="ensemble",
        backbone_threshold=0.4,
        min_occurrences=1,
        min_ratio=0.0,
    )
    # w1 = 4.0 * sqrt(0.5*0.5) = 2.0; w2 = 2.0 * sqrt(0.25*0.75) = sqrt(0.75) ~= 0.866025
    w1 = 2.0
    w2 = 2.0 * (0.25 * 0.75) ** 0.5
    edges = {(e.feature_a, e.feature_b): e for e in summary.edges}
    expected_s12 = w1 / (( (w1 + w2) * w1 ) ** 0.5)
    assert edges[("X1", "X2")].weight == pytest.approx(w1, abs=1e-4)
    assert edges[("X1", "X2")].strength == pytest.approx(expected_s12, abs=1e-4)


def test_rulegp_regressor_pareto_archive():
    from scoredrulesets.estimators import RuleGPRegressor

    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, size=(80, 4))
    y = 10.0 * X[:, 0] * X[:, 1] + 2.0 * X[:, 2] + rng.normal(0, 0.2, size=80)
    reg = RuleGPRegressor(population_size=30, max_generations=10, random_state=42)
    reg.fit(X, y)

    assert hasattr(reg, "final_archive_") and len(reg.final_archive_) >= 1
    assert hasattr(reg, "pareto_archive_") and len(reg.pareto_archive_) >= 1
    summary = reg.extract_pareto_interactions(
        feature_names=["X1", "X2", "X3", "X4"],
        backbone_threshold=0.3,
        min_occurrences=1,
        min_ratio=0.0,
    )
    assert summary.n_models == len(reg.pareto_archive_)
    assert summary.n_models >= 1


def test_skill_relative_competence_gate_and_drop_isolated():
    from scoredrulesets.estimators.auto import MasterParetoArchive, ParetoCandidate

    fn = ["x0", "x1", "x2", "x3"]

    # 1. Classification with chance baseline 0.5 and gamma = 0.5
    arch_cls = MasterParetoArchive(higher_is_better=True)
    rs_strong = _make_sample_ruleset([["x0", "x1"]], fn)
    rs_weak = _make_sample_ruleset([["x2", "x3"]], fn)
    arch_cls.candidate_pool_.append(
        ParetoCandidate(ruleset=rs_strong, score=0.90, atoms=2, rules=1, backend="strong")
    )
    # Gain = 0.65 - 0.50 = 0.15 vs max_gain = 0.40 -> ratio 0.375 < 0.50 -> excluded
    arch_cls.candidate_pool_.append(
        ParetoCandidate(ruleset=rs_weak, score=0.65, atoms=2, rules=1, backend="weak")
    )
    sum_cls = extract_pareto_interactions(
        arch_cls,
        archive_source="balanced_pool",
        min_relative_skill=0.50,
        skill_baseline=0.50,
        backbone_threshold=0.5,
        min_occurrences=1,
        min_ratio=0.0,
    )
    assert sum_cls.active_backends == ["strong"]
    assert sum_cls.n_models == 1

    # 2. Regression (R^2) with baseline 0.0 and gamma = 0.5 + drop_isolated
    arch_reg = MasterParetoArchive(higher_is_better=True)
    rs_reg_best = _make_sample_ruleset([["x0", "x1"], ["x2"]], fn)
    rs_reg_ok = _make_sample_ruleset([["x0", "x1"]], fn)
    rs_reg_poor = _make_sample_ruleset([["x2", "x3"]], fn)
    arch_reg.candidate_pool_.extend(
        [
            ParetoCandidate(ruleset=rs_reg_best, score=0.80, atoms=3, rules=2, backend="greedy_residual"),
            ParetoCandidate(ruleset=rs_reg_ok, score=0.50, atoms=2, rules=1, backend="hs"),
            ParetoCandidate(ruleset=rs_reg_poor, score=0.30, atoms=2, rules=1, backend="rulegp"),
        ]
    )
    sum_reg = extract_pareto_interactions(
        arch_reg,
        archive_source="balanced_pool",
        min_relative_skill=0.50,
        skill_baseline=0.0,
        backbone_threshold=0.5,
        min_occurrences=1,
        min_ratio=0.0,
    )
    assert sum_reg.active_backends == ["greedy_residual", "hs"]
    assert "x2" in sum_reg.backbone_features

    dot_with_isolated = sum_reg.to_dot(max_edges=1, drop_isolated=False)
    dot_no_isolated = sum_reg.to_dot(max_edges=1, drop_isolated=True)
    assert '"x2"' in dot_with_isolated
    assert '"x2"' not in dot_no_isolated
    assert '"x0"' in dot_no_isolated and '"x1"' in dot_no_isolated


def test_multiclass_class_purity_and_score_intensity():
    fn = ["F0", "F1", "F2", "F3", "F4", "F5"]
    rules = [
        # Pure class A high margin
        Rule(atoms=[Atom(feature="F0", op="==", value=1), Atom(feature="F1", op="==", value=1)], scores=[10.0, 0.0, 0.0]),
        # Equally split across all 3 classes -> low purity
        Rule(atoms=[Atom(feature="F2", op="==", value=1), Atom(feature="F3", op="==", value=1)], scores=[3.0, 0.0, 0.0]),
        Rule(atoms=[Atom(feature="F2", op="==", value=0), Atom(feature="F3", op="==", value=1)], scores=[0.0, 3.0, 0.0]),
        Rule(atoms=[Atom(feature="F2", op="==", value=1), Atom(feature="F3", op="==", value=0)], scores=[0.0, 0.0, 3.0]),
        # Low-margin rule for class C
        Rule(atoms=[Atom(feature="F4", op="==", value=1), Atom(feature="F5", op="==", value=1)], scores=[0.1, 0.1, 0.6]),
    ]
    m1 = ScoredRuleSet(class_labels=["A", "B", "C"], rules=rules, feature_names=fn)
    m1.metadata["backend"] = "cart"
    m2 = ScoredRuleSet(class_labels=["A", "B", "C"], rules=rules, feature_names=fn)
    m2.metadata["backend"] = "hs"

    summary = extract_pareto_interactions(
        [m1, m2],
        backbone_threshold=0.5,
        min_occurrences=1,
        min_ratio=0.0,
    )
    assert summary.class_labels == ["A", "B", "C"]
    edges = {(e.feature_a, e.feature_b): e for e in summary.edges}

    pure_edge = edges[("F0", "F1")]
    assert pure_edge.dominant_class == "A"
    assert pure_edge.class_purity == pytest.approx(1.0, abs=1e-6)
    assert pure_edge.score_intensity == pytest.approx(1.0, abs=1e-6)
    assert dict(pure_edge.class_distribution)["A"] == pytest.approx(1.0, abs=1e-6)

    mixed_edge = edges[("F2", "F3")]
    assert mixed_edge.class_purity == pytest.approx(0.0, abs=1e-6)
    assert mixed_edge.score_intensity == pytest.approx(0.3, abs=1e-6)

    low_edge = edges[("F4", "F5")]
    assert low_edge.dominant_class == "C"
    assert low_edge.class_purity == pytest.approx(1.0, abs=1e-6)
    assert low_edge.score_intensity == pytest.approx(0.05, abs=1e-6)

    # Check DOT and TikZ color encodings and validation
    dot_cls = summary.to_dot(color_by="class_score")
    assert "subgraph cluster_legend" in dot_cls
    assert "#0072b2" in dot_cls  # pure class A full color
    assert "[c=A 100%]" in dot_cls

    dot_score = summary.to_dot(color_by="score")
    assert "[m=1.00]" in dot_score
    assert "#1d4ed8" in dot_score

    tikz_cls = summary.to_tikz(color_by="class_score", standalone=True)
    assert r"\definecolor{prinedge0}{HTML}{" in tikz_cls
    assert "draw=prinedge" in tikz_cls

    with pytest.raises(ValueError, match="Invalid color_by"):
        summary.to_dot(color_by="bogus")


def test_score_weighting_cross_backend_scale_invariance():
    fn = ["X1", "X2", "N1", "N2"]
    # Backend 1 uses large logits [100, 0] and [1, 0]; Backend 2 uses tiny logits [0.1, 0] and [0.001, 0]
    m_big = ScoredRuleSet(
        class_labels=[0, 1],
        rules=[
            Rule(atoms=[Atom(feature="X1", op="==", value=1), Atom(feature="X2", op="==", value=1)], scores=[100.0, 0.0]),
            Rule(atoms=[Atom(feature="N1", op="==", value=1), Atom(feature="N2", op="==", value=1)], scores=[1.0, 0.0]),
        ],
        feature_names=fn,
    )
    m_big.metadata["backend"] = "rulegp"

    m_small = ScoredRuleSet(
        class_labels=[0, 1],
        rules=[
            Rule(atoms=[Atom(feature="X1", op="==", value=1), Atom(feature="X2", op="==", value=1)], scores=[0.1, 0.0]),
            Rule(atoms=[Atom(feature="N1", op="==", value=1), Atom(feature="N2", op="==", value=1)], scores=[0.001, 0.0]),
        ],
        feature_names=fn,
    )
    m_small.metadata["backend"] = "cart"

    # Legacy uniform rule mass gives equal strength to (X1, X2) and (N1, N2)
    sum_uniform = extract_pareto_interactions(
        [m_big, m_small],
        score_weighting=False,
        min_occurrences=1,
        min_ratio=0.0,
    )
    edges_u = {(e.feature_a, e.feature_b): e for e in sum_uniform.edges}
    assert edges_u[("X1", "X2")].strength == pytest.approx(edges_u[("N1", "N2")].strength, abs=1e-6)

    # Margin-normalized score weighting boosts high-margin (X1, X2) symmetrically across scales
    sum_weighted = extract_pareto_interactions(
        [m_big, m_small],
        score_weighting=True,
        min_occurrences=1,
        min_ratio=0.0,
    )
    edges_w = {(e.feature_a, e.feature_b): e for e in sum_weighted.edges}
    assert edges_w[("X1", "X2")].strength > edges_w[("N1", "N2")].strength * 10.0

    # Symmetry check if rules are swapped across backends
    m_small_swapped = ScoredRuleSet(
        class_labels=[0, 1],
        rules=[
            Rule(atoms=[Atom(feature="N1", op="==", value=1), Atom(feature="N2", op="==", value=1)], scores=[0.1, 0.0]),
            Rule(atoms=[Atom(feature="X1", op="==", value=1), Atom(feature="X2", op="==", value=1)], scores=[0.001, 0.0]),
        ],
        feature_names=fn,
    )
    m_small_swapped.metadata["backend"] = "cart"
    sum_sym = extract_pareto_interactions(
        [m_big, m_small_swapped],
        score_weighting=True,
        min_occurrences=1,
        min_ratio=0.0,
    )
    edges_sym = {(e.feature_a, e.feature_b): e for e in sum_sym.edges}
    assert edges_sym[("X1", "X2")].strength == pytest.approx(edges_sym[("N1", "N2")].strength, abs=1e-6)


def test_multiclass_wine_auto_estimator_end_to_end():
    from sklearn.datasets import load_wine
    from scoredrulesets.estimators import AutoScoredRuleSetClassifier

    wine = load_wine()
    clf = AutoScoredRuleSetClassifier(
        candidate_backends=["cart", "hs", "greedy"],
        scoring="balanced_accuracy",
        feature_selection="none",
        cv=3,
        random_state=42,
    )
    clf.fit(wine.data, wine.target)
    summary = extract_pareto_interactions(
        clf,
        mode="pareto",
        feature_names=list(wine.feature_names),
        archive_source="balanced_pool",
        skill_baseline=1.0 / 3.0,
        min_relative_skill=0.50,
        backbone_threshold=0.30,
        min_occurrences=1,
        min_ratio=0.05,
        score_weighting=True,
    )
    assert len(summary.class_labels) == 3
    assert len(summary.edges) > 0
    assert any(e.dominant_class is not None for e in summary.edges)
    assert all(0.0 <= e.class_purity <= 1.0 for e in summary.edges)
    assert all(0.0 <= e.score_intensity <= 1.0 for e in summary.edges)
    dot = summary.to_dot(color_by="class_score")
    assert "cluster_legend" in dot





