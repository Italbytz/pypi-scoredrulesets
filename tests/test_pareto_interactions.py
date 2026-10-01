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

