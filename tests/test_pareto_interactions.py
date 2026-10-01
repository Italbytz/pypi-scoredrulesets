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
