"""Tests for LogicGP interaction analysis (extract_interactions)."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest


def _make_simple_data():
    rng = np.random.default_rng(42)
    X = rng.integers(0, 3, size=(80, 6)).astype(object)
    y = np.array([0] * 40 + [1] * 40)
    return X, y


def _fit_logicgp(X, y):
    from scoredrulesets import LogicGPClassifier

    clf = LogicGPClassifier(
        trainer="rlcw",
        max_generations=20,
        stagnation_generations=5,
        n_bins=3,
        random_state=0,
    )
    clf.fit(X, y)
    return clf


def _fit_gpas(X, y):
    from scoredrulesets.estimators.logicgp import GPASClassifier

    clf = GPASClassifier(
        max_generations=20,
        stagnation_generations=5,
        n_bins=3,
        random_state=0,
    )
    clf.fit(X, y)
    return clf


# ---------------------------------------------------------------------------
# Basic return structure
# ---------------------------------------------------------------------------

def test_extract_interactions_returns_dict():
    X, y = _make_simple_data()
    clf = _fit_logicgp(X, y)

    result = clf.extract_interactions(min_occurrences=1, min_ratio=0.0)

    assert isinstance(result, dict)
    assert "edges" in result
    assert "feature_counts" in result
    assert "pair_counts" in result


def test_extract_interactions_edge_structure():
    X, y = _make_simple_data()
    clf = _fit_logicgp(X, y)

    result = clf.extract_interactions(min_occurrences=1, min_ratio=0.0)

    for edge in result["edges"]:
        assert "feature_a" in edge
        assert "feature_b" in edge
        assert "count" in edge
        assert "ratio_a" in edge
        assert "ratio_b" in edge
        assert edge["count"] >= 1
        assert 0.0 <= edge["ratio_a"] <= 1.0
        assert 0.0 <= edge["ratio_b"] <= 1.0


def test_feature_names_in_edges():
    X, y = _make_simple_data()
    clf = _fit_logicgp(X, y)

    result = clf.extract_interactions(min_occurrences=1, min_ratio=0.0)

    valid_names = set(clf.feature_names_in_)
    for edge in result["edges"]:
        assert edge["feature_a"] in valid_names
        assert edge["feature_b"] in valid_names


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def test_high_min_occurrences_produces_fewer_edges():
    X, y = _make_simple_data()
    clf = _fit_logicgp(X, y)

    result_loose = clf.extract_interactions(min_occurrences=1, min_ratio=0.0)
    result_strict = clf.extract_interactions(min_occurrences=1000, min_ratio=0.0)

    assert len(result_strict["edges"]) <= len(result_loose["edges"])


def test_high_min_ratio_produces_fewer_edges():
    X, y = _make_simple_data()
    clf = _fit_logicgp(X, y)

    result_loose = clf.extract_interactions(min_occurrences=1, min_ratio=0.0)
    result_strict = clf.extract_interactions(min_occurrences=1, min_ratio=1.1)

    assert len(result_strict["edges"]) == 0
    assert len(result_loose["edges"]) >= 0


# ---------------------------------------------------------------------------
# Not-fitted guard
# ---------------------------------------------------------------------------

def test_extract_interactions_before_fit_raises():
    from scoredrulesets import LogicGPClassifier
    from scoredrulesets.analysis import extract_interactions

    clf = LogicGPClassifier()
    with pytest.raises((ValueError, AttributeError)):
        extract_interactions(clf)


# ---------------------------------------------------------------------------
# Works on GPASClassifier (subclass)
# ---------------------------------------------------------------------------

def test_gpas_extract_interactions():
    X, y = _make_simple_data()
    clf = _fit_gpas(X, y)

    result = clf.extract_interactions(min_occurrences=1, min_ratio=0.0)
    assert "edges" in result


# ---------------------------------------------------------------------------
# File output: DOT
# ---------------------------------------------------------------------------

def test_extract_interactions_writes_dot():
    X, y = _make_simple_data()
    clf = _fit_logicgp(X, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        dot_path = os.path.join(tmpdir, "interactions.dot")
        clf.extract_interactions(
            min_occurrences=1, min_ratio=0.0, out_dot=dot_path
        )
        assert os.path.isfile(dot_path)
        content = open(dot_path).read()
        assert "graph interactions" in content


# ---------------------------------------------------------------------------
# File output: CSV
# ---------------------------------------------------------------------------

def test_extract_interactions_writes_csv():
    X, y = _make_simple_data()
    clf = _fit_logicgp(X, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "interactions.csv")
        clf.extract_interactions(
            min_occurrences=1, min_ratio=0.0, out_csv=csv_path
        )
        assert os.path.isfile(csv_path)
        with open(csv_path) as fh:
            header = fh.readline().strip()
        assert header == "feature_a,feature_b,count,ratio_a,ratio_b"


# ---------------------------------------------------------------------------
# Standalone function from analysis module
# ---------------------------------------------------------------------------

def test_standalone_extract_interactions():
    from scoredrulesets.analysis import extract_interactions

    X, y = _make_simple_data()
    clf = _fit_logicgp(X, y)

    result = extract_interactions(clf, min_occurrences=1, min_ratio=0.0)
    assert isinstance(result, dict)
    assert "edges" in result
