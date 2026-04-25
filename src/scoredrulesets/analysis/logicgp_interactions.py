"""
Interaction analysis for fitted LogicGP estimators.

Mirrors the GPASInteractions / setInteractionR postprocessor from the original
R/Java FREAK implementation (Nunkesser et al. 2007): counts co-occurrences of
feature pairs within the same monomial across all individuals in the final GP
population, then filters by a minimum count and a minimum ratio relative to
each feature's individual occurrence count.

Works for any fitted ``LogicGPClassifier`` (and its subclasses such as
``GPASClassifier``).
"""

from __future__ import annotations

import csv
import itertools
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from scoredrulesets.estimators.logicgp import LogicGPClassifier


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def extract_interactions(
    estimator: "LogicGPClassifier",
    *,
    min_occurrences: int = 10,
    min_ratio: float = 0.1,
    out_dot: Optional[Union[str, Path]] = None,
    out_csv: Optional[Union[str, Path]] = None,
) -> dict:
    """Extract pairwise feature interactions from a fitted LogicGP estimator.

    Parameters
    ----------
    estimator:
        A fitted ``LogicGPClassifier`` (or subclass).  Must have
        ``_final_population`` and ``feature_names_in_`` set by ``fit()``.
    min_occurrences:
        Minimum number of times a feature pair must co-occur in the same
        monomial across the final population.  Mirrors the ``occurences``
        parameter of ``GPASInteractions`` / ``setInteractionR``.
    min_ratio:
        Minimum ratio of the pair count to the individual feature count
        (``pair_count / min(count_i, count_j)``).  Mirrors the ``ratio``
        parameter of ``GPASInteractions``.
    out_dot:
        Optional path for a GraphViz DOT file.  Written only when provided.
    out_csv:
        Optional path for a CSV file with columns
        ``feature_a, feature_b, count, ratio_a, ratio_b``.
        Written only when provided.

    Returns
    -------
    dict with keys:
        ``edges``   – list of dicts, each with keys
                      ``feature_a``, ``feature_b``, ``count``,
                      ``ratio_a``, ``ratio_b``
        ``feature_counts`` – dict mapping feature name → occurrence count
        ``pair_counts``    – dict mapping (name_a, name_b) → count
    """
    if not hasattr(estimator, "_final_population"):
        raise ValueError(
            "estimator has no '_final_population' attribute. "
            "Call fit() before extract_interactions()."
        )

    feature_names = estimator.feature_names_in_

    # --- count occurrences -------------------------------------------------
    feature_count: dict[int, int] = defaultdict(int)
    pair_count: dict[tuple[int, int], int] = defaultdict(int)

    for entry in estimator._final_population:
        # _final_population contains (polynomial, fitness) tuples.
        polynomial = entry[0] if isinstance(entry, tuple) else entry
        for monomial in polynomial.monomials:
            feat_indices = [lit.feature_idx for lit in monomial.literals]
            # Individual feature counts
            for fi in feat_indices:
                feature_count[fi] += 1
            # Pair counts (ordered, smaller index first)
            for fi, fj in itertools.combinations(sorted(set(feat_indices)), 2):
                pair_count[(fi, fj)] += 1

    # --- filter ------------------------------------------------------------
    edges: list[dict] = []
    for (fi, fj), count in pair_count.items():
        if count < min_occurrences:
            continue
        ratio_a = count / feature_count[fi] if feature_count[fi] else 0.0
        ratio_b = count / feature_count[fj] if feature_count[fj] else 0.0
        if max(ratio_a, ratio_b) < min_ratio:
            continue
        edges.append(
            {
                "feature_a": str(feature_names[fi]),
                "feature_b": str(feature_names[fj]),
                "count": count,
                "ratio_a": ratio_a,
                "ratio_b": ratio_b,
            }
        )

    # --- build named maps for return value ---------------------------------
    named_feature_counts = {
        str(feature_names[fi]): cnt for fi, cnt in feature_count.items()
    }
    named_pair_counts = {
        (str(feature_names[fi]), str(feature_names[fj])): cnt
        for (fi, fj), cnt in pair_count.items()
    }

    # --- optional DOT output -----------------------------------------------
    if out_dot is not None:
        _write_dot(edges, named_feature_counts, Path(out_dot))

    # --- optional CSV output -----------------------------------------------
    if out_csv is not None:
        _write_csv(edges, Path(out_csv))

    return {
        "edges": edges,
        "feature_counts": named_feature_counts,
        "pair_counts": named_pair_counts,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_dot(
    edges: list[dict],
    feature_counts: dict[str, int],
    path: Path,
) -> None:
    """Write a GraphViz DOT file from the filtered interaction edges."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Collect all node names that appear in at least one edge.
    node_names: set[str] = set()
    for e in edges:
        node_names.add(e["feature_a"])
        node_names.add(e["feature_b"])

    with path.open("w", encoding="utf-8") as fh:
        fh.write("graph interactions {\n")
        fh.write("  node [shape=ellipse];\n")
        for name in sorted(node_names):
            cnt = feature_counts.get(name, 0)
            safe = _dot_id(name)
            fh.write(f'  {safe} [label="{name}\\n({cnt})"];\n')
        for e in edges:
            a = _dot_id(e["feature_a"])
            b = _dot_id(e["feature_b"])
            cnt = e["count"]
            fh.write(f"  {a} -- {b} [label={cnt}];\n")
        fh.write("}\n")


def _write_csv(edges: list[dict], path: Path) -> None:
    """Write the filtered edges as a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["feature_a", "feature_b", "count", "ratio_a", "ratio_b"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(edges)


def _dot_id(name: str) -> str:
    """Return a safe DOT node identifier (quoted)."""
    escaped = name.replace('"', '\\"')
    return f'"{escaped}"'
