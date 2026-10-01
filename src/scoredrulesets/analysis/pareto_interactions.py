"""Pareto front and Rashomon set interaction analysis for rule-based estimators.

Inspired by Cynthia Rudin's research on the Rashomon effect, model multiplicity,
and variable importance clouds, as well as the GPASInteractions methodology
(Nunkesser et al. 2007).

Analyzes the structural anatomy of an entire Pareto front (or collection of
near-optimal models / population):
1. **Feature Backbone:** Identification of core features that are persistently
   included across models of varying complexity.
2. **Conjunctive Interactions:** Pairwise co-occurrences of features within
   the same rule or conjunctive monomial.
3. **Multi-format Exports:** GraphViz DOT graphs (with node styles for backbone
   features and weighted edges for interaction strength), CSV, and Markdown tables.
"""

from __future__ import annotations

import csv
import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class ParetoInteractionEdge:
    """A pairwise feature interaction identified across rules."""

    feature_a: str
    feature_b: str
    count: int
    ratio_a: float
    ratio_b: float
    strength: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_a": self.feature_a,
            "feature_b": self.feature_b,
            "count": self.count,
            "ratio_a": round(self.ratio_a, 4),
            "ratio_b": round(self.ratio_b, 4),
            "strength": round(self.strength, 4),
        }


@dataclass
class ParetoFrontSummary:
    """Compact summary of feature importance and interactions across a Pareto front."""

    n_models: int
    feature_counts: dict[str, int]
    model_feature_frequencies: dict[str, float]
    backbone_features: list[str]
    peripheral_features: list[str]
    edges: list[ParetoInteractionEdge]
    feature_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_models": self.n_models,
            "backbone_features": self.backbone_features,
            "peripheral_features": self.peripheral_features,
            "feature_counts": self.feature_counts,
            "model_feature_frequencies": {
                k: round(v, 4) for k, v in self.model_feature_frequencies.items()
            },
            "edges": [e.to_dict() for e in self.edges],
        }

    def to_dot(self, title: str = "Pareto Interaction Graph") -> str:
        """Generate a GraphViz DOT graph string."""
        lines = [
            f'graph "{title}" {{',
            "  rankdir=LR;",
            '  node [fontname="Helvetica", fontsize=11];',
            '  edge [fontname="Helvetica", fontsize=9];',
        ]

        active_nodes = set(self.backbone_features)
        for e in self.edges:
            active_nodes.add(e.feature_a)
            active_nodes.add(e.feature_b)

        for name in sorted(active_nodes):
            is_backbone = name in self.backbone_features
            freq = self.model_feature_frequencies.get(name, 0.0)
            cnt = self.feature_counts.get(name, 0)
            lbl = f"{name}\\n({freq:.0%}, {cnt}r)"
            safe = _dot_id(name)
            if is_backbone:
                lines.append(
                    f'  {safe} [label="{lbl}", shape=box, style="filled,rounded", '
                    f'fillcolor="#d4edda", color="#28a745", penwidth=2.0];'
                )
            else:
                lines.append(
                    f'  {safe} [label="{lbl}", shape=ellipse, style=filled, '
                    f'fillcolor="#f8f9fa", color="#6c757d"];'
                )

        for e in self.edges:
            a = _dot_id(e.feature_a)
            b = _dot_id(e.feature_b)
            pw = max(1.0, min(5.0, 1.0 + 4.0 * e.strength))
            lines.append(
                f'  {a} -- {b} [label="{e.count}", penwidth={pw:.1f}, color="#495057"];'
            )

        lines.append("}\n")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Format a human-readable Markdown summary table."""
        lines = [
            f"### Pareto Front Interaction Summary ({self.n_models} models)",
            "",
            f"- **Backbone features:** {', '.join(f'`{f}`' for f in self.backbone_features) if self.backbone_features else '*(none)*'}",
            f"- **Peripheral features:** {len(self.peripheral_features)}",
            f"- **Discovered interactions:** {len(self.edges)}",
            "",
            "#### Feature Inclusion & Backbone Anatomy",
            "",
            "| Feature | Model Presence | Rule Count | Role |",
            "|---|---|---|---|",
        ]

        all_feats = sorted(
            self.model_feature_frequencies.keys(),
            key=lambda f: (-self.model_feature_frequencies[f], -self.feature_counts[f], f),
        )
        for f in all_feats:
            freq = self.model_feature_frequencies[f]
            cnt = self.feature_counts[f]
            role = "**Core Backbone**" if f in self.backbone_features else "Peripheral"
            lines.append(f"| `{f}` | {freq:.1%} | {cnt} | {role} |")

        if self.edges:
            lines.extend(
                [
                    "",
                    "#### Top Feature Interactions (Co-occurrences in Rules)",
                    "",
                    "| Feature A | Feature B | Co-occurrences | Strength | Coverage Ratios |",
                    "|---|---|---|---|---|",
                ]
            )
            for e in self.edges:
                lines.append(
                    f"| `{e.feature_a}` | `{e.feature_b}` | {e.count} | {e.strength:.2f} | "
                    f"A: {e.ratio_a:.1%}, B: {e.ratio_b:.1%} |"
                )

        return "\n".join(lines)


def _dot_id(name: str) -> str:
    """Return a safely quoted node identifier for GraphViz DOT."""
    escaped = name.replace('"', '\\"')
    return f'"{escaped}"'


def _extract_rules_from_single_model(
    model: Any, feature_names: Sequence[str] | None = None
) -> list[set[str]]:
    """Extract set of feature names per rule from a model or ruleset."""
    rules_feats: list[set[str]] = []

    # 1. Rule sets with a list of rules (ScoredRuleSet or RuleGP _RuleSet2)
    if hasattr(model, "rules") and isinstance(model.rules, list):
        for rule in model.rules:
            feats = set()
            for atom in getattr(rule, "atoms", []):
                feat = getattr(atom, "feature", None)
                if feat is not None:
                    if feature_names and isinstance(feat, int) and feat < len(feature_names):
                        feats.add(str(feature_names[feat]))
                    elif (
                        feature_names
                        and isinstance(feat, str)
                        and feat.startswith("f")
                        and feat[1:].isdigit()
                    ):
                        idx = int(feat[1:])
                        if idx < len(feature_names):
                            feats.add(str(feature_names[idx]))
                        else:
                            feats.add(str(feat))
                    else:
                        feats.add(str(feat))
                else:
                    f_idx = getattr(atom, "feature_idx", None)
                    if f_idx is not None:
                        name = (
                            feature_names[f_idx]
                            if feature_names and f_idx < len(feature_names)
                            else f"f{f_idx}"
                        )
                        feats.add(str(name))
            if feats:
                rules_feats.append(feats)
        return rules_feats

    # 2. LogicGP polynomial / individual (has 'monomials' with 'literals')
    poly = model[0] if isinstance(model, tuple) else model
    if hasattr(poly, "monomials"):
        for monomial in getattr(poly, "monomials", []):
            feats = set()
            for lit in getattr(monomial, "literals", []):
                f_idx = getattr(lit, "feature_idx", None)
                if f_idx is not None:
                    name = (
                        feature_names[f_idx]
                        if feature_names and f_idx < len(feature_names)
                        else f"f{f_idx}"
                    )
                    feats.add(str(name))
            if feats:
                rules_feats.append(feats)
        return rules_feats

    return rules_feats


def _collect_models(estimator_or_models: Any) -> tuple[list[Any], list[str] | None]:
    """Collect individual models and feature names from an estimator or sequence."""
    feature_names: list[str] | None = None

    # Handle sequence / collection of models
    if isinstance(estimator_or_models, (list, tuple)):
        # Check if first element has feature_names_in_
        for item in estimator_or_models:
            if hasattr(item, "feature_names_in_"):
                feature_names = [str(f) for f in item.feature_names_in_]
                break
        return list(estimator_or_models), feature_names

    # Estimator object
    est = estimator_or_models
    if hasattr(est, "feature_names_in_"):
        feature_names = [str(f) for f in est.feature_names_in_]

    # RuleGP final_archive_ (Pareto front)
    if hasattr(est, "final_archive_") and est.final_archive_:
        return list(est.final_archive_), feature_names

    # LogicGP _final_population
    if hasattr(est, "_final_population") and est._final_population:
        return list(est._final_population), feature_names

    # Standard fitted estimator with ruleset_
    if hasattr(est, "ruleset_") and est.ruleset_ is not None:
        if not feature_names and getattr(est.ruleset_, "feature_names", None):
            feature_names = [str(f) for f in est.ruleset_.feature_names]
        return [est.ruleset_], feature_names

    # Single ScoredRuleSet
    if hasattr(est, "rules"):
        if not feature_names and getattr(est, "feature_names", None):
            feature_names = [str(f) for f in est.feature_names]
        return [est], feature_names

    raise ValueError(
        "Could not extract models from the provided object. "
        "Expected a fitted RuleGPClassifier/LogicGPClassifier, a ScoredRuleSet, "
        "or a sequence of models."
    )


def extract_pareto_interactions(
    estimator_or_models: Any,
    *,
    feature_names: Sequence[str] | None = None,
    min_occurrences: int = 1,
    min_ratio: float = 0.05,
    backbone_threshold: float = 0.5,
    out_dot: str | Path | None = None,
    out_csv: str | Path | None = None,
) -> ParetoFrontSummary:
    """Analyze feature inclusions and interactions across a Pareto front or ensemble.

    Parameters
    ----------
    estimator_or_models:
        A fitted estimator with a Pareto archive (`RuleGPClassifier` with
        `final_archive_` or `LogicGPClassifier` with `_final_population`),
        an estimator with a fitted `ruleset_`, a single `ScoredRuleSet`, or a list
        of model instances.
    feature_names:
        Optional sequence of feature names. If omitted, inferred from the
        estimator's `feature_names_in_` or rule specifications.
    min_occurrences:
        Minimum number of times a feature pair must co-occur in the same rule.
    min_ratio:
        Minimum ratio of pair count relative to the smaller individual feature
        count (`pair_count / min(count_a, count_b)`).
    backbone_threshold:
        Fraction of models (between 0.0 and 1.0) in which a feature must appear
        to be classified as a core "backbone" feature. Default is 0.5 (50%).
    out_dot:
        Optional file path to write a GraphViz DOT representation.
    out_csv:
        Optional file path to write an interactions CSV.

    Returns
    -------
    ParetoFrontSummary
        Structured object containing backbone features, interaction edges,
        frequencies, and export methods (`to_dot`, `to_markdown`, `to_dict`).
    """
    models, inferred_names = _collect_models(estimator_or_models)
    if feature_names is None:
        feature_names = inferred_names

    n_models = len(models)
    if n_models == 0:
        return ParetoFrontSummary(
            n_models=0,
            feature_counts={},
            model_feature_frequencies={},
            backbone_features=[],
            peripheral_features=[],
            edges=[],
            feature_names=[],
        )

    # Track feature presence across models and rules
    model_presence: dict[str, int] = defaultdict(int)
    feature_rule_counts: dict[str, int] = defaultdict(int)
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)

    all_discovered_features: set[str] = set()

    for model in models:
        rules_feats = _extract_rules_from_single_model(model, feature_names)
        model_feats: set[str] = set()

        for feats in rules_feats:
            model_feats.update(feats)
            all_discovered_features.update(feats)
            for f in feats:
                feature_rule_counts[f] += 1

            for fa, fb in itertools.combinations(sorted(feats), 2):
                pair_counts[(fa, fb)] += 1

        for f in model_feats:
            model_presence[f] += 1

    # Compute model frequencies and backbone
    frequencies = {f: model_presence[f] / float(n_models) for f in all_discovered_features}
    backbone = [
        f for f in sorted(all_discovered_features) if frequencies.get(f, 0.0) >= backbone_threshold
    ]
    backbone.sort(key=lambda f: (-frequencies[f], -feature_rule_counts[f], f))

    peripheral = [
        f for f in sorted(all_discovered_features) if frequencies.get(f, 0.0) < backbone_threshold
    ]
    peripheral.sort(key=lambda f: (-frequencies[f], -feature_rule_counts[f], f))

    # Build interaction edges
    edges: list[ParetoInteractionEdge] = []
    for (fa, fb), count in pair_counts.items():
        if count < min_occurrences:
            continue
        cnt_a = feature_rule_counts[fa]
        cnt_b = feature_rule_counts[fb]
        ratio_a = count / cnt_a if cnt_a > 0 else 0.0
        ratio_b = count / cnt_b if cnt_b > 0 else 0.0

        if max(ratio_a, ratio_b) < min_ratio:
            continue

        # Harmonic mean of coverage ratios as interaction strength
        strength = (
            2.0 * (ratio_a * ratio_b) / (ratio_a + ratio_b)
            if (ratio_a + ratio_b) > 0
            else 0.0
        )
        edges.append(
            ParetoInteractionEdge(
                feature_a=fa,
                feature_b=fb,
                count=count,
                ratio_a=ratio_a,
                ratio_b=ratio_b,
                strength=strength,
            )
        )

    # Sort edges by strength descending, then count descending
    edges.sort(key=lambda e: (-e.strength, -e.count, e.feature_a, e.feature_b))

    summary = ParetoFrontSummary(
        n_models=n_models,
        feature_counts=dict(feature_rule_counts),
        model_feature_frequencies=frequencies,
        backbone_features=backbone,
        peripheral_features=peripheral,
        edges=edges,
        feature_names=sorted(all_discovered_features),
    )

    if out_dot is not None:
        dot_path = Path(out_dot)
        dot_path.parent.mkdir(parents=True, exist_ok=True)
        dot_path.write_text(summary.to_dot(), encoding="utf-8")

    if out_csv is not None:
        csv_path = Path(out_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["feature_a", "feature_b", "count", "ratio_a", "ratio_b", "strength"]
            )
            for e in edges:
                writer.writerow(
                    [
                        e.feature_a,
                        e.feature_b,
                        e.count,
                        f"{e.ratio_a:.4f}",
                        f"{e.ratio_b:.4f}",
                        f"{e.strength:.4f}",
                    ]
                )

    return summary
