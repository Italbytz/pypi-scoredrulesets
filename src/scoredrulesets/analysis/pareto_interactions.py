"""Pareto front and Rashomon set interaction analysis for rule-based estimators.

Inspired by Cynthia Rudin's research on the Rashomon effect, model multiplicity,
and variable importance clouds, as well as the GPASInteractions methodology
(Nunkesser et al. 2007).

Analyzes the structural anatomy of rule-based models in two complementary modes:
1. **Mode A: Inter-Model Consensus (``mode="pareto"``)**
   Analyzes an entire Pareto front or Rashomon set of models (e.g., from
   ``RuleGPClassifier``, ``LogicGPClassifier``, or a collection of rule sets).
   - Core Backbone: Features appearing persistently across models of varying
     complexity (``model_frequency >= backbone_threshold``).
   - Conjunctive Interactions: Pairwise co-occurrences of features within rules
     across the archive models.
2. **Mode B: Intra-Ensemble Distillation (``mode="ensemble"``)**
   Compresses a large ensemble of rules from a single model (e.g., ``RuleFitClassifier``
   from imodels with Lasso-weighted basis functions, ``ExSTraCS`` with rule populations,
   or a single dense ``ScoredRuleSet``) into an interpretable interaction network.
   - Core Backbone: Features with high rule frequency or concentrated coefficient/score
     weight (``rule_frequency >= backbone_threshold`` or relative weight threshold).
   - Epistatic Interactions: Weighted co-occurrences of features within conjunctive rules.
"""

from __future__ import annotations

import csv
import itertools
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


DEFAULT_CLASS_PALETTE: tuple[str, ...] = (
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#8C564B",  # brown
    "#009E73",  # bluish green
    "#332288",  # indigo
)
_VALID_COLOR_BY: set[str | None] = {None, "score", "class", "class_score"}


def _validate_color_by(color_by: str | None) -> None:
    if color_by not in _VALID_COLOR_BY:
        raise ValueError(
            f"Invalid color_by='{color_by}'. Expected None, 'score', 'class', or 'class_score'."
        )


def _blend_hex(hex_low: str, hex_high: str, t: float) -> str:
    """Linearly interpolate between two ``#rrggbb`` hex colors for ``t`` in ``[0, 1]``."""
    alpha = max(0.0, min(1.0, float(t)))
    c0 = hex_low.lstrip("#")
    c1 = hex_high.lstrip("#")
    r0, g0, b0 = int(c0[0:2], 16), int(c0[2:4], 16), int(c0[4:6], 16)
    r1, g1, b1 = int(c1[0:2], 16), int(c1[2:4], 16), int(c1[4:6], 16)
    r = round(r0 + (r1 - r0) * alpha)
    g = round(g0 + (g1 - g0) * alpha)
    b = round(b0 + (b1 - b0) * alpha)
    return f"#{r:02x}{g:02x}{b:02x}"


def _compute_class_purity(dist: dict[str, float], n_classes: int) -> float:
    """Compute normalized inverse Shannon entropy in ``[0, 1]`` (1 = single class)."""
    if not dist:
        return 0.0
    k = max(int(n_classes), len(dist), 2)
    probs = [p for p in dist.values() if p > 0.0]
    if len(probs) <= 1:
        return 1.0
    entropy = -sum(p * math.log(p) for p in probs)
    max_entropy = math.log(float(k))
    if max_entropy <= 0.0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - (entropy / max_entropy)))


def _build_class_color_map(
    class_labels: Sequence[str],
    class_palette: Sequence[str] | None = None,
) -> dict[str, str]:
    palette = tuple(class_palette) if class_palette else DEFAULT_CLASS_PALETTE
    return {
        str(lbl): palette[idx % len(palette)]
        for idx, lbl in enumerate(class_labels)
    }


def _finalize_class_stats(
    class_mass_map: dict[str, float],
    ordered_labels: Sequence[str],
) -> tuple[str | None, float, dict[str, float]]:
    """Normalize per-class mass into (dominant_class, purity, distribution)."""
    total = sum(v for v in class_mass_map.values() if v > 0.0)
    if total <= 1e-12:
        return None, 0.0, {}
    order_index = {lbl: idx for idx, lbl in enumerate(ordered_labels)}
    ordered_keys = sorted(
        (k for k, v in class_mass_map.items() if v > 0.0),
        key=lambda k: (order_index.get(k, len(order_index)), str(k)),
    )
    dist = {k: class_mass_map[k] / total for k in ordered_keys}
    dom_class = max(
        ordered_keys,
        key=lambda k: (dist[k], -order_index.get(k, len(order_index))),
    )
    purity = _compute_class_purity(dist, len(ordered_labels) or len(ordered_keys))
    return dom_class, purity, dist


def _resolve_edge_color(
    edge: "ParetoInteractionEdge",
    color_by: str | None,
    class_color_map: dict[str, str],
) -> str:
    """Resolve edge stroke color according to ``color_by``."""
    _validate_color_by(color_by)
    if color_by is None:
        return "#495057"
    if color_by == "score":
        return _blend_hex("#cbd5e1", "#1d4ed8", edge.score_intensity)
    if edge.dominant_class is not None and edge.dominant_class in class_color_map:
        base_hex = class_color_map[edge.dominant_class]
        if color_by == "class":
            return _blend_hex("#9ca3af", base_hex, edge.class_purity)
        t = edge.class_purity * (0.35 + 0.65 * edge.score_intensity)
        return _blend_hex("#9ca3af", base_hex, t)
    if color_by == "class_score" and edge.score_intensity > 0.0:
        return _blend_hex("#cbd5e1", "#1d4ed8", edge.score_intensity)
    return "#6c757d"


@dataclass(frozen=True)
class ParetoInteractionEdge:
    """A pairwise feature interaction identified across rules."""

    feature_a: str
    feature_b: str
    count: int
    ratio_a: float
    ratio_b: float
    strength: float
    weight: float = 0.0
    backends: tuple[str, ...] = ()
    score_intensity: float = 0.0
    dominant_class: str | None = None
    class_purity: float = 0.0
    class_distribution: tuple[tuple[str, float], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        res: dict[str, Any] = {
            "feature_a": self.feature_a,
            "feature_b": self.feature_b,
            "count": self.count,
            "ratio_a": round(self.ratio_a, 4),
            "ratio_b": round(self.ratio_b, 4),
            "strength": round(self.strength, 4),
        }
        if self.weight > 0.0:
            res["weight"] = round(self.weight, 4)
        if self.backends:
            res["backends"] = list(self.backends)
        if self.score_intensity > 0.0:
            res["score_intensity"] = round(self.score_intensity, 4)
        if self.dominant_class is not None:
            res["dominant_class"] = self.dominant_class
            res["class_purity"] = round(self.class_purity, 4)
        if self.class_distribution:
            res["class_distribution"] = {
                str(cls): round(prob, 4) for cls, prob in self.class_distribution
            }
        return res


@dataclass(frozen=True)
class ParetoInteractionTriplet:
    """A 3-way higher-order feature interaction with Interaction Lift."""

    features: tuple[str, str, str]
    count: int
    mass: float
    lift: float
    backends: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        res: dict[str, Any] = {
            "features": list(self.features),
            "count": self.count,
            "mass": round(self.mass, 4),
            "lift": round(self.lift, 4),
        }
        if self.backends:
            res["backends"] = list(self.backends)
        return res


@dataclass
class ParetoFrontSummary:
    """Compact summary of feature importance and interactions across a Pareto front or ensemble."""

    n_models: int
    feature_counts: dict[str, int]
    model_feature_frequencies: dict[str, float]
    backbone_features: list[str]
    peripheral_features: list[str]
    edges: list[ParetoInteractionEdge]
    feature_names: list[str] = field(default_factory=list)
    mode: str = "pareto"
    n_rules: int = 0
    feature_weights: dict[str, float] = field(default_factory=dict)
    pair_weights: dict[tuple[str, str], float] = field(default_factory=dict)
    triplets: list[ParetoInteractionTriplet] = field(default_factory=list)
    active_backends: list[str] = field(default_factory=list)
    pool_scores: list[float] = field(default_factory=list)
    class_labels: list[str] = field(default_factory=list)
    feature_score_intensity: dict[str, float] = field(default_factory=dict)
    feature_dominant_class: dict[str, str] = field(default_factory=dict)
    feature_class_purity: dict[str, float] = field(default_factory=dict)
    feature_class_distribution: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        res: dict[str, Any] = {
            "mode": self.mode,
            "n_models": self.n_models,
            "n_rules": self.n_rules,
            "backbone_features": self.backbone_features,
            "peripheral_features": self.peripheral_features,
            "feature_counts": self.feature_counts,
            "model_feature_frequencies": {
                k: round(v, 4) for k, v in self.model_feature_frequencies.items()
            },
            "edges": [e.to_dict() for e in self.edges],
        }
        if self.feature_weights:
            res["feature_weights"] = {
                k: round(v, 4) for k, v in self.feature_weights.items()
            }
        if self.triplets:
            res["triplets"] = [t.to_dict() for t in self.triplets]
        if self.active_backends:
            res["active_backends"] = list(self.active_backends)
        if self.class_labels:
            res["class_labels"] = list(self.class_labels)
        if self.feature_score_intensity:
            res["feature_score_intensity"] = {
                k: round(v, 4) for k, v in self.feature_score_intensity.items()
            }
        if self.feature_dominant_class:
            res["feature_dominant_class"] = dict(self.feature_dominant_class)
        if self.feature_class_purity:
            res["feature_class_purity"] = {
                k: round(v, 4) for k, v in self.feature_class_purity.items()
            }
        if self.feature_class_distribution:
            res["feature_class_distribution"] = {
                f: {c: round(p, 4) for c, p in dist.items()}
                for f, dist in self.feature_class_distribution.items()
            }
        return res

    def to_dot(
        self,
        title: str | None = None,
        *,
        max_edges: int | None = None,
        drop_isolated: bool = False,
        fontname: str = "Arial",
        fontsize_node: int = 11,
        fontsize_edge: int = 9,
        nodesep: float = 0.5,
        ranksep: float = 0.7,
        color_by: str | None = None,
        class_palette: Sequence[str] | None = None,
    ) -> str:
        """Generate a GraphViz DOT graph string."""
        _validate_color_by(color_by)
        if title is None:
            title = (
                "Pareto Interaction Graph"
                if self.mode == "pareto"
                else "Intra-Ensemble Interaction Graph"
            )
        lines = [
            f'graph "{title}" {{',
            "  rankdir=LR;",
            f"  nodesep={nodesep};",
            f"  ranksep={ranksep};",
            f'  node [fontname="{fontname}", fontsize={fontsize_node}];',
            f'  edge [fontname="{fontname}", fontsize={fontsize_edge}];',
        ]

        edges_to_draw = self.edges[:max_edges] if max_edges is not None else self.edges

        if drop_isolated and edges_to_draw:
            active_nodes: set[str] = set()
        else:
            active_nodes = set(self.backbone_features)
        for e in edges_to_draw:
            active_nodes.add(e.feature_a)
            active_nodes.add(e.feature_b)

        for name in sorted(active_nodes):
            is_backbone = name in self.backbone_features
            freq = self.model_feature_frequencies.get(name, 0.0)
            cnt = self.feature_counts.get(name, 0)
            w = self.feature_weights.get(name, 0.0)
            if self.mode == "ensemble" and w > 0.0:
                lbl = f"{name}\\n(w={w:.1f}, {freq:.0%})"
            else:
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

        class_labels_ordered = list(self.class_labels)
        if not class_labels_ordered:
            seen_cls: set[str] = set()
            for e in edges_to_draw:
                if e.dominant_class and e.dominant_class not in seen_cls:
                    seen_cls.add(e.dominant_class)
                    class_labels_ordered.append(e.dominant_class)
        class_color_map = _build_class_color_map(class_labels_ordered, class_palette)

        max_s = max((e.strength for e in edges_to_draw), default=1.0) or 1.0
        for e in edges_to_draw:
            a = _dot_id(e.feature_a)
            b = _dot_id(e.feature_b)
            pw = max(1.0, min(5.0, 1.0 + 4.0 * (e.strength / max_s)))
            b_tag = f"\\n[{','.join(e.backends)}]" if e.backends else ""
            c_tag = ""
            if color_by in ("class", "class_score") and e.dominant_class is not None:
                c_tag = (
                    f"\\n[c={e.dominant_class} {e.class_purity:.0%}]"
                    if e.class_purity >= 0.10
                    else "\\n[mixed]"
                )
            elif color_by == "score" and e.score_intensity > 0.0:
                c_tag = f"\\n[m={e.score_intensity:.2f}]"
            edge_lbl = (
                f"{e.count} (S={e.strength:.2f}){b_tag}{c_tag}"
                if e.weight <= 0.0
                else f"{e.count} (S={e.strength:.2f}, w={e.weight:.1f}){b_tag}{c_tag}"
            )
            edge_color = _resolve_edge_color(e, color_by, class_color_map)
            lines.append(
                f'  {a} -- {b} [label="{edge_lbl}", penwidth={pw:.1f}, color="{edge_color}"];'
            )

        if color_by in ("class", "class_score") and class_labels_ordered:
            legend_title = (
                "Target Class (hue=class, saturation=purity)"
                if color_by == "class"
                else "Target Class (hue=class, sat.=purity x margin)"
            )
            lines.append("  subgraph cluster_legend {")
            lines.append(f'    label="{legend_title}";')
            lines.append('    style="rounded,dashed";')
            lines.append('    color="#cbd5e1";')
            lines.append(f'    fontsize={max(8, fontsize_edge)};')
            prev_lid: str | None = None
            for idx, cls_lbl in enumerate(class_labels_ordered):
                lid = f'"__legend_cls_{idx}"'
                chexx = class_color_map[cls_lbl]
                lines.append(
                    f'    {lid} [label="{cls_lbl}", shape=box, style="filled,rounded", '
                    f'fillcolor="{chexx}", fontcolor="#ffffff", color="{chexx}", fontsize={max(8, fontsize_edge)}];'
                )
                if prev_lid is not None:
                    lines.append(f"    {prev_lid} -- {lid} [style=invis];")
                prev_lid = lid
            lines.append("  }")

        lines.append("}\n")
        return "\n".join(lines)

    def to_tikz(
        self,
        title: str | None = None,
        standalone: bool = False,
        *,
        color_by: str | None = None,
        class_palette: Sequence[str] | None = None,
    ) -> str:
        """Generate a publication-ready LaTeX TikZ network graph string."""
        _validate_color_by(color_by)
        active_nodes = sorted(set(self.backbone_features) | {e.feature_a for e in self.edges} | {e.feature_b for e in self.edges})
        n_nodes = len(active_nodes)
        if n_nodes == 0:
            return "% Empty interaction network\n"

        class_labels_ordered = list(self.class_labels)
        if not class_labels_ordered:
            seen_cls: set[str] = set()
            for e in self.edges:
                if e.dominant_class and e.dominant_class not in seen_cls:
                    seen_cls.add(e.dominant_class)
                    class_labels_ordered.append(e.dominant_class)
        class_color_map = _build_class_color_map(class_labels_ordered, class_palette)

        lines = []
        if standalone:
            lines.extend([
                r"\documentclass[tikz,border=5pt]{standalone}",
                r"\usetikzlibrary{positioning,shapes,calc}",
                r"\begin{document}",
            ])

        if color_by is not None:
            seen_colors: dict[str, str] = {}
            for idx, e in enumerate(self.edges):
                chexx = _resolve_edge_color(e, color_by, class_color_map)
                if chexx not in seen_colors:
                    cname = f"prinedge{len(seen_colors)}"
                    seen_colors[chexx] = cname
                    lines.append(f"\\definecolor{{{cname}}}{{HTML}}{{{chexx.lstrip('#').upper()}}}")
        else:
            seen_colors = {}

        lines.extend([
            r"\begin{tikzpicture}[",
            r"  font=\sffamily\footnotesize,",
            r"  bbnode/.style={draw=green!60!black, fill=green!15, rounded corners=2pt, thick, align=center, inner sep=3pt},",
            r"  pernode/.style={draw=gray!80, fill=gray!10, ellipse, align=center, inner sep=2.5pt},",
            r"  intedge/.style={draw=darkgray, line cap=round},",
            r"]",
        ])

        # Place nodes in a regular polygon layout for clean symmetry
        import math
        radius = max(2.5, min(5.5, 1.0 + 0.35 * n_nodes))
        coords = {}
        for idx, name in enumerate(active_nodes):
            angle = 90.0 - (360.0 * idx / max(1, n_nodes))
            rad = math.radians(angle)
            x = radius * math.cos(rad)
            y = radius * math.sin(rad)
            safe_id = re.sub(r"\W+", "_", name).strip("_")
            coords[name] = safe_id
            is_bb = name in self.backbone_features
            style = "bbnode" if is_bb else "pernode"
            freq = self.model_feature_frequencies.get(name, 0.0)
            cnt = self.feature_counts.get(name, 0)
            pct = f"{freq:.0%}".replace("%", r"\%")
            lbl = f"\\textbf{{{name}}}\\\\{pct}" if is_bb else f"{name}\\\\{pct}"
            lines.append(f"  \\node[{style}] ({safe_id}) at ({x:.2f}, {y:.2f}) {{{lbl}}};")

        # Draw edges
        for e in self.edges:
            id_a = coords.get(e.feature_a)
            id_b = coords.get(e.feature_b)
            if not id_a or not id_b:
                continue
            lw = max(0.8, min(4.5, 0.8 + 3.5 * e.strength))
            b_str = f" [{','.join(e.backends)}]" if e.backends else ""
            lbl = f"{e.count}{b_str}" if e.weight <= 0.0 else f"{e.count}"
            if color_by is not None:
                chexx = _resolve_edge_color(e, color_by, class_color_map)
                cname = seen_colors[chexx]
                draw_opt = f"intedge, draw={cname}, line width={lw:.1f}pt"
            else:
                draw_opt = f"intedge, line width={lw:.1f}pt"
            lines.append(
                f"  \\draw[{draw_opt}] ({id_a}) -- node[midway, fill=white, inner sep=1pt, text=darkgray] {{\\tiny {lbl}}} ({id_b});"
            )

        lines.append(r"\end{tikzpicture}")
        if standalone:
            lines.append(r"\end{document}")
        lines.append("")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Format a human-readable Markdown summary table."""
        title = (
            f"### Pareto Front Interaction Summary ({self.n_models} models)"
            if self.mode == "pareto"
            else f"### Intra-Ensemble Interaction Summary ({self.n_rules} rules)"
        )
        freq_col = "Model Presence" if self.mode == "pareto" else "Rule Frequency"
        lines = [
            title,
            "",
            f"- **Mode:** `{self.mode}`",
            f"- **Backbone features:** {', '.join(f'`{f}`' for f in self.backbone_features) if self.backbone_features else '*(none)*'}",
            f"- **Peripheral features:** {len(self.peripheral_features)}",
            f"- **Discovered interactions:** {len(self.edges)}",
            "",
            "#### Feature Inclusion & Backbone Anatomy",
            "",
        ]
        if self.feature_weights:
            lines.append(f"| Feature | {freq_col} | Rule Count | Weight | Role |")
            lines.append("|---|---|---|---|---|")
        else:
            lines.append(f"| Feature | {freq_col} | Rule Count | Role |")
            lines.append("|---|---|---|---|")

        all_feats = sorted(
            self.model_feature_frequencies.keys(),
            key=lambda f: (-self.model_feature_frequencies[f], -self.feature_counts[f], f),
        )
        for f in all_feats:
            freq = self.model_feature_frequencies[f]
            cnt = self.feature_counts[f]
            role = "**Core Backbone**" if f in self.backbone_features else "Peripheral"
            if self.feature_weights:
                w = self.feature_weights.get(f, 0.0)
                lines.append(f"| `{f}` | {freq:.1%} | {cnt} | {w:.2f} | {role} |")
            else:
                lines.append(f"| `{f}` | {freq:.1%} | {cnt} | {role} |")

        if self.edges:
            has_w = any(e.weight > 0 for e in self.edges)
            has_b = any(bool(e.backends) for e in self.edges)
            has_cls = any(e.dominant_class is not None for e in self.edges)
            has_sc = any(e.score_intensity > 0.0 for e in self.edges)
            header = "| Feature A | Feature B | Co-occurrences | Strength | Coverage Ratios |"
            sep = "|---|---|---|---|---|"
            if has_w:
                header += " Weight |"
                sep += "---|"
            if has_b:
                header += " Supporting Backends |"
                sep += "---|"
            if has_cls:
                header += " Dominant Class (Purity) |"
                sep += "---|"
            if has_sc:
                header += " Score Intensity |"
                sep += "---|"
            lines.extend(
                [
                    "",
                    "#### Top Feature Interactions (Co-occurrences in Rules)",
                    "",
                    header,
                    sep,
                ]
            )
            for e in self.edges:
                w_col = f" {e.weight:.2f} |" if has_w else ""
                b_col = f" {', '.join(e.backends)} |" if has_b else ""
                cls_col = (
                    f" `{e.dominant_class}` ({e.class_purity:.0%}) |"
                    if has_cls and e.dominant_class is not None
                    else (" — |" if has_cls else "")
                )
                sc_col = f" {e.score_intensity:.2f} |" if has_sc else ""
                lines.append(
                    f"| `{e.feature_a}` | `{e.feature_b}` | {e.count} | {e.strength:.2f} | "
                    f"A: {e.ratio_a:.1%}, B: {e.ratio_b:.1%} |{w_col}{b_col}{cls_col}{sc_col}"
                )

        return "\n".join(lines)


def _dot_id(name: str) -> str:
    """Return a safely quoted node identifier for GraphViz DOT."""
    escaped = name.replace('"', '\\"')
    return f'"{escaped}"'


@dataclass
class _RuleMeta:
    """Internal per-rule metadata extracted from a rule set or tree."""

    features: set[str]
    raw_margin: float
    dominant_class: str | None
    legacy_weight: float


def _parse_rule_scores(
    scores: Any,
    class_labels: list[str],
    task_type: str,
) -> tuple[float, str | None, float]:
    """Compute (raw_margin, dominant_class, legacy_weight) from rule scores."""
    if not scores:
        return 0.0, None, 1.0
    try:
        scores_f = [
            float(v) if math.isfinite(float(v)) else 0.0
            for v in scores
        ]
    except Exception:
        return 0.0, None, 1.0
    if not scores_f:
        return 0.0, None, 1.0

    legacy_weight = float(max(scores_f))
    if task_type != "regression" and len(scores_f) >= 2:
        sorted_s = sorted(scores_f, reverse=True)
        raw_margin = max(0.0, sorted_s[0] - sorted_s[1])
        if raw_margin > 1e-12:
            best_idx = max(range(len(scores_f)), key=lambda i: scores_f[i])
            dom_cls = (
                class_labels[best_idx]
                if best_idx < len(class_labels)
                else str(best_idx)
            )
        else:
            dom_cls = None
        return raw_margin, dom_cls, legacy_weight

    raw_margin = (
        abs(scores_f[0])
        if len(scores_f) == 1
        else math.sqrt(sum(v * v for v in scores_f))
    )
    return raw_margin, None, legacy_weight


def _extract_rules_meta_from_scored_ruleset(
    ruleset: Any,
    feature_names: Sequence[str] | None = None,
) -> tuple[list[_RuleMeta], list[str] | None, list[str]]:
    """Extract feature sets, score margins, and dominant classes from a ScoredRuleSet."""
    inferred_names = (
        list(feature_names)
        if feature_names
        else getattr(ruleset, "feature_names", None)
    )
    if inferred_names is not None:
        inferred_names = [str(f) for f in inferred_names]

    raw_labels = getattr(ruleset, "class_labels", None) or []
    class_labels = [str(c) for c in raw_labels]
    task_type = str(
        getattr(
            ruleset,
            "task_type",
            "classification" if class_labels else "regression",
        )
    )

    rule_metas: list[_RuleMeta] = []
    rules = getattr(ruleset, "rules", [])
    for rule in rules:
        feats: set[str] = set()
        for atom in getattr(rule, "atoms", []):
            feat = getattr(atom, "feature", None)
            if feat is not None:
                if inferred_names and isinstance(feat, int) and feat < len(inferred_names):
                    feats.add(str(inferred_names[feat]))
                elif inferred_names and isinstance(feat, str):
                    m = re.match(r"^(?:x|fx|f|ff|var_?)(\d+)$", feat)
                    if m and int(m.group(1)) < len(inferred_names):
                        feats.add(str(inferred_names[int(m.group(1))]))
                    else:
                        feats.add(str(feat))
                else:
                    feats.add(str(feat))
            else:
                f_idx = getattr(atom, "feature_idx", None)
                if f_idx is not None:
                    name = (
                        inferred_names[f_idx]
                        if inferred_names and f_idx < len(inferred_names)
                        else f"f{f_idx}"
                    )
                    feats.add(str(name))
        if not feats:
            continue

        scores = getattr(rule, "scores", None)
        if task_type != "regression" and not class_labels and scores and len(scores) >= 2:
            class_labels = [str(i) for i in range(len(scores))]
        raw_margin, dom_cls, legacy_weight = _parse_rule_scores(scores, class_labels, task_type)
        rule_metas.append(
            _RuleMeta(
                features=feats,
                raw_margin=raw_margin,
                dominant_class=dom_cls,
                legacy_weight=legacy_weight,
            )
        )
    return rule_metas, inferred_names, class_labels


def _extract_from_scored_ruleset(
    ruleset: Any,
    feature_names: Sequence[str] | None = None,
) -> tuple[list[tuple[set[str], float]], list[str] | None]:
    """Extract features and legacy weights from a ScoredRuleSet or object with rules."""
    metas, inferred_names, _ = _extract_rules_meta_from_scored_ruleset(
        ruleset, feature_names
    )
    return [(rm.features, rm.legacy_weight) for rm in metas], inferred_names


def _extract_rule_metas_from_single_model(
    model: Any, feature_names: Sequence[str] | None = None
) -> tuple[list[_RuleMeta], list[str]]:
    """Extract rule metadata and inferred class labels from a model or ruleset."""
    if hasattr(model, "ruleset"):
        model = getattr(model, "ruleset")

    if hasattr(model, "rules") and isinstance(model.rules, list):
        metas, _, cls_labels = _extract_rules_meta_from_scored_ruleset(model, feature_names)
        return metas, cls_labels

    poly = model[0] if isinstance(model, tuple) else model
    if hasattr(poly, "monomials"):
        metas = []
        for monomial in getattr(poly, "monomials", []):
            feats: set[str] = set()
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
                metas.append(
                    _RuleMeta(
                        features=feats,
                        raw_margin=0.0,
                        dominant_class=None,
                        legacy_weight=1.0,
                    )
                )
        return metas, []

    return [], []


def _extract_rules_from_single_model(
    model: Any, feature_names: Sequence[str] | None = None
) -> list[set[str]]:
    """Extract set of feature names per rule from a model or ruleset."""
    metas, _ = _extract_rule_metas_from_single_model(model, feature_names)
    return [rm.features for rm in metas]


def _extract_ensemble_rule_metas(
    estimator_or_object: Any,
    feature_names: Sequence[str] | None = None,
) -> tuple[list[_RuleMeta], list[str] | None, list[str]]:
    """Extract rules, weights, margins, and class labels from an ensemble or rule set."""
    inferred_names = list(feature_names) if feature_names else None

    # Case 1: RuleFit (prefer _get_rules() with empirical support, else visualize())
    if hasattr(estimator_or_object, "_get_rules") and callable(estimator_or_object._get_rules):
        try:
            df = estimator_or_object._get_rules()
            if hasattr(df, "iterrows") and "rule" in df.columns and "coef" in df.columns:
                if inferred_names is None:
                    if hasattr(estimator_or_object, "feature_names_in_"):
                        inferred_names = [str(f) for f in estimator_or_object.feature_names_in_]
                    elif hasattr(estimator_or_object, "feature_names_"):
                        inferred_names = [str(f) for f in estimator_or_object.feature_names_]

                metas: list[_RuleMeta] = []
                for _, row in df.iterrows():
                    rule_str = str(row.get("rule", "")).strip()
                    coef = float(row.get("coef", 1.0))
                    abs_coef = abs(coef)
                    if abs_coef < 1e-4:
                        continue
                    rule_type = str(row.get("type", "rule")).lower()
                    support = float(row.get("support", float("nan"))) if "support" in df.columns else float("nan")
                    if rule_type == "rule" and not math.isnan(support) and 0.0 < support < 1.0:
                        weight = abs_coef * math.sqrt(support * (1.0 - support))
                    else:
                        weight = abs_coef
                    if inferred_names:
                        feats = {
                            f
                            for f in inferred_names
                            if re.search(r"\b" + re.escape(f) + r"\b", rule_str)
                        }
                    else:
                        tokens = re.findall(
                            r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:<=|>=|<|>|==|!=)", rule_str
                        )
                        feats = set(tokens)
                    if feats:
                        metas.append(
                            _RuleMeta(
                                features=feats,
                                raw_margin=weight,
                                dominant_class=None,
                                legacy_weight=weight,
                            )
                        )
                return metas, inferred_names, []
        except Exception:
            pass

    if hasattr(estimator_or_object, "visualize") and callable(estimator_or_object.visualize):
        try:
            df = estimator_or_object.visualize()
            if hasattr(df, "iterrows"):
                if inferred_names is None:
                    if hasattr(estimator_or_object, "feature_names_in_"):
                        inferred_names = [str(f) for f in estimator_or_object.feature_names_in_]
                    elif hasattr(estimator_or_object, "feature_names_"):
                        inferred_names = [str(f) for f in estimator_or_object.feature_names_]

                metas = []
                for _, row in df.iterrows():
                    rule_str = str(row.get("rule", "")).strip()
                    coef = float(row.get("coef", 1.0))
                    abs_coef = abs(coef)
                    if abs_coef < 1e-4:
                        continue
                    if inferred_names:
                        feats = {
                            f
                            for f in inferred_names
                            if re.search(r"\b" + re.escape(f) + r"\b", rule_str)
                        }
                    else:
                        tokens = re.findall(
                            r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:<=|>=|<|>|==|!=)", rule_str
                        )
                        feats = set(tokens)
                    if feats:
                        metas.append(
                            _RuleMeta(
                                features=feats,
                                raw_margin=abs_coef,
                                dominant_class=None,
                                legacy_weight=abs_coef,
                            )
                        )
                return metas, inferred_names, []
        except Exception:
            pass

    # Case 2: ExSTraCS or LCS model with population
    if hasattr(estimator_or_object, "population") or type(estimator_or_object).__name__.startswith("ExSTraCS"):
        try:
            from scoredrulesets.estimators.ruleset_transform import exstracs_to_scored_ruleset
            srs = exstracs_to_scored_ruleset(
                estimator_or_object,
                class_labels=[0, 1],
                feature_names=inferred_names,
            )
            return _extract_rules_meta_from_scored_ruleset(srs, inferred_names)
        except Exception:
            pass

    # Case 3: Estimator with ruleset_
    if hasattr(estimator_or_object, "ruleset_") and estimator_or_object.ruleset_ is not None:
        return _extract_rules_meta_from_scored_ruleset(
            estimator_or_object.ruleset_, inferred_names
        )

    # Case 4: Direct ScoredRuleSet
    if hasattr(estimator_or_object, "rules") and isinstance(estimator_or_object.rules, list):
        return _extract_rules_meta_from_scored_ruleset(
            estimator_or_object, inferred_names
        )

    # Case 5: List/collection of models - pool all their rules
    if isinstance(estimator_or_object, (list, tuple)):
        all_metas: list[_RuleMeta] = []
        cls_labels: list[str] = []
        seen_cls: set[str] = set()
        for m in estimator_or_object:
            m_metas, m_names, m_cls = _extract_ensemble_rule_metas(m, inferred_names)
            all_metas.extend(m_metas)
            if inferred_names is None and m_names:
                inferred_names = m_names
            for c in m_cls:
                if c not in seen_cls:
                    seen_cls.add(c)
                    cls_labels.append(c)
        return all_metas, inferred_names, cls_labels

    # Fallback to standard model extraction
    models, names = _collect_models(estimator_or_object)
    fallback_metas: list[_RuleMeta] = []
    cls_labels = []
    seen_cls = set()
    for m in models:
        m_metas, m_cls = _extract_rule_metas_from_single_model(m, feature_names or names)
        fallback_metas.extend(m_metas)
        for c in m_cls:
            if c not in seen_cls:
                seen_cls.add(c)
                cls_labels.append(c)
    return fallback_metas, names, cls_labels


def _extract_ensemble_rules(
    estimator_or_object: Any,
    feature_names: Sequence[str] | None = None,
) -> tuple[list[tuple[set[str], float]], list[str] | None]:
    """Extract rules and weights from an ensemble estimator, population, or ScoredRuleSet."""
    metas, inferred_names, _ = _extract_ensemble_rule_metas(
        estimator_or_object, feature_names
    )
    return [(rm.features, rm.legacy_weight) for rm in metas], inferred_names


def _select_equidistant_tiers(ordered: list[Any], quota: int) -> list[Any]:
    """Select up to ``quota`` complexity-ordered models evenly across the spectrum."""
    if quota <= 0 or not ordered:
        return []
    if len(ordered) <= quota:
        return list(ordered)
    if quota == 1:
        return [ordered[-1]]
    idxs = np.round(np.linspace(0, len(ordered) - 1, quota)).astype(int)
    seen: set[int] = set()
    selected: list[Any] = []
    for idx in idxs:
        i = int(idx)
        if i not in seen:
            seen.add(i)
            selected.append(ordered[i])
    return selected


def _collect_models(
    estimator_or_models: Any,
    archive_source: str = "balanced_pool",
    max_models_per_backend: int = 5,
    rashomon_epsilon: float = 0.05,
    min_backend_score: float | None = None,
    min_relative_skill: float | None = None,
    skill_baseline: float = 0.0,
) -> tuple[list[Any], list[str] | None]:
    """Collect individual models and feature names from an estimator or sequence."""
    feature_names: list[str] | None = None

    # Handle sequence / collection of models
    if isinstance(estimator_or_models, (list, tuple)):
        for item in estimator_or_models:
            if hasattr(item, "feature_names_in_"):
                feature_names = [str(f) for f in item.feature_names_in_]
                break
        return list(estimator_or_models), feature_names

    # Estimator object
    est = estimator_or_models
    if hasattr(est, "feature_names_in_"):
        feature_names = [str(f) for f in est.feature_names_in_]

    # Check for MasterParetoArchive on estimator or passed directly
    arch = getattr(est, "master_archive_", None)
    if arch is None and hasattr(est, "candidates_") and hasattr(est, "candidate_pool_"):
        arch = est

    if arch is not None:
        if archive_source == "strict_front":
            candidates = list(arch.candidates_)
        elif archive_source == "full_pool":
            candidates = list(arch.candidate_pool_) or list(arch.candidates_)
        else:  # "balanced_pool" (default): per-backend epsilon-Rashomon + equidistant tiers
            pool = list(arch.candidate_pool_) or list(arch.candidates_)
            higher_is_better = bool(getattr(arch, "higher_is_better", True))
            by_b: dict[str, list[Any]] = defaultdict(list)
            for c in pool:
                b = str(getattr(c, "backend", "unknown"))
                by_b[b].append(c)

            backend_best: dict[str, float] = {}
            for b, b_cands in by_b.items():
                scores_b = [float(getattr(c, "score", 0.0)) for c in b_cands]
                backend_best[b] = max(scores_b) if higher_is_better else min(scores_b)

            eligible_backends = list(by_b.keys())
            if min_backend_score is not None:
                filtered = [
                    b for b in eligible_backends
                    if (backend_best[b] >= min_backend_score if higher_is_better else backend_best[b] <= min_backend_score)
                ]
                if filtered:
                    eligible_backends = filtered
                elif eligible_backends:
                    best_b = max(eligible_backends, key=lambda b: backend_best[b]) if higher_is_better else min(eligible_backends, key=lambda b: backend_best[b])
                    eligible_backends = [best_b]

            if min_relative_skill is not None and eligible_backends:
                best_global = (
                    max(backend_best[b] for b in eligible_backends)
                    if higher_is_better
                    else min(backend_best[b] for b in eligible_backends)
                )
                max_gain = (best_global - skill_baseline) if higher_is_better else (skill_baseline - best_global)
                if max_gain > 1e-12:
                    filtered_rel = [
                        b for b in eligible_backends
                        if (
                            ((backend_best[b] - skill_baseline) / max_gain) >= min_relative_skill
                            if higher_is_better
                            else ((skill_baseline - backend_best[b]) / max_gain) >= min_relative_skill
                        )
                    ]
                    if filtered_rel:
                        eligible_backends = filtered_rel
                    else:
                        best_b = max(eligible_backends, key=lambda b: backend_best[b]) if higher_is_better else min(eligible_backends, key=lambda b: backend_best[b])
                        eligible_backends = [best_b]
                else:
                    best_b = max(eligible_backends, key=lambda b: backend_best[b]) if higher_is_better else min(eligible_backends, key=lambda b: backend_best[b])
                    eligible_backends = [best_b]

            candidates = []
            for b in eligible_backends:
                b_cands = by_b[b]
                b_best = backend_best[b]
                if rashomon_epsilon is not None and rashomon_epsilon >= 0.0:
                    rash = [
                        c for c in b_cands
                        if (
                            float(getattr(c, "score", 0.0)) >= b_best - rashomon_epsilon
                            if higher_is_better
                            else float(getattr(c, "score", 0.0)) <= b_best + rashomon_epsilon
                        )
                    ]
                else:
                    rash = list(b_cands)
                if not rash:
                    best_c = max(b_cands, key=lambda c: float(getattr(c, "score", 0.0))) if higher_is_better else min(b_cands, key=lambda c: float(getattr(c, "score", 0.0)))
                    rash = [best_c]

                # Deduplicate by atom count, keeping best score per complexity tier
                by_atoms: dict[int, Any] = {}
                for c in rash:
                    comp = int(getattr(c, "atoms", 0))
                    sc = float(getattr(c, "score", 0.0))
                    if comp not in by_atoms:
                        by_atoms[comp] = c
                    else:
                        prev_sc = float(getattr(by_atoms[comp], "score", 0.0))
                        if (higher_is_better and sc > prev_sc) or (not higher_is_better and sc < prev_sc):
                            by_atoms[comp] = c
                ordered = [by_atoms[a] for a in sorted(by_atoms)]
                candidates.extend(_select_equidistant_tiers(ordered, max_models_per_backend))
        return candidates, feature_names

    # Prefer deduplicated complexity Pareto archive if exposed
    pareto_arch = getattr(est, "pareto_archive_", None)
    if isinstance(pareto_arch, dict) and pareto_arch:
        return [pareto_arch[k] for k in sorted(pareto_arch.keys())], feature_names

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
        "Expected a fitted RuleGPClassifier/LogicGPClassifier, AutoScoredRuleSet, "
        "MasterParetoArchive, ScoredRuleSet, or a sequence of models."
    )


def extract_pareto_interactions(
    estimator_or_models: Any,
    *,
    mode: str = "auto",
    feature_names: Sequence[str] | None = None,
    archive_source: str = "balanced_pool",
    max_models_per_backend: int = 5,
    rashomon_epsilon: float = 0.05,
    min_backend_score: float | None = None,
    min_relative_skill: float | None = None,
    skill_baseline: float = 0.0,
    min_occurrences: int = 1,
    min_ratio: float = 0.05,
    laplace_alpha: float = 1.0,
    normalize_per_model: bool = True,
    provenance_weighting: bool = True,
    score_weighting: bool = False,
    color_by: str | None = None,
    max_edges: int | None = None,
    backbone_threshold: float = 0.5,
    backbone_criterion: str = "frequency",
    out_dot: str | Path | None = None,
    out_tikz: str | Path | None = None,
    out_csv: str | Path | None = None,
) -> ParetoFrontSummary:
    """Analyze feature inclusions and interactions across a Pareto front or rule ensemble.

    Parameters
    ----------
    estimator_or_models:
        A fitted estimator with a Pareto archive (`RuleGPClassifier` with
        `final_archive_` or `LogicGPClassifier` with `_final_population`),
        an `AutoScoredRuleSetClassifier` / `AutoScoredRuleSetRegressor`,
        a direct `MasterParetoArchive`, an ensemble estimator (`RuleFitClassifier`,
        `ExSTraCS`), a single `ScoredRuleSet`, or a list of model instances.
    mode:
        Operation mode:
        - ``"auto"`` (default): Selects ``"ensemble"`` if input is a RuleFit / ExSTraCS
          model or a single dense rule set (>1 rule), and ``"pareto"`` if multiple
          models or a Pareto archive are detected.
        - ``"pareto"``: Inter-model consensus across the Pareto front / population.
          Backbone threshold applies to the fraction of models containing the feature.
        - ``"ensemble"``: Intra-ensemble distillation across the rules of the model.
          Backbone threshold applies to the rule frequency or coefficient weight.
    feature_names:
        Optional sequence of feature names. If omitted, inferred from the
        estimator's `feature_names_in_` or rule specifications.
    archive_source:
        Selection strategy when extracting models from an AutoScoredRuleSet or MasterParetoArchive:
        - ``"balanced_pool"`` (default): Balanced representation taking up to `max_models_per_backend`
          Pareto-optimal variants from each active backend. Prevents multi-backend pool dilution.
        - ``"strict_front"``: Global non-dominated Pareto front candidates only.
        - ``"full_pool"``: All harvested candidate variants across all backends.
    max_models_per_backend:
        Maximum number of models to admit per backend when ``archive_source="balanced_pool"`` (default: 5).
    min_occurrences:
        Minimum number of times a feature pair must co-occur in the same rule.
    min_ratio:
        Minimum ratio of pair count relative to the smaller individual feature
        count (`pair_count / min(count_a, count_b)`).
    score_weighting:
        If ``True``, weight rules within each model by their normalized decision
        margin (preserving total unit mass per model when ``normalize_per_model=True``).
        Defaults to ``False`` for uniform rule-mass behavior.
    color_by:
        Optional edge color encoding passed to `to_dot`/`to_tikz`: ``None`` (default
        gray), ``"score"``, ``"class"``, or ``"class_score"``.
    backbone_threshold:
        Fraction (between 0.0 and 1.0) required to be classified as a core "backbone"
        feature. In Mode A: fraction of models. In Mode B: rule frequency (or relative
        weight if ``backbone_criterion="weight"``).
    backbone_criterion:
        Criterion for backbone classification in Mode B:
        - ``"frequency"`` (default): Fraction of rules containing the feature >= backbone_threshold.
        - ``"weight"``: Feature weight relative to max feature weight >= backbone_threshold.
    out_dot:
        Optional file path to write a GraphViz DOT representation.
    out_tikz:
        Optional file path to write a publication-ready LaTeX TikZ network representation.
    out_csv:
        Optional file path to write an interactions CSV.

    Returns
    -------
    ParetoFrontSummary
        Structured object containing backbone features, interaction edges,
        frequencies, supporting backends, and export methods (`to_dot`, `to_tikz`,
        `to_markdown`, `to_dict`).
    """
    if mode not in ("auto", "pareto", "ensemble"):
        raise ValueError(
            f"Invalid mode '{mode}'. Expected 'auto', 'pareto', or 'ensemble'."
        )
    _validate_color_by(color_by)

    # Resolve mode if "auto"
    effective_mode = mode
    if effective_mode == "auto":
        # RuleFit from imodels or skope-rules
        if hasattr(estimator_or_models, "visualize") or (
            hasattr(estimator_or_models, "rules_")
            and (
                hasattr(estimator_or_models, "coef_")
                or hasattr(estimator_or_models, "_coefs")
            )
        ):
            effective_mode = "ensemble"
        # ExSTraCS
        elif (
            hasattr(estimator_or_models, "population")
            or type(estimator_or_models).__name__.startswith("ExSTraCS")
        ):
            effective_mode = "ensemble"
        # AutoScoredRuleSet / MasterParetoArchive
        elif hasattr(estimator_or_models, "master_archive_") or (
            hasattr(estimator_or_models, "candidates_") and hasattr(estimator_or_models, "candidate_pool_")
        ):
            effective_mode = "pareto"
        # Multi-model archive
        elif (
            hasattr(estimator_or_models, "final_archive_")
            and len(estimator_or_models.final_archive_) > 1
        ):
            effective_mode = "pareto"
        elif (
            hasattr(estimator_or_models, "_final_population")
            and len(estimator_or_models._final_population) > 1
        ):
            effective_mode = "pareto"
        elif isinstance(estimator_or_models, (list, tuple)) and len(estimator_or_models) > 1:
            effective_mode = "pareto"
        # Single rule set or single model
        elif hasattr(estimator_or_models, "rules") and len(getattr(estimator_or_models, "rules", [])) > 1:
            effective_mode = "ensemble"
        elif (
            hasattr(estimator_or_models, "ruleset_")
            and estimator_or_models.ruleset_ is not None
            and len(getattr(estimator_or_models.ruleset_, "rules", [])) > 1
        ):
            effective_mode = "ensemble"
        else:
            effective_mode = "pareto"

    # =========================================================================
    # MODE B: INTRA-ENSEMBLE DISTILLATION
    # =========================================================================
    if effective_mode == "ensemble":
        rule_metas, inferred_names, discovered_class_labels = _extract_ensemble_rule_metas(
            estimator_or_models, feature_names
        )
        if feature_names is None:
            feature_names = inferred_names

        n_rules = len(rule_metas)
        if n_rules == 0:
            return ParetoFrontSummary(
                n_models=1,
                n_rules=0,
                feature_counts={},
                model_feature_frequencies={},
                backbone_features=[],
                peripheral_features=[],
                edges=[],
                feature_names=[],
                mode="ensemble",
                class_labels=discovered_class_labels,
            )

        max_raw_margin = max((rm.raw_margin for rm in rule_metas), default=0.0)
        has_positive_margin = max_raw_margin > 1e-12
        max_legacy_weight = max((abs(rm.legacy_weight) for rm in rule_metas), default=0.0)

        feature_rule_counts: dict[str, int] = defaultdict(int)
        feature_weights: dict[str, float] = defaultdict(float)
        feature_intensity_num: dict[str, float] = defaultdict(float)
        feature_intensity_den: dict[str, float] = defaultdict(float)
        feature_class_mass: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        pair_weights: dict[tuple[str, str], float] = defaultdict(float)
        pair_intensity_num: dict[tuple[str, str], float] = defaultdict(float)
        pair_intensity_den: dict[tuple[str, str], float] = defaultdict(float)
        pair_class_mass: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
        all_discovered_features: set[str] = set()

        for rm in rule_metas:
            feats = rm.features
            if score_weighting and has_positive_margin:
                weight = rm.raw_margin
            else:
                weight = rm.legacy_weight
            if has_positive_margin:
                norm_int = rm.raw_margin / max_raw_margin
            elif max_legacy_weight > 1e-12:
                norm_int = abs(rm.legacy_weight) / max_legacy_weight
            else:
                norm_int = 0.0
            cls_w = rm.raw_margin if has_positive_margin else abs(weight)

            all_discovered_features.update(feats)
            for f in feats:
                feature_rule_counts[f] += 1
                feature_weights[f] += weight
                feature_intensity_num[f] += abs(weight) * norm_int
                feature_intensity_den[f] += abs(weight)
                if rm.dominant_class is not None:
                    feature_class_mass[f][rm.dominant_class] += cls_w

            for fa, fb in itertools.combinations(sorted(feats), 2):
                pair_counts[(fa, fb)] += 1
                pair_weights[(fa, fb)] += weight
                pair_intensity_num[(fa, fb)] += abs(weight) * norm_int
                pair_intensity_den[(fa, fb)] += abs(weight)
                if rm.dominant_class is not None:
                    pair_class_mass[(fa, fb)][rm.dominant_class] += cls_w

        rule_frequencies = {
            f: feature_rule_counts[f] / float(n_rules) for f in all_discovered_features
        }
        max_feat_weight = (
            max(feature_weights.values()) if feature_weights else 1.0
        )

        if backbone_criterion == "weight" and max_feat_weight > 0.0:
            backbone = [
                f
                for f in sorted(all_discovered_features)
                if (feature_weights.get(f, 0.0) / max_feat_weight) >= backbone_threshold
            ]
        else:
            backbone = [
                f
                for f in sorted(all_discovered_features)
                if rule_frequencies.get(f, 0.0) >= backbone_threshold
            ]
        backbone.sort(
            key=lambda f: (-feature_weights.get(f, 0.0), -rule_frequencies.get(f, 0.0), f)
        )

        peripheral = [f for f in sorted(all_discovered_features) if f not in backbone]
        peripheral.sort(
            key=lambda f: (-feature_weights.get(f, 0.0), -rule_frequencies.get(f, 0.0), f)
        )

        feature_score_intensity: dict[str, float] = {}
        feature_dominant_class: dict[str, str] = {}
        feature_class_purity: dict[str, float] = {}
        feature_class_distribution: dict[str, dict[str, float]] = {}
        for f in all_discovered_features:
            den_f = feature_intensity_den[f]
            if den_f > 1e-12:
                feature_score_intensity[f] = feature_intensity_num[f] / den_f
            dom_f, pur_f, dist_f = _finalize_class_stats(
                feature_class_mass.get(f, {}), discovered_class_labels
            )
            if dom_f is not None:
                feature_dominant_class[f] = dom_f
                feature_class_purity[f] = pur_f
                feature_class_distribution[f] = dist_f

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

            wa = feature_weights[fa]
            wb = feature_weights[fb]
            denom = math.sqrt(wa * wb)
            strength = (pair_weights[(fa, fb)] / denom) if denom > 0.0 else 0.0
            den_p = pair_intensity_den[(fa, fb)]
            edge_intensity = (pair_intensity_num[(fa, fb)] / den_p) if den_p > 1e-12 else 0.0
            dom_e, pur_e, dist_e = _finalize_class_stats(
                pair_class_mass.get((fa, fb), {}), discovered_class_labels
            )
            edges.append(
                ParetoInteractionEdge(
                    feature_a=fa,
                    feature_b=fb,
                    count=count,
                    ratio_a=ratio_a,
                    ratio_b=ratio_b,
                    strength=strength,
                    weight=pair_weights.get((fa, fb), 0.0),
                    score_intensity=edge_intensity,
                    dominant_class=dom_e,
                    class_purity=pur_e,
                    class_distribution=tuple(dist_e.items()),
                )
            )

        edges.sort(
            key=lambda e: (-e.strength, -e.weight, -e.count, e.feature_a, e.feature_b)
        )
        if max_edges is not None:
            edges = edges[:max_edges]

        summary = ParetoFrontSummary(
            n_models=1,
            n_rules=n_rules,
            feature_counts=dict(feature_rule_counts),
            model_feature_frequencies=rule_frequencies,
            feature_weights=dict(feature_weights),
            pair_weights=dict(pair_weights),
            backbone_features=backbone,
            peripheral_features=peripheral,
            edges=edges,
            feature_names=sorted(all_discovered_features),
            mode="ensemble",
            class_labels=discovered_class_labels,
            feature_score_intensity=feature_score_intensity,
            feature_dominant_class=feature_dominant_class,
            feature_class_purity=feature_class_purity,
            feature_class_distribution=feature_class_distribution,
        )

    # =========================================================================
    # MODE A: INTER-MODEL CONSENSUS ACROSS PARETO FRONT
    # =========================================================================
    else:
        models, inferred_names = _collect_models(
            estimator_or_models,
            archive_source=archive_source,
            max_models_per_backend=max_models_per_backend,
            rashomon_epsilon=rashomon_epsilon,
            min_backend_score=min_backend_score,
            min_relative_skill=min_relative_skill,
            skill_baseline=skill_baseline,
        )
        if feature_names is None:
            feature_names = inferred_names

        n_models = len(models)
        if n_models == 0:
            return ParetoFrontSummary(
                n_models=0,
                n_rules=0,
                feature_counts={},
                model_feature_frequencies={},
                backbone_features=[],
                peripheral_features=[],
                edges=[],
                feature_names=[],
                mode="pareto",
            )

        model_presence: dict[str, int] = defaultdict(int)
        feature_rule_counts: dict[str, int] = defaultdict(int)
        feature_mass: dict[str, float] = defaultdict(float)
        feature_intensity_num: dict[str, float] = defaultdict(float)
        feature_intensity_den: dict[str, float] = defaultdict(float)
        feature_class_mass: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        pair_mass: dict[tuple[str, str], float] = defaultdict(float)
        pair_intensity_num: dict[tuple[str, str], float] = defaultdict(float)
        pair_intensity_den: dict[tuple[str, str], float] = defaultdict(float)
        pair_class_mass: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
        pair_backends: dict[tuple[str, str], set[str]] = defaultdict(set)
        triplet_counts: dict[tuple[str, str, str], int] = defaultdict(int)
        triplet_mass: dict[tuple[str, str, str], float] = defaultdict(float)
        triplet_backends: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        seen_backends: set[str] = set()
        discovered_class_labels = []
        seen_class_labels: set[str] = set()
        pool_scores: list[float] = []
        all_discovered_features: set[str] = set()
        total_rules_across_models = 0
        total_effective_mass = 0.0

        for model in models:
            b = str(getattr(model, "backend", "") or getattr(model, "metadata", {}).get("backend", ""))
            if b:
                seen_backends.add(b)
            sc = getattr(model, "score", None)
            if sc is not None:
                try:
                    pool_scores.append(float(sc))
                except Exception:
                    pass
            rule_metas, m_cls_labels = _extract_rule_metas_from_single_model(model, feature_names)
            for c_lbl in m_cls_labels:
                if c_lbl not in seen_class_labels:
                    seen_class_labels.add(c_lbl)
                    discovered_class_labels.append(c_lbl)

            k_rules = len(rule_metas)
            sum_m = sum(rm.raw_margin for rm in rule_metas)
            max_m = max((rm.raw_margin for rm in rule_metas), default=0.0)
            uniform_mass = (
                1.0 / float(k_rules)
                if (normalize_per_model and k_rules > 0)
                else 1.0
            )
            model_feats: set[str] = set()

            for rm in rule_metas:
                feats = rm.features
                total_rules_across_models += 1
                model_feats.update(feats)
                all_discovered_features.update(feats)

                norm_int = (rm.raw_margin / max_m) if max_m > 1e-12 else 0.0
                score_mass = (
                    (rm.raw_margin / sum_m)
                    if sum_m > 1e-12
                    else ((1.0 / float(k_rules)) if k_rules > 0 else 1.0)
                )
                if score_weighting:
                    eff_mass = (
                        score_mass
                        if normalize_per_model
                        else (rm.raw_margin if sum_m > 1e-12 else 1.0)
                    )
                else:
                    eff_mass = uniform_mass
                total_effective_mass += eff_mass

                cls_mass = (
                    score_mass
                    if (normalize_per_model and sum_m > 1e-12)
                    else (rm.raw_margin if sum_m > 1e-12 else uniform_mass)
                )
                if rm.dominant_class is not None and rm.dominant_class not in seen_class_labels:
                    seen_class_labels.add(rm.dominant_class)
                    discovered_class_labels.append(rm.dominant_class)

                for f in feats:
                    feature_rule_counts[f] += 1
                    feature_mass[f] += eff_mass
                    if max_m > 1e-12:
                        feature_intensity_num[f] += eff_mass * norm_int
                        feature_intensity_den[f] += eff_mass
                    if rm.dominant_class is not None:
                        feature_class_mass[f][rm.dominant_class] += cls_mass

                sorted_feats = sorted(feats)
                for fa, fb in itertools.combinations(sorted_feats, 2):
                    pair_counts[(fa, fb)] += 1
                    pair_mass[(fa, fb)] += eff_mass
                    if max_m > 1e-12:
                        pair_intensity_num[(fa, fb)] += eff_mass * norm_int
                        pair_intensity_den[(fa, fb)] += eff_mass
                    if rm.dominant_class is not None:
                        pair_class_mass[(fa, fb)][rm.dominant_class] += cls_mass
                    if b:
                        pair_backends[(fa, fb)].add(b)

                for fa, fb, fc in itertools.combinations(sorted_feats, 3):
                    triplet_counts[(fa, fb, fc)] += 1
                    triplet_mass[(fa, fb, fc)] += eff_mass
                    if b:
                        triplet_backends[(fa, fb, fc)].add(b)

            for f in model_feats:
                model_presence[f] += 1

        active_backends = sorted(seen_backends)
        n_backends = max(len(active_backends), 1)

        frequencies = {
            f: model_presence[f] / float(n_models) for f in all_discovered_features
        }
        backbone = [
            f
            for f in sorted(all_discovered_features)
            if frequencies.get(f, 0.0) >= backbone_threshold
        ]
        backbone.sort(key=lambda f: (-frequencies[f], -feature_mass[f], -feature_rule_counts[f], f))

        peripheral = [
            f
            for f in sorted(all_discovered_features)
            if frequencies.get(f, 0.0) < backbone_threshold
        ]
        peripheral.sort(key=lambda f: (-frequencies[f], -feature_mass[f], -feature_rule_counts[f], f))

        feature_score_intensity = {}
        feature_dominant_class = {}
        feature_class_purity = {}
        feature_class_distribution = {}
        for f in all_discovered_features:
            den_f = feature_intensity_den[f]
            if den_f > 1e-12:
                feature_score_intensity[f] = feature_intensity_num[f] / den_f
            dom_f, pur_f, dist_f = _finalize_class_stats(
                feature_class_mass.get(f, {}), discovered_class_labels
            )
            if dom_f is not None:
                feature_dominant_class[f] = dom_f
                feature_class_purity[f] = pur_f
                feature_class_distribution[f] = dist_f

        edges = []
        for (fa, fb), count in pair_counts.items():
            if count < min_occurrences:
                continue
            cnt_a = feature_rule_counts[fa]
            cnt_b = feature_rule_counts[fb]
            ratio_a = count / cnt_a if cnt_a > 0 else 0.0
            ratio_b = count / cnt_b if cnt_b > 0 else 0.0

            if max(ratio_a, ratio_b) < min_ratio:
                continue

            denom = feature_mass[fa] + feature_mass[fb] + 2.0 * laplace_alpha
            base_strength = (2.0 * pair_mass[(fa, fb)] / denom) if denom > 0.0 else 0.0
            b_set = pair_backends.get((fa, fb), set())
            if provenance_weighting and len(active_backends) > 1:
                strength = base_strength * (len(b_set) / float(n_backends))
            else:
                strength = base_strength

            den_p = pair_intensity_den[(fa, fb)]
            edge_intensity = (pair_intensity_num[(fa, fb)] / den_p) if den_p > 1e-12 else 0.0
            dom_e, pur_e, dist_e = _finalize_class_stats(
                pair_class_mass.get((fa, fb), {}), discovered_class_labels
            )
            edges.append(
                ParetoInteractionEdge(
                    feature_a=fa,
                    feature_b=fb,
                    count=count,
                    ratio_a=ratio_a,
                    ratio_b=ratio_b,
                    strength=strength,
                    backends=tuple(sorted(b_set)),
                    score_intensity=edge_intensity,
                    dominant_class=dom_e,
                    class_purity=pur_e,
                    class_distribution=tuple(dist_e.items()),
                )
            )

        edges.sort(key=lambda e: (-e.strength, -e.count, e.feature_a, e.feature_b))
        if max_edges is not None:
            edges = edges[:max_edges]

        if score_weighting and total_effective_mass > 1e-12:
            total_mass = total_effective_mass
        else:
            total_mass = float(n_models) if normalize_per_model else float(total_rules_across_models)
        triplets: list[ParetoInteractionTriplet] = []
        for (fa, fb, fc), t_cnt in triplet_counts.items():
            if t_cnt < min_occurrences:
                continue
            denom_t = feature_mass[fa] * feature_mass[fb] * feature_mass[fc]
            lift = (triplet_mass[(fa, fb, fc)] * (total_mass ** 2) / denom_t) if denom_t > 0.0 else 0.0
            triplets.append(
                ParetoInteractionTriplet(
                    features=(fa, fb, fc),
                    count=t_cnt,
                    mass=triplet_mass[(fa, fb, fc)],
                    lift=lift,
                    backends=tuple(sorted(triplet_backends.get((fa, fb, fc), set()))),
                )
            )
        triplets.sort(key=lambda t: (-len(t.backends), -t.mass, -t.lift, t.features))

        summary = ParetoFrontSummary(
            n_models=n_models,
            n_rules=total_rules_across_models,
            feature_counts=dict(feature_rule_counts),
            model_feature_frequencies=frequencies,
            backbone_features=backbone,
            peripheral_features=peripheral,
            edges=edges,
            feature_names=sorted(all_discovered_features),
            mode="pareto",
            triplets=triplets,
            active_backends=active_backends,
            pool_scores=pool_scores,
            class_labels=discovered_class_labels,
            feature_score_intensity=feature_score_intensity,
            feature_dominant_class=feature_dominant_class,
            feature_class_purity=feature_class_purity,
            feature_class_distribution=feature_class_distribution,
        )

    # Export formats
    if out_dot is not None:
        dot_path = Path(out_dot)
        dot_path.parent.mkdir(parents=True, exist_ok=True)
        dot_path.write_text(summary.to_dot(color_by=color_by), encoding="utf-8")

    if out_tikz is not None:
        tikz_path = Path(out_tikz)
        tikz_path.parent.mkdir(parents=True, exist_ok=True)
        tikz_path.write_text(summary.to_tikz(color_by=color_by), encoding="utf-8")

    if out_csv is not None:
        csv_path = Path(out_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            has_w = any(e.weight > 0 for e in summary.edges)
            has_b = any(bool(e.backends) for e in summary.edges)
            has_cls = any(e.dominant_class is not None for e in summary.edges)
            has_sc = any(e.score_intensity > 0.0 for e in summary.edges)
            header = ["feature_a", "feature_b", "count", "ratio_a", "ratio_b", "strength"]
            if has_w:
                header.append("weight")
            if has_b:
                header.append("backends")
            if has_cls:
                header.extend(["dominant_class", "class_purity"])
            if has_sc:
                header.append("score_intensity")
            writer.writerow(header)
            for e in summary.edges:
                row = [
                    e.feature_a,
                    e.feature_b,
                    e.count,
                    f"{e.ratio_a:.4f}",
                    f"{e.ratio_b:.4f}",
                    f"{e.strength:.4f}",
                ]
                if has_w:
                    row.append(f"{e.weight:.4f}")
                if has_b:
                    row.append(";".join(e.backends))
                if has_cls:
                    row.extend([e.dominant_class or "", f"{e.class_purity:.4f}"])
                if has_sc:
                    row.append(f"{e.score_intensity:.4f}")
                writer.writerow(row)

    return summary
