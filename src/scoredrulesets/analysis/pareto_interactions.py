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
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


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

    def to_dict(self) -> dict[str, Any]:
        res = {
            "feature_a": self.feature_a,
            "feature_b": self.feature_b,
            "count": self.count,
            "ratio_a": round(self.ratio_a, 4),
            "ratio_b": round(self.ratio_b, 4),
            "strength": round(self.strength, 4),
        }
        if self.weight > 0.0:
            res["weight"] = round(self.weight, 4)
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
        return res

    def to_dot(self, title: str | None = None) -> str:
        """Generate a GraphViz DOT graph string."""
        if title is None:
            title = (
                "Pareto Interaction Graph"
                if self.mode == "pareto"
                else "Intra-Ensemble Interaction Graph"
            )
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

        for e in self.edges:
            a = _dot_id(e.feature_a)
            b = _dot_id(e.feature_b)
            pw = max(1.0, min(5.0, 1.0 + 4.0 * e.strength))
            edge_lbl = (
                f"{e.count}" if e.weight <= 0.0 else f"{e.count} (w={e.weight:.1f})"
            )
            lines.append(
                f'  {a} -- {b} [label="{edge_lbl}", penwidth={pw:.1f}, color="#495057"];'
            )

        lines.append("}\n")
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
            lines.extend(
                [
                    "",
                    "#### Top Feature Interactions (Co-occurrences in Rules)",
                    "",
                    "| Feature A | Feature B | Co-occurrences | Strength | Coverage Ratios |"
                    + (" Weight |" if has_w else ""),
                    "|---|---|---|---|---|" + ("---|" if has_w else ""),
                ]
            )
            for e in self.edges:
                extra = f" {e.weight:.2f} |" if has_w else ""
                lines.append(
                    f"| `{e.feature_a}` | `{e.feature_b}` | {e.count} | {e.strength:.2f} | "
                    f"A: {e.ratio_a:.1%}, B: {e.ratio_b:.1%} |{extra}"
                )

        return "\n".join(lines)


def _dot_id(name: str) -> str:
    """Return a safely quoted node identifier for GraphViz DOT."""
    escaped = name.replace('"', '\\"')
    return f'"{escaped}"'


def _extract_from_scored_ruleset(
    ruleset: Any,
    feature_names: Sequence[str] | None = None,
) -> tuple[list[tuple[set[str], float]], list[str] | None]:
    """Extract features and weights from a ScoredRuleSet or object with rules."""
    inferred_names = (
        list(feature_names)
        if feature_names
        else getattr(ruleset, "feature_names", None)
    )
    if inferred_names is not None:
        inferred_names = [str(f) for f in inferred_names]

    rules_with_weights: list[tuple[set[str], float]] = []
    rules = getattr(ruleset, "rules", [])
    for rule in rules:
        scores = getattr(rule, "scores", None)
        try:
            w = float(max(scores)) if scores else 1.0
        except Exception:
            w = 1.0

        feats = set()
        for atom in getattr(rule, "atoms", []):
            feat = getattr(atom, "feature", None)
            if feat is not None:
                if inferred_names and isinstance(feat, int) and feat < len(inferred_names):
                    feats.add(str(inferred_names[feat]))
                elif (
                    inferred_names
                    and isinstance(feat, str)
                    and feat.startswith("f")
                    and feat[1:].isdigit()
                ):
                    idx = int(feat[1:])
                    if idx < len(inferred_names):
                        feats.add(str(inferred_names[idx]))
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
        if feats:
            rules_with_weights.append((feats, w))
    return rules_with_weights, inferred_names


def _extract_rules_from_single_model(
    model: Any, feature_names: Sequence[str] | None = None
) -> list[set[str]]:
    """Extract set of feature names per rule from a model or ruleset."""
    # 1. Rule sets with a list of rules (ScoredRuleSet or RuleGP _RuleSet2)
    if hasattr(model, "rules") and isinstance(model.rules, list):
        rules_w, _ = _extract_from_scored_ruleset(model, feature_names)
        return [r[0] for r in rules_w]

    # 2. LogicGP polynomial / individual (has 'monomials' with 'literals')
    poly = model[0] if isinstance(model, tuple) else model
    if hasattr(poly, "monomials"):
        rules_feats: list[set[str]] = []
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

    return []


def _extract_ensemble_rules(
    estimator_or_object: Any,
    feature_names: Sequence[str] | None = None,
) -> tuple[list[tuple[set[str], float]], list[str] | None]:
    """Extract rules and weights from an ensemble estimator, population, or ScoredRuleSet."""
    inferred_names = list(feature_names) if feature_names else None

    # Case 1: RuleFit (e.g., from imodels with visualize() DataFrame)
    if hasattr(estimator_or_object, "visualize") and callable(estimator_or_object.visualize):
        try:
            df = estimator_or_object.visualize()
            if hasattr(df, "iterrows"):
                if inferred_names is None:
                    if hasattr(estimator_or_object, "feature_names_in_"):
                        inferred_names = [str(f) for f in estimator_or_object.feature_names_in_]
                    elif hasattr(estimator_or_object, "feature_names_"):
                        inferred_names = [str(f) for f in estimator_or_object.feature_names_]

                rules_with_weights: list[tuple[set[str], float]] = []
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
                        rules_with_weights.append((feats, abs_coef))
                return rules_with_weights, inferred_names
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
            return _extract_from_scored_ruleset(srs, inferred_names)
        except Exception:
            pass

    # Case 3: Estimator with ruleset_
    if hasattr(estimator_or_object, "ruleset_") and estimator_or_object.ruleset_ is not None:
        return _extract_from_scored_ruleset(estimator_or_object.ruleset_, inferred_names)

    # Case 4: Direct ScoredRuleSet
    if hasattr(estimator_or_object, "rules") and isinstance(estimator_or_object.rules, list):
        return _extract_from_scored_ruleset(estimator_or_object, inferred_names)

    # Case 5: List/collection of models - pool all their rules
    if isinstance(estimator_or_object, (list, tuple)):
        all_rules: list[tuple[set[str], float]] = []
        for m in estimator_or_object:
            m_rules, m_names = _extract_ensemble_rules(m, inferred_names)
            all_rules.extend(m_rules)
            if inferred_names is None and m_names:
                inferred_names = m_names
        return all_rules, inferred_names

    # Fallback to standard model extraction
    models, names = _collect_models(estimator_or_object)
    rules_with_weights = []
    for m in models:
        for r_feats in _extract_rules_from_single_model(m, feature_names or names):
            rules_with_weights.append((r_feats, 1.0))
    return rules_with_weights, names


def _collect_models(estimator_or_models: Any) -> tuple[list[Any], list[str] | None]:
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
    mode: str = "auto",
    feature_names: Sequence[str] | None = None,
    min_occurrences: int = 1,
    min_ratio: float = 0.05,
    backbone_threshold: float = 0.5,
    backbone_criterion: str = "frequency",
    out_dot: str | Path | None = None,
    out_csv: str | Path | None = None,
) -> ParetoFrontSummary:
    """Analyze feature inclusions and interactions across a Pareto front or rule ensemble.

    Parameters
    ----------
    estimator_or_models:
        A fitted estimator with a Pareto archive (`RuleGPClassifier` with
        `final_archive_` or `LogicGPClassifier` with `_final_population`),
        an ensemble estimator (`RuleFitClassifier`, `ExSTraCS`), a single
        `ScoredRuleSet`, or a list of model instances.
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
    min_occurrences:
        Minimum number of times a feature pair must co-occur in the same rule.
    min_ratio:
        Minimum ratio of pair count relative to the smaller individual feature
        count (`pair_count / min(count_a, count_b)`).
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
    out_csv:
        Optional file path to write an interactions CSV.

    Returns
    -------
    ParetoFrontSummary
        Structured object containing backbone features, interaction edges,
        frequencies, and export methods (`to_dot`, `to_markdown`, `to_dict`).
    """
    if mode not in ("auto", "pareto", "ensemble"):
        raise ValueError(
            f"Invalid mode '{mode}'. Expected 'auto', 'pareto', or 'ensemble'."
        )

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
        rules_with_weights, inferred_names = _extract_ensemble_rules(
            estimator_or_models, feature_names
        )
        if feature_names is None:
            feature_names = inferred_names

        n_rules = len(rules_with_weights)
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
            )

        feature_rule_counts: dict[str, int] = defaultdict(int)
        feature_weights: dict[str, float] = defaultdict(float)
        pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        pair_weights: dict[tuple[str, str], float] = defaultdict(float)
        all_discovered_features: set[str] = set()

        for feats, weight in rules_with_weights:
            all_discovered_features.update(feats)
            for f in feats:
                feature_rule_counts[f] += 1
                feature_weights[f] += weight

            for fa, fb in itertools.combinations(sorted(feats), 2):
                pair_counts[(fa, fb)] += 1
                pair_weights[(fa, fb)] += weight

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
                    weight=pair_weights.get((fa, fb), 0.0),
                )
            )

        edges.sort(
            key=lambda e: (-e.weight, -e.strength, -e.count, e.feature_a, e.feature_b)
        )

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
        )

    # =========================================================================
    # MODE A: INTER-MODEL CONSENSUS ACROSS PARETO FRONT
    # =========================================================================
    else:
        models, inferred_names = _collect_models(estimator_or_models)
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
        pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        all_discovered_features: set[str] = set()
        total_rules_across_models = 0

        for model in models:
            rules_feats = _extract_rules_from_single_model(model, feature_names)
            model_feats: set[str] = set()

            for feats in rules_feats:
                total_rules_across_models += 1
                model_feats.update(feats)
                all_discovered_features.update(feats)
                for f in feats:
                    feature_rule_counts[f] += 1

                for fa, fb in itertools.combinations(sorted(feats), 2):
                    pair_counts[(fa, fb)] += 1

            for f in model_feats:
                model_presence[f] += 1

        frequencies = {
            f: model_presence[f] / float(n_models) for f in all_discovered_features
        }
        backbone = [
            f
            for f in sorted(all_discovered_features)
            if frequencies.get(f, 0.0) >= backbone_threshold
        ]
        backbone.sort(key=lambda f: (-frequencies[f], -feature_rule_counts[f], f))

        peripheral = [
            f
            for f in sorted(all_discovered_features)
            if frequencies.get(f, 0.0) < backbone_threshold
        ]
        peripheral.sort(key=lambda f: (-frequencies[f], -feature_rule_counts[f], f))

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

        edges.sort(key=lambda e: (-e.strength, -e.count, e.feature_a, e.feature_b))

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
        )

    # Export formats
    if out_dot is not None:
        dot_path = Path(out_dot)
        dot_path.parent.mkdir(parents=True, exist_ok=True)
        dot_path.write_text(summary.to_dot(), encoding="utf-8")

    if out_csv is not None:
        csv_path = Path(out_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            has_w = any(e.weight > 0 for e in summary.edges)
            header = ["feature_a", "feature_b", "count", "ratio_a", "ratio_b", "strength"]
            if has_w:
                header.append("weight")
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
                writer.writerow(row)

    return summary
