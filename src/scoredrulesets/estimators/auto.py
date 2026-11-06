"""AutoML meta-estimators for Scored Rule Sets.

Provides:
- MasterParetoArchive: Container maintaining a non-dominated Pareto front over
  structural complexity (atoms) and predictive performance.
- AutoScoredRuleSetClassifier: Multi-backend classifier with intent profiles,
  multi-fidelity probing, and cross-backend Master Pareto Fusion.
- AutoScoredRuleSetRegressor: Multi-backend regressor with intent profiles and
  cross-backend Pareto Fusion.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import time
from typing import Any, Callable, Literal, Sequence
import warnings
import zipfile

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import f1_score, get_scorer, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.utils.validation import check_is_fitted, check_X_y

from ..formatting import format_ruleset_markdown, format_ruleset_table
from ..runtime import predict as predict_from_ruleset
from ..runtime import predict_proba as predict_proba_from_ruleset
from ..runtime import predict_regression as predict_regression_from_ruleset
from ..schema import ScoredRuleSet
from .base import BaseRuleSetEstimator
from .sklearn_wrapper import ScoredRuleSetClassifier, ScoredRuleSetRegressor


_DEFAULT_CLASSIFIER_BACKENDS = ["greedy_pareto", "cart", "hs", "ruleplcs"]
_DEFAULT_REGRESSOR_BACKENDS = ["greedy_residual", "greedy_pareto", "cart", "ruleplcs"]

# Architectural top-k sweep harvested for the neural backend when the user
# does not pin ``max_atoms_per_rule`` explicitly.  Each k contributes its own
# point to the Master Pareto Archive; dominated variants are pruned by the
# archive itself, so the sweep doubles as an automatic front sampler.
_RULENLN_SWEEP_K = (2, 3, 4, 6)


def _transform_for_estimator(estimator: Any, X: Any) -> np.ndarray:
    """Apply a fitted wrapper's own preprocessing/feature selection/encoding to X.

    Each archive candidate must be evaluated (and served) in the input space of
    the estimator that produced it; atom feature indices refer to that space.
    """
    X_arr = np.asarray(X)
    if estimator is None:
        return X_arr
    pipeline = getattr(estimator, "preprocess_pipeline_", None)
    if pipeline is not None:
        X_arr = np.asarray(pipeline.transform(X_arr), dtype=None)
    selector = getattr(estimator, "feature_selector_", None)
    if selector is not None:
        X_arr = np.asarray(selector.transform(X_arr), dtype=None)
    elif getattr(estimator, "selected_feature_indices_", None) is not None:
        X_arr = X_arr[:, estimator.selected_feature_indices_]
    if hasattr(estimator, "_prepare_X_for_prediction"):
        X_arr = estimator._prepare_X_for_prediction(X_arr)
    return X_arr


def _harvest_variants(fitted: Any) -> dict[tuple, ScoredRuleSet]:
    """Return all rule sets a fitted wrapper offers, keyed stably across refits.

    Backends exposing an internal Pareto front (``pareto_archive_``, keyed by
    complexity) contribute every front member; all others contribute their
    single fitted rule set.
    """
    underlying = getattr(fitted, "estimator_", fitted)
    front = getattr(underlying, "pareto_archive_", None)
    if not (isinstance(front, dict) and front):
        front = getattr(fitted, "pareto_archive_", None)
    if isinstance(front, dict) and front:
        return {("front", int(comp)): rs for comp, rs in front.items()}
    return {("model",): fitted.to_ruleset()}


def _macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _archive_metric_from_scoring(scoring: str | None) -> Callable[[Any, Any], float]:
    """Derive the archive metric from the estimator's ``scoring`` string.

    Historically the Master Pareto Archive was always scored with macro-F1
    while backend ranking used ``self.scoring`` (default ``f1_weighted``).
    The two metrics can disagree by a hair (Ionosphere: greedy's 7-atom front
    member beats CART on weighted but loses on macro), flipping the accuracy
    profile's pick.  Harmonize: the archive now follows ``scoring`` for the
    f1/accuracy families; unknown scorers fall back to macro-F1 with a warning.
    """
    s = (scoring or "f1_weighted").lower()
    base = s.split("@")[0]  # strip possible decorator suffix
    if base.startswith("f1"):
        average = base.split("_", 1)[1] if "_" in base else "binary"
        return lambda y_true, y_pred, _a=average: float(
            f1_score(y_true, y_pred, average=_a, zero_division=0)
        )
    if base in ("accuracy", "acc"):
        from sklearn.metrics import accuracy_score
        return lambda y_true, y_pred: float(accuracy_score(y_true, y_pred))
    warnings.warn(
        f"AutoScoredRuleSet: archive metric falls back to macro-F1 for "
        f"scoring='{scoring}'.",
        UserWarning,
    )
    return _macro_f1


def _r2(y_true, y_pred) -> float:
    return float(r2_score(y_true, y_pred))


def _hypervolume_2d(points: list[tuple[float, float]], ref: tuple[float, float]) -> float:
    """HV of 2D points (complexity minimized, score maximized) vs reference.

    ``points`` are ``(atoms, score)`` pairs; ``ref`` is the dominated reference
    point ``(atoms_ref, score_ref)`` with ``atoms_ref`` >= every considered
    complexity and ``score_ref`` below every score.  Returns 0.0 for an empty
    or fully dominated front.
    """
    if not points:
        return 0.0
    # Keep non-dominated staircase: sort by atoms asc, keep strictly rising scores
    staircase: list[tuple[float, float]] = []
    best = -np.inf
    for atoms, score in sorted(points, key=lambda p: (p[0], -p[1])):
        if score > best:
            staircase.append((atoms, score))
            best = score
    hv = 0.0
    for i, (atoms, score) in enumerate(staircase):
        next_atoms = staircase[i + 1][0] if i + 1 < len(staircase) else ref[0]
        width = next_atoms - atoms
        if width <= 0:
            continue
        height = score - ref[1]
        if height <= 0:
            continue
        hv += width * height
    return float(hv)


def _probe_hv_contribution(
    make_estimator: Callable[[], Any],
    X_probe: np.ndarray,
    y_probe: np.ndarray,
    splitter: Any,
    *,
    archive_metric: Callable[[Any, Any], float],
    predict_fn: Callable[[ScoredRuleSet, np.ndarray], Any],
) -> dict[str, float]:
    """Per-backend hypervolume contribution on the probe set.

    For every backend: fit per probe fold, harvest all rule sets the wrapper
    offers (front members where exposed, else the single served model), score
    each on the held-out fold, and collect ``(atoms, score)`` points.  The
    backend's contribution is the hypervolume *gain* its points add to the
    union front of all other backends (leave-one-out).  Backends that only
    produce dominated candidates contribute 0.0; a backend holding a unique
    front corner contributes that corner's slab.

    This is the fusion-aware probing criterion: unlike mean-score ranking, it
    retains engines that are weak on average but non-dominated at one end of
    the complexity spectrum (e.g. greedy at the compact end).
    """
    per_backend: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for train_idx, val_idx in splitter.split(X_probe, y_probe):
        X_tr, y_tr = X_probe[train_idx], y_probe[train_idx]
        X_val, y_val = X_probe[val_idx], y_probe[val_idx]
        for backend, mk in make_estimator.items():
            try:
                est = mk()
                est.fit(X_tr, y_tr)
            except Exception:
                continue
            X_val_t = _transform_for_estimator(est, X_val)
            for rs in _harvest_variants(est).values():
                try:
                    score = float(archive_metric(y_val, predict_fn(rs, X_val_t)))
                except Exception:
                    continue
                atoms = sum(len(r.atoms) for r in rs.rules)
                per_backend[backend].append((float(atoms), score))
    # Collapse per-backend points to one (atoms, mean score) staircase per atom count
    collapsed: dict[str, list[tuple[float, float]]] = {}
    for backend, pts in per_backend.items():
        by_atoms: dict[float, list[float]] = defaultdict(list)
        for atoms, score in pts:
            by_atoms[atoms].append(score)
        collapsed[backend] = [(a, float(np.mean(v))) for a, v in sorted(by_atoms.items())]
    # Leave-one-out HV contribution
    all_backends = list(collapsed.keys())
    contributions: dict[str, float] = {}
    for backend in all_backends:
        others = [p for b in all_backends if b != backend for p in collapsed[b]]
        own = collapsed[backend]
        if not own:
            contributions[backend] = 0.0
            continue
        max_atoms = max(p[0] for p in (others + own)) + 1
        min_score = min(p[1] for p in (others + own)) - 1e-6
        ref = (max_atoms, min_score)
        hv_without = _hypervolume_2d(others, ref) if others else 0.0
        hv_with = _hypervolume_2d(others + own, ref)
        contributions[backend] = max(0.0, hv_with - hv_without)
    return contributions


def _oof_evaluate_config(
    make_estimator: Callable[[], Any],
    X: np.ndarray,
    y: np.ndarray,
    splitter: Any,
    *,
    main_scorer: Callable | None,
    archive_metric: Callable[[Any, Any], float],
    predict_fn: Callable[[ScoredRuleSet, np.ndarray], Any],
) -> tuple[float, dict[tuple, list[float]]]:
    """Cross-validate one backend configuration.

    Returns the mean held-out ``main_scorer`` score of the fitted model (NaN if
    no scorer is given) and, for every harvested variant key, the list of
    held-out ``archive_metric`` scores.  Every variant is evaluated through its
    own rule set in its own estimator's input space, so backend-internal fronts
    are scored member by member.
    """
    main_scores: list[float] = []
    per_key: dict[tuple, list[float]] = defaultdict(list)
    for train_idx, val_idx in splitter.split(X, y):
        est = make_estimator()
        est.fit(X[train_idx], y[train_idx])
        X_val, y_val = X[val_idx], y[val_idx]
        if main_scorer is not None:
            main_scores.append(float(main_scorer(est, X_val, y_val)))
        X_val_t = _transform_for_estimator(est, X_val)
        for key, rs in _harvest_variants(est).items():
            per_key[key].append(archive_metric(y_val, predict_fn(rs, X_val_t)))
    main = float(np.mean(main_scores)) if main_scores else float("nan")
    return main, dict(per_key)


def _admit_full_fit(
    archive: "MasterParetoArchive",
    est_full: Any,
    label: str,
    per_key: dict[tuple, list[float]],
    *,
    n_splits: int,
    candidate_pool: list["ParetoCandidate"] | None = None,
) -> int:
    """Offer the full-data rule sets of ``est_full`` to the archive with OOF scores.

    Complexity is taken from the full-data rule set (the one actually served);
    the score is the mean out-of-fold score of the same variant key.  Variants
    observed in fewer than half of the folds are skipped as unreliable.
    """
    min_folds = max(1, (n_splits + 1) // 2)
    admitted = 0
    for key, rs in _harvest_variants(est_full).items():
        scores = per_key.get(key)
        if not scores or len(scores) < min_folds:
            continue
        mean_score = float(np.mean(scores))
        meta = {
            "estimator": est_full,
            "variant": key,
            "oof_folds": len(scores),
            "oof_std": float(np.std(scores)),
        }
        atoms = sum(len(r.atoms) for r in rs.rules)
        non_default_rules = len([r for r in rs.rules if r.atoms])
        pool_cand = ParetoCandidate(
            ruleset=rs,
            score=mean_score,
            atoms=atoms,
            rules=non_default_rules,
            backend=label,
            metadata=dict(meta),
        )
        archive.candidate_pool_.append(pool_cand)
        if candidate_pool is not None:
            candidate_pool.append(pool_cand)
        if archive.add(
            rs,
            mean_score,
            backend=label,
            metadata=meta,
        ):
            admitted += 1
    return admitted


@dataclass
class ParetoCandidate:
    """A single non-dominated ScoredRuleSet candidate."""

    ruleset: ScoredRuleSet
    score: float
    atoms: int
    rules: int
    backend: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def effective_nodes(self) -> int:
        """Return internal split-node count if candidate originates from a tree, else path atoms."""
        if isinstance(self.ruleset.metadata, dict):
            n_splits = self.ruleset.metadata.get("tree_n_splits")
            if n_splits is not None:
                return int(n_splits)
        return int(self.atoms)

    def to_dict(self) -> dict[str, Any]:
        safe_meta: dict[str, Any] = {}
        for k, v in (self.metadata or {}).items():
            if k == "estimator":
                continue
            if isinstance(v, tuple):
                safe_meta[k] = list(v)
            elif isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                safe_meta[k] = v
            else:
                safe_meta[k] = str(v)
        return {
            "ruleset": self.ruleset.to_dict(),
            "score": float(self.score),
            "atoms": int(self.atoms),
            "effective_nodes": int(self.effective_nodes),
            "rules": int(self.rules),
            "backend": str(self.backend),
            "metadata": safe_meta,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ParetoCandidate":
        rs = ScoredRuleSet.from_dict(payload["ruleset"])
        meta = dict(payload.get("metadata", {}))
        if "variant" in meta and isinstance(meta["variant"], list):
            meta["variant"] = tuple(meta["variant"])
        return cls(
            ruleset=rs,
            score=float(payload["score"]),
            atoms=int(payload["atoms"]),
            rules=int(payload["rules"]),
            backend=str(payload["backend"]),
            metadata=meta,
        )


class MasterParetoArchive:
    """Maintains a non-dominated Pareto front over complexity (atoms) and performance."""

    def __init__(
        self,
        higher_is_better: bool = True,
        compact_tolerance: float = 0.02,
        metric_name: str = "Validation Score",
        *,
        candidates_: list[ParetoCandidate] | None = None,
        candidate_pool_: list[ParetoCandidate] | None = None,
        metadata_: dict[str, Any] | None = None,
    ):
        self.higher_is_better = higher_is_better
        self.compact_tolerance = float(compact_tolerance)
        self.metric_name = str(metric_name)
        self.candidates_: list[ParetoCandidate] = list(candidates_) if candidates_ else []
        self.candidate_pool_: list[ParetoCandidate] = list(candidate_pool_) if candidate_pool_ else []
        self.metadata_: dict[str, Any] = dict(metadata_) if metadata_ else {}

    def _dominates(self, a_atoms: int, a_score: float, b_atoms: int, b_score: float) -> bool:
        """Return True if candidate A strictly dominates candidate B."""
        if self.higher_is_better:
            score_better_or_equal = a_score >= b_score
            score_strictly_better = a_score > b_score
        else:
            score_better_or_equal = a_score <= b_score
            score_strictly_better = a_score < b_score

        fewer_or_equal_atoms = a_atoms <= b_atoms
        strictly_fewer_atoms = a_atoms < b_atoms

        return (fewer_or_equal_atoms and score_better_or_equal) and (strictly_fewer_atoms or score_strictly_better)

    def add(
        self,
        ruleset: ScoredRuleSet,
        score: float,
        backend: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Add candidate to archive. Prunes dominated candidates. Returns True if admitted."""
        meta = dict(metadata or {})
        atoms = sum(len(r.atoms) for r in ruleset.rules)
        # Count non-default rules
        non_default_rules = len([r for r in ruleset.rules if r.atoms])

        # Check if dominated by any existing candidate
        for existing in self.candidates_:
            if self._dominates(existing.atoms, existing.score, atoms, score):
                return False

        # Remove existing candidates dominated by the new one
        self.candidates_ = [
            c for c in self.candidates_
            if not self._dominates(atoms, score, c.atoms, c.score)
        ]

        # Avoid exact duplicate points (same atoms and identical score)
        for c in self.candidates_:
            if c.atoms == atoms and abs(c.score - score) < 1e-9:
                return False

        self.candidates_.append(
            ParetoCandidate(
                ruleset=ruleset,
                score=score,
                atoms=atoms,
                rules=non_default_rules,
                backend=backend,
                metadata=meta,
            )
        )
        # Keep candidates sorted by complexity ascending
        self.candidates_.sort(key=lambda c: (c.atoms, -c.score if self.higher_is_better else c.score))
        return True

    def get_spectrum(self) -> list[dict[str, Any]]:
        """Return a structured summary list of the non-dominated Pareto front."""
        return [
            {
                "complexity": c.atoms,
                "effective_nodes": c.effective_nodes,
                "rules": c.rules,
                "score": c.score,
                "backend": c.backend,
                "ruleset": c.ruleset,
            }
            for c in self.candidates_
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the Master Pareto Archive to a JSON-compatible dictionary."""
        payload: dict[str, Any] = {
            "format": "master_pareto_archive",
            "version": "1.1",
            "higher_is_better": bool(self.higher_is_better),
            "compact_tolerance": float(self.compact_tolerance),
            "metric_name": str(self.metric_name),
            "metadata": dict(self.metadata_),
            "candidates": [c.to_dict() for c in self.candidates_],
        }
        if self.candidate_pool_:
            payload["candidate_pool"] = [c.to_dict() for c in self.candidate_pool_]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MasterParetoArchive":
        """Reconstruct a MasterParetoArchive from a dictionary produced by ``to_dict()``."""
        arch = cls(
            higher_is_better=bool(payload.get("higher_is_better", True)),
            compact_tolerance=float(payload.get("compact_tolerance", 0.02)),
            metric_name=str(payload.get("metric_name", "Validation Score")),
            metadata_=dict(payload.get("metadata", {})),
        )
        arch.candidates_ = [
            ParetoCandidate.from_dict(c) for c in payload.get("candidates", [])
        ]
        arch.candidate_pool_ = [
            ParetoCandidate.from_dict(c) for c in payload.get("candidate_pool", [])
        ]
        return arch

    def get_profile_summary(
        self,
        profiles: Sequence[str] = ("compact", "balanced", "performance", "expressive", "accuracy"),
    ) -> dict[str, dict[str, Any]]:
        """Return a dictionary summarizing the candidate selected by each operational profile."""
        if not self.candidates_:
            return {}
        summary: dict[str, dict[str, Any]] = {}
        for prof in profiles:
            cand = self.select_candidate(preference=prof)
            idx = next(
                (i for i, c in enumerate(self.candidates_) if c is cand or (c.atoms == cand.atoms and abs(c.score - cand.score) < 1e-12)),
                0,
            )
            summary[prof] = {
                "candidate_index": idx,
                "atoms": int(cand.atoms),
                "effective_nodes": int(cand.effective_nodes),
                "rules": int(cand.rules),
                "score": float(cand.score),
                "backend": str(cand.backend),
            }
        return summary

    def format_markdown_summary(self) -> str:
        """Render a human-readable Markdown summary of the Master Pareto Archive and its profile menu."""
        lines = [
            "# Master Pareto Archive Summary",
            "",
            f"- **Metric:** `{self.metric_name}` ({'higher is better' if self.higher_is_better else 'lower is better'})",
            f"- **Compact tolerance:** `{self.compact_tolerance}`",
            f"- **Non-dominated Pareto candidates:** `{len(self.candidates_)}`",
            f"- **Total harvested pool candidates:** `{len(self.candidate_pool_)}`",
            "",
            "## Post-Hoc Operational Profile Menu",
            "",
            "| Profile | Candidate # | Backend | Score | Path Atoms ($A$) | Split Nodes ($N_{\\text{split}}$) | Rules ($K$) |",
            "| :--- | ---: | :--- | ---: | ---: | ---: | ---: |",
        ]
        prof_summary = self.get_profile_summary()
        for prof, info in prof_summary.items():
            lines.append(
                f"| `{prof}` | {info['candidate_index']} | `{info['backend']}` | "
                f"{info['score']:.4f} | {info['atoms']} | {info['effective_nodes']} | {info['rules']} |"
            )
        lines.extend([
            "",
            "## Non-Dominated Master Pareto Frontier",
            "",
            "| # | Backend | Score | Path Atoms ($A$) | Split Nodes ($N_{\\text{split}}$) | Rules ($K$) |",
            "| ---: | :--- | ---: | ---: | ---: | ---: |",
        ])
        for idx, c in enumerate(self.candidates_):
            lines.append(
                f"| {idx} | `{c.backend}` | {c.score:.4f} | {c.atoms} | {c.effective_nodes} | {c.rules} |"
            )
        return "\n".join(lines) + "\n"

    def save_zip(
        self,
        path: str | Path,
        *,
        include_candidate_pool: bool = True,
        include_markdown_reports: bool = True,
    ) -> Path:
        """Save the entire MasterParetoArchive as a self-contained ZIP bundle.

        The ZIP bundle contains:
        - ``manifest.json``: Archive metadata, profile selections, and candidate descriptors.
        - ``candidates/candidate_XXX_<backend>_a<atoms>_r<rules>.json``: Individual
          JSON rule sets compatible with ``load_ruleset_json``.
        - ``candidates/candidate_XXX_<backend>_a<atoms>_r<rules>.md``: Human-readable
          Markdown index cards for each non-dominated rule set.
        - ``candidate_pool.json``: All harvested per-backend candidates (when available)
          so per-backend sub-fronts can be plotted post-hoc.
        - ``archive.json``: Complete single-file JSON representation.
        - ``SUMMARY.md``: Markdown report of the frontier and operational profiles.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        task_type = (
            self.candidates_[0].ruleset.task_type
            if self.candidates_
            else (self.candidate_pool_[0].ruleset.task_type if self.candidate_pool_ else "classification")
        )

        candidate_entries: list[dict[str, Any]] = []
        with zipfile.ZipFile(target, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for idx, cand in enumerate(self.candidates_):
                safe_backend = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(cand.backend)).strip("_") or "backend"
                stem = f"candidates/candidate_{idx:03d}_{safe_backend}_a{cand.atoms}_r{cand.rules}"
                rs_json_file = f"{stem}.json"
                rs_md_file = f"{stem}.md"

                zf.writestr(rs_json_file, json.dumps(cand.ruleset.to_dict(), indent=2))
                if include_markdown_reports:
                    try:
                        zf.writestr(rs_md_file, format_ruleset_markdown(cand.ruleset))
                    except Exception:
                        pass

                c_dict = cand.to_dict()
                candidate_entries.append({
                    "index": idx,
                    "ruleset_file": rs_json_file,
                    "markdown_file": rs_md_file if include_markdown_reports else None,
                    "score": c_dict["score"],
                    "atoms": c_dict["atoms"],
                    "effective_nodes": c_dict["effective_nodes"],
                    "rules": c_dict["rules"],
                    "backend": c_dict["backend"],
                    "metadata": c_dict["metadata"],
                })

            manifest = {
                "format": "master_pareto_archive_zip",
                "version": "1.1",
                "task_type": task_type,
                "higher_is_better": bool(self.higher_is_better),
                "compact_tolerance": float(self.compact_tolerance),
                "metric_name": str(self.metric_name),
                "metadata": dict(self.metadata_),
                "n_candidates": len(self.candidates_),
                "n_pool_candidates": len(self.candidate_pool_),
                "profiles": self.get_profile_summary(),
                "candidates": candidate_entries,
            }
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            zf.writestr("archive.json", json.dumps(self.to_dict(), indent=2))
            if include_candidate_pool and self.candidate_pool_:
                zf.writestr(
                    "candidate_pool.json",
                    json.dumps([c.to_dict() for c in self.candidate_pool_], indent=2),
                )
            if include_markdown_reports:
                zf.writestr("SUMMARY.md", self.format_markdown_summary())

        return target

    @classmethod
    def load_zip(cls, path: str | Path) -> "MasterParetoArchive":
        """Load a MasterParetoArchive from a ZIP bundle created by ``save_zip()``."""
        source = Path(path)
        with zipfile.ZipFile(source, mode="r") as zf:
            names = set(zf.namelist())
            if "manifest.json" in names:
                manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
                if manifest.get("format") not in ("master_pareto_archive_zip", "master_pareto_archive"):
                    raise ValueError(
                        f"Unsupported archive format in manifest.json: {manifest.get('format')!r}"
                    )
                arch = cls(
                    higher_is_better=bool(manifest.get("higher_is_better", True)),
                    compact_tolerance=float(manifest.get("compact_tolerance", 0.02)),
                    metric_name=str(manifest.get("metric_name", "Validation Score")),
                    metadata_=dict(manifest.get("metadata", {})),
                )
                candidates: list[ParetoCandidate] = []
                for entry in manifest.get("candidates", []):
                    rs_file = entry.get("ruleset_file")
                    if rs_file and rs_file in names:
                        rs_payload = json.loads(zf.read(rs_file).decode("utf-8"))
                    elif "ruleset" in entry:
                        rs_payload = entry["ruleset"]
                    else:
                        raise ValueError(f"Missing ruleset file {rs_file!r} in archive ZIP.")
                    cand_payload = {
                        "ruleset": rs_payload,
                        "score": entry["score"],
                        "atoms": entry["atoms"],
                        "rules": entry["rules"],
                        "backend": entry["backend"],
                        "metadata": entry.get("metadata", {}),
                    }
                    candidates.append(ParetoCandidate.from_dict(cand_payload))
                arch.candidates_ = candidates
                if "candidate_pool.json" in names:
                    pool_list = json.loads(zf.read("candidate_pool.json").decode("utf-8"))
                    arch.candidate_pool_ = [ParetoCandidate.from_dict(c) for c in pool_list]
                return arch

            if "archive.json" in names:
                payload = json.loads(zf.read("archive.json").decode("utf-8"))
                return cls.from_dict(payload)

            raise ValueError(f"ZIP file {source} does not contain 'manifest.json' or 'archive.json'.")

    def save(self, path: str | Path) -> Path:
        """Save the archive to ``path`` (ZIP format by default, or JSON if path ends with ``.json``)."""
        target = Path(path)
        if target.suffix.lower() == ".json":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
            return target
        return self.save_zip(target)

    @classmethod
    def load(cls, path: str | Path) -> "MasterParetoArchive":
        """Load a MasterParetoArchive from a ``.zip`` bundle or ``.json`` file."""
        source = Path(path)
        if zipfile.is_zipfile(source):
            return cls.load_zip(source)
        payload = json.loads(source.read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    def select(
        self,
        preference: str = "balanced",
        max_rules: int | None = None,
        max_atoms: int | None = None,
        *,
        max_effective_nodes: int | None = None,
        target_atoms: int | None = None,
        target_rules: int | None = None,
        target_effective_nodes: int | None = None,
        exact_size: bool = False,
    ) -> ScoredRuleSet:
        """Select a single ScoredRuleSet from the archive according to user intent or target size."""
        return self.select_candidate(
            preference=preference,
            max_rules=max_rules,
            max_atoms=max_atoms,
            max_effective_nodes=max_effective_nodes,
            target_atoms=target_atoms,
            target_rules=target_rules,
            target_effective_nodes=target_effective_nodes,
            exact_size=exact_size,
        ).ruleset

    def select_by_size(
        self,
        *,
        atoms: int | None = None,
        rules: int | None = None,
        effective_nodes: int | None = None,
        mode: Literal["closest", "max", "exact"] = "closest",
        preference: str = "accuracy",
        return_candidate: bool = False,
    ) -> ScoredRuleSet | ParetoCandidate:
        """Post-hoc model selection by fixed or bounded model size.

        Parameters
        ----------
        atoms : int | None
            Desired path-expanded atom count $A(c)$.
        rules : int | None
            Desired rule count $K(c)$.
        effective_nodes : int | None
            Desired effective split-node count $N_{\\text{split}}(c)$.
        mode : {"closest", "max", "exact"}, default="closest"
            - ``"closest"``: candidate minimizing absolute distance to the requested size
              (ties broken by best score, then smallest complexity).
            - ``"max"``: candidate selected by ``preference`` (default ``"accuracy"``)
              among all candidates whose complexity does not exceed the requested size budget(s).
            - ``"exact"``: highest-scoring candidate matching the exact requested size(s);
              raises ``ValueError`` if no candidate matches.
        preference : str, default="accuracy"
            Operational profile applied within the bounded subset when ``mode="max"``.
        return_candidate : bool, default=False
            If True, returns the ``ParetoCandidate`` wrapper; otherwise returns the ``ScoredRuleSet``.
        """
        if atoms is None and rules is None and effective_nodes is None:
            raise ValueError(
                "At least one size parameter ('atoms', 'rules', or 'effective_nodes') must be specified."
            )
        if mode == "max":
            cand = self.select_candidate(
                preference=preference,
                max_atoms=atoms,
                max_rules=rules,
                max_effective_nodes=effective_nodes,
            )
        elif mode == "exact":
            cand = self.select_candidate(
                target_atoms=atoms,
                target_rules=rules,
                target_effective_nodes=effective_nodes,
                exact_size=True,
            )
        elif mode == "closest":
            cand = self.select_candidate(
                target_atoms=atoms,
                target_rules=rules,
                target_effective_nodes=effective_nodes,
                exact_size=False,
            )
        else:
            raise ValueError(f"Unsupported size selection mode {mode!r}; expected 'closest', 'max', or 'exact'.")
        return cand if return_candidate else cand.ruleset

    def predict(
        self,
        X: Any,
        preference: str = "balanced",
        **select_kwargs: Any,
    ) -> np.ndarray:
        """Run inference directly using a model selected post-hoc from the archive."""
        rs = self.select(preference=preference, **select_kwargs)
        X_arr = np.asarray(X)
        if rs.task_type == "regression":
            return predict_regression_from_ruleset(rs, X_arr)
        return predict_from_ruleset(rs, X_arr)

    def predict_proba(
        self,
        X: Any,
        preference: str = "balanced",
        **select_kwargs: Any,
    ) -> np.ndarray:
        """Run probability inference directly using a classification model selected post-hoc."""
        rs = self.select(preference=preference, **select_kwargs)
        if rs.task_type == "regression":
            raise ValueError("predict_proba is not available for regression archives.")
        return predict_proba_from_ruleset(rs, np.asarray(X))

    def _signed(self, score: float) -> float:
        return score if self.higher_is_better else -score

    def _select_compact(self, eligible: list[ParetoCandidate]) -> ParetoCandidate:
        """Most parsimonious candidate within ``compact_tolerance`` of the best score."""
        best = max(self._signed(c.score) for c in eligible)
        within = [c for c in eligible if self._signed(c.score) >= best - self.compact_tolerance]
        return min(within, key=lambda c: (c.atoms, -self._signed(c.score)))

    def _select_balanced(self, eligible: list[ParetoCandidate]) -> ParetoCandidate:
        """Knee point in normalized objective space (maximal gain above chord)."""
        if len(eligible) <= 2:
            return self._select_compact(eligible)

        atoms_arr = np.array([c.atoms for c in eligible], dtype=float)
        scores_arr = np.array([self._signed(c.score) for c in eligible], dtype=float)
        a_span = atoms_arr[-1] - atoms_arr[0]
        s_span = scores_arr[-1] - scores_arr[0]
        if a_span <= 0 or s_span <= 1e-12:
            return self._select_compact(eligible)

        # After min-max normalization the extremes are (0, 0) and (1, 1); the
        # signed distance above the chord y = x is proportional to (s - a).
        a_norm = (atoms_arr - atoms_arr[0]) / a_span
        s_norm = (scores_arr - scores_arr[0]) / s_span
        gain = s_norm - a_norm
        best_idx = int(np.argmax(gain))
        if gain[best_idx] <= 1e-12:
            return self._select_compact(eligible)
        return eligible[best_idx]

    def _select_performance(
        self, eligible: list[ParetoCandidate], weight: float = 0.70
    ) -> ParetoCandidate:
        """Asymmetric knee / weighted trade-off favoring performance (default weight=0.70)."""
        if len(eligible) <= 2:
            return max(eligible, key=lambda c: (self._signed(c.score), -c.atoms))

        atoms_arr = np.array([c.atoms for c in eligible], dtype=float)
        scores_arr = np.array([self._signed(c.score) for c in eligible], dtype=float)
        a_span = atoms_arr[-1] - atoms_arr[0]
        s_span = scores_arr[-1] - scores_arr[0]
        if a_span <= 0 or s_span <= 1e-12:
            return max(eligible, key=lambda c: (self._signed(c.score), -c.atoms))

        a_norm = (atoms_arr - atoms_arr[0]) / a_span
        s_norm = (scores_arr - scores_arr[0]) / s_span
        gain = weight * s_norm - (1.0 - weight) * a_norm
        best_idx = int(np.argmax(gain))
        cand = eligible[best_idx]
        bal = self._select_balanced(eligible)
        if cand.atoms < bal.atoms:
            return bal
        return cand

    def _select_expressive(self, eligible: list[ParetoCandidate]) -> ParetoCandidate:
        """Parsimonious high-performance candidate within tighter tolerance (0.005) of apex, bounded below by balanced."""
        best = max(self._signed(c.score) for c in eligible)
        tight_tolerance = max(0.005, self.compact_tolerance * 0.25)
        within = [c for c in eligible if self._signed(c.score) >= best - tight_tolerance]
        cand = min(within, key=lambda c: (c.atoms, -self._signed(c.score)))
        bal = self._select_balanced(eligible)
        if cand.atoms < bal.atoms:
            return bal
        return cand

    def select_candidate(
        self,
        preference: str = "balanced",
        max_rules: int | None = None,
        max_atoms: int | None = None,
        *,
        max_effective_nodes: int | None = None,
        target_atoms: int | None = None,
        target_rules: int | None = None,
        target_effective_nodes: int | None = None,
        exact_size: bool = False,
    ) -> ParetoCandidate:
        """Select a single archive candidate according to user intent or size specification.

        - ``compact``: fewest atoms among candidates whose score lies within
          ``compact_tolerance`` of the best score, bounded above by ``balanced``
          to guarantee monotonicity (atoms(compact) <= atoms(balanced)).
        - ``balanced``: knee point (maximal signed vertical gain above the chord).
        - ``performance``: asymmetric 70/30 knee favoring predictive accuracy,
          bounded below by ``balanced``.
        - ``expressive``: fewest atoms within tight apex tolerance (0.005),
          bounded below by ``balanced``.
        - ``accuracy`` / ``manual`` / ``best_single_cv``: highest score.
        - Fixed / target size: if ``target_atoms``, ``target_rules``, or
          ``target_effective_nodes`` is specified, selects the candidate matching
          (or closest to) the requested size, breaking ties by highest score.
        """
        if not self.candidates_:
            raise RuntimeError("MasterParetoArchive is empty; no models available.")

        eligible = list(self.candidates_)
        if max_rules is not None:
            eligible = [c for c in eligible if c.rules <= max_rules]
        if max_atoms is not None:
            eligible = [c for c in eligible if c.atoms <= max_atoms]
        if max_effective_nodes is not None:
            eligible = [c for c in eligible if c.effective_nodes <= max_effective_nodes]

        if not eligible:
            warnings.warn(
                f"No models satisfy constraints (max_rules={max_rules}, max_atoms={max_atoms}, "
                f"max_effective_nodes={max_effective_nodes}); falling back to unconstrained archive.",
                UserWarning,
            )
            eligible = list(self.candidates_)

        eligible = sorted(eligible, key=lambda c: (c.atoms, -self._signed(c.score)))

        # Fixed / target size selection takes precedence when explicitly requested
        if (
            target_atoms is not None
            or target_rules is not None
            or target_effective_nodes is not None
        ):
            if exact_size:
                exact_matches = [
                    c for c in eligible
                    if (target_atoms is None or c.atoms == target_atoms)
                    and (target_rules is None or c.rules == target_rules)
                    and (target_effective_nodes is None or c.effective_nodes == target_effective_nodes)
                ]
                if not exact_matches:
                    raise ValueError(
                        f"No candidate in MasterParetoArchive matches exact size "
                        f"(target_atoms={target_atoms}, target_rules={target_rules}, "
                        f"target_effective_nodes={target_effective_nodes})."
                    )
                return max(exact_matches, key=lambda c: (self._signed(c.score), -c.atoms))

            def _size_dist_key(c: ParetoCandidate) -> tuple[float, float, int]:
                dist = 0.0
                if target_atoms is not None:
                    dist += abs(c.atoms - target_atoms)
                if target_rules is not None:
                    dist += abs(c.rules - target_rules)
                if target_effective_nodes is not None:
                    dist += abs(c.effective_nodes - target_effective_nodes)
                return (dist, -self._signed(c.score), c.atoms)

            return min(eligible, key=_size_dist_key)

        if preference == "compact":
            cand = self._select_compact(eligible)
            # Monotonicity guard: when using a positive tolerance, a user requesting
            # "compact" expects a model at least as parsimonious as "balanced".
            if self.compact_tolerance > 0.0:
                bal = self._select_balanced(eligible)
                if cand.atoms > bal.atoms:
                    return bal
            return cand

        if preference in ("performance", "performance_tilted"):
            return self._select_performance(eligible, weight=0.70)

        if preference in ("expressive", "upper_knee"):
            return self._select_expressive(eligible)

        if preference in ("accuracy", "manual", "custom", "best_single_cv"):
            return max(eligible, key=lambda c: (self._signed(c.score), -c.atoms))

        # Default: "balanced" (knee point in normalized objective space)
        bal = self._select_balanced(eligible)
        acc = max(eligible, key=lambda c: (self._signed(c.score), -c.atoms))
        if bal.atoms > acc.atoms:
            return acc
        return bal

    @staticmethod
    def _candidate_complexity(
        c: ParetoCandidate,
        complexity_metric: Literal["atoms", "effective_nodes", "rules"],
    ) -> int:
        if complexity_metric == "effective_nodes":
            return int(c.effective_nodes)
        if complexity_metric == "rules":
            return int(c.rules)
        return int(c.atoms)

    def plot_pareto_front(
        self,
        output_path: str | Path | None = None,
        title: str = "Master Pareto Frontier",
        metric_name: str | None = None,
        *,
        complexity_metric: Literal["atoms", "effective_nodes", "rules"] = "atoms",
        profiles: Sequence[str] = ("compact", "balanced", "performance", "expressive", "accuracy"),
        show_backend_fronts: bool = True,
        highlight_size: int | None = None,
        ax: Any | None = None,
    ):
        """Plot the non-dominated Pareto front with post-hoc intent and size highlights.

        Parameters
        ----------
        output_path : str | Path | None
            If provided, saves the figure to this file (PDF, PNG, SVG).
        title : str
            Title of the plot.
        metric_name : str | None
            Y-axis label (defaults to ``self.metric_name``).
        complexity_metric : {"atoms", "effective_nodes", "rules"}, default="atoms"
            Structural complexity axis to plot on the X-axis.
        profiles : Sequence[str]
            Operational intent profiles to highlight on the frontier.
        show_backend_fronts : bool, default=True
            If True and ``self.candidate_pool_`` is populated, plots per-backend
            Pareto fronts as dashed step curves beneath the Master Pareto Envelope.
        highlight_size : int | None, default=None
            If provided, highlights the candidate closest to ``highlight_size`` on the
            chosen ``complexity_metric`` axis.
        ax : matplotlib.axes.Axes | None, default=None
            Optional existing Axes to draw on.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes objects
        """
        import matplotlib.pyplot as plt

        if not self.candidates_:
            raise RuntimeError("Cannot plot empty MasterParetoArchive.")

        y_label = metric_name or self.metric_name or "Validation Score"
        if ax is None:
            fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
        else:
            fig = ax.figure

        all_cands_for_colors = list(self.candidate_pool_) + list(self.candidates_)
        unique_backends = sorted({c.backend for c in all_cands_for_colors})
        cmap = plt.get_cmap("tab10")
        color_map = {b: cmap(i % 10) for i, b in enumerate(unique_backends)}

        # Optional per-backend sub-fronts from candidate_pool_
        if show_backend_fronts and self.candidate_pool_:
            for b in unique_backends:
                b_pool = [c for c in self.candidate_pool_ if c.backend == b]
                if not b_pool:
                    continue
                pts = sorted(
                    [(self._candidate_complexity(c, complexity_metric), c.score) for c in b_pool],
                    key=lambda p: (p[0], -self._signed(p[1])),
                )
                front: list[tuple[int, float]] = []
                best_s = -float("inf")
                for comp_val, sc_val in pts:
                    signed_s = self._signed(sc_val)
                    if signed_s > best_s:
                        front.append((comp_val, sc_val))
                        best_s = signed_s
                if len(front) > 1:
                    ax.step(
                        [p[0] for p in front],
                        [p[1] for p in front],
                        where="post",
                        color=color_map[b],
                        linestyle=":",
                        linewidth=1.3,
                        alpha=0.65,
                        zorder=2,
                    )

        # Plot all Master Pareto candidates by backend
        for b in unique_backends:
            b_cands = [c for c in self.candidates_ if c.backend == b]
            if not b_cands:
                continue
            ax.scatter(
                [self._candidate_complexity(c, complexity_metric) for c in b_cands],
                [c.score for c in b_cands],
                label=f"Backend: {b}",
                color=color_map[b],
                s=70,
                alpha=0.88,
                edgecolors="white",
                linewidths=0.6,
                zorder=4,
            )

        # Plot Master Pareto step envelope
        sorted_cands = sorted(
            self.candidates_,
            key=lambda c: (self._candidate_complexity(c, complexity_metric), -self._signed(c.score)),
        )
        x_steps = [self._candidate_complexity(c, complexity_metric) for c in sorted_cands]
        y_steps = [c.score for c in sorted_cands]
        ax.step(
            x_steps,
            y_steps,
            where="post",
            color="#222222",
            linestyle="--",
            linewidth=1.8,
            alpha=0.75,
            zorder=3,
            label="Pareto Envelope",
        )

        profile_styles: dict[str, dict[str, Any]] = {
            "compact":     {"color": "#1b9e77", "marker": "o", "size": 155, "short": "c"},
            "balanced":    {"color": "#e41a1c", "marker": "s", "size": 195, "short": "b"},
            "performance": {"color": "#1f77b4", "marker": "D", "size": 175, "short": "p"},
            "expressive":  {"color": "#d95f02", "marker": "P", "size": 185, "short": "e"},
            "accuracy":    {"color": "#7570b3", "marker": "^", "size": 175, "short": "a"},
        }

        for prof in profiles:
            try:
                p_cand = self.select_candidate(prof)
                style = profile_styles.get(
                    prof,
                    {"color": "black", "marker": "o", "size": 160, "short": prof[:1]},
                )
                x_val = self._candidate_complexity(p_cand, complexity_metric)
                ax.scatter(
                    [x_val],
                    [p_cand.score],
                    s=style["size"],
                    facecolors="none",
                    edgecolors=style["color"],
                    linewidths=2.1,
                    marker=style["marker"],
                    label=f"Intent: {prof} ({x_val})",
                    zorder=5,
                )
            except Exception:
                pass

        if highlight_size is not None:
            try:
                size_kwargs: dict[str, Any] = {complexity_metric: int(highlight_size), "return_candidate": True}
                s_cand = self.select_by_size(**size_kwargs)
                assert isinstance(s_cand, ParetoCandidate)
                sx = self._candidate_complexity(s_cand, complexity_metric)
                ax.scatter(
                    [sx],
                    [s_cand.score],
                    s=230,
                    facecolors="none",
                    edgecolors="#ff9900",
                    linewidths=2.6,
                    marker="*",
                    label=f"Size target ({complexity_metric}={highlight_size} -> {sx})",
                    zorder=6,
                )
            except Exception:
                pass

        x_labels = {
            "atoms": "Complexity (Total Path Atoms $A$ in Rule Set)",
            "effective_nodes": "Complexity (Effective Split Nodes $N_{\\mathrm{split}}$)",
            "rules": "Complexity (Non-Default Rules $K$)",
        }
        ax.set_xlabel(x_labels.get(complexity_metric, "Complexity"), fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(frameon=True, fontsize=8, loc="best")
        fig.tight_layout()

        if output_path is not None:
            fig.savefig(output_path, bbox_inches="tight")

        return fig, ax


class AutoScoredRuleSetClassifier(BaseRuleSetEstimator, ClassifierMixin):
    """AutoML meta-estimator for Scored Rule Sets classification.

    Evaluates candidate backends, harvests non-dominated solutions into a
    Master Pareto Archive, and selects the optimal model matching the user's
    operational intent (compact, balanced, accuracy, pareto_menu).

    Parameters
    ----------
    candidate_backends : list[str] | None
        List of backend names to evaluate (default: ``["greedy_pareto", "cart",
        "hs", "ruleplcs"]``).  Additional backends are reachable on request:
        ``rulenln`` (gradient-based; when no ``max_atoms_per_rule`` is pinned,
        an architectural top-k sweep k in {2, 3, 4, 6} is harvested into the
        Master Pareto Archive), ``rulegp`` / ``rulensga2`` (evolutionary) and
        ``exact`` (certified CP-SAT; requires ``scoredrulesets[exact]``).
        ``rulenln`` was demoted from the default after a three-generation
        fusion study showed zero non-dominated contributions once the greedy
        beam search's rule<->weight pairing defect was repaired; the
        evolutionary and exact families are excluded for runtime reasons.
    preference : {"compact", "balanced", "performance", "expressive", "accuracy", "pareto_menu", "manual", "best_single_cv"}, default="balanced"
        Operational intent profile (all evaluated on out-of-fold archive scores):
        - "compact": fewest atoms within ``compact_tolerance`` of the best score.
        - "balanced": knee point of the normalized Pareto front (falls back to
          "compact" when the front has fewer than three points).
        - "performance": asymmetric weighted knee (w=0.70) favoring predictive power
          while penalizing gratuitous complexity explosions.
        - "expressive": parsimonious high-performance candidate within tighter tolerance
          (0.005) of the apex, bounded below by "balanced".
        - "accuracy": highest out-of-fold score on the Master Pareto Front.
        - "best_single_cv": classical scalar CV winner across candidate backends,
          activating the top-scoring single-backend model without Pareto pruning.
        - "pareto_menu": fits full front for interactive post-fit inspection.
    compact_tolerance : float, default=0.02
        Score tolerance used by the "compact" intent (and the "balanced"
        fallback), in units of the archive metric (macro-F1).
    enable_pareto_fusion : bool, default=True
        Whether to merge non-dominated rule sets across all evaluated backends
        into a unified Master Pareto Front.
    max_rules : int | None, default=None
        Upper bound on number of rules (for preference="manual").
    max_atoms : int | None, default=None
        Upper bound on total atomic conditions (for preference="manual").
    backend_params : dict[str, dict] | None
        Per-backend constructor parameters.
    cv : int, default=5
        Number of cross-validation folds.
    scoring : str, default="f1_weighted"
        Sklearn scoring metric.
    feature_selection : {"auto", "none", "kbest_mi", "kbest_f", "variance", "tree"} | Any, default="auto"
        Upstream search-space reduction strategy. When set to "auto", automatically
        activates mutual-information feature selection if n_features >= 50 or
        n_features > n_samples, preventing combinatorial collapse on high-dimensional
        or omics data (grounded in bioinformatics findings).
    max_features : int | float | None, default=None
        Maximum features to retain when feature selection is active.
        If float between 0.0 and 1.0, treated as percentage of input features.
        If None, adaptively scales (min 10, max 50).
    preprocessing : dict | None
        Preprocessing configuration forwarded to backends.
    timeout_per_backend : float | None
        Maximum seconds for a single backend's CV loop.
    random_state : int | None
        Random seed for cross-validation splits.
    """

    def __init__(
        self,
        candidate_backends: list[str] | None = None,
        preference: Literal["compact", "balanced", "performance", "expressive", "accuracy", "pareto_menu", "manual", "best_single_cv"] = "balanced",
        enable_pareto_fusion: bool = True,
        max_rules: int | None = None,
        max_atoms: int | None = None,
        backend_params: dict[str, dict[str, Any]] | None = None,
        cv: int = 5,
        scoring: str = "f1_weighted",
        feature_selection: Literal["auto", "none", "kbest_mi", "kbest_f", "variance", "tree"] | Any = "auto",
        max_features: int | float | None = None,
        preprocessing: dict[str, Any] | None = None,
        timeout_per_backend: float | None = None,
        probing_strategy: Literal["none", "subsample", "hv_contribution"] = "none",
        probing_subsample: float = 0.25,
        probing_threshold_samples: int = 500,
        compact_tolerance: float = 0.02,
        random_state: int | None = None,
    ):
        self.candidate_backends = candidate_backends
        self.preference = preference
        self.enable_pareto_fusion = enable_pareto_fusion
        self.max_rules = max_rules
        self.max_atoms = max_atoms
        self.backend_params = backend_params
        self.cv = cv
        self.scoring = scoring
        self.feature_selection = feature_selection
        self.max_features = max_features
        self.preprocessing = preprocessing
        self.timeout_per_backend = timeout_per_backend
        self.probing_strategy = probing_strategy
        self.probing_subsample = probing_subsample
        self.probing_threshold_samples = probing_threshold_samples
        self.compact_tolerance = compact_tolerance
        self.random_state = random_state

    def _resolve_feature_selection(self, n_samples: int, n_features: int) -> dict[str, Any] | None:
        """Derive upstream feature selection settings based on data dimensionality.

        Grounded in bioinformatics and high-dimensional omics findings, rule induction
        in high-D regimes (D >= 50 or D > N) collapses in runtime, predictive quality,
        or model size unless coarse variable-level selection is applied prior to
        candidate atom construction.
        """
        fs = self.feature_selection
        if fs in (None, False, "none"):
            return None

        if not isinstance(fs, str):
            return {"feature_selector": fs}

        fs_str = fs.lower()
        if fs_str == "auto":
            if n_features < 50 and n_features <= n_samples:
                return None
            method = "kbest"
            from sklearn.feature_selection import mutual_info_classif
            score_func = mutual_info_classif
        elif fs_str in ("kbest_mi", "kbest"):
            method = "kbest"
            from sklearn.feature_selection import mutual_info_classif
            score_func = mutual_info_classif
        elif fs_str == "kbest_f":
            method = "kbest"
            from sklearn.feature_selection import f_classif
            score_func = f_classif
        elif fs_str == "variance":
            from sklearn.feature_selection import VarianceThreshold
            return {"feature_selector": VarianceThreshold()}
        elif fs_str == "tree":
            method = "boruta"
            score_func = None
        else:
            method = fs_str
            score_func = None

        if self.max_features is None:
            # Default heuristic: choose k in [10, 50], preserving sub-second tractability
            k = min(n_features, max(10, min(50, max(20, n_samples // 4))))
        elif isinstance(self.max_features, float) and 0.0 < self.max_features <= 1.0:
            k = max(2, int(n_features * self.max_features))
        else:
            k = min(n_features, int(self.max_features))

        cfg: dict[str, Any] = {"feature_selection": method, "k": k}
        if score_func is not None:
            cfg["feature_selection_params"] = {"score_func": score_func}
        return cfg

    def fit(self, X, y):
        X_valid, y_valid = check_X_y(X, y, dtype=None)
        self.n_features_in_ = X_valid.shape[1]

        # Resolve automated upstream search-space reduction
        effective_preprocessing = dict(self.preprocessing or {})
        if "feature_selection" not in effective_preprocessing and "feature_selector" not in effective_preprocessing:
            fs_cfg = self._resolve_feature_selection(len(X_valid), self.n_features_in_)
            if fs_cfg is not None:
                effective_preprocessing.update(fs_cfg)
        self.effective_preprocessing_ = effective_preprocessing

        backends = list(self.candidate_backends or _DEFAULT_CLASSIFIER_BACKENDS)
        per_backend_params = dict(self.backend_params or {})

        # Multi-fidelity probing to filter unpromising backends on larger datasets.
        # ``subsample`` ranks by mean held-out score; ``hv_contribution`` ranks by
        # the hypervolume a backend adds to the pooled probe front (leave-one-out).
        # The latter is fusion-aware: an engine that is weak on average but owns
        # one end of the complexity spectrum (e.g. greedy at the compact end)
        # survives HV probing and is evicted by mean-score probing.
        if (
            self.probing_strategy in ("subsample", "hv_contribution")
            and len(X_valid) >= self.probing_threshold_samples
            and len(backends) > 2
        ):
            probe_size = max(50, int(len(X_valid) * self.probing_subsample))
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, train_size=probe_size, random_state=self.random_state)
            probe_idx, _ = next(sss.split(X_valid, y_valid))
            X_probe, y_probe = X_valid[probe_idx], y_valid[probe_idx]
            probe_splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=self.random_state)

            def _probe_factory(backend: str, bp: dict[str, Any] | None):
                return lambda: ScoredRuleSetClassifier(
                    backend=backend,
                    backend_params=bp,
                    preprocessing=self.effective_preprocessing_,
                    random_state=self.random_state,
                )

            probe_ranks: dict[str, float] = {}
            if self.probing_strategy == "hv_contribution":
                factories = {}
                for backend in backends:
                    bp = per_backend_params.get(backend)
                    if bp is None and backend == "cart":
                        bp = {"max_depth": 3}
                    factories[backend] = _probe_factory(backend, bp)
                try:
                    probe_ranks = _probe_hv_contribution(
                        factories, X_probe, y_probe, probe_splitter,
                        archive_metric=_archive_metric_from_scoring(self.scoring),
                        predict_fn=predict_from_ruleset,
                    )
                except Exception:
                    probe_ranks = {}
            if not probe_ranks:  # "subsample" or HV probing failed -> mean-score fallback
                for backend in backends:
                    bp = per_backend_params.get(backend)
                    if bp is None and backend == "cart":
                        bp = {"max_depth": 3}
                    try:
                        sc = cross_val_score(
                            _probe_factory(backend, bp)(), X_probe, y_probe,
                            cv=probe_splitter,
                            scoring=self.scoring,
                            error_score="raise",
                        )
                        probe_ranks[backend] = float(np.mean(sc))
                    except Exception:
                        probe_ranks[backend] = float("-inf")

            sorted_backends = sorted(backends, key=lambda b: probe_ranks.get(b, float("-inf")), reverse=True)
            cutoff = max(2, len(backends) // 2)
            survivors = [b for b in sorted_backends[:cutoff] if probe_ranks.get(b, float("-inf")) > float("-inf")]
            if survivors:
                backends = survivors

        cv_results: dict[str, float] = {}
        best_backend: str | None = None
        best_score = -np.inf

        cv_splitter = StratifiedKFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )
        scorer = get_scorer(self.scoring)
        # Archive metric follows the estimator's scoring so the Pareto front
        # and the backend ranking never disagree on which candidate is best
        # (review finding: macro-F1 archive vs f1_weighted ranking flipped the
        # Ionosphere accuracy pick by 0.004).
        archive_metric = _archive_metric_from_scoring(self.scoring)

        self.master_archive_ = MasterParetoArchive(
            higher_is_better=True,
            compact_tolerance=self.compact_tolerance,
            metric_name=str(self.scoring or "Validation Score"),
        )
        self.all_harvested_candidates_: list[ParetoCandidate] = []
        self.backend_timings_: dict[str, dict[str, float]] = {}
        fitted_estimators: dict[str, Any] = {}

        for backend in backends:
            bp = per_backend_params.get(backend)
            if bp is None and backend == "cart":
                bp = {"max_depth": 3}

            # Configurations evaluated for this backend.  The neural backend
            # additionally contributes an architectural top-k sweep; each k is
            # its own candidate point and the archive prunes dominated widths.
            configs: list[tuple[str, dict[str, Any] | None]] = [(backend, bp)]
            if backend == "rulenln" and not (bp or {}).get("max_atoms_per_rule"):
                configs += [
                    (f"rulenln(k={k})", {**(bp or {}), "max_atoms_per_rule": k})
                    for k in _RULENLN_SWEEP_K
                ]

            for label, params in configs:
                is_main = label == backend

                def make(params=params, backend=backend):
                    return ScoredRuleSetClassifier(
                        backend=backend,
                        backend_params=params,
                        preprocessing=self.effective_preprocessing_,
                        random_state=self.random_state,
                    )

                t0 = time.monotonic()
                try:
                    main_score, per_key = _oof_evaluate_config(
                        make, X_valid, y_valid, cv_splitter,
                        main_scorer=scorer if is_main else None,
                        archive_metric=archive_metric,
                        predict_fn=predict_from_ruleset,
                    )
                except Exception as exc:  # noqa: BLE001
                    if is_main:
                        warnings.warn(
                            f"AutoScoredRuleSet: backend '{backend}' failed during CV: {exc}",
                            UserWarning,
                        )
                        cv_results[backend] = float("-inf")
                        break  # skip sweep variants of a failing backend
                    warnings.warn(
                        f"AutoScoredRuleSet: {label} harvest failed: {exc}",
                        UserWarning,
                    )
                    continue

                if is_main:
                    elapsed = time.monotonic() - t0
                    self.backend_timings_.setdefault(backend, {})["cv_s"] = float(elapsed)
                    if self.timeout_per_backend is not None and elapsed > self.timeout_per_backend:
                        warnings.warn(
                            f"AutoScoredRuleSet: backend '{backend}' exceeded "
                            f"timeout ({elapsed:.1f}s > {self.timeout_per_backend:.1f}s).",
                            UserWarning,
                        )
                    cv_results[backend] = main_score
                    if main_score > best_score:
                        best_score = main_score
                        best_backend = backend

                if not (self.enable_pareto_fusion or is_main):
                    continue
                # Refit on all data for serving; archive scores stay OOF.
                try:
                    t_refit = time.monotonic()
                    est_full = make()
                    est_full.fit(X_valid, y_valid)
                    refit_elapsed = time.monotonic() - t_refit
                    if is_main:
                        fitted_estimators[backend] = est_full
                        self.backend_timings_.setdefault(backend, {})["refit_s"] = float(refit_elapsed)
                    if self.enable_pareto_fusion:
                        _admit_full_fit(
                            self.master_archive_, est_full, label, per_key,
                            n_splits=cv_splitter.get_n_splits(),
                            candidate_pool=self.all_harvested_candidates_,
                        )
                except Exception as exc:  # noqa: BLE001
                    warnings.warn(
                        f"AutoScoredRuleSet: failed to harvest Pareto models from '{label}': {exc}",
                        UserWarning,
                    )

        self.fitted_estimators_ = fitted_estimators

        if best_backend is None:
            raise RuntimeError(
                "AutoScoredRuleSet: all candidate backends failed during "
                "cross-validation. Check warnings for details."
            )

        # Refit or retrieve winner
        if best_backend in fitted_estimators:
            winner = fitted_estimators[best_backend]
        else:
            bp = per_backend_params.get(best_backend)
            winner = ScoredRuleSetClassifier(
                backend=best_backend,
                backend_params=bp,
                preprocessing=self.effective_preprocessing_,
                random_state=self.random_state,
            )
            winner.fit(X_valid, y_valid)

        self.best_backend_ = best_backend
        self.best_score_ = best_score
        self.cv_results_ = cv_results
        self.best_estimator_ = winner
        self.classes_ = winner.classes_
        self.feature_names_in_ = winner.feature_names_in_
        self.selected_feature_indices_ = getattr(winner, "selected_feature_indices_", None)
        self.feature_selector_ = getattr(winner, "feature_selector_", None)

        # Select model according to preference and Pareto fusion
        if self.preference == "best_single_cv":
            self.ruleset_ = winner.ruleset_
            self.active_estimator_ = winner
            self.active_backend_ = best_backend
        elif self.enable_pareto_fusion and self.master_archive_.candidates_:
            chosen = self.master_archive_.select_candidate(
                preference=self.preference,
                max_rules=self.max_rules,
                max_atoms=self.max_atoms,
            )
            self._activate(chosen)
        else:
            self.ruleset_ = winner.ruleset_
            self.active_estimator_ = winner
            self.active_backend_ = best_backend

        return self

    def _activate(self, candidate: ParetoCandidate) -> None:
        self.active_candidate_ = candidate
        self.ruleset_ = candidate.ruleset
        self.active_estimator_ = candidate.metadata.get("estimator", getattr(self, "best_estimator_", None))
        self.active_backend_ = candidate.backend

    def get_pareto_spectrum(self) -> list[dict[str, Any]]:
        """Return the Master Pareto spectrum of non-dominated rule sets."""
        check_is_fitted(self, ["master_archive_"])
        return self.master_archive_.get_spectrum()

    def set_active_model(
        self,
        complexity: int | None = None,
        *,
        preference: str | None = None,
        max_atoms: int | None = None,
        max_rules: int | None = None,
        max_effective_nodes: int | None = None,
        target_atoms: int | None = None,
        target_rules: int | None = None,
        target_effective_nodes: int | None = None,
        exact_size: bool = False,
    ):
        """Switch active rule set post-hoc by operational intent profile or model size."""
        check_is_fitted(self, ["master_archive_"])
        candidates = self.master_archive_.candidates_
        if not candidates:
            raise RuntimeError("No models available in Master Pareto Archive.")

        if complexity is not None and target_atoms is None:
            target_atoms = int(complexity)

        chosen = self.master_archive_.select_candidate(
            preference=preference or self.preference,
            max_atoms=max_atoms,
            max_rules=max_rules,
            max_effective_nodes=max_effective_nodes,
            target_atoms=target_atoms,
            target_rules=target_rules,
            target_effective_nodes=target_effective_nodes,
            exact_size=exact_size,
        )
        self._activate(chosen)
        return self

    def save_archive(self, path: str | Path) -> Path:
        """Save the fitted MasterParetoArchive to a ZIP bundle (or JSON if ``.json``)."""
        check_is_fitted(self, ["master_archive_"])
        return self.master_archive_.save(path)

    @classmethod
    def from_archive(
        cls,
        archive_or_path: str | Path | MasterParetoArchive,
        preference: str = "balanced",
        **select_kwargs: Any,
    ) -> "AutoScoredRuleSetClassifier":
        """Instantiate a ready-to-predict classifier post-hoc from a saved MasterParetoArchive."""
        archive = (
            archive_or_path
            if isinstance(archive_or_path, MasterParetoArchive)
            else MasterParetoArchive.load(archive_or_path)
        )
        if not archive.candidates_:
            raise ValueError("Cannot instantiate AutoScoredRuleSetClassifier from an empty archive.")
        est = cls(preference=preference, compact_tolerance=archive.compact_tolerance)  # type: ignore[arg-type]
        est.master_archive_ = archive
        est.all_harvested_candidates_ = list(archive.candidate_pool_) or list(archive.candidates_)
        chosen = archive.select_candidate(preference=preference, **select_kwargs)
        est.best_estimator_ = None
        est.best_backend_ = chosen.backend
        est.best_score_ = chosen.score
        est.classes_ = np.asarray(chosen.ruleset.class_labels)
        est.feature_names_in_ = np.asarray(chosen.ruleset.feature_names) if chosen.ruleset.feature_names else None
        est.n_features_in_ = len(chosen.ruleset.feature_names) if chosen.ruleset.feature_names else None
        est._activate(chosen)
        return est

    def export_index_card(self) -> str:
        """Export active model formatted as a human-simulatable index card."""
        check_is_fitted(self, ["ruleset_"])
        return format_ruleset_markdown(self.ruleset_)

    def _prepare_X_for_prediction(self, X) -> np.ndarray:
        """Transform X into the input space of the estimator that produced the active rule set."""
        est = getattr(self, "active_estimator_", None) or getattr(self, "best_estimator_", None)
        return _transform_for_estimator(est, X)

    def predict(self, X):
        check_is_fitted(self, ["ruleset_"])
        X_prep = self._prepare_X_for_prediction(X)
        return predict_from_ruleset(self.ruleset_, X_prep)

    def predict_proba(self, X):
        check_is_fitted(self, ["ruleset_"])
        X_prep = self._prepare_X_for_prediction(X)
        return predict_proba_from_ruleset(self.ruleset_, X_prep)

    def to_ruleset(self) -> ScoredRuleSet:
        check_is_fitted(self, ["ruleset_"])
        return self.ruleset_

    def plot_pareto_front(self, output_path: str | Path | None = None, title: str | None = None, **kwargs: Any):
        """Plot the non-dominated Master Pareto front."""
        check_is_fitted(self, ["master_archive_"])
        t = title or f"{self.__class__.__name__} Master Pareto Frontier"
        metric = "Validation Score" if self.scoring is None else str(self.scoring)
        return self.master_archive_.plot_pareto_front(output_path=output_path, title=t, metric_name=metric, **kwargs)


class AutoScoredRuleSetRegressor(RegressorMixin, BaseRuleSetEstimator):
    """AutoML meta-estimator for Scored Rule Sets regression.

    Evaluates candidate regression backends and manages the Master Pareto Archive.

    Parameters
    ----------
    candidate_backends : list[str] | None
        List of backend names to evaluate (default: ``["greedy_residual",
        "greedy_pareto", "cart", "ruleplcs"]``).
    preference : {"compact", "balanced", "performance", "expressive", "accuracy", "pareto_menu", "manual", "best_single_cv"}, default="balanced"
        Operational intent profile:
        - "compact": fewest atoms within ``compact_tolerance`` of the best score.
        - "balanced": knee point of the normalized Pareto front.
        - "performance": asymmetric weighted knee (w=0.70) favoring predictive power.
        - "expressive": parsimonious high-performance candidate within tighter tolerance (0.005) of apex.
        - "accuracy": highest out-of-fold score on the Master Pareto Front.
        - "best_single_cv": classical scalar CV winner across candidate backends,
          activating the top-scoring single-backend model without Pareto pruning.
        - "pareto_menu": fits full front for interactive post-fit inspection.
    enable_pareto_fusion : bool, default=True
        Whether to merge models into the Master Pareto Front.
    max_rules : int | None, default=None
        Upper bound on rules.
    max_atoms : int | None, default=None
        Upper bound on total atoms.
    backend_params : dict[str, dict] | None
        Per-backend parameters.
    cv : int, default=5
        Cross-validation folds.
    scoring : str, default="r2"
        Regression scoring metric.
    compact_tolerance : float, default=0.02
        Score tolerance used by the "compact" intent (and the "balanced"
        fallback), in units of the archive metric (R^2).
    feature_selection : {"auto", "none", "kbest_mi", "kbest_f", "variance", "tree"} | Any, default="auto"
        Upstream search-space reduction strategy. When set to "auto", automatically
        activates mutual-information regression feature selection if n_features >= 50 or
        n_features > n_samples.
    max_features : int | float | None, default=None
        Maximum features to retain when feature selection is active.
        If float between 0.0 and 1.0, treated as percentage of input features.
        If None, adaptively scales (min 10, max 50).
    preprocessing : dict | None
        Preprocessing configuration forwarded to backends.
    timeout_per_backend : float | None
        Warn if a single backend's CV loop exceeds this many seconds.
    random_state : int | None
        Random seed for splitting.
    """

    def __init__(
        self,
        candidate_backends: list[str] | None = None,
        preference: Literal["compact", "balanced", "performance", "expressive", "accuracy", "pareto_menu", "manual", "best_single_cv"] = "balanced",
        enable_pareto_fusion: bool = True,
        max_rules: int | None = None,
        max_atoms: int | None = None,
        backend_params: dict[str, dict[str, Any]] | None = None,
        cv: int = 5,
        scoring: str = "r2",
        feature_selection: Literal["auto", "none", "kbest_mi", "kbest_f", "variance", "tree"] | Any = "auto",
        max_features: int | float | None = None,
        preprocessing: dict[str, Any] | None = None,
        probing_strategy: Literal["none", "subsample", "hv_contribution"] = "none",
        probing_subsample: float = 0.25,
        probing_threshold_samples: int = 500,
        compact_tolerance: float = 0.02,
        timeout_per_backend: float | None = None,
        random_state: int | None = None,
    ):
        self.candidate_backends = candidate_backends
        self.preference = preference
        self.enable_pareto_fusion = enable_pareto_fusion
        self.max_rules = max_rules
        self.max_atoms = max_atoms
        self.backend_params = backend_params
        self.cv = cv
        self.scoring = scoring
        self.feature_selection = feature_selection
        self.max_features = max_features
        self.preprocessing = preprocessing
        self.probing_strategy = probing_strategy
        self.probing_subsample = probing_subsample
        self.probing_threshold_samples = probing_threshold_samples
        self.compact_tolerance = compact_tolerance
        self.timeout_per_backend = timeout_per_backend
        self.random_state = random_state

    def _resolve_feature_selection(self, n_samples: int, n_features: int) -> dict[str, Any] | None:
        """Derive upstream feature selection settings based on data dimensionality.

        For continuous regression in high-D regimes (D >= 50 or D > N), applying
        coarse variable-level selection via mutual information or F-regression
        prevents exponential growth in quantile thresholds and beam-search paths.
        """
        fs = self.feature_selection
        if fs in (None, False, "none"):
            return None

        if not isinstance(fs, str):
            return {"feature_selector": fs}

        fs_str = fs.lower()
        if fs_str == "auto":
            if n_features < 50 and n_features <= n_samples:
                return None
            method = "kbest"
            from sklearn.feature_selection import mutual_info_regression
            score_func = mutual_info_regression
        elif fs_str in ("kbest_mi", "kbest"):
            method = "kbest"
            from sklearn.feature_selection import mutual_info_regression
            score_func = mutual_info_regression
        elif fs_str == "kbest_f":
            method = "kbest"
            from sklearn.feature_selection import f_regression
            score_func = f_regression
        elif fs_str == "variance":
            from sklearn.feature_selection import VarianceThreshold
            return {"feature_selector": VarianceThreshold()}
        elif fs_str == "tree":
            method = "boruta"
            score_func = None
        else:
            method = fs_str
            score_func = None

        if self.max_features is None:
            k = min(n_features, max(10, min(50, max(20, n_samples // 4))))
        elif isinstance(self.max_features, float) and 0.0 < self.max_features <= 1.0:
            k = max(2, int(n_features * self.max_features))
        else:
            k = min(n_features, int(self.max_features))

        cfg: dict[str, Any] = {"feature_selection": method, "k": k}
        if score_func is not None:
            cfg["feature_selection_params"] = {"score_func": score_func}
        return cfg

    def fit(self, X, y):
        X_valid, y_valid = check_X_y(X, y, dtype=None, y_numeric=True)
        self.n_features_in_ = X_valid.shape[1]

        # Resolve automated upstream search-space reduction
        effective_preprocessing = dict(self.preprocessing or {})
        if "feature_selection" not in effective_preprocessing and "feature_selector" not in effective_preprocessing:
            fs_cfg = self._resolve_feature_selection(len(X_valid), self.n_features_in_)
            if fs_cfg is not None:
                effective_preprocessing.update(fs_cfg)
        self.effective_preprocessing_ = effective_preprocessing

        backends = list(self.candidate_backends or _DEFAULT_REGRESSOR_BACKENDS)
        per_backend_params = dict(self.backend_params or {})

        # Multi-fidelity probing (see classifier: "subsample" = mean score,
        # "hv_contribution" = leave-one-out hypervolume gain on the probe set)
        if (
            self.probing_strategy in ("subsample", "hv_contribution")
            and len(X_valid) >= self.probing_threshold_samples
            and len(backends) > 2
        ):
            probe_size = max(50, int(len(X_valid) * self.probing_subsample))
            from sklearn.model_selection import ShuffleSplit
            ss = ShuffleSplit(n_splits=1, train_size=probe_size, random_state=self.random_state)
            probe_idx, _ = next(ss.split(X_valid, y_valid))
            X_probe, y_probe = X_valid[probe_idx], y_valid[probe_idx]
            probe_splitter = KFold(n_splits=2, shuffle=True, random_state=self.random_state)

            def _probe_factory_reg(backend: str, bp: dict[str, Any] | None):
                return lambda: ScoredRuleSetRegressor(
                    backend=backend,
                    backend_params=bp,
                    preprocessing=self.effective_preprocessing_,
                    random_state=self.random_state,
                )

            probe_ranks_reg: dict[str, float] = {}
            if self.probing_strategy == "hv_contribution":
                factories = {}
                for backend in backends:
                    bp = per_backend_params.get(backend)
                    if bp is None and backend == "cart":
                        bp = {"max_depth": 3}
                    factories[backend] = _probe_factory_reg(backend, bp)
                try:
                    probe_ranks_reg = _probe_hv_contribution(
                        factories, X_probe, y_probe, probe_splitter,
                        archive_metric=_r2,
                        predict_fn=predict_regression_from_ruleset,
                    )
                except Exception:
                    probe_ranks_reg = {}
            if not probe_ranks_reg:  # "subsample" or HV probing failed -> mean-score fallback
                for backend in backends:
                    bp = per_backend_params.get(backend)
                    if bp is None and backend == "cart":
                        bp = {"max_depth": 3}
                    try:
                        sc = cross_val_score(
                            _probe_factory_reg(backend, bp)(), X_probe, y_probe,
                            cv=probe_splitter,
                            scoring=self.scoring,
                            error_score="raise",
                        )
                        probe_ranks_reg[backend] = float(np.mean(sc))
                    except Exception:
                        probe_ranks_reg[backend] = float("-inf")

            sorted_backends = sorted(backends, key=lambda b: probe_ranks_reg.get(b, float("-inf")), reverse=True)
            cutoff = max(2, len(backends) // 2)
            survivors = [b for b in sorted_backends[:cutoff] if probe_ranks_reg.get(b, float("-inf")) > float("-inf")]
            if survivors:
                backends = survivors

        cv_results: dict[str, float] = {}
        best_backend: str | None = None
        best_score = -np.inf

        cv_splitter = KFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )
        scorer = get_scorer(self.scoring)

        self.master_archive_ = MasterParetoArchive(
            higher_is_better=True,
            compact_tolerance=self.compact_tolerance,
            metric_name=str(self.scoring or "Validation Score"),
        )
        self.all_harvested_candidates_: list[ParetoCandidate] = []
        self.backend_timings_: dict[str, dict[str, float]] = {}
        fitted_estimators: dict[str, Any] = {}

        for backend in backends:
            bp = per_backend_params.get(backend)
            if bp is None and backend == "cart":
                bp = {"max_depth": 3}

            def make(params=bp, backend=backend):
                return ScoredRuleSetRegressor(
                    backend=backend,
                    backend_params=params,
                    preprocessing=self.effective_preprocessing_,
                    random_state=self.random_state,
                )

            t0 = time.monotonic()
            try:
                main_score, per_key = _oof_evaluate_config(
                    make, X_valid, y_valid, cv_splitter,
                    main_scorer=scorer,
                    archive_metric=_r2,
                    predict_fn=predict_regression_from_ruleset,
                )
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"AutoScoredRuleSetRegressor: backend '{backend}' failed during CV: {exc}",
                    UserWarning,
                )
                main_score = float("-inf")
                per_key = {}

            elapsed = time.monotonic() - t0
            self.backend_timings_.setdefault(backend, {})["cv_s"] = float(elapsed)
            if (
                self.timeout_per_backend is not None
                and elapsed > self.timeout_per_backend
            ):
                warnings.warn(
                    f"AutoScoredRuleSetRegressor: backend '{backend}' exceeded "
                    f"timeout ({elapsed:.1f}s > {self.timeout_per_backend:.1f}s).",
                    UserWarning,
                )

            cv_results[backend] = main_score
            if main_score > best_score:
                best_score = main_score
                best_backend = backend

            if self.enable_pareto_fusion and main_score > float("-inf"):
                try:
                    t_refit = time.monotonic()
                    fitted_reg = make()
                    fitted_reg.fit(X_valid, y_valid)
                    refit_elapsed = time.monotonic() - t_refit
                    fitted_estimators[backend] = fitted_reg
                    self.backend_timings_.setdefault(backend, {})["refit_s"] = float(refit_elapsed)
                    _admit_full_fit(
                        self.master_archive_, fitted_reg, backend, per_key,
                        n_splits=cv_splitter.get_n_splits(),
                        candidate_pool=self.all_harvested_candidates_,
                    )
                except Exception as exc:  # noqa: BLE001
                    warnings.warn(
                        f"AutoScoredRuleSetRegressor: failed to harvest model from '{backend}': {exc}",
                        UserWarning,
                    )

        self.fitted_estimators_ = fitted_estimators

        if best_backend is None:
            raise RuntimeError(
                "AutoScoredRuleSetRegressor: all candidate backends failed during "
                "cross-validation. Check warnings for details."
            )

        if best_backend in fitted_estimators:
            winner = fitted_estimators[best_backend]
        else:
            bp = per_backend_params.get(best_backend)
            winner = ScoredRuleSetRegressor(
                backend=best_backend,
                backend_params=bp,
                preprocessing=self.effective_preprocessing_,
                random_state=self.random_state,
            )
            winner.fit(X_valid, y_valid)

        self.best_backend_ = best_backend
        self.best_score_ = best_score
        self.cv_results_ = cv_results
        self.best_estimator_ = winner
        self.feature_names_in_ = winner.feature_names_in_
        self.selected_feature_indices_ = getattr(winner, "selected_feature_indices_", None)
        self.feature_selector_ = getattr(winner, "feature_selector_", None)

        if self.preference == "best_single_cv":
            self.ruleset_ = winner.ruleset_
            self.active_estimator_ = winner
            self.active_backend_ = best_backend
        elif self.enable_pareto_fusion and self.master_archive_.candidates_:
            chosen = self.master_archive_.select_candidate(
                preference=self.preference,
                max_rules=self.max_rules,
                max_atoms=self.max_atoms,
            )
            self.active_candidate_ = chosen
            self.ruleset_ = chosen.ruleset
            self.active_estimator_ = chosen.metadata.get("estimator", winner)
            self.active_backend_ = chosen.backend
        else:
            self.active_candidate_ = None
            self.ruleset_ = winner.ruleset_
            self.active_estimator_ = winner
            self.active_backend_ = best_backend

        return self

    def get_pareto_spectrum(self) -> list[dict[str, Any]]:
        check_is_fitted(self, ["master_archive_"])
        return self.master_archive_.get_spectrum()

    def set_active_model(
        self,
        complexity: int | None = None,
        *,
        preference: str | None = None,
        max_atoms: int | None = None,
        max_rules: int | None = None,
        max_effective_nodes: int | None = None,
        target_atoms: int | None = None,
        target_rules: int | None = None,
        target_effective_nodes: int | None = None,
        exact_size: bool = False,
    ):
        """Switch active regression rule set post-hoc by operational intent profile or model size."""
        check_is_fitted(self, ["master_archive_"])
        candidates = self.master_archive_.candidates_
        if not candidates:
            raise RuntimeError("No models available in Master Pareto Archive.")
        if complexity is not None and target_atoms is None:
            target_atoms = int(complexity)
        chosen = self.master_archive_.select_candidate(
            preference=preference or self.preference,
            max_atoms=max_atoms,
            max_rules=max_rules,
            max_effective_nodes=max_effective_nodes,
            target_atoms=target_atoms,
            target_rules=target_rules,
            target_effective_nodes=target_effective_nodes,
            exact_size=exact_size,
        )
        self.active_candidate_ = chosen
        self.ruleset_ = chosen.ruleset
        self.active_estimator_ = chosen.metadata.get(
            "estimator", getattr(self, "best_estimator_", None)
        )
        self.active_backend_ = chosen.backend
        return self

    def save_archive(self, path: str | Path) -> Path:
        """Save the fitted MasterParetoArchive to a ZIP bundle (or JSON if ``.json``)."""
        check_is_fitted(self, ["master_archive_"])
        return self.master_archive_.save(path)

    @classmethod
    def from_archive(
        cls,
        archive_or_path: str | Path | MasterParetoArchive,
        preference: str = "balanced",
        **select_kwargs: Any,
    ) -> "AutoScoredRuleSetRegressor":
        """Instantiate a ready-to-predict regressor post-hoc from a saved MasterParetoArchive."""
        archive = (
            archive_or_path
            if isinstance(archive_or_path, MasterParetoArchive)
            else MasterParetoArchive.load(archive_or_path)
        )
        if not archive.candidates_:
            raise ValueError("Cannot instantiate AutoScoredRuleSetRegressor from an empty archive.")
        reg = cls(preference=preference, compact_tolerance=archive.compact_tolerance)  # type: ignore[arg-type]
        reg.master_archive_ = archive
        reg.all_harvested_candidates_ = list(archive.candidate_pool_) or list(archive.candidates_)
        chosen = archive.select_candidate(preference=preference, **select_kwargs)
        reg.best_estimator_ = None
        reg.best_backend_ = chosen.backend
        reg.best_score_ = chosen.score
        reg.active_candidate_ = chosen
        reg.ruleset_ = chosen.ruleset
        reg.active_estimator_ = None
        reg.active_backend_ = chosen.backend
        reg.feature_names_in_ = np.asarray(chosen.ruleset.feature_names) if chosen.ruleset.feature_names else None
        reg.n_features_in_ = len(chosen.ruleset.feature_names) if chosen.ruleset.feature_names else None
        return reg

    def export_index_card(self) -> str:
        check_is_fitted(self, ["ruleset_"])
        return format_ruleset_markdown(self.ruleset_)

    def _prepare_X_for_prediction(self, X) -> np.ndarray:
        """Transform X into the input space of the estimator that produced the active rule set."""
        est = getattr(self, "active_estimator_", None) or getattr(self, "best_estimator_", None)
        return _transform_for_estimator(est, X)

    def predict(self, X):
        check_is_fitted(self, ["ruleset_"])
        X_prep = self._prepare_X_for_prediction(X)
        return predict_regression_from_ruleset(self.ruleset_, X_prep)

    def to_ruleset(self) -> ScoredRuleSet:
        check_is_fitted(self, ["ruleset_"])
        return self.ruleset_

    def plot_pareto_front(self, output_path: str | Path | None = None, title: str | None = None, **kwargs: Any):
        """Plot the non-dominated Master Pareto front."""
        check_is_fitted(self, ["master_archive_"])
        t = title or f"{self.__class__.__name__} Master Pareto Frontier"
        metric = "Validation Score" if self.scoring is None else str(self.scoring)
        return self.master_archive_.plot_pareto_front(output_path=output_path, title=t, metric_name=metric, **kwargs)
