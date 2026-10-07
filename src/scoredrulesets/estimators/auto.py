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
import time
from typing import Any, Callable, Literal
import warnings

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
_DEFAULT_REGRESSOR_BACKENDS = ["greedy_cascaded", "greedy_pareto", "cart", "ruleplcs"]

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
        if archive.add(
            rs,
            float(np.mean(scores)),
            backend=label,
            metadata={
                "estimator": est_full,
                "variant": key,
                "oof_folds": len(scores),
                "oof_std": float(np.std(scores)),
            },
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


class MasterParetoArchive:
    """Maintains a non-dominated Pareto front over complexity (atoms) and performance."""

    def __init__(self, higher_is_better: bool = True, compact_tolerance: float = 0.02):
        self.higher_is_better = higher_is_better
        self.compact_tolerance = float(compact_tolerance)
        self.candidates_: list[ParetoCandidate] = []

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
                "rules": c.rules,
                "score": c.score,
                "backend": c.backend,
                "ruleset": c.ruleset,
            }
            for c in self.candidates_
        ]

    def select(
        self,
        preference: str = "balanced",
        max_rules: int | None = None,
        max_atoms: int | None = None,
    ) -> ScoredRuleSet:
        """Select a single model from the archive according to user intent."""
        return self.select_candidate(preference, max_rules=max_rules, max_atoms=max_atoms).ruleset

    def _signed(self, score: float) -> float:
        return score if self.higher_is_better else -score

    def _select_compact(self, eligible: list[ParetoCandidate]) -> ParetoCandidate:
        """Most parsimonious candidate within ``compact_tolerance`` of the best score."""
        best = max(self._signed(c.score) for c in eligible)
        within = [c for c in eligible if self._signed(c.score) >= best - self.compact_tolerance]
        return min(within, key=lambda c: (c.atoms, -self._signed(c.score)))

    def select_candidate(
        self,
        preference: str = "balanced",
        max_rules: int | None = None,
        max_atoms: int | None = None,
    ) -> ParetoCandidate:
        """Select a single archive candidate according to user intent.

        - ``compact``: fewest atoms among candidates whose score lies within
          ``compact_tolerance`` of the best score.
        - ``accuracy`` / ``manual``: highest score.
        - ``balanced``: knee point, i.e. the candidate with maximal signed
          distance above the chord between the two extreme front points in
          min-max normalized (atoms, score) space.  Falls back to ``compact``
          for fronts with fewer than three points or without a convex knee.
        """
        if not self.candidates_:
            raise RuntimeError("MasterParetoArchive is empty; no models available.")

        eligible = list(self.candidates_)
        if max_rules is not None:
            eligible = [c for c in eligible if c.rules <= max_rules]
        if max_atoms is not None:
            eligible = [c for c in eligible if c.atoms <= max_atoms]

        if not eligible:
            warnings.warn(
                f"No models satisfy constraints (max_rules={max_rules}, max_atoms={max_atoms}); "
                f"falling back to unconstrained archive.",
                UserWarning,
            )
            eligible = list(self.candidates_)

        eligible = sorted(eligible, key=lambda c: (c.atoms, -self._signed(c.score)))

        if preference == "compact":
            return self._select_compact(eligible)

        if preference in ("accuracy", "manual", "custom"):
            return max(eligible, key=lambda c: (self._signed(c.score), -c.atoms))

        # Default: "balanced" (knee point in normalized objective space)
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

    def plot_pareto_front(
        self,
        output_path: str | None = None,
        title: str = "Master Pareto Frontier",
        metric_name: str = "Validation Score",
    ):
        """Plot the non-dominated Pareto front with intent highlights.

        Parameters
        ----------
        output_path : str | None
            If provided, saves the figure to this file (PDF, PNG, SVG).
        title : str
            Title of the plot.
        metric_name : str
            Y-axis label.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes objects
        """
        import matplotlib.pyplot as plt

        if not self.candidates_:
            raise RuntimeError("Cannot plot empty MasterParetoArchive.")

        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

        # Distinct backend colors
        unique_backends = sorted(list({c.backend for c in self.candidates_}))
        cmap = plt.get_cmap("tab10")
        color_map = {b: cmap(i % 10) for i, b in enumerate(unique_backends)}

        # Plot all candidates by backend
        for b in unique_backends:
            b_cands = [c for c in self.candidates_ if c.backend == b]
            ax.scatter(
                [c.atoms for c in b_cands],
                [c.score for c in b_cands],
                label=f"Backend: {b}",
                color=color_map[b],
                s=70,
                alpha=0.85,
                edgecolors="none",
                zorder=3,
            )

        # Plot Pareto step curve
        sorted_cands = sorted(self.candidates_, key=lambda c: c.atoms)
        x_steps = [c.atoms for c in sorted_cands]
        y_steps = [c.score for c in sorted_cands]
        ax.step(
            x_steps,
            y_steps,
            where="post",
            color="gray",
            linestyle="--",
            alpha=0.6,
            zorder=2,
            label="Pareto Envelope",
        )

        # Highlight Intent profiles
        try:
            compact_rs = self.select("compact")
            c_cand = next(c for c in self.candidates_ if c.ruleset is compact_rs)
            ax.scatter(
                [c_cand.atoms],
                [c_cand.score],
                s=160,
                facecolors="none",
                edgecolors="blue",
                linewidths=2,
                label="Intent: compact",
                zorder=4,
            )
        except Exception:
            pass

        try:
            balanced_rs = self.select("balanced")
            b_cand = next(c for c in self.candidates_ if c.ruleset is balanced_rs)
            ax.scatter(
                [b_cand.atoms],
                [b_cand.score],
                s=200,
                facecolors="none",
                edgecolors="crimson",
                linewidths=2.5,
                marker="s",
                label="Intent: balanced (knee)",
                zorder=4,
            )
        except Exception:
            pass

        try:
            acc_rs = self.select("accuracy")
            a_cand = next(c for c in self.candidates_ if c.ruleset is acc_rs)
            ax.scatter(
                [a_cand.atoms],
                [a_cand.score],
                s=160,
                facecolors="none",
                edgecolors="forestgreen",
                linewidths=2,
                marker="^",
                label="Intent: accuracy",
                zorder=4,
            )
        except Exception:
            pass

        ax.set_xlabel("Complexity (Total Atoms in Rule Set)", fontsize=10)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(frameon=True, fontsize=8, loc="best")
        plt.tight_layout()

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
    preference : {"compact", "balanced", "accuracy", "pareto_menu", "manual"}, default="balanced"
        Operational intent profile (all evaluated on out-of-fold archive scores):
        - "compact": fewest atoms within ``compact_tolerance`` of the best score.
        - "balanced": knee point of the normalized Pareto front (falls back to
          "compact" when the front has fewer than three points).
        - "accuracy": highest out-of-fold score.
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
        preference: Literal["compact", "balanced", "accuracy", "pareto_menu", "manual"] = "balanced",
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
            higher_is_better=True, compact_tolerance=self.compact_tolerance
        )
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
                    est_full = make()
                    est_full.fit(X_valid, y_valid)
                    if is_main:
                        fitted_estimators[backend] = est_full
                    if self.enable_pareto_fusion:
                        _admit_full_fit(
                            self.master_archive_, est_full, label, per_key,
                            n_splits=cv_splitter.get_n_splits(),
                        )
                except Exception as exc:  # noqa: BLE001
                    warnings.warn(
                        f"AutoScoredRuleSet: failed to harvest Pareto models from '{label}': {exc}",
                        UserWarning,
                    )

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
        if self.enable_pareto_fusion and self.master_archive_.candidates_:
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
        self.ruleset_ = candidate.ruleset
        self.active_estimator_ = candidate.metadata.get("estimator", self.best_estimator_)
        self.active_backend_ = candidate.backend

    def get_pareto_spectrum(self) -> list[dict[str, Any]]:
        """Return the Master Pareto spectrum of non-dominated rule sets."""
        check_is_fitted(self, ["master_archive_"])
        return self.master_archive_.get_spectrum()

    def set_active_model(self, complexity: int):
        """Switch active rule set to a specific complexity point on the Pareto front."""
        check_is_fitted(self, ["master_archive_"])
        candidates = self.master_archive_.candidates_
        if not candidates:
            raise RuntimeError("No models available in Master Pareto Archive.")

        # Find closest match by complexity
        closest = min(candidates, key=lambda c: abs(c.atoms - complexity))
        self._activate(closest)
        return self

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

    def plot_pareto_front(self, output_path: str | None = None, title: str | None = None):
        """Plot the non-dominated Master Pareto front."""
        check_is_fitted(self, ["master_archive_"])
        t = title or f"{self.__class__.__name__} Master Pareto Frontier"
        metric = "Validation Score" if self.scoring is None else str(self.scoring)
        return self.master_archive_.plot_pareto_front(output_path=output_path, title=t, metric_name=metric)


class AutoScoredRuleSetRegressor(RegressorMixin, BaseRuleSetEstimator):
    """AutoML meta-estimator for Scored Rule Sets regression.

    Evaluates candidate regression backends and manages the Master Pareto Archive.

    Parameters
    ----------
    candidate_backends : list[str] | None
        List of backend names to evaluate (default: ``["greedy_cascaded",
        "cart", "ruleplcs"]``).
    preference : {"compact", "balanced", "accuracy", "pareto_menu", "manual"}, default="balanced"
        Operational intent profile.
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
        preference: Literal["compact", "balanced", "accuracy", "pareto_menu", "manual"] = "balanced",
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
            higher_is_better=True, compact_tolerance=self.compact_tolerance
        )
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
                    fitted_reg = make()
                    fitted_reg.fit(X_valid, y_valid)
                    fitted_estimators[backend] = fitted_reg
                    _admit_full_fit(
                        self.master_archive_, fitted_reg, backend, per_key,
                        n_splits=cv_splitter.get_n_splits(),
                    )
                except Exception as exc:  # noqa: BLE001
                    warnings.warn(
                        f"AutoScoredRuleSetRegressor: failed to harvest model from '{backend}': {exc}",
                        UserWarning,
                    )

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

        if self.enable_pareto_fusion and self.master_archive_.candidates_:
            chosen = self.master_archive_.select_candidate(
                preference=self.preference,
                max_rules=self.max_rules,
                max_atoms=self.max_atoms,
            )
            self.ruleset_ = chosen.ruleset
            self.active_estimator_ = chosen.metadata.get("estimator", winner)
            self.active_backend_ = chosen.backend
        else:
            self.ruleset_ = winner.ruleset_
            self.active_estimator_ = winner
            self.active_backend_ = best_backend

        return self

    def get_pareto_spectrum(self) -> list[dict[str, Any]]:
        check_is_fitted(self, ["master_archive_"])
        return self.master_archive_.get_spectrum()

    def set_active_model(self, complexity: int):
        check_is_fitted(self, ["master_archive_"])
        candidates = self.master_archive_.candidates_
        if not candidates:
            raise RuntimeError("No models available in Master Pareto Archive.")
        closest = min(candidates, key=lambda c: abs(c.atoms - complexity))
        self.ruleset_ = closest.ruleset
        self.active_estimator_ = closest.metadata.get(
            "estimator", self.best_estimator_)
        self.active_backend_ = closest.backend
        return self

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

    def plot_pareto_front(self, output_path: str | None = None, title: str | None = None):
        """Plot the non-dominated Master Pareto front."""
        check_is_fitted(self, ["master_archive_"])
        t = title or f"{self.__class__.__name__} Master Pareto Frontier"
        metric = "Validation Score" if self.scoring is None else str(self.scoring)
        return self.master_archive_.plot_pareto_front(output_path=output_path, title=t, metric_name=metric)
