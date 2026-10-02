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

from dataclasses import dataclass, field
import time
from typing import Any, Literal
import warnings

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import f1_score, r2_score
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
_DEFAULT_REGRESSOR_BACKENDS = ["greedy_cascaded", "cart"]


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

    def __init__(self, higher_is_better: bool = True):
        self.higher_is_better = higher_is_better
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

        if preference == "compact":
            # Highest score among solutions with atoms <= 6 (or minimum available)
            compact_set = [c for c in eligible if c.atoms <= 6]
            pool = compact_set if compact_set else eligible
            chosen = max(pool, key=lambda c: c.score if self.higher_is_better else -c.score)
            return chosen.ruleset

        if preference == "accuracy":
            chosen = max(eligible, key=lambda c: c.score if self.higher_is_better else -c.score)
            return chosen.ruleset

        if preference in ("manual", "custom"):
            chosen = max(eligible, key=lambda c: c.score if self.higher_is_better else -c.score)
            return chosen.ruleset

        # Default: "balanced" (knee / elbow point)
        if len(eligible) <= 2:
            chosen = max(eligible, key=lambda c: c.score if self.higher_is_better else -c.score)
            return chosen.ruleset

        # Multi-objective knee point via perpendicular distance to chord
        atoms_arr = np.array([c.atoms for c in eligible], dtype=float)
        scores_arr = np.array([c.score for c in eligible], dtype=float)

        c_min, c_max = atoms_arr[0], atoms_arr[-1]
        s_min, s_max = scores_arr[0], scores_arr[-1]

        denom = np.hypot(c_max - c_min, s_max - s_min)
        if denom < 1e-9:
            chosen = max(eligible, key=lambda c: c.score if self.higher_is_better else -c.score)
            return chosen.ruleset

        # Distance from (c_i, s_i) to line (c_min, s_min)-(c_max, s_max)
        distances = np.abs((s_max - s_min) * atoms_arr - (c_max - c_min) * scores_arr + c_max * s_min - s_max * c_min) / denom
        best_idx = int(np.argmax(distances))
        return eligible[best_idx].ruleset

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
        "hs", "ruleplcs"]``).
    preference : {"compact", "balanced", "accuracy", "pareto_menu", "manual"}, default="balanced"
        Operational intent profile:
        - "compact": minimizes rule complexity (target atoms <= 6).
        - "balanced": knee point of Pareto front (optimal marginal utility).
        - "accuracy": pushes performance to maximum on validation score.
        - "pareto_menu": fits full front for interactive post-fit inspection.
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
        preprocessing: dict[str, Any] | None = None,
        timeout_per_backend: float | None = None,
        probing_strategy: Literal["none", "subsample"] = "none",
        probing_subsample: float = 0.25,
        probing_threshold_samples: int = 500,
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
        self.preprocessing = preprocessing
        self.timeout_per_backend = timeout_per_backend
        self.probing_strategy = probing_strategy
        self.probing_subsample = probing_subsample
        self.probing_threshold_samples = probing_threshold_samples
        self.random_state = random_state

    def fit(self, X, y):
        X_valid, y_valid = check_X_y(X, y, dtype=None)
        self.n_features_in_ = X_valid.shape[1]

        backends = list(self.candidate_backends or _DEFAULT_CLASSIFIER_BACKENDS)
        per_backend_params = dict(self.backend_params or {})

        # Multi-fidelity probing to filter unpromising backends on larger datasets
        if (
            self.probing_strategy == "subsample"
            and len(X_valid) >= self.probing_threshold_samples
            and len(backends) > 2
        ):
            probe_size = max(50, int(len(X_valid) * self.probing_subsample))
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, train_size=probe_size, random_state=self.random_state)
            probe_idx, _ = next(sss.split(X_valid, y_valid))
            X_probe, y_probe = X_valid[probe_idx], y_valid[probe_idx]
            probe_splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=self.random_state)

            probe_scores: dict[str, float] = {}
            for backend in backends:
                bp = per_backend_params.get(backend)
                if bp is None and backend == "cart":
                    bp = {"max_depth": 3}
                clf = ScoredRuleSetClassifier(
                    backend=backend,
                    backend_params=bp,
                    preprocessing=self.preprocessing,
                    random_state=self.random_state,
                )
                try:
                    sc = cross_val_score(
                        clf, X_probe, y_probe,
                        cv=probe_splitter,
                        scoring=self.scoring,
                        error_score="raise",
                    )
                    probe_scores[backend] = float(np.mean(sc))
                except Exception:
                    probe_scores[backend] = float("-inf")

            sorted_backends = sorted(backends, key=lambda b: probe_scores.get(b, float("-inf")), reverse=True)
            cutoff = max(2, len(backends) // 2)
            survivors = [b for b in sorted_backends[:cutoff] if probe_scores.get(b, float("-inf")) > float("-inf")]
            if survivors:
                backends = survivors

        cv_results: dict[str, float] = {}
        best_backend: str | None = None
        best_score = -np.inf

        cv_splitter = StratifiedKFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )

        self.master_archive_ = MasterParetoArchive(higher_is_better=True)
        fitted_estimators: dict[str, Any] = {}

        for backend in backends:
            bp = per_backend_params.get(backend)
            if bp is None and backend == "cart":
                bp = {"max_depth": 3}
            clf = ScoredRuleSetClassifier(
                backend=backend,
                backend_params=bp,
                preprocessing=self.preprocessing,
                random_state=self.random_state,
            )
            t0 = time.monotonic()
            try:
                scores = cross_val_score(
                    clf, X_valid, y_valid,
                    cv=cv_splitter,
                    scoring=self.scoring,
                    error_score="raise",
                )
                elapsed = time.monotonic() - t0
                if (
                    self.timeout_per_backend is not None
                    and elapsed > self.timeout_per_backend
                ):
                    warnings.warn(
                        f"AutoScoredRuleSet: backend '{backend}' exceeded "
                        f"timeout ({elapsed:.1f}s > {self.timeout_per_backend:.1f}s).",
                        UserWarning,
                    )
                mean_score = float(scores.mean())
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"AutoScoredRuleSet: backend '{backend}' failed during CV: {exc}",
                    UserWarning,
                )
                mean_score = float("-inf")

            cv_results[backend] = mean_score
            if mean_score > best_score:
                best_score = mean_score
                best_backend = backend

            # If pareto fusion is active, fit on full data and harvest models
            if self.enable_pareto_fusion and mean_score > float("-inf"):
                try:
                    fitted_clf = ScoredRuleSetClassifier(
                        backend=backend,
                        backend_params=bp,
                        preprocessing=self.preprocessing,
                        random_state=self.random_state,
                    )
                    fitted_clf.fit(X_valid, y_valid)
                    fitted_estimators[backend] = fitted_clf

                    # Check for multi-model Pareto front on underlying estimator
                    underlying = getattr(fitted_clf, "estimator_", fitted_clf)
                    pareto_dict = getattr(underlying, "pareto_archive_", None)
                    if isinstance(pareto_dict, dict) and pareto_dict:
                        for comp, rs in pareto_dict.items():
                            preds_cand = predict_from_ruleset(rs, X_valid)
                            cand_s = float(f1_score(y_valid, preds_cand, average="macro", zero_division=0))
                            self.master_archive_.add(rs, cand_s, backend=backend)
                    else:
                        rs = fitted_clf.to_ruleset()
                        preds_cand = predict_from_ruleset(rs, X_valid)
                        cand_s = float(f1_score(y_valid, preds_cand, average="macro", zero_division=0))
                        self.master_archive_.add(rs, cand_s, backend=backend)
                except Exception as exc:  # noqa: BLE001
                    warnings.warn(
                        f"AutoScoredRuleSet: failed to harvest Pareto models from '{backend}': {exc}",
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
                preprocessing=self.preprocessing,
                random_state=self.random_state,
            )
            winner.fit(X_valid, y_valid)

        self.best_backend_ = best_backend
        self.best_score_ = best_score
        self.cv_results_ = cv_results
        self.best_estimator_ = winner
        self.classes_ = winner.classes_
        self.feature_names_in_ = winner.feature_names_in_

        # Select model according to preference and Pareto fusion
        if self.enable_pareto_fusion and self.master_archive_.candidates_:
            self.ruleset_ = self.master_archive_.select(
                preference=self.preference,
                max_rules=self.max_rules,
                max_atoms=self.max_atoms,
            )
        else:
            self.ruleset_ = winner.ruleset_

        return self

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
        self.ruleset_ = closest.ruleset
        return self

    def export_index_card(self) -> str:
        """Export active model formatted as a human-simulatable index card."""
        check_is_fitted(self, ["ruleset_"])
        return format_ruleset_markdown(self.ruleset_)

    def predict(self, X):
        check_is_fitted(self, ["ruleset_"])
        return predict_from_ruleset(self.ruleset_, X)

    def predict_proba(self, X):
        check_is_fitted(self, ["ruleset_"])
        return predict_proba_from_ruleset(self.ruleset_, X)

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
        List of backend names to evaluate (default: ``["greedy_cascaded", "cart"]``).
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
        probing_strategy: Literal["none", "subsample"] = "none",
        probing_subsample: float = 0.25,
        probing_threshold_samples: int = 500,
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
        self.probing_strategy = probing_strategy
        self.probing_subsample = probing_subsample
        self.probing_threshold_samples = probing_threshold_samples
        self.random_state = random_state

    def fit(self, X, y):
        X_valid, y_valid = check_X_y(X, y, dtype=None, y_numeric=True)
        self.n_features_in_ = X_valid.shape[1]

        backends = list(self.candidate_backends or _DEFAULT_REGRESSOR_BACKENDS)
        per_backend_params = dict(self.backend_params or {})

        # Multi-fidelity probing to filter unpromising backends on larger datasets
        if (
            self.probing_strategy == "subsample"
            and len(X_valid) >= self.probing_threshold_samples
            and len(backends) > 2
        ):
            probe_size = max(50, int(len(X_valid) * self.probing_subsample))
            from sklearn.model_selection import ShuffleSplit
            ss = ShuffleSplit(n_splits=1, train_size=probe_size, random_state=self.random_state)
            probe_idx, _ = next(ss.split(X_valid, y_valid))
            X_probe, y_probe = X_valid[probe_idx], y_valid[probe_idx]
            probe_splitter = KFold(n_splits=2, shuffle=True, random_state=self.random_state)

            probe_scores_reg: dict[str, float] = {}
            for backend in backends:
                bp = per_backend_params.get(backend)
                if bp is None and backend == "cart":
                    bp = {"max_depth": 3}
                reg = ScoredRuleSetRegressor(
                    backend=backend,
                    backend_params=bp,
                    random_state=self.random_state,
                )
                try:
                    sc = cross_val_score(
                        reg, X_probe, y_probe,
                        cv=probe_splitter,
                        scoring=self.scoring,
                        error_score="raise",
                    )
                    probe_scores_reg[backend] = float(np.mean(sc))
                except Exception:
                    probe_scores_reg[backend] = float("-inf")

            sorted_backends = sorted(backends, key=lambda b: probe_scores_reg.get(b, float("-inf")), reverse=True)
            cutoff = max(2, len(backends) // 2)
            survivors = [b for b in sorted_backends[:cutoff] if probe_scores_reg.get(b, float("-inf")) > float("-inf")]
            if survivors:
                backends = survivors

        cv_results: dict[str, float] = {}
        best_backend: str | None = None
        best_score = -np.inf

        cv_splitter = KFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )

        self.master_archive_ = MasterParetoArchive(higher_is_better=True)
        fitted_estimators: dict[str, Any] = {}

        for backend in backends:
            bp = per_backend_params.get(backend)
            if bp is None and backend == "cart":
                bp = {"max_depth": 3}
            reg = ScoredRuleSetRegressor(
                backend=backend,
                backend_params=bp,
                random_state=self.random_state,
            )
            try:
                scores = cross_val_score(
                    reg, X_valid, y_valid,
                    cv=cv_splitter,
                    scoring=self.scoring,
                    error_score="raise",
                )
                mean_score = float(scores.mean())
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"AutoScoredRuleSetRegressor: backend '{backend}' failed during CV: {exc}",
                    UserWarning,
                )
                mean_score = float("-inf")

            cv_results[backend] = mean_score
            if mean_score > best_score:
                best_score = mean_score
                best_backend = backend

            if self.enable_pareto_fusion and mean_score > float("-inf"):
                try:
                    fitted_reg = ScoredRuleSetRegressor(
                        backend=backend,
                        backend_params=bp,
                        random_state=self.random_state,
                    )
                    fitted_reg.fit(X_valid, y_valid)
                    fitted_estimators[backend] = fitted_reg
                    rs = fitted_reg.to_ruleset()
                    preds_cand = predict_regression_from_ruleset(rs, X_valid)
                    cand_s = float(r2_score(y_valid, preds_cand))
                    self.master_archive_.add(rs, cand_s, backend=backend)
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
                random_state=self.random_state,
            )
            winner.fit(X_valid, y_valid)

        self.best_backend_ = best_backend
        self.best_score_ = best_score
        self.cv_results_ = cv_results
        self.best_estimator_ = winner
        self.feature_names_in_ = winner.feature_names_in_

        if self.enable_pareto_fusion and self.master_archive_.candidates_:
            self.ruleset_ = self.master_archive_.select(
                preference=self.preference,
                max_rules=self.max_rules,
                max_atoms=self.max_atoms,
            )
        else:
            self.ruleset_ = winner.ruleset_

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
        return self

    def export_index_card(self) -> str:
        check_is_fitted(self, ["ruleset_"])
        return format_ruleset_markdown(self.ruleset_)

    def predict(self, X):
        check_is_fitted(self, ["ruleset_"])
        return predict_regression_from_ruleset(self.ruleset_, X)

    def to_ruleset(self) -> ScoredRuleSet:
        check_is_fitted(self, ["ruleset_"])
        return self.ruleset_

    def plot_pareto_front(self, output_path: str | None = None, title: str | None = None):
        """Plot the non-dominated Master Pareto front."""
        check_is_fitted(self, ["master_archive_"])
        t = title or f"{self.__class__.__name__} Master Pareto Frontier"
        metric = "Validation Score" if self.scoring is None else str(self.scoring)
        return self.master_archive_.plot_pareto_front(output_path=output_path, title=t, metric_name=metric)
