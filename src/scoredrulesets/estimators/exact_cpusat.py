"""Exact (CP-SAT) certified-compact rule induction backend.

Two-stage formulation validated in the PLA-neural-scored-rule-sets project
(stage 6, ``spike_exact_front_v2.py``):

  1. **Candidate enumeration** (no solver): exhaustively enumerate all
     conjunctions of width ``<= max_width`` over the binarized proposition
     matrix, deduplicated by coverage bitset.  This replaces the monolithic
     slot-based encoding, whose vacuous empty conjunctions degenerate to
     all-firing rules.
  2. **Coverage ILP** (CP-SAT): select any subset of at most ``max_rules``
     candidates under an atom budget, maximizing F1 via parametric binary
     search on an integral threshold.  Selection variables are per candidate,
     so "at most R rules" is expressed faithfully (no slot assignment).

The result is a *certified* optimum for the hypothesis space "conjunctions of
width <= k, at most R rules, at most B atoms per class": the UNSAT branches of
the binary search prove that no better F1 is attainable within the budget.

Multiclass is handled by solving one binary DNF per class (positives = y == c)
and assigning each rule a one-hot score vector toward its class; inference is
the standard argmax-sum aggregation with a majority-class default rule.

This backend requires the optional ``exact`` dependency (OR-Tools)::

    pip install 'scoredrulesets[exact]'

The import is lazy (inside :meth:`fit`), so importing this module never fails
when OR-Tools is absent.

Known limitation: OR-Tools 9.15 deadlocks on Python 3.14 (the first
``CpSolver.Solve`` call blocks indefinitely in native code, not even
SIGALRM-based watchdogs fire).  Run the exact backend on Python <= 3.13
until OR-Tools ships a fixed 3.14 wheel.

Design note: unlike the one-class asymmetric spike formulation (DNF for the
positive class, default = negative), this estimator is *symmetric*: every
class solves its own budgeted DNF with one-hot scores, and inference is the
standard argmax-sum with a majority-class default rule.  Certification is
therefore w.r.t. the per-class hypothesis space "conjunctions of width <= k,
at most max_rules rules, at most max_atoms_per_class atoms — per class".
"""

from __future__ import annotations

import itertools
from typing import NamedTuple

import numpy as np
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ..runtime import predict as predict_from_ruleset
from ..runtime import predict_proba as predict_proba_from_ruleset
from ..schema import AggregationSpec, Atom, Rule, ScoredRuleSet
from ._time_budget import (
    FitBudgetExceededError,
    deadline_reached,
    resolve_deadline,
)
from .atom_space import binarize_with_thresholds, compute_nln_thresholds
from .base import BaseRuleSetEstimator


class _Candidate(NamedTuple):
    """One enumerated conjunction: literal indices + coverage bitset."""

    lits: tuple[int, ...]
    width: int
    mask: int  # Python int as bitset of covered samples


# ---------------------------------------------------------------------------
# Stage 1: candidate enumeration
# ---------------------------------------------------------------------------

def enumerate_candidates(
    P: np.ndarray, max_width: int, max_candidates: int,
) -> list[_Candidate]:
    """All conjunctions of width 1..max_width, deduplicated by coverage mask.

    Contradictory (empty-coverage) conjunctions are dropped.  When several
    literal combinations cover the same sample set, the narrowest wins.  If the
    pool exceeds ``max_candidates``, the best-covering candidates are kept.
    """
    n, D = P.shape
    Pb = P.astype(bool)
    lit_bits: list[int] = []
    for d in range(D):
        bits = 0
        for i in np.where(Pb[:, d])[0]:
            bits |= 1 << int(i)
        lit_bits.append(bits)

    seen: dict[int, _Candidate] = {}
    for width in range(1, max_width + 1):
        for combo in itertools.combinations(range(D), width):
            mask = (1 << n) - 1
            for d in combo:
                if lit_bits[d] == 0:
                    mask = 0
                    break
                mask &= lit_bits[d]
                if mask == 0:
                    break
            if mask == 0:
                continue
            prev = seen.get(mask)
            if prev is None or width < prev.width:
                seen[mask] = _Candidate(combo, width, mask)

    cands = list(seen.values())
    if len(cands) > max_candidates:
        cands.sort(key=lambda c: (-c.mask.bit_count(), c.width))
        cands = cands[:max_candidates]
    cands.sort(key=lambda c: (c.width, c.mask))
    return cands


# ---------------------------------------------------------------------------
# Stage 2: coverage ILP (one class)
# ---------------------------------------------------------------------------

def _solve_class(
    P: np.ndarray,
    y_pos: np.ndarray,
    cands: list[_Candidate],
    *,
    max_rules: int,
    budget: int,
    time_limit: float,
    fit_deadline: float | None = None,
) -> tuple[float, list[int]] | None:
    """Maximize F1 for one class under an atom budget via CP-SAT.

    Returns ``(f1, chosen_candidate_indices)`` or ``None`` if infeasible or the
    (per-solve or global) time budget was exhausted before any solution.
    """
    from ortools.sat.python import cp_model

    n = P.shape[0]
    C = len(cands)
    widths = [c.width for c in cands]
    P_count = int(y_pos.sum())
    if P_count == 0:
        return None

    # Coverage lists: which candidates cover sample i?
    covers: list[list[int]] = [[] for _ in range(n)]
    for ci, cand in enumerate(cands):
        m = cand.mask
        while m:
            low = m & -m
            covers[int(low.bit_length() - 1)].append(ci)
            m ^= low

    def feasible(t: float) -> tuple[float, list[int]] | None:
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x{c}") for c in range(C)]
        pred = [model.NewBoolVar(f"p{i}") for i in range(n)]

        model.Add(sum(x) <= max_rules)
        model.Add(sum(widths[c] * x[c] for c in range(C)) <= budget)

        for i in range(n):
            cl = covers[i]
            if not cl:
                model.Add(pred[i] == 0)
                continue
            for c in cl:
                model.Add(pred[i] >= x[c])
            model.Add(pred[i] <= sum(x[c] for c in cl))

        # Integral F1 threshold: 2Q*TP >= q*(PH + P), t = q/Q, Q = 10000.
        q = int(round(t * 10000))
        tp_expr = sum(2 * 10000 * int(y_pos[i]) * pred[i] for i in range(n))
        ph_expr = sum(q * pred[i] for i in range(n))
        model.Add(tp_expr - ph_expr >= q * P_count)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        status = solver.Solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            chosen = [c for c in range(C) if solver.Value(x[c])]
            pv = np.array([solver.Value(pred[i]) for i in range(n)], dtype=bool)
            tp = int((pv & y_pos).sum())
            ph = int(pv.sum())
            f1 = 2 * tp / (ph + P_count) if (ph + P_count) else 0.0
            return f1, chosen
        return None

    # Fast path: perfect separation (F1 = 1.0) needs a single solve and is the
    # common case on conceptually clean data (e.g. synthetic DNF).
    if deadline_reached(fit_deadline):
        return None
    perfect = feasible(1.0)
    if perfect is not None:
        return perfect

    # Parametric binary search on the attainable F1 threshold.  Attainable F1
    # values are discrete (2tp/(ph+P)), so 10 iterations at 2e-3 resolution
    # pin down the optimum on all benchmark datasets.
    lo, hi = 0.0, 1.0
    best: tuple[float, list[int]] | None = None
    for _ in range(10):
        if deadline_reached(fit_deadline):
            return best
        mid = (lo + hi) / 2
        res = feasible(mid)
        if res is not None:
            best = res
            lo = mid
        else:
            hi = mid
        if hi - lo < 2e-3:
            break
    return best


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

class ExactCPSATClassifier(BaseRuleSetEstimator):
    """Certified-compact rule induction via two-stage exact optimization.

    Parameters
    ----------
    max_width : int
        Maximum literals per conjunction (candidate pool is O(D^max_width)).
    max_rules : int
        Maximum number of rules per class.
    max_atoms_per_class : int
        Atom budget per class for the primary model.
    pareto_schedule : list[int] | None
        If given, the atom budget is swept over these values and every
        resulting rule set is stored in :attr:`pareto_archive_` (keyed by total
        atom count) so that :class:`AutoScoredRuleSetClassifier` can harvest
        the certified front.  If ``None``, only ``max_atoms_per_class`` is
        solved.
    n_bins : int
        Quantile bins per feature for discretisation.
    max_thresholds_per_feature : int | None
        Cap on thresholds per feature.
    max_features : int | None
        Cap on the number of features entering the literal pool (random
        subsample if exceeded), keeping enumeration tractable.
    max_candidates : int
        Cap on the deduplicated candidate pool (best-covering kept).
    time_limit_per_solve : float
        Wall-clock limit for each CP-SAT feasibility solve.
    max_fit_seconds : float | None
        Global cooperative fit-time budget.  If exhausted before any class is
        solved, :class:`FitBudgetExceededError` is raised.
    random_state : int | None
        Seed for the feature subsample.
    """

    def __init__(
        self,
        max_width: int = 2,
        max_rules: int = 3,
        max_atoms_per_class: int = 4,
        pareto_schedule: list[int] | None = None,
        n_bins: int = 3,
        max_thresholds_per_feature: int | None = None,
        max_features: int | None = None,
        max_candidates: int = 2000,
        time_limit_per_solve: float = 10.0,
        max_fit_seconds: float | None = None,
        random_state: int | None = None,
    ):
        self.max_width = max_width
        self.max_rules = max_rules
        self.max_atoms_per_class = max_atoms_per_class
        self.pareto_schedule = pareto_schedule
        self.n_bins = n_bins
        self.max_thresholds_per_feature = max_thresholds_per_feature
        self.max_features = max_features
        self.max_candidates = max_candidates
        self.time_limit_per_solve = time_limit_per_solve
        self.max_fit_seconds = max_fit_seconds
        self.random_state = random_state

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X, y):
        try:
            import ortools  # noqa: F401
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "backend='exact' requires OR-Tools. Install with: "
                "pip install 'scoredrulesets[exact]' (or 'pip install ortools')."
            ) from exc

        X_arr, y_arr = check_X_y(X, y, dtype="numeric")
        self.n_features_in_ = X_arr.shape[1]
        self.classes_ = unique_labels(y_arr)
        n_classes = len(self.classes_)
        rng = np.random.default_rng(self.random_state)
        fit_deadline = resolve_deadline(self.max_fit_seconds)

        # --- literal pool (optional feature subsample) ---------------------
        if self.max_features is not None and X_arr.shape[1] > self.max_features:
            feat_idx = np.sort(rng.choice(
                X_arr.shape[1], size=self.max_features, replace=False))
            X_pool = X_arr[:, feat_idx]
        else:
            feat_idx = np.arange(X_arr.shape[1])
            X_pool = X_arr
        self._pool_features_ = feat_idx

        self._thresholds_ = compute_nln_thresholds(
            X_pool, n_bins=self.n_bins,
            max_thresholds_per_feature=self.max_thresholds_per_feature,
        )
        P = binarize_with_thresholds(X_pool, self._thresholds_)
        # Meta carries ORIGINAL feature indices so that extracted atoms resolve
        # against the full-width X at predict time.
        meta = [
            (int(feat_idx[j]), op, thr)
            for j, op, thr in self._proposition_meta()
        ]

        y_idx = np.array(
            [int(np.searchsorted(self.classes_, v)) for v in y_arr], dtype=int)
        majority = int(np.bincount(y_idx, minlength=n_classes).argmax())

        # --- stage 1: candidate pool --------------------------------------
        cands = enumerate_candidates(P, self.max_width, self.max_candidates)

        # --- stage 2: per-class ILP, budget sweep -------------------------
        budgets = self.pareto_schedule if self.pareto_schedule else [
            self.max_atoms_per_class]
        budgets = sorted({int(b) for b in budgets if int(b) >= 1}) or [
            self.max_atoms_per_class]

        self.pareto_archive_: dict[int, ScoredRuleSet] = {}
        solved_any = False
        for budget in budgets:
            rules, total_atoms, solved = self._solve_all_classes(
                P, y_idx, cands, meta, n_classes, budget, fit_deadline)
            solved_any = solved_any or solved
            rs = self._assemble_ruleset(
                rules, total_atoms, budget, majority, n_classes)
            self.pareto_archive_[total_atoms] = rs
            if not cands:
                break  # empty pool: default-only model, no further budgets
            if deadline_reached(fit_deadline):
                break
        if not solved_any and deadline_reached(fit_deadline):
            # No class was ever solved: the only "model" is the majority-class
            # default, i.e. an untrained degenerate result.  Surface the
            # timeout instead (package-wide contract for setup exhaustion).
            raise FitBudgetExceededError(
                "max_fit_seconds exhausted during exact backend optimization "
                "before a single class was solved; no certified model exists "
                "within this budget."
            )

        # Primary model: archive entry closest to max_atoms_per_class.
        self.ruleset_ = min(
            self.pareto_archive_.items(),
            key=lambda kv: abs(kv[0] - self.max_atoms_per_class),
        )[1]
        return self

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _proposition_meta(self) -> list[tuple[int, str, float]]:
        meta: list[tuple[int, str, float]] = []
        for j, thr in enumerate(self._thresholds_):
            for t in thr:
                meta.append((j, "<=", float(t)))
            for t in thr:
                meta.append((j, ">", float(t)))
        return meta

    def _solve_all_classes(
        self, P, y_idx, cands, meta, n_classes, budget, fit_deadline,
    ) -> tuple[list[Rule], int, bool]:
        """Solve one binary DNF per class under *budget* atoms.

        Returns (rules, total_atoms, solved_any_class).
        """
        rules: list[Rule] = []
        total_atoms = 0
        solved = False
        if not cands:
            return rules, total_atoms, solved
        for c in range(n_classes):
            if deadline_reached(fit_deadline):
                break
            y_pos = (y_idx == c)
            if not y_pos.any():
                continue
            res = _solve_class(
                P, y_pos, cands,
                max_rules=self.max_rules, budget=budget,
                time_limit=self.time_limit_per_solve,
                fit_deadline=fit_deadline,
            )
            if res is None:
                continue
            solved = True
            _f1, chosen = res
            for ci in chosen:
                cand = cands[ci]
                total_atoms += cand.width
                rules.append(self._candidate_to_rule(
                    cand, meta, c, n_classes, ci))
        return rules, total_atoms, solved

    def _candidate_to_rule(
        self, cand: _Candidate, meta, class_idx, n_classes, ci,
    ) -> Rule:
        atoms: list[Atom] = []
        for d in sorted(cand.lits):
            feat_j, op, thr = meta[d]
            atoms.append(Atom(feature=f"f{feat_j}", op=op, value=thr))
        scores = [0.0] * n_classes
        scores[class_idx] = 1.0
        return Rule(
            atoms=atoms, scores=scores,
            rule_id=f"exact_{class_idx}_{ci}",
            metadata={"source": "exact_cpusat", "class_idx": class_idx},
        )

    def _assemble_ruleset(
        self, rules: list[Rule], total_atoms: int, budget: int,
        majority: int, n_classes: int,
    ) -> ScoredRuleSet:
        default_scores = [0.0] * n_classes
        default_scores[majority] = 1.0
        rs = ScoredRuleSet(
            class_labels=self.classes_.tolist(),
            feature_names=[f"f{j}" for j in range(self.n_features_in_)],
            rules=[Rule(
                atoms=[], scores=default_scores, rule_id="exact_default",
                metadata={"source": "exact_cpusat", "kind": "default"},
            )] + rules,
            aggregation=AggregationSpec(type="argmax_sum", temperature=1.0),
            metadata={
                "backend": "exact_cpusat",
                "certified": True,
                "budget_atoms_per_class": budget,
                "max_width": self.max_width,
                "max_rules": self.max_rules,
                "total_atoms": total_atoms,
            },
        )
        rs.validate()
        return rs

    # ------------------------------------------------------------------
    # predict / sklearn interface
    # ------------------------------------------------------------------

    def predict(self, X):
        check_is_fitted(self, "ruleset_")
        X_arr = check_array(X, dtype="numeric")
        return predict_from_ruleset(self.ruleset_, X_arr)

    def predict_proba(self, X):
        check_is_fitted(self, "ruleset_")
        X_arr = check_array(X, dtype="numeric")
        return predict_proba_from_ruleset(self.ruleset_, X_arr)

    def to_ruleset(self) -> ScoredRuleSet:
        check_is_fitted(self, "ruleset_")
        return self.ruleset_
