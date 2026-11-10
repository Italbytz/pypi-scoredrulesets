"""Cooperative time-budget helpers for evolutionary estimators."""

from __future__ import annotations

import time

try:
    import evolutionary_ml.common.time_budget as _emb
    FitBudgetExceededError = _emb.FitBudgetExceededError
    resolve_deadline = _emb.resolve_deadline
    deadline_reached = _emb.deadline_reached
    remaining_seconds = _emb.remaining_seconds
except ImportError:
    class FitBudgetExceededError(TimeoutError):
        """Raised when ``max_fit_seconds`` is exhausted during setup."""

    def resolve_deadline(max_fit_seconds: float | None) -> float | None:
        if max_fit_seconds is None:
            return None
        budget = float(max_fit_seconds)
        now = time.monotonic()
        if budget <= 0.0:
            return now
        return now + budget

    def deadline_reached(deadline: float | None) -> bool:
        return deadline is not None and time.monotonic() >= deadline

    def remaining_seconds(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        return max(0.0, deadline - time.monotonic())

__all__ = [
    "resolve_deadline",
    "deadline_reached",
    "remaining_seconds",
    "FitBudgetExceededError",
]
