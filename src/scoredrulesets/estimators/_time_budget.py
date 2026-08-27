"""Cooperative time-budget helpers for evolutionary estimators.

These utilities give every generation/epoch/iteration-based estimator a
uniform and reliable way to honour a ``max_fit_seconds`` fit-time budget:

* the deadline is an absolute :func:`time.monotonic` timestamp computed once
  at the start of ``fit`` – it therefore covers data setup *and* the search
  loop, so the total fit stays close to the requested budget;
* :func:`deadline_reached` is cheap and is meant to be called frequently,
  including inside inner loops, so the actual stop time never overshoots the
  budget by more than a single individual/batch evaluation;
* when the deadline is hit the caller is expected to break out of its loops
  and return the best model found so far.

Using a single shared implementation keeps the timeout behaviour consistent
across all estimators instead of each one re-implementing it slightly
differently.
"""

from __future__ import annotations

import time

__all__ = ["resolve_deadline", "deadline_reached", "remaining_seconds"]


def resolve_deadline(max_fit_seconds: float | None) -> float | None:
    """Return an absolute monotonic deadline for ``max_fit_seconds``.

    Parameters
    ----------
    max_fit_seconds : float or None
        Fit-time budget in seconds. ``None`` disables the limit. A
        non-positive budget maps to "already expired" so the caller stops
        before doing any real search work.

    Returns
    -------
    float or None
        The absolute :func:`time.monotonic` timestamp at which the budget is
        exhausted, or ``None`` when no budget is configured.
    """
    if max_fit_seconds is None:
        return None
    budget = float(max_fit_seconds)
    now = time.monotonic()
    if budget <= 0.0:
        return now
    return now + budget


def deadline_reached(deadline: float | None) -> bool:
    """Return ``True`` if ``deadline`` is set and the current time is past it."""
    return deadline is not None and time.monotonic() >= deadline


def remaining_seconds(deadline: float | None) -> float | None:
    """Return the seconds left until ``deadline`` (never negative), or ``None``."""
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())
