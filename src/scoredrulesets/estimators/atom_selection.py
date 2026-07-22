from __future__ import annotations

from importlib import metadata as importlib_metadata
from typing import Any, Callable, Hashable, Iterable

import numpy as np


AtomSelectionStrategy = Callable[
    [Iterable[tuple[Hashable, np.ndarray]], np.ndarray, int, int, int],
    set[Hashable],
]


_ENTRY_POINT_GROUP = "scoredrulesets.atom_selection"
_ATOM_SELECTION_REGISTRY: dict[str, AtomSelectionStrategy] = {}
_ENTRY_POINTS_LOADED = False


def iter_native_candidate_signatures(specs: list[dict[str, Any]]) -> Iterable[tuple[int, str, object]]:
    """Yield generic native atom candidate signatures from feature specs."""
    for spec in specs:
        fi = int(spec["idx"])

        if spec["kind"] in ("num", "both"):
            for thr in spec.get("thresholds", []):
                yield (fi, "<=", float(thr))
                yield (fi, ">", float(thr))
            for lo, hi in spec.get("intervals", []):
                yield (fi, "between", [float(lo), float(hi)])

        if spec["kind"] in ("cat", "both"):
            cats = list(spec.get("categories", []))
            for cat in cats:
                yield (fi, "==", cat)
            if len(cats) >= 3:
                yield (fi, "in", sorted(cats[:2]))
                yield (fi, "in", sorted(cats[-2:]))
                if len(cats) >= 4:
                    mid = len(cats) // 2
                    yield (fi, "in", sorted(cats[mid - 1:mid + 1]))


def signature_key(signature: tuple[int, str, object]) -> tuple[int, str, str]:
    fi, op, value = signature
    return int(fi), str(op), str(value)


def select_top_c2_signatures(
    candidates: Iterable[tuple[Hashable, np.ndarray]],
    y_idx: np.ndarray,
    n_classes: int,
    min_samples_leaf: int,
    top_k: int,
) -> set[Hashable]:
    """Select top candidate signatures by C2 quality across classes.

    Each candidate contributes one score per class using
    C2 = (p - n) / (P + N) with full dataset class counts P/N.
    """
    if int(top_k) <= 0:
        raise ValueError("top_k must be > 0.")

    class_counts = np.bincount(y_idx, minlength=n_classes).astype(float)
    n_total = float(len(y_idx))

    scored: list[tuple[float, int, Hashable]] = []
    for sig, mask in candidates:
        support = int(mask.sum())
        if support < int(min_samples_leaf):
            continue
        covered = y_idx[mask]
        if covered.size == 0:
            continue
        covered_counts = np.bincount(covered, minlength=n_classes).astype(float)

        for c in range(n_classes):
            p = float(covered_counts[c])
            n = float(support - p)
            P = float(class_counts[c])
            N = float(n_total - P)
            denom = P + N
            if denom <= 0:
                continue
            c2 = (p - n) / denom
            scored.append((float(c2), support, sig))

    if not scored:
        return set()

    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    k = int(min(top_k, len(scored)))
    return {sig for _, _, sig in scored[:k]}


def register_atom_selection_strategy(
    name: str,
    selector: AtomSelectionStrategy,
    *,
    overwrite: bool = False,
) -> None:
    """Register an atom-selection strategy by name."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Strategy name must be a non-empty string.")
    if not callable(selector):
        raise TypeError("selector must be callable.")

    key = name.strip()
    if key in _ATOM_SELECTION_REGISTRY and not overwrite:
        raise ValueError(f"Atom-selection strategy '{key}' is already registered.")
    _ATOM_SELECTION_REGISTRY[key] = selector


def _register_builtin_strategies() -> None:
    # Intentionally empty: public core should not ship embargoed strategies.
    return


def _iter_entry_points(group: str):
    eps = importlib_metadata.entry_points()
    if hasattr(eps, "select"):
        return eps.select(group=group)
    return eps.get(group, [])


def _load_entry_point_strategies() -> None:
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True

    for ep in _iter_entry_points(_ENTRY_POINT_GROUP):
        try:
            loaded = ep.load()
        except Exception:
            continue

        strategy_name = ep.name
        selector: AtomSelectionStrategy | None = None

        if callable(loaded):
            selector = loaded
        elif hasattr(loaded, "select_signatures") and callable(loaded.select_signatures):
            selector = loaded.select_signatures
            if hasattr(loaded, "name") and isinstance(loaded.name, str) and loaded.name.strip():
                strategy_name = loaded.name.strip()

        if selector is None:
            continue

        try:
            register_atom_selection_strategy(strategy_name, selector, overwrite=False)
        except ValueError:
            continue


def available_atom_selection_strategies() -> tuple[str, ...]:
    """Return all registered atom-selection strategies, including plugins."""
    _register_builtin_strategies()
    _load_entry_point_strategies()
    return tuple(sorted(_ATOM_SELECTION_REGISTRY.keys()))


def is_atom_selection_strategy_available(name: str) -> bool:
    """Return whether a strategy is available in the current environment."""
    if not isinstance(name, str) or not name.strip():
        return False
    _register_builtin_strategies()
    _load_entry_point_strategies()
    return name.strip() in _ATOM_SELECTION_REGISTRY


def select_signatures_by_strategy(
    strategy: str,
    candidates: Iterable[tuple[Hashable, np.ndarray]],
    y_idx: np.ndarray,
    n_classes: int,
    min_samples_leaf: int,
    top_k: int,
) -> set[Hashable]:
    """Dispatch candidate selection to a registered strategy."""
    if not isinstance(strategy, str) or not strategy.strip():
        raise ValueError("strategy must be a non-empty string.")
    if int(top_k) <= 0:
        raise ValueError("top_k must be > 0.")

    _register_builtin_strategies()
    _load_entry_point_strategies()

    key = strategy.strip()
    selector = _ATOM_SELECTION_REGISTRY.get(key)
    if selector is None:
        available = ", ".join(sorted(_ATOM_SELECTION_REGISTRY.keys()))
        raise ValueError(
            f"Unknown atom-selection strategy '{key}'. Available: [{available}]"
        )

    return selector(
        candidates,
        y_idx,
        int(n_classes),
        int(min_samples_leaf),
        int(top_k),
    )
