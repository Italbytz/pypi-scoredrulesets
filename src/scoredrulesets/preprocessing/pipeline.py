from __future__ import annotations

from typing import Any

from sklearn.base import clone
from sklearn.pipeline import Pipeline


def _is_transformer(obj: Any) -> bool:
    return hasattr(obj, "fit") and hasattr(obj, "transform")


def build_preprocessing_step(step: Any, random_state: int | None = None) -> tuple[str, Any]:
    """Build one sklearn-compatible preprocessing step.

    Supported step forms:
    - transformer object with fit/transform
    - dict with {'name': ..., 'params': {...}, 'id': ...}
    - dict with {'transformer': ..., 'id': ...}
    """
    if _is_transformer(step):
        return (step.__class__.__name__.lower(), clone(step))

    if not isinstance(step, dict):
        raise ValueError("Each pipeline step must be a transformer or a dict configuration.")

    step_id = step.get("id")
    if "transformer" in step:
        transformer = step["transformer"]
        if not _is_transformer(transformer):
            raise ValueError("'transformer' must implement fit() and transform().")
        name = step_id or transformer.__class__.__name__.lower()
        return name, clone(transformer)

    name = str(step.get("name", "")).strip().lower()
    if not name:
        raise ValueError("Pipeline step dict must provide either 'transformer' or 'name'.")

    params = dict(step.get("params") or {})
    name_out = step_id or name

    if name == "impute":
        from sklearn.impute import SimpleImputer

        strategy = params.pop("strategy", "median")
        return name_out, SimpleImputer(strategy=strategy, **params)

    if name == "standard_scale":
        from sklearn.preprocessing import StandardScaler

        return name_out, StandardScaler(**params)

    if name == "minmax_scale":
        from sklearn.preprocessing import MinMaxScaler

        return name_out, MinMaxScaler(**params)

    if name == "robust_scale":
        from sklearn.preprocessing import RobustScaler

        return name_out, RobustScaler(**params)

    if name == "quantile":
        from sklearn.preprocessing import QuantileTransformer

        params.setdefault("random_state", random_state)
        return name_out, QuantileTransformer(**params)

    if name == "power":
        from sklearn.preprocessing import PowerTransformer

        return name_out, PowerTransformer(**params)

    raise ValueError(
        f"Unknown preprocessing pipeline step '{name}'. Supported names: "
        "'impute', 'standard_scale', 'minmax_scale', 'robust_scale', 'quantile', 'power'."
    )


def build_preprocessing_pipeline(
    steps: list[Any],
    random_state: int | None = None,
) -> Pipeline:
    """Build sklearn Pipeline from configurable preprocessing steps."""
    if not isinstance(steps, list) or len(steps) == 0:
        raise ValueError("pipeline_steps must be a non-empty list.")

    built_steps: list[tuple[str, Any]] = []
    seen_names: set[str] = set()

    for idx, raw_step in enumerate(steps):
        name, transformer = build_preprocessing_step(raw_step, random_state=random_state)
        if name in seen_names:
            name = f"{name}_{idx}"
        seen_names.add(name)
        built_steps.append((name, transformer))

    return Pipeline(steps=built_steps)
