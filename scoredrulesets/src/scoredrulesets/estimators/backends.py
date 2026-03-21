from __future__ import annotations

import importlib
from typing import Any

from sklearn.tree import DecisionTreeClassifier


def build_backend_estimator(
    backend: str,
    backend_params: dict[str, Any] | None,
    random_state: int | None,
):
    params = dict(backend_params or {})
    backend_key = backend.lower()

    if backend_key == "cart":
        params.setdefault("random_state", random_state)
        return DecisionTreeClassifier(**params)

    if backend_key == "hs":
        hs_cls = _resolve_hs_class()
        params.setdefault("random_state", random_state)
        return hs_cls(**params)

    raise ValueError(
        f"Unknown backend '{backend}'. Supported backends: 'hs', 'cart'."
    )


def _resolve_hs_class():
    candidate_locations = [
        ("imodels", "HSTreeClassifier"),
        ("imodels", "HSTreeClassifierCV"),
        ("imodels.tree.hierarchical_shrinkage", "HSTreeClassifier"),
        ("imodels.tree.hierarchical_shrinkage", "HSTreeClassifierCV"),
    ]

    tried: list[str] = []
    for module_name, class_name in candidate_locations:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            tried.append(f"{module_name}.{class_name}")
            continue

        cls = getattr(module, class_name, None)
        if cls is not None:
            return cls
        tried.append(f"{module_name}.{class_name}")

    raise ImportError(
        "backend='hs' braucht eine imodels-HS-Klasse, konnte aber keine finden. "
        "Installiere/aktualisiere mit: pip install -e '.[hs]'. "
        "Geprueft wurden: "
        + ", ".join(tried)
    )

