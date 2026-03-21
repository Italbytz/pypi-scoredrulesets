from __future__ import annotations

import importlib
import subprocess
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

    if backend_key == "rulekit":
        rulekit_cls = _resolve_rulekit_class()
        params.setdefault("random_state", random_state)
        return rulekit_cls(**params)

    if backend_key == "exstracs":
        exstracs_cls = _resolve_exstracs_class()
        params.setdefault("random_state", random_state)
        return exstracs_cls(**params)

    raise ValueError(
        f"Unknown backend '{backend}'. Supported backends: 'cart', 'hs', 'rulekit', 'exstracs'."
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


def _resolve_rulekit_class():
    """
    Versuche RuleKit-Klasse zu laden.
    RuleKit benötigt Java - gebe aussagekräftige Fehlermeldung aus.
    """
    try:
        import subprocess
        # Überprüfe ob Java installiert ist
        result = subprocess.run(
            ["java", "-version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            raise RuntimeError("Java nicht gefunden oder nicht lauffähig")
    except (FileNotFoundError, subprocess.TimeoutExpired, RuntimeError) as e:
        raise ImportError(
            "backend='rulekit' benötigt Java, aber kein Java gefunden oder funktionsfähig. "
            "Bitte installiere Java (JDK 11+). "
            f"Fehler: {e}"
        ) from e

    # Versuche rulekit zu importieren
    try:
        from rulekit.classifier import RuleKit
        return RuleKit
    except ImportError as e:
        raise ImportError(
            "backend='rulekit' braucht das 'rulekit' Paket. "
            "Installiere mit: pip install rulekit. "
            "Beachte: RuleKit benötigt Java (JDK 11+). "
            f"Import-Fehler: {e}"
        ) from e


def _resolve_exstracs_class():
    """
    Versuche ExSTraCS-Klasse zu laden.
    ExSTraCS wird durch skExSTraCS bereitgestellt.
    """
    try:
        from skexstracs import ExSTraCSClassifier
        return ExSTraCSClassifier
    except ImportError as e:
        raise ImportError(
            "backend='exstracs' braucht das 'skexstracs' Paket. "
            "Installiere mit: pip install scikit-exstracs. "
            f"Import-Fehler: {e}"
        ) from e

