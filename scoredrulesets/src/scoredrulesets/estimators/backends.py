from __future__ import annotations

import importlib
import inspect
import os
import subprocess as sp
from pathlib import Path
from typing import Any

from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------------------------
# Java / JVM helpers  (needed by rulekit backend — optional extra [rulekit])
# These functions are only called when backend="rulekit" is selected.
# Install with: pip install 'scoredrulesets[rulekit]'
# ---------------------------------------------------------------------------

def _ensure_java_home() -> None:
    """Make sure JAVA_HOME points to a JDK whose libjli.dylib JPype can load.

    Problem: sdkman (and some Homebrew setups) may set JAVA_HOME to a JDK
    built for the wrong CPU architecture (e.g. x86_64 on an arm64 Mac).
    This function:
      1. Checks whether a **loadable** JVM library exists at the current JAVA_HOME.
      2. If not, tries ``/usr/libexec/java_home`` (macOS) to get a working path.
      3. Updates ``os.environ["JAVA_HOME"]`` accordingly.
      4. Tells JPype where the JVM library is (monkey-patches getDefaultJVMPath).
    """
    current = os.environ.get("JAVA_HOME", "")

    # Step 1: If JAVA_HOME is set, check that a loadable JVM lib exists
    if current and _find_jvm_dll(current):
        _configure_jpype(current)
        return

    # Step 2: Try /usr/libexec/java_home (macOS) for a compatible JDK
    resolved = _java_home_from_system()
    if resolved and _find_jvm_dll(resolved):
        os.environ["JAVA_HOME"] = resolved
        _configure_jpype(resolved)
        return

    # Step 3: JAVA_HOME was not set at all → use system result anyway
    if not current and resolved:
        os.environ["JAVA_HOME"] = resolved
        _configure_jpype(resolved)
        return

    # Nothing helped – leave env as-is; the caller will get the JVM error.


def _java_home_from_system() -> str | None:
    """Return JAVA_HOME as reported by the OS (macOS ``/usr/libexec/java_home``)."""
    try:
        proc = sp.run(
            ["/usr/libexec/java_home"],
            capture_output=True, text=True, timeout=5,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except (FileNotFoundError, sp.TimeoutExpired):
        pass
    return None


def _find_jvm_dll(java_home: str) -> str | None:
    """Return the path to a **loadable** JVM shared library under *java_home*.

    On macOS this also verifies that the library's CPU architecture matches
    the running Python process (arm64 vs x86_64).  Returns None if no
    compatible library is found.
    """
    import platform as _platform
    import struct

    base = Path(java_home)
    if not base.is_dir():
        return None

    # Determine Python's pointer size → architecture expectation
    # 8 bytes → 64-bit; struct.calcsize("P") is reliable across platforms.
    _py_arch = _platform.machine().lower()  # e.g. "arm64", "x86_64", "aarch64"

    # Well-known relative paths  (checked in order)
    _patterns = (
        "lib/libjli.dylib",
        "lib/server/libjvm.dylib",
        "lib/server/libjvm.so",
        "lib/amd64/server/libjvm.so",
        "jre/lib/server/libjvm.so",
        "jre/lib/amd64/server/libjvm.so",
    )

    for pattern in _patterns:
        candidate = base / pattern
        if candidate.exists() and _dll_arch_ok(candidate, _py_arch):
            return str(candidate)

    # Fallback: recursive glob
    for dll in base.rglob("libjli.dylib"):
        if _dll_arch_ok(dll, _py_arch):
            return str(dll)
    for dll in base.rglob("libjvm.*"):
        if dll.suffix in (".dylib", ".so") and _dll_arch_ok(dll, _py_arch):
            return str(dll)

    return None


def _dll_arch_ok(dll_path: Path, python_arch: str) -> bool:
    """Check whether *dll_path*'s CPU architecture is compatible with *python_arch*.

    On macOS we use ``file`` to inspect Mach-O headers.  On other platforms
    we optimistically return True (worst case: JPype will fail with a clear
    error message).
    """
    import sys as _sys
    if _sys.platform != "darwin":
        return True  # Skip check on Linux/Windows

    try:
        result = sp.run(
            ["file", str(dll_path)],
            capture_output=True, text=True, timeout=5,
        )
        output = result.stdout.lower()
        # Map Python's platform.machine() to what `file` reports
        if python_arch in ("arm64", "aarch64"):
            return "arm64" in output
        if python_arch in ("x86_64", "amd64"):
            return "x86_64" in output
        # Unknown arch – accept optimistically
        return True
    except Exception:
        return True  # If `file` fails, don't block


def _configure_jpype(java_home: str) -> None:
    """Tell JPype where to find the JVM (if jpype is installed).

    JPype uses JAVA_HOME first, but its ``find_libjvm`` may fail on some
    setups (sdkman, Homebrew symlinks).  We patch ``getDefaultJVMPath``
    directly if we already know the correct path.
    """
    try:
        import jpype          # type: ignore[import-untyped]
        if jpype.isJVMStarted():
            return  # Already running, nothing we can do.
        jvm_path = _find_jvm_dll(java_home)
        if jvm_path:
            # Monkey-patch getDefaultJVMPath so rulekit (and anything else
            # that calls it) receives the verified path.
            _original = jpype.getDefaultJVMPath

            def _patched() -> str:
                return jvm_path

            jpype.getDefaultJVMPath = _patched  # type: ignore[assignment]
    except ImportError:
        pass



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
        if random_state is not None and _supports_kwarg(rulekit_cls, "random_state"):
            params.setdefault("random_state", random_state)
        try:
            return rulekit_cls(**params)
        except FileNotFoundError as e:
            jvm_hint = _rulekit_jvm_hint()
            raise ImportError(
                "backend='rulekit' could not start the JVM. "
                "Install with: pip install 'scoredrulesets[rulekit]' and check "
                "JAVA_HOME and the JPype JVM path. "
                f"{jvm_hint} Error: {e}"
            ) from e

    if backend_key == "exstracs":
        exstracs_cls = _resolve_exstracs_class()
        if random_state is not None and _supports_kwarg(exstracs_cls, "random_state"):
            params.setdefault("random_state", random_state)
        return exstracs_cls(**params)

    if backend_key == "logicgp":
        logicgp_cls = _resolve_logicgp_class()
        if random_state is not None and _supports_kwarg(logicgp_cls, "random_state"):
            params.setdefault("random_state", random_state)
        return logicgp_cls(**params)

    if backend_key == "rulelcs":
        rulelcs_cls = _resolve_rulelcs_class()
        if random_state is not None and _supports_kwarg(rulelcs_cls, "random_state"):
            params.setdefault("random_state", random_state)
        return rulelcs_cls(**params)


    if backend_key == "rulekit_native":
        rulekit_native_cls = _resolve_rulekit_native_class()
        if random_state is not None and _supports_kwarg(rulekit_native_cls, "random_state"):
            params.setdefault("random_state", random_state)
        return rulekit_native_cls(**params)


    if backend_key == "rulenln":
        rulenln_cls = _resolve_rulenln_class()
        if random_state is not None and _supports_kwarg(rulenln_cls, "random_state"):
            params.setdefault("random_state", random_state)
        return rulenln_cls(**params)

    if backend_key == "rulegp":
        from .rulegp import RuleGPClassifier
        if random_state is not None:
            params.setdefault("random_state", random_state)
        return RuleGPClassifier(**params)

    if backend_key == "rulegp2":
        from .rulegp2 import RuleGP2Classifier
        if random_state is not None:
            params.setdefault("random_state", random_state)
        return RuleGP2Classifier(**params)

    if backend_key == "rulelcs2":
        from .rulelcs2 import RuleLCS2Classifier
        if random_state is not None:
            params.setdefault("random_state", random_state)
        return RuleLCS2Classifier(**params)

    raise ValueError(
        f"Unknown backend '{backend}'. Supported backends: "
        f"'cart', 'hs', 'rulekit', 'rulekit_native', 'exstracs', 'logicgp', "
        f"'rulelcs', 'rulelcs2', 'rulenln', 'rulegp', 'rulegp2'."
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
        "backend='hs' requires an imodels HS class, but none could be found. "
        "Install/update with: pip install -e '.[hs]'. "
        "Checked: "
        + ", ".join(tried)
    )


def _resolve_rulekit_class():
    """
    Try to load the RuleKit class.
    RuleKit requires Java; provide actionable error messages.
    """
    _ensure_java_home()

    try:
        # Check whether Java is installed.
        result = sp.run(
            ["java", "-version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            raise RuntimeError("Java not found or not runnable")
    except (FileNotFoundError, sp.TimeoutExpired, RuntimeError) as e:
        raise ImportError(
            "backend='rulekit' requires Java, but no working Java was found. "
            "Install with: pip install 'scoredrulesets[rulekit]' and ensure "
            "Java (JDK 11+) is installed. "
            f"Error: {e}"
        ) from e

    candidate_locations = [
        ("rulekit.classification", "RuleClassifier"),
        ("rulekit.classifier", "RuleKit"),
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
        "backend='rulekit' requires the 'rulekit' package. "
        "Install with: pip install 'scoredrulesets[rulekit]'. "
        "Note: RuleKit requires Java (JDK 11+). "
        "Checked: "
        + ", ".join(tried)
    )


def _resolve_exstracs_class():
    """
    Try to load the ExSTraCS class.
    ExSTraCS is provided by skExSTraCS.
    """
    candidate_locations = [
        ("skExSTraCS", "ExSTraCS"),
        ("skexstracs", "ExSTraCSClassifier"),
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
        "backend='exstracs' requires scikit-exstracs (import names: skExSTraCS or skexstracs). "
        "Install with: pip install 'scoredrulesets[exstracs]'. "
        "Checked: "
        + ", ".join(tried)
    )


def _supports_kwarg(cls: type[Any], param_name: str) -> bool:
    try:
        signature = inspect.signature(cls)
    except (TypeError, ValueError):
        return False
    return param_name in signature.parameters


def _resolve_logicgp_class():
    """Load the LogicGPClassifier class from this project."""
    try:
        from .logicgp import LogicGPClassifier
        return LogicGPClassifier
    except ImportError as e:
        raise ImportError(
            "backend='logicgp' could not load LogicGPClassifier. "
            f"Import error: {e}"
        ) from e


def _resolve_rulelcs_class():
    try:
        from .rulelcs import RuleLCSClassifier

        return RuleLCSClassifier
    except ImportError as e:
        raise ImportError(
            "backend='rulelcs' could not load RuleLCSClassifier. "
            f"Import error: {e}"
        ) from e


def _rulekit_jvm_hint() -> str:
    java_home = os.environ.get("JAVA_HOME", "<unset>")
    hint = f"JAVA_HOME={java_home}."
    try:
        import jpype

        hint += f" jpype.getDefaultJVMPath()={jpype.getDefaultJVMPath()}."
    except Exception:
        hint += " jpype.getDefaultJVMPath()=<unavailable>."
    return hint


def _resolve_rulenln_class():
    """Load the RuleNLNClassifier from this package."""
    try:
        from .rulenln import RuleNLNClassifier
        return RuleNLNClassifier
    except ImportError as e:
        raise ImportError(
            "backend='rulenln' could not load RuleNLNClassifier. "
            f"Import error: {e}"
        ) from e


def _resolve_rulekit_native_class():
    """Load the pure-Python RuleKitNativeClassifier from this package."""
    try:
        from .rulekit_native import RuleKitNativeClassifier
        return RuleKitNativeClassifier
    except ImportError as e:
        raise ImportError(
            "backend='rulekit_native' could not load RuleKitNativeClassifier. "
            f"Import error: {e}"
        ) from e


