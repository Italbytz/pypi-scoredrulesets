#!/usr/bin/env python3
"""First-step SNP scaling experiment for scoredrulesets.

Goal:
- quantify quality/runtime degradation when increasing SNP count
- compare against prior feature selection baselines (kbest, rfe)

This first step intentionally excludes 1000G and uses synthetic epistasis
for controlled SNP scaling.
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import ssl
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel


REPO_ROOT = Path(__file__).resolve().parents[4]
PKG_SRC = REPO_ROOT / "artifacts/packages/python/pypi-scoredrulesets/src"
if str(PKG_SRC) not in sys.path:
    sys.path.insert(0, str(PKG_SRC))

CACHE_ROOT = REPO_ROOT / ".cache" / "benchmark"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("SCIKIT_LEARN_DATA", str(CACHE_ROOT / "scikit_learn_data"))
(CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(CACHE_ROOT / "scikit_learn_data").mkdir(parents=True, exist_ok=True)

from scoredrulesets.benchmarking.datasets import generate_epistasis_dataset  # noqa: E402
from scoredrulesets.estimators.sklearn_wrapper import ScoredRuleSetClassifier  # noqa: E402
from scoredrulesets.estimators._time_budget import FitBudgetExceededError  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    name: str
    backend: str
    backend_params: dict[str, Any]


@dataclass(frozen=True)
class SelectionSpec:
    name: str
    k: int


MODEL_LIBRARY: dict[str, ModelSpec] = {
    "wrapper_cart": ModelSpec(
        name="wrapper_cart",
        backend="cart",
        backend_params={"max_depth": 4},
    ),
    "wrapper_rulegp": ModelSpec(
        name="wrapper_rulegp",
        backend="rulegp",
        backend_params={
            "f1_averaging": "macro",
            "max_generations": 120,
            "stagnation_generations": 30,
            "population_size": 40,
            "n_adaptations_per_gen": 10,
        },
    ),
    "wrapper_ruleplcs": ModelSpec(
        name="wrapper_ruleplcs",
        backend="ruleplcs",
        backend_params={
            "population_size": 100,
            "n_iterations": 20,
            "n_repetitions": 1,
            "tournament_size": 4,
            "max_rules": 10,
        },
    ),
    "wrapper_rulensga2": ModelSpec(
        name="wrapper_rulensga2",
        backend="rulensga2",
        backend_params={
            "population_size": 40,
            "generations": 30,
            "max_rules": 8,
            "max_atoms_per_rule": 4,
            "enable_compaction": True,
        },
    ),
    "wrapper_rulekit": ModelSpec(
        name="wrapper_rulekit",
        backend="rulekit",
        backend_params={},
    ),
}

DEFAULT_MODELS = (
    MODEL_LIBRARY["wrapper_cart"],
    MODEL_LIBRARY["wrapper_ruleplcs"],
    MODEL_LIBRARY["wrapper_rulensga2"],
)

SELECTED_PAPER_MODELS = (
    MODEL_LIBRARY["wrapper_rulegp"],
    MODEL_LIBRARY["wrapper_rulensga2"],
    MODEL_LIBRARY["wrapper_ruleplcs"],
    MODEL_LIBRARY["wrapper_rulekit"],
)

DEFAULT_SELECTIONS = (
    SelectionSpec(name="none", k=0),
    SelectionSpec(name="kbest20_mi", k=20),
    SelectionSpec(name="l1logreg20", k=20),
)

DEFAULT_ARMS = ("none", "kbest20_mi", "l1logreg20")


_DOROTHEA_CACHE: dict[str, Any] = {}


class FitTimeoutError(TimeoutError):
    """Raised when an individual model fit exceeds the configured wall-clock budget."""


def _fit_timeout_handler(_signum, _frame):
    raise FitTimeoutError("fit timed out")


def _cooperative_fit_budget(fit_budget: float) -> float:
    """Cooperative ``max_fit_seconds`` kept slightly below the hard cap.

    The CLI ``--fit-time-budget-seconds`` value is the hard wall-clock cap
    enforced by a SIGALRM safety net. We hand the estimator a slightly smaller
    cooperative budget so it stops its search on its own and can still finish
    final model selection + ruleset construction before the alarm fires. This
    guarantees a reported model whenever the search loop is reached inside the
    budget, instead of a hard interrupt that yields no model.
    """
    margin = min(max(0.1 * fit_budget, 2.0), 30.0)
    return max(0.5, fit_budget - margin)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SNP scaling experiment")
    parser.add_argument(
        "--dataset-source",
        choices=["synth_epistasis", "sklearn_classification", "dorothea_uci"],
        default="synth_epistasis",
        help="Simulation backend for generating high-dimensional benchmark data.",
    )
    parser.add_argument(
        "--snp-counts",
        nargs="+",
        type=int,
        default=[20, 50, 100, 200],
        help="SNP counts to evaluate",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Random seeds for repeats",
    )
    parser.add_argument("--n-samples", type=int, default=1600)
    parser.add_argument("--n-interacting", type=int, default=2)
    parser.add_argument("--heritability", type=float, default=0.4)
    parser.add_argument(
        "--sklearn-informative",
        type=int,
        default=6,
        help="Number of informative features for sklearn_classification source.",
    )
    parser.add_argument(
        "--sklearn-flip-y",
        type=float,
        default=0.05,
        help="Label noise for sklearn_classification source (0..1).",
    )
    parser.add_argument(
        "--sklearn-class-sep",
        type=float,
        default=1.0,
        help="Class separation for sklearn_classification source.",
    )
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument(
        "--results-dir",
        default=str(REPO_ROOT / "agent-workbench/projects/feature-vs-atom-selection-bioinformatics/results"),
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=None,
        help="Subset of selection arm names to run (defaults to all selections).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Model names to run. Known: "
            + ", ".join(sorted(MODEL_LIBRARY.keys()))
            + ". If omitted, defaults to historical model set."
        ),
    )
    parser.add_argument(
        "--use-selected-paper-models",
        action="store_true",
        help="Shortcut for selected internal estimators plus RuleKit baseline.",
    )
    parser.add_argument(
        "--fit-time-budget-seconds",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Optional per-fit wall-clock budget(s) in seconds. "
            "Provide one or more values to evaluate F1 differences across fixed budgets. "
            "Exceeding fits are marked as timeout."
        ),
    )
    parser.add_argument(
        "--insecure-ssl-downloads",
        action="store_true",
        help=(
            "Disable TLS certificate verification for dataset downloads (OpenML/UCI). "
            "Use only as a local workaround for certificate issues."
        ),
    )
    return parser.parse_args()


def _parse_fit_budget_values(value: Any | None) -> list[float | None]:
    if value is None:
        return [None]
    if isinstance(value, (list, tuple)):
        values = [float(v) for v in value]
    else:
        values = [float(value)]
    if not values:
        return [None]
    if any(v <= 0 for v in values if v is not None):
        raise SystemExit("--fit-time-budget-seconds values must be > 0 when provided")
    return values


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.mean(values))


def _safe_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    return float(statistics.stdev(values))


def _checkpoint_outputs(
    results_dir: Path,
    raw_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    selections: list[SelectionSpec],
    models: list[ModelSpec],
) -> tuple[Path, Path, Path]:
    raw_csv = results_dir / "snp_scaling_feature_selection_raw.csv"
    _write_csv(
        raw_csv,
        raw_rows,
        [
            "dataset",
            "n_snps",
            "seed",
            "model",
            "backend",
            "selection",
            "fit_time_budget_seconds",
            "status",
            "error",
            "fit_seconds",
            "predict_seconds",
            "f1_macro",
            "n_rules",
            "n_atoms",
        ],
    )

    summary_rows = _aggregate_summary(raw_rows)
    summary_csv = results_dir / "snp_scaling_feature_selection_summary.csv"
    _write_csv(
        summary_csv,
        summary_rows,
        [
            "fit_time_budget_seconds",
            "n_snps",
            "model",
            "selection",
            "n_runs",
            "ok_runs",
            "f1_macro_mean",
            "f1_macro_std",
            "fit_seconds_mean",
            "fit_seconds_std",
            "predict_seconds_mean",
            "predict_seconds_std",
            "n_rules_mean",
            "n_atoms_mean",
            "f1_delta_vs_none",
            "fit_seconds_delta_vs_none",
            "f1_delta_vs_20",
            "f1_drop_pct_vs_20",
            "fit_time_ratio_vs_20",
        ],
    )

    report_md = results_dir / "snp_scaling_feature_selection_report.md"
    _write_markdown_report(report_md, summary_rows, args, selections, models)
    return raw_csv, summary_csv, report_md


def _generate_dataset(
    args: argparse.Namespace,
    snp_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    if args.dataset_source == "dorothea_uci":
        X_full, y_full = _load_dorothea_uci()
        n_total_features = int(X_full.shape[1])
        requested = int(snp_count)
        if requested != n_total_features:
            raise SystemExit(
                "dataset-source dorothea_uci uses original feature space only. "
                f"Requested n_snps={requested}, but DOROTHEA has {n_total_features} features. "
                "Use --snp-counts 100000."
            )

        # Keep original feature space; seed controls split only.
        return X_full, y_full, "dorothea_uci"

    if args.dataset_source == "synth_epistasis":
        X, y = generate_epistasis_dataset(
            n_samples=args.n_samples,
            n_snps=int(snp_count),
            n_interacting=args.n_interacting,
            heritability=args.heritability,
            random_state=int(seed),
        )
        return X, y, "synth_epistasis"

    informative = int(min(max(2, args.sklearn_informative), int(snp_count)))
    X, y = make_classification(
        n_samples=int(args.n_samples),
        n_features=int(snp_count),
        n_informative=informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        flip_y=float(args.sklearn_flip_y),
        class_sep=float(args.sklearn_class_sep),
        shuffle=True,
        random_state=int(seed),
    )
    # Keep memory usage manageable for large SNP scales.
    return X.astype(np.float32, copy=False), y.astype(np.int64, copy=False), "sklearn_classification"


def _load_sparse_weka_arff(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a Weka sparse ARFF file into dense feature and label arrays.

    Sparse rows look like:
        {190 1,366 1,613 1,...,1}
    where each 'index value' pair is a non-zero feature and the final scalar is the class label.
    """
    feature_count = 0
    rows: list[np.ndarray] = []
    labels: list[int] = []
    in_data = False

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            if line.lower() == "@data":
                in_data = True
                continue
            if not in_data:
                if line.lower().startswith("@attribute"):
                    attr_name = line.split()[1].lower()
                    if attr_name != "class" and "numeric" in line.lower():
                        feature_count += 1
                continue

            if not line.startswith("{") or not line.endswith("}"):
                raise ValueError(f"Unsupported sparse ARFF row format in {path}: {line!r}")

            contents = line[1:-1].strip()
            if not contents:
                continue

            dense = np.zeros(feature_count, dtype=np.float32)
            label_value = None
            for item in contents.split(","):
                token = item.strip()
                if not token:
                    continue
                if " " in token:
                    idx_str, val_str = token.split(None, 1)
                    idx = int(idx_str)
                    if 0 <= idx < feature_count:
                        dense[idx] = float(val_str)
                    continue
                label_value = float(token)

            if label_value is None:
                raise ValueError(f"Missing class label in sparse ARFF row: {line!r}")
            rows.append(dense)
            labels.append(int(label_value))

    if not rows:
        raise ValueError(f"No data rows were parsed from sparse ARFF file: {path}")

    X = np.vstack(rows).astype(np.float32, copy=False)
    y = np.asarray(labels, dtype=np.int64)
    return X, y


def _load_dorothea_uci() -> tuple[np.ndarray, np.ndarray]:
    cached_X = _DOROTHEA_CACHE.get("X")
    cached_y = _DOROTHEA_CACHE.get("y")
    if cached_X is not None and cached_y is not None:
        return cached_X, cached_y

    local_arff = REPO_ROOT / "articles/2027/bioinformatics/data/dorothea.arff"
    X_raw = None
    y_raw = None

    if local_arff.exists():
        try:
            X_raw, y_raw = _load_sparse_weka_arff(local_arff)
        except Exception:
            X_raw = None
            y_raw = None

    if X_raw is None or y_raw is None:
        try:
            from ucimlrepo import fetch_ucirepo

            dataset = fetch_ucirepo(id=169)
            X_raw = dataset.data.features
            y_raw = dataset.data.targets
        except Exception:
            # UCI currently lists DOROTHEA but does not always expose it via ucimlrepo import.
            # Fallback to OpenML's DOROTHEA mirror (data_id=4137) using direct ARFF parsing.
            try:
                import arff
                import openml

                ds = openml.datasets.get_dataset(4137, download_data=True)
                with open(ds.data_file, "r", encoding="utf-8", errors="ignore") as f:
                    parsed = arff.load(f)
                mat = np.asarray(parsed["data"], dtype=object)
                X_raw = mat[:, :-1]
                y_raw = mat[:, -1]
            except Exception as exc:  # pragma: no cover - depends on network/OpenML availability
                raise SystemExit(
                    "Could not load real DOROTHEA data from the local ARFF file, ucimlrepo(id=169), or OpenML(data_id=4137). "
                    "Ensure articles/2027/bioinformatics/data/dorothea.arff exists or install openml/liac-arff for the remote fallback."
                ) from exc

    # Robustly support DataFrame/Series-like objects and plain arrays.
    if hasattr(X_raw, "to_numpy"):
        X_np = X_raw.to_numpy()
    else:
        X_np = np.asarray(X_raw)

    if hasattr(y_raw, "to_numpy"):
        y_np = y_raw.to_numpy()
    else:
        y_np = np.asarray(y_raw)

    if y_np.ndim > 1:
        y_np = y_np[:, 0]

    # Try direct numeric conversion for labels first.
    try:
        y_num = y_np.astype(np.int64)
    except Exception:
        y_text = np.asarray([str(v).strip().lower() for v in y_np])
        mapped = np.full(y_text.shape[0], -1, dtype=np.int64)
        mapped[np.isin(y_text, ["positive", "pos", "1", "true", "yes", "up"])] = 1
        mapped[np.isin(y_text, ["negative", "neg", "0", "false", "no", "down"])] = 0
        if np.any(mapped < 0):
            uniques, inv = np.unique(y_text, return_inverse=True)
            if uniques.shape[0] != 2:
                raise SystemExit(
                    "dorothea_uci target mapping failed: expected binary labels, got "
                    f"{uniques.shape[0]} unique values"
                )
            y_num = inv.astype(np.int64)
        else:
            y_num = mapped

    # Normalize common binary encodings to 0/1 for backend compatibility.
    uniq = np.unique(y_num)
    if uniq.shape[0] != 2:
        raise SystemExit(
            "dorothea_uci target mapping failed: expected binary labels, got "
            f"{uniq.shape[0]} unique values"
        )
    if np.any(y_num < 0):
        y_num = (y_num == np.max(uniq)).astype(np.int64, copy=False)
    elif not np.array_equal(uniq, np.array([0, 1], dtype=np.int64)):
        y_num = (y_num == np.max(uniq)).astype(np.int64, copy=False)

    # Convert feature matrix to numeric and replace missing values deterministically.
    if X_np.dtype == object:
        X_num = np.empty(X_np.shape, dtype=np.float32)
        for j in range(X_np.shape[1]):
            col = np.asarray(X_np[:, j])
            vals = np.empty(col.shape[0], dtype=np.float32)
            for i, v in enumerate(col):
                try:
                    vals[i] = float(v)
                except Exception:
                    vals[i] = np.nan
            X_num[:, j] = vals
        X_np = X_num
    else:
        X_np = X_np.astype(np.float32, copy=False)

    X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)

    # If effectively binary, compress to uint8 to keep memory pressure low.
    unique_probe = np.unique(X_np[: min(5000, X_np.shape[0]), : min(2000, X_np.shape[1])])
    if unique_probe.size <= 3 and np.all(np.isin(unique_probe, [0.0, 1.0, -1.0])):
        if np.any(X_np < 0.0):
            X_np = (X_np > 0.0).astype(np.uint8, copy=False)
        else:
            X_np = X_np.astype(np.uint8, copy=False)

    _DOROTHEA_CACHE["X"] = X_np
    _DOROTHEA_CACHE["y"] = y_num.astype(np.int64, copy=False)
    return _DOROTHEA_CACHE["X"], _DOROTHEA_CACHE["y"]


def run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    _maybe_enable_insecure_ssl(args.insecure_ssl_downloads)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    selection_names = set(args.arms) if args.arms else set(DEFAULT_ARMS)
    selections = [selection for selection in DEFAULT_SELECTIONS if selection.name in selection_names]
    if not selections:
        raise SystemExit(f"No valid selections requested. Available: {', '.join(DEFAULT_ARMS)}")

    if args.models and args.use_selected_paper_models:
        raise SystemExit("Use either --models or --use-selected-paper-models, not both.")

    if args.use_selected_paper_models:
        models = list(SELECTED_PAPER_MODELS)
    elif args.models:
        unknown = sorted(set(args.models) - set(MODEL_LIBRARY.keys()))
        if unknown:
            raise SystemExit(
                "Unknown models: " + ", ".join(unknown)
                + ". Known: " + ", ".join(sorted(MODEL_LIBRARY.keys()))
            )
        models = [MODEL_LIBRARY[name] for name in args.models]
    else:
        models = list(DEFAULT_MODELS)

    fit_budgets = _parse_fit_budget_values(args.fit_time_budget_seconds)

    raw_rows: list[dict[str, Any]] = []

    for fit_budget in fit_budgets:
        for snp_count in args.snp_counts:
            for seed in args.seeds:
                X, y, dataset_name = _generate_dataset(args, int(snp_count), int(seed))

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=args.test_size,
                    random_state=int(seed),
                    stratify=y,
                )

                for selection in selections:
                    X_train_sel, X_test_sel = _apply_selection(
                        X_train,
                        y_train,
                        X_test,
                        selection,
                    )

                    for model in models:
                        row = {
                            "dataset": dataset_name,
                            "n_snps": int(snp_count),
                            "seed": int(seed),
                            "model": model.name,
                            "backend": model.backend,
                            "selection": selection.name,
                            "fit_time_budget_seconds": "" if fit_budget is None else str(float(fit_budget)),
                            "status": "ok",
                            "error": "",
                            "fit_seconds": "",
                            "predict_seconds": "",
                            "f1_macro": "",
                            "n_rules": "",
                            "n_atoms": "",
                        }
                        previous_handler = None
                        try:
                            cooperative_budget = (
                                _cooperative_fit_budget(float(fit_budget))
                                if fit_budget is not None
                                else None
                            )
                            clf = ScoredRuleSetClassifier(
                                backend=model.backend,
                                backend_params=dict(model.backend_params),
                                max_fit_seconds=cooperative_budget,
                                random_state=int(seed),
                            )

                            if fit_budget is not None:
                                previous_handler = signal.getsignal(signal.SIGALRM)
                                signal.signal(signal.SIGALRM, _fit_timeout_handler)
                                # Hard wall-clock safety net; the cooperative
                                # budget above should normally stop earlier.
                                signal.setitimer(signal.ITIMER_REAL, float(fit_budget))

                            t0 = time.perf_counter()
                            clf.fit(X_train_sel, y_train)
                            fit_seconds = time.perf_counter() - t0

                            if fit_budget is not None:
                                signal.setitimer(signal.ITIMER_REAL, 0.0)
                                if previous_handler is not None:
                                    signal.signal(signal.SIGALRM, previous_handler)

                            t1 = time.perf_counter()
                            y_pred = clf.predict(X_test_sel)
                            predict_seconds = time.perf_counter() - t1

                            f1 = f1_score(y_test, y_pred, average="macro")

                            ruleset = clf.to_ruleset()
                            n_rules = len(ruleset.rules)
                            n_atoms = int(sum(len(rule.atoms) for rule in ruleset.rules))

                            row["fit_seconds"] = f"{fit_seconds:.6f}"
                            row["predict_seconds"] = f"{predict_seconds:.6f}"
                            row["f1_macro"] = f"{f1:.6f}"
                            row["n_rules"] = str(n_rules)
                            row["n_atoms"] = str(n_atoms)
                        except FitTimeoutError as exc:
                            row["status"] = "timeout"
                            row["error"] = f"{type(exc).__name__}: {exc}"
                        except FitBudgetExceededError as exc:
                            # Cooperative setup-phase timeout: the estimator could
                            # not build a viable model within max_fit_seconds.
                            row["status"] = "timeout"
                            row["error"] = f"{type(exc).__name__}: {exc}"
                        except Exception as exc:
                            row["status"] = "error"
                            row["error"] = f"{type(exc).__name__}: {exc}"
                        finally:
                            if fit_budget is not None:
                                signal.setitimer(signal.ITIMER_REAL, 0.0)
                                if previous_handler is not None:
                                    signal.signal(signal.SIGALRM, previous_handler)

                        raw_rows.append(row)
                        _checkpoint_outputs(results_dir, raw_rows, args, selections, models)

    return _checkpoint_outputs(results_dir, raw_rows, args, selections, models)


def _maybe_enable_insecure_ssl(enabled: bool) -> None:
    if not enabled:
        return
    ssl._create_default_https_context = ssl._create_unverified_context


def _aggregate_summary(raw_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, int, str, str], list[dict[str, Any]]] = {}
    for row in raw_rows:
        budget = str(row.get("fit_time_budget_seconds", ""))
        key = (budget, int(row["n_snps"]), row["model"], row["selection"])
        grouped.setdefault(key, []).append(row)

    summaries: list[dict[str, str]] = []
    baseline_map: dict[tuple[str, str, str], tuple[float | None, float | None]] = {}

    for budget, n_snps, model, selection in sorted(grouped.keys(), key=lambda item: (item[0], item[1], item[2], item[3])):
        rows = grouped[(budget, n_snps, model, selection)]
        ok_rows = [r for r in rows if r["status"] == "ok"]

        f1_vals = [float(r["f1_macro"]) for r in ok_rows if r["f1_macro"] != ""]
        fit_vals = [float(r["fit_seconds"]) for r in ok_rows if r["fit_seconds"] != ""]
        pred_vals = [float(r["predict_seconds"]) for r in ok_rows if r["predict_seconds"] != ""]
        rules_vals = [float(r["n_rules"]) for r in ok_rows if r["n_rules"] != ""]
        atoms_vals = [float(r["n_atoms"]) for r in ok_rows if r["n_atoms"] != ""]

        f1_mean = _safe_mean(f1_vals)
        fit_mean = _safe_mean(fit_vals)

        if n_snps == 20:
            baseline_map[(budget, model, selection)] = (f1_mean, fit_mean)

        base_f1, base_fit = baseline_map.get((budget, model, selection), (None, None))

        f1_delta = None
        f1_drop_pct = None
        fit_ratio = None
        f1_delta_vs_none = None
        fit_delta_vs_none = None

        if f1_mean is not None and base_f1 is not None:
            f1_delta = f1_mean - base_f1
            if base_f1 != 0.0:
                f1_drop_pct = 100.0 * (base_f1 - f1_mean) / base_f1
        if fit_mean is not None and base_fit is not None and base_fit > 0:
            fit_ratio = fit_mean / base_fit

        none_rows = grouped.get((budget, n_snps, model, "none"), [])
        none_ok = [r for r in none_rows if r["status"] == "ok"]
        none_f1 = _safe_mean([float(r["f1_macro"]) for r in none_ok if r["f1_macro"] != ""])
        none_fit = _safe_mean([float(r["fit_seconds"]) for r in none_ok if r["fit_seconds"] != ""])
        if selection != "none" and f1_mean is not None and none_f1 is not None:
            f1_delta_vs_none = f1_mean - none_f1
        if selection != "none" and fit_mean is not None and none_fit is not None:
            fit_delta_vs_none = fit_mean - none_fit

        summaries.append(
            {
                "fit_time_budget_seconds": budget,
                "n_snps": str(n_snps),
                "model": model,
                "selection": selection,
                "n_runs": str(len(rows)),
                "ok_runs": str(len(ok_rows)),
                "f1_macro_mean": _fmt(f1_mean),
                "f1_macro_std": _fmt(_safe_std(f1_vals)),
                "fit_seconds_mean": _fmt(fit_mean),
                "fit_seconds_std": _fmt(_safe_std(fit_vals)),
                "predict_seconds_mean": _fmt(_safe_mean(pred_vals)),
                "predict_seconds_std": _fmt(_safe_std(pred_vals)),
                "n_rules_mean": _fmt(_safe_mean(rules_vals)),
                "n_atoms_mean": _fmt(_safe_mean(atoms_vals)),
                "f1_delta_vs_none": _fmt(f1_delta_vs_none),
                "fit_seconds_delta_vs_none": _fmt(fit_delta_vs_none),
                "f1_delta_vs_20": _fmt(f1_delta),
                "f1_drop_pct_vs_20": _fmt(f1_drop_pct),
                "fit_time_ratio_vs_20": _fmt(fit_ratio),
            }
        )

    return summaries


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown_report(
    path: Path,
    summary_rows: list[dict[str, str]],
    args: argparse.Namespace,
    selections: list[SelectionSpec],
    models: list[ModelSpec],
) -> None:
    lines: list[str] = []
    lines.append("# SNP Scaling vs Feature Selection (First Step, no 1000G)")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    if args.dataset_source == "synth_epistasis":
        lines.append(f"- Dataset: synthetic epistasis (`n_samples={args.n_samples}`, `heritability={args.heritability}`)")
    elif args.dataset_source == "dorothea_uci":
        lines.append("- Dataset: UCI DOROTHEA (`ucimlrepo(id=169)`, real sparse drug-discovery classification)")
        lines.append("- Features: original full DOROTHEA space (no slicing)")
    else:
        lines.append(
            "- Dataset: sklearn_classification "
            f"(`n_samples={args.n_samples}`, `informative={args.sklearn_informative}`, "
            f"`flip_y={args.sklearn_flip_y}`, `class_sep={args.sklearn_class_sep}`)"
        )
    lines.append(f"- SNP counts: {', '.join(str(x) for x in args.snp_counts)}")
    lines.append(f"- Seeds: {', '.join(str(x) for x in args.seeds)}")
    lines.append(f"- Models: {', '.join(model.name for model in models)}")
    lines.append(f"- Feature selection: {', '.join(selection.name for selection in selections)}")
    lines.append("- Metric focus: fixed-budget F1 deltas against the same no-selection baseline, then fit_seconds and model size (rules/atoms)")
    lines.append("")

    by_key: dict[tuple[str, int, str, str], dict[str, str]] = {}
    for row in summary_rows:
        by_key[(str(row.get("fit_time_budget_seconds", "")), int(row["n_snps"]), row["model"], row["selection"])] = row

    comparison_rows: list[dict[str, str]] = []
    candidate_selections = [selection.name for selection in selections if selection.name != "none"]

    for budget in sorted({row.get("fit_time_budget_seconds", "") for row in summary_rows}):
        for n_snps in sorted({int(row["n_snps"]) for row in summary_rows if str(row.get("fit_time_budget_seconds", "")) == str(budget)}):
            for model in sorted({row["model"] for row in summary_rows if str(row.get("fit_time_budget_seconds", "")) == str(budget)}):
                none_row = by_key.get((str(budget), n_snps, model, "none"))
                if none_row is None:
                    continue

                none_f1 = float(none_row["f1_macro_mean"]) if none_row["f1_macro_mean"] else None
                none_fit = float(none_row["fit_seconds_mean"]) if none_row["fit_seconds_mean"] else None

                for sel_name in candidate_selections:
                    sel_row = by_key.get((str(budget), n_snps, model, sel_name))
                    if sel_row is None:
                        continue

                    sel_f1 = float(sel_row["f1_macro_mean"]) if sel_row["f1_macro_mean"] else None
                    sel_fit = float(sel_row["fit_seconds_mean"]) if sel_row["fit_seconds_mean"] else None

                    delta_f1 = None if (sel_f1 is None or none_f1 is None) else (sel_f1 - none_f1)
                    delta_fit = None if (sel_fit is None or none_fit is None) else (sel_fit - none_fit)

                    comparison_rows.append(
                        {
                            "fit_time_budget_seconds": str(budget),
                            "n_snps": str(n_snps),
                            "model": model,
                            "selection": sel_name,
                            "ok_none": none_row["ok_runs"],
                            "ok_sel": sel_row["ok_runs"],
                            "f1_none": none_row["f1_macro_mean"],
                            "f1_sel": sel_row["f1_macro_mean"],
                            "delta_f1": _fmt(delta_f1),
                            "fit_none": none_row["fit_seconds_mean"],
                            "fit_sel": sel_row["fit_seconds_mean"],
                            "delta_fit": _fmt(delta_fit),
                        }
                    )

    lines.append("## Within-Model Selection Deltas at Fixed Time Budgets (reference: none)")
    lines.append("")
    lines.append("Interpret this table row-wise: each row compares one model with and without feature selection at the same SNP count.")
    lines.append("")

    header = (
        "| budget_s | n_snps | model | selection | ok_none | ok_sel | f1_none | f1_sel | delta_f1_sel_minus_none | "
        "fit_none_s | fit_sel_s | delta_fit_s_sel_minus_none |"
    )
    sep = "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines.append(header)
    lines.append(sep)

    if comparison_rows:
        for row in sorted(comparison_rows, key=lambda r: (float(r["fit_time_budget_seconds"] or 0.0), int(r["n_snps"]), r["model"], r["selection"])):
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["fit_time_budget_seconds"],
                        row["n_snps"],
                        row["model"],
                        row["selection"],
                        row["ok_none"],
                        row["ok_sel"],
                        row["f1_none"],
                        row["f1_sel"],
                        row["delta_f1"],
                        row["fit_none"],
                        row["fit_sel"],
                        row["delta_fit"],
                    ]
                )
                + " |"
            )
    else:
        lines.append("| - | - | - | - | - | - | - | - | - | - | - | - |")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Report focus is strictly within-model under the same fixed fit budget (none vs selection) to isolate the cost of feature-selection decisions.")
    lines.append("- The primary comparison is F1 delta against the same no-selection baseline at identical budget and SNP count, not timeout counts alone.")
    lines.append("- Different model families keep their own internal stopping criteria; compare models only with explicit fairness caveats.")
    lines.append("- This experiment intentionally compares fixed-budget behavior rather than hard timeout frequency as the primary outcome.")
    lines.append("- The same runner can be reused for synthetic and real DOROTHEA benchmarks by varying dataset-source and budget values.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _apply_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    selection: SelectionSpec,
) -> tuple[np.ndarray, np.ndarray]:
    if selection.name == "none":
        return X_train, X_test

    k = int(min(selection.k, X_train.shape[1]))
    if k <= 0:
        return X_train, X_test

    if selection.name == "kbest20_mi":
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_train_sel = selector.fit_transform(X_train, y_train)
        X_test_sel = selector.transform(X_test)
        return X_train_sel, X_test_sel

    if selection.name == "l1logreg20":
        estimator = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            random_state=0,
            max_iter=200,
            C=0.5,
        )
        estimator.fit(X_train, y_train)
        abs_coef = np.abs(estimator.coef_)
        if abs_coef.ndim == 2:
            importance = abs_coef.mean(axis=0)
        else:
            importance = abs_coef

        if np.allclose(importance, 0.0):
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
            X_train_sel = selector.fit_transform(X_train, y_train)
            X_test_sel = selector.transform(X_test)
            return X_train_sel, X_test_sel

        threshold = np.sort(importance)[-k] if k < len(importance) else 0.0
        selector = SelectFromModel(estimator, prefit=True, threshold=threshold)
        X_train_sel = selector.transform(X_train)
        X_test_sel = selector.transform(X_test)

        if X_train_sel.shape[1] > k:
            top_idx = np.argsort(importance)[-k:]
            X_train_sel = X_train[:, top_idx]
            X_test_sel = X_test[:, top_idx]

        return X_train_sel, X_test_sel

    raise ValueError(f"Unknown selection method: {selection.name}")


def main() -> int:
    args = parse_args()
    raw_csv, summary_csv, report_md = run(args)
    print(f"Wrote raw results: {raw_csv}")
    print(f"Wrote summary:     {summary_csv}")
    print(f"Wrote report:      {report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
