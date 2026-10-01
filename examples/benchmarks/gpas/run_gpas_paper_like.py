#!/usr/bin/env python3
"""Paper-like GPAS reproduction workflow (HapMap, simulation, optional GENICA).

Goal:
- Run Python GPAS experiments replicating Nunkesser et al. (2007), Table 1.
- Validate results against GPAS mean misclassification rates:
  - HapMap:     0.011
  - Simulation: 0.335
  - GENICA:     0.392 (optional, requires local private dataset)

Notes:
- HapMap data is included in examples/snp/data/hapmap157.csv (Open Data).
- Simulation data is generated on the fly via `catgen.simulate_snp_glm`.
- GENICA is a protected clinical cohort and is NOT bundled in this repository.
  To evaluate on GENICA, pass `--genica-path /path/to/genica63.csv`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

try:
    from scoredrulesets.estimators.logicgp import GPASClassifier
except Exception as exc:  # pragma: no cover - startup guard
    raise SystemExit(
        "Could not import scoredrulesets. "
        "Install artifacts/packages/python/pypi-scoredrulesets first. "
        f"Original error: {exc}"
    )

try:
    from catgen import simulate_snp_glm
except Exception as exc:  # pragma: no cover - startup guard
    raise SystemExit(
        "Could not import catgen. "
        "Install artifacts/packages/python/pypi-catgen first. "
        f"Original error: {exc}"
    )


PAPER_TARGETS = {
    "genica": 0.392,
    "hapmap": 0.011,
    "simulation": 0.335,
}


@dataclass
class DatasetResult:
    name: str
    n_evaluations: int
    mean_misclassification: float
    std_misclassification: float
    min_misclassification: float
    max_misclassification: float
    paper_target: float
    abs_delta_to_target: float
    within_tolerance: bool


@dataclass
class RunConfig:
    profile: str
    seed: int
    output_dir: str
    datasets: list[str]
    genica_generations: int
    hapmap_generations: int
    simulation_generations: int
    hapmap_bagging_runs: int
    gpas_model_selection: str
    simulation_n_datasets: int
    simulation_n_obs: int
    simulation_n_snp: int
    genica_cv_folds: int
    genica_cv_repeats: int
    hapmap_cv_folds: int
    tolerance_genica: float
    tolerance_hapmap: float
    tolerance_simulation: float


def _resolve_profile_defaults(profile: str) -> dict[str, int]:
    if profile == "smoke":
        return {
            "genica_generations": 80,
            "hapmap_generations": 120,
            "simulation_generations": 80,
            "hapmap_bagging_runs": 2,
            "simulation_n_datasets": 6,
            "genica_cv_repeats": 1,
            "tolerance_ppm": 250,
        }
    if profile == "quick":
        return {
            "genica_generations": 600,
            "hapmap_generations": 1200,
            "simulation_generations": 500,
            "hapmap_bagging_runs": 6,
            "simulation_n_datasets": 20,
            "genica_cv_repeats": 1,
            "tolerance_ppm": 150,
        }
    if profile == "paper":
        return {
            "genica_generations": 5000,
            "hapmap_generations": 10000,
            "simulation_generations": 2000,
            "hapmap_bagging_runs": 25,
            "simulation_n_datasets": 50,
            "genica_cv_repeats": 1,
            "tolerance_ppm": 80,
        }
    raise ValueError(f"Unsupported profile: {profile}")


def _resolve_repo_root() -> pathlib.Path:
    # examples/benchmarks/gpas/run_gpas_paper_like.py -> parents[3] is repo root
    return pathlib.Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paper-like GPAS reproduction with validation against GPAS Table-1 targets."
    )
    parser.add_argument("--profile", choices=["smoke", "quick", "paper"], default="quick")
    parser.add_argument(
        "--datasets",
        type=str,
        default="hapmap,simulation",
        help="Comma-separated datasets to evaluate: 'hapmap', 'simulation', 'genica', or 'all'. Default: hapmap,simulation",
    )
    parser.add_argument(
        "--genica-path",
        type=pathlib.Path,
        default=None,
        help="Optional path to genica63.csv (required if evaluating GENICA).",
    )
    parser.add_argument(
        "--hapmap-path",
        type=pathlib.Path,
        default=None,
        help="Path to hapmap157.csv. Default: examples/snp/data/hapmap157.csv",
    )
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Output directory. Default: benchmarks/gpas/<timestamp>",
    )

    parser.add_argument("--genica-generations", type=int, default=None)
    parser.add_argument("--hapmap-generations", type=int, default=None)
    parser.add_argument("--simulation-generations", type=int, default=None)
    parser.add_argument("--hapmap-bagging-runs", type=int, default=None)
    parser.add_argument(
        "--gpas-model-selection",
        type=str,
        choices=["paper", "best_f1", "shortest_zero_train_mcr"],
        default="paper",
        help=(
            "Final GPAS model selection strategy. "
            "'shortest_zero_train_mcr' replicates the paper-style preference "
            "for the shortest model with training MCR=0.0."
        ),
    )
    parser.add_argument("--simulation-n-datasets", type=int, default=None)

    parser.add_argument("--simulation-n-obs", type=int, default=1000)
    parser.add_argument("--simulation-n-snp", type=int, default=50)

    parser.add_argument("--genica-cv-folds", type=int, default=10)
    parser.add_argument("--genica-cv-repeats", type=int, default=None)
    parser.add_argument("--hapmap-cv-folds", type=int, default=9)

    parser.add_argument("--tolerance-genica", type=float, default=None)
    parser.add_argument("--tolerance-hapmap", type=float, default=None)
    parser.add_argument("--tolerance-simulation", type=float, default=None)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress every N loop iterations (minimum 1).",
    )

    return parser.parse_args()


def load_genica(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        rows = [row for row in reader]
    raw = np.asarray(rows, dtype=int)
    y = raw[:, 0].astype(int)
    X = (raw[:, 1:64] - 1).astype(int)
    return X, y


def load_hapmap(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        rows = [row for row in reader]
    raw = np.asarray(rows, dtype=int)
    y = raw[:, 0].astype(int)
    X = (raw[:, 1:158] - 1).astype(int)
    return X, y


def _misclassification(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true != y_pred))


def _should_log_progress(idx: int, total: int, every: int) -> bool:
    step = max(1, int(every))
    return idx == 1 or idx == total or (idx % step == 0)


def _ensemble_predict_binary(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    generations: int,
    seed: int,
    n_models: int,
    model_selection: str,
    stagnation_generations: int,
    progress_every: int,
) -> np.ndarray:
    if n_models <= 1:
        model = GPASClassifier(
            max_generations=generations,
            stagnation_generations=stagnation_generations,
            model_selection=model_selection,
            random_state=seed,
        )
        model.fit(X_train, y_train)
        return model.predict(X_test)

    probs_acc = np.zeros((X_test.shape[0], 2), dtype=float)
    for run_idx in range(n_models):
        model_no = run_idx + 1
        started = time.perf_counter()
        model = GPASClassifier(
            max_generations=generations,
            stagnation_generations=stagnation_generations,
            model_selection=model_selection,
            random_state=seed + run_idx,
        )
        model.fit(X_train, y_train)
        p = model.predict_proba(X_test)

        aligned = np.zeros_like(probs_acc)
        for col_idx, cls in enumerate(model.classes_):
            aligned[:, int(cls)] = p[:, col_idx]
        probs_acc += aligned

        if _should_log_progress(model_no, n_models, progress_every):
            elapsed = time.perf_counter() - started
            print(
                f"  HapMap ensemble model {model_no}/{n_models} finished in {elapsed:.2f}s",
                flush=True,
            )

    probs_acc /= float(n_models)
    return np.argmax(probs_acc, axis=1)


def evaluate_genica(
    path: pathlib.Path,
    *,
    generations: int,
    cv_folds: int,
    cv_repeats: int,
    seed: int,
    model_selection: str,
    progress_every: int,
    output_dir: pathlib.Path,
) -> list[float]:
    X, y = load_genica(path)
    splitter = RepeatedStratifiedKFold(
        n_splits=cv_folds,
        n_repeats=cv_repeats,
        random_state=seed,
    )
    errors: list[float] = []
    total_splits = cv_folds * cv_repeats

    for split_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        started = time.perf_counter()
        model = GPASClassifier(
            max_generations=generations,
            stagnation_generations=max(50, generations // 10),
            model_selection=model_selection,
            random_state=seed + split_idx,
        )
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        fold_err = _misclassification(y[test_idx], pred)
        errors.append(fold_err)
        _write_dataset_checkpoint(
            output_dir,
            "genica",
            errors,
            completed=split_idx,
            total=total_splits,
        )

        if _should_log_progress(split_idx, total_splits, progress_every):
            elapsed = time.perf_counter() - started
            print(
                f"GENICA split {split_idx}/{total_splits} done: "
                f"err={fold_err:.4f}, elapsed={elapsed:.2f}s",
                flush=True,
            )

    return errors


def evaluate_hapmap(
    path: pathlib.Path,
    *,
    generations: int,
    cv_folds: int,
    seed: int,
    bagging_runs: int,
    model_selection: str,
    progress_every: int,
    output_dir: pathlib.Path,
) -> list[float]:
    X, y = load_hapmap(path)
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    errors: list[float] = []

    for split_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        started = time.perf_counter()
        pred = _ensemble_predict_binary(
            X[train_idx],
            y[train_idx],
            X[test_idx],
            generations=generations,
            seed=seed + split_idx * 100,
            n_models=bagging_runs,
            model_selection=model_selection,
            stagnation_generations=max(100, generations // 8),
            progress_every=progress_every,
        )
        fold_err = _misclassification(y[test_idx], pred)
        errors.append(fold_err)
        _write_dataset_checkpoint(
            output_dir,
            "hapmap",
            errors,
            completed=split_idx,
            total=cv_folds,
        )

        if _should_log_progress(split_idx, cv_folds, progress_every):
            elapsed = time.perf_counter() - started
            print(
                f"HapMap split {split_idx}/{cv_folds} done: "
                f"err={fold_err:.4f}, elapsed={elapsed:.2f}s",
                flush=True,
            )

    return errors


def _build_sim_dataset(seed: int, n_obs: int, n_snp: int) -> tuple[np.ndarray, np.ndarray]:
    sim = simulate_snp_glm(
        n_obs=n_obs,
        n_snp=n_snp,
        list_ia=[[-1, 1], [1, 1, 1]],
        list_snp=[[6, 7], [3, 9, 10]],
        beta0=-0.5,
        beta=[1.5, 1.5],
        maf=0.25,
        random_state=seed,
    )
    return np.asarray(sim.x, dtype=int), np.asarray(sim.y, dtype=int)


def evaluate_simulation(
    *,
    generations: int,
    n_datasets: int,
    seed: int,
    model_selection: str,
    n_obs: int,
    n_snp: int,
    progress_every: int,
    output_dir: pathlib.Path,
) -> list[float]:
    errors: list[float] = []
    for d_idx in range(1, n_datasets + 1):
        started = time.perf_counter()
        sim_seed = seed + d_idx * 17
        X, y = _build_sim_dataset(sim_seed, n_obs=n_obs, n_snp=n_snp)

        splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=sim_seed)
        train_idx, test_idx = next(splitter.split(X, y))

        model = GPASClassifier(
            max_generations=generations,
            stagnation_generations=max(50, generations // 10),
            model_selection=model_selection,
            random_state=sim_seed + 1,
        )
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        err = _misclassification(y[test_idx], pred)
        errors.append(err)
        _write_dataset_checkpoint(
            output_dir,
            "simulation",
            errors,
            completed=d_idx,
            total=n_datasets,
        )

        if _should_log_progress(d_idx, n_datasets, progress_every):
            elapsed = time.perf_counter() - started
            print(
                f"Simulation dataset {d_idx}/{n_datasets} done: "
                f"err={err:.4f}, elapsed={elapsed:.2f}s",
                flush=True,
            )

    return errors


def summarize_dataset(
    name: str,
    errors: list[float],
    paper_target: float,
    tolerance: float,
) -> DatasetResult:
    mean_val = float(statistics.mean(errors))
    std_val = float(statistics.stdev(errors)) if len(errors) > 1 else 0.0
    min_val = float(min(errors))
    max_val = float(max(errors))
    delta = abs(mean_val - paper_target)
    within = delta <= tolerance
    return DatasetResult(
        name=name,
        n_evaluations=len(errors),
        mean_misclassification=mean_val,
        std_misclassification=std_val,
        min_misclassification=min_val,
        max_misclassification=max_val,
        paper_target=paper_target,
        abs_delta_to_target=delta,
        within_tolerance=within,
    )


def _write_errors_csv(path: pathlib.Path, errors: list[float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["evaluation_index", "misclassification_rate"])
        for idx, err in enumerate(errors, start=1):
            writer.writerow([idx, f"{err:.8f}"])


def _write_dataset_checkpoint(
    output_dir: pathlib.Path,
    dataset: str,
    values: list[float],
    *,
    completed: int,
    total: int,
) -> None:
    _write_errors_csv(output_dir / f"{dataset}_errors.csv", values)
    progress = {
        "dataset": dataset,
        "completed": int(completed),
        "total": int(total),
        "updated_at": datetime.now(UTC).isoformat(),
        "mean_misclassification": float(statistics.mean(values)) if values else None,
    }
    (output_dir / f"{dataset}_progress.json").write_text(
        json.dumps(progress, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    defaults = _resolve_profile_defaults(args.profile)
    repo_root = _resolve_repo_root()

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (repo_root / f"benchmarks/gpas/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine requested datasets
    raw_ds = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    if "all" in raw_ds:
        requested_datasets = ["hapmap", "simulation", "genica"]
    else:
        requested_datasets = raw_ds

    hapmap_path = args.hapmap_path or (repo_root / "examples/snp/data/hapmap157.csv")

    genica_generations = args.genica_generations or defaults["genica_generations"]
    hapmap_generations = args.hapmap_generations or defaults["hapmap_generations"]
    simulation_generations = args.simulation_generations or defaults["simulation_generations"]
    hapmap_bagging_runs = args.hapmap_bagging_runs or defaults["hapmap_bagging_runs"]
    simulation_n_datasets = args.simulation_n_datasets or defaults["simulation_n_datasets"]
    genica_cv_repeats = args.genica_cv_repeats or defaults["genica_cv_repeats"]

    tol_default = defaults["tolerance_ppm"] / 1000.0
    tolerance_genica = args.tolerance_genica if args.tolerance_genica is not None else tol_default
    tolerance_hapmap = args.tolerance_hapmap if args.tolerance_hapmap is not None else tol_default
    tolerance_sim = args.tolerance_simulation if args.tolerance_simulation is not None else tol_default
    progress_every = max(1, args.progress_every)

    cfg = RunConfig(
        profile=args.profile,
        seed=args.seed,
        output_dir=str(output_dir),
        datasets=requested_datasets,
        genica_generations=genica_generations,
        hapmap_generations=hapmap_generations,
        simulation_generations=simulation_generations,
        hapmap_bagging_runs=hapmap_bagging_runs,
        gpas_model_selection=args.gpas_model_selection,
        simulation_n_datasets=simulation_n_datasets,
        simulation_n_obs=args.simulation_n_obs,
        simulation_n_snp=args.simulation_n_snp,
        genica_cv_folds=args.genica_cv_folds,
        genica_cv_repeats=genica_cv_repeats,
        hapmap_cv_folds=args.hapmap_cv_folds,
        tolerance_genica=tolerance_genica,
        tolerance_hapmap=tolerance_hapmap,
        tolerance_simulation=tolerance_sim,
    )

    print(f"Run output directory: {output_dir}", flush=True)
    print(f"Active profile: {args.profile}", flush=True)
    print(f"Requested datasets: {', '.join(requested_datasets)}", flush=True)

    results: dict[str, DatasetResult] = {}

    # 1. GENICA
    if "genica" in requested_datasets:
        if args.genica_path and args.genica_path.exists():
            print("Running GENICA experiment...", flush=True)
            genica_errors = evaluate_genica(
                args.genica_path,
                generations=genica_generations,
                cv_folds=args.genica_cv_folds,
                cv_repeats=genica_cv_repeats,
                seed=args.seed,
                model_selection=args.gpas_model_selection,
                progress_every=progress_every,
                output_dir=output_dir,
            )
            results["genica"] = summarize_dataset(
                "genica", genica_errors, PAPER_TARGETS["genica"], tolerance_genica
            )
            _write_errors_csv(output_dir / "genica_errors.csv", genica_errors)
        else:
            print(
                "NOTE: GENICA is a protected clinical cohort dataset and is NOT bundled in this repository.\n"
                "Skipping GENICA. (Pass --genica-path /path/to/genica63.csv to evaluate on local private data.)\n",
                flush=True,
            )

    # 2. HapMap
    if "hapmap" in requested_datasets:
        if not hapmap_path.exists():
            print(f"ERROR: HapMap dataset not found at {hapmap_path}", file=sys.stderr)
            return 1
        print("Running HapMap experiment...", flush=True)
        hapmap_errors = evaluate_hapmap(
            hapmap_path,
            generations=hapmap_generations,
            cv_folds=args.hapmap_cv_folds,
            seed=args.seed + 2000,
            bagging_runs=hapmap_bagging_runs,
            model_selection=args.gpas_model_selection,
            progress_every=progress_every,
            output_dir=output_dir,
        )
        results["hapmap"] = summarize_dataset(
            "hapmap", hapmap_errors, PAPER_TARGETS["hapmap"], tolerance_hapmap
        )
        _write_errors_csv(output_dir / "hapmap_errors.csv", hapmap_errors)

    # 3. Simulation
    if "simulation" in requested_datasets:
        print("Running simulation experiment...", flush=True)
        sim_errors = evaluate_simulation(
            generations=simulation_generations,
            n_datasets=simulation_n_datasets,
            seed=args.seed + 4000,
            model_selection=args.gpas_model_selection,
            n_obs=args.simulation_n_obs,
            n_snp=args.simulation_n_snp,
            progress_every=progress_every,
            output_dir=output_dir,
        )
        results["simulation"] = summarize_dataset(
            "simulation", sim_errors, PAPER_TARGETS["simulation"], tolerance_sim
        )
        _write_errors_csv(output_dir / "simulation_errors.csv", sim_errors)

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config": asdict(cfg),
        "paper_targets": PAPER_TARGETS,
        "results": {k: asdict(v) for k, v in results.items()},
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\nValidation summary (misclassification mean vs paper target):")
    all_ok = True
    for item in results.values():
        status = "OK" if item.within_tolerance else "WARN"
        if not item.within_tolerance:
            all_ok = False
        print(
            f"- {item.name:10s} mean={item.mean_misclassification:.3f} "
            f"target={item.paper_target:.3f} "
            f"delta={item.abs_delta_to_target:.3f} [{status}]"
        )

    print(f"\nArtifacts written to: {output_dir}")
    print(f"Summary JSON: {summary_path}")

    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
