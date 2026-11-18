#!/usr/bin/env python3
"""Section 4 experiment: Extreme-dimensional SNP scaling (p up to 100,000)
comparing Feature Selection with Generic vs. Feature-Type-Aware Atom Construction.

Setup:
- SNP dataset generated via catgen.simulate_snp_glm (n=2000, n_causal=5 dominant SNPs)
- p in {2000, 10000, 50000, 100000}
- Feature Selection: SelectKBest(f_classif, k=20) (or mutual_info_classif via --score-func)
- Arms:
  * fs_generic : FS (kbest20_f) + generic atom construction (hybrid / quantile)
  * fs_typed   : FS (kbest20_f) + feature-type-aware construction (genotype_aware)
- Models: RuleGP, RuleNSGA2, RulePLCS
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[4]
PKG_SRC = REPO_ROOT / "artifacts/packages/python/pypi-scoredrulesets/src"
CATGEN_SRC = REPO_ROOT / "artifacts/packages/python/pypi-catgen/src"
for _p in (PKG_SRC, CATGEN_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from catgen import simulate_snp_glm  # noqa: E402
from scoredrulesets.estimators.sklearn_wrapper import ScoredRuleSetClassifier  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    name: str
    backend: str
    backend_params: dict[str, Any]


NATIVE_BACKENDS = {"rulegp", "rulensga2"}

ARM_OVERRIDES: dict[str, dict[str, dict[str, Any]]] = {
    "fs_generic": {
        "native": {"atom_space_strategy": "hybrid"},
        "ruleplcs": {"feature_typing_strategy": "auto_low_cardinality"},
    },
    "fs_typed": {
        "native": {"atom_space_strategy": "genotype_aware"},
        "ruleplcs": {"feature_typing_strategy": "genotype_aware"},
    },
}

DEFAULT_ARMS = ("fs_generic", "fs_typed")

DEFAULT_MODELS = (
    ModelSpec(
        name="wrapper_rulegp",
        backend="rulegp",
        backend_params={
            "population_size": 40,
            "max_generations": 30,
            "max_rules": 8,
            "max_atoms_per_rule": 4,
        },
    ),
    ModelSpec(
        name="wrapper_rulensga2",
        backend="rulensga2",
        backend_params={
            "population_size": 50,
            "generations": 40,
            "max_rules": 8,
            "max_atoms_per_rule": 4,
        },
    ),
    ModelSpec(
        name="wrapper_ruleplcs",
        backend="ruleplcs",
        backend_params={
            "population_size": 100,
            "n_iterations": 20,
            "n_repetitions": 1,
            "max_rules": 10,
        },
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extreme-dimensional SNP scaling: FS Generic vs FS Typed"
    )
    parser.add_argument(
        "--snp-counts", nargs="+", type=int, default=[2000, 10000, 50000, 100000]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--n-obs", type=int, default=2000)
    parser.add_argument("--n-causal", type=int, default=5)
    parser.add_argument("--k-features", type=int, default=20)
    parser.add_argument("--maf-min", type=float, default=0.15)
    parser.add_argument("--maf-max", type=float, default=0.45)
    parser.add_argument("--beta0", type=float, default=-1.2)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--score-func",
        choices=["f_classif", "mutual_info_classif"],
        default="f_classif",
    )
    parser.add_argument(
        "--results-dir",
        default=str(
            REPO_ROOT
            / "agent-workbench/projects/feature-vs-atom-selection-bioinformatics/results/snp_glm_extreme_fs_vs_typed"
        ),
    )
    return parser.parse_args()


def _simulate_main_scenario(
    n_obs: int,
    n_snp: int,
    n_causal: int,
    maf: tuple[float, float],
    beta0: float,
    beta: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    if n_causal > n_snp:
        raise ValueError("n_causal must not exceed n_snp")
    list_ia = [[-1] for _ in range(n_causal)]
    list_snp = [[i + 1] for i in range(n_causal)]  # scrime 1-based
    sim = simulate_snp_glm(
        n_obs=n_obs,
        n_snp=n_snp,
        list_ia=list_ia,
        list_snp=list_snp,
        beta0=beta0,
        beta=[beta] * n_causal,
        maf=maf,
        random_state=seed,
    )
    causal_indices = list(range(n_causal))
    return sim.x, sim.y, causal_indices


_FEATURE_RE = re.compile(r"^[A-Za-z]+(\d+)$")


def _referenced_feature_indices(ruleset: Any) -> set[int]:
    referenced: set[int] = set()
    for rule in ruleset.rules:
        for atom in rule.atoms:
            feat = atom.feature
            if isinstance(feat, (int, np.integer)):
                referenced.add(int(feat))
                continue
            match = _FEATURE_RE.match(str(feat))
            if match:
                referenced.add(int(match.group(1)))
    return referenced


def _resolve_overrides(backend: str, arm: str) -> dict[str, Any]:
    family = "native" if backend in NATIVE_BACKENDS else backend
    arm_map = ARM_OVERRIDES.get(arm)
    if arm_map is None:
        raise ValueError(f"Unknown arm: {arm}")
    return dict(arm_map.get(family, {}))


def _safe_mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _safe_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    return float(statistics.stdev(values))


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


RAW_FIELDS = [
    "dataset",
    "n_snps",
    "n_causal",
    "seed",
    "arm",
    "model",
    "backend",
    "status",
    "error",
    "fs_seconds",
    "fit_seconds",
    "total_fit_seconds",
    "predict_seconds",
    "f1_macro",
    "n_rules",
    "n_atoms",
    "fs_causal_recall",
    "model_causal_recall",
    "spurious_feats",
]


def run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    arms = list(args.arms) if args.arms else list(DEFAULT_ARMS)
    models = (
        [m for m in DEFAULT_MODELS if m.name in set(args.models)]
        if args.models
        else list(DEFAULT_MODELS)
    )

    raw_rows: list[dict[str, Any]] = []

    for snp_count in args.snp_counts:
        for seed in args.seeds:
            X, y, causal = _simulate_main_scenario(
                n_obs=args.n_obs,
                n_snp=int(snp_count),
                n_causal=int(args.n_causal),
                maf=(args.maf_min, args.maf_max),
                beta0=args.beta0,
                beta=args.beta,
                seed=int(seed),
            )
            causal_set = set(causal)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=args.test_size, random_state=int(seed), stratify=y
            )

            # Univariate Feature Selection on Train
            t_fs_0 = time.perf_counter()
            k = min(args.k_features, int(snp_count))
            score_func = (
                f_classif
                if args.score_func == "f_classif"
                else mutual_info_classif
            )
            selector = SelectKBest(score_func=score_func, k=k)
            X_train_fs = selector.fit_transform(X_train, y_train)
            X_test_fs = selector.transform(X_test)
            fs_seconds = time.perf_counter() - t_fs_0

            selected_indices = np.where(selector.get_support())[0]
            fs_causal_recovered = set(selected_indices) & causal_set
            fs_recall = len(fs_causal_recovered) / len(causal_set) if causal_set else 0.0

            for arm in arms:
                for model in models:
                    row: dict[str, Any] = {
                        "dataset": "snp_glm_extreme",
                        "n_snps": int(snp_count),
                        "n_causal": int(args.n_causal),
                        "seed": int(seed),
                        "arm": arm,
                        "model": model.name,
                        "backend": model.backend,
                        "status": "ok",
                        "error": "",
                        "fs_seconds": f"{fs_seconds:.6f}",
                        "fit_seconds": "",
                        "total_fit_seconds": "",
                        "predict_seconds": "",
                        "f1_macro": "",
                        "n_rules": "",
                        "n_atoms": "",
                        "fs_causal_recall": f"{fs_recall:.6f}",
                        "model_causal_recall": "",
                        "spurious_feats": "",
                    }
                    try:
                        backend_params = dict(model.backend_params)
                        backend_params.update(_resolve_overrides(model.backend, arm))
                        clf = ScoredRuleSetClassifier(
                            backend=model.backend,
                            backend_params=backend_params,
                            random_state=int(seed),
                        )

                        t0 = time.perf_counter()
                        clf.fit(X_train_fs, y_train)
                        fit_seconds = time.perf_counter() - t0
                        total_fit_seconds = fit_seconds + fs_seconds

                        t1 = time.perf_counter()
                        y_pred = clf.predict(X_test_fs)
                        predict_seconds = time.perf_counter() - t1

                        f1 = f1_score(y_test, y_pred, average="macro")
                        rs = clf.to_ruleset()
                        rel_feats = _referenced_feature_indices(rs)
                        # map relative feature indices to original feature space
                        orig_referenced = {selected_indices[idx] for idx in rel_feats if idx < len(selected_indices)}
                        model_causal_recovered = orig_referenced & causal_set
                        model_recall = len(model_causal_recovered) / len(causal_set) if causal_set else 0.0
                        spurious = orig_referenced - causal_set

                        row["fit_seconds"] = f"{fit_seconds:.6f}"
                        row["total_fit_seconds"] = f"{total_fit_seconds:.6f}"
                        row["predict_seconds"] = f"{predict_seconds:.6f}"
                        row["f1_macro"] = f"{f1:.6f}"
                        row["n_rules"] = str(len(rs.rules))
                        row["n_atoms"] = str(int(sum(len(r.atoms) for r in rs.rules)))
                        row["model_causal_recall"] = f"{model_recall:.6f}"
                        row["spurious_feats"] = str(len(spurious))
                    except Exception as exc:  # noqa: BLE001
                        row["status"] = "error"
                        row["error"] = f"{type(exc).__name__}: {exc}"

                    raw_rows.append(row)

    raw_csv = results_dir / "snp_glm_extreme_fs_vs_typed_raw.csv"
    _write_csv(raw_csv, raw_rows, RAW_FIELDS)

    summary_rows = _aggregate_summary(raw_rows)
    summary_csv = results_dir / "snp_glm_extreme_fs_vs_typed_summary.csv"
    _write_csv(
        summary_csv,
        summary_rows,
        [
            "n_snps",
            "model",
            "arm",
            "n_runs",
            "ok_runs",
            "f1_macro_mean",
            "f1_macro_std",
            "fit_seconds_mean",
            "total_fit_seconds_mean",
            "n_rules_mean",
            "n_atoms_mean",
            "model_causal_recall_mean",
            "spurious_feats_mean",
            "f1_delta_vs_fs_generic",
        ],
    )

    report_md = results_dir / "snp_glm_extreme_fs_vs_typed_report.md"
    _write_report(report_md, summary_rows, args)
    return raw_csv, summary_csv, report_md


def _aggregate_summary(raw_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for row in raw_rows:
        key = (int(row["n_snps"]), row["model"], row["arm"])
        grouped.setdefault(key, []).append(row)

    generic_f1: dict[tuple[int, str], float | None] = {}
    for (n_snps, model, arm), rows in grouped.items():
        if arm != "fs_generic":
            continue
        ok = [r for r in rows if r["status"] == "ok" and r["f1_macro"] != ""]
        generic_f1[(n_snps, model)] = _safe_mean([float(r["f1_macro"]) for r in ok])

    summaries: list[dict[str, str]] = []
    for n_snps, model, arm in sorted(grouped.keys()):
        rows = grouped[(n_snps, model, arm)]
        ok = [r for r in rows if r["status"] == "ok"]

        f1_vals = [float(r["f1_macro"]) for r in ok if r["f1_macro"] != ""]
        fit_vals = [float(r["fit_seconds"]) for r in ok if r["fit_seconds"] != ""]
        tot_vals = [float(r["total_fit_seconds"]) for r in ok if r["total_fit_seconds"] != ""]
        rules_vals = [float(r["n_rules"]) for r in ok if r["n_rules"] != ""]
        atoms_vals = [float(r["n_atoms"]) for r in ok if r["n_atoms"] != ""]
        recall_vals = [float(r["model_causal_recall"]) for r in ok if r["model_causal_recall"] != ""]
        spur_vals = [float(r["spurious_feats"]) for r in ok if r["spurious_feats"] != ""]

        f1_mean = _safe_mean(f1_vals)
        base = generic_f1.get((n_snps, model))
        f1_delta = (
            f1_mean - base if (f1_mean is not None and base is not None) else None
        )

        summaries.append(
            {
                "n_snps": str(n_snps),
                "model": model,
                "arm": arm,
                "n_runs": str(len(rows)),
                "ok_runs": str(len(ok)),
                "f1_macro_mean": _fmt(f1_mean),
                "f1_macro_std": _fmt(_safe_std(f1_vals)),
                "fit_seconds_mean": _fmt(_safe_mean(fit_vals)),
                "total_fit_seconds_mean": _fmt(_safe_mean(tot_vals)),
                "n_rules_mean": _fmt(_safe_mean(rules_vals)),
                "n_atoms_mean": _fmt(_safe_mean(atoms_vals)),
                "model_causal_recall_mean": _fmt(_safe_mean(recall_vals)),
                "spurious_feats_mean": _fmt(_safe_mean(spur_vals)),
                "f1_delta_vs_fs_generic": _fmt(f1_delta),
            }
        )
    return summaries


def _write_report(
    path: Path, summary_rows: list[dict[str, str]], args: argparse.Namespace
) -> None:
    lines: list[str] = []
    lines.append("# Extreme-Dimensional SNP Scaling: FS Generic vs. FS Type-Aware")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- SNP counts: {', '.join(str(x) for x in args.snp_counts)}")
    lines.append(f"- Seeds: {', '.join(str(x) for x in args.seeds)}")
    lines.append(f"- n_obs: {args.n_obs}, n_causal: {args.n_causal} (dominant model)")
    lines.append(f"- Feature Selection: kbest20_mi (top {args.k_features} features)")
    lines.append("- Arms: fs_generic (hybrid) vs fs_typed (genotype_aware)")
    lines.append("")
    lines.append("## Aggregated Results")
    lines.append("")
    lines.append(
        "| n_snps | model | arm | ok | f1_mean | f1_delta | "
        "causal_recall | spurious | n_rules | n_atoms | fit_s | total_s |"
    )
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(
        summary_rows, key=lambda r: (int(r["n_snps"]), r["model"], r["arm"])
    ):
        lines.append(
            "| "
            + " | ".join(
                [
                    row["n_snps"],
                    row["model"],
                    row["arm"],
                    row["ok_runs"],
                    row["f1_macro_mean"],
                    row["f1_delta_vs_fs_generic"],
                    row["model_causal_recall_mean"],
                    row["spurious_feats_mean"],
                    row["n_rules_mean"],
                    row["n_atoms_mean"],
                    row["fit_seconds_mean"],
                    row["total_fit_seconds_mean"],
                ]
            )
            + " |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    raw_csv, summary_csv, report_md = run(args)
    print(f"raw:     {raw_csv}")
    print(f"summary: {summary_csv}")
    print(f"report:  {report_md}")


if __name__ == "__main__":
    main()
