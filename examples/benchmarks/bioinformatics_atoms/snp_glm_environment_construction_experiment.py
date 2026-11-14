#!/usr/bin/env python3
"""Section 4 experiment: feature-type-aware atom construction on the SNP-GLM
`environment` scenario (dominant SNP main effects + continuous covariates).

Motivation
----------
The reframed Section 4 argues that feature-type-aware construction should treat
each feature according to its type: compact *genetic-model* atoms for genotype
SNPs, and *supervised* (Fayyad-Irani MDLP) cut-points for continuous
covariates. The ``main`` scenario only exercises the genotype half. This
experiment adds continuous gene-environment covariates via
``catgen.simulate_snp_glm_with_covariates`` so the supervised cut-point half is
exercised too.

The feature matrix is ``[SNPs | covariates]``. Causal SNPs act through a
dominant model; the first ``n_causal_cov`` covariates carry a linear main
effect, the rest are noise. Genotype columns have 2-3 integer levels; covariate
columns are continuous.

Arms (construction strategies)
------------------------------
- ``generic`` : native  -> atom_space_strategy="hybrid",
                           continuous_threshold_strategy="quantile_midpoint"
                 RulePLCS -> feature_typing_strategy="auto_low_cardinality"
- ``typed``   : native  -> atom_space_strategy="genotype_aware",
                           continuous_threshold_strategy="supervised_mdl"
                 RulePLCS -> feature_typing_strategy="genotype_aware"

Note: RulePLCS has no ``continuous_threshold_strategy`` axis, so its two arms
differ only in genotype typing; the continuous covariates are handled the same
way in both arms.

Metrics
-------
- f1_macro, fit_seconds, predict_seconds, n_rules, n_atoms
- snp_recall  : fraction of causal SNPs referenced by the ruleset
- cov_recall  : fraction of causal covariates referenced by the ruleset
- spurious_feats : number of noise features referenced by the ruleset
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
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[4]
PKG_SRC = REPO_ROOT / "artifacts/packages/python/pypi-scoredrulesets/src"
CATGEN_SRC = REPO_ROOT / "artifacts/packages/python/pypi-catgen/src"
for _p in (PKG_SRC, CATGEN_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from catgen import simulate_snp_glm_with_covariates  # noqa: E402
from scoredrulesets.estimators.sklearn_wrapper import ScoredRuleSetClassifier  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    name: str
    backend: str
    backend_params: dict[str, Any]


NATIVE_BACKENDS = {"rulegp", "rulensga2"}

ARM_OVERRIDES: dict[str, dict[str, dict[str, Any]]] = {
    "generic": {
        "native": {
            "atom_space_strategy": "hybrid",
            "continuous_threshold_strategy": "quantile_midpoint",
        },
        "ruleplcs": {"feature_typing_strategy": "auto_low_cardinality"},
    },
    "typed": {
        "native": {
            "atom_space_strategy": "genotype_aware",
            "continuous_threshold_strategy": "supervised_mdl",
        },
        "ruleplcs": {"feature_typing_strategy": "genotype_aware"},
    },
}

DEFAULT_ARMS = ("generic", "typed")

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
        description="Feature-type-aware construction on the SNP-GLM environment scenario"
    )
    parser.add_argument("--snp-counts", nargs="+", type=int, default=[50, 100])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n-obs", type=int, default=2000)
    parser.add_argument("--n-causal-snp", type=int, default=2)
    parser.add_argument("--n-cov", type=int, default=3)
    parser.add_argument("--n-causal-cov", type=int, default=1)
    parser.add_argument("--maf-min", type=float, default=0.15)
    parser.add_argument("--maf-max", type=float, default=0.45)
    parser.add_argument("--beta0", type=float, default=-1.0)
    parser.add_argument("--beta", type=float, default=0.9,
                        help="Per-causal-SNP log-odds effect (dominant model).")
    parser.add_argument("--covariate-beta", type=float, default=1.2,
                        help="Log-odds effect per causal covariate.")
    parser.add_argument("--covariate-corr", type=float, default=0.5,
                        help="Pairwise correlation among covariates.")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--results-dir",
        default=str(
            REPO_ROOT
            / "agent-workbench/projects/feature-vs-atom-selection-bioinformatics/results/snp_glm_environment_construction"
        ),
    )
    return parser.parse_args()


def _simulate_environment_scenario(
    n_obs: int,
    n_snp: int,
    n_causal_snp: int,
    n_cov: int,
    n_causal_cov: int,
    maf: tuple[float, float],
    beta0: float,
    beta: float,
    covariate_beta: float,
    covariate_corr: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Simulate the gene-environment scenario.

    Returns ``(X, y, causal_snp_idx, causal_cov_idx)`` with 0-based indices into
    the combined ``[SNPs | covariates]`` matrix.
    """
    if n_causal_snp > n_snp:
        raise ValueError("n_causal_snp must not exceed n_snp")
    if n_causal_cov > n_cov:
        raise ValueError("n_causal_cov must not exceed n_cov")

    list_ia = [[-1] for _ in range(n_causal_snp)]
    list_snp = [[i + 1] for i in range(n_causal_snp)]  # scrime 1-based
    cov_beta = [covariate_beta] * n_causal_cov + [0.0] * (n_cov - n_causal_cov)
    corr = np.full((n_cov, n_cov), covariate_corr, dtype=float)
    np.fill_diagonal(corr, 1.0)

    sim = simulate_snp_glm_with_covariates(
        n_obs=n_obs,
        n_snp=n_snp,
        list_ia=list_ia,
        list_snp=list_snp,
        beta0=beta0,
        beta=[beta] * n_causal_snp,
        maf=maf,
        covariate_mean=[0.0] * n_cov,
        covariate_cov=corr,
        covariate_beta=cov_beta,
        random_state=seed,
    )
    X = np.hstack([sim.x.astype(float), sim.covariates])
    causal_snp_idx = list(range(n_causal_snp))
    causal_cov_idx = [n_snp + i for i in range(n_causal_cov)]
    return X, sim.y, causal_snp_idx, causal_cov_idx


# Backends export atom features with different single-letter prefixes but the
# same 0-based index (RuleGP/RuleNSGA2: "f{idx}", RulePLCS: "x{idx}").
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
    "n_causal_snp",
    "n_cov",
    "n_causal_cov",
    "seed",
    "arm",
    "model",
    "backend",
    "status",
    "error",
    "fit_seconds",
    "predict_seconds",
    "f1_macro",
    "n_rules",
    "n_atoms",
    "snp_recall",
    "cov_recall",
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
            X, y, causal_snp, causal_cov = _simulate_environment_scenario(
                n_obs=args.n_obs,
                n_snp=int(snp_count),
                n_causal_snp=int(args.n_causal_snp),
                n_cov=int(args.n_cov),
                n_causal_cov=int(args.n_causal_cov),
                maf=(args.maf_min, args.maf_max),
                beta0=args.beta0,
                beta=args.beta,
                covariate_beta=args.covariate_beta,
                covariate_corr=args.covariate_corr,
                seed=int(seed),
            )
            snp_set = set(causal_snp)
            cov_set = set(causal_cov)
            causal_all = snp_set | cov_set

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=args.test_size, random_state=int(seed), stratify=y
            )

            for arm in arms:
                for model in models:
                    row: dict[str, Any] = {
                        "dataset": "snp_glm_environment",
                        "n_snps": int(snp_count),
                        "n_causal_snp": int(args.n_causal_snp),
                        "n_cov": int(args.n_cov),
                        "n_causal_cov": int(args.n_causal_cov),
                        "seed": int(seed),
                        "arm": arm,
                        "model": model.name,
                        "backend": model.backend,
                        "status": "ok",
                        "error": "",
                        "fit_seconds": "",
                        "predict_seconds": "",
                        "f1_macro": "",
                        "n_rules": "",
                        "n_atoms": "",
                        "snp_recall": "",
                        "cov_recall": "",
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
                        clf.fit(X_train, y_train)
                        fit_seconds = time.perf_counter() - t0

                        t1 = time.perf_counter()
                        y_pred = clf.predict(X_test)
                        predict_seconds = time.perf_counter() - t1

                        f1 = f1_score(y_test, y_pred, average="macro")
                        rs = clf.to_ruleset()
                        referenced = _referenced_feature_indices(rs)
                        snp_recall = (
                            len(referenced & snp_set) / len(snp_set) if snp_set else 0.0
                        )
                        cov_recall = (
                            len(referenced & cov_set) / len(cov_set) if cov_set else 0.0
                        )
                        spurious = referenced - causal_all

                        row["fit_seconds"] = f"{fit_seconds:.6f}"
                        row["predict_seconds"] = f"{predict_seconds:.6f}"
                        row["f1_macro"] = f"{f1:.6f}"
                        row["n_rules"] = str(len(rs.rules))
                        row["n_atoms"] = str(int(sum(len(r.atoms) for r in rs.rules)))
                        row["snp_recall"] = f"{snp_recall:.6f}"
                        row["cov_recall"] = f"{cov_recall:.6f}"
                        row["spurious_feats"] = str(len(spurious))
                    except Exception as exc:  # noqa: BLE001 - record and continue
                        row["status"] = "error"
                        row["error"] = f"{type(exc).__name__}: {exc}"

                    raw_rows.append(row)

    raw_csv = results_dir / "snp_glm_environment_construction_raw.csv"
    _write_csv(raw_csv, raw_rows, RAW_FIELDS)

    summary_rows = _aggregate_summary(raw_rows)
    summary_csv = results_dir / "snp_glm_environment_construction_summary.csv"
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
            "n_rules_mean",
            "n_atoms_mean",
            "snp_recall_mean",
            "cov_recall_mean",
            "spurious_feats_mean",
            "f1_delta_vs_generic",
        ],
    )

    report_md = results_dir / "snp_glm_environment_construction_report.md"
    _write_report(report_md, summary_rows, args)
    return raw_csv, summary_csv, report_md


def _aggregate_summary(raw_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for row in raw_rows:
        key = (int(row["n_snps"]), row["model"], row["arm"])
        grouped.setdefault(key, []).append(row)

    generic_f1: dict[tuple[int, str], float | None] = {}
    for (n_snps, model, arm), rows in grouped.items():
        if arm != "generic":
            continue
        ok = [r for r in rows if r["status"] == "ok" and r["f1_macro"] != ""]
        generic_f1[(n_snps, model)] = _safe_mean([float(r["f1_macro"]) for r in ok])

    summaries: list[dict[str, str]] = []
    for n_snps, model, arm in sorted(grouped.keys()):
        rows = grouped[(n_snps, model, arm)]
        ok = [r for r in rows if r["status"] == "ok"]

        f1_vals = [float(r["f1_macro"]) for r in ok if r["f1_macro"] != ""]
        fit_vals = [float(r["fit_seconds"]) for r in ok if r["fit_seconds"] != ""]
        rules_vals = [float(r["n_rules"]) for r in ok if r["n_rules"] != ""]
        atoms_vals = [float(r["n_atoms"]) for r in ok if r["n_atoms"] != ""]
        snp_vals = [float(r["snp_recall"]) for r in ok if r["snp_recall"] != ""]
        cov_vals = [float(r["cov_recall"]) for r in ok if r["cov_recall"] != ""]
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
                "n_rules_mean": _fmt(_safe_mean(rules_vals)),
                "n_atoms_mean": _fmt(_safe_mean(atoms_vals)),
                "snp_recall_mean": _fmt(_safe_mean(snp_vals)),
                "cov_recall_mean": _fmt(_safe_mean(cov_vals)),
                "spurious_feats_mean": _fmt(_safe_mean(spur_vals)),
                "f1_delta_vs_generic": _fmt(f1_delta),
            }
        )
    return summaries


def _write_report(
    path: Path, summary_rows: list[dict[str, str]], args: argparse.Namespace
) -> None:
    lines: list[str] = []
    lines.append("# SNP-GLM Environment Scenario: Feature-Type-Aware Atom Construction")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- SNP counts: {', '.join(str(x) for x in args.snp_counts)}")
    lines.append(f"- Seeds: {', '.join(str(x) for x in args.seeds)}")
    lines.append(
        f"- n_obs: {args.n_obs}, causal SNPs: {args.n_causal_snp} (dominant), "
        f"covariates: {args.n_cov} ({args.n_causal_cov} causal)"
    )
    lines.append(f"- MAF range: [{args.maf_min}, {args.maf_max}], covariate corr: {args.covariate_corr}")
    lines.append(
        f"- beta0: {args.beta0}, beta (SNP): {args.beta}, beta (covariate): {args.covariate_beta}"
    )
    lines.append("- Arms: generic (hybrid + quantile) vs typed (genotype_aware + supervised MDL)")
    lines.append("")
    lines.append("## Aggregated Results")
    lines.append("")
    lines.append(
        "| n_snps | model | arm | ok | f1_mean | f1_delta_vs_generic | "
        "snp_recall | cov_recall | spurious | n_rules | n_atoms | fit_s |"
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
                    row["f1_delta_vs_generic"],
                    row["snp_recall_mean"],
                    row["cov_recall_mean"],
                    row["spurious_feats_mean"],
                    row["n_rules_mean"],
                    row["n_atoms_mean"],
                    row["fit_seconds_mean"],
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
