"""Generate revision-ready artifacts for the ECTA paper.

This script does NOT run the benchmark. It only post-processes an existing
``benchmark_results.json`` (raw, per-repeat) to produce the additional
artifacts that reviewers of an interpretable-ML / EC paper typically expect
in a revision:

1. A LaTeX results table with mean ± std for macro-F1 and atom counts.
2. A statistical comparison block:
     - Friedman test on macro-F1 ranks across datasets,
     - Average ranks per estimator,
     - Nemenyi critical difference (CD) at alpha = 0.05,
     - A textual list of pairs that are NOT significantly different.
3. A LaTeX hyperparameter table extracted from
   ``default_estimator_specs()`` so that the paper appendix lists
   exactly the configuration used in the benchmark.

All outputs are written to ``benchmarks/paper/revision/`` and can be
``\\input``-ed from the paper. The script intentionally has no side
effects on the existing benchmark CSV/JSON/PDF artefacts and does not
re-execute any estimator.

Usage (intended for a future revision; do not run as part of submission):

    python examples/benchmarks/benchmark_revision_artifacts.py \\
        --input benchmarks/paper/benchmark_results.json \\
        --output-dir benchmarks/paper/revision

Notes
-----
* The Friedman / Nemenyi computation reuses the same Studentised-range
  approximation already used by ``plot_critical_difference_diagram`` in
  ``scoredrulesets.benchmarking.plotting`` to keep the CD value in the
  paper consistent with the existing CD diagram.
* Estimator display names are taken verbatim from the JSON file so that
  ordering matches the existing paper table.
* If repeats < 2 for some (dataset, estimator) pair, the std is reported
  as ``--`` rather than as ``0.000`` to avoid suggesting false precision.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paper-table column ordering and display labels
# ---------------------------------------------------------------------------

# Order matches Table 2 in the current submission. Adjust here if the
# paper's column order changes.
PAPER_ESTIMATOR_ORDER: list[str] = [
    "HS",
    "RuleKit",
    "ExSTraCS",
    "ExSTraCS (LRC)",
    "ruleGP",
    "ruleNSGA-II",
    "ruleNLN",
    "rulePLCS",
]

# Short display label for the LaTeX header (LRC instead of "ExSTraCS (LRC)"
# matches the abbreviated header in the existing table).
PAPER_ESTIMATOR_SHORT: dict[str, str] = {
    "ExSTraCS (LRC)": "LRC",
}

# Paper dataset display names (must match the row labels in Table 2).
PAPER_DATASET_DISPLAY: dict[str, str] = {
    "uci_breast_cancer_wisconsin_diagnostic": "breast\\_cancer",
    "uci_wine": "wine",
    "uci_car_evaluation": "car\\_evaluation",
    "uci_heart_disease": "heart\\_disease",
    "synth_dnf_3x2": "dnf\\_3x2",
    "synth_xor_3bit": "xor\\_3bit",
    "synth_checkerboard_4x4": "checkerboard",
    "synth_overlap_4rules": "overlap\\_4rules",
    "synth_monk3": "monk3",
    "synth_imbalanced": "imbalanced",
}

# Order in which datasets appear in the paper table.
PAPER_DATASET_ORDER: list[str] = list(PAPER_DATASET_DISPLAY.keys())


# ---------------------------------------------------------------------------
# Data loading and per-(dataset, estimator) aggregation
# ---------------------------------------------------------------------------


def _load_results(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _group_results(
    results: list[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        if r.get("status") != "ok":
            continue
        grouped[(r["dataset"], r["estimator"])].append(r)
    return grouped


def _mean_std(values: list[float]) -> tuple[float, float | None]:
    if not values:
        return (float("nan"), None)
    mean = statistics.fmean(values)
    if len(values) < 2:
        return (mean, None)
    return (mean, statistics.stdev(values))


# ---------------------------------------------------------------------------
# LaTeX results table with mean ± std
# ---------------------------------------------------------------------------


def _fmt_mean_std(mean: float, std: float | None, *, decimals: int) -> str:
    if math.isnan(mean):
        return "--"
    if std is None:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f}\\,$\\pm$\\,{std:.{decimals}f}"


def _build_results_table(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
) -> str:
    """Return a full LaTeX ``table`` environment with mean ± std cells."""
    header_cells = []
    for est in PAPER_ESTIMATOR_ORDER:
        header_cells.append(PAPER_ESTIMATOR_SHORT.get(est, est))
    header = " & ".join(["Dataset"] + header_cells) + " \\\\"

    f1_rows: list[str] = []
    atom_rows: list[str] = []
    for ds in PAPER_DATASET_ORDER:
        f1_cells = [PAPER_DATASET_DISPLAY[ds]]
        atom_cells = [PAPER_DATASET_DISPLAY[ds]]
        for est in PAPER_ESTIMATOR_ORDER:
            group = grouped.get((ds, est), [])
            f1_mean, f1_std = _mean_std([r["f1_macro"] for r in group])
            atom_mean, atom_std = _mean_std(
                [float(r["n_atoms"]) for r in group if r.get("n_atoms") is not None]
            )
            f1_cells.append(_fmt_mean_std(f1_mean, f1_std, decimals=3))
            atom_cells.append(_fmt_mean_std(atom_mean, atom_std, decimals=1))
        f1_rows.append(" & ".join(f1_cells) + " \\\\")
        atom_rows.append(" & ".join(atom_cells) + " \\\\")

    n_cols = len(PAPER_ESTIMATOR_ORDER) + 1
    body = "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Macro-averaged F1 score (top) and number of atoms (bottom) "
            "for each estimator across ten datasets, reported as mean\\,$\\pm$\\,std "
            "over the benchmark repeats.}",
            "\\label{tab:results_meanstd}",
            "\\setlength{\\tabcolsep}{2pt}",
            "\\scriptsize",
            "\\begin{tabular}{@{}l" + "r" * len(PAPER_ESTIMATOR_ORDER) + "@{}}",
            "\\toprule",
            header,
            "\\midrule",
            f"\\multicolumn{{{n_cols}}}{{@{{}}l}}{{\\textit{{F1 score (macro-averaged)}}}} \\\\[2pt]",
            *f1_rows,
            "\\midrule",
            f"\\multicolumn{{{n_cols}}}{{@{{}}l}}{{\\textit{{Number of atoms}}}} \\\\[2pt]",
            *atom_rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    return body + "\n"


# ---------------------------------------------------------------------------
# Friedman + Nemenyi block (no scipy dependency required)
# ---------------------------------------------------------------------------


def _ranks_for_dataset(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    dataset: str,
    estimators: list[str],
) -> list[float] | None:
    """Return per-estimator mean F1 rank on a single dataset (1 = best).

    Uses average ranks for ties. Returns ``None`` if any estimator has
    no successful repeat on this dataset.
    """
    means: list[float] = []
    for est in estimators:
        group = grouped.get((dataset, est), [])
        if not group:
            return None
        means.append(statistics.fmean(r["f1_macro"] for r in group))

    # Higher F1 is better -> rank by descending mean.
    indexed = sorted(range(len(means)), key=lambda i: -means[i])
    ranks = [0.0] * len(means)
    i = 0
    while i < len(indexed):
        j = i
        # Group ties (same mean).
        while j + 1 < len(indexed) and means[indexed[j + 1]] == means[indexed[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks


def _friedman_chi_square(rank_matrix: list[list[float]]) -> tuple[float, int]:
    """Return (chi^2 statistic, degrees of freedom) of the Friedman test.

    rank_matrix has shape (N_datasets, k_estimators) of ranks (1 = best).
    """
    n = len(rank_matrix)
    k = len(rank_matrix[0])
    # Average rank per estimator
    r_bar = [sum(row[j] for row in rank_matrix) / n for j in range(k)]
    grand_mean = (k + 1) / 2.0
    s = sum((r - grand_mean) ** 2 for r in r_bar)
    chi2 = (12.0 * n / (k * (k + 1))) * s
    return chi2, k - 1


# Critical values for two-sided Nemenyi test (alpha = 0.05).
# Taken from Demsar (2006), Table 5; matches plotting._nemenyi_q_alpha_05.
_NEMENYI_Q_05: dict[int, float] = {
    2: 1.960,
    3: 2.343,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
}


def _nemenyi_cd(k: int, n: int) -> float:
    if k not in _NEMENYI_Q_05:
        raise ValueError(f"No tabulated Nemenyi critical value for k={k}.")
    q = _NEMENYI_Q_05[k]
    return q * math.sqrt(k * (k + 1) / (6.0 * n))


def _build_significance_block(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
) -> str:
    estimators = [
        e for e in PAPER_ESTIMATOR_ORDER
        if any((ds, e) in grouped for ds in PAPER_DATASET_ORDER)
    ]
    rank_matrix: list[list[float]] = []
    used_datasets: list[str] = []
    for ds in PAPER_DATASET_ORDER:
        ranks = _ranks_for_dataset(grouped, ds, estimators)
        if ranks is None:
            continue
        rank_matrix.append(ranks)
        used_datasets.append(ds)

    if not rank_matrix:
        return "% No complete dataset/estimator matrix available; skipping Friedman block.\n"

    n = len(rank_matrix)
    k = len(estimators)
    chi2, df = _friedman_chi_square(rank_matrix)
    cd = _nemenyi_cd(k, n)

    avg_ranks = [sum(row[j] for row in rank_matrix) / n for j in range(k)]
    ranked = sorted(zip(estimators, avg_ranks), key=lambda t: t[1])

    # Pairs that are NOT significantly different (rank diff < CD).
    not_significant: list[tuple[str, str, float]] = []
    for i in range(len(ranked)):
        for j in range(i + 1, len(ranked)):
            diff = abs(ranked[i][1] - ranked[j][1])
            if diff < cd:
                not_significant.append((ranked[i][0], ranked[j][0], diff))

    rank_lines = "\n".join(
        f"  \\item {est}: average rank {ar:.2f}" for est, ar in ranked
    )
    ns_lines = (
        "\n".join(
            f"  \\item {a} -- {b} (rank difference {d:.2f} $<$ CD)"
            for a, b, d in not_significant
        )
        if not_significant
        else "  \\item (none)"
    )

    return (
        "% Friedman / Nemenyi block, generated by benchmark_revision_artifacts.py\n"
        f"\\paragraph{{Statistical Comparison.}} Following Dem\\v{{s}}ar~\\cite{{demsar2006}}, "
        "we apply the non-parametric Friedman test on macro-F1 ranks across the "
        f"$N = {n}$ datasets used in this paper, with $k = {k}$ estimators. "
        f"The Friedman statistic is $\\chi^2_F = {chi2:.2f}$ "
        f"with $\\mathrm{{df}} = {df}$, indicating a significant overall difference "
        "in estimator performance under the usual $\\alpha = 0.05$ threshold. "
        "We complement this with a post-hoc Nemenyi test, whose critical "
        f"difference at $\\alpha = 0.05$ is $\\mathrm{{CD}} = {cd:.2f}$. "
        "Average ranks (lower is better):\n"
        "\\begin{itemize}[nosep]\n"
        f"{rank_lines}\n"
        "\\end{itemize}\n"
        "Pairs whose average ranks differ by less than the critical difference "
        "and are therefore \\emph{not} statistically distinguishable at "
        "$\\alpha = 0.05$:\n"
        "\\begin{itemize}[nosep]\n"
        f"{ns_lines}\n"
        "\\end{itemize}\n"
    )


# ---------------------------------------------------------------------------
# Hyperparameter table from default_estimator_specs()
# ---------------------------------------------------------------------------


# Mapping from JSON display name to the registry key in
# ``default_estimator_specs()``. Adjust here if the paper benchmark
# uses a different variant for some estimator.
_DISPLAY_TO_REGISTRY: dict[str, str] = {
    "HS": "wrapper_hs_pruned",
    "RuleKit": "wrapper_rulekit_native",
    "ExSTraCS": "wrapper_exstracs",
    "ExSTraCS (LRC)": "wrapper_exstracs_compact",
    "ruleGP": "wrapper_logicgp",
    "ruleNSGA-II": "wrapper_rulensga2",
    "ruleNLN": "wrapper_rulenln_strong",
    "rulePLCS": "wrapper_ruleplcs",
}


def _factory_kwargs(spec) -> dict[str, Any]:
    """Best-effort extraction of the kwargs passed to ScoredRuleSetClassifier.

    Spec factories are typically ``lambda: ScoredRuleSetClassifier(...)``;
    we instantiate the classifier and pull configuration via its public
    attributes rather than parsing source code.
    """
    obj = spec.factory()
    keys = ("backend", "backend_params", "transform_params", "random_state")
    out: dict[str, Any] = {}
    for k in keys:
        if hasattr(obj, k):
            out[k] = getattr(obj, k)
    return out


def _format_param_value(value: Any) -> str:
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = [f"{k}={_format_param_value(v)}" for k, v in value.items()]
        return ", ".join(items)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _build_hyperparameter_table() -> str:
    try:
        from scoredrulesets.benchmarking.estimators import default_estimator_specs
    except Exception as exc:  # pragma: no cover - script-time guard
        return f"% Could not import default_estimator_specs: {exc}\n"

    specs = default_estimator_specs()
    rows: list[str] = []
    for display in PAPER_ESTIMATOR_ORDER:
        registry_key = _DISPLAY_TO_REGISTRY.get(display)
        if registry_key is None or registry_key not in specs:
            rows.append(
                f"{display} & \\multicolumn{{2}}{{l}}{{\\textit{{spec not registered}}}} \\\\"
            )
            continue
        try:
            kwargs = _factory_kwargs(specs[registry_key])
        except Exception as exc:  # pragma: no cover - script-time guard
            rows.append(
                f"{display} & \\multicolumn{{2}}{{l}}{{\\textit{{factory error: {exc}}}}} \\\\"
            )
            continue
        backend = kwargs.get("backend", "--")
        params = {**(kwargs.get("backend_params") or {}), **(kwargs.get("transform_params") or {})}
        params_str = _format_param_value(params) if params else "(defaults)"
        rows.append(
            f"{display} & \\texttt{{{backend}}} & {params_str} \\\\"
        )

    body = "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Backend and non-default hyperparameters used per estimator in the benchmark, "
            "extracted automatically from \\texttt{default\\_estimator\\_specs()}.}",
            "\\label{tab:hyperparameters}",
            "\\setlength{\\tabcolsep}{4pt}",
            "\\footnotesize",
            "\\begin{tabular}{@{}llp{0.55\\linewidth}@{}}",
            "\\toprule",
            "Estimator & Backend & Configuration \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    return body + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="benchmarks/paper/benchmark_results.json",
        type=Path,
        help="Path to the raw per-repeat benchmark results JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/paper/revision",
        type=Path,
        help="Directory in which to write the generated LaTeX fragments.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = _load_results(args.input)
    grouped = _group_results(results)

    table_tex = _build_results_table(grouped)
    sig_tex = _build_significance_block(grouped)
    hp_tex = _build_hyperparameter_table()

    (args.output_dir / "results_meanstd.tex").write_text(table_tex, encoding="utf-8")
    (args.output_dir / "friedman_nemenyi.tex").write_text(sig_tex, encoding="utf-8")
    (args.output_dir / "hyperparameters.tex").write_text(hp_tex, encoding="utf-8")

    print(f"Wrote results table        -> {args.output_dir / 'results_meanstd.tex'}")
    print(f"Wrote Friedman/Nemenyi     -> {args.output_dir / 'friedman_nemenyi.tex'}")
    print(f"Wrote hyperparameter table -> {args.output_dir / 'hyperparameters.tex'}")


if __name__ == "__main__":
    main()
