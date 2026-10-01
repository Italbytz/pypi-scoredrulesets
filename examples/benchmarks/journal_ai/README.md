# AI Journal Benchmark Suite (MDPI AI 2026)

This directory contains the complete and self-contained reproduction benchmark suite for the journal paper:

> **Scored Rule Sets: A Unified Framework for Interpretable Classification, Regression, and Cluster Explanation**  
> Journal: *AI* (MDPI, 2026)

---

## Overview of Tables & Figures Mapping

| Item in Paper | Description | Script | Primary Output / Result File |
| :--- | :--- | :--- | :--- |
| **Figure 1** | Classification Pareto trade-off (Macro-F1 vs. Atoms) | `generate_all_pareto_figures.py` | `figures/classification_f1_vs_complexity.{pdf,png}` |
| **Figure 2** | Regression Pareto trade-off ($R^2$ vs. Atoms) | `generate_all_pareto_figures.py` | `figures/regression_r2_vs_complexity.{pdf,png}` |
| **Figure 3** | Cluster Explanation Pareto trade-off (Fidelity vs. Atoms) | `generate_all_pareto_figures.py` | `figures/cluster_fidelity_vs_complexity.{pdf,png}` |
| **Figure (CD)** | Demšar Critical-Difference Diagram ($R^2$, Nemenyi CD) | `make_cd_diagram.py` | `figures/regression_cd_diagram.{pdf,png}` |
| **Table 1** | Framework novelty matrix | Conceptual (in text) | Section 1, Table 1 |
| **Table 2** | Benchmark datasets summary | Protocol summary | Section 4, Table 2 |
| **Table 3** | Multiclass classification 5-fold benchmark (9 datasets) | `run_classification_benchmark.py` | `results_classification/classification_5fold_benchmark_results.csv` |
| **Table 4** | EBM classification baseline comparison | `run_ebm_baseline.py` | `results_classification/ebm_classification_results.csv` |
| **Table 5** | Evolutionary generation budget ablation (`ruleGP`) | `run_ablation_search_budget.py` | `results_classification/ablation_search_budget.csv` |
| **Table 6** | Continuous regression 5-fold benchmark ($R^2$, Atoms) | `run_regression_benchmark.py` | `results_regression/regression_benchmark_results.csv` |
| **Table 7** | EBM regression baseline comparison | `run_ebm_baseline.py` | `results_regression/ebm_regression_results.csv` |
| **Table 8** | Multi-target regression (UCI Energy Efficiency) | `run_multioutput_energy_efficiency.py` | `results_regression/multioutput_energy_efficiency.csv` |
| **Table 9** | Multi-seed robustness (5 seeds: 42, 100, 200, 300, 400) | `run_multi_seed_robustness.py` | `results_classification/multiseed_robustness_cls.csv`<br>`results_regression/multiseed_robustness_reg.csv` |
| **Table 10** | Cluster explanation benchmark under $k$-means | `run_cluster_surrogate_benchmark.py` | `results_cluster/cluster_explanation_benchmark_results.csv` |
| **Table 11** | Native & external cluster explanation (GMM, etc.) | `run_native_clustering_benchmark.py`<br>`run_native_clustering_external_benchmark.py` | `results_cluster/native_clustering_benchmark.csv`<br>`results_cluster/native_clustering_external_benchmark.csv` |
| **Tables 12–13** | Subgroup discovery (Diabetes & California Housing) | `run_subgroup_discovery_benchmark.py`<br>`run_subgroup_pysubgroup_baseline.py` | `results_regression/subgroups_rulegp_*.csv`<br>`results_regression/subgroups_nsga2_*.csv` |

---

## Directory Structure

```text
journal_ai/
├── README.md                              # This document
├── figures/                               # Generated publication figures (PDF and PNG)
├── results_classification/                # Pre-computed classification benchmark CSVs
├── results_regression/                    # Pre-computed regression benchmark CSVs
├── results_cluster/                       # Pre-computed cluster explanation benchmark CSVs
├── generate_all_pareto_figures.py         # Regenerates Figures 1, 2, and 3 from CSVs
├── make_cd_diagram.py                     # Regenerates the Critical-Difference diagram
├── make_classification_pareto_plot.py     # Standalone classification Pareto generator
├── run_classification_benchmark.py        # 5-fold CV classification benchmark
├── run_regression_benchmark.py            # 5-fold CV regression benchmark
├── run_cluster_surrogate_benchmark.py     # Cluster explanation benchmark
├── run_ebm_baseline.py                    # Explainable Boosting Machines baselines
├── run_ablation_search_budget.py          # Evolutionary generation budget ablation
├── run_multi_seed_robustness.py           # Multi-seed stability (5 seeds)
├── run_multioutput_energy_efficiency.py   # Multi-target regression benchmark
├── run_multioutput_benchmark.py           # Multi-output regression suite
├── run_native_5fold_classification.py     # Native classification runner
├── run_native_clustering_benchmark.py     # Native clustering evaluation
├── run_native_clustering_external_benchmark.py # External cluster fidelity
├── run_rulefit_benchmark.py               # RuleFit regression baseline
├── run_subgroup_discovery_benchmark.py    # Evolutionary subgroup discovery
├── run_subgroup_pysubgroup_baseline.py    # pysubgroup baseline comparison
├── run_takagi_sugeno_benchmark.py         # Takagi-Sugeno piecewise linear models
└── run_takagi_sugeno_countermeasure.py    # TS regularized countermeasure evaluation
```

---

## Quick Start: Regenerate All Paper Figures

Since pre-computed benchmark results are included in the subdirectories, you can regenerate all publication figures immediately:

```bash
python generate_all_pareto_figures.py
python make_cd_diagram.py
```

Generated files will be saved in `figures/`:
- `figures/classification_f1_vs_complexity.pdf` (Figure 1)
- `figures/regression_r2_vs_complexity.pdf` (Figure 2)
- `figures/cluster_fidelity_vs_complexity.pdf` (Figure 3)
- `figures/regression_cd_diagram.pdf` (Critical Difference diagram)

---

## Re-Running Individual Benchmarks

Each script can be run standalone. It will update the corresponding CSV in `results_*/`:

```bash
# Classification benchmark (Table 3)
python run_classification_benchmark.py

# Regression benchmark (Table 6)
python run_regression_benchmark.py

# Cluster explanation benchmark (Table 10)
python run_cluster_surrogate_benchmark.py

# EBM baselines (Tables 4 & 7)
python run_ebm_baseline.py

# Multi-seed robustness across 5 seeds (Table 9)
python run_multi_seed_robustness.py

# Evolutionary subgroup discovery (Tables 12–13)
python run_subgroup_discovery_benchmark.py
```
