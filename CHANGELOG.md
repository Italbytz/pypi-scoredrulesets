# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] – 2026-10-01

### Added
- **Continuous Regression Estimators**: Native evolutionary regressors `RuleGPRegressor`, `RuleNSGA2Regressor`, `RulePLCSRegressor`, and two-stage `CascadedRuleGPRegressor` (with closed-form ridge score calibration and `cascaded_sum` aggregation).
- **Subgroup Discovery & Pattern Mining**: `RuleNSGA2Subgroup` (with `RuleEvoSubgroup` alias) and `RuleGPSubgroup` for multi-objective exploration of localized exceptional sub-populations.
- **Interpretable Cluster Explanation**: `RuleEvoCluster` for evolutionary surrogate rule extraction approximating cluster assignments.
- **Tree-Guided Ensemble Warmstart Utilities**: `scoredrulesets.warmstart` with multi-estimator extraction, prefix ladders, class-balanced seed selection, and crowding trim controls.
- **AI Journal Benchmark Suite**: Complete, self-contained reproduction suite in `examples/benchmarks/journal_ai/` for classification, regression, cluster explanation, multi-seed robustness, and subgroup experiments.
- **Execution Budget Controls**: Robust `max_fit_seconds` time-budgeting and `FitBudgetExceededError` handling across evolutionary setups.
- **Supervised Discretization & Genotype-Aware Features**: Fayyad-Irani supervised cut-points and compact genetic-model atom construction for ordinal and genomic features.

## [0.1.0] – 2026-07-22

### Added
- `ScoredRuleSetClassifier` wrapper converting scikit-learn rule learners to scored rule sets
- `ScoredRuleSetRegressor` — regression wrapper for scored rule sets
- `ScoredRuleSetClusterer` — cluster-label approximation with interpretable rulesets
- `RuleNSGA2Classifier` — NSGA-II genetic programming over rule populations
- `RuleGPClassifier` — logicGP-style evolution on native atom/rule/rule-set structures
- `RulePLCSClassifier` — Sequential covering with a genetic algorithm (LCS-style)
- `RuleNLNClassifier` — Neural rule extraction via Neural Logic Networks
- `AutoScoredRuleSetClassifier` — automatic estimator selection via cross-validation
- `LogicGPClassifier` — wrapper for logicGP JSON model import
- JSON serialisation format for scored rule sets (`dump_ruleset_json` / `load_ruleset_json`)
- LRC (Local Rule Compaction) post-processing step for rule merging
- Benchmarking utilities: `run_benchmarks`, `benchmark_cluster_approximation`, leaderboard, comparison reports
- Benchmarking dataset registry (UCI, OpenML, multiplexer, synthetic) — generators from `catgen`
- Optional backends: `hs` (imodels), `exstracs`, `rulekit`
- scikit-learn estimator API compliance (fit / predict / score / get_params / set_params)

[Unreleased]: https://github.com/Italbytz/pypi-scoredrulesets/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Italbytz/pypi-scoredrulesets/releases/tag/v0.1.0
