# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] – 2025-01-01

### Added
- `ScoredRuleSetClassifier` wrapper converting scikit-learn rule learners to scored rule sets
- `RuleNSGA2Classifier` — NSGA-II genetic programming over rule populations
- `RulePLCSClassifier` — Sequential covering with a genetic algorithm (LCS-style)
- `RuleNLNClassifier` — Neural rule extraction via Neural Logic Networks
- `AutoScoredRuleSetClassifier` — automatic estimator selection via cross-validation
- `LogicGPClassifier` — wrapper for logicGP JSON model import
- JSON serialisation format for scored rule sets (`dump_ruleset_json` / `load_ruleset_json`)
- LRC (Local Rule Compaction) post-processing step for rule merging
- Benchmarking utilities: `run_benchmarks`, leaderboard, comparison reports
- Synthetic dataset generators (checkerboard, MONK, MUX, …)
- Optional backends: `hs` (imodels), `exstracs`, `rulekit`
- scikit-learn estimator API compliance (fit / predict / score / get_params / set_params)

[Unreleased]: https://github.com/scoredrulesets/scoredrulesets/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/scoredrulesets/scoredrulesets/releases/tag/v0.1.0
