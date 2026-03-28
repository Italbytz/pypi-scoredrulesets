# ScoredRuleSets Paper Benchmark - Rule-Based Classifiers Comparison

## Summary

- **datasets**: `3`
- **estimators**: `2`
- **warning_runs**: `0`
- **warning_models**: `0`
- **top_1_model**: `uci_wine / logicGP` (f1=0.9450, rules=7.0000, fit_s=4.2205)

## Configuration

- **datasets**: `sklearn_wine, uci_heart_disease, synth_checkerboard_4x4`
- **estimators**: `logicGP, ruleGP2`
- **repeats**: `1`
- **timeout_seconds**: `120s`
- **design**: `8 Rule-Based Classifiers x 3 Datasets (2 real-world, 1 synthetic), 1 repeats`

## Artifacts

- **raw_csv**: [benchmark_results.csv](benchmark_results.csv)
- **raw_json**: [benchmark_results.json](benchmark_results.json)
- **aggregated_csv**: [benchmark_results_aggregated.csv](benchmark_results_aggregated.csv)
- **aggregated_json**: [benchmark_results_aggregated.json](benchmark_results_aggregated.json)
- **plot_png**: [benchmark_results.png](benchmark_results.png)
- **plot_pdf**: [benchmark_results.pdf](benchmark_results.pdf)
- **heatmap_png**: [benchmark_results_heatmap.png](benchmark_results_heatmap.png)
- **heatmap_pdf**: [benchmark_results_heatmap.pdf](benchmark_results_heatmap.pdf)
- **combined_heatmap_png**: [benchmark_results_heatmap_combined.png](benchmark_results_heatmap_combined.png)
- **combined_heatmap_pdf**: [benchmark_results_heatmap_combined.pdf](benchmark_results_heatmap_combined.pdf)
- **combined_dot_png**: [benchmark_results_combined_dot.png](benchmark_results_combined_dot.png)
- **combined_dot_pdf**: [benchmark_results_combined_dot.pdf](benchmark_results_combined_dot.pdf)
- **pareto_png**: [benchmark_results_pareto.png](benchmark_results_pareto.png)
- **pareto_pdf**: [benchmark_results_pareto.pdf](benchmark_results_pareto.pdf)
- **cd_png**: [benchmark_results_cd.png](benchmark_results_cd.png)
- **cd_pdf**: [benchmark_results_cd.pdf](benchmark_results_cd.pdf)
- **wtl_png**: [benchmark_results_wtl.png](benchmark_results_wtl.png)
- **wtl_pdf**: [benchmark_results_wtl.pdf](benchmark_results_wtl.pdf)
- **wtl_size_png**: [benchmark_results_wtl_size.png](benchmark_results_wtl_size.png)
- **wtl_size_pdf**: [benchmark_results_wtl_size.pdf](benchmark_results_wtl_size.pdf)
- **wtl_pareto_png**: [benchmark_results_wtl_pareto.png](benchmark_results_wtl_pareto.png)
- **wtl_pareto_pdf**: [benchmark_results_wtl_pareto.pdf](benchmark_results_wtl_pareto.pdf)
- **wtl_triangular_png**: [benchmark_results_wtl_triangular.png](benchmark_results_wtl_triangular.png)
- **wtl_triangular_pdf**: [benchmark_results_wtl_triangular.pdf](benchmark_results_wtl_triangular.pdf)
- **efficiency_png**: [benchmark_results_efficiency.png](benchmark_results_efficiency.png)
- **efficiency_pdf**: [benchmark_results_efficiency.pdf](benchmark_results_efficiency.pdf)

## Plot Preview

![Benchmark plot](benchmark_results.png)

_Heatmap add-on: compact overview of aggregated F1 values and fit times per dataset/estimator._

![Benchmark heatmap](benchmark_results_heatmap.png)

## Notes

- Paper Benchmark: 8 rule-based classifiers on 10 selected datasets.
- Real-world datasets: sklearn_breast_cancer, sklearn_wine, uci_car_evaluation, uci_heart_disease.
- Synthetic datasets chosen for concept diversity: DNF rules, overlapping rules, MONK-3 noise, XOR/parity, class imbalance, geometric complexity.
- ExSTraCS (LRC) applies Lossy Rule Compaction post-hoc (interval merge + conservative pruning; 0-6% F1 loss, 29-98% rule reduction).
- Timeout per run: 120s. 1 repeats, random_state=42.

## Top per Dataset

### uci_wine

- **best_model**: `logicGP` (f1=0.9450, rules=7.0000, fit_s=4.2205)
- **smallest_model**: `logicGP` (rules=7.0000, atoms=6.0000, f1=0.9450)
- **fastest_model**: `ruleGP2` (fit_s=3.2879, f1=0.9094, rules=8.0000)

### uci_heart_disease

- **best_model**: `logicGP` (f1=0.7337, rules=8.0000, fit_s=8.0919)
- **smallest_model**: `ruleGP2` (rules=2.0000, atoms=3.0000, f1=0.7100)
- **fastest_model**: `ruleGP2` (fit_s=3.6721, f1=0.7100, rules=2.0000)

### synth_checkerboard_4x4

- **best_model**: `logicGP` (f1=0.6938, rules=3.0000, fit_s=4.1429)
- **smallest_model**: `logicGP` (rules=3.0000, atoms=4.0000, f1=0.6938)
- **fastest_model**: `logicGP` (fit_s=4.1429, f1=0.6938, rules=3.0000)

## Pareto Front (F1 vs Model Size)

### uci_wine

| estimator | f1_mean | atoms | rules |
| --- | ---: | ---: | ---: |
| logicGP | 0.9450 | 6.0000 | 7.0000 |

### uci_heart_disease

| estimator | f1_mean | atoms | rules |
| --- | ---: | ---: | ---: |
| logicGP | 0.7337 | 7.0000 | 8.0000 |
| ruleGP2 | 0.7100 | 3.0000 | 2.0000 |

### synth_checkerboard_4x4

| estimator | f1_mean | atoms | rules |
| --- | ---: | ---: | ---: |
| logicGP | 0.6938 | 4.0000 | 3.0000 |

## Leaderboard

| rank | dataset | estimator | repeats | f1_mean | f1_err | fit_s | rules | atoms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | uci_wine | logicGP | 1 | 0.9450 | 0.0000 | 4.2205 | 7.0000 | 6.0000 |
| 2 | uci_wine | ruleGP2 | 1 | 0.9094 | 0.0000 | 3.2879 | 8.0000 | 8.0000 |
| 3 | uci_heart_disease | logicGP | 1 | 0.7337 | 0.0000 | 8.0919 | 8.0000 | 7.0000 |
| 4 | uci_heart_disease | ruleGP2 | 1 | 0.7100 | 0.0000 | 3.6721 | 2.0000 | 3.0000 |
| 5 | synth_checkerboard_4x4 | logicGP | 1 | 0.6938 | 0.0000 | 4.1429 | 3.0000 | 4.0000 |
| 6 | synth_checkerboard_4x4 | ruleGP2 | 1 | 0.4770 | 0.0000 | 5.9531 | 5.0000 | 4.0000 |

## Dataset: uci_wine

- **best_model**: `logicGP` (f1=0.9450, rules=7.0000, fit_s=4.2205)
- **smallest_model**: `logicGP` (rules=7.0000, atoms=6.0000, f1=0.9450)
- **fastest_model**: `ruleGP2` (fit_s=3.2879, f1=0.9094, rules=8.0000)

| rank | dataset | estimator | repeats | f1_mean | f1_err | fit_s | rules | atoms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | uci_wine | logicGP | 1 | 0.9450 | 0.0000 | 4.2205 | 7.0000 | 6.0000 |
| 2 | uci_wine | ruleGP2 | 1 | 0.9094 | 0.0000 | 3.2879 | 8.0000 | 8.0000 |

## Dataset: uci_heart_disease

- **best_model**: `logicGP` (f1=0.7337, rules=8.0000, fit_s=8.0919)
- **smallest_model**: `ruleGP2` (rules=2.0000, atoms=3.0000, f1=0.7100)
- **fastest_model**: `ruleGP2` (fit_s=3.6721, f1=0.7100, rules=2.0000)

| rank | dataset | estimator | repeats | f1_mean | f1_err | fit_s | rules | atoms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | uci_heart_disease | logicGP | 1 | 0.7337 | 0.0000 | 8.0919 | 8.0000 | 7.0000 |
| 2 | uci_heart_disease | ruleGP2 | 1 | 0.7100 | 0.0000 | 3.6721 | 2.0000 | 3.0000 |

## Dataset: synth_checkerboard_4x4

- **best_model**: `logicGP` (f1=0.6938, rules=3.0000, fit_s=4.1429)
- **smallest_model**: `logicGP` (rules=3.0000, atoms=4.0000, f1=0.6938)
- **fastest_model**: `logicGP` (fit_s=4.1429, f1=0.6938, rules=3.0000)

| rank | dataset | estimator | repeats | f1_mean | f1_err | fit_s | rules | atoms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | synth_checkerboard_4x4 | logicGP | 1 | 0.6938 | 0.0000 | 4.1429 | 3.0000 | 4.0000 |
| 2 | synth_checkerboard_4x4 | ruleGP2 | 1 | 0.4770 | 0.0000 | 5.9531 | 5.0000 | 4.0000 |
