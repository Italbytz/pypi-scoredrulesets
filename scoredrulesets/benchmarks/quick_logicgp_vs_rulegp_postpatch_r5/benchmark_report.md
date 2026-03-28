# ScoredRuleSets Paper Benchmark - Rule-Based Classifiers Comparison

## Summary

- **datasets**: `6`
- **estimators**: `2`
- **warning_runs**: `0`
- **warning_models**: `0`
- **top_1_model**: `synth_dnf_3x2 / ruleGP` (f1=0.9466, rules=4.6000, fit_s=9.4069)

## Configuration

- **datasets**: `sklearn_breast_cancer, sklearn_wine, uci_heart_disease, synth_dnf_3x2, synth_xor_3bit, synth_checkerboard_4x4`
- **estimators**: `logicGP, ruleGP`
- **repeats**: `5`
- **timeout_seconds**: `90s`
- **design**: `8 Rule-Based Classifiers x 6 Datasets (3 real-world, 3 synthetic), 5 repeats`

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
- Timeout per run: 90s. 5 repeats, random_state=42.

## Top per Dataset

### synth_dnf_3x2

- **best_model**: `ruleGP` (f1=0.9466, rules=4.6000, fit_s=9.4069)
- **smallest_model**: `ruleGP` (rules=4.6000, atoms=5.6000, f1=0.9466)
- **fastest_model**: `logicGP` (fit_s=6.2679, f1=0.8848, rules=4.6000)

### uci_wine

- **best_model**: `logicGP` (f1=0.9427, rules=6.8000, fit_s=4.6790)
- **smallest_model**: `ruleGP` (rules=3.8000, atoms=3.2000, f1=0.8590)
- **fastest_model**: `logicGP` (fit_s=4.6790, f1=0.9427, rules=6.8000)

### uci_breast_cancer_wisconsin_diagnostic

- **best_model**: `logicGP` (f1=0.9163, rules=4.6000, fit_s=6.7494)
- **smallest_model**: `ruleGP` (rules=4.0000, atoms=3.8000, f1=0.9105)
- **fastest_model**: `logicGP` (fit_s=6.7494, f1=0.9163, rules=4.6000)

### synth_xor_3bit

- **best_model**: `ruleGP` (f1=0.7870, rules=4.2000, fit_s=13.8815)
- **smallest_model**: `ruleGP` (rules=4.2000, atoms=6.6000, f1=0.7870)
- **fastest_model**: `logicGP` (fit_s=7.4696, f1=0.7013, rules=4.8000)

### uci_heart_disease

- **best_model**: `logicGP` (f1=0.7830, rules=7.8000, fit_s=8.4632)
- **smallest_model**: `ruleGP` (rules=5.6000, atoms=6.6000, f1=0.7038)
- **fastest_model**: `logicGP` (fit_s=8.4632, f1=0.7830, rules=7.8000)

### synth_checkerboard_4x4

- **best_model**: `logicGP` (f1=0.6112, rules=5.0000, fit_s=7.2942)
- **smallest_model**: `logicGP` (rules=5.0000, atoms=5.6000, f1=0.6112)
- **fastest_model**: `logicGP` (fit_s=7.2942, f1=0.6112, rules=5.0000)

## Pareto Front (F1 vs Model Size)

### synth_dnf_3x2

| estimator | f1_mean | atoms | rules |
| --- | ---: | ---: | ---: |
| ruleGP | 0.9466 | 5.6000 | 4.6000 |

### uci_wine

| estimator | f1_mean | atoms | rules |
| --- | ---: | ---: | ---: |
| logicGP | 0.9427 | 6.0000 | 6.8000 |
| ruleGP | 0.8590 | 3.2000 | 3.8000 |

### uci_breast_cancer_wisconsin_diagnostic

| estimator | f1_mean | atoms | rules |
| --- | ---: | ---: | ---: |
| logicGP | 0.9163 | 4.2000 | 4.6000 |
| ruleGP | 0.9105 | 3.8000 | 4.0000 |

### synth_xor_3bit

| estimator | f1_mean | atoms | rules |
| --- | ---: | ---: | ---: |
| ruleGP | 0.7870 | 6.6000 | 4.2000 |
| logicGP | 0.7013 | 6.4000 | 4.8000 |

### uci_heart_disease

| estimator | f1_mean | atoms | rules |
| --- | ---: | ---: | ---: |
| logicGP | 0.7830 | 7.4000 | 7.8000 |
| ruleGP | 0.7038 | 6.6000 | 5.6000 |

### synth_checkerboard_4x4

| estimator | f1_mean | atoms | rules |
| --- | ---: | ---: | ---: |
| logicGP | 0.6112 | 5.6000 | 5.0000 |

## Leaderboard

| rank | dataset | estimator | repeats | f1_mean | f1_err | fit_s | rules | atoms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | synth_dnf_3x2 | ruleGP | 5 | 0.9466 | 0.0734 | 9.4069 | 4.6000 | 5.6000 |
| 2 | uci_wine | logicGP | 5 | 0.9427 | 0.0143 | 4.6790 | 6.8000 | 6.0000 |
| 3 | uci_breast_cancer_wisconsin_diagnostic | logicGP | 5 | 0.9163 | 0.0232 | 6.7494 | 4.6000 | 4.2000 |
| 4 | uci_breast_cancer_wisconsin_diagnostic | ruleGP | 5 | 0.9105 | 0.0078 | 11.9310 | 4.0000 | 3.8000 |
| 5 | synth_dnf_3x2 | logicGP | 5 | 0.8848 | 0.0463 | 6.2679 | 4.6000 | 5.8000 |
| 6 | uci_wine | ruleGP | 5 | 0.8590 | 0.0698 | 9.1981 | 3.8000 | 3.2000 |
| 7 | synth_xor_3bit | ruleGP | 5 | 0.7870 | 0.1373 | 13.8815 | 4.2000 | 6.6000 |
| 8 | uci_heart_disease | logicGP | 5 | 0.7830 | 0.0472 | 8.4632 | 7.8000 | 7.4000 |
| 9 | uci_heart_disease | ruleGP | 5 | 0.7038 | 0.0546 | 14.4492 | 5.6000 | 6.6000 |
| 10 | synth_xor_3bit | logicGP | 5 | 0.7013 | 0.1159 | 7.4696 | 4.8000 | 6.4000 |
| 11 | synth_checkerboard_4x4 | logicGP | 5 | 0.6112 | 0.0791 | 7.2942 | 5.0000 | 5.6000 |
| 12 | synth_checkerboard_4x4 | ruleGP | 5 | 0.5242 | 0.0181 | 39.0221 | 10.2000 | 17.2000 |

## Dataset: synth_dnf_3x2

- **best_model**: `ruleGP` (f1=0.9466, rules=4.6000, fit_s=9.4069)
- **smallest_model**: `ruleGP` (rules=4.6000, atoms=5.6000, f1=0.9466)
- **fastest_model**: `logicGP` (fit_s=6.2679, f1=0.8848, rules=4.6000)

| rank | dataset | estimator | repeats | f1_mean | f1_err | fit_s | rules | atoms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | synth_dnf_3x2 | ruleGP | 5 | 0.9466 | 0.0734 | 9.4069 | 4.6000 | 5.6000 |
| 2 | synth_dnf_3x2 | logicGP | 5 | 0.8848 | 0.0463 | 6.2679 | 4.6000 | 5.8000 |

## Dataset: uci_wine

- **best_model**: `logicGP` (f1=0.9427, rules=6.8000, fit_s=4.6790)
- **smallest_model**: `ruleGP` (rules=3.8000, atoms=3.2000, f1=0.8590)
- **fastest_model**: `logicGP` (fit_s=4.6790, f1=0.9427, rules=6.8000)

| rank | dataset | estimator | repeats | f1_mean | f1_err | fit_s | rules | atoms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | uci_wine | logicGP | 5 | 0.9427 | 0.0143 | 4.6790 | 6.8000 | 6.0000 |
| 2 | uci_wine | ruleGP | 5 | 0.8590 | 0.0698 | 9.1981 | 3.8000 | 3.2000 |

## Dataset: uci_breast_cancer_wisconsin_diagnostic

- **best_model**: `logicGP` (f1=0.9163, rules=4.6000, fit_s=6.7494)
- **smallest_model**: `ruleGP` (rules=4.0000, atoms=3.8000, f1=0.9105)
- **fastest_model**: `logicGP` (fit_s=6.7494, f1=0.9163, rules=4.6000)

| rank | dataset | estimator | repeats | f1_mean | f1_err | fit_s | rules | atoms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | uci_breast_cancer_wisconsin_diagnostic | logicGP | 5 | 0.9163 | 0.0232 | 6.7494 | 4.6000 | 4.2000 |
| 2 | uci_breast_cancer_wisconsin_diagnostic | ruleGP | 5 | 0.9105 | 0.0078 | 11.9310 | 4.0000 | 3.8000 |

## Dataset: synth_xor_3bit

- **best_model**: `ruleGP` (f1=0.7870, rules=4.2000, fit_s=13.8815)
- **smallest_model**: `ruleGP` (rules=4.2000, atoms=6.6000, f1=0.7870)
- **fastest_model**: `logicGP` (fit_s=7.4696, f1=0.7013, rules=4.8000)

| rank | dataset | estimator | repeats | f1_mean | f1_err | fit_s | rules | atoms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | synth_xor_3bit | ruleGP | 5 | 0.7870 | 0.1373 | 13.8815 | 4.2000 | 6.6000 |
| 2 | synth_xor_3bit | logicGP | 5 | 0.7013 | 0.1159 | 7.4696 | 4.8000 | 6.4000 |

## Dataset: uci_heart_disease

- **best_model**: `logicGP` (f1=0.7830, rules=7.8000, fit_s=8.4632)
- **smallest_model**: `ruleGP` (rules=5.6000, atoms=6.6000, f1=0.7038)
- **fastest_model**: `logicGP` (fit_s=8.4632, f1=0.7830, rules=7.8000)

| rank | dataset | estimator | repeats | f1_mean | f1_err | fit_s | rules | atoms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | uci_heart_disease | logicGP | 5 | 0.7830 | 0.0472 | 8.4632 | 7.8000 | 7.4000 |
| 2 | uci_heart_disease | ruleGP | 5 | 0.7038 | 0.0546 | 14.4492 | 5.6000 | 6.6000 |

## Dataset: synth_checkerboard_4x4

- **best_model**: `logicGP` (f1=0.6112, rules=5.0000, fit_s=7.2942)
- **smallest_model**: `logicGP` (rules=5.0000, atoms=5.6000, f1=0.6112)
- **fastest_model**: `logicGP` (fit_s=7.2942, f1=0.6112, rules=5.0000)

| rank | dataset | estimator | repeats | f1_mean | f1_err | fit_s | rules | atoms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | synth_checkerboard_4x4 | logicGP | 5 | 0.6112 | 0.0791 | 7.2942 | 5.0000 | 5.6000 |
| 2 | synth_checkerboard_4x4 | ruleGP | 5 | 0.5242 | 0.0181 | 39.0221 | 10.2000 | 17.2000 |
