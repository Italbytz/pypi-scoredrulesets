# Bioinformatics Search-Space Reduction Benchmarks

Reproduction scripts for the paper *"Search-Space Reduction for Scored Rule Sets on High-Dimensional Bioinformatics Data: Feature Selection and Feature-Type-Aware Atom Construction"* (compatible with `scoredrulesets >= 0.2.0` and `catgen >= 0.1`).

## Scripts

- `snp_scaling_feature_selection_experiment.py`: Unfiltered vs. univariate mutual-information (`kbest20_mi`) scaling on `make_classification` ($p \le 100{,}000$), DOROTHEA, and Leukemia.
- `snp_glm_main_construction_experiment.py`: Generic (`hybrid`) vs. feature-type-aware (`genotype_aware`) atom construction on the SNP Main Effects track ($p \le 1{,}000$).
- `snp_glm_environment_construction_experiment.py`: Generic vs. feature-type-aware (`genotype_aware` + `supervised_mdl`) atom construction on the Gene--Environment track ($p \le 1{,}000$).
- `snp_glm_extreme_fs_vs_typed_experiment.py`: Extreme-dimensional SNP scaling ($p \le 100{,}000$) combining `kbest20_mi` with generic vs. type-aware atom construction.
- `generate_paper_plots.py`: Generates publication figures from the aggregated CSV summaries.
