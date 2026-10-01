# 1000 Genomes Data Preparation Tools

Tools to download, filter, and extract cohort-level genotype matrices from the 1000 Genomes Project Phase 3 data for reproducible benchmark studies.

For the theoretical and statistical background, see the protocol in [`docs/1000g_snp_reduction_protocol.md`](../../../docs/1000g_snp_reduction_protocol.md).

## Prerequisites

- `curl` (for download)
- `bcftools` (for VCF parsing and filtering)
- Python 3.10+
- R with `glmnet` (optional, for nested CV SNP selection via `select_optimal_k_nested_cv.R`)

## Workflow

### 1. Download raw VCFs
```bash
./tools/data_prep/1000g/download_1000g_phase3_autosomes.sh --out-dir data/1000g/raw
```

### 2. Build Standard Cohort CSVs
Creates binary classification datasets (GBR vs CEU, IBS vs TSI) with genotypes coded as 1/2/3:
```bash
./tools/data_prep/1000g/build_1000g_standard_cohort_csvs.sh
```

### 3. Select Optimal k via Nested CV (optional)
```bash
Rscript tools/data_prep/1000g/select_optimal_k_nested_cv.R \
  --data data/1000g/gbr_ceu_autosomal_genotypes.csv \
  --outdir runs/1000g_kopt_gbr_ceu
```
