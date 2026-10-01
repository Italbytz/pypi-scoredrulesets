# Reproducible SNP Reduction Protocol (1000 Genomes, publication-ready)

This protocol defines a leakage-safe and reproducible procedure to derive an optimally sized SNP panel (`k`) from real open data for binary classification tasks.

## Why this protocol

- Uses real, openly accessible genotype data (no simulation).
- Separates cohort construction from model selection.
- Uses nested cross-validation for unbiased model selection.
- Adds stability selection to report robust markers.
- Produces versioned artifacts that can be reused in future papers.

## Data source

Preferred source:
- IGSR / 1000 Genomes data portal: https://www.internationalgenome.org/data
- Open mirror: https://registry.opendata.aws/1000-genomes/

Recommended non-trivial binary tasks:
- `GBR` vs `CEU`
- `IBS` vs `TSI`

These are intentionally harder than continental contrasts.

## Cohort construction (outside model selection)

1. Select two populations and define labels (`0`, `1`).
2. Restrict to autosomal biallelic SNPs.
3. Keep sample manifest (sample IDs, population labels, source files, timestamps).
4. Freeze one analysis dataset version before any feature selection.

## Statistical protocol

All thresholds below must be pre-registered per experiment and logged in artifacts.

1. Outer stratified CV split (default: 5 folds)
- Outer test folds are never used for feature selection or `k` tuning.

2. Inner stratified CV split on outer-train (default: 5 folds)
- Used to choose `k` from a predefined grid.

3. Training-only filtering per fold
- SNP missingness filter (default <= 2%).
- MAF filter (default >= 1%).
- Optional additional technical filters handled upstream (e.g., PLINK QC).

4. Training-only feature ranking
- Univariate chi-square ranking over genotype states (`0/1/2`) versus class.

5. Candidate `k` grid
- Example: `16,32,48,63,96,128,192,256`.

6. Model evaluation per `k`
- Train ridge logistic model (`glmnet`, `alpha = 0`) on selected SNPs.
- Inner score: balanced accuracy.

7. Select optimal `k`
- Choose `k` with best mean inner balanced accuracy.
- Tie-breaker: choose smaller `k`.

8. Outer performance estimation
- Refit on outer-train with selected `k`, evaluate once on outer-test.
- Report mean and SD across outer folds.

9. Stability selection (post-hoc robustness)
- Bootstrap stratified resampling (default: 100 runs).
- Re-run training-only ranking and select top `k_opt` SNPs.
- Keep SNPs with selection frequency >= threshold (default: 0.60).

## Reproducibility requirements

For each run, persist:
- Data manifest (source URLs, cohort definition, sample IDs).
- Full parameter set (`seed`, folds, thresholds, `k` grid).
- Inner and outer CV results.
- Final `k_opt` decision with tie-break rationale.
- Stable SNP list with selection frequencies.
- Software versions (`R`, package versions).

## Script in this repository

Use:
- `scripts/r/select_optimal_k_nested_cv.R`
- `scripts/prepare_1000g_cohort_csv.py` (to build the required cohort CSVs from raw 1000G VCFs)

Expected input CSV format:
- Semicolon-separated (`read.csv2` compatible)
- Column 1: binary label (`0`/`1`)
- Columns 2..p: SNP genotype encoded `1/2/3` (internally recoded to `0/1/2`)

Preparation command pattern:

```bash
cd artifacts/consumers/production/paper-code
python3 scripts/prepare_1000g_cohort_csv.py \
  --panel reference/open-data/1000g/raw/integrated_call_samples_v3.20130502.ALL.panel \
  --vcf-pattern "reference/open-data/1000g/raw/ALL.chr{chr}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz" \
  --populations GBR,CEU \
  --out-csv reference/open-data/1000g/gbr_ceu_autosomal_genotypes.csv \
  --chromosomes 1-22 \
  --max-snps 25000 \
  --maf-min 0.01 \
  --missing-max 0.02
```

Dependency:

- `bcftools`

## Example command

```bash
cd artifacts/consumers/production/paper-code
Rscript scripts/r/select_optimal_k_nested_cv.R \
  --data reference/nunkesser-2007-gpas/data/genica/genica63.csv \
  --outdir runs/snp_k_study \
  --seed 20260425 \
  --outer-folds 5 \
  --inner-folds 5 \
  --k-grid 16,32,48,63 \
  --stability-runs 100 \
  --stability-threshold 0.60 \
  --maf-min 0.01 \
  --missing-max 0.02
```

## Notes for papers

Recommended reporting block:
- Cohort definition and class balance.
- Frozen pre-processing rules.
- Nested CV design and metric.
- `k` grid and tie-break policy.
- Final `k_opt` and confidence intervals over outer folds.
- Stability threshold and number of stable SNPs.
- Link to artifact folder and exact script commit.
