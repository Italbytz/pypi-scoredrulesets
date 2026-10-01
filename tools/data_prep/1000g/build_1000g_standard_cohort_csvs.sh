#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/prepare_1000g_cohort_csv.py"
DATA_DIR="${REPO_ROOT}/data/1000g"

PANEL_PATH="${DATA_DIR}/raw/integrated_call_samples_v3.20130502.ALL.panel"
VCF_PATTERN="${DATA_DIR}/raw/ALL.chr{chr}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"

OUT_GBR_CEU="${DATA_DIR}/gbr_ceu_autosomal_genotypes.csv"
OUT_IBS_TSI="${DATA_DIR}/ibs_tsi_autosomal_genotypes.csv"

CHROMOSOMES="1-22"
MAX_SNPS=25000
MAF_MIN=0.01
MISSING_MAX=0.02
DRY_RUN=0

usage() {
  cat <<'EOF'
Build both standard 1000G cohort CSVs for downstream SNP-k optimization.

Usage:
  tools/data_prep/1000g/build_1000g_standard_cohort_csvs.sh [options]

Options:
  --panel <path>          Panel file path
  --vcf-pattern <pattern> VCF pattern with {chr}
  --chromosomes <spec>    Chromosome selector (default: 1-22)
  --max-snps <n>          Max SNPs per cohort (default: 25000)
  --maf-min <f>           MAF minimum (default: 0.01)
  --missing-max <f>       Missingness maximum (default: 0.02)
  --dry-run               Print commands only
  -h, --help              Show this help

Outputs:
  - data/1000g/gbr_ceu_autosomal_genotypes.csv
  - data/1000g/ibs_tsi_autosomal_genotypes.csv
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --panel)
      PANEL_PATH="$2"
      shift 2
      ;;
    --vcf-pattern)
      VCF_PATTERN="$2"
      shift 2
      ;;
    --chromosomes)
      CHROMOSOMES="$2"
      shift 2
      ;;
    --max-snps)
      MAX_SNPS="$2"
      shift 2
      ;;
    --maf-min)
      MAF_MIN="$2"
      shift 2
      ;;
    --missing-max)
      MISSING_MAX="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
  echo "Missing script: ${PYTHON_SCRIPT}" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found in PATH." >&2
  exit 1
fi

if [[ ! -f "${PANEL_PATH}" ]]; then
  echo "Panel file not found: ${PANEL_PATH}" >&2
  exit 1
fi

cmd_common=(
  python3 "${PYTHON_SCRIPT}"
  --panel "${PANEL_PATH}"
  --vcf-pattern "${VCF_PATTERN}"
  --chromosomes "${CHROMOSOMES}"
  --max-snps "${MAX_SNPS}"
  --maf-min "${MAF_MIN}"
  --missing-max "${MISSING_MAX}"
)

run_cmd() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf 'DRY-RUN: '
    printf '%q ' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

echo "Building cohort CSV: GBR vs CEU"
run_cmd "${cmd_common[@]}" \
  --populations GBR,CEU \
  --out-csv "${OUT_GBR_CEU}"

echo "Building cohort CSV: IBS vs TSI"
run_cmd "${cmd_common[@]}" \
  --populations IBS,TSI \
  --out-csv "${OUT_IBS_TSI}"

echo "Done. Generated files:"
echo "  - ${OUT_GBR_CEU}"
echo "  - ${OUT_IBS_TSI}"