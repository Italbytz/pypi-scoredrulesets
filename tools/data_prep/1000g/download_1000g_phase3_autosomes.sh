#!/usr/bin/env bash
set -euo pipefail

BASE_URL="https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502"
OUT_DIR="data/1000g/raw"
START_CHR=1
END_CHR=22
WITH_INDEX=1
DRY_RUN=0

usage() {
  cat <<'EOF'
Download 1000 Genomes Phase 3 autosomal VCFs (chr1-22) and optional .tbi indices.

Usage:
  tools/data_prep/1000g/download_1000g_phase3_autosomes.sh [options]

Options:
  --out-dir <path>     Output directory (default: data/1000g/raw)
  --start-chr <n>      Start chromosome (default: 1)
  --end-chr <n>        End chromosome (default: 22)
  --no-index           Do not download .tbi index files
  --dry-run            Print commands without downloading
  -h, --help           Show this help

Notes:
  - Uses curl with resume support (-C -) and retries.
  - Existing partial files are continued, complete files are reused.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --start-chr)
      START_CHR="$2"
      shift 2
      ;;
    --end-chr)
      END_CHR="$2"
      shift 2
      ;;
    --no-index)
      WITH_INDEX=0
      shift
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

if ! command -v curl >/dev/null 2>&1; then
  echo "curl not found in PATH." >&2
  exit 1
fi

if [[ "$START_CHR" -lt 1 || "$END_CHR" -gt 22 || "$START_CHR" -gt "$END_CHR" ]]; then
  echo "Invalid chromosome range: ${START_CHR}-${END_CHR} (allowed 1-22)." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

download_file() {
  local url="$1"
  local target="$2"
  local cmd=(
    curl
    --fail
    --location
    --continue-at -
    --retry 5
    --retry-delay 3
    --retry-connrefused
    --output "$target"
    "$url"
  )

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'DRY-RUN: '
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  "${cmd[@]}"
}

for chr in $(seq "$START_CHR" "$END_CHR"); do
  file="ALL.chr${chr}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
  url="${BASE_URL}/${file}"
  target="${OUT_DIR}/${file}"

  echo "Downloading chr${chr} VCF..."
  download_file "$url" "$target"

  if [[ "$WITH_INDEX" -eq 1 ]]; then
    echo "Downloading chr${chr} index..."
    download_file "${url}.tbi" "${target}.tbi"
  fi
done

echo "Done. Files are in: ${OUT_DIR}"