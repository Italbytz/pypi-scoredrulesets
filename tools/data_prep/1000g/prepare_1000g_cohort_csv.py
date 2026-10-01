#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    population: str
    label: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Builds a semicolon CSV for 1000G cohort studies (label in col1, "
            "genotypes coded 1/2/3 in columns 2..p)."
        )
    )
    parser.add_argument(
        "--panel",
        required=True,
        help="Path to 1000G sample panel TSV (must contain sample and population columns).",
    )
    parser.add_argument(
        "--vcf-pattern",
        required=True,
        help=(
            "Path pattern for chromosome VCFs, e.g. "
            "raw/ALL.chr{chr}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
        ),
    )
    parser.add_argument(
        "--populations",
        required=True,
        help="Two populations as CSV, e.g. GBR,CEU",
    )
    parser.add_argument(
        "--label-map",
        default="",
        help="Optional explicit map like GBR:0,CEU:1. Defaults to first pop=0, second pop=1.",
    )
    parser.add_argument("--out-csv", required=True, help="Output CSV path (semicolon-separated).")
    parser.add_argument(
        "--manifest",
        default="",
        help="Optional output JSON manifest path. Defaults to <out-csv>.manifest.json",
    )
    parser.add_argument(
        "--sample-manifest",
        default="",
        help="Optional output sample manifest TSV path. Defaults to <out-csv>.samples.tsv",
    )
    parser.add_argument(
        "--chromosomes",
        default="1-22",
        help="Chromosome selection, e.g. 1-22 or 1,2,3,10",
    )
    parser.add_argument(
        "--max-snps",
        type=int,
        default=25000,
        help="Maximum number of SNP columns to keep after filtering (0 means unlimited).",
    )
    parser.add_argument("--maf-min", type=float, default=0.01, help="Minimum MAF per SNP.")
    parser.add_argument("--missing-max", type=float, default=0.02, help="Maximum missingness per SNP.")
    parser.add_argument("--bcftools", default="bcftools", help="bcftools executable name/path.")
    return parser


def parse_populations(populations_arg: str) -> list[str]:
    populations = [part.strip().upper() for part in populations_arg.split(",") if part.strip()]
    if len(populations) != 2:
        raise ValueError("--populations must define exactly two populations, e.g. GBR,CEU")
    if populations[0] == populations[1]:
        raise ValueError("--populations must contain two distinct populations")
    return populations


def parse_label_map(label_map_arg: str, populations: list[str]) -> dict[str, int]:
    if not label_map_arg.strip():
        return {populations[0]: 0, populations[1]: 1}

    mapping: dict[str, int] = {}
    for pair in label_map_arg.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError("--label-map must be formatted as POP:LABEL,POP:LABEL")
        pop, label = pair.split(":", 1)
        pop = pop.strip().upper()
        label = int(label.strip())
        mapping[pop] = label

    missing = [pop for pop in populations if pop not in mapping]
    if missing:
        raise ValueError(f"--label-map missing populations: {', '.join(missing)}")
    return mapping


def parse_chromosomes(chromosomes_arg: str) -> list[str]:
    tokens = [part.strip() for part in chromosomes_arg.split(",") if part.strip()]
    chromosomes: list[str] = []
    for token in tokens:
        if "-" in token:
            left, right = token.split("-", 1)
            start = int(left)
            end = int(right)
            if start > end:
                raise ValueError("Invalid chromosome range in --chromosomes")
            chromosomes.extend([str(value) for value in range(start, end + 1)])
        else:
            chromosomes.append(token)
    if not chromosomes:
        raise ValueError("--chromosomes resolved to an empty set")
    return chromosomes


def load_sample_records(panel_path: Path, populations: list[str], labels: dict[str, int]) -> list[SampleRecord]:
    if not panel_path.exists():
        raise FileNotFoundError(f"Panel file not found: {panel_path}")

    records: list[SampleRecord] = []
    with panel_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for index, row in enumerate(reader):
            if not row:
                continue
            if index == 0 and row[0].strip().lower() in {"sample", "sample_id"}:
                continue
            if len(row) < 2:
                continue
            sample_id = row[0].strip()
            population = row[1].strip().upper()
            if not sample_id or population not in labels:
                continue
            records.append(SampleRecord(sample_id=sample_id, population=population, label=labels[population]))

    if not records:
        raise RuntimeError("No samples selected from panel. Check --panel and --populations.")

    records.sort(key=lambda item: (populations.index(item.population), item.sample_id))
    return records


def genotype_to_dosage(genotype: str) -> int | None:
    value = genotype.strip()
    if value in {".", "./.", ".|."}:
        return None
    normalized = value.replace("/", "|")
    alleles = normalized.split("|")
    if len(alleles) != 2:
        return None
    if any(allele == "." for allele in alleles):
        return None
    if any(allele not in {"0", "1"} for allele in alleles):
        return None
    return int(alleles[0]) + int(alleles[1])


def variant_passes(dosages: list[int | None], maf_min: float, missing_max: float) -> bool:
    total = len(dosages)
    non_missing = [value for value in dosages if value is not None]
    missing_fraction = 1.0 - (len(non_missing) / float(total))
    if missing_fraction > missing_max:
        return False
    if not non_missing:
        return False
    allele_frequency = sum(non_missing) / float(2 * len(non_missing))
    maf = min(allele_frequency, 1.0 - allele_frequency)
    return maf >= maf_min


def ensure_tool(executable: str) -> None:
    if shutil.which(executable) is None:
        raise RuntimeError(
            f"Required tool not found: {executable}. Install bcftools and retry."
        )


def query_vcf(
    bcftools_exe: str,
    vcf_path: Path,
    sample_file: Path,
    stderr_path: Path,
) -> tuple[subprocess.Popen[str], TextIO]:
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_handle = stderr_path.open("a", encoding="utf-8")
    command = [
        bcftools_exe,
        "query",
        "-S",
        str(sample_file),
        "-i",
        'TYPE="snp" && N_ALT=1',
        "-f",
        "%CHROM:%POS:%REF:%ALT[\t%GT]\n",
        str(vcf_path),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=stderr_handle,
        text=True,
    )
    return process, stderr_handle


def resolve_output_path(out_csv: Path, explicit_path: str, suffix: str) -> Path:
    if explicit_path.strip():
        return Path(explicit_path)
    return out_csv.with_suffix(out_csv.suffix + suffix)


def main() -> int:
    args = build_parser().parse_args()
    populations = parse_populations(args.populations)
    labels = parse_label_map(args.label_map, populations)
    chromosomes = parse_chromosomes(args.chromosomes)

    ensure_tool(args.bcftools)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = resolve_output_path(out_csv, args.manifest, ".manifest.json")
    sample_manifest_path = resolve_output_path(out_csv, args.sample_manifest, ".samples.tsv")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    sample_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_sample_records(Path(args.panel), populations, labels)
    ordered_sample_ids = [record.sample_id for record in records]
    sample_lookup = {record.sample_id: record for record in records}

    selected_samples_path = out_csv.with_suffix(out_csv.suffix + ".samples.keep.txt")
    with selected_samples_path.open("w", encoding="utf-8") as handle:
        for sample_id in ordered_sample_ids:
            handle.write(sample_id)
            handle.write("\n")

    sample_columns: dict[str, list[str]] = {sample_id: [] for sample_id in ordered_sample_ids}
    variant_ids: list[str] = []
    vcf_files_used: list[str] = []
    scanned_variants = 0
    retained_variants = 0
    stop_after_max = False

    for chromosome in chromosomes:
        vcf_path = Path(args.vcf_pattern.format(chr=chromosome))
        if not vcf_path.exists():
            raise FileNotFoundError(f"VCF file missing for chromosome {chromosome}: {vcf_path}")

        chrom_started = time.perf_counter()
        print(f"[{chromosome}] scanning {vcf_path.name}", flush=True)
        vcf_files_used.append(str(vcf_path))
        stderr_log_path = out_csv.with_suffix(out_csv.suffix + f".chr{chromosome}.bcftools.stderr.log")
        process, stderr_handle = query_vcf(args.bcftools, vcf_path, selected_samples_path, stderr_log_path)
        assert process.stdout is not None

        scanned_before = scanned_variants
        retained_before = retained_variants

        for line in process.stdout:
            line = line.rstrip("\n")
            if not line:
                continue

            fields = line.split("\t")
            if len(fields) != len(ordered_sample_ids) + 1:
                continue

            scanned_variants += 1
            if scanned_variants % 100000 == 0:
                print(
                    f"[{chromosome}] progress: scanned={scanned_variants}, retained={retained_variants}",
                    flush=True,
                )
            variant_id = fields[0].replace(":", "_")
            dosages = [genotype_to_dosage(value) for value in fields[1:]]
            if not variant_passes(dosages, maf_min=args.maf_min, missing_max=args.missing_max):
                continue

            retained_variants += 1
            variant_ids.append(variant_id)
            for index, sample_id in enumerate(ordered_sample_ids):
                dosage = dosages[index]
                sample_columns[sample_id].append("" if dosage is None else str(dosage + 1))

            if args.max_snps > 0 and len(variant_ids) >= args.max_snps:
                stop_after_max = True
                break

        if process.stdout is not None:
            process.stdout.close()
        exit_code = process.wait()
        try:
            stderr_handle.close()
        except Exception:
            pass
        is_expected_sigpipe = stop_after_max and exit_code in (-13, 141)
        if exit_code != 0 and not is_expected_sigpipe:
            stderr_output = ""
            if stderr_log_path.exists():
                try:
                    stderr_output = "\n".join(stderr_log_path.read_text(encoding="utf-8").splitlines()[-20:])
                except OSError:
                    stderr_output = ""
            raise RuntimeError(
                f"bcftools query failed for {vcf_path} with exit code {exit_code}: {stderr_output.strip()}"
            )

        elapsed = time.perf_counter() - chrom_started
        scanned_chr = scanned_variants - scanned_before
        retained_chr = retained_variants - retained_before
        print(
            f"[{chromosome}] done in {elapsed:.1f}s: scanned_chr={scanned_chr}, retained_chr={retained_chr}, retained_total={retained_variants}",
            flush=True,
        )

        if stop_after_max:
            break

    if not variant_ids:
        raise RuntimeError("No SNPs passed filters. Relax thresholds or verify input files.")

    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["label", *variant_ids])
        for sample_id in ordered_sample_ids:
            sample_record = sample_lookup[sample_id]
            writer.writerow([sample_record.label, *sample_columns[sample_id]])

    with sample_manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["sample_id", "population", "label"])
        for sample_id in ordered_sample_ids:
            sample_record = sample_lookup[sample_id]
            writer.writerow([sample_record.sample_id, sample_record.population, sample_record.label])

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": {
            "panel": str(Path(args.panel)),
            "vcf_pattern": args.vcf_pattern,
            "vcf_files_used": vcf_files_used,
        },
        "cohort": {
            "populations": populations,
            "labels": labels,
            "n_samples": len(ordered_sample_ids),
            "n_samples_per_population": {
                population: sum(1 for record in records if record.population == population)
                for population in populations
            },
        },
        "filters": {
            "maf_min": args.maf_min,
            "missing_max": args.missing_max,
            "max_snps": args.max_snps,
            "chromosomes": chromosomes,
        },
        "results": {
            "scanned_variants": scanned_variants,
            "retained_variants": retained_variants,
            "selected_snps": len(variant_ids),
            "output_csv": str(out_csv),
            "sample_manifest": str(sample_manifest_path),
        },
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())