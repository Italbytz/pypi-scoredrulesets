# GPAS Paper Reproduction Benchmarks

Reproduces the benchmarks reported in:
> Nunkesser, R., Bernholt, T., Schwender, H., Bieber, K., & Wieczorek, S. (2007). *Detecting complex interactions in high-dimensional genomic data using genetic programming*. Bioinformatics, 23(24), 3280–3288.

Target values from Table 1 (GPAS mean misclassification rate):
- **HapMap**: 0.011 (Open Data included in `examples/snp/data/hapmap157.csv`)
- **Simulation**: 0.335 (Synthetically generated via `catgen`)
- **GENICA**: 0.392 (Protected clinical study cohort, not included)

## Usage

### Smoke test (runs in seconds)
```bash
python examples/benchmarks/gpas/run_gpas_paper_like.py --profile smoke
```

### Standard reproduction (HapMap + Simulation)
```bash
python examples/benchmarks/gpas/run_gpas_paper_like.py --profile quick
```

### Running with private GENICA data (if locally available)
GENICA data is sensitive patient data and must never be committed to public repositories. If you have authorized access to `genica63.csv`:
```bash
python examples/benchmarks/gpas/run_gpas_paper_like.py \
  --profile quick \
  --datasets all \
  --genica-path /path/to/private/genica63.csv
```
