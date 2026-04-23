# SNP Example Scripts

Quick-run scripts that demonstrate `GPASClassifier` and `LogicGPClassifier`
on real and simulated SNP data.  These scripts are the Python counterpart of
the R reference implementations in the
[paper-code](https://github.com/RobinNunkesser/paper-code) companion repo.

## Scripts

| Script | Classifier | Data | Equivalent R script |
|---|---|---|---|
| `quick_hapmap_run.py` | `GPASClassifier` (binary) | HapMap CHB vs JPT | `quick_hapmap_run.R` |
| `quick_scrime_run.py` | `LogicGPClassifier` (3-class) | scrime simulation | `quick_scrime_run.R` |

## Data

| File | Source | License |
|---|---|---|
| `data/hapmap157.csv` | [International HapMap Project](https://www.genome.gov/10001688/international-hapmap-project) | Public domain |
| `data/scrime.csv` | Simulated via R `scrime` package | Simulated |

Columns in `hapmap157.csv`: semicolon-separated; column 0 = population label
(0 = CHB, 1 = JPT); columns 1–157 = SNP genotypes coded 1–3.

Columns in `scrime.csv`: comma-separated; columns `x.SNP1`…`x.SNP50` = SNP
genotypes coded 0–2; column `y` = class label (0, 1, 2).

## Usage

Run from the repository root:

```bash
python examples/snp/quick_hapmap_run.py
python examples/snp/quick_scrime_run.py
```

The package must be installed (or the repo must be on `PYTHONPATH`):

```bash
pip install -e .
```
