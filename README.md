# scoredrulesets

[![PyPI version](https://img.shields.io/pypi/v/scoredrulesets.svg)](https://pypi.org/project/scoredrulesets/)
[![Python versions](https://img.shields.io/pypi/pyversions/scoredrulesets.svg)](https://pypi.org/project/scoredrulesets/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/Italbytz/pypi-scoredrulesets/actions/workflows/ci.yml/badge.svg)](https://github.com/Italbytz/pypi-scoredrulesets/actions/workflows/ci.yml)

A scikit-learn compatible toolkit for **scored rule set** classification, regression, and clustering — interpretable models that express predictions as weighted combinations of human-readable IF-THEN rules.

## What are Scored Rule Sets?

A scored rule set is a set of rules of the form:

> **IF** condition **THEN** score vector

Each rule assigns real-valued scores to every class/cluster. At prediction time, the score vectors of all *firing* rules are aggregated, and the class or value with the highest total score is returned.

**Why scored rule sets?**

- **Full interpretability** — every prediction traces back to the rules that fired.
- **Single model for multi-class** — no one-vs-rest decomposition needed.
- **Compact models** — typically 5–20 rules capture relevant patterns.
- **Supports classification, regression, and clustering** — unified interface.

## Core Estimators

### Classification

| Estimator | Algorithm | Description |
|---|---|---|
| `RulePLCSClassifier` | Sequential covering + GA | Fast, no external deps; suitable for large datasets |
| `RuleNSGA2Classifier` | NSGA-II genetic programming | Multi-objective optimization; tunable rule set complexity |
| `RuleGPClassifier` | logicGP-style evolution | Direct atom/rule/rule-set optimization; domain-aware |
| `RuleNLNClassifier` | Neural Logic Networks | Neural extraction; requires `torch` |
| `ScoredRuleSetClassifier` | Wrapper for scikit-learn learners | Convert CART, HS, RuleKit, ExSTraCS, logicGP to rulesets |
| `AutoScoredRuleSetClassifier` | Auto-selection via CV | Finds best estimator for your data |

### Regression & Clustering

| Estimator | Task |
|---|---|
| `ScoredRuleSetRegressor` | Regression wrapper; works with all backends |
| `ScoredRuleSetClusterer` | Cluster-label approximation; explains k-means, etc. |

## Installation

```bash
pip install scoredrulesets
```

**Optional backends:**

```bash
# imodels (HS, RIPPER, etc.)
pip install "scoredrulesets[hs]"

# scikit-ExSTraCS (learning classifier systems)
pip install "scoredrulesets[exstracs]"

# RuleKit (requires JDK)
pip install "scoredrulesets[rulekit]"

# All backends
pip install "scoredrulesets[all]"

# Benchmarking utilities (matplotlib, pandas)
pip install "scoredrulesets[benchmark]"
```

## Quick Start

### Example 1: RulePLCS Classifier (no dependencies)

```python
from sklearn.datasets import load_iris
from scoredrulesets import RulePLCSClassifier, format_ruleset_table

X, y = load_iris(return_X_y=True)

# Train
clf = RulePLCSClassifier(max_rules=6, random_state=42)
clf.fit(X, y)

# Predict
print(clf.predict(X[:3]))

# View rules
ruleset = clf.to_ruleset()
print(format_ruleset_table(ruleset))
```

### Example 2: Multi-Objective Genetic Programming

```python
from sklearn.datasets import load_iris
from scoredrulesets import RuleNSGA2Classifier

X, y = load_iris(return_X_y=True)

clf = RuleNSGA2Classifier(
    max_rules=8,
    n_generations=50,
    random_state=42
)
clf.fit(X, y)
print(f"Accuracy: {clf.score(X, y):.3f}")
```

### Example 3: Wrapper Backend (HS)

```python
from sklearn.datasets import load_iris
from scoredrulesets import ScoredRuleSetClassifier, dump_ruleset_json

X, y = load_iris(return_X_y=True)

# Requires: pip install scoredrulesets[hs]
clf = ScoredRuleSetClassifier(backend="hs")
clf.fit(X, y)

# Save to JSON
ruleset = clf.to_ruleset()
dump_ruleset_json(ruleset, "iris_rules.json")
```

### Example 4: Regression

```python
from sklearn.datasets import load_diabetes
from scoredrulesets import ScoredRuleSetRegressor

X, y = load_diabetes(return_X_y=True)

reg = ScoredRuleSetRegressor(backend="cart")
reg.fit(X, y)
print(f"R² score: {reg.score(X, y):.3f}")
```

### Example 5: Load & Reuse

```python
from scoredrulesets import load_ruleset_json, ScoredRuleSetClassifier

# Load previously saved ruleset
ruleset = load_ruleset_json("iris_rules.json")

# Create estimator from it
clf = ScoredRuleSetClassifier.from_ruleset(ruleset)

# Use for predictions
print(clf.predict(X[:3]))
```

### Example 6: Export for Papers

```python
from scoredrulesets import dump_ruleset_markdown, dump_ruleset_latex

ruleset = clf.to_ruleset()

# Export to Markdown (documentation)
dump_ruleset_markdown(ruleset, "rules.md")

# Export to LaTeX (papers)
dump_ruleset_latex(ruleset, "rules.tex")
```

## Rule Set JSON Format

Rule sets are serialized to human-readable JSON:

```json
{
  "format": "scoredrulesets",
  "version": "0.1",
  "class_labels": ["setosa", "versicolor", "virginica"],
  "feature_names": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
  "aggregation": {"type": "argmax_sum"},
  "rules": [
    {
      "atoms": [{"feature": "petal_length", "op": "<=", "value": 2.45}],
      "scores": [1.0, 0.0, 0.0]
    },
    {
      "atoms": [{"feature": "petal_length", "op": ">", "value": 2.45}],
      "scores": [0.0, 0.4, 0.6]
    }
  ]
}
```

## Benchmarking

Run standard benchmarks (see `Makefile`):

```bash
make benchmark              # Full suite
make benchmark-standard     # Standard datasets
make reports-standard       # Regenerate reports
```

Direct invocation:

```bash
python examples/benchmarks/benchmark_standard.py
python examples/benchmarks/generate_reports.py normal-lite
```

## Running Tests

```bash
# Core tests (no optional deps)
pytest -q

# With HS backend
pytest -q -m hs

# sklearn compliance checks
pytest -q tests/test_estimator_checks.py
```

## Examples

Runnable demos in `examples/`:

- `examples/estimators/example_ruleplcs_backend.py` — RulePLCS direct use
- `examples/estimators/example_ruleplcs_wrapper.py` — RulePLCS via wrapper
- `examples/estimators/example_rulenln_backend.py` — RuleNLN (neural extraction)
- `examples/estimators/example_rule_shrinking.py` — LRC rule compaction

## Configuration & Advanced Use

For detailed documentation on feature encoding, atom preselection, fit-time budgets, and plugin mechanisms, see the [API documentation](https://github.com/Italbytz/pypi-scoredrulesets/wiki).

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

## Citation

If you use scoredrulesets in research, please cite the accompanying paper:

```bibtex
@inproceedings{nunkesser2026scoredrulesets,
  title={Scored Rule Sets for Interpretable Machine Learning},
  author={Nunkesser, Christoph and others},
  year={2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
