# scoredrulesets

[![PyPI version](https://img.shields.io/pypi/v/scoredrulesets.svg)](https://pypi.org/project/scoredrulesets/)
[![Python versions](https://img.shields.io/pypi/pyversions/scoredrulesets.svg)](https://pypi.org/project/scoredrulesets/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/scoredrulesets/scoredrulesets/actions/workflows/ci.yml/badge.svg)](https://github.com/scoredrulesets/scoredrulesets/actions/workflows/ci.yml)

A scikit-learn compatible toolkit for **scored rule set** classification — interpretable models that express each prediction as a weighted combination of human-readable IF-THEN rules.

## What are Scored Rule Sets?

A scored rule set is a set of rules of the form

> **IF** condition **THEN** score vector

Each rule assigns a real-valued score to every class. At prediction time the score vectors of all *firing* rules are summed and the class with the highest total score is returned.  This representation supports:

- **Full interpretability** — every prediction can be traced back to the rules that fired.
- **Multi-class via a single model** — no one-vs-rest decomposition needed.
- **Compact models** — typically 5–20 rules cover the relevant patterns.

## Estimators

| Estimator | Algorithm | Requires |
|---|---|---|
| `ScoredRuleSetClassifier` | Wrapper: converts any scikit-learn rule learner into a scored rule set | depends on `backend=` |
| `RuleGPClassifier` | NSGA-II genetic programming over rule populations | – (pure Python) |
| `RuleLCSClassifier` | Sequential covering with a genetic algorithm (LCS-style) | – (pure Python) |
| `RuleNLNClassifier` | Neural rule extraction (Neural Logic Networks) | `torch` |
| `AutoScoredRuleSetClassifier` | Auto-selects best estimator via cross-validation | – |

### Wrapper backends

`ScoredRuleSetClassifier(backend=...)` delegates training to an external rule learner and post-hoc converts the result:

| `backend=` | Library |
|---|---|
| `"hs"` | `imodels` (Hierarchical Shrinkage / RIPPER / …) |
| `"cart"` | scikit-learn `DecisionTreeClassifier` |
| `"exstracs"` | `scikit-ExSTraCS` |
| `"rulekit"` | `rulekit` + `jpype1` (Java) |
| `"logicgp"` | logicGP JSON import |

## Installation

```bash
pip install scoredrulesets
```

With optional backends:

```bash
# imodels backend (HS, RIPPER, …)
pip install "scoredrulesets[hs]"

# ExSTraCS backend
pip install "scoredrulesets[exstracs]"

# RuleKit backend (requires a JDK)
pip install "scoredrulesets[rulekit]"

# All optional backends
pip install "scoredrulesets[all]"

# Benchmarking utilities (matplotlib, pandas)
pip install "scoredrulesets[benchmark]"
```

## Quick Start

### RuleLCS (no extra dependencies)

```python
from sklearn.datasets import load_iris
from scoredrulesets import RuleLCSClassifier, format_ruleset_table

X, y = load_iris(return_X_y=True)

clf = RuleLCSClassifier(max_rules=6, random_state=42)
clf.fit(X, y)
print(clf.predict(X[:3]))

ruleset = clf.to_ruleset()
print(format_ruleset_table(ruleset))
```

### Wrapper with HS backend

```python
from sklearn.datasets import load_iris
from scoredrulesets import ScoredRuleSetClassifier, dump_ruleset_json

X, y = load_iris(return_X_y=True)

clf = ScoredRuleSetClassifier(backend="hs")   # requires: pip install scoredrulesets[hs]
clf.fit(X, y)

ruleset = clf.to_ruleset()
dump_ruleset_json(ruleset, "iris_ruleset.json")
```

### RuleGP (NSGA-II genetic programming)

```python
from sklearn.datasets import load_iris
from scoredrulesets import RuleGPClassifier

X, y = load_iris(return_X_y=True)

clf = RuleGPClassifier(max_rules=8, n_generations=50, random_state=42)
clf.fit(X, y)
print(clf.score(X, y))
```

### Loading a saved rule set

```python
from scoredrulesets import load_ruleset_json, ScoredRuleSetClassifier

ruleset = load_ruleset_json("iris_ruleset.json")
clf = ScoredRuleSetClassifier.from_ruleset(ruleset)
print(clf.predict(X[:3]))
```

## Rule Set JSON Format

Rule sets can be serialised to a human-readable JSON format:

```json
{
  "format": "scoredrulesets",
  "version": "0.1",
  "class_labels": ["setosa", "versicolor", "virginica"],
  "feature_names": ["sepal length", "sepal width", "petal length", "petal width"],
  "aggregation": {"type": "argmax_sum"},
  "rules": [
    {
      "atoms": [{"feature": "petal length", "op": "<=", "value": 2.45}],
      "scores": [1.0, 0.0, 0.0]
    }
  ]
}
```

## Benchmarking

Benchmark entry points are under `examples/benchmarks/` and wrapped in `Makefile` targets:

```bash
make benchmark            # standard benchmark suite
make benchmark-standard   # standard datasets only
make reports-standard     # regenerate reports from existing results
```

Direct invocation:

```bash
python examples/benchmarks/benchmark_standard.py
python examples/benchmarks/generate_reports.py normal-lite
```

## Running Tests

```bash
# core tests (no optional dependencies required)
pytest -q

# include HS integration tests
pytest -q -m hs

# sklearn estimator interface checks
pytest -q tests/test_estimator_checks.py
```

## Examples

See [`examples/`](examples/) for runnable demos:

- `examples/estimators/example_rulelcs_backend.py` — RuleLCS direct estimator
- `examples/estimators/example_rulelcs_wrapper.py` — RuleLCS via wrapper
- `examples/estimators/example_rulenln_backend.py` — RuleNLN (neural)
- `examples/estimators/example_rule_shrinking.py` — LRC rule compaction

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

## License

MIT — see [LICENSE](LICENSE).

