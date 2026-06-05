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
| `RuleNSGA2Classifier` | NSGA-II genetic programming over rule populations | – (pure Python) |
| `RulePLCSClassifier` | Sequential covering with a genetic algorithm (LCS-style) | – (pure Python) |
| `RuleNLNClassifier` | Neural rule extraction (Neural Logic Networks) | `torch` |
| `AutoScoredRuleSetClassifier` | Auto-selects best estimator via cross-validation | – |

### Atom-Space Strategy Guide

Atom-based estimators now expose explicit pre-fit strategy parameters so you
can control how non-categorical features are transformed into candidate atoms.

| Estimator | Parameter | Options (default first) |
|---|---|---|
| `LogicGPClassifier` | `feature_encoding_strategy` | `"auto_low_cardinality"`, `"force_numeric_bins"` |
| `RuleNSGA2Classifier` | `atom_space_strategy` | `"hybrid"`, `"numeric_only"`, `"categorical_low_cardinality_only"` |
| `RuleNSGA2Classifier` | `atom_preselection_strategy` | `"none"`, any registered atom-selection strategy (e.g. `"top_c2_private"`) |
| `RuleGPClassifier` | `atom_space_strategy` | `"hybrid"`, `"numeric_only"`, `"categorical_low_cardinality_only"` |
| `RuleGPClassifier` | `atom_preselection_strategy` | `"none"`, `"logicgp_singleton"`, `"logicgp_binned_sets"`, any registered atom-selection strategy (e.g. `"top_c2_private"`) |
| `RuleNLNClassifier` | `threshold_strategy` | `"quantile_midpoint"`, `"quantile_only"`, `"midpoint_only"` |
| `RuleKitNativeClassifier` | `atom_preselection_strategy` | `"none"`, any registered atom-selection strategy (e.g. `"top_c2_private"`) |
| `RulePLCSClassifier` | `feature_typing_strategy` | `"auto_low_cardinality"`, `"all_numeric"`, `"all_integer_categorical"` |
| `RulePLCSClassifier` | `atom_preselection_strategy` | `"none"`, any registered atom-selection strategy (e.g. `"top_c2_private"`) |

### Atom-Selection Plugins

Atom-selection strategies can be extended without shipping the implementation
inside the public core package. This is useful for private or embargoed
research variants.

Built-in strategy names are available via:

```python
from scoredrulesets import available_atom_selection_strategies

print(available_atom_selection_strategies())
```

You can register strategies at runtime:

```python
from scoredrulesets import register_atom_selection_strategy

def my_selector(candidates, y_idx, n_classes, min_samples_leaf, top_k):
  selected = set()
  for signature, mask in candidates:
    if mask.sum() < min_samples_leaf:
      continue
    selected.add(signature)
    if len(selected) >= top_k:
      break
  return selected

register_atom_selection_strategy("my_selector", my_selector)
```

Or via package entry points (recommended for separate private packages):

```toml
[project.entry-points."scoredrulesets.atom_selection"]
my_selector = "my_private_pkg.atomsel:my_selector"
```

Recommended starting points:

| Data profile | Recommended strategy choices |
|---|---|
| Mixed continuous + encoded categoricals | Keep defaults (`auto_low_cardinality`, `hybrid`, `quantile_midpoint`) |
| Mostly continuous signals | Use `force_numeric_bins` (LogicGP), `numeric_only` (RuleNSGA2/RuleGP), `quantile_only` (RuleNLN), `all_numeric` (RulePLCS) |
| Mostly integer-coded symbolic features | Use `hybrid` or `categorical_low_cardinality_only` (RuleNSGA2/RuleGP), `midpoint_only` (RuleNLN), `all_integer_categorical` (RulePLCS) |

Minimal example:

```python
from scoredrulesets import RuleGPClassifier

clf = RuleGPClassifier(
  atom_space_strategy="numeric_only",
  random_state=42,
)
```

### Wrapper backends

`ScoredRuleSetClassifier(backend=...)` delegates training to an external rule learner and post-hoc converts the result:

| `backend=` | Library |
|---|---|
| `"hs"` | `imodels` (Hierarchical Shrinkage / RIPPER / …) |
| `"cart"` | scikit-learn `DecisionTreeClassifier` |
| `"exstracs"` | `scikit-ExSTraCS` |
| `"rulekit"` | `rulekit` + `jpype1` (Java) |
| `"rulekit_native"` | native RuleKit-style learner (pure Python) |
| `"logicgp"` | logicGP JSON import |

## Fit-Time Budgets

Several native or integrated estimators support cooperative wall-clock limits
via `max_fit_seconds`. When the budget is exhausted, training stops cleanly and
the best model found so far is returned.

Supported backends currently include:

- `LogicGPClassifier`
- `RuleGPClassifier`
- `RuleNSGA2Classifier`
- `RulePLCSClassifier`
- `RuleNLNClassifier`
- `RuleKitNativeClassifier`

You can set the budget directly on a native estimator:

```python
from scoredrulesets import RulePLCSClassifier

clf = RulePLCSClassifier(max_rules=8, max_fit_seconds=120, random_state=42)
```

Or uniformly through the wrapper:

```python
from scoredrulesets import ScoredRuleSetClassifier

clf = ScoredRuleSetClassifier(
  backend="ruleplcs",
  backend_params={"max_rules": 8},
  max_fit_seconds=120,
  random_state=42,
)
```

Backends without cooperative timeout support ignore `max_fit_seconds`; for
hard per-run stopping in experiments, the benchmarking runner provides an
additional external timeout layer.

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

### RulePLCS (no extra dependencies)

```python
from sklearn.datasets import load_iris
from scoredrulesets import RulePLCSClassifier, format_ruleset_table

X, y = load_iris(return_X_y=True)

clf = RulePLCSClassifier(max_rules=6, random_state=42)
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
from scoredrulesets import RuleNSGA2Classifier

X, y = load_iris(return_X_y=True)

clf = RuleNSGA2Classifier(max_rules=8, n_generations=50, random_state=42)
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

### Exporting a learned model as Markdown / LaTeX

```python
from scoredrulesets import dump_ruleset_latex, dump_ruleset_markdown

ruleset = clf.to_ruleset()
dump_ruleset_markdown(ruleset, "restaurant_ruleset.md")
dump_ruleset_latex(ruleset, "restaurant_ruleset.tex")
```

This is useful for lecture material, papers, and documentation where the
same learned model should be shown in different output formats.

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

- `examples/estimators/example_ruleplcs_backend.py` — RulePLCS direct estimator
- `examples/estimators/example_ruleplcs_wrapper.py` — RulePLCS via wrapper
- `examples/estimators/example_rulenln_backend.py` — RuleNLN (neural)
- `examples/estimators/example_rule_shrinking.py` — LRC rule compaction

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

## License

MIT — see [LICENSE](LICENSE).

