# Contributing to scoredrulesets

Thank you for your interest in contributing! This document explains how to get started.

## Setting Up a Development Environment

```bash
git clone https://github.com/scoredrulesets/scoredrulesets.git
cd scoredrulesets

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e ".[dev,hs]"
```

## Running Tests

```bash
# core tests
pytest -q

# include optional backend tests (requires imodels)
pytest -q -m hs

# scikit-learn API compliance
pytest -q tests/test_estimator_checks.py
```

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/).
- Keep public APIs scikit-learn compatible (fit / predict / score / get_params / set_params).
- New estimators should pass `sklearn.utils.estimator_checks.check_estimator`.

## Submitting Changes

1. Fork the repository and create a feature branch:
   ```bash
   git checkout -b feature/my-feature
   ```
2. Make your changes and add tests where appropriate.
3. Ensure the full test suite passes: `pytest -q`
4. Open a pull request against `main` with a clear description of the change.

## Reporting Issues

Please open an issue on [GitHub Issues](https://github.com/scoredrulesets/scoredrulesets/issues) and include:

- Python version and OS
- `scoredrulesets` version (`pip show scoredrulesets`)
- A minimal reproducible example
- The full traceback

## Adding a New Backend

1. Create `src/scoredrulesets/estimators/my_backend.py` implementing the `BaseEstimator` + `ClassifierMixin` interface and a `to_ruleset()` method returning a `ScoredRuleSet`.
2. Register the backend in `src/scoredrulesets/estimators/auto.py`.
3. Add an optional dependency group in `pyproject.toml`.
4. Add integration tests under `tests/` (marked with a custom pytest marker).
5. Add an example under `examples/estimators/`.

## License

By contributing you agree that your contributions will be licensed under the MIT License.
