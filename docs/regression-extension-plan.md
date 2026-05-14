# Scored Rule Sets: Regression Extension Plan

## Goal

Extend scored rule sets from classification-only to a task-aware framework that also supports regression, while preserving interpretability and sklearn compatibility.

## Current constraints in the codebase

- Estimator base is classification-specific (`ClassifierMixin`).
- Rule set schema assumes `class_labels` and per-rule class score vectors.
- Runtime prediction uses class aggregation (`argmax_sum`, `softmax_sum`) and label decoding.
- Auto estimator uses stratified CV and classification scoring defaults.

## Design principles

- Keep backward compatibility for existing classification users.
- Introduce task-aware abstractions with explicit `task_type`.
- Avoid mixing classification and regression semantics in one implicit runtime path.
- Reuse current atom/rule extraction and backend bridges where possible.

## Proposed architecture

### 1) Schema evolution (non-breaking)

Add task metadata to the rule set format:

- `task_type`: `"classification" | "regression"`
- Keep `class_labels` required only for classification.
- For regression, allow exactly one scalar output per rule.

Suggested approach:

- Keep current `ScoredRuleSet` for compatibility.
- Introduce `TaskAwareRuleSet` (or `ScoredRuleSetV2`) with validation by `task_type`.
- Add serializer support for both versions during migration.

### 2) Runtime split

Keep current runtime functions for classification, add regression runtime:

- `decision_function_classification(...)`
- `predict_proba_classification(...)`
- `predict_classification(...)`
- `predict_regression(...)`

Regression aggregators (initial set):

- `weighted_sum`
- `mean_active`
- `default_plus_sum` (default rule value + active adjustments)

### 3) Estimator hierarchy

Refactor base estimator into neutral core + task-specific mixins:

- `BaseRuleSetEstimatorCore` (shared utilities)
- `BaseRuleSetClassifier` (`ClassifierMixin`)
- `BaseRuleSetRegressor` (`RegressorMixin`)

Add first public regressor wrappers:

- `ScoredRuleSetRegressor`
- `AutoScoredRuleSetRegressor`

### 4) Backend strategy for phase 1

Best initial backend candidates:

- Native: `rulegp` (fitness can be switched from F1/recall to MAE/RMSE/R2 objectives)
- Optional fast baseline: tree wrapper with `DecisionTreeRegressor` plus post-hoc rule conversion

Deferred to phase 2:

- `ruleplcs` regression variant
- additional evolutionary backends for regression

### 5) Evaluation changes

Regression metrics:

- primary: MAE
- secondary: RMSE, R2

Auto regressor CV:

- `KFold` instead of `StratifiedKFold`
- scoring default: `neg_mean_absolute_error`

### 6) Backward compatibility

- Existing `ScoredRuleSetClassifier` behavior unchanged.
- Existing JSON files remain loadable.
- New regression JSON files should be rejected by old runtime with clear error messages.

## Implementation roadmap

### Milestone A: Core model and runtime

- Add task-aware schema validation.
- Add regression runtime functions and tests.
- Add migration tests for old JSON format.

### Milestone B: Regressor wrapper

- Implement `ScoredRuleSetRegressor` using a tree surrogate baseline.
- Add sklearn estimator checks for regressor API compliance.

### Milestone C: Native evolutionary regressor

- Implement `RuleGPRegressor` fitness and rule-weight optimization.
- Add benchmark scripts for tabular regression datasets.

### Milestone D: Auto regressor

- Implement `AutoScoredRuleSetRegressor` with per-backend timeouts.
- Add report templates analogous to classification benchmark reports.

## Risks and mitigations

- Risk: semantic drift between classification and regression aggregators.
- Mitigation: separate runtime entry points and strict schema validation.

- Risk: increased maintenance from dual-task support.
- Mitigation: shared core utilities and strict task-specific interfaces.

- Risk: overfitting in native evolutionary regression.
- Mitigation: regularization objectives and validation-aware fitness.

## Recommended immediate next step

Implement Milestone A first, then a minimal tree-based `ScoredRuleSetRegressor` to validate end-to-end API design before evolutionary optimization work.
