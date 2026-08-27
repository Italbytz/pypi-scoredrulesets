__version__ = "0.1.0"

from .benchmarking import (
    AggregatedBenchmarkResult,
    BenchmarkComparisonRow,
    BenchmarkConfig,
    BenchmarkResult,
    ClusterApproxRunResult,
    ClusterApproxSummary,
    aggregate_benchmark_results,
    aggregated_results_as_dicts,
    benchmark_cluster_approximation,
    build_benchmark_leaderboard,
    compare_benchmark_runs,
    format_cluster_approximation_runs_markdown,
    format_benchmark_comparison_report_html,
    format_benchmark_comparison_report_markdown,
    format_benchmark_leaderboard_html,
    format_benchmark_leaderboard_markdown,
    format_benchmark_leaderboard_table,
    format_benchmark_meta_report_html,
    format_benchmark_meta_report_markdown,
    format_benchmark_report_html,
    format_benchmark_report_markdown,
    load_aggregated_results_json,
    plot_benchmark_heatmap,
    plot_benchmark_heatmap_combined,
    plot_benchmark_results,
    run_benchmarks,
    summarize_cluster_approximation_runs,
)
from .estimators.auto import AutoScoredRuleSetClassifier
from .estimators.cluster_wrapper import ScoredRuleSetClusterer
from .estimators.logicgp import GPASClassifier, LogicGPClassifier
from .estimators.rulenln import RuleNLNClassifier
from .estimators.rulegp import RuleGPClassifier
from .estimators.ruleplcs import RulePLCSClassifier
from .estimators.rulensga2 import RuleNSGA2Classifier
from .estimators.rulekit_native import RuleKitNativeClassifier
from .estimators.atom_selection import (
    available_atom_selection_strategies,
    is_atom_selection_strategy_available,
    register_atom_selection_strategy,
)
from .estimators.sklearn_wrapper import ScoredRuleSetClassifier, ScoredRuleSetRegressor
from .estimators._time_budget import FitBudgetExceededError
from .formatting import format_ruleset_latex, format_ruleset_markdown, format_ruleset_table
from .io import dump_ruleset_json, dump_ruleset_latex, dump_ruleset_markdown, load_ruleset_json
from .preprocessing import BorutaSelector, build_feature_selector, build_preprocessing_pipeline
from .runtime import predict_regression
from .schema import AggregationSpec, Atom, Rule, ScoredRuleSet
from .importers.logicgp import import_logicgp_json

__all__ = [
    "AggregationSpec",
    "Atom",
    "Rule",
    "ScoredRuleSet",
    "AggregatedBenchmarkResult",
    "AutoScoredRuleSetClassifier",
    "available_atom_selection_strategies",
    "ScoredRuleSetClusterer",
    "BenchmarkComparisonRow",
    "BenchmarkConfig",
    "BenchmarkResult",
    "ClusterApproxRunResult",
    "ClusterApproxSummary",
    "BorutaSelector",
    "GPASClassifier",
    "LogicGPClassifier",
    "RuleGPClassifier",
    "RuleNLNClassifier",
    "RulePLCSClassifier",
    "RuleNSGA2Classifier",
    "RuleKitNativeClassifier",
    "ScoredRuleSetClassifier",
    "ScoredRuleSetRegressor",
    "FitBudgetExceededError",
    "is_atom_selection_strategy_available",
    "aggregate_benchmark_results",
    "aggregated_results_as_dicts",
    "benchmark_cluster_approximation",
    "build_benchmark_leaderboard",
    "build_feature_selector",
    "build_preprocessing_pipeline",
    "compare_benchmark_runs",
    "dump_ruleset_json",
    "dump_ruleset_latex",
    "dump_ruleset_markdown",
    "format_benchmark_comparison_report_html",
    "format_benchmark_comparison_report_markdown",
    "format_benchmark_leaderboard_html",
    "format_benchmark_leaderboard_markdown",
    "format_benchmark_leaderboard_table",
    "format_benchmark_meta_report_html",
    "format_benchmark_meta_report_markdown",
    "format_benchmark_report_html",
    "format_benchmark_report_markdown",
    "format_cluster_approximation_runs_markdown",
    "format_ruleset_latex",
    "format_ruleset_markdown",
    "format_ruleset_table",
    "import_logicgp_json",
    "load_aggregated_results_json",
    "load_ruleset_json",
    "plot_benchmark_heatmap",
    "plot_benchmark_heatmap_combined",
    "plot_benchmark_results",
    "predict_regression",
    "register_atom_selection_strategy",
    "run_benchmarks",
    "summarize_cluster_approximation_runs",
]

