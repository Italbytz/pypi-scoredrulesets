from __future__ import annotations

from .schema import ScoredRuleSet


def _format_condition(ruleset: ScoredRuleSet, rule_index: int) -> str:
    rule = ruleset.rules[rule_index]
    if not rule.atoms:
        return "<default>"
    chunks = []
    for atom in rule.atoms:
        chunks.append(f"{atom.feature} {atom.op} {atom.value}")
    return " AND ".join(chunks)


def _format_scores(ruleset: ScoredRuleSet, rule_index: int) -> str:
    rule = ruleset.rules[rule_index]
    labels = ruleset.class_labels
    entries = []
    for label, score in zip(labels, rule.scores):
        entries.append(f"{label}:{score:.3f}")
    return ", ".join(entries)


def format_ruleset_table(ruleset: ScoredRuleSet, max_condition_len: int = 80) -> str:
    """Format a ScoredRuleSet as an ASCII table for terminal output."""
    rows: list[list[str]] = []
    for idx, rule in enumerate(ruleset.rules):
        condition = _format_condition(ruleset, idx)
        if len(condition) > max_condition_len:
            condition = condition[: max(0, max_condition_len - 3)] + "..."
        rows.append([
            str(idx),
            rule.rule_id or "-",
            condition,
            _format_scores(ruleset, idx),
        ])

    headers = ["idx", "rule_id", "condition", "scores"]
    widths = [len(h) for h in headers]
    for row in rows:
        for col_idx, value in enumerate(row):
            widths[col_idx] = max(widths[col_idx], len(value))

    def fmt_row(values: list[str]) -> str:
        padded = [v.ljust(widths[i]) for i, v in enumerate(values)]
        return "| " + " | ".join(padded) + " |"

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    lines = [sep, fmt_row(headers), sep]
    for row in rows:
        lines.append(fmt_row(row))
    lines.append(sep)
    return "\n".join(lines)

