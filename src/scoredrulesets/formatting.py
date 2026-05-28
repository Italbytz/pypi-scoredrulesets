from __future__ import annotations

from .schema import ScoredRuleSet


def _format_scalar(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, int):
        return str(value)
    return str(value)


def _format_atom(atom) -> str:
    if atom.op == "between" and isinstance(atom.value, (list, tuple)) and len(atom.value) == 2:
        lower, upper = atom.value
        return f"{atom.feature} in [{_format_scalar(lower)}, {_format_scalar(upper)}]"
    if atom.op in {"in", "not_in"} and isinstance(atom.value, (list, tuple)):
        values = ", ".join(_format_scalar(value) for value in atom.value)
        operator = "in" if atom.op == "in" else "not in"
        return f"{atom.feature} {operator} {{{values}}}"
    return f"{atom.feature} {atom.op} {_format_scalar(atom.value)}"


def _format_condition(ruleset: ScoredRuleSet, rule_index: int) -> str:
    rule = ruleset.rules[rule_index]
    if not rule.atoms:
        return "emptyset"
    chunks = [_format_atom(atom) for atom in rule.atoms]
    return "{ " + ", ".join(chunks) + " }"


def _format_score_columns(ruleset: ScoredRuleSet, rule_index: int) -> list[str]:
    rule = ruleset.rules[rule_index]
    return [f"{score:.2f}" for score in rule.scores]


def _is_zero_one_label(label: object) -> bool:
    if isinstance(label, bool):
        return True
    if isinstance(label, int):
        return label in {0, 1}
    if isinstance(label, float):
        return label in {0.0, 1.0}
    if isinstance(label, str):
        normalized = label.strip().lower()
        return normalized in {"0", "1", "false", "true"}
    return False


def _normalize_zero_one_label(label: object) -> int:
    if isinstance(label, bool):
        return int(label)
    if isinstance(label, int):
        return label
    if isinstance(label, float):
        return int(label)
    if isinstance(label, str):
        normalized = label.strip().lower()
        if normalized in {"1", "true"}:
            return 1
    return 0


def _display_class_labels(class_labels: list[object]) -> list[str]:
    if len(class_labels) == 2 and all(_is_zero_one_label(label) for label in class_labels):
        rendered = []
        for label in class_labels:
            rendered.append("Yes" if _normalize_zero_one_label(label) == 1 else "No")
        return rendered
    return [str(label) for label in class_labels]


def _table_headers(ruleset: ScoredRuleSet) -> list[str]:
    return [*_display_class_labels(ruleset.class_labels), "S_r"]


def _table_row(ruleset: ScoredRuleSet, rule_index: int) -> list[str]:
    return [
        *_format_score_columns(ruleset, rule_index),
        _format_condition(ruleset, rule_index),
    ]


def _escape_md(text: str) -> str:
    return text.replace("|", "\\|")


def _escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in text:
        out.append(replacements.get(ch, ch))
    return "".join(out)


def _truncate_condition(condition: str, max_condition_len: int) -> str:
    if len(condition) <= max_condition_len:
        return condition
    return condition[: max(0, max_condition_len - 3)] + "..."


def format_ruleset_table(ruleset: ScoredRuleSet, max_condition_len: int = 80) -> str:
    """Format a ScoredRuleSet as an ASCII table for terminal output."""
    headers = _table_headers(ruleset)
    rows = []
    for idx in range(len(ruleset.rules)):
        row = _table_row(ruleset, idx)
        row[-1] = _truncate_condition(row[-1], max_condition_len)
        rows.append(row)

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


def format_ruleset_markdown(ruleset: ScoredRuleSet, max_condition_len: int = 120) -> str:
    """Format a ScoredRuleSet as a Markdown table."""
    headers = _table_headers(ruleset)
    lines = ["| " + " | ".join(_escape_md(header) for header in headers) + " |"]
    lines.append("|" + "---|" * len(headers))
    for idx in range(len(ruleset.rules)):
        row = _table_row(ruleset, idx)
        row[-1] = _truncate_condition(row[-1], max_condition_len)
        lines.append("| " + " | ".join(_escape_md(value) for value in row) + " |")
    return "\n".join(lines)


def format_ruleset_latex(ruleset: ScoredRuleSet, max_condition_len: int = 120) -> str:
    """Format a ScoredRuleSet as a LaTeX tabular block."""
    column_spec = "r l " + " ".join("r" for _ in ruleset.class_labels) + " p{0.38\\linewidth}"
    lines = [
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\hline",
        " & ".join(_escape_latex(header) for header in _table_headers(ruleset)) + r" \\",
        r"\hline",
    ]
    for idx in range(len(ruleset.rules)):
        row = _table_row(ruleset, idx)
        row[-1] = _truncate_condition(row[-1], max_condition_len)
        lines.append(" & ".join(_escape_latex(value) for value in row) + r" \\")
    lines.extend([
        r"\hline",
        r"\end{tabular}",
    ])
    return "\n".join(lines)

