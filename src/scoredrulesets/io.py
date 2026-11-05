from __future__ import annotations

import json
from pathlib import Path

from .formatting import format_ruleset_latex, format_ruleset_markdown
from .schema import ScoredRuleSet


def dump_ruleset_json(ruleset: ScoredRuleSet, path: str | Path, indent: int = 2) -> None:
    target = Path(path)
    target.write_text(json.dumps(ruleset.to_dict(), indent=indent), encoding="utf-8")


def load_ruleset_json(path: str | Path) -> ScoredRuleSet:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    return ScoredRuleSet.from_dict(payload)


def dump_ruleset_markdown(ruleset: ScoredRuleSet, path: str | Path) -> None:
    target = Path(path)
    target.write_text(format_ruleset_markdown(ruleset), encoding="utf-8")


def dump_ruleset_latex(ruleset: ScoredRuleSet, path: str | Path) -> None:
    target = Path(path)
    target.write_text(format_ruleset_latex(ruleset), encoding="utf-8")


def dump_archive_json(archive: object, path: str | Path, indent: int = 2) -> None:
    """Serialize a MasterParetoArchive (including all non-dominated ScoredRuleSets) to JSON."""
    target = Path(path)
    to_dict = getattr(archive, "to_dict", None)
    if to_dict is None:
        raise TypeError("Expected an archive object exposing to_dict().")
    target.write_text(json.dumps(to_dict(), indent=indent), encoding="utf-8")


def load_archive_json(path: str | Path):
    """Deserialize a MasterParetoArchive from JSON."""
    from .estimators.auto import MasterParetoArchive

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    return MasterParetoArchive.from_dict(payload)


