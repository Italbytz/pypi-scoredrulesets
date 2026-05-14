from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Sequence


SUPPORTED_OPS = {"==", "!=", "<=", ">", "<", ">=", "in", "not_in", "between"}
SUPPORTED_TASK_TYPES = {"classification", "regression"}
TaskType = Literal["classification", "regression"]


@dataclass
class AggregationSpec:
    type: str = "argmax_sum"
    temperature: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "temperature": float(self.temperature)}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AggregationSpec":
        return cls(
            type=str(payload.get("type", "argmax_sum")),
            temperature=float(payload.get("temperature", 1.0)),
        )


@dataclass
class Atom:
    feature: str | int
    op: str
    value: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.op not in SUPPORTED_OPS:
            raise ValueError(f"Unsupported atom operator: {self.op}")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        payload = {
            "feature": self.feature,
            "op": self.op,
            "value": self.value,
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Atom":
        atom = cls(
            feature=payload["feature"],
            op=str(payload["op"]),
            value=payload.get("value"),
            metadata=dict(payload.get("metadata", {})),
        )
        atom.validate()
        return atom


@dataclass
class Rule:
    atoms: List[Atom]
    scores: List[float]
    rule_id: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "atoms": [atom.to_dict() for atom in self.atoms],
            "scores": [float(v) for v in self.scores],
        }
        if self.rule_id is not None:
            payload["id"] = self.rule_id
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Rule":
        return cls(
            atoms=[Atom.from_dict(a) for a in payload.get("atoms", [])],
            scores=[float(v) for v in payload.get("scores", [])],
            rule_id=payload.get("id"),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class ScoredRuleSet:
    class_labels: List[Any]
    rules: List[Rule]
    task_type: TaskType = "classification"
    feature_names: List[str] = field(default_factory=list)
    aggregation: AggregationSpec = field(default_factory=AggregationSpec)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.task_type not in SUPPORTED_TASK_TYPES:
            raise ValueError(
                f"Unsupported task_type '{self.task_type}'. "
                f"Expected one of {sorted(SUPPORTED_TASK_TYPES)}"
            )

        if self.task_type == "classification":
            if not self.class_labels:
                raise ValueError("class_labels must not be empty for classification")
            n_classes = len(self.class_labels)
            for idx, rule in enumerate(self.rules):
                if len(rule.scores) != n_classes:
                    raise ValueError(
                        f"Rule at index {idx} has {len(rule.scores)} scores; expected {n_classes}"
                    )
            return

        # Regression: one scalar output per rule.
        if self.class_labels:
            raise ValueError("class_labels must be empty for regression")
        for idx, rule in enumerate(self.rules):
            if len(rule.scores) != 1:
                raise ValueError(
                    f"Rule at index {idx} has {len(rule.scores)} scores; expected 1"
                )

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "format": "scoredrulesets",
            "version": "0.1",
            "task_type": self.task_type,
            "class_labels": list(self.class_labels),
            "feature_names": list(self.feature_names),
            "aggregation": self.aggregation.to_dict(),
            "rules": [rule.to_dict() for rule in self.rules],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ScoredRuleSet":
        if payload.get("format") != "scoredrulesets":
            raise ValueError("Unsupported format. Expected format='scoredrulesets'")

        if "task_type" in payload:
            task_type = str(payload["task_type"])
        elif "class_labels" in payload:
            task_type = "classification"
        else:
            task_type = "regression"

        ruleset = cls(
            class_labels=list(payload.get("class_labels", [])),
            task_type=task_type,  # type: ignore[arg-type]
            feature_names=list(payload.get("feature_names", [])),
            aggregation=AggregationSpec.from_dict(payload.get("aggregation", {})),
            rules=[Rule.from_dict(r) for r in payload.get("rules", [])],
            metadata=dict(payload.get("metadata", {})),
        )
        ruleset.validate()
        return ruleset

    @classmethod
    def empty(cls, class_labels: Sequence[Any]) -> "ScoredRuleSet":
        return cls(class_labels=list(class_labels), rules=[])

