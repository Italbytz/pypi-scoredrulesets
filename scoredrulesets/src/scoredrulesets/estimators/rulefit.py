"""
RuleFit-Klassifikator-Wrapper und Transformation zu ScoredRuleSet.

RuleFit (Friedman & Popescu, 2008) ist ein Regressionsmodell, das
Regeln aus Entscheidungsbäumen extrahiert und per Lasso-Regression
gewichtet. Dieser Wrapper macht es sklearn-kompatibel für Klassifikation.

Binäre Klassifikation:
  - Einzelnes RuleFit-Modell, Threshold bei 0.5
  - Score = intercept + sum(coef_i * rule_i(x))

Multi-Klasse (One-vs-Rest):
  - Ein RuleFit-Modell pro Klasse
  - Prediction = argmax der OvR-Scores
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ..schema import AggregationSpec, Atom, Rule, ScoredRuleSet


def _import_rulefit():
    """Importiere RuleFit mit aussagekräftiger Fehlermeldung."""
    try:
        from rulefit import RuleFit
        return RuleFit
    except ImportError as e:
        raise ImportError(
            "backend='rulefit' benötigt das Paket 'rulefit'. "
            "Installiere mit: pip install rulefit"
        ) from e


class RuleFitClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-kompatibler Klassifikator basierend auf RuleFit.

    Binäre Klassifikation: Ein einzelnes RuleFit-Modell.
    Multi-Klasse: One-vs-Rest mit einem RuleFit-Modell pro Klasse.
    """

    def __init__(
        self,
        max_rules: int = 50,
        tree_size: int = 4,
        random_state: int | None = None,
        rfmode: str = "classify",
        lin_trim_quantile: float = 0.025,
        memory_par: float = 0.01,
    ):
        self.max_rules = max_rules
        self.tree_size = tree_size
        self.random_state = random_state
        self.rfmode = rfmode
        self.lin_trim_quantile = lin_trim_quantile
        self.memory_par = memory_par

    def fit(self, X, y):
        X_valid, y_valid = check_X_y(X, y, dtype="numeric")
        self.classes_ = unique_labels(y_valid)
        self.n_features_in_ = X_valid.shape[1]
        self.feature_names_in_ = [f"f{i}" for i in range(self.n_features_in_)]

        RuleFit = _import_rulefit()

        if len(self.classes_) == 2:
            # Binäre Klassifikation
            self._is_binary_ = True
            y_bin = (y_valid == self.classes_[1]).astype(float)
            self._models_ = [self._build_rulefit()]
            self._models_[0].fit(X_valid, y_bin, feature_names=self.feature_names_in_)
        else:
            # Multi-Klasse: One-vs-Rest
            self._is_binary_ = False
            self._models_ = []
            for cls in self.classes_:
                y_ovr = (y_valid == cls).astype(float)
                model = self._build_rulefit()
                model.fit(X_valid, y_ovr, feature_names=self.feature_names_in_)
                self._models_.append(model)

        return self

    def predict(self, X):
        check_is_fitted(self, "_models_")
        X_valid = check_array(X, dtype="numeric")
        proba = self.predict_proba(X_valid)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

    def predict_proba(self, X):
        check_is_fitted(self, "_models_")
        X_valid = check_array(X, dtype="numeric")

        if self._is_binary_:
            raw = self._models_[0].predict(X_valid)
            p1 = np.clip(raw, 0.0, 1.0)
            return np.column_stack([1.0 - p1, p1])
        else:
            # Multi-Klasse: OvR Scores
            raw_scores = np.column_stack(
                [model.predict(X_valid) for model in self._models_]
            )
            # Normalisiere zu Wahrscheinlichkeiten via Softmax
            raw_scores = np.clip(raw_scores, -10, 10)
            exp_scores = np.exp(raw_scores - np.max(raw_scores, axis=1, keepdims=True))
            return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def get_rules(self, class_index: int = 0):
        """Gibt die Regeln des Modells als DataFrame zurück."""
        check_is_fitted(self, "_models_")
        return self._models_[class_index].get_rules()

    def to_ruleset(self) -> ScoredRuleSet:
        """Konvertiert das trainierte Modell in ein ScoredRuleSet."""
        check_is_fitted(self, "_models_")
        return rulefit_to_scored_ruleset(self)

    def _build_rulefit(self):
        RuleFit = _import_rulefit()
        return RuleFit(
            max_rules=self.max_rules,
            tree_size=self.tree_size,
            random_state=self.random_state,
            lin_trim_quantile=self.lin_trim_quantile,
            memory_par=self.memory_par,
        )


# ---------------------------------------------------------------------------
# Regel-Parsing: RuleFit Regel-Strings → Atome
# ---------------------------------------------------------------------------

_RULE_PATTERN = re.compile(
    r"(\w+)\s*(<=|>=|<|>)\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)


def _parse_rule_string(rule_str: str, feature_names: list[str]) -> list[Atom]:
    """Parst einen RuleFit-Regelstring wie 'f2 > 2.5 & f3 <= 1.75' in Atome."""
    atoms: list[Atom] = []
    # Trenne an ' & ' (RuleFit-Format)
    parts = re.split(r"\s*&\s*", rule_str.strip())
    for part in parts:
        match = _RULE_PATTERN.match(part.strip())
        if match:
            fname = match.group(1).strip()
            op = match.group(2).strip()
            val = float(match.group(3))
            atoms.append(Atom(feature=fname, op=op, value=val))
    return atoms


# ---------------------------------------------------------------------------
# Transformation: RuleFitClassifier → ScoredRuleSet
# ---------------------------------------------------------------------------

def rulefit_to_scored_ruleset(estimator: RuleFitClassifier) -> ScoredRuleSet:
    """Transformiere trainiertes RuleFitClassifier-Modell zu ScoredRuleSet.

    Regeln mit type='rule' werden in Atome geparst.
    Regeln mit type='linear' werden ignoriert (nicht als Atome darstellbar).
    Regeln mit coef≈0 werden übersprungen.
    """
    n_classes = len(estimator.classes_)
    feature_names = list(estimator.feature_names_in_)
    rules: list[Rule] = []
    coef_threshold = 1e-8  # Ignoriere Regeln mit sehr kleinem Koeffizienten

    if estimator._is_binary_:
        # Binäre Klassifikation: Ein Modell
        model = estimator._models_[0]
        rules_df = model.get_rules()
        intercept = float(model.intercept_)

        # Default-Regel aus Intercept
        # Für argmax_sum: Verschiebe Intercept relativ zu Threshold 0.5
        bias = intercept - 0.5
        if bias >= 0:
            default_scores = [0.0, bias]
        else:
            default_scores = [abs(bias), 0.0]
        rules.append(Rule(
            atoms=[],
            scores=default_scores,
            rule_id="rulefit_default",
            metadata={"source": "rulefit", "kind": "intercept", "intercept": intercept},
        ))

        # Regeln
        for idx, row in rules_df.iterrows():
            if row["type"] != "rule":
                continue  # Überspringe Linear-Terms
            coef = float(row["coef"])
            if abs(coef) < coef_threshold:
                continue

            atoms = _parse_rule_string(str(row["rule"]), feature_names)
            if not atoms:
                continue

            # Score-Vektor: positiver Koeffizient → Klasse 1, negativer → Klasse 0
            if coef >= 0:
                scores = [0.0, coef]
            else:
                scores = [abs(coef), 0.0]

            rules.append(Rule(
                atoms=atoms,
                scores=scores,
                rule_id=f"rulefit_{idx}",
                metadata={
                    "source": "rulefit",
                    "kind": "rule",
                    "coef": coef,
                    "support": float(row["support"]),
                    "importance": float(row["importance"]),
                    "rule_string": str(row["rule"]),
                },
            ))

    else:
        # Multi-Klasse: One-vs-Rest
        # Default-Regel: Summe der Intercepts
        default_scores = [0.0] * n_classes
        for class_idx, model in enumerate(estimator._models_):
            intercept = float(model.intercept_)
            default_scores[class_idx] = max(intercept, 0.0)
        rules.append(Rule(
            atoms=[],
            scores=default_scores,
            rule_id="rulefit_default",
            metadata={"source": "rulefit", "kind": "intercept_ovr"},
        ))

        # Regeln pro Klasse
        rule_counter = 0
        for class_idx, model in enumerate(estimator._models_):
            rules_df = model.get_rules()
            for _, row in rules_df.iterrows():
                if row["type"] != "rule":
                    continue
                coef = float(row["coef"])
                if abs(coef) < coef_threshold:
                    continue

                atoms = _parse_rule_string(str(row["rule"]), feature_names)
                if not atoms:
                    continue

                scores = [0.0] * n_classes
                scores[class_idx] = max(coef, 0.0)

                rules.append(Rule(
                    atoms=atoms,
                    scores=scores,
                    rule_id=f"rulefit_{rule_counter}",
                    metadata={
                        "source": "rulefit",
                        "kind": "rule_ovr",
                        "class_index": class_idx,
                        "coef": coef,
                        "support": float(row["support"]),
                        "importance": float(row["importance"]),
                        "rule_string": str(row["rule"]),
                    },
                ))
                rule_counter += 1

    ruleset = ScoredRuleSet(
        class_labels=estimator.classes_.tolist(),
        feature_names=feature_names,
        rules=rules,
        aggregation=AggregationSpec(type="argmax_sum", temperature=1.0),
        metadata={
            "transform": "rulefit_to_scored_ruleset",
            "n_rules": len(rules),
            "is_binary": estimator._is_binary_,
        },
    )
    ruleset.validate()
    return ruleset


