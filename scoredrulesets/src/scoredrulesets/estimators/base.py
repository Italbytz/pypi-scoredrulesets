from __future__ import annotations

from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator, ClassifierMixin

from ..schema import ScoredRuleSet


class BaseRuleSetEstimator(BaseEstimator, ClassifierMixin, ABC):
    """Erweiterungspunkt fuer kuenftige native Scored-Rule-Set-Schaetzer."""

    @abstractmethod
    def to_ruleset(self) -> ScoredRuleSet:
        raise NotImplementedError

