import numpy as np
import pytest

from scoredrulesets.benchmarking.datasets import (
    DatasetBundle,
    missing_paper_uci_dataset_keys,
    resolve_dataset_names,
)


def test_resolve_dataset_names_expands_paper_uci_alias():
    registry = {
        "sklearn_iris": DatasetBundle("sklearn_iris", np.zeros((10, 2)), np.zeros(10), "sklearn"),
        "sklearn_wine": DatasetBundle("sklearn_wine", np.zeros((10, 2)), np.zeros(10), "sklearn"),
        "sklearn_breast_cancer": DatasetBundle("sklearn_breast_cancer", np.zeros((10, 2)), np.zeros(10), "sklearn"),
    }

    resolved = resolve_dataset_names(["paper_uci"], registry)

    assert "sklearn_iris" in resolved
    assert "sklearn_wine" in resolved
    assert "sklearn_breast_cancer" in resolved


def test_resolve_dataset_names_keeps_order_and_deduplicates():
    registry = {
        "sklearn_iris": DatasetBundle("sklearn_iris", np.zeros((10, 2)), np.zeros(10), "sklearn"),
        "sklearn_wine": DatasetBundle("sklearn_wine", np.zeros((10, 2)), np.zeros(10), "sklearn"),
        "sklearn_breast_cancer": DatasetBundle("sklearn_breast_cancer", np.zeros((10, 2)), np.zeros(10), "sklearn"),
    }

    resolved = resolve_dataset_names(["sklearn_iris", "paper_uci", "sklearn_iris"], registry)

    assert resolved[0] == "sklearn_iris"
    assert len(resolved) == len(set(resolved))


def test_resolve_dataset_names_paper_uci_strict_raises_when_missing():
    registry = {
        "sklearn_iris": DatasetBundle("sklearn_iris", np.zeros((10, 2)), np.zeros(10), "sklearn"),
    }

    with pytest.raises(ValueError, match="paper_uci strict mode"):
        resolve_dataset_names(["paper_uci"], registry, paper_uci_strict=True)


def test_missing_paper_uci_dataset_keys_reports_unavailable_entries():
    registry = {
        "sklearn_iris": DatasetBundle("sklearn_iris", np.zeros((10, 2)), np.zeros(10), "sklearn"),
        "sklearn_wine": DatasetBundle("sklearn_wine", np.zeros((10, 2)), np.zeros(10), "sklearn"),
    }

    missing = missing_paper_uci_dataset_keys(registry)
    assert "heart_disease" in missing
    assert "car_evaluation" in missing


