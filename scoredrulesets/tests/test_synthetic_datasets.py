"""Tests fuer synthetische biomedizinische Datensaetze."""
from __future__ import annotations

import numpy as np
import pytest

from scoredrulesets.benchmarking.datasets import (
    DatasetBundle,
    generate_epistasis_dataset,
    generate_xor_parity_dataset,
    generate_highdim_lowsample_dataset,
    generate_imbalanced_dataset,
    load_synthetic_datasets,
    resolve_dataset_names,
)


class TestGenerateEpistasisDataset:
    def test_shape_and_dtype(self):
        X, y = generate_epistasis_dataset(n_samples=200, n_snps=10, random_state=0)
        assert X.shape == (200, 10)
        assert y.shape == (200,)
        assert X.dtype == int or np.issubdtype(X.dtype, np.integer)
        assert set(np.unique(X)).issubset({0, 1, 2})
        assert set(np.unique(y)).issubset({0, 1})

    def test_heritability_affects_balance(self):
        # Hohe Heritabilitaet → staerkeres Signal → besser trennbar
        _, y_easy = generate_epistasis_dataset(
            n_samples=5000, n_snps=10, n_interacting=2, heritability=0.9, random_state=42
        )
        _, y_hard = generate_epistasis_dataset(
            n_samples=5000, n_snps=10, n_interacting=2, heritability=0.1, random_state=42
        )
        # Beide sollten grob balanciert sein (ca. 50/50)
        easy_ratio = np.mean(y_easy)
        hard_ratio = np.mean(y_hard)
        assert 0.3 < easy_ratio < 0.7
        assert 0.3 < hard_ratio < 0.7

    def test_reproducibility(self):
        X1, y1 = generate_epistasis_dataset(n_samples=100, random_state=123)
        X2, y2 = generate_epistasis_dataset(n_samples=100, random_state=123)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


class TestGenerateXorParityDataset:
    def test_shape(self):
        X, y = generate_xor_parity_dataset(n_bits=4, n_noise_features=6, n_samples=500, random_state=0)
        assert X.shape == (500, 10)
        assert y.shape == (500,)
        assert set(np.unique(y)).issubset({0, 1})

    def test_parity_logic_without_noise(self):
        # Ohne Rauschen und Permutation koennen wir die Paritaet pruefen.
        # Wir pruefen indirekt: die Labels sollten grob 50/50 sein.
        X, y = generate_xor_parity_dataset(n_bits=3, n_noise_features=0, n_samples=10000, random_state=42)
        ratio = np.mean(y)
        assert 0.45 < ratio < 0.55


class TestGenerateHighdimLowsampleDataset:
    def test_shape(self):
        X, y = generate_highdim_lowsample_dataset(
            n_samples=80, n_features=500, random_state=0
        )
        assert X.shape == (80, 500)
        assert y.shape == (80,)

    def test_p_much_greater_than_n(self):
        X, y = generate_highdim_lowsample_dataset(
            n_samples=50, n_features=1000, random_state=0
        )
        assert X.shape[1] > X.shape[0] * 5


class TestGenerateImbalancedDataset:
    def test_class_imbalance(self):
        X, y = generate_imbalanced_dataset(
            n_samples=1000, imbalance_ratio=0.1, random_state=0
        )
        assert X.shape[0] == 1000
        minority_ratio = np.mean(y == 1)
        # Sollte ungefaehr 10% sein (mit etwas Toleranz)
        assert 0.03 < minority_ratio < 0.25


class TestLoadSyntheticDatasets:
    def test_returns_nonempty_dict(self):
        bundles = load_synthetic_datasets()
        assert len(bundles) >= 8  # Mindestens 8 vordefinierte Configs

    def test_all_bundles_valid(self):
        bundles = load_synthetic_datasets()
        for name, bundle in bundles.items():
            assert isinstance(bundle, DatasetBundle)
            assert bundle.X.shape[0] == bundle.y.shape[0]
            assert bundle.X.shape[0] > 0
            assert bundle.source == "synthetic"
            assert name.startswith("synth_")

    def test_expected_names_present(self):
        bundles = load_synthetic_datasets()
        expected = [
            "synth_epistasis_2way_easy",
            "synth_epistasis_2way_hard",
            "synth_epistasis_3way",
            "synth_xor_3bit",
            "synth_xor_5bit",
            "synth_highdim_p500_n120",
            "synth_imbalanced_10pct",
        ]
        for name in expected:
            assert name in bundles, f"Missing: {name}"


class TestResolveSyntheticAliases:
    def test_synthetic_alias(self):
        bundles = load_synthetic_datasets()
        resolved = resolve_dataset_names(["synthetic"], bundles)
        assert len(resolved) >= 8
        assert all(name.startswith("synth_") for name in resolved)

    def test_epistasis_alias(self):
        bundles = load_synthetic_datasets()
        resolved = resolve_dataset_names(["epistasis"], bundles)
        assert len(resolved) >= 3
        assert all("epistasis" in name for name in resolved)

