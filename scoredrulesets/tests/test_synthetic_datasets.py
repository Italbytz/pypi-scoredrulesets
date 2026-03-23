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
    generate_dnf_concept_dataset,
    generate_checkerboard_dataset,
    generate_monk1_dataset,
    generate_monk3_dataset,
    generate_overlapping_rules_dataset,
    generate_modular_sum_dataset,
    generate_deep_tree_dataset,
    generate_sequential_threshold_dataset,
    generate_hierarchical_interaction_dataset,
    generate_circle_boundary_dataset,
    generate_diagonal_boundary_dataset,
    generate_spiral_dataset,
    generate_concentric_rings_dataset,
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

    def test_cart_hard_alias(self):
        bundles = load_synthetic_datasets()
        resolved = resolve_dataset_names(["cart_hard"], bundles)
        assert len(resolved) >= 10
        expected_prefixes = ("synth_dnf_", "synth_checkerboard_", "synth_monk",
                             "synth_overlap_", "synth_modsum_")
        for name in resolved:
            assert any(name.startswith(p) for p in expected_prefixes), f"Unexpected: {name}"


# ---------------------------------------------------------------------------
# Tests fuer CART-schwierige Datensaetze
# ---------------------------------------------------------------------------


class TestGenerateDnfConceptDataset:
    def test_shape_and_dtype(self):
        X, y = generate_dnf_concept_dataset(n_disjuncts=3, n_conjuncts=2,
                                            n_noise_features=5, n_samples=500, random_state=0)
        assert X.shape == (500, 11)  # 3*2 + 5
        assert y.shape == (500,)
        assert set(np.unique(y)).issubset({0, 1})

    def test_positive_class_present(self):
        X, y = generate_dnf_concept_dataset(n_disjuncts=2, n_conjuncts=2,
                                            n_samples=2000, random_state=42)
        # DNF mit 2 Disjunkten → positive Klasse sollte substantiell vertreten sein
        assert 0.2 < np.mean(y) < 0.9

    def test_noise_flips(self):
        _, y_clean = generate_dnf_concept_dataset(n_disjuncts=2, n_conjuncts=2,
                                                  n_samples=5000, noise_rate=0.0, random_state=42)
        _, y_noisy = generate_dnf_concept_dataset(n_disjuncts=2, n_conjuncts=2,
                                                  n_samples=5000, noise_rate=0.2, random_state=42)
        # Bei 20% Rauschen sollten sich die Labels merklich unterscheiden
        diff_rate = np.mean(y_clean != y_noisy)
        assert 0.1 < diff_rate < 0.35

    def test_reproducibility(self):
        X1, y1 = generate_dnf_concept_dataset(random_state=7)
        X2, y2 = generate_dnf_concept_dataset(random_state=7)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


class TestGenerateCheckerboardDataset:
    def test_shape(self):
        X, y = generate_checkerboard_dataset(n_tiles=4, n_noise_features=6,
                                             n_samples=1000, random_state=0)
        assert X.shape == (1000, 8)  # 2 real + 6 noise
        assert y.shape == (1000,)
        assert set(np.unique(y)).issubset({0, 1})

    def test_balanced(self):
        _, y = generate_checkerboard_dataset(n_tiles=4, n_samples=10000, random_state=42)
        assert 0.45 < np.mean(y) < 0.55


class TestGenerateMonk1Dataset:
    def test_full_enumeration(self):
        X, y = generate_monk1_dataset(n_samples=5000, random_state=0)
        # Volle Enumeration hat 432 Instanzen
        assert X.shape[0] == 432
        assert X.shape[1] == 6

    def test_concept_correct(self):
        X, y = generate_monk1_dataset(n_samples=5000, random_state=0)
        # Manuell pruefen: (a1==a2) OR (a5==1) auf einem Sample
        for i in range(X.shape[0]):
            expected = int((X[i, 0] == X[i, 1]) or (X[i, 4] == 1))
            assert y[i] == expected, f"Row {i}: expected {expected}, got {y[i]}"


class TestGenerateMonk3Dataset:
    def test_shape(self):
        X, y = generate_monk3_dataset(n_samples=5000, noise_rate=0.0, random_state=0)
        assert X.shape[0] == 432
        assert X.shape[1] == 6

    def test_noise_effect(self):
        _, y_clean = generate_monk3_dataset(noise_rate=0.0, random_state=42)
        _, y_noisy = generate_monk3_dataset(noise_rate=0.1, random_state=42)
        diff = np.mean(y_clean != y_noisy)
        assert 0.02 < diff < 0.25


class TestGenerateOverlappingRulesDataset:
    def test_shape(self):
        X, y = generate_overlapping_rules_dataset(n_rules=3, n_features_per_rule=2,
                                                  n_noise_features=5, n_samples=800, random_state=0)
        assert X.shape == (800, 11)  # 3*2 + 5
        assert y.shape == (800,)

    def test_positive_class(self):
        _, y = generate_overlapping_rules_dataset(n_rules=4, n_samples=5000, random_state=42)
        # Mindestens einige Positive erwartet
        assert np.sum(y == 1) > 50


class TestGenerateModularSumDataset:
    def test_shape(self):
        X, y = generate_modular_sum_dataset(n_relevant=4, n_noise_features=6,
                                            n_samples=1000, modulus=3, random_state=0)
        assert X.shape == (1000, 10)
        assert y.shape == (1000,)
        assert set(np.unique(y)).issubset({0, 1})

    def test_approx_balance(self):
        # Bei mod 3 sollte ca. 1/3 positiv sein
        _, y = generate_modular_sum_dataset(n_relevant=4, n_samples=10000,
                                            modulus=3, random_state=42)
        assert 0.25 < np.mean(y) < 0.45


class TestCartHardDatasetsInRegistry:
    def test_all_present(self):
        bundles = load_synthetic_datasets()
        expected = [
            "synth_dnf_3x2", "synth_dnf_5x3", "synth_dnf_3x2_noisy",
            "synth_checkerboard_4x4", "synth_checkerboard_6x6",
            "synth_monk1", "synth_monk3",
            "synth_overlap_4rules", "synth_overlap_6rules",
            "synth_modsum_mod3", "synth_modsum_mod4",
        ]
        for name in expected:
            assert name in bundles, f"Missing CART-hard dataset: {name}"


# ---------------------------------------------------------------------------
# Tests fuer Ruleset-schwierige Datensaetze (CART besser als Rule Sets)
# ---------------------------------------------------------------------------


class TestGenerateDeepTreeDataset:
    def test_shape(self):
        X, y = generate_deep_tree_dataset(depth=4, n_noise_features=5,
                                          n_samples=800, random_state=0)
        assert X.shape == (800, 9)  # 4 relevant + 5 noise
        assert y.shape == (800,)
        assert set(np.unique(y)).issubset({0, 1})

    def test_balanced(self):
        _, y = generate_deep_tree_dataset(depth=5, n_samples=10000, random_state=42)
        assert 0.35 < np.mean(y) < 0.65

    def test_reproducibility(self):
        X1, y1 = generate_deep_tree_dataset(random_state=99)
        X2, y2 = generate_deep_tree_dataset(random_state=99)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


class TestGenerateSequentialThresholdDataset:
    def test_shape(self):
        X, y = generate_sequential_threshold_dataset(n_bins=5, n_noise_features=4,
                                                     n_samples=500, random_state=0)
        assert X.shape == (500, 5)  # 1 relevant + 4 noise
        assert y.shape == (500,)
        assert set(np.unique(y)).issubset({0, 1})

    def test_bins_alternate(self):
        # Mit 4 Bins: Bin 0 → 0, 1 → 1, 2 → 0, 3 → 1 → ca. 50/50
        _, y = generate_sequential_threshold_dataset(n_bins=4, n_samples=10000, random_state=42)
        assert 0.45 < np.mean(y) < 0.55


class TestGenerateHierarchicalInteractionDataset:
    def test_shape(self):
        X, y = generate_hierarchical_interaction_dataset(
            n_context_features=3, n_response_features=3,
            n_noise_features=4, n_samples=500, random_state=0)
        assert X.shape == (500, 8)  # 1 context + 3 response + 4 noise
        assert y.shape == (500,)
        assert set(np.unique(y)).issubset({0, 1})


# ---------------------------------------------------------------------------
# Tests fuer alle-regelbasierten-schwierige Datensaetze (SVM/kNN besser)
# ---------------------------------------------------------------------------


class TestGenerateCircleBoundaryDataset:
    def test_shape(self):
        X, y = generate_circle_boundary_dataset(n_noise_features=5,
                                                n_samples=800, random_state=0)
        assert X.shape == (800, 7)  # 2 real + 5 noise
        assert y.shape == (800,)
        assert set(np.unique(y)).issubset({0, 1})

    def test_both_classes_present(self):
        _, y = generate_circle_boundary_dataset(n_samples=2000, random_state=42)
        assert np.sum(y == 0) > 100
        assert np.sum(y == 1) > 100


class TestGenerateDiagonalBoundaryDataset:
    def test_shape(self):
        X, y = generate_diagonal_boundary_dataset(n_relevant=4, n_noise_features=6,
                                                  n_samples=1000, random_state=0)
        assert X.shape == (1000, 10)
        assert y.shape == (1000,)

    def test_balanced(self):
        _, y = generate_diagonal_boundary_dataset(n_relevant=4, n_samples=10000, random_state=42)
        assert 0.40 < np.mean(y) < 0.60


class TestGenerateSpiralDataset:
    def test_shape(self):
        X, y = generate_spiral_dataset(n_samples=600, n_noise_features=4, random_state=0)
        assert X.shape == (600, 6)  # 2 real + 4 noise
        assert y.shape == (600,)
        assert set(np.unique(y)).issubset({0, 1})

    def test_balanced(self):
        _, y = generate_spiral_dataset(n_samples=2000, random_state=42)
        assert 0.45 < np.mean(y) < 0.55


class TestGenerateConcentricRingsDataset:
    def test_shape(self):
        X, y = generate_concentric_rings_dataset(n_rings=3, n_noise_features=4,
                                                 n_samples=900, random_state=0)
        assert X.shape == (900, 6)  # 2 real + 4 noise
        assert y.shape == (900,)
        assert set(np.unique(y)).issubset({0, 1})


class TestRulesetHardDatasetsInRegistry:
    def test_all_present(self):
        bundles = load_synthetic_datasets()
        expected = [
            "synth_deeptree_d5", "synth_deeptree_d7",
            "synth_seqthresh_5bin", "synth_seqthresh_8bin",
            "synth_hierarch_3x3", "synth_hierarch_5x5",
        ]
        for name in expected:
            assert name in bundles, f"Missing ruleset-hard dataset: {name}"

    def test_alias(self):
        bundles = load_synthetic_datasets()
        resolved = resolve_dataset_names(["ruleset_hard"], bundles)
        assert len(resolved) >= 6
        expected_prefixes = ("synth_deeptree_", "synth_seqthresh_", "synth_hierarch_")
        for name in resolved:
            assert any(name.startswith(p) for p in expected_prefixes), f"Unexpected: {name}"


class TestRuleHardDatasetsInRegistry:
    def test_all_present(self):
        bundles = load_synthetic_datasets()
        expected = [
            "synth_circle", "synth_circle_noisy",
            "synth_diagonal_4d", "synth_diagonal_8d",
            "synth_spiral",
            "synth_rings_3", "synth_rings_5",
        ]
        for name in expected:
            assert name in bundles, f"Missing rule-hard dataset: {name}"

    def test_alias(self):
        bundles = load_synthetic_datasets()
        resolved = resolve_dataset_names(["rule_hard"], bundles)
        assert len(resolved) >= 7
        expected_prefixes = ("synth_circle", "synth_diagonal_", "synth_spiral", "synth_rings_")
        for name in resolved:
            assert any(name.startswith(p) for p in expected_prefixes), f"Unexpected: {name}"


