"""Native Interpretable Clustering — External Validation Benchmark.

Complements ``run_native_clustering_benchmark.py`` (which reports the *internal*
Silhouette coefficient) by evaluating RuleEvoCluster, KMeans and GMM against
*ground-truth* class labels using the Adjusted Rand Index (ARI) and Normalized
Mutual Information (NMI). Because these external metrics are computed against
labels that none of the methods optimize, they remove the circularity of
comparing a Silhouette-optimizing clusterer on the Silhouette metric.

The datasets mirror those of ``get_all_datasets`` exactly (same seeds), but each
loader additionally returns the latent/true partition ``y_true``.
"""

import sys
import unittest.mock
from pathlib import Path

# Mock catgen so scoredrulesets.benchmarking doesn't fail (mirrors sibling script)
sys.modules['catgen'] = unittest.mock.MagicMock()
sys.modules['catgen.datasets'] = unittest.mock.MagicMock()

_SCRIPT_DIR = Path(__file__).resolve().parent
_PKG_SRC = _SCRIPT_DIR.parents[2] / "src"
if _PKG_SRC.exists() and str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    load_wine,
    make_blobs,
)
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from scoredrulesets.estimators.evocluster import RuleEvoCluster

RANDOM_STATE = 42


def _load_seeds():
    """Seeds dataset with ground-truth variety labels (OpenML, synthetic fallback)."""
    from sklearn.datasets import fetch_openml
    try:
        data = fetch_openml(name="seeds", version=1, as_frame=False, parser="auto")
        y = np.asarray(data.target)
        # Map string/categorical targets to integer codes
        _, y_int = np.unique(y, return_inverse=True)
        return data.data, y_int, 3, "Seeds"
    except Exception:
        X, y = make_blobs(n_samples=210, n_features=7, centers=3,
                          cluster_std=1.2, random_state=42)
        return X, y, 3, "Seeds (Synthetic)"


def get_labeled_datasets():
    """Return (X, y_true, K, name) tuples mirroring get_all_datasets seeds."""
    datasets = []

    iris = load_iris()
    datasets.append((iris.data, iris.target, 3, "Iris"))

    wine = load_wine()
    datasets.append((wine.data, wine.target, 3, "Wine"))

    datasets.append(_load_seeds())

    cancer = load_breast_cancer()
    datasets.append((cancer.data, cancer.target, 2, "Breast Cancer"))

    X_blobs, y_blobs = make_blobs(n_samples=300, n_features=2, centers=4,
                                  cluster_std=0.85, random_state=42)
    datasets.append((X_blobs, y_blobs, 4, "Synthetic 2D (4 Clusters)"))

    X_aniso, y_aniso = make_blobs(n_samples=300, n_features=2, centers=3,
                                  cluster_std=0.7, random_state=100)
    X_aniso = np.dot(X_aniso, [[0.6, -0.6], [-0.4, 0.8]])
    datasets.append((X_aniso, y_aniso, 3, "Anisotropic 2D (3 Clusters)"))

    return datasets


def run():
    results = []

    for X, y_true, k, ds_name in get_labeled_datasets():
        print(f"\n{'=' * 66}")
        print(f"Dataset: {ds_name} (N={X.shape[0]}, D={X.shape[1]}, K={k})")
        print(f"{'=' * 66}")

        X_scaled = StandardScaler().fit_transform(X)

        def _score(preds):
            try:
                sil = float(silhouette_score(X_scaled, preds))
            except ValueError:
                sil = float("nan")
            return (sil,
                    float(adjusted_rand_score(y_true, preds)),
                    float(normalized_mutual_info_score(y_true, preds)))

        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        sil_km, ari_km, nmi_km = _score(km.fit_predict(X_scaled))

        gmm = GaussianMixture(n_components=k, random_state=RANDOM_STATE)
        sil_gmm, ari_gmm, nmi_gmm = _score(gmm.fit_predict(X_scaled))

        evo = RuleEvoCluster(n_clusters=k, random_state=RANDOM_STATE,
                             max_generations=50, population_size=50)
        t0 = time.time()
        evo.fit(X_scaled)
        t_evo = time.time() - t0
        sil_evo, ari_evo, nmi_evo = _score(evo.labels_)
        atoms_evo = sum(len(r.atoms) for r in evo.ruleset_.rules)

        print(f"KMeans : Sil={sil_km:.3f} ARI={ari_km:.3f} NMI={nmi_km:.3f}")
        print(f"GMM    : Sil={sil_gmm:.3f} ARI={ari_gmm:.3f} NMI={nmi_gmm:.3f}")
        print(f"EvoClus: Sil={sil_evo:.3f} ARI={ari_evo:.3f} NMI={nmi_evo:.3f} "
              f"| Atoms={atoms_evo} | Time={t_evo:.2f}s")

        results.append({
            "Dataset": ds_name,
            "KMeans ARI": ari_km, "KMeans NMI": nmi_km,
            "GMM ARI": ari_gmm, "GMM NMI": nmi_gmm,
            "RuleEvoCluster ARI": ari_evo, "RuleEvoCluster NMI": nmi_evo,
            "RuleEvoCluster Atoms": atoms_evo,
        })

    df = pd.DataFrame(results)
    out_path = _SCRIPT_DIR / "results_cluster" / "native_clustering_external_benchmark.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved CSV to {out_path}")


if __name__ == "__main__":
    run()
