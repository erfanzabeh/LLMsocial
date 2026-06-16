"""
Population-level decoding analysis.

Answers the question: can we decode *which interlocutor* the target is reading
from the population activity vector at each layer?

Methods:
  1. Linear decoder (logistic regression) — like psychophysics population decoding
  2. Cross-validated accuracy per layer
  3. PCA / UMAP geometry of population vectors coloured by condition
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Linear decoder
# ---------------------------------------------------------------------------

def decode_layer(
    data: np.ndarray,     # (N, hidden_dim)
    labels: np.ndarray,   # (N,) int
    n_folds: int = 5,
    max_iter: int = 1000,
    C: float = 1.0,
    random_state: int = 42,
) -> Dict:
    """
    Train and cross-validate a linear decoder (logistic regression)
    to predict interlocutor identity from population activation vectors.

    Returns dict with:
      - mean_accuracy, std_accuracy
      - chance_level (1 / n_conditions)
      - fold_accuracies
    """
    n_conditions = len(np.unique(labels))
    chance = 1.0 / n_conditions

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver="lbfgs",
            random_state=random_state,
        )),
    ])

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipeline, data, labels, cv=cv, scoring="accuracy")

    return {
        "mean_accuracy": float(scores.mean()),
        "std_accuracy": float(scores.std()),
        "fold_accuracies": scores.tolist(),
        "chance_level": chance,
        "above_chance": float(scores.mean()) > chance,
        "n_folds": n_folds,
    }


# ---------------------------------------------------------------------------
# Cross-layer decoding sweep
# ---------------------------------------------------------------------------

def decoding_sweep(
    h5_path: str,
    n_folds: int = 5,
    layer_filter: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Run linear decoding on every layer in the HDF5 file.

    Args:
        h5_path: path to activation HDF5
        n_folds: cross-validation folds
        layer_filter: if given, only process layers whose name contains this string
                      (e.g. "residual" or "mlp_hidden")

    Returns:
        dict mapping layer_name -> decode_result_dict
    """
    from src.activation.extractor import load_layer, list_layers

    layers = list_layers(h5_path)
    if layer_filter:
        layers = [l for l in layers if layer_filter in l]

    results: Dict[str, Dict] = {}
    for layer_name in sorted(layers):
        logger.info(f"Decoding layer: {layer_name}")
        data, labels, label_map = load_layer(h5_path, layer_name)
        result = decode_layer(data, labels, n_folds=n_folds)
        result["label_map"] = label_map
        results[layer_name] = result
        logger.info(
            f"  acc={result['mean_accuracy']:.3f} ± {result['std_accuracy']:.3f} "
            f"(chance={result['chance_level']:.3f})"
        )

    return results


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def pca_population_geometry(
    data: np.ndarray,
    labels: np.ndarray,
    n_components: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project population vectors into PCA space.

    Returns:
        projected: (N, n_components)
        explained_variance_ratio: (n_components,)
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(data)
    pca = PCA(n_components=min(n_components, data.shape[1], data.shape[0]))
    projected = pca.fit_transform(X)
    return projected, pca.explained_variance_ratio_


def umap_population_geometry(
    data: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """
    2-D UMAP projection of population vectors.

    Returns:
        projected: (N, 2)
    """
    try:
        import umap
    except ImportError:
        raise ImportError("Install umap-learn: pip install umap-learn")

    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(data)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state,
    )
    return reducer.fit_transform(X)


# ---------------------------------------------------------------------------
# Representational Similarity Analysis (RSA)
# ---------------------------------------------------------------------------

def compute_rdm(
    condition_means: np.ndarray,   # (N_conditions, N_units)
    metric: str = "correlation",
) -> np.ndarray:
    """
    Compute the Representational Dissimilarity Matrix (RDM).

    Args:
        condition_means: mean activation per condition
        metric: "correlation" | "euclidean" | "cosine"

    Returns:
        rdm: (N_conditions, N_conditions) symmetric dissimilarity matrix
    """
    from scipy.spatial.distance import pdist, squareform
    rdm = squareform(pdist(condition_means, metric=metric))
    return rdm.astype(np.float32)


def rsa_across_layers(
    reports: Dict[str, Dict],
    metric: str = "correlation",
) -> Dict[str, np.ndarray]:
    """
    Compute RDMs for every layer using pre-computed condition means.

    Args:
        reports: output of selectivity.run_full_analysis()

    Returns:
        dict mapping layer_name -> rdm array
    """
    rdms: Dict[str, np.ndarray] = {}
    for layer_name, report in reports.items():
        means = report["condition_means"]   # (n_conditions, hidden_dim)
        rdm = compute_rdm(means, metric)
        rdms[layer_name] = rdm
    return rdms
