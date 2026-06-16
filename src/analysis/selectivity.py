"""
Neuroscience-style selectivity analysis.

Analogous to how neuroscientists identify neurons (or multi-unit sites) that
fire selectively to a particular stimulus identity, we find units (hidden
dimensions) in GPT-2-XL whose activations are significantly modulated by
*which interlocutor* the target is reading.

Metrics implemented:
  1. One-way ANOVA F-statistic & p-value across conditions
  2. Selectivity Index (SI):  (R_max - R_second) / (R_max + R_second)
     — borrowing from visual neuroscience (face-selective cells)
  3. d-prime (d'):  signal-to-noise ratio for binary discrimination
  4. Lifetime sparseness:  how concentrated is a unit's response across conditions
  5. Population Signal-to-Noise ratio (pSNR)

All computations work on numpy arrays and are fully vectorised over the
unit / hidden-dimension axis.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-unit metrics (vectorised over units)
# ---------------------------------------------------------------------------

def compute_condition_means(
    data: np.ndarray,         # (N_samples, N_units)
    labels: np.ndarray,       # (N_samples,)  int condition labels
    n_conditions: int,
) -> np.ndarray:
    """
    Compute mean activation per condition.

    Returns:
        means: (N_conditions, N_units)
    """
    means = np.zeros((n_conditions, data.shape[1]), dtype=np.float32)
    for c in range(n_conditions):
        mask = labels == c
        if mask.sum() > 0:
            means[c] = data[mask].mean(axis=0)
    return means


def anova_selectivity(
    data: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    One-way ANOVA for each unit: does the unit's activation differ
    significantly across interlocutor conditions?

    Returns:
        f_stats: (N_units,) F-statistics
        p_values: (N_units,) p-values (uncorrected)
    """
    conditions = sorted(np.unique(labels).tolist())
    n_units = data.shape[1]
    f_stats = np.zeros(n_units)
    p_values = np.ones(n_units)

    # Group data by condition
    groups = [data[labels == c] for c in conditions]

    # Vectorised ANOVA: scipy.stats.f_oneway operates column-wise
    # We need to handle it unit by unit for varying group sizes, but
    # since groups may have different sizes we do a fast batch via scipy.
    for u in range(n_units):
        g_vecs = [g[:, u] for g in groups if len(g) > 0]
        if len(g_vecs) < 2:
            continue
        try:
            f, p = stats.f_oneway(*g_vecs)
            f_stats[u] = f if np.isfinite(f) else 0.0
            p_values[u] = p if np.isfinite(p) else 1.0
        except Exception:
            pass

    return f_stats, p_values


def selectivity_index(
    means: np.ndarray,    # (N_conditions, N_units)
) -> np.ndarray:
    """
    Selectivity Index (SI) inspired by visual cortex studies.

    SI = (R_max - R_second_max) / (R_max + R_second_max)

    Range: 0 (equal response to all) … 1 (responds only to one condition).
    If only one condition elicits non-zero response → SI ≈ 1.

    Returns:
        si: (N_units,)
    """
    # Shift to be non-negative (activations can be negative after LayerNorm)
    shifted = means - means.min(axis=0, keepdims=True)
    sorted_means = np.sort(shifted, axis=0)[::-1]  # descending
    r_max = sorted_means[0]
    r_second = sorted_means[1] if len(sorted_means) > 1 else np.zeros_like(r_max)
    denom = r_max + r_second
    si = np.where(denom > 1e-8, (r_max - r_second) / denom, 0.0)
    return si.astype(np.float32)


def dprime(
    data: np.ndarray,
    labels: np.ndarray,
    class_a: int,
    class_b: int,
) -> np.ndarray:
    """
    d-prime (sensitivity index) for discriminating condition A from B.

    d' = (mu_A - mu_B) / sqrt(0.5 * (sigma_A^2 + sigma_B^2))

    Returns:
        dp: (N_units,)
    """
    a = data[labels == class_a]
    b = data[labels == class_b]
    mu_a, mu_b = a.mean(0), b.mean(0)
    var_a = a.var(0) + 1e-8
    var_b = b.var(0) + 1e-8
    dp = (mu_a - mu_b) / np.sqrt(0.5 * (var_a + var_b))
    return dp.astype(np.float32)


def lifetime_sparseness(
    means: np.ndarray,    # (N_conditions, N_units)
) -> np.ndarray:
    """
    Lifetime sparseness (Rolls & Tovee 1995):

    S = (1 - (sum r_i / N)^2 / sum(r_i^2 / N)) / (1 - 1/N)

    where r_i are mean responses across conditions (non-negative).
    S = 0 → dense (responds equally to all); S → 1 → very sparse (one condition).

    Returns:
        sparseness: (N_units,)
    """
    # Shift to non-negative
    r = means - means.min(axis=0, keepdims=True)    # (C, U)
    N = r.shape[0]
    sum_r = r.sum(axis=0) / N                        # (U,)
    sum_r2 = (r ** 2).sum(axis=0) / N               # (U,)
    denom = sum_r2 + 1e-8
    s_num = 1.0 - (sum_r ** 2) / denom
    s_denom = 1.0 - 1.0 / N
    sparseness = np.where(s_denom > 1e-8, s_num / s_denom, 0.0)
    return sparseness.clip(0, 1).astype(np.float32)


def preferred_condition(
    means: np.ndarray,    # (N_conditions, N_units)
    label_map: Dict[str, int],
) -> np.ndarray:
    """
    For each unit, return the index of the condition with the highest mean.

    Returns:
        pref: (N_units,)  int array of preferred condition indices
    """
    return np.argmax(means, axis=0).astype(np.int32)


# ---------------------------------------------------------------------------
# Full selectivity report for one layer
# ---------------------------------------------------------------------------

def layer_selectivity_report(
    data: np.ndarray,
    labels: np.ndarray,
    label_map: Dict[str, int],
    fdr_alpha: float = 0.05,
) -> Dict:
    """
    Run all selectivity metrics for a single layer and return a summary dict.

    Args:
        data:      (N, hidden_dim) float32
        labels:    (N,) int32
        label_map: interlocutor_id -> int
        fdr_alpha: FDR threshold for calling units 'selective'

    Returns:
        report dict with per-unit metrics and summary statistics
    """
    n_conditions = len(label_map)
    inv_label_map = {v: k for k, v in label_map.items()}

    means = compute_condition_means(data, labels, n_conditions)
    f_stats, p_values = anova_selectivity(data, labels)
    si = selectivity_index(means)
    ls = lifetime_sparseness(means)
    pref = preferred_condition(means, label_map)

    # FDR correction (Benjamini-Hochberg)
    p_sorted_idx = np.argsort(p_values)
    m = len(p_values)
    bh_threshold = np.arange(1, m + 1) / m * fdr_alpha
    sig_mask = np.zeros(m, dtype=bool)
    for rank, idx in enumerate(p_sorted_idx):
        if p_values[idx] <= bh_threshold[rank]:
            sig_mask[idx] = True
        else:
            break   # BH: once we exceed, all subsequent are also non-significant

    n_selective = sig_mask.sum()

    # Which condition "owns" the most selective units?
    pref_selective = pref[sig_mask] if n_selective > 0 else np.array([])
    condition_counts = {
        inv_label_map[c]: int((pref_selective == c).sum())
        for c in range(n_conditions)
    }

    return {
        # Per-unit arrays (hidden_dim,)
        "f_stats": f_stats,
        "p_values": p_values,
        "p_values_significant": sig_mask,
        "selectivity_index": si,
        "lifetime_sparseness": ls,
        "preferred_condition": pref,
        "condition_means": means,   # (n_conditions, hidden_dim)
        # Summary scalars
        "n_units": data.shape[1],
        "n_selective_units": int(n_selective),
        "fraction_selective": float(n_selective) / data.shape[1],
        "mean_si_all": float(si.mean()),
        "mean_si_selective": float(si[sig_mask].mean()) if n_selective > 0 else 0.0,
        "selective_units_per_condition": condition_counts,
        "label_map": label_map,
        "inv_label_map": inv_label_map,
    }


# ---------------------------------------------------------------------------
# Cross-layer summary
# ---------------------------------------------------------------------------

def run_full_analysis(
    h5_path: str,
    fdr_alpha: float = 0.05,
) -> Dict[str, Dict]:
    """
    Load activation data from HDF5 and run selectivity analysis on every layer.

    Returns:
        dict mapping layer_name -> report_dict
    """
    from src.activation.extractor import load_layer, list_layers, get_label_map

    layers = list_layers(h5_path)
    label_map = get_label_map(h5_path)
    reports: Dict[str, Dict] = {}

    for layer_name in sorted(layers):
        logger.info(f"Analysing layer: {layer_name}")
        data, labels, _ = load_layer(h5_path, layer_name)
        report = layer_selectivity_report(data, labels, label_map, fdr_alpha)
        reports[layer_name] = report
        logger.info(
            f"  {report['n_selective_units']}/{report['n_units']} selective units "
            f"({report['fraction_selective']:.1%}), "
            f"mean SI={report['mean_si_all']:.3f}"
        )

    return reports
