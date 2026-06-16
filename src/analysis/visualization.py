"""
Visualization utilities for SocialLLM analysis results.

All functions return matplotlib Figure objects so they can be saved
or displayed in notebooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONDITION_COLORS = plt.cm.tab10.colors


def _condition_palette(n: int) -> List:
    return [CONDITION_COLORS[i % len(CONDITION_COLORS)] for i in range(n)]


def save_fig(fig: plt.Figure, path: str, dpi: int = 150):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Decoding accuracy across layers
# ---------------------------------------------------------------------------

def plot_decoding_accuracy(
    decoding_results: Dict[str, Dict],
    title: str = "Linear Decoding Accuracy Across Layers",
    layer_filter: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart of cross-validated decoding accuracy per layer.
    """
    items = sorted(decoding_results.items())
    if layer_filter:
        items = [(k, v) for k, v in items if layer_filter in k]

    layer_names = [k for k, _ in items]
    accs = [v["mean_accuracy"] for _, v in items]
    stds = [v["std_accuracy"] for _, v in items]
    chance = items[0][1]["chance_level"] if items else 0.25

    # Extract block numbers for x-axis labels
    labels = []
    for ln in layer_names:
        parts = ln.split("_")
        block_num = parts[1] if len(parts) > 1 else ln
        labels.append(block_num)

    fig, ax = plt.subplots(figsize=(14, 4))
    x = np.arange(len(accs))
    bars = ax.bar(x, accs, yerr=stds, capsize=3,
                  color="steelblue", alpha=0.8, ecolor="black")
    ax.axhline(chance, color="red", linestyle="--", linewidth=1.2,
               label=f"Chance ({chance:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Selectivity distribution across layers
# ---------------------------------------------------------------------------

def plot_selectivity_heatmap(
    reports: Dict[str, Dict],
    metric: str = "selectivity_index",
    layer_filter: Optional[str] = None,
    n_top_units: int = 200,
) -> plt.Figure:
    """
    Heatmap: layers (y-axis) x top-N units (x-axis), colour = selectivity metric.
    """
    items = sorted(reports.items())
    if layer_filter:
        items = [(k, v) for k, v in items if layer_filter in k]

    layer_names = [k for k, _ in items]
    # Collect the metric for each layer, taking top N units by that metric
    matrix_rows = []
    for _, report in items:
        vals = report[metric]   # (n_units,)
        top_idx = np.argsort(vals)[-n_top_units:]
        row = vals[top_idx]
        matrix_rows.append(row)

    # Pad rows to same length
    max_len = max(len(r) for r in matrix_rows)
    matrix = np.zeros((len(matrix_rows), max_len), dtype=np.float32)
    for i, row in enumerate(matrix_rows):
        matrix[i, :len(row)] = row

    fig, ax = plt.subplots(figsize=(14, len(layer_names) * 0.35 + 2))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels(layer_names, fontsize=7)
    ax.set_xlabel(f"Top-{n_top_units} units (sorted by {metric})")
    ax.set_title(f"{metric} — top units per layer")
    plt.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Fraction of selective units per layer
# ---------------------------------------------------------------------------

def plot_fraction_selective(
    reports: Dict[str, Dict],
    layer_filter: Optional[str] = None,
) -> plt.Figure:
    """Line plot of the fraction of selective units per layer."""
    items = sorted(reports.items())
    if layer_filter:
        items = [(k, v) for k, v in items if layer_filter in k]

    fracs = [v["fraction_selective"] for _, v in items]
    block_labels = []
    for k, _ in items:
        parts = k.split("_")
        block_labels.append("_".join(parts[:3]))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(range(len(fracs)), fracs, marker="o", color="darkorange", linewidth=2)
    ax.set_xticks(range(len(fracs)))
    ax.set_xticklabels(block_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Fraction of selective units (FDR-corrected)")
    ax.set_ylim(0, max(fracs) * 1.2 + 0.01)
    ax.set_title("Fraction of Interlocutor-Selective Units per Layer")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Tuning curves for most selective units
# ---------------------------------------------------------------------------

def plot_tuning_curves(
    report: Dict,
    layer_name: str,
    n_top: int = 10,
) -> plt.Figure:
    """
    Plot mean activation (tuning curve) across interlocutor conditions
    for the top-N most selective units in a given layer.
    """
    means = report["condition_means"]   # (n_conditions, n_units)
    si = report["selectivity_index"]
    inv_label_map = report["inv_label_map"]
    n_conditions = means.shape[0]
    condition_names = [inv_label_map[i] for i in range(n_conditions)]

    top_unit_idx = np.argsort(si)[-n_top:][::-1]
    colors = _condition_palette(n_conditions)

    fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharey=False)
    axes = axes.flatten()

    for plot_i, unit_idx in enumerate(top_unit_idx):
        ax = axes[plot_i]
        unit_means = means[:, unit_idx]
        bars = ax.bar(range(n_conditions), unit_means, color=colors)
        ax.set_xticks(range(n_conditions))
        ax.set_xticklabels(
            [cn.split("_")[2] if "_" in cn else cn for cn in condition_names],
            rotation=30, ha="right", fontsize=7,
        )
        ax.set_title(f"Unit {unit_idx}\nSI={si[unit_idx]:.3f}", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.5)

    # Hide unused subplots
    for j in range(len(top_unit_idx), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Tuning curves — top {n_top} selective units in {layer_name}")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. PCA / UMAP scatter of population vectors
# ---------------------------------------------------------------------------

def plot_population_geometry(
    projected: np.ndarray,         # (N, 2 or 3)
    labels: np.ndarray,
    label_map: Dict[str, int],
    title: str = "Population geometry",
    method: str = "PCA",
) -> plt.Figure:
    """
    Scatter plot of population vectors coloured by interlocutor condition.
    """
    inv = {v: k for k, v in label_map.items()}
    n_conditions = len(label_map)
    colors = _condition_palette(n_conditions)

    fig, ax = plt.subplots(figsize=(8, 6))
    for c in range(n_conditions):
        mask = labels == c
        label_name = inv.get(c, str(c))
        # Shorten label for legend
        short = label_name.split("_")[2] if label_name.count("_") >= 2 else label_name
        ax.scatter(
            projected[mask, 0], projected[mask, 1],
            c=[colors[c]], label=short, alpha=0.6, s=20, edgecolors="none",
        )

    ax.set_xlabel(f"{method} dim 1")
    ax.set_ylabel(f"{method} dim 2")
    ax.set_title(title)
    ax.legend(markerscale=2, fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. RDM (Representational Dissimilarity Matrix)
# ---------------------------------------------------------------------------

def plot_rdm(
    rdm: np.ndarray,
    condition_names: List[str],
    title: str = "Representational Dissimilarity Matrix",
) -> plt.Figure:
    """Plot a single RDM as a heatmap."""
    short_names = [
        cn.split("_")[2] if cn.count("_") >= 2 else cn
        for cn in condition_names
    ]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(rdm, cmap="RdYlBu_r", vmin=0)
    ax.set_xticks(range(len(short_names)))
    ax.set_yticks(range(len(short_names)))
    ax.set_xticklabels(short_names, rotation=45, ha="right")
    ax.set_yticklabels(short_names)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def plot_rdm_across_layers(
    rdms: Dict[str, np.ndarray],
    condition_names: List[str],
    layer_filter: Optional[str] = "residual",
    n_cols: int = 4,
) -> plt.Figure:
    """
    Grid of RDMs for every layer (filtered by layer_filter).
    """
    items = sorted(rdms.items())
    if layer_filter:
        items = [(k, v) for k, v in items if layer_filter in k]

    n = len(items)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.5))
    axes = np.array(axes).flatten()

    short_names = [
        cn.split("_")[2] if cn.count("_") >= 2 else cn
        for cn in condition_names
    ]

    for i, (layer_name, rdm) in enumerate(items):
        ax = axes[i]
        im = ax.imshow(rdm, cmap="RdYlBu_r", vmin=0)
        block_label = "_".join(layer_name.split("_")[:3])
        ax.set_title(block_label, fontsize=8)
        ax.set_xticks(range(len(short_names)))
        ax.set_yticks(range(len(short_names)))
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(short_names, fontsize=6)

    for j in range(len(items), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("RDMs across layers (residual stream)")
    fig.tight_layout()
    return fig
