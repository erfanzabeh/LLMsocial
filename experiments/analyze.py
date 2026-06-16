"""
Step 2 — Run neuroscience-style analysis on the collected activation dataset.

Usage:
    python experiments/analyze.py [--config config/config.yaml]
                                  [--h5 data/activations/activations.h5]
                                  [--skip_decoding]
                                  [--skip_umap]
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.activation.extractor import load_layer, list_layers, get_label_map
from src.analysis.selectivity import run_full_analysis
from src.analysis.decoding import decoding_sweep, pca_population_geometry, rsa_across_layers
from src.analysis.visualization import (
    plot_decoding_accuracy,
    plot_fraction_selective,
    plot_selectivity_heatmap,
    plot_tuning_curves,
    plot_population_geometry,
    plot_rdm_across_layers,
    save_fig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--h5", default=None, help="Override HDF5 path")
    p.add_argument("--skip_decoding", action="store_true")
    p.add_argument("--skip_umap", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    h5_path = args.h5 or cfg["activation"]["output_path"]
    figs_dir = cfg["output"]["figures_dir"]
    results_dir = cfg["output"]["results_dir"]
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    Path(figs_dir).mkdir(parents=True, exist_ok=True)

    label_map = get_label_map(h5_path)
    condition_names = [k for k, _ in sorted(label_map.items(), key=lambda x: x[1])]
    logger.info(f"Conditions: {condition_names}")

    # ------------------------------------------------------------------
    # 1. Selectivity analysis
    # ------------------------------------------------------------------
    logger.info("=== Running selectivity analysis ===")
    reports = run_full_analysis(h5_path, fdr_alpha=cfg["analysis"]["fdr_alpha"])

    # Save reports (without large numpy arrays for JSON; use pickle for full)
    with open(f"{results_dir}/selectivity_reports.pkl", "wb") as f:
        pickle.dump(reports, f)

    # JSON summary (scalars only)
    summary = {}
    for layer_name, r in reports.items():
        summary[layer_name] = {
            "n_units": r["n_units"],
            "n_selective_units": r["n_selective_units"],
            "fraction_selective": r["fraction_selective"],
            "mean_si_all": r["mean_si_all"],
            "mean_si_selective": r["mean_si_selective"],
            "selective_units_per_condition": r["selective_units_per_condition"],
        }
    with open(f"{results_dir}/selectivity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Figures
    fig = plot_fraction_selective(reports, layer_filter="residual")
    save_fig(fig, f"{figs_dir}/fraction_selective_residual.png")

    fig = plot_fraction_selective(reports, layer_filter="mlp_hidden")
    save_fig(fig, f"{figs_dir}/fraction_selective_mlp.png")

    fig = plot_selectivity_heatmap(reports, metric="selectivity_index",
                                   layer_filter="residual")
    save_fig(fig, f"{figs_dir}/si_heatmap_residual.png")

    # Tuning curves for the layer with the most selective units
    best_layer = max(
        reports.keys(),
        key=lambda k: reports[k]["fraction_selective"]
    )
    logger.info(f"Most selective layer: {best_layer}")
    fig = plot_tuning_curves(
        reports[best_layer], best_layer,
        n_top=cfg["analysis"]["n_top_units_plot"]
    )
    save_fig(fig, f"{figs_dir}/tuning_curves_best_layer.png")

    # ------------------------------------------------------------------
    # 2. RSA
    # ------------------------------------------------------------------
    logger.info("=== Computing RDMs ===")
    rdms = rsa_across_layers(reports)
    fig = plot_rdm_across_layers(rdms, condition_names, layer_filter="residual")
    save_fig(fig, f"{figs_dir}/rdms_residual.png")

    # ------------------------------------------------------------------
    # 3. Population decoding
    # ------------------------------------------------------------------
    if not args.skip_decoding:
        logger.info("=== Running population decoding ===")
        decode_results = decoding_sweep(
            h5_path,
            n_folds=cfg["analysis"]["n_decoding_folds"],
            layer_filter="residual",
        )
        with open(f"{results_dir}/decoding_results.json", "w") as f:
            json.dump(
                {k: {kk: vv for kk, vv in v.items() if kk != "label_map"}
                 for k, v in decode_results.items()},
                f, indent=2
            )

        fig = plot_decoding_accuracy(decode_results, layer_filter="residual")
        save_fig(fig, f"{figs_dir}/decoding_accuracy_residual.png")

    # ------------------------------------------------------------------
    # 4. PCA geometry for the best layer
    # ------------------------------------------------------------------
    logger.info(f"=== PCA geometry for {best_layer} ===")
    data, labels, lm = load_layer(h5_path, best_layer)
    projected, evr = pca_population_geometry(data, labels, n_components=10)
    logger.info(f"  Explained variance (first 3 PCs): {evr[:3]}")

    fig = plot_population_geometry(
        projected[:, :2], labels, label_map,
        title=f"Population geometry — {best_layer}", method="PCA"
    )
    save_fig(fig, f"{figs_dir}/pca_geometry_best_layer.png")

    # ------------------------------------------------------------------
    # 5. Optional UMAP
    # ------------------------------------------------------------------
    if not args.skip_umap:
        try:
            from src.analysis.decoding import umap_population_geometry
            logger.info("=== UMAP geometry ===")
            umap_proj = umap_population_geometry(data, labels)
            fig = plot_population_geometry(
                umap_proj, labels, label_map,
                title=f"UMAP geometry — {best_layer}", method="UMAP"
            )
            save_fig(fig, f"{figs_dir}/umap_geometry_best_layer.png")
        except ImportError:
            logger.warning("umap-learn not installed — skipping UMAP.")

    logger.info(f"Analysis complete. Figures saved to {figs_dir}/")
    logger.info(f"Results saved to {results_dir}/")


if __name__ == "__main__":
    main()
