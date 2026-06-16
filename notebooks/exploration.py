"""
Interactive exploration script (also usable as a Jupyter notebook via jupytext).
Run sections manually or convert with: jupytext --to notebook notebooks/exploration.py

Sections:
  A. Quick sanity check — single interaction + activation extraction
  B. Load saved activation dataset and inspect
  C. Per-unit selectivity (tuning curves, SI, ANOVA)
  D. Population decoding
  E. PCA geometry
"""

# %%
import sys
sys.path.insert(0, "..")   # run from notebooks/ directory

import torch
import numpy as np
import matplotlib.pyplot as plt

# %%  A. Quick sanity check ------------------------------------------------
# Load models (use tiny GPT-2 for speed)

from src.social.agents import AgentConfig, LLMAgent, TargetAgent

target_cfg = AgentConfig(
    model_name="gpt2",          # use gpt2 (not XL) for quick test
    agent_id="target_gpt2",
    temperature=0.8,
    max_new_tokens=64,
    device="cpu",
)
target = TargetAgent(target_cfg)
print("Target loaded:", target.agent_id)

interlocutor_cfg = AgentConfig(
    model_name="gpt2",
    agent_id="interlocutor_a",
    persona="You are a curious assistant.",
    temperature=0.9,
    max_new_tokens=64,
    device="cpu",
)
interlocutor = LLMAgent(interlocutor_cfg)
print("Interlocutor loaded:", interlocutor.agent_id)

# %%
# Run a single reading + extraction
test_text = "Can you explain how neurons process information in the brain?"
acts = target.read_and_extract(test_text, interlocutor_id="interlocutor_a")

print("Captured activation layers:")
for name, tensor in acts.items():
    print(f"  {name}: shape={tuple(tensor.shape)}, "
          f"mean={tensor.mean():.4f}, std={tensor.std():.4f}")

# %%  B. Load activation dataset -------------------------------------------
from src.activation.extractor import load_layer, list_layers, get_label_map

H5_PATH = "../data/activations/activations.h5"  # run run_interactions.py first

label_map = get_label_map(H5_PATH)
print("Label map:", label_map)

layers = list_layers(H5_PATH)
print(f"Layers available ({len(layers)}):", layers[:6], "...")

# %%  C. Selectivity analysis -----------------------------------------------
from src.analysis.selectivity import layer_selectivity_report

# Pick one layer to inspect
layer_name = "block_20_residual"
data, labels, lm = load_layer(H5_PATH, layer_name)
print(f"Layer {layer_name}: data shape = {data.shape}")

report = layer_selectivity_report(data, labels, label_map, fdr_alpha=0.05)
print(f"  Selective units: {report['n_selective_units']} / {report['n_units']} "
      f"({report['fraction_selective']:.1%})")
print(f"  Mean SI: {report['mean_si_all']:.4f}")
print(f"  Selective units per condition: {report['selective_units_per_condition']}")

# %%
from src.analysis.visualization import plot_tuning_curves, save_fig

fig = plot_tuning_curves(report, layer_name, n_top=10)
plt.show()

# %%  D. Population decoding ------------------------------------------------
from src.analysis.decoding import decode_layer

result = decode_layer(data, labels, n_folds=5)
print(f"Decoding accuracy: {result['mean_accuracy']:.3f} ± {result['std_accuracy']:.3f}")
print(f"Chance level:      {result['chance_level']:.3f}")

# %%  E. PCA geometry -------------------------------------------------------
from src.analysis.decoding import pca_population_geometry
from src.analysis.visualization import plot_population_geometry

projected, evr = pca_population_geometry(data, labels)
print("Explained variance (first 3 PCs):", evr[:3])

fig = plot_population_geometry(
    projected[:, :2], labels, label_map,
    title=f"PCA — {layer_name}", method="PCA"
)
plt.show()
