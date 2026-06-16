"""
Activation extraction and storage.

Converts raw per-turn activation snapshots (from Interaction objects) into
a structured dataset stored as HDF5, ready for neuroscience-style analysis.

Dataset structure (HDF5):
  /layer_name/
    data      float32  (N_samples, hidden_dim)
    labels    int32    (N_samples,)   index into label_map
    label_map attrs    JSON string -> label_map dict

where label_map maps interlocutor_id (str) -> int.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Aggregation strategies for converting (n_tokens, dim) → (dim,)
# ---------------------------------------------------------------------------

def aggregate_tokens(
    tensor: torch.Tensor,
    strategy: str = "mean",
) -> np.ndarray:
    """
    Collapse the token dimension so each reading event becomes one vector.

    Args:
        tensor: shape (n_tokens, hidden_dim)
        strategy: "mean" | "last" | "max"

    Returns:
        numpy array of shape (hidden_dim,)
    """
    if strategy == "mean":
        vec = tensor.float().mean(dim=0)
    elif strategy == "last":
        vec = tensor.float()[-1]
    elif strategy == "max":
        vec = tensor.float().max(dim=0).values
    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy}")
    return vec.numpy()


# ---------------------------------------------------------------------------
# ActivationDataset builder
# ---------------------------------------------------------------------------

class ActivationDataset:
    """
    Collects activation vectors from multiple interactions and writes them
    to an HDF5 file.

    Usage::

        ds = ActivationDataset("data/activations/dataset.h5")
        ds.add_from_interactions(all_interactions, aggregation="mean")
        ds.save()
    """

    def __init__(self, output_path: str, aggregation: str = "mean"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.aggregation = aggregation

        # interlocutor_id -> int
        self._label_map: Dict[str, int] = {}
        # layer_name -> list of (vector, label_int)
        self._buffer: Dict[str, List[Tuple[np.ndarray, int]]] = {}

    # ------------------------------------------------------------------

    def _get_or_add_label(self, interlocutor_id: str) -> int:
        if interlocutor_id not in self._label_map:
            self._label_map[interlocutor_id] = len(self._label_map)
        return self._label_map[interlocutor_id]

    # ------------------------------------------------------------------

    def add_from_interactions(self, interactions) -> "ActivationDataset":
        """
        Process a list of Interaction objects (with activation_snapshots)
        and buffer the vectors.
        """
        from src.social.environment import Interaction  # avoid circular at module level

        for interaction in interactions:
            label_int = self._get_or_add_label(interaction.interlocutor_id)
            for snapshot in interaction.activation_snapshots:
                for key, val in snapshot.items():
                    if key == "turn_idx":
                        continue
                    if not isinstance(val, torch.Tensor):
                        continue
                    vec = aggregate_tokens(val, self.aggregation)
                    if key not in self._buffer:
                        self._buffer[key] = []
                    self._buffer[key].append((vec, label_int))

        return self

    def add_snapshot(
        self,
        activations: Dict[str, torch.Tensor],
        interlocutor_id: str,
    ) -> "ActivationDataset":
        """Add a single activation snapshot (from one reading event)."""
        label_int = self._get_or_add_label(interlocutor_id)
        for layer_name, tensor in activations.items():
            vec = aggregate_tokens(tensor, self.aggregation)
            if layer_name not in self._buffer:
                self._buffer[layer_name] = []
            self._buffer[layer_name].append((vec, label_int))
        return self

    # ------------------------------------------------------------------

    def save(self):
        """Write the buffered data to HDF5."""
        label_map_json = json.dumps(self._label_map)

        with h5py.File(self.output_path, "w") as f:
            f.attrs["label_map"] = label_map_json
            f.attrs["aggregation"] = self.aggregation
            f.attrs["n_conditions"] = len(self._label_map)

            for layer_name, samples in self._buffer.items():
                # Filter out samples with shape mismatch
                if not samples:
                    logger.warning(f"Skipping layer {layer_name}: no samples")
                    continue
                
                # Get the expected shape from the first sample
                expected_shape = samples[0][0].shape
                
                # Filter samples to only those matching the expected shape
                valid_samples = [s for s in samples if s[0].shape == expected_shape]
                
                if len(valid_samples) < len(samples):
                    logger.warning(
                        f"Filtered {layer_name}: kept {len(valid_samples)}/{len(samples)} samples "
                        f"(discarded {len(samples) - len(valid_samples)} with inconsistent shapes)"
                    )
                
                if not valid_samples:
                    logger.warning(f"Skipping layer {layer_name}: no valid samples after filtering")
                    continue
                
                vecs = np.stack([s[0] for s in valid_samples], axis=0).astype(np.float32)
                labels = np.array([s[1] for s in valid_samples], dtype=np.int32)

                grp = f.create_group(layer_name)
                grp.create_dataset("data", data=vecs, compression="gzip")
                grp.create_dataset("labels", data=labels)
                grp.attrs["label_map"] = label_map_json
                grp.attrs["n_samples"] = len(valid_samples)
                grp.attrs["hidden_dim"] = vecs.shape[1]

        logger.info(f"Saved activation dataset: {self.output_path}")
        logger.info(f"  Conditions: {self._label_map}")
        logger.info(f"  Layers: {len(self._buffer)}")
        for ln, s in self._buffer.items():
            if s:
                logger.info(f"    {ln}: {len(s)} samples, dim={s[0][0].shape[0]}")


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_layer(
    h5_path: str,
    layer_name: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Load data for one layer from a saved HDF5 activation dataset.

    Returns:
        data:      float32 array (N, hidden_dim)
        labels:    int32 array (N,)
        label_map: dict mapping interlocutor_id -> int label
    """
    with h5py.File(h5_path, "r") as f:
        grp = f[layer_name]
        data = grp["data"][:]
        labels = grp["labels"][:]
        label_map = json.loads(grp.attrs["label_map"])
    return data, labels, label_map


def list_layers(h5_path: str) -> List[str]:
    """Return all layer names stored in an HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        return list(f.keys())


def get_label_map(h5_path: str) -> Dict[str, int]:
    """Return the interlocutor_id -> int label mapping."""
    with h5py.File(h5_path, "r") as f:
        return json.loads(f.attrs["label_map"])
