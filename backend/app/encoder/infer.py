"""Deterministic inference for a trained encoder checkpoint."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .data import eeg_to_feature_sequence
from .model import EEGStateEncoder

DEFAULT_CHECKPOINT = Path("models/encoder/synthetic_demo.pt")


def load_encoder(checkpoint_path: str | Path = DEFAULT_CHECKPOINT, device: str = "cpu") -> tuple[EEGStateEncoder, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = EEGStateEncoder(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


@torch.inference_mode()
def infer_latent(raw_eeg: np.ndarray, sample_rate: float, checkpoint_path: str | Path = DEFAULT_CHECKPOINT) -> np.ndarray:
    """Return a stable latent vector from raw EEG using checkpoint normalisation."""
    model, checkpoint = load_encoder(checkpoint_path)
    features = eeg_to_feature_sequence(raw_eeg, sample_rate)
    if features.shape[1:] != (model.channels, model.bands):
        raise ValueError(f"Checkpoint expects {model.channels} channels and {model.bands} bands; received {features.shape[1:]}.")
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    normalised = (features - mean) / std
    return model(torch.from_numpy(normalised).unsqueeze(0))["latent"].squeeze(0).cpu().numpy()

