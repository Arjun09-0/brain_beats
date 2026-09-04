"""Train the Phase 2 EEG encoder on synthetic data or supplied datasets."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .data import EEGFeatureDataset, make_deap_dataset, make_seed_dataset, make_synthetic_dataset
from .model import EEGStateEncoder

DEFAULT_CHECKPOINT = Path("models/encoder/synthetic_demo.pt")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _normalise(dataset: EEGFeatureDataset, train_indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    train_values = dataset.features[train_indices]
    mean = train_values.mean(axis=(0, 1), keepdims=True)
    std = train_values.std(axis=(0, 1), keepdims=True) + 1e-6
    dataset.features = (dataset.features - mean) / std
    return mean.squeeze(0), std.squeeze(0)


def train_dataset(dataset: EEGFeatureDataset, epochs: int = 18, batch_size: int = 16, seed: int = 123, checkpoint_path: str | Path = DEFAULT_CHECKPOINT, source: str = "synthetic mock_stream presets only; not DEAP or SEED") -> dict:
    """Train and persist an encoder from a prepared feature dataset."""
    set_seed(seed)
    split = max(2, int(len(dataset) * 0.8))
    train_indices, val_indices = list(range(split)), list(range(split, len(dataset)))
    mean, std = _normalise(dataset, train_indices)
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False)
    model = EEGStateEncoder(channels=dataset.features.shape[2], bands=dataset.features.shape[3])
    optimiser = torch.optim.Adam(model.parameters(), lr=2e-3)
    criterion = nn.BCEWithLogitsLoss()
    losses: list[float] = []
    for _ in range(epochs):
        model.train()
        running, count = 0.0, 0
        for features, labels in train_loader:
            output = model(features)
            loss = criterion(output["valence_logits"], labels[:, 0]) + criterion(output["arousal_logits"], labels[:, 1])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running += loss.item() * len(features)
            count += len(features)
        losses.append(running / count)
    model.eval()
    correct, total = 0, 0
    with torch.inference_mode():
        for features, labels in val_loader:
            output = model(features)
            predictions = torch.stack((output["valence_logits"] > 0, output["arousal_logits"] > 0), dim=1)
            correct += (predictions == labels.bool()).sum().item()
            total += labels.numel()
    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": {"channels": model.channels, "bands": model.bands, "latent_dim": model.latent_dim},
        "feature_mean": mean,
        "feature_std": std,
        "source": source,
        "seed": seed,
    }
    destination = Path(checkpoint_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)
    metrics = {"train_loss": losses, "validation_label_accuracy": correct / total, "checkpoint": str(destination)}
    destination.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def train_synthetic(epochs: int = 18, samples: int = 64, batch_size: int = 16, seed: int = 123, checkpoint_path: str | Path = DEFAULT_CHECKPOINT) -> dict:
    """Train and persist a compact, reproducible synthetic demonstration model."""
    set_seed(seed)
    return train_dataset(make_synthetic_dataset(samples=samples, seed=seed), epochs, batch_size, seed, checkpoint_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BrainBeats EEG encoder.")
    parser.add_argument("--dataset", choices=("synthetic", "deap", "seed"), default="synthetic")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    if args.dataset == "synthetic":
        metrics = train_synthetic(args.epochs, args.samples, checkpoint_path=args.checkpoint)
    elif args.dataset == "deap":
        from ..eeg.datasets.deap_loader import load_deap
        data, labels = load_deap()
        metrics = train_dataset(make_deap_dataset(data, labels), args.epochs, checkpoint_path=args.checkpoint, source="user-supplied DEAP subject files")
    else:
        from ..eeg.datasets.seed_loader import load_seed
        trials, labels = load_seed()
        metrics = train_dataset(make_seed_dataset(trials, labels), args.epochs, checkpoint_path=args.checkpoint, source="user-supplied SEED raw trial matrices")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
