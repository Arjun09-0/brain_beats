"""Feature datasets for synthetic and supplied EEG recordings."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from ..eeg.mock_stream import MockEEGConfig, generate_synthetic_eeg
from ..eeg.preprocessing import epoch_data, extract_band_powers


@dataclass(frozen=True)
class EncoderExample:
    features: np.ndarray  # (epochs, channels, bands)
    labels: np.ndarray  # (valence, arousal), binary


class EEGFeatureDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, examples: list[EncoderExample]) -> None:
        if not examples:
            raise ValueError("Encoder dataset is empty.")
        self.features = np.stack([example.features for example in examples]).astype(np.float32)
        self.labels = np.stack([example.labels for example in examples]).astype(np.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.features[index]), torch.from_numpy(self.labels[index])


def eeg_to_feature_sequence(data: np.ndarray, sample_rate: float, epoch_seconds: float = 2.0) -> np.ndarray:
    """Convert raw `(samples, channels)` data to relative-power epoch maps."""
    epochs = epoch_data(data, sample_rate, epoch_seconds)
    return np.stack(
        [extract_band_powers(epoch, sample_rate).relative for epoch in epochs], axis=0
    ).astype(np.float32)


def make_synthetic_dataset(samples: int = 64, channels: int = 8, sample_rate: float = 128.0, duration: float = 8.0, seed: int = 123) -> EEGFeatureDataset:
    """Build labelled mock examples without claiming they are DEAP/SEED data.

    Labels describe generator presets only: calm maps to positive-valence / low
    arousal and aroused maps to lower-valence / high-arousal. They are useful
    solely as a pipeline-learning sanity check.
    """
    examples: list[EncoderExample] = []
    for index in range(samples):
        state = "calm" if index % 2 == 0 else "aroused"
        labels = np.array((1.0, 0.0) if state == "calm" else (0.0, 1.0), dtype=np.float32)
        recording, _ = generate_synthetic_eeg(MockEEGConfig(channels, sample_rate, duration, state, seed + index))
        examples.append(EncoderExample(eeg_to_feature_sequence(recording, sample_rate), labels))
    return EEGFeatureDataset(examples)


def make_deap_dataset(data: np.ndarray, labels: np.ndarray, sample_rate: float = 128.0) -> EEGFeatureDataset:
    """Convert standard DEAP trial arrays to encoder features.

    DEAP subject files store 40 channels by samples. The first 32 are EEG; the
    last eight peripheral channels are excluded. Valence and arousal are split
    at the conventional five-point threshold.
    """
    if data.ndim != 3 or data.shape[1] < 32 or labels.ndim != 2 or labels.shape[1] < 2:
        raise ValueError("Expected standard DEAP data shaped (trials, >=32 channels, samples) and labels with valence/arousal columns.")
    examples: list[EncoderExample] = []
    for trial, target in zip(data, labels, strict=True):
        # DEAP begins each trial with a three-second baseline.
        eeg = trial[:32, int(sample_rate * 3) :].T
        examples.append(EncoderExample(eeg_to_feature_sequence(eeg, sample_rate), (np.asarray(target[:2]) >= 5.0).astype(np.float32)))
    return EEGFeatureDataset(examples)


def make_seed_dataset(trials: list[np.ndarray], labels: np.ndarray, sample_rate: float = 200.0) -> EEGFeatureDataset:
    """Convert raw-channel SEED trial matrices to feature sequences.

    This supports releases whose trial arrays are `(channels, samples)` (or the
    transpose). Pre-extracted SEED DE features need a separate documented
    adapter because they are not raw EEG and have release-specific dimensions.
    """
    labels = np.asarray(labels).reshape(-1)
    if len(trials) < len(labels):
        raise ValueError("SEED has fewer trial matrices than labels; verify the selected release directory.")
    examples: list[EncoderExample] = []
    for trial, label in zip(trials, labels, strict=False):
        values = np.asarray(trial, dtype=np.float32)
        if values.ndim != 2:
            continue
        # SEED recordings normally have 62 channels; detect either orientation.
        eeg = values.T if values.shape[0] <= values.shape[1] else values
        if eeg.shape[1] < 2 or eeg.shape[0] < int(sample_rate * 2):
            continue
        # SEED's {-1, 0, 1} emotion classes are mapped to transparent auxiliary
        # targets: positive-vs-not and high-arousal (non-neutral)-vs-neutral.
        examples.append(EncoderExample(eeg_to_feature_sequence(eeg, sample_rate), np.array((label > 0, label != 0), dtype=np.float32)))
    if not examples:
        raise ValueError("No raw EEG-shaped SEED trials were found. This release appears to contain pre-extracted features, which require a release-specific adapter.")
    return EEGFeatureDataset(examples)
