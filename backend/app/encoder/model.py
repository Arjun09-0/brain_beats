"""Compact CNN+LSTM encoder for sequences of EEG band-power maps."""
from __future__ import annotations

import torch
from torch import nn


class EEGStateEncoder(nn.Module):
    """Encode `(batch, epochs, channels, bands)` into a latent state vector.

    A small 2-D CNN learns spatial/spectral patterns independently for every
    epoch; an LSTM then summarizes their evolution over time.  Two binary heads
    provide valence and arousal supervision during training.
    """

    def __init__(self, channels: int = 8, bands: int = 5, latent_dim: int = 12, cnn_dim: int = 24, lstm_dim: int = 32) -> None:
        super().__init__()
        self.channels, self.bands, self.latent_dim = channels, bands, latent_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, cnn_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.temporal = nn.LSTM(cnn_dim, lstm_dim, batch_first=True)
        self.latent = nn.Sequential(nn.Linear(lstm_dim, latent_dim), nn.Tanh())
        self.valence_head = nn.Linear(latent_dim, 1)
        self.arousal_head = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError("Expected input shaped (batch, epochs, channels, bands).")
        batch, epochs, channels, bands = x.shape
        if channels != self.channels or bands != self.bands:
            raise ValueError(f"Expected {self.channels} channels and {self.bands} bands, got {channels} and {bands}.")
        maps = x.reshape(batch * epochs, 1, channels, bands)
        cnn_features = self.cnn(maps).flatten(1).reshape(batch, epochs, -1)
        temporal_features, _ = self.temporal(cnn_features)
        latent = self.latent(temporal_features[:, -1])
        return {
            "latent": latent,
            "valence_logits": self.valence_head(latent).squeeze(-1),
            "arousal_logits": self.arousal_head(latent).squeeze(-1),
        }

