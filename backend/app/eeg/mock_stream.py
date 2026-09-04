"""Synthetic EEG source for hardware-free BrainBeats demonstrations.

The generator deliberately models spectral composition, not physiology.  It is
useful for exercising signal-processing code and must not be interpreted as a
real EEG recording.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
from scipy.signal import butter, sosfiltfilt

BandName = Literal["delta", "theta", "alpha", "beta", "gamma"]
BANDS: dict[BandName, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


@dataclass(frozen=True)
class MockEEGConfig:
    channels: int = 8
    sample_rate: float = 128.0
    duration: float = 20.0
    state: str = "calm"
    seed: int = 7


_STATE_WEIGHTS: dict[str, dict[str, float]] = {
    "calm": {"delta": 0.5, "theta": 0.8, "alpha": 3.0, "beta": 0.55, "gamma": 0.25},
    "neutral": {"delta": 0.8, "theta": 1.0, "alpha": 1.1, "beta": 1.0, "gamma": 0.55},
    "aroused": {"delta": 0.35, "theta": 0.55, "alpha": 0.5, "beta": 2.4, "gamma": 1.5},
}


def _band_limited_noise(samples: int, sample_rate: float, low: float, high: float, rng: np.random.Generator) -> np.ndarray:
    """Return unit-variance noise confined to a frequency band."""
    noise = rng.standard_normal(samples)
    sos = butter(4, (low, high), btype="bandpass", fs=sample_rate, output="sos")
    filtered = sosfiltfilt(sos, noise)
    return filtered / (np.std(filtered) + np.finfo(float).eps)


def generate_synthetic_eeg(config: MockEEGConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Generate `(samples, channels)` data in arbitrary microvolt-like units.

    Each channel varies slightly around the selected spectral-state preset.
    """
    config = config or MockEEGConfig()
    if config.state not in _STATE_WEIGHTS:
        raise ValueError(f"Unknown state {config.state!r}. Choose one of: {', '.join(_STATE_WEIGHTS)}.")
    if config.channels < 1 or config.sample_rate <= 90 or config.duration <= 2:
        raise ValueError("Use at least 1 channel, sample_rate > 90 Hz, and duration > 2 seconds.")
    samples = int(round(config.sample_rate * config.duration))
    rng = np.random.default_rng(config.seed)
    signal = np.zeros((samples, config.channels), dtype=np.float64)
    for channel in range(config.channels):
        channel_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        for band, (low, high) in BANDS.items():
            variability = channel_rng.uniform(0.9, 1.1)
            signal[:, channel] += _STATE_WEIGHTS[config.state][band] * variability * _band_limited_noise(samples, config.sample_rate, low, high, channel_rng)
        signal[:, channel] += 0.04 * channel_rng.standard_normal(samples)
    times = np.arange(samples, dtype=float) / config.sample_rate
    return signal, times


def iter_chunks(data: np.ndarray, sample_rate: float, chunk_seconds: float = 1.0) -> Iterator[tuple[np.ndarray, float]]:
    """Yield LSL-adapter-friendly sample chunks and their starting timestamps."""
    if data.ndim != 2:
        raise ValueError("Expected data shaped (samples, channels).")
    chunk_size = max(1, int(round(sample_rate * chunk_seconds)))
    for start in range(0, len(data), chunk_size):
        yield data[start : start + chunk_size], start / sample_rate


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mock EEG and a band-power sanity-check plot.")
    parser.add_argument("--preset", choices=sorted(_STATE_WEIGHTS), default="calm")
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--sample-rate", type=float, default=128.0)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--plot", type=Path, default=Path("reports/mock_bandpower.png"))
    args = parser.parse_args()
    from .preprocessing import extract_band_powers, plot_bandpower_sanity_check

    data, _ = generate_synthetic_eeg(MockEEGConfig(args.channels, args.sample_rate, args.duration, args.preset))
    features = extract_band_powers(data, args.sample_rate)
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    plot_bandpower_sanity_check(data, args.sample_rate, features, args.plot)
    relative = features.relative.mean(axis=0)
    print(f"Generated {args.preset} synthetic EEG: {data.shape[0]} samples x {data.shape[1]} channels")
    print("Mean relative band powers: " + ", ".join(f"{band}={value:.3f}" for band, value in zip(features.band_names, relative)))
    print(f"Plot saved to: {args.plot}")


if __name__ == "__main__":
    main()

