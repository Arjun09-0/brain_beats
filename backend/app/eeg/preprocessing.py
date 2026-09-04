"""Array-first EEG preprocessing and spectral feature extraction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import butter, sosfiltfilt, welch

from .mock_stream import BANDS


@dataclass(frozen=True)
class BandPowerFeatures:
    """Per-channel band-power summary, with columns ordered by `band_names`."""
    absolute: np.ndarray
    relative: np.ndarray
    band_names: tuple[str, ...]


def bandpass_filter(data: np.ndarray, sample_rate: float, low: float = 1.0, high: float = 45.0) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass to `(samples, channels)` EEG."""
    values = _validate_data(data, sample_rate)
    if not 0 < low < high < sample_rate / 2:
        raise ValueError("Filter bounds must lie below the Nyquist frequency.")
    sos = butter(4, (low, high), btype="bandpass", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, values, axis=0)


def epoch_data(data: np.ndarray, sample_rate: float, epoch_seconds: float = 2.0) -> np.ndarray:
    """Split data into complete epochs: `(epochs, samples_per_epoch, channels)`."""
    values = _validate_data(data, sample_rate)
    width = int(round(sample_rate * epoch_seconds))
    if width < 2:
        raise ValueError("epoch_seconds is too short.")
    count = len(values) // width
    if count == 0:
        raise ValueError("Recording is shorter than one epoch.")
    return values[: count * width].reshape(count, width, values.shape[1])


def extract_band_powers(data: np.ndarray, sample_rate: float, apply_filter: bool = True) -> BandPowerFeatures:
    """Compute Welch absolute and relative power for five canonical EEG bands."""
    values = bandpass_filter(data, sample_rate) if apply_filter else _validate_data(data, sample_rate)
    frequencies, psd = welch(values, fs=sample_rate, nperseg=min(int(sample_rate * 2), len(values)), axis=0)
    names = tuple(BANDS)
    absolute = np.empty((values.shape[1], len(names)), dtype=float)
    for index, name in enumerate(names):
        low, high = BANDS[name]
        mask = (frequencies >= low) & (frequencies < high)
        absolute[:, index] = trapezoid(psd[mask], frequencies[mask], axis=0)
    relative = absolute / (absolute.sum(axis=1, keepdims=True) + np.finfo(float).eps)
    return BandPowerFeatures(absolute=absolute, relative=relative, band_names=names)


def plot_bandpower_sanity_check(data: np.ndarray, sample_rate: float, features: BandPowerFeatures, output_path: Path) -> None:
    """Save a spectrum plus mean relative-band-power plot for visual inspection."""
    values = _validate_data(data, sample_rate)
    frequencies, psd = welch(values, fs=sample_rate, nperseg=min(int(sample_rate * 2), len(values)), axis=0)
    figure, (spectrum_ax, bands_ax) = plt.subplots(1, 2, figsize=(12, 4.5))
    spectrum_ax.semilogy(frequencies, psd.mean(axis=1), color="#335c81")
    for name, (low, high) in BANDS.items():
        spectrum_ax.axvspan(low, high, alpha=0.12, label=name)
    spectrum_ax.set(xlim=(0, 50), xlabel="Frequency (Hz)", ylabel="PSD", title="Mean-channel Welch spectrum")
    spectrum_ax.legend(ncol=2, fontsize="small")
    mean_relative = features.relative.mean(axis=0)
    bands_ax.bar(features.band_names, mean_relative, color="#4f772d")
    bands_ax.set(ylim=(0, 1), ylabel="Relative power", title="Mean relative band power")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _validate_data(data: np.ndarray, sample_rate: float) -> np.ndarray:
    values = np.asarray(data, dtype=float)
    if values.ndim != 2 or values.shape[0] < 32 or values.shape[1] < 1:
        raise ValueError("Expected EEG shaped (at least 32 samples, at least 1 channel).")
    if not np.isfinite(values).all() or sample_rate <= 0:
        raise ValueError("EEG must be finite and sample_rate must be positive.")
    return values
