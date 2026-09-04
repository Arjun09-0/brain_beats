from pathlib import Path

from backend.app.eeg.mock_stream import MockEEGConfig, generate_synthetic_eeg
from backend.app.eeg.preprocessing import epoch_data, extract_band_powers, plot_bandpower_sanity_check


def test_calm_synthetic_eeg_is_alpha_dominant() -> None:
    data, _ = generate_synthetic_eeg(MockEEGConfig(channels=6, duration=20, state="calm", seed=21))
    features = extract_band_powers(data, sample_rate=128)
    assert features.band_names[features.relative.mean(axis=0).argmax()] == "alpha"


def test_aroused_synthetic_eeg_emphasizes_beta_or_gamma() -> None:
    data, _ = generate_synthetic_eeg(MockEEGConfig(channels=6, duration=20, state="aroused", seed=33))
    features = extract_band_powers(data, sample_rate=128)
    assert features.band_names[features.relative.mean(axis=0).argmax()] in {"beta", "gamma"}


def test_epoching_and_sanity_plot(tmp_path: Path) -> None:
    data, _ = generate_synthetic_eeg(MockEEGConfig(channels=2, duration=6, state="calm"))
    assert epoch_data(data, 128, epoch_seconds=2).shape == (3, 256, 2)
    features = extract_band_powers(data, 128)
    output = tmp_path / "spectrum.png"
    plot_bandpower_sanity_check(data, 128, features, output)
    assert output.is_file() and output.stat().st_size > 0

