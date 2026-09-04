import numpy as np
import pytest

from backend.app.eeg.datasets.deap_loader import load_deap
from backend.app.eeg.datasets.seed_loader import load_seed
from backend.app.eeg.mock_stream import MockEEGConfig, generate_synthetic_eeg
from backend.app.encoder.infer import infer_latent
from backend.app.encoder.model import EEGStateEncoder
from backend.app.encoder.train import train_synthetic


def test_encoder_latent_shape() -> None:
    import torch
    output = EEGStateEncoder(latent_dim=12)(torch.zeros(3, 4, 8, 5))
    assert output["latent"].shape == (3, 12)


def test_synthetic_training_decreases_loss_and_inference_is_deterministic(tmp_path) -> None:
    checkpoint = tmp_path / "demo.pt"
    metrics = train_synthetic(epochs=8, samples=32, checkpoint_path=checkpoint)
    assert checkpoint.is_file()
    assert metrics["train_loss"][-1] < metrics["train_loss"][0]
    eeg, _ = generate_synthetic_eeg(MockEEGConfig(duration=8, state="calm", seed=999))
    first = infer_latent(eeg, 128, checkpoint)
    second = infer_latent(eeg, 128, checkpoint)
    assert first.shape == (12,)
    np.testing.assert_array_equal(first, second)


def test_empty_deap_loader_has_actionable_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="licence-gated.*--dataset synthetic"):
        load_deap(tmp_path)


def test_empty_seed_loader_has_actionable_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="not downloaded.*--dataset synthetic"):
        load_seed(tmp_path)
