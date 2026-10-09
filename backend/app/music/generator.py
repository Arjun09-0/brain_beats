"""Phase 3 FiLM-conditioned symbolic music generator."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pretty_midi
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DEFAULT_CHECKPOINT = Path("models/music/synthetic_demo.pt")
CALM_LATENT = np.array([-0.75, -0.75, 0.0, 0.75, *([0.0] * 8)], dtype=np.float32)
AGITATED_LATENT = np.array([0.75, 0.75, 0.0, -0.75, *([0.0] * 8)], dtype=np.float32)
MAJOR = (0, 2, 4, 5, 7, 9, 11)
MINOR = (0, 2, 3, 5, 7, 8, 10)


def music_controls(latent: np.ndarray | list[float]) -> dict[str, Any]:
    """Map 12-D EEG latent vector to deterministic macro-musical controls."""
    z = np.asarray(latent, dtype=np.float32).reshape(-1)
    if z.size != 12:
        raise ValueError(f"Expected a 12-D latent vector; received {z.size} values.")
    return {
        "tempo_bpm": float(np.clip(110 + 40 * z[0], 60, 170)),
        "notes_per_beat": float(np.clip(2.5 + 1.5 * z[1], 1, 4)),
        "key_root": int(np.clip(round(6 + 5 * z[2]), 0, 11)),
        "mode": "major" if z[3] >= 0 else "minor",
    }


class FiLMMusicGenerator(nn.Module):
    """GRU music sequence generator conditioned via Feature-wise Linear Modulation (FiLM)."""

    def __init__(self, hidden_size: int = 48, latent_dim: int = 12) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(128, 24)
        self.gru = nn.GRU(24, hidden_size, batch_first=True)
        self.film = nn.Linear(latent_dim, hidden_size * 2)
        self.pitch_head = nn.Linear(hidden_size, 128)
        self.duration_head = nn.Linear(hidden_size, 4)

    def forward(self, pitches: torch.Tensor, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, _ = self.gru(self.embedding(pitches))
        scale, bias = self.film(latent).chunk(2, -1)
        hidden = hidden * (1 + scale.unsqueeze(1)) + bias.unsqueeze(1)
        return self.pitch_head(hidden), self.duration_head(hidden)


def _examples(corpus: str | Path, limit: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    xs: list[list[int]] = []
    ys: list[list[int]] = []
    ds: list[list[int]] = []
    used = 0
    midi_files = sorted((*Path(corpus).rglob("*.mid"), *Path(corpus).rglob("*.midi")))[:limit]
    for path in midi_files:
        try:
            pm = pretty_midi.PrettyMIDI(str(path))
            notes = sorted((n for inst in pm.instruments for n in inst.notes), key=lambda n: n.start)
        except Exception:
            continue
        if len(notes) < 33:
            continue
        used += 1
        pitches = [n.pitch for n in notes]
        durations = [
            min(3, int(n.end - n.start > 0.25) + int(n.end - n.start > 0.75) + int(n.end - n.start > 1.5))
            for n in notes
        ]
        for s in range(0, min(len(pitches) - 33, 96), 32):
            xs.append(pitches[s : s + 32])
            ys.append(pitches[s + 1 : s + 33])
            ds.append(durations[s + 1 : s + 33])
    if not xs:
        raise ValueError(f"No usable MIDI sequences found in {corpus}.")
    return np.array(xs), np.array(ys), np.array(ds), used


def train_music(
    corpus: str | Path,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    limit: int = 64,
    epochs: int = 4,
    seed: int = 123,
) -> dict[str, Any]:
    """Train the compact FiLM-conditioned GRU generator on a slice of MIDI files."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    x, y, d, used = _examples(corpus, limit)
    z = torch.randn(len(x), 12, generator=torch.Generator().manual_seed(seed))
    dataset = TensorDataset(torch.tensor(x), torch.tensor(y), torch.tensor(d), z)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(seed))

    model = FiLMMusicGenerator()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    ce = nn.CrossEntropyLoss()
    losses: list[float] = []

    for _ in range(epochs):
        total = 0.0
        for bx, by, bd, bz in loader:
            lp, ld = model(bx, bz)
            loss = ce(lp.reshape(-1, 128), by.reshape(-1)) + ce(ld.reshape(-1, 4), bd.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        losses.append(total / len(loader))

    dest = Path(checkpoint_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": {"hidden_size": model.hidden_size, "latent_dim": model.latent_dim},
            "source": str(corpus),
            "file_limit": limit,
            "files_used": used,
            "seed": seed,
        },
        dest,
    )
    return {"train_loss": losses, "checkpoint": str(dest), "files_used": used, "examples": len(x)}


def load_music_model(checkpoint_path: str | Path = DEFAULT_CHECKPOINT) -> FiLMMusicGenerator:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = FiLMMusicGenerator(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def _scale(pitch: int, root: int, mode: str) -> int:
    allowed = MAJOR if mode == "major" else MINOR
    return min((p for p in range(36, 85) if (p - root) % 12 in allowed), key=lambda p: abs(p - pitch))


@torch.inference_mode()
def generate_midi(
    latent: np.ndarray | list[float],
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    beats: int = 16,
) -> tuple[pretty_midi.PrettyMIDI, dict[str, Any]]:
    """Generate MIDI notes conditioned on the 12-D latent vector."""
    controls = music_controls(latent)
    model = load_music_model(checkpoint_path)
    count = max(1, round(beats * controls["notes_per_beat"]))
    spacing = 60.0 / controls["tempo_bpm"] / controls["notes_per_beat"]
    history = [60] * 32
    z = torch.tensor(np.asarray(latent, dtype=np.float32)).unsqueeze(0)

    pm = pretty_midi.PrettyMIDI(initial_tempo=controls["tempo_bpm"])
    inst = pretty_midi.Instrument(program=0)

    for i in range(count):
        logits, _ = model(torch.tensor([history]), z)
        raw_pitch = int(logits[0, -1].argmax())
        pitch = _scale(raw_pitch, controls["key_root"], controls["mode"])
        start = i * spacing
        inst.notes.append(pretty_midi.Note(velocity=88, pitch=pitch, start=start, end=start + spacing * 0.8))
        history = history[1:] + [pitch]

    pm.instruments.append(inst)
    return pm, controls


def midi_metrics(pm: pretty_midi.PrettyMIDI) -> dict[str, Any]:
    """Compute summary metrics for a generated PrettyMIDI object."""
    tempo = float(pm.get_tempo_changes()[1][0])
    notes = [n for inst in pm.instruments for n in inst.notes]
    end = max((n.end for n in notes), default=0.0)
    density = len(notes) / (end * tempo / 60.0) if end > 0 else 0.0
    return {"tempo_bpm": tempo, "note_density": density, "notes": len(notes)}
