"""DEAP `.dat` loader; files must be supplied manually by the user."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def load_deap(directory: str | Path = "data/raw/deap") -> tuple[np.ndarray, np.ndarray]:
    """Load and concatenate DEAP subject files into `(trials, channels, samples)`.

    Labels retain DEAP's four continuous dimensions in their published order.
    No dataset download or synthetic substitution is attempted.
    """
    folder = Path(directory)
    files = sorted(path for path in folder.glob("*.dat") if path.is_file())
    if not files:
        raise FileNotFoundError(
            f"No DEAP .dat files found in {folder.resolve()}. DEAP is licence-gated: "
            "download it yourself under its terms, then place subject files (for example s01.dat) there. "
            "For a hardware-free demo, run training with --dataset synthetic."
        )
    trials, labels = [], []
    for file in files:
        with file.open("rb") as handle:
            payload = pickle.load(handle, encoding="latin1")
        if not isinstance(payload, dict) or "data" not in payload or "labels" not in payload:
            raise ValueError(f"{file} is not a recognised DEAP subject .dat file (expected data and labels keys).")
        trials.append(np.asarray(payload["data"], dtype=np.float32))
        labels.append(np.asarray(payload["labels"], dtype=np.float32))
    return np.concatenate(trials, axis=0), np.concatenate(labels, axis=0)

