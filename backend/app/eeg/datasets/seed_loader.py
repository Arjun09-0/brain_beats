"""SEED `.mat` loader; files must be supplied manually by the user."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat


def load_seed(directory: str | Path = "data/raw/seed") -> tuple[list[np.ndarray], np.ndarray]:
    """Load supplied SEED trial matrices and labels from `label.mat`.

    SEED releases vary in feature representation, so this intentionally returns
    raw trial matrices for the train adapter to validate/standardise explicitly.
    """
    folder = Path(directory)
    files = sorted(path for path in folder.glob("*.mat") if path.name.lower() != "label.mat")
    label_file = folder / "label.mat"
    if not files or not label_file.is_file():
        raise FileNotFoundError(
            f"SEED data is incomplete in {folder.resolve()}. Place the supplied trial .mat files and label.mat there; "
            "SEED is not downloaded by this project. Use --dataset synthetic for the demo checkpoint."
        )
    labels_data = loadmat(label_file)
    label_values = next((value for key, value in labels_data.items() if not key.startswith("__") and isinstance(value, np.ndarray)), None)
    if label_values is None:
        raise ValueError(f"{label_file} contains no numeric label array.")
    trials: list[np.ndarray] = []
    for file in files:
        contents = loadmat(file)
        trials.extend(np.asarray(value, dtype=np.float32) for key, value in contents.items() if not key.startswith("__") and isinstance(value, np.ndarray) and value.ndim >= 2)
    return trials, np.asarray(label_values).squeeze()

