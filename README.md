# BrainBeats

BrainBeats is a research/course pilot application for exploring an EEG-to-music
closed loop. It is not a medical or therapeutic device.

## Phase 1: synthetic EEG and band-power features

The current implementation generates deterministic, configurable synthetic
multi-channel EEG-like signals and extracts relative delta, theta, alpha, beta,
and gamma band powers. It runs without a headset or gated research dataset.

### Setup (Python 3.11)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
```

### Generate a sanity-check spectrum

```powershell
python -m backend.app.eeg.mock_stream --preset calm --duration 20 --plot reports/calm_bandpower.png
```

This writes a PNG spectrum/band-power plot and prints relative band powers. A
calm preset intentionally has alpha as its dominant relative band; an aroused
preset emphasizes beta/gamma.

### Run tests

```powershell
pytest backend/tests
```

## Phase 2: encoder demo

Train the included synthetic-data demonstration checkpoint:

```powershell
python -m backend.app.encoder.train --dataset synthetic --epochs 14 --samples 48
```

This writes `models/encoder/synthetic_demo.pt`. It learns labels assigned to
the mock generator's calm/aroused presets only, so its perfect mock validation
score is a pipeline check, not evidence of real-world emotion recognition.

Later, after manually adding licenced data, invoke the same script with
`--dataset deap` or `--dataset seed`. Empty data directories give an actionable
message; neither dataset is downloaded or fabricated by this project.

## Data policy

DEAP and SEED are not downloaded by this project. When later phases add their
loaders, user-supplied files will belong in `data/raw/deap/` and
`data/raw/seed/`. The synthetic generator is independent of those datasets.

## Planned optional hardware path

The eventual real-headset path will consume an LSL stream (for example, a Muse
published through `muselsl`). Hardware is not required for the mock pipeline.

## Phase 3: music generator

Fetch the documented corpus, then train the compact demo checkpoint:

```powershell
git clone --depth 1 https://github.com/narcisse-dev/classical_ancient_midifiles.git data/raw/classical_ancient_midifiles
python -m backend.app.music.train --epochs 4 --limit 64
python -m backend.app.music.infer --latent -0.75 -0.75 0 0.75 0 0 0 0 0 0 0 0 --output reports/calm.mid
```

`infer` writes MIDI and, when `pretty_midi`'s built-in sine-wave synthesis works, a WAV preview beside it. The model uses FiLM with the Phase 2-compatible 12-D latent; tempo, density, key, and mode are explicitly documented in `PROGRESS.md`.
