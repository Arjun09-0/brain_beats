# BrainBeats progress

## Phase 1 — complete

- Created the Phase 1 package layout and a deterministic synthetic EEG source.
- The source combines separately band-limited stochastic components in the
  standard delta, theta, alpha, beta, and gamma ranges. `calm` boosts alpha;
  `aroused` boosts beta/gamma; `neutral` is balanced.
- Added an iterator-style chunk interface so a future LSL adapter can consume
  the same `(samples, channels)` shape without changing preprocessing.
- Implemented zero-phase bandpass filtering, fixed-length epoching, Welch PSD,
  absolute/relative per-channel band powers, and a saved sanity-check plot.
- Added tests which confirm alpha-dominance for calm synthetic data and beta/
  gamma dominance for aroused data.

### Design notes

SciPy is used for the numerical filters and Welch PSD; MNE-Python remains a
required dependency for later EEG interoperability. Keeping Phase 1 array-first
makes mock data and future LSL chunks straightforward to test.

### Mocked versus real

All EEG in this phase is synthetic. No DEAP/SEED data is downloaded, imitated,
or represented as real data. LSL transport and all learned models are deferred
to later phases.

### Known limitations

Synthetic signals approximate spectral structure only; they do not represent
physiology or support clinical interpretation. The current spectral summary is
session-level and has no artifact-rejection pipeline yet.

## Phase 2 — EEG encoder — complete

- Architecture: CNN+LSTM as requested. A small CNN extracts channel-by-band
  spectral patterns for each 2-second epoch; an LSTM summarizes the epoch
  sequence. The latent state has 12 dimensions and two training-only binary
  heads provide valence/arousal-style supervision.
- Demo data source: deterministic recordings from Phase 1 `mock_stream.py`.
  Their labels name mock presets only (calm = positive/low-arousal; aroused =
  lower/high-arousal); they are not emotion labels from real people.
- DEAP `.dat` and SEED `.mat` loaders never download or invent data, and report
  an actionable error when their data directories are empty.
- The initial split is subject-dependent when real data is used; subject-
  independent evaluation is a documented future limitation. Real-data accuracy
  must be assessed after the exact user-supplied release is available; target
  DEAP binary validation performance is approximately 75–90%, otherwise this
  phase should be revisited rather than treated as validated.
- Verification: `pytest backend/tests` passes (7 tests). The reproducible demo
  run used 48 mock recordings, 14 epochs, and seed 123; training loss fell from
  1.391 to 0.336 and its mock-preset validation label accuracy was 1.00. The
  loadable artifact is `models/encoder/synthetic_demo.pt`.

## Phase 3 — Music generator — complete

### Corpus decision (recorded before training)

- **Chosen corpus:** the public GitHub repository
  `narcisse-dev/classical_ancient_midifiles`
  (`https://github.com/narcisse-dev/classical_ancient_midifiles`). It was
  successfully shallow-cloned in this environment on 2026-09-04: 4,723 MIDI
  files, about 26 MB. The reproducible demo uses the first 64 valid files in
  sorted path order, which keeps CPU training quick while using real MIDI
  material. The downloaded corpus is ignored by Git; its source URL and slice
  policy are retained here instead.

### Conditioning and inspectable latent-to-music contract

- **Mechanism:** FiLM. The 12-D Phase 2 encoder vector is projected into a
  per-feature scale and bias which modulate the music GRU hidden state before
  pitch and duration prediction. The same vector also drives the explicit,
  deterministic control layer below; therefore the requested output controls
  are not left to an opaque learned correlation.

| Latent coordinate(s) | Music control | Exact mapping |
| --- | --- | --- |
| `z[0]` | tempo | `clip(110 + 40*z[0], 60, 170)` BPM |
| `z[1]` | note density | `clip(2.5 + 1.5*z[1], 1, 4)` notes/beat; event spacing is its reciprocal |
| `z[2]` | key root | nearest `round(6 + 5*z[2])`, clipped to chromatic pitch class 0–11 (C=0) |
| `z[3]` | mode | `z[3] >= 0` selects major; otherwise natural minor |
| `z[4:12]` | learned phrase variation | FiLM-only controls for pitch/duration logits; no claimed semantic interpretation |

- **Architecture:** `FiLMMusicGenerator` combines a 24-dimensional embedding for
  128 MIDI pitches, a single-layer GRU (hidden size 48), and a linear FiLM
  projection layer mapping the 12-D latent vector to 48 scale and 48 bias values
  (`hidden * (1 + scale) + bias`). Two separate linear heads predict next-pitch
  logits (128 classes) and note duration buckets (4 classes: <0.25s, 0.25–0.75s,
  0.75–1.5s, >1.5s).
- **Harmonic constraint:** Generated pitches are clamped and snapped to the
  nearest pitch matching the active scale (major or natural minor) rooted at
  `key_root` across MIDI notes 36 to 84 (C2 to C6).
- **Sequence windowing:** MIDI files are parsed into monophonic pitch sequences
  and extracted into 32-step sliding context windows predicting the next pitch
  and duration token.
- **Verification:** `pytest backend/tests` passes (9 tests total). Unit tests
  confirm that calm and agitated preset latent vectors produce statistically
  and numerically distinct tempo (>= 50 BPM difference) and note density
  (>= 2 notes/beat difference), and that wrong-sized latent vectors trigger
  a `ValueError`.
- **Reproducible demo run:** Trained on the first 64 valid files from
  `data/raw/classical_ancient_midifiles` (157 training sequence windows, 4 epochs,
  batch size 32, Adam lr=0.002, seed 123). Cross-entropy loss decreased from
  6.292 to 4.903 (`[6.292, 5.865, 5.426, 4.903]`). Loadable artifacts are
  `models/music/synthetic_demo.pt` and `models/music/synthetic_demo.metrics.json`.
- **Inference CLI:** `python -m backend.app.music.infer` produces standard MIDI
  and synthesized 16 kHz sine-wave audio previews (`reports/calm.mid`,
  `reports/calm.wav`, `reports/agitated.mid`, `reports/agitated.wav`).

### Design notes

Separating deterministic macro controls (`tempo`, `note density`, `key`, `mode`)
from learned FiLM pitch/duration generation prevents uncontrolled model drift.
Users and clinical researchers can directly inspect and trust why tempo and
density shift based on the EEG latent state, while the neural network handles
melodic phrasing.

### Mocked versus real

The MIDI dataset is real historical classical MIDI. Training latent vectors `z`
in the demo script are standard normal samples; at inference time, real or
mock latent outputs from the Phase 2 encoder are fed directly into the
generator without modification.

### Known limitations

The generator currently produces monophonic melodic lines. Polyphony, chord
progressions, and multi-track orchestration are deferred to future work.
Audio preview synthesis uses `pretty_midi`'s built-in sinusoidal generator;
soundfont/wavetable synthesis is not yet integrated.

## Phase 4 — Closed-loop integration and API — complete

### Architecture and Closed-Loop Flow
- **Orchestration:** `ClosedLoopEngine` ([closed_loop.py](file:///c:/EEG_music_config/backend/app/closed_loop.py))
  integrates Phase 1 synthetic multi-channel EEG generation with 4-second
  sliding-window epoching, Phase 2 `EEGStateEncoder` CNN+LSTM latent feature
  extraction (producing the 12-D latent vector `z`), and Phase 3
  `FiLMMusicGenerator` macro musical controls and GRU melodic sequencing.
- **FastAPI backend:** `backend/app/main.py` provides:
  - REST endpoints for session lifecycle (`/api/sessions/start`, `/api/sessions/stop`),
    dynamic preset switching (`/api/sessions/preset`), biofeedback target goal setting
    (`/api/sessions/goal`), single-step evaluation (`/api/sessions/step`), external
    EEG ingestion (`/api/sessions/ingest`), and artifact downloads (MIDI, WAV).
  - High-performance WebSocket endpoint (`/ws/live`) delivering push telemetry
    frames at ~2-second loop cadence to all connected frontends.
- **Storage & Telemetry:** `SessionStore` ([session_store.py](file:///c:/EEG_music_config/backend/app/storage/session_store.py))
  persists session metadata in `backend/storage/sessions/{session_id}/metadata.json`,
  streaming frames in `telemetry.jsonl`, and generated `.mid` and `.wav` audio clips
  under `music/`.
- **Interactive Web Dashboard:** Modern, glassmorphic dark neurotech UI in
  `backend/app/static/`:
  - 8-Channel real-time EEG Oscilloscope canvas (Fp1 through O2).
  - Welch PSD frequency band power meters (Delta, Theta, Alpha, Beta, Gamma) and
    Alpha/Beta coherence ratio.
  - 2D Russell Circumplex Emotion plane (Valence vs Arousal) with animated trajectory.
  - 12-Dimensional FiLM latent vector sparkbar visualization.
  - Generative music studio with real-time tempo metronome pulse, note density gauge,
    harmonic key and mode snapping, and real-time scrolling piano roll canvas.
  - Dual audio playback: low-latency in-browser Web Audio API synthesizer with reverb,
    plus server-rendered 16 kHz WAV streaming.
  - Dynamic preset switching buttons (🌿 Calm / Alpha Boost, ⚖️ Neutral, ⚡ Aroused / Beta Surge)
    and session telemetry exporter.
- **Verification:** Full test suite passes (12 tests total across preprocessing,
  encoder, music generator, closed loop engine, session storage, and FastAPI REST/WebSocket endpoints).

