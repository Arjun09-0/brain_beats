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
