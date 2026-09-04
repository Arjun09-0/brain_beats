# Synthetic EEG samples

Generate reproducible Phase 1 samples and sanity-check plots with:

```powershell
python -m backend.app.eeg.mock_stream --preset calm --plot reports/calm_bandpower.png
python -m backend.app.eeg.mock_stream --preset aroused --plot reports/aroused_bandpower.png
```

The generator's source and fixed default seed are the checked-in reproducible
sample definition; no real EEG data is included.

