@echo off
if exist .venv\Scripts\python.exe (
    echo Starting BrainBeats...
    .venv\Scripts\python.exe -m backend.app.main
) else (
    echo Virtual environment .venv not found.
    pause
)
