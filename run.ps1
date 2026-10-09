# BrainBeats Startup Script
$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
    Write-Host "Starting BrainBeats with virtual environment..." -ForegroundColor Cyan
    & $VenvPython -m backend.app.main
} else {
    Write-Host "Virtual environment not found. Please setup .venv first." -ForegroundColor Red
}
