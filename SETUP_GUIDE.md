# BrainBeats — Device Setup & Installation Guide

This guide walks you through setting up and running **BrainBeats** on a fresh machine (Windows, macOS, or Linux).

---

## 1. Prerequisites

Make sure the target device has the following installed:
* **Git**: [https://git-scm.com/](https://git-scm.com/)
* **Python 3.10 or 3.11** (Python 3.11 recommended): [https://www.python.org/downloads/](https://www.python.org/downloads/)
  * *On Windows, ensure you check **"Add python.exe to PATH"** during installation.*

---

## 2. Clone the Repository

Open a terminal (PowerShell, Command Prompt, or Bash) and clone the repository:

```bash
git clone https://github.com/Arjun09-0/brain_beats.git
cd brain_beats
```

---

## 3. Virtual Environment Setup & Installation

### On Windows (PowerShell)

```powershell
# Create virtual environment
python -m venv .venv

# Install dependencies using the venv pip
.\.venv\Scripts\pip install -r backend/requirements.txt
```

### On macOS / Linux (Terminal)

```bash
# Create virtual environment
python3 -m venv .venv

# Install dependencies using the venv pip
./.venv/bin/pip install -r backend/requirements.txt
```

---

## 4. Launching the Application

### Option A: Direct Command (Cross-Platform)

* **Windows:**
  ```powershell
  .\.venv\Scripts\python.exe -m backend.app.main
  ```

* **macOS / Linux:**
  ```bash
  ./.venv/bin/python -m backend.app.main
  ```

### Option B: Using Windows Shortcut Scripts

* Double-click `run.bat` or run:
  ```powershell
  .\run.bat
  ```

### Option C: Using Uvicorn Directly

* **Windows:**
  ```powershell
  .\.venv\Scripts\uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
  ```

* **macOS / Linux:**
  ```bash
  ./.venv/bin/uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
  ```

---

## 5. Opening the Web Dashboard

Once the terminal outputs:
```text
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Open your browser and navigate to:
👉 **[http://localhost:8000](http://localhost:8000)**

### First Time in Dashboard:
1. Click **▶ Start Session** in the top header.
2. Ensure your browser audio is enabled (some browsers block autoplay until you interact with the page).
3. Toggle between **🌿 Calm**, **⚖️ Neutral**, and **⚡ Aroused** presets to see the real-time EEG waves, band powers, emotional state, and generated music adapt dynamically.

---

## 6. Running Automated Tests

To verify that all signal processing, deep learning models, storage, and API endpoints are working properly:

* **Windows:**
  ```powershell
  .\.venv\Scripts\pytest backend/tests
  ```

* **macOS / Linux:**
  ```bash
  ./.venv/bin/pytest backend/tests
  ```

Expected result: **All 12 tests pass**.

---

## 7. Troubleshooting & FAQ

### Issue: `ModuleNotFoundError: No module named 'fastapi'`
* **Cause**: You ran system Python instead of the virtual environment Python.
* **Fix**: Use `.\.venv\Scripts\python.exe` (Windows) or `./.venv/bin/python` (Mac/Linux) rather than just `python`.

### Issue: `running scripts is disabled on this system` (PowerShell)
* **Cause**: Windows PowerShell security execution policy.
* **Fix**: Run `.\run.bat` or `.\.venv\Scripts\python.exe -m backend.app.main` directly (executables are not blocked by script policies).

### Issue: Port 8000 is already in use
* **Fix**: Specify another port:
  ```powershell
  .\.venv\Scripts\uvicorn backend.app.main:app --port 8080
  ```
  Then open `http://localhost:8080`.

### Issue: No audio heard in browser
* **Fix**: Click anywhere on the webpage or press the **🔊 Sound On** button to grant browser Web Audio context permission.
