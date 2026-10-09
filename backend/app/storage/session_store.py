"""Session storage and telemetry logging for BrainBeats."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Default storage directory under backend/storage
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STORAGE_DIR = PROJECT_ROOT / "backend" / "storage"


class SessionStore:
    """Manages persistent telemetry, metadata, and music files for sessions."""

    def __init__(self, base_dir: str | Path = DEFAULT_STORAGE_DIR) -> None:
        self.base_dir = Path(base_dir)
        self.sessions_dir = self.base_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _session_dir(self, session_id: str) -> Path:
        return self.sessions_dir / session_id

    def create_session(self, session_id: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Initialize session directory structure and metadata file."""
        s_dir = self._session_dir(session_id)
        music_dir = s_dir / "music"
        music_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "session_id": session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "active",
            "frame_count": 0,
            **(metadata or {}),
        }
        with open(s_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return meta

    def append_telemetry(self, session_id: str, frame: dict[str, Any]) -> None:
        """Append a single telemetry frame to telemetry.jsonl."""
        s_dir = self._session_dir(session_id)
        if not s_dir.exists():
            self.create_session(session_id)

        # Do not write large audio bytes/base64 to telemetry.jsonl
        persist_frame = {k: v for k, v in frame.items() if k not in ("audio_base64", "raw_eeg_preview")}
        telemetry_file = s_dir / "telemetry.jsonl"
        with open(telemetry_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(persist_frame) + "\n")

    def save_music_artifacts(
        self,
        session_id: str,
        frame_id: int,
        midi_data: bytes | None,
        wav_data: bytes | None,
    ) -> tuple[str | None, str | None]:
        """Save generated MIDI and WAV files for a given frame."""
        s_dir = self._session_dir(session_id)
        music_dir = s_dir / "music"
        music_dir.mkdir(parents=True, exist_ok=True)

        midi_filename = None
        if midi_data:
            midi_filename = f"clip_{frame_id:04d}.mid"
            with open(music_dir / midi_filename, "wb") as f:
                f.write(midi_data)

        wav_filename = None
        if wav_data:
            wav_filename = f"clip_{frame_id:04d}.wav"
            with open(music_dir / wav_filename, "wb") as f:
                f.write(wav_data)

        return midi_filename, wav_filename

    def finish_session(self, session_id: str, summary: dict[str, Any] | None = None) -> dict[str, Any]:
        """Mark session completed and update metadata summary."""
        s_dir = self._session_dir(session_id)
        meta_file = s_dir / "metadata.json"
        if not meta_file.exists():
            return {}

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        meta["status"] = "finished"
        meta["finished_at"] = datetime.now(timezone.utc).isoformat()
        if summary:
            meta.update(summary)

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return meta

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Retrieve session metadata."""
        meta_file = self._session_dir(session_id) / "metadata.json"
        if not meta_file.exists():
            return None
        with open(meta_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_sessions(self) -> list[dict[str, Any]]:
        """List metadata for all recorded sessions."""
        sessions = []
        if not self.sessions_dir.exists():
            return sessions

        for item in sorted(self.sessions_dir.iterdir(), reverse=True):
            if item.is_dir():
                meta_file = item / "metadata.json"
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            sessions.append(json.load(f))
                    except Exception:
                        continue
        return sessions

    def get_telemetry(self, session_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """Read recent telemetry frames for a session."""
        telemetry_file = self._session_dir(session_id) / "telemetry.jsonl"
        if not telemetry_file.exists():
            return []

        frames = []
        with open(telemetry_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        frames.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return frames[-limit:]

    def get_music_artifact_path(self, session_id: str, filename: str) -> Path | None:
        """Resolve path to a specific music artifact."""
        safe_name = Path(filename).name
        target = self._session_dir(session_id) / "music" / safe_name
        return target if target.is_file() else None
