import json
from pathlib import Path

from backend.app.storage.session_store import SessionStore


def test_session_store_lifecycle(tmp_path: Path) -> None:
    store = SessionStore(base_dir=tmp_path)
    session_id = "test_sess_001"

    # Create session
    meta = store.create_session(session_id, {"eeg_source": "synthetic", "preset": "calm"})
    assert meta["session_id"] == session_id
    assert meta["status"] == "active"

    # Append telemetry
    store.append_telemetry(session_id, {"frame_id": 1, "valence": 0.8, "arousal": 0.2})
    store.append_telemetry(session_id, {"frame_id": 2, "valence": 0.75, "arousal": 0.25})

    telemetry = store.get_telemetry(session_id)
    assert len(telemetry) == 2
    assert telemetry[0]["valence"] == 0.8

    # Save music artifacts
    midi_name, wav_name = store.save_music_artifacts(session_id, 1, b"MThd...", b"RIFF...")
    assert midi_name == "clip_0001.mid"
    assert wav_name == "clip_0001.wav"
    assert store.get_music_artifact_path(session_id, midi_name).exists()

    # Finish session
    finished = store.finish_session(session_id, {"total_frames": 2})
    assert finished["status"] == "finished"
    assert finished["total_frames"] == 2

    # List sessions
    sessions = store.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == session_id
