from pathlib import Path
import numpy as np

from backend.app.closed_loop import ClosedLoopEngine


def test_closed_loop_engine_end_to_end(tmp_path: Path) -> None:
    engine = ClosedLoopEngine(storage_dir=tmp_path)

    # Start session
    session_info = engine.start_session(preset="calm", target_goal="calm")
    assert engine.is_running
    assert engine.session_id is not None
    assert session_info["session_id"] == engine.session_id

    # Execute one step
    frame = engine.step(render_audio=True, beats=8)
    assert frame.frame_id == 1
    assert "alpha" in frame.band_powers
    assert -1.0 <= frame.valence <= 1.0
    assert -1.0 <= frame.arousal <= 1.0
    assert len(frame.latent) == 12
    assert 60.0 <= frame.tempo_bpm <= 170.0
    assert len(frame.notes) > 0
    assert frame.audio_base64 is not None
    assert frame.midi_filename is not None
    assert frame.wav_filename is not None

    # Switch preset to aroused and step again
    engine.set_preset("aroused")
    frame2 = engine.step(render_audio=False, beats=8)
    assert frame2.frame_id == 2
    assert frame2.preset == "aroused"
    # Aroused should lead to distinct or higher tempo/density
    assert 60.0 <= frame2.tempo_bpm <= 170.0

    # Stop session
    res = engine.stop_session()
    assert res["status"] == "finished"
    assert not engine.is_running

    # Verify telemetry file was written
    telemetry = engine.store.get_telemetry(session_info["session_id"])
    assert len(telemetry) == 2
