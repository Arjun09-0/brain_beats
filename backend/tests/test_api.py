from pathlib import Path
from fastapi.testclient import TestClient

from backend.app.main import app, engine


def test_api_endpoints_and_lifecycle(tmp_path: Path) -> None:
    # Use temporary storage for clean tests
    engine.store.base_dir = tmp_path
    engine.store.sessions_dir = tmp_path / "sessions"
    engine.store.sessions_dir.mkdir(parents=True, exist_ok=True)

    with TestClient(app) as client:
        # 1. Health status
        res = client.get("/api/status")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ready"

        # 2. Start session
        res = client.post("/api/sessions/start", json={"preset": "calm", "target_goal": "calm", "interval": 1.0})
        assert res.status_code == 200
        start_data = res.json()
        assert start_data["status"] == "started"
        session_id = start_data["session"]["session_id"]
        assert engine.session_id == session_id

        # 3. Get current session state
        res = client.get("/api/sessions/current")
        assert res.status_code == 200
        cur_data = res.json()
        assert cur_data["is_running"] is True
        assert cur_data["preset"] == "calm"
        assert cur_data["latest_frame"] is not None

        # 4. Change preset
        res = client.post("/api/sessions/preset", json={"preset": "aroused"})
        assert res.status_code == 200
        assert res.json()["preset"] == "aroused"
        assert engine.preset == "aroused"

        # 5. Trigger single step
        res = client.post("/api/sessions/step")
        assert res.status_code == 200
        step_frame = res.json()
        assert step_frame["preset"] == "aroused"
        assert "tempo_bpm" in step_frame

        # 6. Stop session
        res = client.post("/api/sessions/stop")
        assert res.status_code == 200
        assert res.json()["status"] == "stopped"
        assert engine.session_id is None

        # 7. List historical sessions
        res = client.get("/api/sessions")
        assert res.status_code == 200
        sessions = res.json()
        assert any(s["session_id"] == session_id for s in sessions)

        # 8. Get session details
        res = client.get(f"/api/sessions/{session_id}")
        assert res.status_code == 200
        details = res.json()
        assert details["metadata"]["session_id"] == session_id
        assert len(details["telemetry"]) >= 2

        # 9. Static UI delivery
        res_ui = client.get("/")
        assert res_ui.status_code == 200
        assert "BrainBeats" in res_ui.text

        # 10. WebSocket live streaming
        with client.websocket_connect("/ws/live") as ws:
            msg = ws.receive_json()
            assert "type" in msg
            ws.send_json({"action": "ping"})
            pong = ws.receive_json()
            assert pong.get("type") == "pong"
