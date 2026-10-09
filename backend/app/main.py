"""FastAPI REST and WebSocket server for BrainBeats closed-loop biofeedback."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .closed_loop import ClosedLoopEngine

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Global engine instance
engine = ClosedLoopEngine()

# Connected WebSocket clients
connected_clients: set[WebSocket] = set()
background_task: asyncio.Task | None = None
loop_interval: float = 2.0


async def broadcast_frame(frame_data: dict[str, Any]) -> None:
    """Send telemetry frame to all connected WebSocket clients."""
    if not connected_clients:
        return
    disconnected = set()
    for client in connected_clients:
        try:
            await client.send_json({"type": "telemetry", "data": frame_data})
        except Exception:
            disconnected.add(client)
    connected_clients.difference_update(disconnected)


async def engine_loop() -> None:
    """Background task continuously advancing the biofeedback loop when active."""
    global loop_interval
    while True:
        try:
            if engine.is_running:
                # Run CPU-bound step in threadpool so asyncio event loop isn't blocked
                frame = await asyncio.to_thread(engine.step, render_audio=True, beats=16)
                await broadcast_frame(frame.to_dict())
            await asyncio.sleep(loop_interval)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error in closed-loop tick: {e}")
            await asyncio.sleep(loop_interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global background_task
    background_task = asyncio.create_task(engine_loop())
    yield
    if background_task:
        background_task.cancel()
        try:
            await background_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="BrainBeats API",
    description="Real-time EEG to Generative Music Closed-Loop Biofeedback",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request schemas
class StartSessionRequest(BaseModel):
    preset: str = Field(default="calm", description="calm, neutral, or aroused")
    target_goal: str = Field(default="calm", description="calm or focus")
    interval: float = Field(default=2.0, ge=0.5, le=10.0, description="Loop step interval in seconds")


class PresetRequest(BaseModel):
    preset: str = Field(..., description="calm, neutral, or aroused")


class TargetGoalRequest(BaseModel):
    target_goal: str = Field(..., description="calm or focus")


class IngestChunkRequest(BaseModel):
    data: list[list[float]] = Field(..., description="2D array of (samples, channels)")


# REST Endpoints
@app.get("/api/status")
def get_status() -> dict[str, Any]:
    """Get system and engine health status."""
    return {
        "status": "ready",
        "is_running": engine.is_running,
        "session_id": engine.session_id,
        "preset": engine.preset,
        "target_goal": engine.target_goal,
        "connected_clients": len(connected_clients),
        "latest_frame_id": engine.latest_frame.frame_id if engine.latest_frame else None,
    }


@app.post("/api/sessions/start")
def start_session(req: StartSessionRequest) -> dict[str, Any]:
    """Start a new closed-loop biofeedback session."""
    global loop_interval
    loop_interval = req.interval
    meta = engine.start_session(preset=req.preset, target_goal=req.target_goal)
    # Generate initial frame immediately
    frame = engine.step(render_audio=True)
    return {"status": "started", "session": meta, "initial_frame": frame.to_dict()}


@app.post("/api/sessions/stop")
def stop_session() -> dict[str, Any]:
    """Stop the currently active session."""
    res = engine.stop_session()
    return {"status": "stopped", "summary": res}


@app.post("/api/sessions/preset")
def set_preset(req: PresetRequest) -> dict[str, Any]:
    """Dynamically switch EEG preset."""
    updated = engine.set_preset(req.preset)
    return {"preset": updated}


@app.post("/api/sessions/goal")
def set_target_goal(req: TargetGoalRequest) -> dict[str, Any]:
    """Set biofeedback target goal."""
    updated = engine.set_target_goal(req.target_goal)
    return {"target_goal": updated}


@app.post("/api/sessions/step")
def trigger_step() -> dict[str, Any]:
    """Manually execute a single closed-loop cycle."""
    frame = engine.step(render_audio=True)
    return frame.to_dict()


@app.post("/api/sessions/ingest")
def ingest_chunk(req: IngestChunkRequest) -> dict[str, Any]:
    """Ingest custom external EEG samples."""
    try:
        import numpy as np
        data_arr = np.array(req.data, dtype=np.float64)
        engine.ingest_chunk(data_arr)
        return {"status": "ingested", "samples": len(data_arr)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/sessions/current")
def get_current_state() -> dict[str, Any]:
    """Get the active session's latest telemetry frame."""
    return {
        "is_running": engine.is_running,
        "session_id": engine.session_id,
        "preset": engine.preset,
        "target_goal": engine.target_goal,
        "latest_frame": engine.latest_frame.to_dict() if engine.latest_frame else None,
    }


@app.get("/api/sessions")
def list_sessions() -> list[dict[str, Any]]:
    """List historical sessions recorded in storage."""
    return engine.store.list_sessions()


@app.get("/api/sessions/{session_id}")
def get_session_details(session_id: str) -> dict[str, Any]:
    """Get session metadata and telemetry points."""
    meta = engine.store.get_session(session_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Session not found")
    telemetry = engine.store.get_telemetry(session_id, limit=200)
    return {"metadata": meta, "telemetry": telemetry}


@app.get("/api/sessions/{session_id}/audio/{filename}")
def get_session_audio(session_id: str, filename: str):
    """Retrieve WAV audio recording."""
    path = engine.store.get_music_artifact_path(session_id, filename)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path, media_type="audio/wav", filename=filename)


@app.get("/api/sessions/{session_id}/midi/{filename}")
def get_session_midi(session_id: str, filename: str):
    """Retrieve generated MIDI recording."""
    path = engine.store.get_music_artifact_path(session_id, filename)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="MIDI file not found")
    return FileResponse(path, media_type="audio/midi", filename=filename)


# WebSocket Endpoint
@app.websocket("/ws/live")
async def websocket_live_endpoint(websocket: WebSocket):
    """Real-time bi-directional streaming endpoint for the web dashboard."""
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        # Send current status on connect
        if engine.latest_frame:
            await websocket.send_json({"type": "telemetry", "data": engine.latest_frame.to_dict()})
        else:
            await websocket.send_json({"type": "status", "data": get_status()})

        while True:
            msg = await websocket.receive_json()
            action = msg.get("action")
            if action == "set_preset":
                new_preset = engine.set_preset(msg.get("preset", "calm"))
                await websocket.send_json({"type": "preset_changed", "preset": new_preset})
            elif action == "set_goal":
                new_goal = engine.set_target_goal(msg.get("goal", "calm"))
                await websocket.send_json({"type": "goal_changed", "goal": new_goal})
            elif action == "step":
                frame = await asyncio.to_thread(engine.step, render_audio=True)
                await broadcast_frame(frame.to_dict())
            elif action == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        connected_clients.discard(websocket)
    except Exception:
        connected_clients.discard(websocket)


# Mount static assets
if (STATIC_DIR / "index.html").exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "BrainBeats API is active. UI index.html not yet installed in static directory."}


@app.get("/app")
def serve_app():
    return serve_index()


def main():
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
