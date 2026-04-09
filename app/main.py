from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import time
import os
from pathlib import Path

from nexova_core.models import GridSenseClassifier
from nexova_core.engine import GridSenseEngine
from nexova_core.data_loader import load_replay_waveforms
from nexova_core.features import get_waveform
from app.routes import router, state

app = FastAPI(
    title="GridSense AI",
    description="Multi-Zone Power Quality Intelligence System",
    version="1.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static Files & Root ──
BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = BASE_DIR / "static" / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>404: static/index.html not found</h1>", status_code=404)
    with open(index_path) as f:
        return f.read()

# ── Include API Routes ──
app.include_router(router)

# ── Background Logic ──
async def engine_loop():
    """Background task to simulate the grid and update global state."""
    print("Initializing GridSense Engine...")
    try:
        real_waveforms = load_replay_waveforms()
        classifier = GridSenseClassifier()
        classifier.train_on_real_data(max_per_class=150)
        engine = GridSenseEngine(classifier)
        state["engine"] = engine
        print("Engine Ready.")
    except Exception as e:
        print(f"CRITICAL: Failed to initialize engine: {e}")
        return

    while True:
        try:
            demo = state["demo"]
            zone = demo.get("display_zone", "motor_room")
            noise = demo["noise_snr"] if demo["noise"] else None
            
            # Scripted/Fault logic
            ft = None
            scripted = demo.get("scripted_demo", {})
            scripted_active = scripted.get("active") and (time.time() - scripted.get("started_at", 0.0) <= scripted.get("duration_s", 0.0))
            
            if scripted_active:
                ft = "harmonic"
                zone = "feeder_a"
            elif demo["fault"] and time.time() < demo["fault_until"]:
                ft = demo["fault"]
                zone = demo["fault_zone"]
                if zone in demo.get("isolated", []):
                    ft = None
            
            ft_map = {
                "sag": "voltage_sag", "harmonic": "harmonic_distortion", "transient": "transient",
                "compound": "sag_harmonic", "flicker": "flicker", "interruption": "interruption",
                "notch": "notch", "swell": "voltage_swell"
            }
            wf_class = ft_map.get(ft) if ft else "normal"
            
            # Generate waveform & Process
            samples = get_waveform(real_waveforms, wf_class, noise)
            
            if state["engine"]:
                snapshot = state["engine"].process_window(samples, wf_class if ft else None, zone, state["webcam"], demo)
                # Update global snapshots for API/WS access
                state["last_snapshot"] = snapshot
                state["last_attribution"] = snapshot.get("attribution", {})
            
        except Exception as e:
            print(f"Error in engine loop: {e}")
        
        # Consistent pacing: ~5Hz data rate for dashboard
        await asyncio.sleep(0.2)

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    """Feeds the latest engine snapshot to the active dashboard."""
    await websocket.accept()
    print(f"WebSocket Connected: {websocket.client.host}")
    try:
        while True:
            snapshot = state.get("last_snapshot")
            if snapshot:
                await websocket.send_json(snapshot)
            # Sleep slightly longer than loop to avoid overwhelming UI
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        print(f"WebSocket Disconnected: {websocket.client.host}")
    except Exception as e:
        print(f"WebSocket Error: {e}")

@app.on_event("startup")
async def startup():
    # Start the engine in the background
    asyncio.create_task(engine_loop())
