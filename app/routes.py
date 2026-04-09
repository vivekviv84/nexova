from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import time
import asyncio
import logging

logger = logging.getLogger("nexova.api")

class InjectRequest(BaseModel):
    type: str = Field(..., description="Fault type (e.g., sag, harmonic)")
    duration: float = Field(default=10.0, ge=1, le=60)
    zone: str = Field(..., description="Target zone")

class NoiseRequest(BaseModel):
    snr: Optional[float] = Field(None, ge=1, le=100)

class BreakerRequest(BaseModel):
    zone: str

class WebcamRequest(BaseModel):
    enable: Optional[bool] = None
    zone: Optional[str] = None

class AskRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=500)

class VisionRequest(BaseModel):
    fault_class: Optional[str] = None

class WhatIfRequest(BaseModel):
    action: str
    zone: str
    amount: float = Field(..., ge=0, le=500)
from nexova_core.config import ZONES, DEVICE_FINGERPRINTS
from nexova_core.data_loader import load_replay_waveforms
from nexova_core.features import get_waveform
from nexova_core.reporting import (
    generate_llm_text, generate_vision_analysis, 
    build_maintenance_report, build_executive_summary
)
from nexova_core.forensics import (
    simulate_renewable_connection, generate_compliance_report_pdf, 
    harmonic_ledger
)

router = APIRouter()

# State shared between routes and the main loop
# In a real app, these would be in a DB/Redis or a singleton
state = {
    "engine": None,
    "demo": {
        "fault": None, "fault_until": 0, "fault_zone": "motor_room", "display_zone": "motor_room",
        "noise": False, "noise_snr": 20, "isolated": [],
        "scripted_demo": {"active": False, "started_at": 0.0, "duration_s": 18.0}
    },
    "webcam": {
        "enabled": False, "running": False, "zone": "panel_b", "message": "Stable monitoring.",
        "fire_detected": False, "smoke_detected": False, "confidence": 0.0,
        "anomalies": [], "last_frame_b64": "", "last_update": 0.0
    },
    "last_snapshot": {},
    "last_attribution": {}
}

@router.post("/api/inject")
async def inject(req: InjectRequest):
    if not state["engine"]: 
        raise HTTPException(status_code=503, detail="Engine not initialized")
    state["demo"]["fault"] = req.type
    state["demo"]["fault_until"] = time.time() + req.duration
    state["demo"]["fault_zone"] = req.zone
    state["demo"]["display_zone"] = req.zone
    logger.info(f"Fault injected: {req.type} in {req.zone} for {req.duration}s")
    return {"ok": True}

@router.post("/api/demo/start")
async def start_demo_mode():
    state["demo"]["scripted_demo"] = {"active": True, "started_at": time.time(), "duration_s": 18.0}
    state["demo"]["display_zone"] = "feeder_a"
    state["demo"]["fault"] = None
    state["demo"]["fault_until"] = 0
    state["demo"]["fault_zone"] = "feeder_a"
    state["demo"]["noise"] = False
    state["demo"]["isolated"] = []
    return {"ok": True, "zone": "feeder_a", "duration_s": 18.0}

@router.post("/api/demo/reset")
async def reset_demo_mode():
    state["demo"]["scripted_demo"] = {"active": False, "started_at": 0.0, "duration_s": 18.0}
    state["demo"]["fault"] = None
    state["demo"]["fault_until"] = 0
    state["demo"]["fault_zone"] = "motor_room"
    state["demo"]["display_zone"] = "motor_room"
    state["demo"]["noise"] = False
    state["demo"]["isolated"] = []
    return {"ok": True}

@router.post("/api/noise")
async def noise(req: NoiseRequest):
    if req.snr is not None:
        state["demo"]["noise_snr"] = req.snr
        state["demo"]["noise"] = True
    else:
        state["demo"]["noise"] = not state["demo"]["noise"]
    logger.info(f"Noise state updated: {state['demo']['noise']} at SNR {state['demo']['noise_snr']}")
    return {"enabled": state["demo"]["noise"], "snr": state["demo"]["noise_snr"]}

@router.post("/api/breaker")
async def breaker(req: BreakerRequest):
    z = req.zone
    if z in state["demo"]["isolated"]:
        state["demo"]["isolated"].remove(z)
    else:
        state["demo"]["isolated"].append(z)
    logger.info(f"Breaker toggled for zone {z}. Isolated: {state['demo']['isolated']}")
    return {"isolated": state["demo"]["isolated"]}

@router.post("/api/webcam")
async def webcam(req: WebcamRequest):
    enable = req.enable if req.enable is not None else not state["webcam"]["enabled"]
    zone = req.zone if req.zone else state["webcam"]["zone"]
    state["webcam"]["zone"] = zone if zone in ZONES else "panel_b"
    state["webcam"]["enabled"] = enable
    if not enable:
        state["webcam"].update({"fire_detected": False, "smoke_detected": False, "confidence": 0.0, "anomalies": []})
    logger.info(f"Webcam state updated: {enable} for zone {state['webcam']['zone']}")
    return {
        "enabled": state["webcam"]["enabled"],
        "running": state["webcam"]["running"],
        "zone": state["webcam"]["zone"],
        "message": state["webcam"]["message"],
    }

@router.post("/api/ask")
async def ask(req: AskRequest):
    q = req.q
    log = list(state["engine"].incident_log)[-10:] if state["engine"] else []
    
    if not log:
        prompt = f"User asked: {q}. The system is currently nominal."
        answer = await generate_llm_text(prompt, "No incidents detected yet. The system is monitoring normally.")
        return {"a": answer}
    
    context = "\n".join([f"{i['fault_class']} in {i['zone_name']} | severity: {i.get('score', 0)} | cause: {i['cause']}" for i in log])
    prompt = f"You are an expert electrical engineer AI.\n\nSystem recent incidents:\n{context}\n\nUser question:\n{q}\n\nAnswer in simple language. Explaining what is happening, why it matters, and action to take."
    answer = await generate_llm_text(prompt, "AI unavailable.")
    return {"a": answer}

@router.post("/api/vision")
async def vision(req: VisionRequest):
    requested_fault = req.fault_class
    snapshot = state["last_snapshot"]
    # Logic to adjust snapshot for vision if requested_fault present would go here
    return generate_vision_analysis(snapshot)

@router.post("/api/simulate")
async def api_simulate(req: Request):
    body = await req.json()
    source_type = body.get("source_type", "Solar PV Inverter")
    try:
        capacity_mw = float(body.get("capacity_mw", 10.0))
    except (ValueError, TypeError):
        capacity_mw = 10.0
    feeder = body.get("feeder", "feeder_a")
    baseline_thd = body.get("baseline_thd")
    if baseline_thd is None and state["engine"] and state["engine"].metric_history:
        baseline_thd = state["engine"].metric_history[-1]["thd_percent"]
    else:
        baseline_thd = 3.0
    logger.info(f"Simulation requested: {source_type} at {capacity_mw}MW on {feeder}")
    return simulate_renewable_connection(source_type, capacity_mw, feeder, baseline_thd)

@router.get("/api/report_pdf")
async def api_report_pdf():
    zone = state["last_attribution"].get("zone", "feeder_a")
    thd = state["engine"].metric_history[-1]["thd_percent"] if state["engine"] and state["engine"].metric_history else 0.0
    pdf_b64 = generate_compliance_report_pdf(zone, state["last_attribution"], thd)
    return {"pdf_b64": pdf_b64, "zone": zone, "filename": f"gridsense_harmonic_report_{int(time.time())}.pdf"}

@router.post("/api/ticker")
async def ticker():
    log = list(state["engine"].incident_log)[-1:] if state["engine"] else []
    if not log: return {"insight": "System operating nominally."}
    last = log[0]
    prompt = f"Write one SHORT, dramatic, single-sentence insight for a live news ticker about: {last['fault_class']} in {last['zone_name']}. Keyword: insight."
    insight = await generate_llm_text(prompt, f"Alert: {last['fault_class']} in {last['zone_name']}.")
    return {"insight": insight}

@router.get("/api/report")
async def report():
    if not state["engine"]: return {"error": "Engine not initialized"}
    twin = state["engine"].build_asset_twin_snapshot({})
    return build_maintenance_report(list(state["engine"].incident_log), twin)

@router.get("/api/incidents")
async def incidents():
    return list(state["engine"].incident_log) if state["engine"] else []

@router.get("/api/devices")
async def devices():
    # In a real app, this would query connected sensors
    return {"devices": []}

@router.post("/api/whatif")
async def whatif(req: WhatIfRequest):
    if not state["engine"]: 
        raise HTTPException(status_code=503, detail="Engine not initialized")
    logger.info(f"What-If scenario: {req.action} in {req.zone} (amount: {req.amount})")
    return state["engine"].simulate_what_if(req.action, req.zone, req.amount)
