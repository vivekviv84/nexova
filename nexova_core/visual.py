import cv2
import numpy as np
import base64
from nexova_core.config import ZONE_NAMES

# Try optional OpenCV
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

def make_cctv_frame_svg(zone, anomalies, severity):
    zone_label = ZONE_NAMES.get(zone, zone).replace("&", "and")
    glow = "#ff4d4d" if severity == "critical" else "#ffb833" if severity == "medium" else "#00e68a"
    haze = "0.35" if "smoke" in anomalies else "0.0"
    spark = "<circle cx='150' cy='48' r='8' fill='#ffb833' opacity='0.9'/><circle cx='160' cy='54' r='4' fill='#fff' opacity='0.8'/>" if "sparks" in anomalies else ""
    door = "<rect x='132' y='44' width='20' height='66' fill='none' stroke='#ff4d4d' stroke-width='4'/>" if "panel_door_open" in anomalies else "<rect x='138' y='44' width='14' height='66' fill='#1a1a1a' stroke='#666' stroke-width='2'/>"
    person = "<circle cx='52' cy='70' r='10' fill='#d9d9d9'/><rect x='44' y='82' width='16' height='28' rx='3' fill='#d9d9d9'/>" if "technician_present" in anomalies or "unauthorized_access" in anomalies else ""
    badge = "UNAUTH" if "unauthorized_access" in anomalies else "TECH" if "technician_present" in anomalies else "CLEAR"
    return (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 200 120'>"
        "<rect width='200' height='120' fill='#111'/>"
        f"<rect x='20' y='22' width='48' height='72' rx='6' fill='#0f0f0f' stroke='{glow}' stroke-width='2' opacity='0.9'/>"
        f"<rect x='128' y='36' width='34' height='78' rx='4' fill='#111' stroke='{glow}' stroke-width='2' opacity='0.9'/>"
        f"{door}{person}{spark}"
        f"<rect x='0' y='0' width='200' height='120' fill='#c9c9c9' opacity='{haze}'/>"
        f"<text x='8' y='12' font-size='10' fill='#fff'>CAM {zone_label}</text>"
        "</svg>"
    )


def detect_cctv_anomalies(zone, fault_class, severity, metrics, incident, webcam_state):
    anomalies = []
    summary = "Visual scene stable."
    confidence = 0.58

    if fault_class in {"transient", "sag_transient"}:
        anomalies.append("sparks")
    if fault_class in {"interruption", "voltage_sag"} and severity in {"medium", "critical"}:
        anomalies.append("panel_door_open")
    if metrics["thd_percent"] > 8 or fault_class in {"harmonic_distortion", "sag_harmonic", "swell_harmonic"}:
        anomalies.append("smoke")
    
    if webcam_state["enabled"] and webcam_state["last_frame_b64"]:
        anomalies = list(dict.fromkeys(webcam_state["anomalies"] + anomalies))
        summary = webcam_state["message"]
        frame_b64 = webcam_state["last_frame_b64"]
        frame_kind = "webcam_jpg"
    else:
        frame_b64 = base64.b64encode(make_cctv_frame_svg(zone, anomalies, severity).encode("utf-8")).decode("ascii")
        frame_kind = "mock_svg"
    
    return {
        "zone": zone, "zone_name": ZONE_NAMES.get(zone, zone),
        "anomalies": anomalies, "summary": summary,
        "frame_b64": frame_b64, "frame_kind": frame_kind,
        "status": "alert" if anomalies else "clear",
    }


def analyze_fire_frame(frame, prev_frame=None):
    if not HAS_CV2 or frame is None:
        return {"fire": False, "smoke": False, "confidence": 0.0, "overlay": None, "anomalies": []}
    
    small = cv2.resize(frame, (320, 240))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    
    fire_mask = cv2.inRange(hsv, np.array([0, 110, 150]), np.array([40, 255, 255]))
    smoke_mask = cv2.inRange(hsv, np.array([0, 0, 95]), np.array([179, 40, 190]))
    
    confidence = min(0.99, float(np.count_nonzero(fire_mask) / combined.size * 20.0)) if 'combined' in locals() else 0.0
    
    return {
        "fire": np.count_nonzero(fire_mask) > 500,
        "smoke": np.count_nonzero(smoke_mask) > 1000,
        "confidence": round(confidence, 3),
        "overlay": small,
        "anomalies": (["sparks"] if np.count_nonzero(fire_mask) > 500 else []) + (["smoke"] if np.count_nonzero(smoke_mask) > 1000 else [])
    }


def encode_frame_b64(frame):
    if not HAS_CV2 or frame is None:
        return ""
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")
