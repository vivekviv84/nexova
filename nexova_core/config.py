import numpy as np
from pathlib import Path
from collections import deque

# ── Paths ──
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
XPQRS_DIR = DATA_DIR / "XPQRS"

# ── Standard Limits ──
IEEE519_THD_LIMIT = 8.0   # % – distribution bus limit
INDIA_CO2_INTENSITY = 0.653  # kg/kWh

# ── Zones & Equipment ──
ZONES = ["feeder_a", "motor_room", "panel_b", "production_floor"]
ZONE_NAMES = {
    "feeder_a": "Feeder A",
    "motor_room": "Motor Room",
    "panel_b": "Panel B",
    "production_floor": "Production Floor"
}

EQUIPMENT = {
    "feeder_a": {"id": "transformer_b", "name": "Transformer B", "profile": "transformer"},
    "motor_room": {"id": "motor_a", "name": "Motor A", "profile": "motor"},
    "panel_b": {"id": "capacitor_bank_c", "name": "Capacitor Bank C", "profile": "capacitor_bank"},
    "production_floor": {"id": "vfd_1", "name": "VFD-1", "profile": "vfd"},
}

# ── Harmonic Fingerprints ──
DEVICE_FINGERPRINTS = {
    "Solar PV Inverter":       np.array([0.05, 0.12, 0.03, 0.18, 0.02, 0.08, 0.01, 0.15, 0.01]),
    "Wind DFIG":               np.array([0.14, 0.06, 0.10, 0.05, 0.12, 0.03, 0.08, 0.02, 0.07]),
    "Variable Frequency Drive": np.array([0.03, 0.22, 0.02, 0.20, 0.01, 0.18, 0.01, 0.16, 0.01]),
    "Arc Furnace":             np.array([0.28, 0.18, 0.12, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04]),
    "Switched-Mode PSU":       np.array([0.02, 0.30, 0.01, 0.25, 0.01, 0.22, 0.01, 0.18, 0.01]),
    "EV Charger":              np.array([0.06, 0.25, 0.04, 0.20, 0.03, 0.15, 0.02, 0.12, 0.01]),
    "LED Lighting Array":      np.array([0.01, 0.35, 0.01, 0.28, 0.01, 0.20, 0.01, 0.14, 0.01]),
}

# Normalise all fingerprints
for _k in DEVICE_FINGERPRINTS:
    _v = DEVICE_FINGERPRINTS[_k]
    _n = np.linalg.norm(_v)
    if _n > 0:
        DEVICE_FINGERPRINTS[_k] = _v / _n

# ── Dataset Mapping ──
XPQRS_CLASSES = [
    "Pure_Sinusoidal", "Sag", "Swell", "Interruption", "Transient",
    "Oscillatory_Transient", "Harmonics", "Harmonics_with_Sag",
    "Harmonics_with_Swell", "Flicker", "Flicker_with_Sag",
    "Flicker_with_Swell", "Sag_with_Oscillatory_Transient",
    "Swell_with_Oscillatory_Transient", "Sag_with_Harmonics",
    "Swell_with_Harmonics", "Notch"
]

XPQRS_SIMPLE_MAP = {
    "Pure_Sinusoidal": "normal",
    "Sag": "voltage_sag", "Swell": "voltage_swell",
    "Interruption": "interruption", "Transient": "transient",
    "Oscillatory_Transient": "oscillatory_transient",
    "Harmonics": "harmonic_distortion",
    "Harmonics_with_Sag": "sag_harmonic", "Harmonics_with_Swell": "swell_harmonic",
    "Flicker": "flicker", "Flicker_with_Sag": "flicker_sag",
    "Flicker_with_Swell": "flicker_swell",
    "Sag_with_Oscillatory_Transient": "sag_transient",
    "Swell_with_Oscillatory_Transient": "swell_transient",
    "Sag_with_Harmonics": "sag_harmonic",
    "Swell_with_Harmonics": "swell_harmonic", "Notch": "notch",
}

# ── Narration Templates ──
CAUSE_TEMPLATES = {
    "voltage_sag": "The power dropped to {sag}% of normal — a sudden voltage dip that can trip sensitive equipment.",
    "voltage_swell": "The voltage rose above safe levels — typically caused by a sudden load disconnection.",
    "harmonic_distortion": "Electrical noise (THD: {thd}%) is distorting the power supply — usually from VFDs or power electronics.",
    "transient": "A sharp voltage spike hit the line — could be switching, lightning, or a loose connection.",
    "sag_harmonic": "The voltage dropped AND has high electrical noise — a compound fault indicating VFD + heavy load.",
    "interruption": "Power was completely interrupted — a full outage on this feeder.",
    "flicker": "Voltage is flickering — rapid small variations that can affect lighting and sensitive equipment.",
    "normal": "Power quality is within normal parameters. No issues detected.",
    "notch": "Notching detected in the waveform — typically caused by power electronic converter switching.",
}

ACTION_TEMPLATES = {
    "voltage_sag": "Check motor control panel in {zone}. If a motor just started, monitor for 2 minutes.",
    "voltage_swell": "Check if equipment in {zone} just disconnected. Inspect capacitor banks.",
    "harmonic_distortion": "Inspect VFDs on this feeder. Transformer aging {aging}x faster than normal.",
    "transient": "Inspect all connections in {zone} for loose terminals. Check surge protectors.",
    "sag_harmonic": "Reduce VFD load in {zone}. Compound fault accelerating equipment degradation.",
    "interruption": "Check breaker status for {zone}. Verify upstream supply.",
    "flicker": "Check for arc furnaces or large intermittent loads near {zone}.",
    "normal": "No action required. System operating normally.",
    "notch": "Inspect power converter switching in {zone}. Verify filter operation.",
}

VISION_RESPONSES = {
    "voltage_sag": "The waveform shows a clear voltage depression — peaks drop to about 70% of normal height. This looks like a sudden power drop, possibly from a large motor starting up.",
    "harmonic_distortion": "The waveform is visibly distorted with jagged edges and flat-topped peaks. This 'squished' appearance suggests electrical noise from power electronics.",
    "transient": "I can see a sharp spike shooting above normal peak levels — a brief transient that could damage sensitive equipment.",
    "sag_harmonic": "Two problems at once — voltage is lower than normal AND the shape is distorted. This combination puts extra stress on equipment.",
    "normal": "The waveform appears smooth and regular with clean sinusoidal pattern. No visible disturbances.",
}

# ── Propagation Graph ──
PLANT_NODES = [
    {"id":"main_bus","name":"Main Bus","x":300,"y":40},
    {"id":"feeder_a","name":"Feeder A","x":140,"y":130},
    {"id":"feeder_b","name":"Feeder B","x":460,"y":130},
    {"id":"transformer_b","name":"Transformer B","x":80,"y":230},
    {"id":"motor_a","name":"Motor A","x":210,"y":230},
    {"id":"vfd_1","name":"VFD-1","x":360,"y":230},
    {"id":"panel_c","name":"Panel C","x":520,"y":230},
    {"id":"compressor_d","name":"Compressor D","x":290,"y":330},
    {"id":"lighting","name":"Lighting","x":460,"y":330},
]
PLANT_EDGES = [
    ("main_bus","feeder_a"),("main_bus","feeder_b"),
    ("feeder_a","transformer_b"),("feeder_a","motor_a"),
    ("feeder_b","vfd_1"),("feeder_b","panel_c"),
    ("vfd_1","compressor_d"),("panel_c","lighting"),
    ("feeder_a","feeder_b"),
]
ZONE_TO_NODE = {"feeder_a":"feeder_a","motor_room":"motor_a","panel_b":"panel_c","production_floor":"vfd_1"}

# ── AI API Configuration (Secure) ──
import os
from dotenv import load_dotenv
load_dotenv()  # Auto-load variables from .env file

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

# ── Timing & State Configuration ──
ATTRIBUTION_HOLD_SECONDS = 30.0

# ── Causal Reasoning Rules ──
CORRELATOR_RULES = [
    ("VFD switching signature", lambda m: m["thd_percent"] > 8 and m["power_factor"] < 0.85,
     "THD spike + PF drop → nonlinear load, likely VFD switching → may overheat transformer → reduces lifespan"),
    ("Motor startup inrush", lambda m: m.get("sag_depth_percent") is not None and m["sag_depth_percent"] < 75,
     "Deep voltage sag → large inductive load startup → may trip downstream equipment"),
    ("Capacitor bank switching", lambda m: m.get("swell_magnitude_percent") is not None and m["thd_percent"] > 5,
     "Voltage swell + harmonics → capacitor bank switching → risk of resonance amplification"),
    ("Generator sync issue", lambda m: abs(m["frequency_hz"] - 50) > 1.5,
     "Frequency deviation → generator synchronization problem → risk of equipment damage"),
    ("Persistent harmonic stress", lambda m: m["thd_percent"] > 12,
     "Sustained high THD → accelerated insulation aging → transformer/motor degradation"),
]
