import numpy as np
from nexova_core.config import IEEE519_THD_LIMIT

def classify_ieee1159(metrics):
    if metrics.get("sag_depth_percent"):
        return "Instantaneous Sag (0.5-30 cycles)"
    if metrics.get("swell_magnitude_percent"):
        return "Instantaneous Swell (0.5-30 cycles)"
    if metrics["thd_percent"] > 5:
        return "Harmonic Distortion (steady-state)"
    if abs(metrics["frequency_hz"] - 50) > 0.5:
        return "Frequency Variation"
    return "Normal"


def check_itic(magnitude_pct, duration_cycles=1):
    if magnitude_pct is None:
        return "not_applicable"
    dur_s = duration_cycles / 50.0
    if dur_s < 0.02:
        lower, upper = 70, 200
    elif dur_s < 0.5:
        lower, upper = 80, 120
    else:
        lower, upper = 90, 110
    return "outside_tolerance" if (magnitude_pct < lower or magnitude_pct > upper) else "inside_tolerance"


def compute_aging(thd_pct, load=0.8, ambient=35):
    """IEEE C57.91 accelerated aging factor."""
    k = 1 + (thd_pct/100)**2
    hst = ambient + 65 * (load * np.sqrt(k))**0.8
    faa = float(np.exp(15000/383 - 15000/(hst+273)))
    return faa


def compute_sustainability(thd_pct, duration_s=0.02, load_kva=100):
    k = 1 + (thd_pct/100)**2
    base_loss = load_kva * 0.015
    excess_kw = base_loss * (k - 1)
    co2_per_hr = excess_kw * 0.653
    return {
        "excess_loss_kw": round(excess_kw, 4),
        "co2_kg_per_hr": round(co2_per_hr, 4),
        "co2_this_event": round(co2_per_hr * duration_s / 3600, 6),
    }

def compute_severity(fault_class, metrics, aging, confidence):
    score = 0
    if "sag" in fault_class: score += 25
    if "harmonic" in fault_class: score += 20
    if "transient" in fault_class: score += 30
    if "interruption" in fault_class: score += 40
    if metrics["thd_percent"] > 15: score += 15
    if metrics.get("sag_depth_percent") and metrics["sag_depth_percent"] < 70: score += 15
    if aging > 1.5: score += 10
    score = int(min(score * confidence, 100))
    return score
