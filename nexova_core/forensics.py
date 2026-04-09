import numpy as np
import time
from collections import deque
from datetime import datetime
import base64
import io
from scipy import fft as sp_fft
from nexova_core.config import DEVICE_FINGERPRINTS, IEEE519_THD_LIMIT, ATTRIBUTION_HOLD_SECONDS
from nexova_core.features import extract_harmonic_vector
from nexova_core.math_utils import compute_aging

# Try optional imports
try:
    from sklearn.decomposition import NMF
    HAS_NMF = True
except ImportError:
    HAS_NMF = False

try:
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors as rl_colors
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# Rolling spectral history for NMF, isolated per zone
ZONE_SPECTRAL_HISTORY = {
    _z: deque(maxlen=60) for _z in ["feeder_a", "motor_room", "panel_b", "production_floor"]
}

ZONE_ATTRIBUTION_STATE = {
    _z: {"published": None, "last_refresh_ts": 0.0}
    for _z in ["feeder_a", "motor_room", "panel_b", "production_floor"]
}

# Harmonic Liability Ledger: {zone: {device_class: {violations, budget_pct, penalty_score}}}
harmonic_ledger = {
    _z: {dev: {"violations": 0, "budget_pct": 0.0, "penalty_score": 0.0} for dev in DEVICE_FINGERPRINTS}
    for _z in ["feeder_a", "motor_room", "panel_b", "production_floor"]
}

def nmf_source_separation(spectral_mat, n_components=3):
    """Decompose harmonic mixtures into latent sources and match against fingerprints."""
    if not HAS_NMF or spectral_mat.shape[0] < 5:
        return _rule_based_attribution(spectral_mat[-1] if len(spectral_mat) else np.zeros(9))

    try:
        W = np.clip(spectral_mat, 0, None)
        model = NMF(n_components=min(n_components, W.shape[0], W.shape[1]),
                    init='nndsvda', max_iter=300, random_state=42)
        model.fit(W)
        components = model.components_
    except Exception:
        return _rule_based_attribution(spectral_mat[-1] if len(spectral_mat) else np.zeros(9))

    results = []
    for comp in components:
        norm = np.linalg.norm(comp)
        if norm < 1e-10:
            continue
        comp_n = comp / norm
        best_dev, best_sim = None, -1.0
        for dev, fp in DEVICE_FINGERPRINTS.items():
            sim = float(np.dot(comp_n, fp))
            if sim > best_sim:
                best_sim, best_dev = sim, dev
        if best_dev:
            results.append({
                "device": best_dev, 
                "confidence": round(float(best_sim), 3),
                "component": comp.tolist()
            })

    seen, deduped = set(), []
    for r in sorted(results, key=lambda x: -x["confidence"]):
        if r["device"] not in seen:
            seen.add(r["device"])
            deduped.append(r)
    return deduped[:4]


def _rule_based_attribution(harmonic_vec):
    """Fallback rule-based attribution."""
    results = []
    for dev, fp in DEVICE_FINGERPRINTS.items():
        sim = max(0.0, float(np.dot(harmonic_vec, fp)))
        results.append({"device": dev, "confidence": round(sim, 3)})
    return sorted(results, key=lambda x: -x["confidence"])[:3]


def _sanitize_sources(sources):
    """Ensure weak matches are reported as low-confidence."""
    if not sources:
        return [{"device": "Unknown / insufficient evidence", "confidence": 0.0}]

    strong_sources = [s for s in sources if s.get("confidence", 0.0) >= 0.35]
    if strong_sources:
        return strong_sources

    best = max(sources, key=lambda x: x.get("confidence", 0.0))
    return [{
        "device": "Unknown / insufficient evidence",
        "confidence": round(float(best.get("confidence", 0.0)), 3),
        "reason": f"Best match {best.get('device', 'unknown')} below confidence threshold.",
    }]


def run_harmonic_attribution(samples, zone, live_thd):
    """Main forensics loop for a zone."""
    if zone not in ZONE_SPECTRAL_HISTORY:
        ZONE_SPECTRAL_HISTORY[zone] = deque(maxlen=60)

    hvec = extract_harmonic_vector(samples)
    ZONE_SPECTRAL_HISTORY[zone].append(hvec)

    mat = np.array(list(ZONE_SPECTRAL_HISTORY[zone]))
    sources = _sanitize_sources(nmf_source_separation(mat))
    
    budget_used = round(live_thd / IEEE519_THD_LIMIT * 100, 1) if IEEE519_THD_LIMIT else 0.0
    update_ledger(zone, sources, live_thd)

    current_payload = {
        "sources": sources,
        "ledger_snapshot": {dev: dict(info) for dev, info in harmonic_ledger.get(zone, {}).items()},
        "thd_budget_used": budget_used,
        "ieee519_limit": IEEE519_THD_LIMIT,
        "zone": zone,
    }
    
    # UI Stability Logic
    state = ZONE_ATTRIBUTION_STATE.get(zone)
    if not state:
        state = {"published": None, "last_refresh_ts": 0.0}
        ZONE_ATTRIBUTION_STATE[zone] = state

    now = time.time()
    if state["published"] is None or (now - state["last_refresh_ts"]) >= ATTRIBUTION_HOLD_SECONDS:
        state["published"] = current_payload
        state["last_refresh_ts"] = now

    return state["published"]


def update_ledger(zone, sources, live_thd):
    """Accumulate violations and penalties in the ledger."""
    if zone not in harmonic_ledger:
        harmonic_ledger[zone] = {dev: {"violations": 0, "budget_pct": 0.0, "penalty_score": 0.0}
                                 for dev in DEVICE_FINGERPRINTS}
    bucket = harmonic_ledger[zone]
    for s in sources:
        dev = s["device"]
        conf = s["confidence"]
        if dev == "Unknown / insufficient evidence":
            continue
        if dev not in bucket:
            bucket[dev] = {"violations": 0, "budget_pct": 0.0, "penalty_score": 0.0}
        bucket[dev]["budget_pct"] = round(conf * live_thd / IEEE519_THD_LIMIT * 100, 1)
        if live_thd > IEEE519_THD_LIMIT and conf > 0.35:
            bucket[dev]["violations"] += 1
            bucket[dev]["penalty_score"] = round(
                bucket[dev]["penalty_score"] + conf * (live_thd - IEEE519_THD_LIMIT) * 0.1, 2)


def simulate_renewable_connection(source_type, capacity_mw, feeder, baseline_thd):
    """Model the PQ impact of a new grid-connected plant."""
    fp = DEVICE_FINGERPRINTS.get(source_type)
    if fp is None:
        return {"error": f"Unknown source type. Available: {list(DEVICE_FINGERPRINTS.keys())}"}

    dominant_harm_mag = float(np.max(fp))
    thd_injection = round(dominant_harm_mag * capacity_mw * 0.35, 2)
    predicted_thd  = round(baseline_thd + thd_injection, 2)

    new_aging = float(compute_aging(predicted_thd))
    old_aging = float(compute_aging(baseline_thd))
    aging_delta = round(new_aging - old_aging, 3)

    harm_ranks = {2: float(fp[0]), 3: float(fp[1]), 5: float(fp[3]), 7: float(fp[5])}
    resonance_flag = bool(max(harm_ranks.values()) > 0.25)
    exceeds_ieee519 = bool(predicted_thd > IEEE519_THD_LIMIT)
    annual_cost_inr = float(round(max(0.0, predicted_thd - IEEE519_THD_LIMIT) * capacity_mw * 150000, 0))

    return {
        "source_type": source_type, "capacity_mw": capacity_mw, "feeder": feeder,
        "baseline_thd": baseline_thd, "thd_injection": thd_injection,
        "predicted_thd": predicted_thd, "exceeds_ieee519": exceeds_ieee519,
        "aging_factor_delta": aging_delta, "resonance_risk": resonance_flag,
        "annual_penalty_inr": annual_cost_inr,
        "recommendation": (
            f"REJECT — predicted THD {predicted_thd:.1f}% violates IEEE 519."
            if exceeds_ieee519 else
            f"APPROVE — predicted THD {predicted_thd:.1f}% within limits."
        )
    }


def generate_compliance_report_pdf(zone, attribution, live_thd):
    """Generate a high-fidelity PDF report."""
    buf = io.BytesIO()
    if not HAS_REPORTLAB:
        report_txt = f"Compliance Report — {zone}\nTHD: {live_thd}%\nStatus: {'NON-COMPLIANT' if live_thd > IEEE519_THD_LIMIT else 'COMPLIANT'}"
        return base64.b64encode(report_txt.encode()).decode()

    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    story = []
    
    # Header
    story.append(Paragraph("GridSense AI — Harmonic Compliance Report", styles['Title']))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  Zone: {zone.replace('_',' ').title()}", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Standard Ref
    story.append(Paragraph(f"Regulatory Standard: IEEE 519-2022 (Limit: {IEEE519_THD_LIMIT}%)", styles['Normal']))
    story.append(Spacer(1, 12))

    # Attribution Table
    if attribution.get("sources"):
        story.append(Paragraph("Identified Sources (NMF Attribution)", styles['Heading2']))
        tdata = [["Device Class", "Confidence", "Budget Used (%)", "Violations"]]
        ledger = attribution.get("ledger_snapshot", {})
        for src in attribution["sources"]:
            dev = src["device"]
            if dev == "Unknown / insufficient evidence": continue
            row = [
                dev, 
                f"{src['confidence']:.1%}", 
                f"{ledger.get(dev, {}).get('budget_pct', 0.0):.1f}", 
                str(ledger.get(dev, {}).get('violations', 0))
            ]
            tdata.append(row)
        
        if len(tdata) > 1:
            t = Table(tdata, colWidths=[2.5*inch, 1.2*inch, 1.2*inch, 1*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#222')),
                ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [rl_colors.whitesmoke, rl_colors.white])
            ]))
            story.append(t)
            story.append(Spacer(1, 12))

    # Final Status
    status_font = "red" if live_thd > IEEE519_THD_LIMIT else "green"
    status_txt = "NON-COMPLIANT" if live_thd > IEEE519_THD_LIMIT else "COMPLIANT"
    story.append(Paragraph(f"Status: <font color='{status_font}'><b>{status_txt}</b></font> (Measured THD: {live_thd:.1f}%)", styles['Normal']))
    
    penalty = sum(info.get("penalty_score", 0.0) for info in attribution.get("ledger_snapshot", {}).values())
    story.append(Paragraph(f"Estimated Annual Liability: ₹{penalty * 12000:,.0f}", styles['Normal']))

    doc.build(story)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

