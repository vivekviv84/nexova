import asyncio
import json
import base64
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from nexova_core.config import (
    CAUSE_TEMPLATES, ACTION_TEMPLATES, VISION_RESPONSES, 
    GROQ_API_KEY, GROQ_MODEL, ZONE_NAMES
)
from nexova_core.math_utils import compute_severity

logger = logging.getLogger("nexova.reporting")

async def generate_llm_text(prompt: str, fallback: str, context_metrics: Optional[Dict[str, Any]] = None) -> str:
    """Call Groq for ultra-fast AI narration, with a robust local expert fallback."""
    # ── Attempt Groq API Call ──
    if not GROQ_API_KEY:
        logger.debug("No Groq API key configured. Using local expert fallback.")
    else:
        try:
            import requests
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            }
            data = {
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
            }
            response = requests.post(url, headers=headers, json=data, timeout=5)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
            else:
                logger.warning(f"Groq API error {response.status_code}: {response.text[:100]}")
        except requests.exceptions.Timeout:
            logger.warning("Groq API request timed out. Falling back to local expert.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Groq API request failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during Groq API call: {e}")

    # ── Local Rule-Based Expert (Smart Fallback) ──
    p_lower = prompt.lower()
    
    # Heuristics for common questions
    if "damage" in p_lower or "problem" in p_lower or "cause" in p_lower:
        if "harmonic" in p_lower:
            return "Based on spectral analysis, the primary cause of equipment stress is non-linear loading from VFDs and power converters on this feeder. Continuous THD above 8% leads to accelerated winding degradation."
        if "sag" in p_lower:
            return "Voltage sags are typically caused by large motor start-ups or transient short-circuit conditions elsewhere in the grid. Frequent sags indicate a weak supply point or inadequate soft-starting."
        return "I am detecting power quality deviations that suggest intermittent loading stress in the active zone. Check for loose connections or aging insulation."
    
    if "health" in p_lower or "status" in p_lower:
        return "The asset digital twin indicates stable performance, though persistent harmonics are slowly reducing the remaining useful life (RUL) of the transformer insulation."

    if "insight" in p_lower:
        return f"Operational Insight: Grid stability is currently {'nominal' if 'normal' in p_lower else 'under stress'}. Recommend monitoring THD trends."

    return fallback


def generate_narration(fault_class, metrics, zone, aging_factor, confidence, standards_result, pattern=None, trend=None, incident_pattern=None):
    """Build a rich technical narration for the detected fault."""
    zn = ZONE_NAMES.get(zone, zone)
    # Match the base fault class to our templates
    fc_simple = next((k for k in CAUSE_TEMPLATES if k in fault_class), "normal")
    
    cause = CAUSE_TEMPLATES.get(fc_simple, f"A {fault_class} event was detected.").format(
        sag=metrics.get("sag_depth_percent", "?"), 
        thd=metrics.get("thd_percent", "?"), 
        zone=zn
    )
    
    action = ACTION_TEMPLATES.get(fc_simple, f"Inspect equipment in {zn}.").format(
        zone=zn, 
        aging=f"{aging_factor:.1f}"
    )
    
    parts = [f"<b>{cause}</b> ({confidence:.0%} confidence)"]
    if pattern: 
        parts.append(f"Root Cause: {pattern['explanation']}")
    if aging_factor > 1.2: 
        parts.append(f"Asset Impact: Aging {aging_factor:.1f}x faster. {standards_result}")
    if trend: 
        parts.append(f"Trend Observation: {trend}")
    if incident_pattern: 
        parts.append(f"⚠ {incident_pattern}")
    
    parts.append(f"Next Action: {action}")
    
    return {
        "cause": cause, 
        "action": action, 
        "narration": "\n".join(parts),
        "severity": compute_severity(fault_class, metrics, aging_factor, confidence),
    }


def build_executive_summary(snapshot):
    """Condense whole-system state into a executive snapshot."""
    classification = snapshot.get("classification", {})
    incident = snapshot.get("incident") or {}
    attribution = snapshot.get("attribution", {})
    
    # Get top source from attribution
    sources = attribution.get("sources", [])
    top_source = sources[0].get("device", "Unknown") if sources else "Normal Load"

    confidence = classification.get("confidence", incident.get("confidence", 0.0))
    severity = incident.get("severity", "nominal")
    
    return {
        "fault": classification.get("fault_class", "normal").replace("_", " ").title(),
        "zone": snapshot.get("zone_name", "Unknown"),
        "likely_source": top_source,
        "severity": severity.title(),
        "severity_score": incident.get("score", 0),
        "recommended_action": incident.get("action", "Monitor system for deviations."),
    }


def build_maintenance_report(incident_log, digital_twin):
    """Prepare a structured report for maintenance personnel."""
    incident = incident_log[-1] if incident_log else None
    impacted = [a for a in digital_twin if a.get("risk_driver", 0) > 20 or a.get("last_event")]
    
    return {
        "generated_at": datetime.now().isoformat(),
        "incident_summary": {
            "fault_class": incident["fault_class"] if incident else "normal",
            "zone": incident["zone_name"] if incident else "System",
            "severity": incident["severity"] if incident else "low",
            "narration": incident["narration"] if incident else "Power quality within nominal limits.",
        },
        "affected_assets": impacted,
        "probable_root_cause": incident["pattern"] if incident and "pattern" in incident else "None identified",
    }


def generate_vision_analysis(snapshot):
    """Vision AI Assessment of waveform patterns."""
    classification = snapshot.get("classification", {})
    fault_class = classification.get("fault_class", "normal")
    narrative = VISION_RESPONSES.get(fault_class, "Waveform pattern appears sinusoidal and stable.")
    
    return {
        "headline": f"Vision AI Assessment - {fault_class.replace('_',' ').title()}",
        "analysis": narrative,
        "observations": [
            f"Pattern Match: {fault_class.replace('_',' ')}", 
            f"Edge Confidence: {classification.get('confidence', 0):.0%}"
        ],
        "recommendation": "Perform physical inspection if THD thresholds persist."
    }

