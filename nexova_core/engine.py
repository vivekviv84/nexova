import numpy as np
import uuid
import time
from collections import deque, Counter
from typing import Optional

import platform
import socket

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from nexova_core.config import (
    ZONES, ZONE_NAMES, EQUIPMENT, PLANT_NODES, PLANT_EDGES, 
    ZONE_TO_NODE, CORRELATOR_RULES
)
from nexova_core.features import compute_pq_metrics, generate_scalogram_b64, classify_waveform_image_cv
from nexova_core.math_utils import compute_aging, compute_sustainability, classify_ieee1159, check_itic
from nexova_core.forensics import run_harmonic_attribution
from nexova_core.visual import detect_cctv_anomalies
from nexova_core.reporting import generate_narration, build_executive_summary

class GridSenseEngine:
    def __init__(self, classifier):
        self.classifier = classifier
        self.incident_log = deque(maxlen=50)
        self.metric_history = deque(maxlen=300)
        self.itic_scatter = []
        self.total_stats = {"incidents": 0, "aging": 0.0, "co2": 0.0, "cost": 0.0}
        self.active_incident = {"fault_class": None, "zone": None, "id": None}
        self.equipment_health = {
            info["id"]: {
                "name": info["name"], "health": 100.0, "aging_hrs": 0.0, "zone": zone, 
                "last_event": "nominal", "profile": info["profile"], "thermal_stress": 0.0, 
                "overload_risk": 0.0, "start_stress": 0.0, "harmonic_load": 0.0, "resonance_risk": 0.0
            }
            for zone, info in EQUIPMENT.items()
        }
        self.prev_metrics = None

    def process_window(self, samples, fault_type=None, zone=None, webcam_state=None, demo_state=None):
        """Main processing pipeline for a single waveform window."""
        zone = zone or "motor_room"
        webcam_state = webcam_state or {"enabled": False, "last_frame_b64": "", "anomalies": [], "message": ""}
        demo_state = demo_state or {"isolated": []}

        # 1. Base Metrics
        metrics = compute_pq_metrics(samples)
        raw_harm_mags = metrics.pop("_raw_harm_mags", [])
        self.metric_history.append(metrics)
        
        # 2. ML Classification
        cls, conf, shap, method = self.classifier.predict(samples)
        if fault_type and fault_type != "normal":
            cls = fault_type
        
        # 3. Standards Compliance
        ieee = classify_ieee1159(metrics)
        mag = metrics.get("sag_depth_percent") or metrics.get("swell_magnitude_percent")
        itic = check_itic(mag)
        aging = compute_aging(metrics["thd_percent"])
        sus = compute_sustainability(metrics["thd_percent"])
        
        # 4. Harmonic Forensics
        attribution = run_harmonic_attribution(samples, zone, metrics["thd_percent"])
        waveform_cv = classify_waveform_image_cv(samples, cls, metrics)
        
        # 5. Diagnostic Analytics
        sensor = self._validate_sensor(metrics)
        trend = self._detect_trend(key="thd_percent")
        inc_pattern = self._detect_incident_pattern()
        correlator = self._run_correlator(metrics)
        
        incident = None
        severity_label = "low"
        if cls != "normal":
            is_new = self._should_start_new_incident(cls, zone)
            zn_name = ZONE_NAMES.get(zone, zone)
            narr = generate_narration(cls, metrics, zone, aging, conf, f"IEEE 1159: {ieee}. ITIC: {itic}.")
            severity = narr["severity"]
            severity_label = "critical" if severity > 60 else "medium" if severity > 30 else "low"
            cost_factor = {60: 500000, 30: 250000}.get(severity, 100000)
            cost = cost_factor * 0.5 / 60
            
            incident = {
                "id": str(uuid.uuid4())[:8], "severity": severity_label, "score": severity,
                "fault_class": cls, "zone": zone, "zone_name": zn_name,
                "cause": narr["cause"], "action": narr["action"], "narration": narr["narration"],
                "confidence": round(conf, 3), "aging_factor": round(aging, 3),
                "aging_impact": f"{aging:.2f}x aging", "co2_impact": f"{sus['co2_this_event']:.5f} kg",
                "cost_impact": f"₹{cost:,.0f}"
            }

            if is_new:
                self.incident_log.append(incident)
                self.total_stats["incidents"] += 1
                self.total_stats["aging"] += aging * 0.001
                self.total_stats["co2"] += sus["co2_this_event"]
                self.total_stats["cost"] += cost
                self._update_equipment(zone, aging, cls, metrics)
                self.active_incident.update({"fault_class": cls, "zone": zone, "id": incident["id"]})
            
            if mag and is_new:
                self.itic_scatter.append({"magnitude": mag, "duration": 1, "status": itic, "fault": cls})

        # 6. Snapshots & Visual Components
        cctv = detect_cctv_anomalies(zone, cls, severity_label, metrics, incident, webcam_state)
        scalo_b64 = generate_scalogram_b64(samples) if cls != "normal" else ""
        propagation = self._compute_propagation(zone, conf if cls != "normal" else 0.0, demo_state)

        result = {
            "version": "GridSense Edge Modular v1.2",
            "timestamp": time.time(),
            "samples": samples[-60:],
            "zone": zone, "zone_name": ZONE_NAMES.get(zone, zone),
            "devices": [self._get_device_telemetry()],
            "metrics": metrics,
            "classification": {
                "fault_class": cls, "confidence": round(conf, 3), "method": method, 
                "shap": shap, "scalogram_b64": scalo_b64, "waveform_cv": waveform_cv
            },
            "sensor": sensor,
            "standards": {"ieee": ieee, "itic": itic, "aging_factor": round(aging, 3), "sustainability": sus},
            "incident": incident,
            "propagation": propagation,
            "cctv": cctv,
            "equipment": [{"id": k, **v} for k, v in self.equipment_health.items()],
            "digital_twin": self.build_asset_twin_snapshot(metrics),
            "fleet": self.build_fleet_snapshot(),
            "itic_scatter": list(self.itic_scatter)[-20:],
            "summary": {
                "incidents": self.total_stats["incidents"], 
                "aging_hrs": round(self.total_stats["aging"], 4), 
                "co2_kg": round(self.total_stats["co2"], 5), 
                "cost_inr": round(self.total_stats["cost"], 0)
            },
            "correlator": correlator,
            "trend": trend,
            "incident_pattern": inc_pattern,
            "attribution": attribution,
            "ledger": attribution.get("ledger_snapshot", {})
        }
        result["executive_summary"] = build_executive_summary(result)
        return result

    def build_asset_twin_snapshot(self, metrics=None):
        """Construct digital twin payloads for all equipment."""
        metrics = metrics or {}
        snapshots = []
        for eq_id, eq in self.equipment_health.items():
            profile = eq["profile"]
            if profile == "transformer":
                risk_driver = eq["thermal_stress"]
                concern = "thermal aging stress"
                checks = ["Oil temp trend", "Filter loading"]
            elif profile == "motor":
                risk_driver = eq["start_stress"]
                concern = "startup inrush stress"
                checks = ["Starter history", "Shaft load"]
            elif profile == "vfd":
                risk_driver = eq["harmonic_load"]
                concern = "harmonic injection"
                checks = ["DC bus ripple", "Input reactor"]
            else:
                risk_driver = eq["resonance_risk"]
                concern = "resonance risk"
                checks = ["Capacitor staging", "Swell coincidence"]

            snapshots.append({
                "id": eq_id, "name": eq["name"], "zone": eq["zone"],
                "profile": profile, "health": round(eq["health"], 1),
                "aging_hrs": round(eq["aging_hrs"], 4), "concern": concern,
                "risk_driver": round(risk_driver, 1), "last_event": eq["last_event"],
                "recommended_checks": checks
            })
        return snapshots

    def build_fleet_snapshot(self):
        """Multi-site risk metrics for executive view."""
        worst_asset = min(self.equipment_health.values(), key=lambda eq: eq["health"])
        return {"sites": [
            {
                "site": "Production Plant", 
                "risk_score": 100 - worst_asset["health"], 
                "worst_asset": worst_asset["name"], 
                "top_fault": worst_asset["last_event"],
                "cost_inr": self.total_stats["cost"],
                "co2_kg": self.total_stats["co2"]
            },
            {"site": "Secondary Hub", "risk_score": 5, "worst_asset": "Transformer-7", "top_fault": "nominal", "cost_inr": 2000, "co2_kg": 0.5}
        ]}

    def _should_start_new_incident(self, cls, zone):
        if cls == "normal":
            self.active_incident = {"fault_class": None, "zone": None, "id": None}
            return False
        return self.active_incident["fault_class"] != cls or self.active_incident["zone"] != zone

    def _update_equipment(self, zone, aging_factor, fault_desc, metrics):
        info = EQUIPMENT.get(zone)
        if not info: return
        eq = self.equipment_health.get(info["id"])
        if not eq: return
        eq["aging_hrs"] += aging_factor * 0.001
        eq["health"] = max(0, 100 - eq["aging_hrs"] * 8)
        eq["last_event"] = fault_desc
        # Stress profile updates
        if eq["profile"] == "vfd":
             eq["harmonic_load"] = round(min(100.0, metrics.get("thd_percent", 0) * 5), 2)
        elif eq["profile"] == "motor" and "sag" in fault_desc:
             eq["start_stress"] += 5.0
        elif eq["profile"] == "transformer":
             eq["thermal_stress"] += aging_factor * 0.5

    def _validate_sensor(self, metrics):
        health = 100.0
        flags = []
        if metrics["rms_voltage"] > 450: flags.append("rms_impossible"); health -= 40
        if metrics.get("thd_percent", 0) > 90: flags.append("thd_impossible"); health -= 30
        status = "sensor_error" if health < 50 else "inconsistent" if health < 85 else "valid"
        return {"status": status, "flags": flags, "health": max(0, health)}

    def _detect_trend(self, key="thd_percent", window=20):
        if len(self.metric_history) < window: return None
        recent = [m[key] for m in list(self.metric_history)[-window:]]
        slope = (recent[-1] - recent[0]) / window
        if slope > 0.4: return f"{key.replace('_', ' ').title()} rising rapidly"
        if slope < -0.4: return f"{key.replace('_', ' ').title()} resolving"
        return None

    def _run_correlator(self, metrics):
        for name, check, explanation in CORRELATOR_RULES:
            try:
                if check(metrics): return {"pattern": name, "explanation": explanation}
            except Exception: continue
        return None

    def _detect_incident_pattern(self):
        if len(self.incident_log) < 3: return None
        recent = list(self.incident_log)[-10:]
        classes = [i["fault_class"] for i in recent]
        counts = Counter(classes)
        mc, count = counts.most_common(1)[0]
        if count >= 3 and mc != "normal":
            return f"{count}x {mc.replace('_', ' ')} detected recently — indicates instability"
        return None

    def simulate_what_if(self, action: str, zone: str, amount: float) -> dict:
        """Simulate the effect of an operator action without applying it to live state."""
        base_thd = self.metric_history[-1]["thd_percent"] if self.metric_history else 5.0
        base_pf  = self.metric_history[-1]["power_factor"] if self.metric_history else 0.92

        # Calculate hypothetical metrics based on action
        if action == "isolate":
            sim_thd = base_thd * 0.4
            sim_pf  = min(0.99, base_pf + 0.05)
            note    = f"Isolating {zone} would reduce harmonic coupling. Downstream loads shift to redundant path."
            risk_delta = -40
        elif action == "thd_up":
            sim_thd = base_thd + (amount / 10)
            sim_pf  = max(0.6, base_pf - 0.08)
            note    = f"Raising THD by {amount}% simulates adding a non-linear load. Risk of resonance increases."
            risk_delta = +35
        elif action == "reduce_load":
            sim_thd = max(1.0, base_thd - (amount / 20))
            sim_pf  = min(0.99, base_pf + 0.03)
            note    = f"Reducing load by {amount}% lowers stress on assets and improves power factor."
            risk_delta = -25
        else:
            sim_thd = base_thd
            sim_pf  = base_pf
            note    = "Unknown action — no change simulated."
            risk_delta = 0

        from nexova_core.math_utils import compute_aging
        predicted_aging = compute_aging(sim_thd)
        predicted_severity = max(0, min(100, 30 + risk_delta + (sim_thd * 2)))
        predicted_cost     = predicted_severity * 3500

        # Propagation preview for what-if scenario
        active_node     = ZONE_TO_NODE.get(zone)
        prop_nodes      = []
        affected_nodes  = {active_node}
        # Find neighbours
        for s, t in PLANT_EDGES:
            if s == active_node: affected_nodes.add(t)
            if t == active_node: affected_nodes.add(s)

        for n in PLANT_NODES:
            if n["id"] == active_node:
                status = "critical" if risk_delta > 0 else "warning"
            elif n["id"] in affected_nodes:
                status = "warning" if risk_delta > 0 else "normal"
            else:
                status = "normal"
            risk = max(0.0, (predicted_severity / 100) * (0.9 if n["id"] == active_node else 0.45 if n["id"] in affected_nodes else 0.1))
            prop_nodes.append({**n, "status": status, "risk": round(risk, 2)})

        prop_edges = [{"source": s, "target": t, "active": (s in affected_nodes or t in affected_nodes) and risk_delta > 0} for s, t in PLANT_EDGES]

        return {
            "action": action, "zone": zone, "amount": amount,
            "predicted_severity": round(predicted_severity, 1),
            "predicted_aging_factor": round(predicted_aging, 2),
            "predicted_cost_inr": round(predicted_cost, 0),
            "predicted_thd": round(sim_thd, 2),
            "predicted_pf": round(sim_pf, 3),
            "notes": [note],
            "recommendation": "Proceed with caution — monitor THD and PF for 60s after action." if risk_delta > 0 else "Action is safe to execute. Verify via live waveform post-action.",
            "propagation": {"nodes": prop_nodes, "edges": prop_edges, "narrative": f"{len(affected_nodes)} nodes affected in cascade scenario."}
        }

    def _compute_propagation(self, zone: str, severity: float = 0.8, demo_state: dict = None) -> dict:
        """Compute realistic fault propagation that cascades risk through connected nodes."""
        active_node = ZONE_TO_NODE.get(zone)
        if not active_node or severity < 0.05:
            # No active fault — all nodes nominal
            nodes = [{**n, "status": "normal", "risk": 0.0} for n in PLANT_NODES]
            edges = [{"source": s, "target": t, "active": False} for s, t in PLANT_EDGES]
            return {"nodes": nodes, "edges": edges}

        # BFS to find nodes at 1-hop and 2-hop distance
        adjacency = {n["id"]: [] for n in PLANT_NODES}
        for s, t in PLANT_EDGES:
            adjacency[s].append(t)
            adjacency[t].append(s)

        first_hop  = set(adjacency.get(active_node, []))
        second_hop = set()
        for n in first_hop:
            second_hop.update(adjacency.get(n, []))
        second_hop.discard(active_node)
        second_hop -= first_hop

        nodes = []
        for n in PLANT_NODES:
            nid = n["id"]
            if nid == active_node:
                status = "critical" if severity > 0.5 else "warning"
                risk   = round(severity, 2)
            elif nid in first_hop:
                status = "warning" if severity > 0.3 else "normal"
                risk   = round(severity * 0.55, 2)
            elif nid in second_hop:
                status = "warning" if severity > 0.65 else "normal"
                risk   = round(severity * 0.25, 2)
            else:
                status = "normal"
                risk   = round(severity * 0.08, 2)
            nodes.append({**n, "status": status, "risk": risk})

        active_set = {active_node} | first_hop
        edges = [
            {"source": s, "target": t, "active": (s in active_set or t in active_set) and severity > 0.1}
            for s, t in PLANT_EDGES
        ]
        return {"nodes": nodes, "edges": edges}

    def _get_device_telemetry(self):
        cpu = psutil.cpu_percent() if HAS_PSUTIL else 15.0
        mem = psutil.virtual_memory().percent if HAS_PSUTIL else 42.0
        return {
            "id": "edge-01", "name": platform.node(), "status": "online",
            "cpu_percent": cpu, "memory_percent": mem,
            "voltage_v": 230.1, "current_a": 12.4
        }

