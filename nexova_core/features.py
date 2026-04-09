from scipy import fft as sp_fft
import numpy as np

def extract_harmonic_vector(samples, fs=5000) -> np.ndarray:
    """Return h2..h10 normalised magnitude vector from waveform samples."""
    s = np.array(samples)
    n = len(s)
    freqs = sp_fft.rfftfreq(n, 1.0 / fs)
    mags  = np.abs(sp_fft.rfft(s)) / n
    vec = []
    for h in range(2, 11):
        idx = int(np.argmin(np.abs(freqs - 50.0 * h)))
        vec.append(float(mags[idx]) if idx < len(mags) else 0.0)
    arr = np.array(vec, dtype=float)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 1e-10 else arr
import io
import base64

# Try optional imports
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

def compute_cwt_features(samples, fs=5000):
    """CWT scalogram features from waveform."""
    if not HAS_PYWT or len(samples) < 10:
        return {}
    scales = np.arange(1, min(32, len(samples)//2))
    try:
        coefficients = pywt.cwt(samples, scales, 'morl')[0]
    except Exception:
        return {}
    
    power = np.abs(coefficients) ** 2
    n = len(scales)
    low = power[:n//3].sum()
    mid = power[n//3:2*n//3].sum()
    high = power[2*n//3:].sum()
    total = low + mid + high + 1e-10
    flat = power.flatten()
    
    return {
        "cwt_low": float(low/total), "cwt_mid": float(mid/total),
        "cwt_high": float(high/total), "cwt_total": float(np.log1p(total)),
        "cwt_max": float(np.max(power)), "cwt_std": float(np.std(flat)),
        "cwt_skew": float(np.mean(((flat-np.mean(flat))/(np.std(flat)+1e-10))**3)),
        "cwt_kurt": float(np.mean(((flat-np.mean(flat))/(np.std(flat)+1e-10))**4)),
        "cwt_peak_scale": float(np.unravel_index(np.argmax(power), power.shape)[0]),
    }


def compute_td_features(samples):
    """Time-domain features."""
    s = np.array(samples)
    if len(s) == 0:
        return {
            "td_rms": 0.0, "td_peak": 0.0, "td_crest": 0.0,
            "td_std": 0.0, "td_zc": 0.0, "td_kurt": 0.0, "td_skew": 0.0
        }
    rms = np.sqrt(np.mean(s**2)) + 1e-10
    return {
        "td_rms": float(rms), "td_peak": float(np.max(np.abs(s))),
        "td_crest": float(np.max(np.abs(s)) / rms),
        "td_std": float(np.std(s)),
        "td_zc": float(np.sum(np.diff(np.sign(s)) != 0)),
        "td_kurt": float(np.mean(((s-np.mean(s))/(np.std(s)+1e-10))**4)),
        "td_skew": float(np.mean(((s-np.mean(s))/(np.std(s)+1e-10))**3)),
    }


def extract_features(samples, fs=5000):
    """All features combined."""
    cwt = compute_cwt_features(samples, fs)
    td = compute_td_features(samples)
    return {**cwt, **td}


def compute_scalogram_signature(samples, fs=5000):
    """Compact visual signature that acts like a CV descriptor for waveform images."""
    if not HAS_PYWT or len(samples) < 10:
        return {}
    scales = np.arange(1, min(32, len(samples)//2))
    try:
        coeffs = pywt.cwt(np.array(samples), scales, 'morl')[0]
    except Exception:
        return {}
    power = np.abs(coeffs) ** 2
    rows, cols = power.shape
    total = float(power.sum()) + 1e-10
    high_band = float(power[2 * rows // 3:, :].sum() / total)
    mid_band = float(power[rows // 3:2 * rows // 3, :].sum() / total)
    vertical_peaks = np.max(power, axis=0)
    transient_columns = float(np.mean(vertical_peaks > (vertical_peaks.mean() + vertical_peaks.std())))
    edge_energy = float((power[:, :cols // 5].sum() + power[:, -cols // 5:].sum()) / total)
    center_energy = float(power[:, cols // 3:2 * cols // 3].sum() / total)
    return {
        "high_band_ratio": round(high_band, 4),
        "mid_band_ratio": round(mid_band, 4),
        "transient_columns": round(transient_columns, 4),
        "edge_energy_ratio": round(edge_energy, 4),
        "center_energy_ratio": round(center_energy, 4),
        "visual_entropy": round(float(-np.sum((power/total) * np.log(power/total + 1e-10))), 4),
    }


def classify_waveform_image_cv(samples, predicted_class, metrics):
    """Vision-style reasoning over waveform/scalogram appearance."""
    sig = compute_scalogram_signature(samples)
    if not sig:
        return {
            "label": predicted_class,
            "confidence": 0.45,
            "explanation": "CV waveform mode unavailable; using waveform classifier output.",
            "signature": {},
        }

    label = predicted_class
    explanation = "Waveform image appears nominal with stable low-frequency energy."
    confidence = 0.6

    if sig["transient_columns"] > 0.18 and sig["high_band_ratio"] > 0.22:
        label = "transient"
        confidence = 0.84
        explanation = "Scalogram shows sharp high-frequency streaks consistent with a transient or switching spike."
    elif sig["high_band_ratio"] > 0.18 and metrics["thd_percent"] > 5:
        label = "harmonic_distortion"
        confidence = 0.8
        explanation = "Waveform image has dense high-frequency texture and flattened energy distribution, suggesting harmonics."
    elif metrics.get("sag_depth_percent") is not None and sig["center_energy_ratio"] > 0.32:
        label = "voltage_sag"
        confidence = 0.78
        explanation = "Image intensity compresses around the cycle center, matching a depressed voltage envelope."
    elif metrics.get("swell_magnitude_percent") is not None and sig["edge_energy_ratio"] > 0.45:
        label = "voltage_swell"
        confidence = 0.76
        explanation = "Outer-cycle energy is amplified in the waveform image, consistent with a swell event."

    return {
        "label": label,
        "confidence": round(confidence, 3),
        "explanation": explanation,
        "signature": sig,
    }


def compute_pq_metrics(samples, fs=5000, nominal_v=None):
    """Compute all PQ metrics from a waveform window."""
    s = np.array(samples)
    rms = float(np.sqrt(np.mean(s**2)))
    
    # THD via FFT
    n = len(s)
    freqs = sp_fft.rfftfreq(n, 1.0/fs)
    mags = np.abs(sp_fft.rfft(s)) / n
    fund_idx = np.argmin(np.abs(freqs - 50.0))
    fund_mag = mags[fund_idx] if fund_idx < len(mags) else 1e-10
    harm_sq = sum(mags[np.argmin(np.abs(freqs - 50*h))]**2 for h in range(2, 12) if np.argmin(np.abs(freqs - 50*h)) < len(mags))
    thd = float(np.sqrt(harm_sq) / (fund_mag + 1e-10) * 100)
    
    # Raw harmonic magnitudes h2..h10 (stored for NMF forensics)
    raw_harm_mags = []
    for h in range(2, 11):
        idx = int(np.argmin(np.abs(freqs - 50.0 * h)))
        raw_harm_mags.append(float(mags[idx]) if idx < len(mags) else 0.0)
    
    # Frequency via zero-crossing
    crossings = [i for i in range(1, len(s)) if s[i-1] <= 0 < s[i]]
    freq = fs / np.mean(np.diff(crossings)) if len(crossings) >= 2 else 50.0
    
    # Power factor estimate from THD
    pf = float(0.95 / np.sqrt(1 + (thd/100)**2))
    
    # XPQRS waveforms are amplitude-scaled to [-1, 1], so a healthy sine has RMS = 1/sqrt(2).
    nominal_rms = float(nominal_v) if nominal_v is not None else (1.0 / np.sqrt(2.0))
    ratio = rms / (nominal_rms + 1e-10) * 100
    sag = float(ratio) if ratio < 85 else None
    swell = float(ratio) if ratio > 115 else None
    
    return {
        "thd_percent": round(min(thd, 99), 2),
        "rms_voltage": round(rms * 230, 2),  # scale to 230V for display
        "power_factor": round(max(0.01, min(1.0, pf)), 3),
        "frequency_hz": round(float(freq), 2),
        "sag_depth_percent": round(sag, 1) if sag else None,
        "swell_magnitude_percent": round(swell, 1) if swell else None,
        "_raw_harm_mags": raw_harm_mags,  # internal — used by forensics engine
    }


def generate_scalogram_b64(samples):
    """Generate a base64-encoded scalogram image."""
    if not HAS_MPL or not HAS_PYWT or len(samples) < 10:
        return ""
    try:
        scales = np.arange(1, min(32, len(samples)//2))
        coeffs = pywt.cwt(np.array(samples), scales, 'morl')[0]
        fig, ax = plt.subplots(figsize=(3, 1.8), dpi=72)
        ax.imshow(np.abs(coeffs)**2, aspect='auto', cmap='hot', interpolation='bilinear')
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout(pad=0.1)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor='#111', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception:
        return ""


def get_waveform(real_waveforms, fault_type=None, noise_snr=None):
    """Get a waveform — from real data if available, synthetic if not."""
    if fault_type and fault_type in real_waveforms and real_waveforms[fault_type]:
        wf = real_waveforms[fault_type][np.random.randint(len(real_waveforms[fault_type]))]
    elif real_waveforms.get("normal"):
        wf = real_waveforms["normal"][np.random.randint(len(real_waveforms["normal"]))]
    else:
        t = np.arange(0, 0.02, 1/5000)
        wf = np.sin(2 * np.pi * 50 * t).tolist()
    
    samples = np.array(wf)
    if noise_snr is not None:
        sig_power = np.mean(samples**2)
        noise_power = sig_power / (10**(noise_snr/10))
        samples = samples + np.random.normal(0, np.sqrt(noise_power), len(samples))
    
    return samples.tolist()
