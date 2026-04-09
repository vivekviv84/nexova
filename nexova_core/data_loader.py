import csv
import numpy as np
from nexova_core.config import XPQRS_DIR, XPQRS_CLASSES, XPQRS_SIMPLE_MAP, DATA_DIR

def load_xpqrs_dataset(max_per_class=200):
    """Load XPQRS waveform CSVs. Each CSV = 1000 waveforms × 100 samples."""
    X, y, labels = [], [], []
    if not XPQRS_DIR.exists():
        print(f"  [!] XPQRS directory not found at {XPQRS_DIR}")
        return np.array([]), np.array([]), []
    
    for cls_name in XPQRS_CLASSES:
        fpath = XPQRS_DIR / f"{cls_name}.csv"
        if not fpath.exists():
            continue
        with open(fpath) as f:
            reader = csv.reader(f)
            count = 0
            for row in reader:
                if count >= max_per_class:
                    break
                try:
                    samples = [float(v) for v in row if v.strip()]
                    if len(samples) >= 50:
                        X.append(samples[:100])
                        y.append(XPQRS_SIMPLE_MAP.get(cls_name, cls_name.lower()))
                        count += 1
                except (ValueError, IndexError):
                    continue
        if count > 0:
            if cls_name not in [l for l in labels]:
                labels.append(cls_name)
    
    if not X:
        return np.array([]), np.array([]), []
    
    # Pad/truncate to uniform length
    max_len = max(len(x) for x in X)
    X_padded = [x + [0.0] * (max_len - len(x)) if len(x) < max_len else x[:max_len] for x in X]
    
    unique_labels = sorted(set(y))
    print(f"  Loaded XPQRS: {len(X_padded)} waveforms, {len(unique_labels)} classes")
    return np.array(X_padded), np.array(y), unique_labels


def load_fault_features_dataset():
    """Load the tabular power_quality_fault_dataset.csv."""
    fpath = DATA_DIR / "power_quality_fault_dataset.csv"
    if not fpath.exists():
        return None, None, None
    
    X, y = [], []
    feature_names = None
    with open(fpath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                features = [
                    float(row['RMS_Voltage']), float(row['Peak_Voltage']),
                    float(row['THD']), float(row['Duration_ms']),
                    float(row['DWT_Energy_Level1']), float(row['DWT_Energy_Level2']),
                    float(row['DWT_Entropy']), float(row['Signal_Noise_Ratio_dB']),
                ]
                X.append(features)
                y.append(row['Fault_Type'])
                if feature_names is None:
                    feature_names = ['RMS_Voltage', 'Peak_Voltage', 'THD', 'Duration_ms',
                                    'DWT_Energy_Level1', 'DWT_Energy_Level2', 'DWT_Entropy', 'SNR_dB']
            except (ValueError, KeyError):
                continue
    
    print(f"  Loaded fault features: {len(X)} samples, {len(set(y))} classes")
    return np.array(X), np.array(y), feature_names


def load_replay_waveforms():
    """Load some real waveforms for replay."""
    real_waveforms = {}
    for cls_name in ["Pure_Sinusoidal", "Sag", "Harmonics", "Transient", "Sag_with_Harmonics", "Flicker", "Notch", "Interruption"]:
        fpath = XPQRS_DIR / f"{cls_name}.csv"
        if fpath.exists():
            with open(fpath) as f:
                reader = csv.reader(f)
                wfs = []
                for row in reader:
                    try:
                        wfs.append([float(v) for v in row if v.strip()][:100])
                    except ValueError:
                        continue
                    if len(wfs) >= 50:
                        break
                if wfs:
                    real_waveforms[XPQRS_SIMPLE_MAP.get(cls_name, cls_name.lower())] = wfs
    return real_waveforms
