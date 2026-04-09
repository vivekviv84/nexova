import numpy as np
from nexova_core.features import extract_features
from nexova_core.data_loader import load_xpqrs_dataset, load_fault_features_dataset

# Try optional imports
try:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

class GridSenseClassifier:
    def __init__(self):
        self.waveform_model = None
        self.feature_model = None
        self.waveform_classes = []
        self.feature_classes = []
        self.feature_names = []
        self.wf_feature_names = []
        self.is_trained = False
        self.training_accuracy = 0
    
    def train_on_real_data(self, max_per_class=150):
        """Train on both Hitachi-provided datasets."""
        print("\n  ── Training GridSense Classifier on Real Data ──")
        
        # Dataset 1: XPQRS waveforms
        X_wf, y_wf, wf_labels = load_xpqrs_dataset(max_per_class=max_per_class)
        if len(X_wf) > 0 and HAS_SKLEARN:
            # Extract features from each waveform
            print("  Extracting CWT + time-domain features...")
            X_features = []
            valid_indices = []
            for i, wf in enumerate(X_wf):
                feats = extract_features(wf)
                if feats:
                    X_features.append(list(feats.values()))
                    valid_indices.append(i)
                    if not self.wf_feature_names:
                        self.wf_feature_names = list(feats.keys())
            
            if X_features:
                X_arr = np.array(X_features)
                y_arr = y_wf[valid_indices]
                self.waveform_classes = sorted(set(y_arr))
                
                # Encode labels
                label_map = {l: i for i, l in enumerate(self.waveform_classes)}
                y_encoded = np.array([label_map[l] for l in y_arr])
                
                self.waveform_model = GradientBoostingClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
                )
                self.waveform_model.fit(X_arr, y_encoded)
                
                train_acc = np.mean(self.waveform_model.predict(X_arr) == y_encoded) * 100
                print(f"  Waveform classifier: {train_acc:.1f}% accuracy on {len(X_arr)} samples, {len(self.waveform_classes)} classes")
                self.training_accuracy = train_acc
        
        # Dataset 2: Tabular features
        X_feat, y_feat, feat_names = load_fault_features_dataset()
        if X_feat is not None and HAS_SKLEARN:
            self.feature_classes = sorted(set(y_feat))
            label_map = {l: i for i, l in enumerate(self.feature_classes)}
            y_encoded = np.array([label_map[l] for l in y_feat])
            
            self.feature_model = GradientBoostingClassifier(
                n_estimators=80, max_depth=3, learning_rate=0.1, random_state=42
            )
            self.feature_model.fit(X_feat, y_encoded)
            
            acc = np.mean(self.feature_model.predict(X_feat) == y_encoded) * 100
            print(f"  Feature classifier: {acc:.1f}% accuracy on {len(X_feat)} samples")
            self.feature_names = feat_names or []
        
        self.is_trained = self.waveform_model is not None
        if self.is_trained:
            print(f"  ✓ Classifier trained on REAL Hitachi-provided data")
        else:
            print(f"  [!] No data found — using threshold-only mode")
    
    def predict(self, samples):
        """Classify a waveform."""
        if not self.is_trained:
            return "normal", 0.5, {}, "threshold"
        
        feats = extract_features(samples)
        if not feats:
            return "normal", 0.5, {}, "threshold"
        
        X = np.array([list(feats.values())])
        proba = self.waveform_model.predict_proba(X)[0]
        idx = np.argmax(proba)
        conf = float(proba[idx])
        cls = self.waveform_classes[idx]
        
        # Feature importances as pseudo-SHAP
        imps = self.waveform_model.feature_importances_
        top_idx = np.argsort(imps)[-5:][::-1]
        shap = {}
        for i in top_idx:
            if i < len(self.wf_feature_names):
                shap[self.wf_feature_names[i]] = round(float(imps[i]), 4)
        
        return cls, conf, shap, "cwt_scalogram"
