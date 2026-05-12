"""
Model utilities for the SNN Crop Recommendation dashboard.
Loads the trained SNN, scaler, encoder, config; provides
single + batch prediction with rate-encoded spike inference.
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

# ---------------------------------------------------------------------------
# Paths — adjust if your trained artifacts live elsewhere
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "crop_snn_best.pth")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
LE_PATH     = os.path.join(BASE_DIR, "label_encoder.pkl")
CONFIG_PATH = os.path.join(BASE_DIR, "model_config.pkl")

spike_grad = surrogate.fast_sigmoid(slope=25)


# ---------------------------------------------------------------------------
# Architecture — must match train.py exactly
# ---------------------------------------------------------------------------
class CropSNN(nn.Module):
    def __init__(self, in_size, h1, h2, out_size, beta):
        super().__init__()
        self.bn0   = nn.BatchNorm1d(in_size)
        self.fc1   = nn.Linear(in_size, h1)
        self.lif1  = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)
        self.drop1 = nn.Dropout(0.3)
        self.fc2   = nn.Linear(h1, h2)
        self.lif2  = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)
        self.drop2 = nn.Dropout(0.2)
        self.fc3   = nn.Linear(h2, 64)
        self.lif3  = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)
        self.fc4   = nn.Linear(64, out_size)
        self.lif4  = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)

    def forward(self, x_spikes):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        spk4_acc = torch.zeros(x_spikes.size(1), self.fc4.out_features,
                               device=x_spikes.device)
        for t in range(x_spikes.size(0)):
            xt        = self.bn0(x_spikes[t])
            spk1, mem1 = self.lif1(self.fc1(xt),   mem1)
            spk1       = self.drop1(spk1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            spk2       = self.drop2(spk2)
            spk3, mem3 = self.lif3(self.fc3(spk2), mem3)
            spk4, mem4 = self.lif4(self.fc4(spk3), mem4)
            spk4_acc  += spk4
        return spk4_acc, mem4


def rate_encode(x, T):
    """Rate-encode a tensor of shape (B, F) into spikes of shape (T, B, F)."""
    x_norm = torch.sigmoid(x)
    return torch.stack([torch.bernoulli(x_norm) for _ in range(T)], dim=0)


# ---------------------------------------------------------------------------
# Singleton-style loader — caches the model in memory
# ---------------------------------------------------------------------------
_state = {"model": None, "scaler": None, "le": None, "cfg": None, "device": None}


def is_ready() -> bool:
    return all(os.path.exists(p) for p in (MODEL_PATH, SCALER_PATH, LE_PATH, CONFIG_PATH))


def load_artifacts():
    """Load model + preprocessors once; return cached instance afterwards."""
    if _state["model"] is not None:
        return _state

    if not is_ready():
        missing = [p for p in (MODEL_PATH, SCALER_PATH, LE_PATH, CONFIG_PATH)
                   if not os.path.exists(p)]
        raise FileNotFoundError(f"Missing model artifacts: {missing}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
    with open(LE_PATH,     'rb') as f: le     = pickle.load(f)
    with open(CONFIG_PATH, 'rb') as f: cfg    = pickle.load(f)

    model = CropSNN(cfg['input_size'], cfg['hidden1'], cfg['hidden2'],
                    cfg['output_size'], cfg['beta']).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    _state.update(model=model, scaler=scaler, le=le, cfg=cfg, device=device)
    return _state


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def predict_single(N, P, K, temperature, humidity, ph, rainfall, n_runs: int = 8):
    """Predict a single sample. Returns dict of label, conf, probabilities, top5."""
    s   = load_artifacts()
    cfg, model, scaler, le, device = s['cfg'], s['model'], s['scaler'], s['le'], s['device']

    sample = np.array([[N, P, K, temperature, humidity, ph, rainfall]],
                      dtype=np.float32)
    sc     = scaler.transform(sample)
    x_t    = torch.FloatTensor(sc).to(device)

    prob_acc = np.zeros(cfg['output_size'])
    with torch.no_grad():
        for _ in range(n_runs):
            spikes = rate_encode(x_t, cfg['T']).to(device)
            out, _ = model(spikes)
            prob_acc += torch.softmax(out, dim=1).cpu().numpy()[0]

    probs    = prob_acc / n_runs
    pred_idx = int(probs.argmax())
    label    = str(le.classes_[pred_idx])

    # Top-5 list
    top_idx  = np.argsort(probs)[::-1][:5]
    top5     = [{"label": str(le.classes_[i]),
                 "prob":  float(probs[i] * 100)} for i in top_idx]

    # Full probability list (sorted high → low) for chart
    all_sorted = np.argsort(probs)[::-1]
    all_probs  = [{"label": str(le.classes_[i]),
                   "prob":  float(probs[i] * 100)} for i in all_sorted]

    return {
        "label":      label,
        "confidence": float(probs[pred_idx] * 100),
        "top5":       top5,
        "all_probs":  all_probs,
        "scaled":     sc.flatten().tolist(),
        "raw":        sample.flatten().tolist(),
        "features":   list(cfg['features']),
    }


def predict_dataframe(df: pd.DataFrame, n_runs: int = 5):
    """
    Predict for every row of a DataFrame. Auto-detects 'label' column.
    Returns: dict with results list, true/pred arrays, metrics if labels exist.
    """
    s   = load_artifacts()
    cfg, model, scaler, le, device = s['cfg'], s['model'], s['scaler'], s['le'], s['device']
    feats = cfg['features']

    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Required: {feats}")

    has_labels = 'label' in df.columns
    true_labels = df['label'].astype(str).values if has_labels else None

    X = df[feats].values.astype(np.float32)
    X_sc = scaler.transform(X)
    x_t  = torch.FloatTensor(X_sc).to(device)

    prob_acc = np.zeros((len(df), cfg['output_size']))
    with torch.no_grad():
        for _ in range(n_runs):
            # Loop in chunks to avoid OOM on very large CSVs
            B = 256
            for i in range(0, len(x_t), B):
                chunk = x_t[i:i + B]
                spikes = rate_encode(chunk, cfg['T']).to(device)
                out, _ = model(spikes)
                probs = torch.softmax(out, dim=1).cpu().numpy()
                prob_acc[i:i + B] += probs

    probs_avg   = prob_acc / n_runs
    pred_idx    = probs_avg.argmax(axis=1)
    pred_labels = le.inverse_transform(pred_idx)
    confidences = probs_avg.max(axis=1) * 100

    rows = []
    for i in range(len(df)):
        row = {
            "row":         i + 1,
            "predicted":   str(pred_labels[i]),
            "confidence":  float(confidences[i]),
        }
        for f in feats:
            row[f] = float(df[feats].iloc[i][f])
        if has_labels:
            row["true"]    = str(true_labels[i])
            row["correct"] = bool(pred_labels[i] == true_labels[i])
        rows.append(row)

    out = {
        "rows":       rows,
        "n":          len(df),
        "features":   feats,
        "classes":    [str(c) for c in le.classes_],
        "has_labels": has_labels,
    }

    if has_labels:
        from sklearn.metrics import (confusion_matrix, f1_score,
                                     precision_score, recall_score)
        y_true = le.transform(true_labels)
        y_pred = pred_idx
        cm     = confusion_matrix(y_true, y_pred,
                                  labels=list(range(cfg['output_size'])))
        out.update({
            "accuracy":    float((y_true == y_pred).mean() * 100),
            "f1":          float(f1_score(y_true, y_pred, average='weighted')),
            "precision":   float(precision_score(y_true, y_pred, average='weighted',
                                                 zero_division=0)),
            "recall":      float(recall_score(y_true, y_pred, average='weighted',
                                              zero_division=0)),
            "confusion":   cm.tolist(),
            "f1_per":      f1_score(y_true, y_pred, average=None,
                                    zero_division=0,
                                    labels=list(range(cfg['output_size']))).tolist(),
            "prec_per":    precision_score(y_true, y_pred, average=None,
                                           zero_division=0,
                                           labels=list(range(cfg['output_size']))).tolist(),
        })
    return out


def get_class_list():
    s = load_artifacts()
    return [str(c) for c in s['le'].classes_]


def get_feature_list():
    return load_artifacts()['cfg']['features']


def get_scaler_stats():
    """Return mean and std of the fitted scaler — useful for a baseline UI."""
    s = load_artifacts()
    sc, feats = s['scaler'], s['cfg']['features']
    return {f: {"mean": float(m), "std": float(np.sqrt(v))}
            for f, m, v in zip(feats, sc.mean_, sc.var_)}