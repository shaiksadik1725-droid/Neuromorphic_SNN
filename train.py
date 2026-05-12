
# ─── stdlib ──────────────────────────────────────────────────────────────────
import os, sys, zipfile, io, pickle, warnings
warnings.filterwarnings('ignore')

# ─── third-party ─────────────────────────────────────────────────────────────
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection    import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing      import StandardScaler, LabelEncoder
from sklearn.metrics            import (confusion_matrix, f1_score,
                                        precision_score, classification_report,
                                        accuracy_score)
from sklearn.ensemble           import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm                import SVC
from sklearn.neural_network     import MLPClassifier
from sklearn.pipeline           import Pipeline

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
# Point to either:
#   (a) the extracted CSV:  CSV_PATH = "dataset/Crop_recommendation.csv"
#   (b) the zip file:       ZIP_PATH = "1778500979911_dataset.zip"  (set CSV_PATH = None)
ZIP_PATH        = None           # set to zip filename if you haven't extracted yet
CSV_PATH        = "dataset/Crop_recommendation.csv"

FEATURES        = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
TARGET          = 'label'

# SNN hyper-params
T               = 30
BATCH_SIZE      = 64
EPOCHS          = 80
LR              = 1e-3
BETA            = 0.9
HIDDEN1         = 256
HIDDEN2         = 128

# General
TEST_SIZE       = 0.2
RANDOM_STATE    = 42
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 65)
print("  CROP RECOMMENDATION — MULTI-MODEL TRAINING")
print("=" * 65)
print(f"Device : {DEVICE}")

# ─── 1. LOAD DATA ─────────────────────────────────────────────────────────────
def load_csv(csv_path: str, zip_path: str | None) -> pd.DataFrame:
    """Load the main CSV from disk or from inside a zip archive."""
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV  : {csv_path}  {df.shape}")
        return df
    if zip_path and os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path) as zf:
            # find the Crop_recommendation.csv inside
            targets = [n for n in zf.namelist() if "Crop_recommendation" in n]
            if not targets:
                raise FileNotFoundError("Crop_recommendation.csv not found inside zip.")
            with zf.open(targets[0]) as f:
                df = pd.read_csv(io.TextIOWrapper(f))
        print(f"Loaded CSV from zip: {zip_path!r} → {targets[0]}  {df.shape}")
        return df
    raise FileNotFoundError(
        "Cannot find dataset. Set CSV_PATH to the CSV file or ZIP_PATH to the zip archive.")

df = load_csv(CSV_PATH, ZIP_PATH)

# ─── 2. FEATURE & LABEL PREP ──────────────────────────────────────────────────
X_raw  = df[FEATURES].values
y_raw  = df[TARGET].values

le     = LabelEncoder()
y      = le.fit_transform(y_raw)
num_classes = len(le.classes_)

scaler = StandardScaler()
X      = scaler.fit_transform(X_raw)

print(f"Classes ({num_classes}): {list(le.classes_)}")
print(f"Feature shape  : {X.shape}")

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
print(f"Train / Test   : {len(X_tr)} / {len(X_te)}\n")

# ─── 3. SKLEARN MODELS ────────────────────────────────────────────────────────
print("─" * 65)
print("Training sklearn baseline models …")
print("─" * 65)

sk_models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=1,
        n_jobs=-1, random_state=RANDOM_STATE),

    "SVM (RBF)": SVC(
        C=10, gamma='scale', kernel='rbf',
        probability=True, random_state=RANDOM_STATE),

    "MLP (sklearn)": MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu', solver='adam',
        max_iter=300, early_stopping=True,
        random_state=RANDOM_STATE),
}

sk_results = {}
for name, clf in sk_models.items():
    clf.fit(X_tr, y_tr)
    preds   = clf.predict(X_te)
    acc     = accuracy_score(y_te, preds)
    f1      = f1_score(y_te, preds, average='weighted')
    prec    = precision_score(y_te, preds, average='weighted')
    sk_results[name] = dict(acc=acc, f1=f1, prec=prec, preds=preds, model=clf)
    print(f"  {name:<20} Acc: {acc*100:.2f}%  F1(w): {f1:.4f}  Prec(w): {prec:.4f}")

# ─── 4. SPIKING NEURAL NETWORK ────────────────────────────────────────────────
print("\n" + "─" * 65)
print("Training Spiking Neural Network (SNN) …")
print("─" * 65)

# 4a. DataLoaders
X_tr_t = torch.FloatTensor(X_tr)
X_te_t = torch.FloatTensor(X_te)
y_tr_t = torch.LongTensor(y_tr)
y_te_t = torch.LongTensor(y_te)

train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                          batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
test_loader  = DataLoader(TensorDataset(X_te_t, y_te_t),
                          batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# 4b. Rate encoding
def rate_encode(x: torch.Tensor, T: int) -> torch.Tensor:
    """Convert a real-valued batch into T-step Bernoulli spike trains."""
    x_norm = torch.sigmoid(x)
    return torch.stack([torch.bernoulli(x_norm) for _ in range(T)], dim=0)

# 4c. Model definition
spike_grad = surrogate.fast_sigmoid(slope=25)

class CropSNN(nn.Module):
    def __init__(self, in_size, h1, h2, out_size, beta):
        super().__init__()
        self.bn0   = nn.BatchNorm1d(in_size)
        self.fc1   = nn.Linear(in_size, h1)
        self.lif1  = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)
        self.drop1 = nn.Dropout(0.30)

        self.fc2   = nn.Linear(h1, h2)
        self.lif2  = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)
        self.drop2 = nn.Dropout(0.20)

        self.fc3   = nn.Linear(h2, 64)
        self.lif3  = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)

        self.fc4   = nn.Linear(64, out_size)
        self.lif4  = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)

    def forward(self, x_spikes: torch.Tensor):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        spk4_acc = torch.zeros(
            x_spikes.size(1), self.fc4.out_features, device=x_spikes.device)

        for t in range(x_spikes.size(0)):
            xt        = self.bn0(x_spikes[t])
            spk1, mem1 = self.lif1(self.fc1(xt),        mem1); spk1 = self.drop1(spk1)
            spk2, mem2 = self.lif2(self.fc2(spk1),      mem2); spk2 = self.drop2(spk2)
            spk3, mem3 = self.lif3(self.fc3(spk2),      mem3)
            spk4, mem4 = self.lif4(self.fc4(spk3),      mem4)
            spk4_acc  += spk4

        return spk4_acc, mem4

model     = CropSNN(len(FEATURES), HIDDEN1, HIDDEN2, num_classes, BETA).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

print(f"SNN parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# 4d. Training loop
train_losses, test_losses = [], []
train_accs,   test_accs   = [], []
best_snn_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    # ── train ──
    model.train()
    ep_loss = ep_correct = ep_total = 0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        spikes    = rate_encode(X_b, T).to(DEVICE)
        optimizer.zero_grad()
        out, _    = model(spikes)
        loss      = criterion(out, y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        ep_loss    += loss.item()
        ep_correct += (out.argmax(1) == y_b).sum().item()
        ep_total   += y_b.size(0)
    scheduler.step()
    tr_loss = ep_loss / len(train_loader)
    tr_acc  = ep_correct / ep_total
    train_losses.append(tr_loss)
    train_accs.append(tr_acc)

    # ── eval ──
    model.eval()
    te_loss = te_correct = te_total = 0
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            spikes    = rate_encode(X_b, T).to(DEVICE)
            out, _    = model(spikes)
            te_loss    += criterion(out, y_b).item()
            te_correct += (out.argmax(1) == y_b).sum().item()
            te_total   += y_b.size(0)
    te_loss_avg = te_loss / len(test_loader)
    te_acc      = te_correct / te_total
    test_losses.append(te_loss_avg)
    test_accs.append(te_acc)

    if te_acc > best_snn_acc:
        best_snn_acc = te_acc
        torch.save(model.state_dict(), 'crop_snn_best.pth')

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch [{epoch:3d}/{EPOCHS}]  "
              f"Train  Loss: {tr_loss:.4f}  Acc: {tr_acc*100:.2f}%  |  "
              f"Test   Loss: {te_loss_avg:.4f}  Acc: {te_acc*100:.2f}%")

print(f"\nBest SNN Test Accuracy: {best_snn_acc*100:.2f}%")

# 4e. Final SNN evaluation (load best weights)
model.load_state_dict(torch.load('crop_snn_best.pth', map_location=DEVICE))
model.eval()
snn_preds, snn_labels = [], []
with torch.no_grad():
    for X_b, y_b in test_loader:
        X_b   = X_b.to(DEVICE)
        spikes = rate_encode(X_b, T).to(DEVICE)
        out, _ = model(spikes)
        snn_preds.extend(out.argmax(1).cpu().numpy())
        snn_labels.extend(y_b.numpy())

snn_preds  = np.array(snn_preds)
snn_labels = np.array(snn_labels)

snn_acc  = accuracy_score(snn_labels, snn_preds)
snn_f1   = f1_score(snn_labels, snn_preds, average='weighted')
snn_prec = precision_score(snn_labels, snn_preds, average='weighted')

sk_results["SNN"] = dict(acc=snn_acc, f1=snn_f1, prec=snn_prec,
                          preds=snn_preds, model=model)

# ─── 5. FULL EVALUATION REPORT ────────────────────────────────────────────────
print("\n" + "=" * 65)
print("FINAL EVALUATION — ALL MODELS")
print("=" * 65)
print(f"{'Model':<22} {'Accuracy':>10} {'F1 (w)':>10} {'Precision (w)':>14}")
print("-" * 60)
for name, res in sk_results.items():
    print(f"{name:<22} {res['acc']*100:>9.2f}%  {res['f1']:>9.4f}  {res['prec']:>13.4f}")

print("\nClassification Report (SNN):")
print(classification_report(snn_labels, snn_preds,
                             target_names=le.classes_, digits=4))

# ─── 6. PLOTS ─────────────────────────────────────────────────────────────────
f1_per   = f1_score(snn_labels, snn_preds, average=None)
prec_per = precision_score(snn_labels, snn_preds, average=None)

# 6a. SNN Confusion Matrix
cm = confusion_matrix(snn_labels, snn_preds)
plt.figure(figsize=(18, 14))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_,
            linewidths=0.4, linecolor='gray', annot_kws={'size': 8})
plt.title(f'Confusion Matrix — SNN Crop Recommendation\nAccuracy: {snn_acc*100:.2f}%',
          fontsize=14, fontweight='bold')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual',    fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0,  fontsize=9)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: confusion_matrix.png")

# 6b. SNN Training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(train_losses, label='Train Loss', color='royalblue', lw=2)
axes[0].plot(test_losses,  label='Test Loss',  color='tomato',    lw=2)
axes[0].set_title('SNN — Loss over Epochs', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Cross-Entropy Loss')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot([a*100 for a in train_accs], label='Train', color='seagreen',   lw=2)
axes[1].plot([a*100 for a in test_accs],  label='Test',  color='darkorange', lw=2)
axes[1].axhline(y=best_snn_acc*100, color='red', ls='--', lw=1.2,
                label=f'Best: {best_snn_acc*100:.2f}%')
axes[1].set_title('SNN — Accuracy over Epochs', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: training_curves.png")

# 6c. Per-class F1 & Precision (SNN)
x = np.arange(num_classes); w = 0.38
fig, ax = plt.subplots(figsize=(18, 6))
ax.bar(x - w/2, f1_per,   w, label='F1',        color='steelblue', edgecolor='k', lw=0.5)
ax.bar(x + w/2, prec_per, w, label='Precision', color='coral',     edgecolor='k', lw=0.5)
ax.set_title('F1 & Precision per Crop Class (SNN)', fontsize=13, fontweight='bold')
ax.set_xlabel('Crop'); ax.set_ylabel('Score')
ax.set_xticks(x); ax.set_xticklabels(le.classes_, rotation=45, ha='right', fontsize=9)
ax.set_ylim(0, 1.15)
ax.axhline(1.0, color='gray', ls='--', lw=0.7)
ax.legend(fontsize=11); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('f1_precision_per_class.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: f1_precision_per_class.png")

# 6d. Model comparison bar chart
model_names = list(sk_results.keys())
accs   = [sk_results[n]['acc']  * 100 for n in model_names]
f1s    = [sk_results[n]['f1']   * 100 for n in model_names]
precs  = [sk_results[n]['prec'] * 100 for n in model_names]

x  = np.arange(len(model_names)); w = 0.25
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - w,   accs,  w, label='Accuracy',  color='steelblue', edgecolor='k', lw=0.5)
ax.bar(x,       f1s,   w, label='F1 (w)',    color='seagreen',  edgecolor='k', lw=0.5)
ax.bar(x + w,   precs, w, label='Prec (w)',  color='coral',     edgecolor='k', lw=0.5)

for i, (a, f, p) in enumerate(zip(accs, f1s, precs)):
    ax.text(i - w,   a + 0.3, f'{a:.1f}',  ha='center', va='bottom', fontsize=8)
    ax.text(i,       f + 0.3, f'{f:.1f}',  ha='center', va='bottom', fontsize=8)
    ax.text(i + w,   p + 0.3, f'{p:.1f}',  ha='center', va='bottom', fontsize=8)

ax.set_title('Model Comparison — Crop Recommendation', fontsize=13, fontweight='bold')
ax.set_ylabel('Score (%)')
ax.set_xticks(x); ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylim(0, 115)
ax.axhline(100, color='gray', ls='--', lw=0.7)
ax.legend(fontsize=11); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: model_comparison.png")

# ─── 7. SAVE ARTEFACTS ────────────────────────────────────────────────────────
with open('scaler.pkl',        'wb') as f: pickle.dump(scaler, f)
with open('label_encoder.pkl', 'wb') as f: pickle.dump(le,     f)
with open('rf_model.pkl',      'wb') as f: pickle.dump(sk_results['Random Forest']['model'], f)

model_config = {
    'input_size' : len(FEATURES),
    'hidden1'    : HIDDEN1,
    'hidden2'    : HIDDEN2,
    'output_size': num_classes,
    'beta'       : BETA,
    'T'          : T,
    'features'   : FEATURES,
    'classes'    : list(le.classes_),
}
with open('model_config.pkl', 'wb') as f: pickle.dump(model_config, f)

# ─── 8. SUMMARY ───────────────────────────────────────────────────────────────
best_name = max(sk_results, key=lambda n: sk_results[n]['acc'])
print("\n" + "=" * 65)
print("TRAINING COMPLETE")
print("=" * 65)
print(f"Best model     : {best_name}  ({sk_results[best_name]['acc']*100:.2f}%)")
print("Saved files    : crop_snn_best.pth | scaler.pkl | label_encoder.pkl")
print("                 model_config.pkl  | rf_model.pkl")
print("Plots          : confusion_matrix.png | training_curves.png")
print("                 f1_precision_per_class.png | model_comparison.png")
print("=" * 65)