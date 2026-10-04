"""
LSTM-based anomaly detection & failure prediction on simulated industrial sensor data.
Sensors: temperature, vibration, pressure, motor current.
Label 1 = machine is in the pre-failure (degradation) phase, 0 = healthy.
"""
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
FEATURES = ["temperature", "vibration", "pressure", "motor_current"]

# ---------------------------------------------------------------- 1. SIMULATION
def simulate_machine(n_steps=1000, fails=False):
    """Healthy baseline = daily cycle + noise. Failing machines drift in the last ~25%."""
    t = np.arange(n_steps)
    load = 0.5 + 0.2 * np.sin(2 * np.pi * t / 200)          # operating load cycle
    temp = 60 + 10 * load + np.random.normal(0, 0.8, n_steps)
    vib  = 2.0 + 0.5 * load + np.random.normal(0, 0.15, n_steps)
    pres = 5.0 + 0.8 * load + np.random.normal(0, 0.10, n_steps)
    curr = 10 + 4 * load + np.random.normal(0, 0.3, n_steps)
    labels = np.zeros(n_steps, dtype=int)

    if fails:
        start = int(n_steps * np.random.uniform(0.6, 0.8))   # degradation onset
        ramp = np.linspace(0, 1, n_steps - start) ** 1.5      # gradual worsening
        temp[start:] += 18 * ramp
        vib[start:]  += 2.5 * ramp + np.random.normal(0, 0.3 * ramp)   # growing jitter
        pres[start:] -= 1.2 * ramp
        curr[start:] += 5 * ramp
        labels[start:] = 1
        # sudden spikes (bearing impacts etc.)
        for i in np.random.choice(np.arange(start, n_steps), 8, replace=False):
            vib[i] += np.random.uniform(2, 4)
    return np.stack([temp, vib, pres, curr], 1), labels

def build_dataset(n_machines, fail_ratio=0.6):
    machines = []
    for _ in range(n_machines):
        X, y = simulate_machine(fails=np.random.rand() < fail_ratio)
        machines.append((X, y))
    return machines

def make_windows(machines, win=50, stride=5, mu=None, sd=None):
    allX = np.concatenate([m[0] for m in machines])
    if mu is None:
        mu, sd = allX.mean(0), allX.std(0)
    Xs, ys = [], []
    for X, y in machines:
        Xn = (X - mu) / sd
        for s in range(0, len(Xn) - win, stride):
            Xs.append(Xn[s:s + win])
            ys.append(y[s + win - 1])        # label of the last step in the window
    return np.array(Xs, np.float32), np.array(ys, np.float32), mu, sd

# ---------------------------------------------------------------- 2. MODEL
class LSTMClassifier(nn.Module):
    def __init__(self, n_feat=4, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(32, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)   # logit from last time step

# ---------------------------------------------------------------- 3. TRAIN / EVAL
def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # split BY MACHINE to avoid leakage between train/val/test windows
    train_m, val_m, test_m = build_dataset(40), build_dataset(10), build_dataset(10)
    Xtr, ytr, mu, sd = make_windows(train_m)
    Xva, yva, _, _ = make_windows(val_m, mu=mu, sd=sd)
    Xte, yte, _, _ = make_windows(test_m, mu=mu, sd=sd)
    print(f"Windows  train={Xtr.shape} val={Xva.shape} test={Xte.shape}")
    print(f"Anomaly share (train) = {ytr.mean():.2%}")

    tl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)), batch_size=64, shuffle=True)
    model = LSTMClassifier().to(dev)
    pos_w = torch.tensor((1 - ytr.mean()) / ytr.mean(), device=dev)   # class imbalance
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def predict(X):
        model.eval()
        with torch.no_grad():
            return torch.sigmoid(model(torch.tensor(X).to(dev))).cpu().numpy()

    hist = {"loss": [], "val_loss": [], "val_acc": []}
    best, best_state = 1e9, None
    for ep in range(1, 21):
        model.train(); tot = 0
        for xb, yb in tl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # tame exploding gradients
            opt.step(); tot += loss.item() * len(xb)
        p = predict(Xva)
        vl = nn.functional.binary_cross_entropy(torch.tensor(p), torch.tensor(yva)).item()
        va = ((p > 0.5) == yva).mean()
        hist["loss"].append(tot / len(Xtr)); hist["val_loss"].append(vl); hist["val_acc"].append(va)
        if vl < best:
            best = vl; best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep:02d}  loss={hist['loss'][-1]:.4f}  val_loss={vl:.4f}  val_acc={va:.3f}")

    model.load_state_dict(best_state)
    p = predict(Xte); pred = (p > 0.5).astype(int)
    print("\n=== TEST RESULTS ===")
    print(classification_report(yte, pred, target_names=["Normal", "Anomaly"], digits=3))
    cm = confusion_matrix(yte, pred)
    print("Confusion matrix:\n", cm, "\nROC-AUC:", round(roc_auc_score(yte, p), 4))

    # ------------------------------------------------------------ 4. PLOTS
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0, 0].plot(hist["loss"], label="train"); ax[0, 0].plot(hist["val_loss"], label="val")
    ax[0, 0].set_title("Loss curve"); ax[0, 0].legend()
    ax[0, 1].imshow(cm, cmap="Blues"); ax[0, 1].set_title("Confusion matrix")
    ax[0, 1].set_xticks([0, 1], ["Normal", "Anomaly"]); ax[0, 1].set_yticks([0, 1], ["Normal", "Anomaly"])
    for i in range(2):
        for j in range(2):
            ax[0, 1].text(j, i, cm[i, j], ha="center", va="center", color="red", fontsize=14)
    # one failing machine: sensors + predicted failure probability
    X, y = simulate_machine(fails=True)
    Xw, yw, _, _ = make_windows([(X, y)], mu=mu, sd=sd, stride=1)
    prob = predict(Xw)
    for k, f in enumerate(FEATURES):
        ax[1, 0].plot((X[:, k] - mu[k]) / sd[k], label=f, lw=0.8)
    ax[1, 0].axvline(np.argmax(y), color="k", ls="--", label="degradation start")
    ax[1, 0].set_title("Simulated sensors (normalised) – failing machine"); ax[1, 0].legend(fontsize=7)
    ax[1, 1].plot(np.arange(len(prob)) + 49, prob, color="crimson", label="P(anomaly)")
    ax[1, 1].axhline(0.5, color="gray", ls=":"); ax[1, 1].axvline(np.argmax(y), color="k", ls="--")
    ax[1, 1].set_title("LSTM failure probability over time"); ax[1, 1].legend()
    plt.tight_layout(); plt.savefig("results.png", dpi=130)
    torch.save(model.state_dict(), "lstm_failure_model.pt")
    print("Saved results.png and lstm_failure_model.pt")

if __name__ == "__main__":
    main()