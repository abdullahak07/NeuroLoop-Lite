"""
evaluate.py -- training, prediction, calibration, selective prediction and the
simulated controller used by NeuroLoop-Lite.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix


def standardize_fit(X):
    mean = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2), keepdims=True) + 1e-7
    return mean.astype(np.float32), std.astype(np.float32)


def standardize_apply(X, mean, std):
    return ((X - mean) / std).astype(np.float32)


def train_model(model, X_tr, y_tr, X_va, y_va, *, epochs=40, lr=1e-3,
                batch_size=64, device="cpu", seed=0, verbose=True):
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    Xt = torch.as_tensor(X_tr, device=device)
    yt = torch.as_tensor(y_tr, device=device)
    has_val = len(X_va) > 0
    if has_val:
        Xv = torch.as_tensor(X_va, device=device)
        yv = torch.as_tensor(y_va, device=device)

    n = len(Xt)
    best_state, best_val = None, -1.0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()

        if has_val:
            model.eval()
            with torch.no_grad():
                acc = (model(Xv).argmax(1) == yv).float().mean().item()
            if acc >= best_val:
                best_val = acc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if verbose and (ep % 5 == 0 or ep == epochs - 1):
                print(f"  epoch {ep:2d}  train_loss {loss.item():.3f}  val_acc {acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_logits(model, X, device="cpu", batch_size=256):
    model.eval().to(device)
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[i:i + batch_size], device=device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 2), np.float32)


def _softmax_np(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def fit_temperature(logits_val, y_val, device="cpu", lr=0.05, max_iter=200):
    if len(logits_val) == 0:
        return 1.0
    logits = torch.as_tensor(np.asarray(logits_val), dtype=torch.float32, device=device)
    y = torch.as_tensor(np.asarray(y_val), dtype=torch.long, device=device)
    log_t = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)
    nll = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return max(float(log_t.exp().item()), 1e-3)


def apply_temperature(logits, T):
    return _softmax_np(np.asarray(logits) / max(float(T), 1e-6))


def expected_calibration_error(confidences, correct, n_bins=10):
    confidences = np.asarray(confidences, float)
    correct = np.asarray(correct, float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (confidences > lo) & (confidences <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - confidences[m].mean())
    return float(ece)


def confidence_at_coverage(confidences, coverage):
    conf = np.sort(np.asarray(confidences))[::-1]
    if len(conf) == 0:
        return 1.0
    k = max(1, int(round(float(coverage) * len(conf))))
    return float(conf[k - 1])


def coverage_sweep(confidences, preds, y_true, coverages=(1.0, 0.8, 0.6, 0.4, 0.2)):
    conf = np.asarray(confidences)
    preds = np.asarray(preds)
    y = np.asarray(y_true)
    order = np.argsort(-conf)
    n = len(conf)
    rows = []
    for c in coverages:
        k = max(1, int(round(float(c) * n)))
        idx = order[:k]
        rows.append({
            "coverage": round(float(c), 2),
            "n_accepted": int(k),
            "accuracy_accepted": round(float((preds[idx] == y[idx]).mean()), 4),
            "min_confidence": round(float(conf[idx].min()), 4),
        })
    return rows


def simulate_controller(preds, confidences, target_state, threshold):
    preds = np.asarray(preds)
    confidences = np.asarray(confidences)
    fire = (preds == target_state) & (confidences >= threshold)
    return np.where(fire, "STIMULATE", "WAIT")


def core_metrics(y_true, preds):
    return {
        "accuracy": round(float(accuracy_score(y_true, preds)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, preds)), 4),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
    }
