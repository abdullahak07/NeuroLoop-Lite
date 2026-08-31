"""
visualise.py -- figures for NeuroLoop-Lite.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, FancyBboxPatch
import numpy as np


def plot_pipeline(path):
    steps = ["Public EEG\n(EEGMMIDB)", "Band-pass\n7-30 Hz", "Fixed windows\n64 ch x 320",
             "1D CNN", "State +\nconfidence", "Simulated decision\nSTIMULATE / WAIT"]
    fig, ax = plt.subplots(figsize=(13, 2.6))
    ax.set_xlim(0, len(steps)); ax.set_ylim(0, 1); ax.axis("off")
    for i, s in enumerate(steps):
        last = i == len(steps) - 1
        ax.add_patch(FancyBboxPatch((i + 0.06, 0.28), 0.82, 0.44,
                     boxstyle="round,pad=0.02,rounding_size=0.06",
                     linewidth=1.4, edgecolor="#22506b",
                     facecolor="#dcecf5" if not last else "#f6d9c8"))
        ax.text(i + 0.47, 0.5, s, ha="center", va="center", fontsize=9.5)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrow(i + 0.9, 0.5, 0.16, 0, width=0.012,
                         head_width=0.09, head_length=0.05,
                         length_includes_head=True, color="#22506b"))
    ax.set_title("NeuroLoop-Lite pipeline", fontsize=12, pad=6)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def plot_confusion_matrix(cm, class_names, path):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), labels=class_names)
    ax.set_yticks(range(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    thr = cm.max() / 2 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black", fontsize=13)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_risk_coverage(sweep, baseline_acc, path):
    rows = sorted(sweep, key=lambda r: r["coverage"])
    cov = [r["coverage"] for r in rows]
    acc = [r["accuracy_accepted"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(cov, acc, "o-", color="#22506b", label="Accuracy on accepted")
    ax.axhline(baseline_acc, ls="--", color="#c2542a", lw=1.2,
               label=f"All windows ({baseline_acc:.2f})")
    lo = min(0.5, min(acc) - 0.03) if acc else 0.5
    hi = max(0.7, max(acc) + 0.06) if acc else 0.7
    ax.set_xlim(0.12, 1.03); ax.set_ylim(lo, hi)
    ax.set_xlabel("Coverage (fraction of windows acted on)")
    ax.set_ylabel("Accuracy on accepted")
    ax.set_title("Risk-coverage: does confidence rank correctness?")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _bin_stats(conf, correct, n_bins):
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accs = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        centers.append((lo + hi) / 2)
        accs.append(correct[m].mean() if m.any() else np.nan)
    return np.array(centers), np.array(accs)


def plot_reliability(conf_pre, correct_pre, ece_pre,
                     conf_post, correct_post, ece_post, path, n_bins=10):
    panels = [(conf_pre, correct_pre, ece_pre, "Before (raw softmax)", "#c2542a"),
              (conf_post, correct_post, ece_post, "After (temperature scaled)", "#3c7d4a")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, (c, corr, ece, title, colour) in zip(axes, panels):
        centers, accs = _bin_stats(c, corr, n_bins)
        ax.bar(centers, accs, width=(1.0 / n_bins) * 0.9,
               color=colour, alpha=0.75, edgecolor="#333", linewidth=0.6)
        ax.plot([0.5, 1.0], [0.5, 1.0], "k--", linewidth=1.1, label="Perfect calibration")
        ax.set_xlim(0.5, 1.0); ax.set_ylim(0.0, 1.0); ax.set_aspect("equal")
        ax.set_xlabel("Confidence"); ax.set_title(f"{title}\nECE = {ece:.3f}")
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("Accuracy")
    fig.suptitle("Reliability diagram", y=1.01, fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def plot_example_decision(x_window, ch_names, sfreq, pred_name, confidence,
                          decision, target_name, path, n_show=6):
    x = np.asarray(x_window)
    n_show = min(n_show, x.shape[0])
    t = np.arange(x.shape[1]) / sfreq
    fig, (ax, axt) = plt.subplots(1, 2, figsize=(11, 3.8),
                                  gridspec_kw={"width_ratios": [2.3, 1]})
    off = 0.0
    for c in range(n_show):
        trace = x[c] / (np.abs(x[c]).max() + 1e-9)
        ax.plot(t, trace + off, linewidth=0.8, color="#22506b")
        ax.text(-0.02, off, ch_names[c], ha="right", va="center", fontsize=8)
        off += 2.3
    ax.set_xlabel("Time (s)"); ax.set_yticks([])
    ax.set_title(f"Example EEG window ({n_show} of {x.shape[0]} channels)")

    axt.axis("off")
    fire = decision == "STIMULATE"
    axt.text(0.5, 0.90, "SIMULATED DECISION", ha="center", fontsize=10, color="#555")
    axt.text(0.5, 0.68, f"Predicted state:  {pred_name}", ha="center", fontsize=12)
    axt.text(0.5, 0.53, f"Confidence:  {confidence:.2f}", ha="center", fontsize=12)
    axt.text(0.5, 0.38, f"Target state:  {target_name}", ha="center", fontsize=10, color="#555")
    axt.add_patch(FancyBboxPatch((0.16, 0.06), 0.68, 0.20,
                  boxstyle="round,pad=0.02,rounding_size=0.06",
                  facecolor="#d6ead9" if fire else "#eee2c8",
                  edgecolor="#3c7d4a" if fire else "#a98a3c", linewidth=1.6))
    axt.text(0.5, 0.16, decision, ha="center", va="center", fontsize=16,
             color="#2f6b3c" if fire else "#8a6d2a", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
