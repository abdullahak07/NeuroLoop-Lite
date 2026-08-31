"""
run_demo.py -- one command runs the whole thing.

    python run_demo.py --prepare      # download + cache the EEG windows
    python run_demo.py                # train, evaluate, write results + figures
    python run_demo.py --quick        # tiny subset (works for --prepare and run)

The closed-loop decision produced here is a software SIMULATION only. It does
not drive real brain stimulation and has no clinical validity.
"""

import argparse
import csv
import json
import os
import re
import numpy as np

from src import data as D
from src import evaluate as E
from src import visualise as V
from src.model import build_model

COVERAGES = [1.0, 0.8, 0.6, 0.4, 0.2]
DATASET_NAME = "PhysioNet EEG Motor Movement/Imagery (EEGMMIDB)"
TASK_NAME = "imagined left vs right fist (binary motor imagery)"


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def update_readme_metrics(readme_path, fields):
    """Replace the block between the METRICS markers, if present."""
    if not os.path.exists(readme_path):
        return
    with open(readme_path, encoding="utf-8") as f:
        text = f.read()
    block = "\n".join(f"- **{k}:** {v}" for k, v in fields.items())
    new = re.sub(r"(<!-- METRICS_START -->).*?(<!-- METRICS_END -->)",
                 rf"\1\n{block}\n\2", text, flags=re.S)
    if new != text:
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(new)


def run(args):
    import torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    ds = D.load_windows(args.data_dir)
    X, y, sids = ds["X"], ds["y"], ds["subject_ids"]
    ch_names, sfreq, class_names = ds["ch_names"], ds["sfreq"], ds["class_names"]
    print(f"Loaded {len(X)} windows | {X.shape[1]} channels x {X.shape[2]} samples "
          f"| {len(np.unique(sids))} subjects")

    tr, va, te = D.subject_wise_split(sids, seed=args.seed)
    print(f"Split (subject-wise): train={len(tr)}  val={len(va)}  test={len(te)}")

    mean, std = E.standardize_fit(X[tr])
    Xtr, Xva, Xte = (E.standardize_apply(X[i], mean, std) for i in (tr, va, te))

    device = _device()
    model = build_model(X.shape[1], n_classes=len(class_names))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: TinyEEGCNN ({n_params:,} params) on {device}")
    model = E.train_model(model, Xtr, y[tr], Xva, y[va],
                          epochs=args.epochs, device=device, seed=args.seed)

    logits_va = E.predict_logits(model, Xva, device=device)
    logits_te = E.predict_logits(model, Xte, device=device)
    y_te = y[te]

    T = E.fit_temperature(logits_va, y[va], device=device) if len(Xva) else 1.0
    proba_raw = E.apply_temperature(logits_te, 1.0)
    proba_cal = E.apply_temperature(logits_te, T)

    preds = proba_cal.argmax(1)
    conf_raw = proba_raw.max(1)
    conf = proba_cal.max(1)
    correct = (preds == y_te).astype(int)

    metrics = E.core_metrics(y_te, preds)
    ece_raw = E.expected_calibration_error(conf_raw, correct)
    ece_cal = E.expected_calibration_error(conf, correct)
    sweep = E.coverage_sweep(conf, preds, y_te, COVERAGES)

    target_state = class_names.index(args.target)

    # Choose the gate without peeking at test labels/confidence distribution.
    # With --coverage, the threshold is selected on validation confidence and
    # then frozen before it is applied to the held-out test subjects.
    if args.threshold is not None:
        op_thr = float(args.threshold)
        threshold_source = "user-specified"
    else:
        proba_va_cal = E.apply_temperature(logits_va, T)
        conf_va = proba_va_cal.max(1)
        op_thr = E.confidence_at_coverage(conf_va, args.coverage)
        threshold_source = "validation coverage"
    decisions = E.simulate_controller(preds, conf, target_state, op_thr)

    csv_path = os.path.join(args.results_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window_id", "true_label", "true_name", "pred_label",
                    "pred_name", "confidence", "confidence_raw", "correct",
                    "target_state", "threshold", "decision"])
        for i in range(len(y_te)):
            w.writerow([i, int(y_te[i]), class_names[y_te[i]], int(preds[i]),
                        class_names[preds[i]], round(float(conf[i]), 4),
                        round(float(conf_raw[i]), 4),
                        int(preds[i] == y_te[i]), args.target, round(op_thr, 4),
                        decisions[i]])

    summary = {
        "dataset": DATASET_NAME,
        "task": TASK_NAME,
        "split": "subject-wise (held-out subjects in test)",
        "n_channels": int(X.shape[1]),
        "n_times": int(X.shape[2]),
        "sfreq_hz": sfreq,
        "class_names": class_names,
        "n_windows": {"total": int(len(X)), "train": int(len(tr)),
                      "val": int(len(va)), "test": int(len(te))},
        "model": f"TinyEEGCNN ({n_params} params)",
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "temperature": round(float(T), 4),
        "ece_uncalibrated": round(ece_raw, 4),
        "ece_calibrated": round(ece_cal, 4),
        "confusion_matrix": metrics["confusion_matrix"],
        "risk_coverage_sweep": sweep,
        "controller": {"target_state": args.target,
                       "requested_coverage": args.coverage if args.threshold is None else None,
                       "confidence_threshold": round(float(op_thr), 4),
                       "threshold_source": threshold_source,
                       "rule": "STIMULATE iff pred==target and calibrated_confidence>=confidence_threshold",
                       "note": "SIMULATION ONLY -- no real stimulation, no clinical validity; "
                               "when --threshold is omitted, the confidence gate is chosen on "
                               "validation data for the requested coverage and then frozen for test",
                       "n_stimulate": int((decisions == "STIMULATE").sum()),
                       "n_wait": int((decisions == "WAIT").sum())},
    }
    with open(os.path.join(args.results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    V.plot_pipeline(os.path.join(args.figures_dir, "pipeline.png"))
    V.plot_confusion_matrix(metrics["confusion_matrix"], class_names,
                            os.path.join(args.figures_dir, "confusion_matrix.png"))
    V.plot_risk_coverage(sweep, metrics["accuracy"],
                         os.path.join(args.figures_dir, "risk_coverage.png"))
    V.plot_reliability(conf_raw, correct, ece_raw, conf, correct, ece_cal,
                       os.path.join(args.figures_dir, "calibration.png"))

    fire = (decisions == "STIMULATE") & (preds == y_te)
    j = int(np.where(fire)[0][np.argmax(conf[fire])]) if fire.any() else int(conf.argmax())
    V.plot_example_decision(Xte[j], ch_names, sfreq, class_names[preds[j]],
                            float(conf[j]), str(decisions[j]), args.target,
                            os.path.join(args.figures_dir, "example_decision.png"))

    acc20 = next((r["accuracy_accepted"] for r in sweep
                  if abs(r["coverage"] - 0.2) < 1e-9), None)
    update_readme_metrics("README.md", {
        "Dataset": DATASET_NAME,
        "Model": f"TinyEEGCNN ({n_params:,} params)",
        "EEG windows": f"{len(X)} ({len(te)} in held-out test)",
        "Balanced accuracy": f"{metrics['balanced_accuracy']:.3f}",
        "ECE (raw &rarr; calibrated)": f"{ece_raw:.3f} &rarr; {ece_cal:.3f}  (T = {T:.2f})",
        "Accuracy (most-confident 20%)": (f"{acc20:.3f}" if acc20 is not None else "n/a"),
    })

    print("\n===== RESULTS (held-out subjects) =====")
    print(f"accuracy            {metrics['accuracy']:.3f}")
    print(f"balanced accuracy   {metrics['balanced_accuracy']:.3f}")
    print(f"temperature (val)   {T:.3f}")
    print(f"ECE  raw -> calib   {ece_raw:.3f} -> {ece_cal:.3f}")
    print("coverage  acc_accepted  min_conf   (rank by calibrated confidence)")
    for r in sorted(sweep, key=lambda x: -x["coverage"]):
        print(f"  {r['coverage']:.2f}       {r['accuracy_accepted']:.3f}       {r['min_confidence']:.3f}")
    op = "abs" if args.threshold is not None else f"cov={args.coverage}"
    print(f"\nController (target='{args.target}', {op} -> conf>={op_thr:.3f}): "
          f"{summary['controller']['n_stimulate']} STIMULATE / "
          f"{summary['controller']['n_wait']} WAIT")
    print(f"\nWrote {csv_path}, results/summary.json, and 5 figures/. "
          "Closed-loop decision is a SIMULATION only.")


def main():
    p = argparse.ArgumentParser(description="NeuroLoop-Lite demo")
    p.add_argument("--prepare", action="store_true",
                   help="download + cache EEG windows, then exit")
    p.add_argument("--quick", action="store_true", help="tiny subject subset")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--target", default="right", choices=D.CLASS_NAMES,
                   help="brain state the controller acts on")
    p.add_argument("--coverage", type=float, default=0.30,
                   help="validation coverage used to set the simulated controller gate")
    p.add_argument("--threshold", type=float, default=None,
                   help="optional absolute confidence gate; overrides --coverage")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--figures-dir", default="figures")
    args = p.parse_args()

    if args.prepare:
        print(f"Preparing {DATASET_NAME} ({'quick' if args.quick else 'full'} subset)...")
        D.prepare_data(quick=args.quick, data_dir=args.data_dir)
    else:
        run(args)


if __name__ == "__main__":
    main()
