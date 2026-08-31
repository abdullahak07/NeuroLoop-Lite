"""
data.py -- prepare a small public EEG motor-imagery dataset.

Dataset: PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB), downloaded
through MNE. Runs 4/8/12 contain imagined left/right fist movement.
"""

import os
import numpy as np

RUNS = [4, 8, 12]
L_FREQ, H_FREQ = 7.0, 30.0
SFREQ = 160.0
WINDOW_SEC = 2.0
N_TIMES = int(WINDOW_SEC * SFREQ)
CLASS_NAMES = ["left", "right"]
SUBJECTS_FULL = list(range(1, 21))
SUBJECTS_QUICK = [1, 2, 3, 4]
CACHE_NAME = "eeg_windows.npz"


def _windows_from_subject(subject, verbose="ERROR"):
    import mne
    from mne.datasets import eegbci
    from mne.io import concatenate_raws, read_raw_edf

    fnames = eegbci.load_data(subject, RUNS, verbose=verbose)
    raw = concatenate_raws([read_raw_edf(f, preload=True, verbose=verbose) for f in fnames])
    eegbci.standardize(raw)
    raw.set_montage(mne.channels.make_standard_montage("standard_1005"),
                    on_missing="ignore", verbose=verbose)

    if int(round(raw.info["sfreq"])) != int(SFREQ):
        raw.resample(SFREQ, verbose=verbose)

    raw.set_eeg_reference("average", projection=False, verbose=verbose)
    raw.filter(L_FREQ, H_FREQ, fir_design="firwin", verbose=verbose)

    events, event_id = mne.events_from_annotations(raw, verbose=verbose)
    wanted = {k: event_id[k] for k in ("T1", "T2") if k in event_id}
    if len(wanted) < 2:
        return None, None, None

    epochs = mne.Epochs(raw, events, event_id=wanted,
                        tmin=0.0, tmax=WINDOW_SEC, baseline=None,
                        picks="eeg", preload=True, verbose=verbose)
    X = epochs.get_data(copy=True)[:, :, :N_TIMES].astype(np.float32)
    y = (epochs.events[:, -1] == event_id["T2"]).astype(np.int64)
    ch_names = np.array(epochs.ch_names)
    return (X, y, ch_names) if len(X) else (None, None, None)


def prepare_data(quick=False, data_dir="data", verbose="ERROR"):
    os.makedirs(data_dir, exist_ok=True)
    subjects = SUBJECTS_QUICK if quick else SUBJECTS_FULL

    Xs, ys, sids, ch_names = [], [], [], None
    for s in subjects:
        try:
            X, y, chs = _windows_from_subject(s, verbose=verbose)
        except Exception as e:
            print(f"  [subject {s:02d}] skipped: {e}")
            continue
        if X is None:
            print(f"  [subject {s:02d}] no usable epochs, skipped")
            continue
        Xs.append(X)
        ys.append(y)
        sids.append(np.full(len(y), s, dtype=np.int64))
        ch_names = chs if ch_names is None else ch_names
        print(f"  [subject {s:02d}] {len(y):3d} windows "
              f"({int((y == 0).sum())} left / {int((y == 1).sum())} right)")

    if not Xs:
        raise RuntimeError("No EEG windows were produced -- check the download.")

    X = np.concatenate(Xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.int64)
    subject_ids = np.concatenate(sids).astype(np.int64)

    out = os.path.join(data_dir, CACHE_NAME)
    np.savez_compressed(out, X=X, y=y, subject_ids=subject_ids,
                        ch_names=ch_names, sfreq=np.float32(SFREQ),
                        class_names=np.array(CLASS_NAMES))
    print(f"\nSaved {len(X)} windows of shape "
          f"({X.shape[1]} channels x {X.shape[2]} samples) -> {out}")
    return out


def load_windows(data_dir="data"):
    path = os.path.join(data_dir, CACHE_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python run_demo.py --prepare` first "
            "(add --quick for a tiny subset)."
        )
    d = np.load(path, allow_pickle=True)
    return {
        "X": d["X"].astype(np.float32),
        "y": d["y"].astype(np.int64),
        "subject_ids": d["subject_ids"].astype(np.int64),
        "ch_names": [str(c) for c in d["ch_names"]],
        "sfreq": float(d["sfreq"]),
        "class_names": [str(c) for c in d["class_names"]],
    }


def subject_wise_split(subject_ids, val_frac=0.2, test_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    subjects = np.unique(subject_ids)
    rng.shuffle(subjects)

    n = len(subjects)
    n_test = max(1, int(round(test_frac * n)))
    n_val = max(1, int(round(val_frac * n))) if n - n_test > 1 else 0
    test_s = set(subjects[:n_test])
    val_s = set(subjects[n_test:n_test + n_val])
    train_s = set(subjects[n_test + n_val:]) or set(subjects[:1])

    idx = np.arange(len(subject_ids))
    tr = idx[np.isin(subject_ids, list(train_s))]
    va = idx[np.isin(subject_ids, list(val_s))] if val_s else tr[:0]
    te = idx[np.isin(subject_ids, list(test_s))]
    return tr, va, te
