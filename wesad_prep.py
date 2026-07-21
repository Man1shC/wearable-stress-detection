"""
WESAD prep: load -> explore -> windowed wrist features.

Usage:
    python wesad_prep.py explore /path/to/WESAD S2
    python wesad_prep.py features /path/to/WESAD --out features.csv

Wrist (Empatica E4) sampling rates:
    BVP 64 Hz | EDA 4 Hz | TEMP 4 Hz | ACC 32 Hz
Label track: 700 Hz
    0 = transient/undefined
    1 = baseline, 2 = stress, 3 = amusement, 4 = meditation
    5,6,7 = ignore
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

FS = {"BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32}
FS_LABEL = 700
WINDOW_SEC = 60
STEP_SEC = 30
LABEL_PURITY = 0.90
KEEP_LABELS = {1: "baseline", 2: "stress", 3: "amusement"}

# S1 and S12 do not exist in the released dataset
SUBJECTS = [f"S{i}" for i in range(2, 18) if i != 12]


# ---------------------------------------------------------------- loading

def load_subject(root, subject):
    """Return (wrist_dict, label_array) for one subject."""
    path = os.path.join(root, subject, f"{subject}.pkl")
    with open(path, "rb") as f:
        # WESAD pickles were written under Python 2
        data = pickle.load(f, encoding="latin1")

    wrist = {k: np.asarray(v, dtype=float) for k, v in data["signal"]["wrist"].items()}
    labels = np.asarray(data["label"]).ravel().astype(int)
    return wrist, labels


# ---------------------------------------------------------------- explore

def explore(root, subject):
    """Plot BVP, EDA, TEMP against the label track for one subject."""
    import matplotlib.pyplot as plt

    wrist, labels = load_subject(root, subject)
    duration = len(labels) / FS_LABEL
    print(f"{subject}: {duration/60:.1f} min")
    for name, arr in wrist.items():
        print(f"  {name:5s} shape={arr.shape}")
    uniq, counts = np.unique(labels, return_counts=True)
    for u, c in zip(uniq, counts):
        print(f"  label {u} ({KEEP_LABELS.get(u, 'other'):9s}): {c/FS_LABEL/60:5.1f} min")

    panels = [("BVP", wrist["BVP"][:, 0]),
              ("EDA", wrist["EDA"][:, 0]),
              ("TEMP", wrist["TEMP"][:, 0])]

    fig, axes = plt.subplots(len(panels) + 1, 1, figsize=(14, 9), sharex=True)
    for ax, (name, sig) in zip(axes, panels):
        t = np.arange(len(sig)) / FS[name] / 60
        ax.plot(t, sig, lw=0.4)
        ax.set_ylabel(name)

    t_lab = np.arange(len(labels)) / FS_LABEL / 60
    axes[-1].plot(t_lab, labels, lw=0.8, color="crimson")
    axes[-1].set_ylabel("label")
    axes[-1].set_xlabel("minutes")
    axes[-1].set_yticks(sorted(uniq))

    fig.suptitle(f"WESAD {subject} — wrist signals vs condition")
    fig.tight_layout()
    out = f"{subject}_explore.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


# ---------------------------------------------------------------- features

def _slope(x):
    """Least-squares slope per sample; flat signals return 0."""
    if len(x) < 2:
        return 0.0
    idx = np.arange(len(x))
    return float(np.polyfit(idx, x, 1)[0])


def _bandpass(x, fs, lo, hi, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, x)


def bvp_features(bvp):
    """Heart rate and HRV from the wrist PPG trace."""
    fs = FS["BVP"]
    out = {"hr_mean": np.nan, "hr_std": np.nan,
           "sdnn": np.nan, "rmssd": np.nan, "n_beats": 0}
    if len(bvp) < fs * 5:
        return out

    # 0.7-3.7 Hz keeps 42-222 bpm and strips baseline wander
    filt = _bandpass(bvp, fs, 0.7, 3.7)
    # beats cannot be closer than 0.4 s apart (150 bpm ceiling on spacing)
    peaks, _ = find_peaks(filt, distance=int(0.4 * fs), prominence=np.std(filt) * 0.5)
    if len(peaks) < 4:
        return out

    ibi = np.diff(peaks) / fs          # inter-beat intervals, seconds
    ibi = ibi[(ibi > 0.35) & (ibi < 2.0)]   # drop physiologically implausible
    if len(ibi) < 3:
        return out

    hr = 60.0 / ibi
    out["hr_mean"] = float(np.mean(hr))
    out["hr_std"] = float(np.std(hr))
    out["sdnn"] = float(np.std(ibi) * 1000)                    # ms
    out["rmssd"] = float(np.sqrt(np.mean(np.diff(ibi) ** 2)) * 1000)
    out["n_beats"] = int(len(peaks))
    return out


def eda_features(eda):
    """Tonic level plus a crude phasic peak count."""
    out = {"eda_mean": np.nan, "eda_std": np.nan,
           "eda_slope": np.nan, "eda_min": np.nan,
           "eda_max": np.nan, "eda_peaks": 0}
    if len(eda) < 8:
        return out
    out["eda_mean"] = float(np.mean(eda))
    out["eda_std"] = float(np.std(eda))
    out["eda_slope"] = _slope(eda)
    out["eda_min"] = float(np.min(eda))
    out["eda_max"] = float(np.max(eda))
    # skin conductance responses: small rises above the tonic level
    peaks, _ = find_peaks(eda, prominence=max(np.std(eda) * 0.5, 0.01))
    out["eda_peaks"] = int(len(peaks))
    return out


def temp_features(temp):
    if len(temp) < 4:
        return {"temp_mean": np.nan, "temp_std": np.nan, "temp_slope": np.nan}
    return {"temp_mean": float(np.mean(temp)),
            "temp_std": float(np.std(temp)),
            "temp_slope": _slope(temp)}


def acc_features(acc):
    """Movement magnitude — doubles as a motion-artifact indicator."""
    if len(acc) < 4:
        return {"acc_mag_mean": np.nan, "acc_mag_std": np.nan, "acc_mag_max": np.nan}
    mag = np.linalg.norm(acc, axis=1)
    return {"acc_mag_mean": float(np.mean(mag)),
            "acc_mag_std": float(np.std(mag)),
            "acc_mag_max": float(np.max(mag))}


def window_subject(root, subject):
    wrist, labels = load_subject(root, subject)
    duration = len(labels) / FS_LABEL
    rows = []

    for start in np.arange(0, duration - WINDOW_SEC, STEP_SEC):
        end = start + WINDOW_SEC

        seg = labels[int(start * FS_LABEL):int(end * FS_LABEL)]
        if len(seg) == 0:
            continue
        vals, counts = np.unique(seg, return_counts=True)
        dominant = int(vals[np.argmax(counts)])
        purity = counts.max() / counts.sum()
        if dominant not in KEEP_LABELS or purity < LABEL_PURITY:
            continue

        def slice_sig(name):
            fs = FS[name]
            return wrist[name][int(start * fs):int(end * fs)]

        row = {"subject": subject, "t_start": float(start),
               "label": dominant, "condition": KEEP_LABELS[dominant]}
        row.update(bvp_features(slice_sig("BVP")[:, 0]))
        row.update(eda_features(slice_sig("EDA")[:, 0]))
        row.update(temp_features(slice_sig("TEMP")[:, 0]))
        row.update(acc_features(slice_sig("ACC")))
        rows.append(row)

    return pd.DataFrame(rows)


def features(root, out_path):
    frames = []
    for s in SUBJECTS:
        try:
            df = window_subject(root, s)
        except FileNotFoundError:
            print(f"  {s}: missing, skipped")
            continue
        print(f"  {s}: {len(df):4d} windows  " +
              "  ".join(f"{k}={v}" for k, v in df["condition"].value_counts().items()))
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    n_before = len(all_df)
    all_df = all_df.dropna(subset=["hr_mean", "eda_mean"])
    print(f"\ndropped {n_before - len(all_df)} windows with unusable BVP")
    print(f"total: {len(all_df)} windows, {all_df['subject'].nunique()} subjects")
    print(all_df["condition"].value_counts())
    all_df.to_csv(out_path, index=False)
    print(f"saved {out_path}")


# ---------------------------------------------------------------- cli

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("explore")
    e.add_argument("root")
    e.add_argument("subject", nargs="?", default="S2")

    f = sub.add_parser("features")
    f.add_argument("root")
    f.add_argument("--out", default="features.csv")

    args = p.parse_args()
    if args.cmd == "explore":
        explore(args.root, args.subject)
    else:
        features(args.root, args.out)


if __name__ == "__main__":
    main()
