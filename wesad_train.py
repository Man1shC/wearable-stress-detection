"""
Train the deployable WESAD stress model and persist it for the streaming layer.

Usage:
    python wesad_train.py features.csv --holdout S16
    python wesad_train.py features.csv --holdout S16 --calib-windows 20

Key difference from wesad_model.py: normalization here is CAUSAL. Each subject
is z-scored against a calibration window taken from the start of their own
recording, never against their full session. This is what a deployed wearable
can actually compute, and it is the number you should report.

Outputs:
    artifacts/model.joblib   trained classifier + feature order + metadata
    artifacts/reference.csv  training feature distribution, for drift monitoring
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib

META = {"subject", "t_start", "label", "condition"}
ARTIFACTS = "artifacts"
CALIB_WINDOWS = 10          # 10 windows x 30 s step = first ~5 min of wear


def calibration_stats(block, n_calib):
    """Mean and sd from the first n_calib windows of one subject's recording."""
    calib = block[:n_calib]
    mu = calib.mean(axis=0)
    sd = calib.std(axis=0)
    sd[sd == 0] = 1.0
    return mu, sd


def causal_normalize(df, feat_names, n_calib):
    """Z-score each subject against their own opening calibration window.

    Rows must already be sorted by t_start within subject, which the feature
    extractor guarantees.
    """
    out = []
    for subject, block in df.groupby("subject", sort=False):
        block = block.sort_values("t_start")
        X = block[feat_names].to_numpy(dtype=float)
        if len(X) <= n_calib:
            continue
        mu, sd = calibration_stats(X, n_calib)
        # calibration windows themselves are consumed, not scored
        scored = block.iloc[n_calib:].copy()
        scored[feat_names] = (X[n_calib:] - mu) / sd
        out.append(scored)
    return pd.concat(out, ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--holdout", default="S16",
                   help="subject reserved for the streaming demo")
    p.add_argument("--calib-windows", type=int, default=CALIB_WINDOWS)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    df["target"] = np.where(df["condition"] == "stress", "stress", "non-stress")
    feat_names = [c for c in df.columns if c not in META | {"target"}]

    norm = causal_normalize(df, feat_names, args.calib_windows)

    train = norm[norm["subject"] != args.holdout]
    test = norm[norm["subject"] == args.holdout]
    if test.empty:
        raise SystemExit(f"holdout {args.holdout} not found in {args.csv}")

    print(f"train: {len(train)} windows / {train['subject'].nunique()} subjects")
    print(f"test:  {len(test)} windows / holdout {args.holdout}")
    print(f"calibration: first {args.calib_windows} windows per subject, discarded\n")

    Xtr = train[feat_names].to_numpy(dtype=float)
    ytr = train["target"].to_numpy()
    Xte = test[feat_names].to_numpy(dtype=float)
    yte = test["target"].to_numpy()

    model = RandomForestClassifier(n_estimators=400, random_state=0,
                                   class_weight="balanced", n_jobs=-1)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    print("=" * 60)
    print(f"held-out subject {args.holdout} - causal calibration normalization")
    print("=" * 60)
    print(classification_report(yte, pred, digits=3))

    labels = sorted(np.unique(yte))
    cm = confusion_matrix(yte, pred, labels=labels)
    print("confusion matrix (rows = true, cols = predicted)")
    print(f"{'':>12}" + "".join(f"{l:>12}" for l in labels))
    for l, row in zip(labels, cm):
        print(f"{l:>12}" + "".join(f"{v:>12}" for v in row))
    print(f"\nmacro F1: {f1_score(yte, pred, average='macro'):.3f}")

    os.makedirs(ARTIFACTS, exist_ok=True)
    joblib.dump({
        "model": model,
        "features": feat_names,
        "classes": list(model.classes_),
        "calib_windows": args.calib_windows,
        "holdout": args.holdout,
    }, os.path.join(ARTIFACTS, "model.joblib"))

    # reference distribution for drift monitoring later
    ref = train[feat_names + ["target"]]
    ref.to_csv(os.path.join(ARTIFACTS, "reference.csv"), index=False)

    with open(os.path.join(ARTIFACTS, "meta.json"), "w") as f:
        json.dump({"holdout": args.holdout,
                   "calib_windows": args.calib_windows,
                   "n_train": len(train),
                   "macro_f1_holdout": float(f1_score(yte, pred, average="macro"))},
                  f, indent=2)

    print(f"\nsaved {ARTIFACTS}/model.joblib, reference.csv, meta.json")


if __name__ == "__main__":
    main()
