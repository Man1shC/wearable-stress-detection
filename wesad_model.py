"""
WESAD modeling: leave-one-subject-out evaluation of wrist-only stress detection.

Usage:
    python wesad_model.py features.csv
    python wesad_model.py features.csv --binary

Compares raw features against per-subject z-scored features, which is the
difference between a one-size-fits-all model and one calibrated to the wearer.
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

META = {"subject", "t_start", "label", "condition"}


def load(path, binary):
    df = pd.read_csv(path)
    if binary:
        # stress vs everything else - the framing a product would ship
        df["condition"] = np.where(df["condition"] == "stress", "stress", "non-stress")
    return df


def subject_normalize(X, groups):
    """Z-score each feature within each subject.

    Mirrors a wearable that calibrates to its wearer: what matters is how far
    the current window sits from that person's own distribution, not the
    absolute value.
    """
    out = X.copy()
    for s in np.unique(groups):
        mask = groups == s
        block = out[mask]
        mu = block.mean(axis=0)
        sd = block.std(axis=0)
        sd[sd == 0] = 1.0
        out[mask] = (block - mu) / sd
    return out


def evaluate(X, y, groups, model_fn, name):
    """Leave-one-subject-out CV. Returns pooled predictions."""
    logo = LeaveOneGroupOut()
    y_true_all, y_pred_all, per_subject = [], [], []

    for train_idx, test_idx in logo.split(X, y, groups):
        model = model_fn()
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])

        y_true_all.append(y[test_idx])
        y_pred_all.append(pred)
        per_subject.append((groups[test_idx][0],
                            f1_score(y[test_idx], pred, average="macro")))

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print(classification_report(y_true, y_pred, digits=3))

    labels = sorted(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("confusion matrix (rows = true, cols = predicted)")
    print(f"{'':>12}" + "".join(f"{l:>12}" for l in labels))
    for l, row in zip(labels, cm):
        print(f"{l:>12}" + "".join(f"{v:>12}" for v in row))

    scores = [s for _, s in per_subject]
    print(f"\nmacro F1 pooled: {f1_score(y_true, y_pred, average='macro'):.3f}")
    print(f"per-subject mean: {np.mean(scores):.3f}  (sd {np.std(scores):.3f})")
    worst = min(per_subject, key=lambda t: t[1])
    best = max(per_subject, key=lambda t: t[1])
    print(f"best  {best[0]}: {best[1]:.3f}   worst {worst[0]}: {worst[1]:.3f}")

    return f1_score(y_true, y_pred, average="macro")


def importances(X, y, feat_names):
    rf = RandomForestClassifier(n_estimators=300, random_state=0,
                                class_weight="balanced")
    rf.fit(X, y)
    order = np.argsort(rf.feature_importances_)[::-1]
    print(f"\n{'=' * 60}\ntop features\n{'=' * 60}")
    for i in order[:10]:
        bar = "#" * int(rf.feature_importances_[i] * 120)
        print(f"{feat_names[i]:>14}  {rf.feature_importances_[i]:.3f}  {bar}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--binary", action="store_true",
                   help="collapse to stress vs non-stress")
    args = p.parse_args()

    df = load(args.csv, args.binary)
    feat_names = [c for c in df.columns if c not in META]
    X = df[feat_names].to_numpy(dtype=float)
    y = df["condition"].to_numpy()
    groups = df["subject"].to_numpy()

    print(f"{len(df)} windows | {len(np.unique(groups))} subjects | "
          f"{len(feat_names)} features")
    print(df["condition"].value_counts().to_string())

    Xn = subject_normalize(X, groups)

    def logreg():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"))

    def forest():
        return RandomForestClassifier(n_estimators=300, random_state=0,
                                      class_weight="balanced", n_jobs=-1)

    results = {
        "logreg / raw": evaluate(X, y, groups, logreg, "Logistic regression - raw features"),
        "forest / raw": evaluate(X, y, groups, forest, "Random forest - raw features"),
        "logreg / norm": evaluate(Xn, y, groups, logreg, "Logistic regression - subject-normalized"),
        "forest / norm": evaluate(Xn, y, groups, forest, "Random forest - subject-normalized"),
    }

    importances(Xn, y, feat_names)

    print(f"\n{'=' * 60}\nsummary (macro F1)\n{'=' * 60}")
    for k, v in sorted(results.items(), key=lambda t: -t[1]):
        print(f"  {k:>16}  {v:.3f}")


if __name__ == "__main__":
    main()
