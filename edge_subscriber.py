"""
Edge gateway. Subscribes to raw wearable telemetry, windows it, scores it, alerts.

Usage:
    python edge_subscriber.py --device wrist-s16

Pipeline per device:
    raw 1 s chunks -> rolling 60 s buffer -> features every 30 s
      -> calibration during a guided rest period
      -> z-score against that personal reference -> model -> publish

Two deployment realities this handles, which the offline evaluation did not:

1. Calibration happens during an instructed rest period, not simply the first
   windows of the stream. The opening minutes of a recording are setup and
   movement, and calibrating on them skews every later z-score.

2. The stream contains conditions the model was never trained on (transitions,
   meditation, recovery). The gateway still scores them - a real device cannot
   opt out - but they are excluded from the accuracy figure, since scoring a
   binary model against an unseen class measures nothing. Coverage is reported
   alongside accuracy so the exclusion stays visible.

Publishes:
    wearable/<id>/inference   every scored window
    wearable/<id>/alerts      only when stress is detected
"""

import argparse
import json
from collections import deque

import joblib
import numpy as np
import paho.mqtt.client as mqtt

from wesad_prep import (FS, acc_features, bvp_features, eda_features,
                        temp_features)

BROKER = "127.0.0.1"
PORT = 1883
WINDOW_SEC = 60
STEP_SEC = 30
BASELINE_LABEL = 1
KNOWN_LABELS = {1: "baseline", 2: "stress", 3: "amusement"}
ALL_LABELS = {0: "transition", 1: "baseline", 2: "stress",
              3: "amusement", 4: "meditation", 5: "other",
              6: "recovery", 7: "recovery"}


class DeviceState:
    """Rolling buffers, personal calibration reference, running evaluation."""

    def __init__(self, device_id, feature_names, n_calib):
        self.device_id = device_id
        self.feature_names = feature_names
        self.n_calib = n_calib
        self.buffers = {k: deque(maxlen=WINDOW_SEC * v) for k, v in FS.items()}
        self.truth = deque(maxlen=WINDOW_SEC)
        self.seconds = 0
        self.calib_rows = []
        self.mu = None
        self.sd = None
        self.n_published = 0
        self.n_evaluated = 0
        self.n_correct = 0
        self.tp = self.fp = self.tn = self.fn = 0

    def ingest(self, payload):
        for name in FS:
            self.buffers[name].extend(payload[name])
        self.truth.append(payload.get("_true_label", 0))
        self.seconds += 1

    def ready(self):
        return (self.seconds >= WINDOW_SEC
                and (self.seconds - WINDOW_SEC) % STEP_SEC == 0)

    def extract(self):
        bvp = np.array(self.buffers["BVP"], dtype=float).ravel()
        eda = np.array(self.buffers["EDA"], dtype=float).ravel()
        temp = np.array(self.buffers["TEMP"], dtype=float).ravel()
        acc = np.array(self.buffers["ACC"], dtype=float).reshape(-1, 3)
        feats = {}
        feats.update(bvp_features(bvp))
        feats.update(eda_features(eda))
        feats.update(temp_features(temp))
        feats.update(acc_features(acc))
        return np.array([feats[k] for k in self.feature_names], dtype=float)

    def calibrating(self):
        return self.mu is None

    def add_calibration(self, row):
        """Only accepts windows from the guided rest period."""
        self.calib_rows.append(row)
        if len(self.calib_rows) >= self.n_calib:
            block = np.vstack(self.calib_rows)
            self.mu = np.nanmean(block, axis=0)
            sd = np.nanstd(block, axis=0)
            sd[sd == 0] = 1.0
            self.sd = sd
            return True
        return False

    def normalize(self, row):
        return (row - self.mu) / self.sd

    def dominant_truth(self):
        vals, counts = np.unique(list(self.truth), return_counts=True)
        idx = int(np.argmax(counts))
        return int(vals[idx]), counts[idx] / counts.sum()

    def record(self, pred, truth_label):
        """Evaluate only against conditions the model was trained on."""
        if truth_label not in KNOWN_LABELS:
            return None
        actual = "stress" if truth_label == 2 else "non-stress"
        self.n_evaluated += 1
        self.n_correct += int(pred == actual)
        if pred == "stress" and actual == "stress":
            self.tp += 1
        elif pred == "stress":
            self.fp += 1
        elif actual == "stress":
            self.fn += 1
        else:
            self.tn += 1
        return actual


def build_handler(client, bundle, states):
    model = bundle["model"]
    # single-window inference; parallel trees add overhead, not speed
    model.n_jobs = 1
    feature_names = bundle["features"]
    n_calib = bundle["calib_windows"]
    classes = list(model.classes_)
    stress_idx = classes.index("stress")

    def on_message(_c, _u, msg):
        parts = msg.topic.split("/")
        if len(parts) < 3 or parts[2] != "telemetry":
            return
        device_id = parts[1]

        payload = json.loads(msg.payload)
        st = states.setdefault(
            device_id, DeviceState(device_id, feature_names, n_calib))
        st.ingest(payload)
        if not st.ready():
            return

        row = st.extract()
        if np.isnan(row).any():
            return

        truth_label, purity = st.dominant_truth()
        truth_name = ALL_LABELS.get(truth_label, "other")

        # --- calibration: guided rest period only ---
        if st.calibrating():
            if truth_label == BASELINE_LABEL and purity > 0.9:
                done = st.add_calibration(row)
                state = ("calibration complete" if done
                         else f"calibrating {len(st.calib_rows)}/{n_calib}")
                print(f"[{device_id}] t={st.seconds:5d}s  {state}")
            return

        # --- inference ---
        x = st.normalize(row).reshape(1, -1)
        proba = model.predict_proba(x)[0]
        p_stress = float(proba[stress_idx])
        pred = classes[int(np.argmax(proba))]
        st.n_published += 1

        actual = st.record(pred, truth_label)

        client.publish(f"wearable/{device_id}/inference", json.dumps({
            "device_id": device_id,
            "t": st.seconds,
            "prediction": pred,
            "p_stress": round(p_stress, 3),
            "hr_mean": round(float(row[feature_names.index("hr_mean")]), 1),
            "eda_mean": round(float(row[feature_names.index("eda_mean")]), 3),
            "true_condition": truth_name,
            "evaluated": actual is not None,
        }), qos=0)

        flag = "STRESS" if pred == "stress" else "  ok  "
        if actual is None:
            print(f"[{device_id}] t={st.seconds:5d}s  {flag}  p={p_stress:.2f}  "
                  f"true={truth_name:11s} (untrained condition, not scored)")
        else:
            mark = " " if pred == actual else "X"
            acc = st.n_correct / st.n_evaluated
            print(f"[{device_id}] t={st.seconds:5d}s  {flag}  p={p_stress:.2f}  "
                  f"true={truth_name:11s} {mark}  acc={acc:.3f}")

        if pred == "stress":
            client.publish(f"wearable/{device_id}/alerts", json.dumps({
                "device_id": device_id,
                "t": st.seconds,
                "p_stress": round(p_stress, 3),
                "severity": "high" if p_stress > 0.8 else "moderate",
            }), qos=1)

    return on_message


def summarize(states):
    print("\n" + "=" * 58)
    for did, st in states.items():
        if not st.n_evaluated:
            print(f"{did}: no evaluable windows")
            continue
        acc = st.n_correct / st.n_evaluated
        prec = st.tp / (st.tp + st.fp) if (st.tp + st.fp) else float("nan")
        rec = st.tp / (st.tp + st.fn) if (st.tp + st.fn) else float("nan")
        cov = st.n_evaluated / st.n_published
        print(f"{did}")
        print(f"  windows scored     {st.n_published}")
        print(f"  windows evaluated  {st.n_evaluated}  ({cov:.0%} coverage)")
        print(f"  accuracy           {acc:.3f}")
        print(f"  stress precision   {prec:.3f}")
        print(f"  stress recall      {rec:.3f}")
        print(f"  tp {st.tp}  fp {st.fp}  tn {st.tn}  fn {st.fn}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="+")
    p.add_argument("--model", default="artifacts/model.joblib")
    args = p.parse_args()

    bundle = joblib.load(args.model)
    print(f"model loaded: {len(bundle['features'])} features, "
          f"classes {list(bundle['model'].classes_)}, "
          f"calibration {bundle['calib_windows']} rest-period windows")

    states = {}
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="edge-gateway")
    client.on_message = build_handler(client, bundle, states)
    client.connect(BROKER, PORT, keepalive=60)
    client.subscribe(f"wearable/{args.device}/telemetry", qos=0)
    print(f"subscribed to wearable/{args.device}/telemetry - waiting for data\n")

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        summarize(states)
        client.disconnect()


if __name__ == "__main__":
    main()
