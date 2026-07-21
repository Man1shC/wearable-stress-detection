"""
Simulated wrist wearable. Replays a WESAD subject's raw sensor stream over MQTT.

Usage:
    python device_publisher.py . --subject S16 --speed 60

Publishes one-second chunks of raw sensor samples to:
    wearable/<device_id>/telemetry

The device sends raw samples only - no feature extraction, no inference. That is
the real split in wearable systems: the device is battery-constrained and dumb,
the gateway does the work.
"""

import argparse
import json
import time

import numpy as np
import paho.mqtt.client as mqtt

from wesad_prep import load_subject, FS

BROKER = "127.0.0.1"
PORT = 1883


def chunks(wrist, labels, fs_label=700):
    """Yield one second of every sensor at its native rate."""
    duration = int(len(labels) / fs_label)
    for sec in range(duration):
        payload = {}
        for name, rate in FS.items():
            seg = wrist[name][sec * rate:(sec + 1) * rate]
            # round to 4 dp - real BLE payloads are not float64
            payload[name] = np.round(seg, 4).tolist()
        # ground truth rides along so the subscriber can score itself;
        # a real device would not send this
        seg_lab = labels[sec * fs_label:(sec + 1) * fs_label]
        vals, counts = np.unique(seg_lab, return_counts=True)
        payload["_true_label"] = int(vals[np.argmax(counts)])
        payload["t"] = sec
        yield payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", help="folder containing S2/, S3/, ...")
    p.add_argument("--subject", default="S16")
    p.add_argument("--speed", type=float, default=60.0,
                   help="replay multiplier; 60 = one minute of wear per second")
    p.add_argument("--device-id", default=None)
    args = p.parse_args()

    device_id = args.device_id or f"wrist-{args.subject.lower()}"
    topic = f"wearable/{device_id}/telemetry"

    print(f"loading {args.subject}...")
    wrist, labels = load_subject(args.root, args.subject)
    total = int(len(labels) / 700)
    print(f"{total} s of wear, replaying at {args.speed}x -> ~{total/args.speed:.0f} s")

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                         client_id=device_id)
    # last will: if the device drops off, subscribers learn about it
    client.will_set(f"wearable/{device_id}/status",
                    json.dumps({"online": False}), qos=1, retain=True)
    client.connect(BROKER, PORT, keepalive=60)
    client.loop_start()
    client.publish(f"wearable/{device_id}/status",
                   json.dumps({"online": True, "subject": args.subject}),
                   qos=1, retain=True)

    interval = 1.0 / args.speed
    sent = 0
    try:
        for payload in chunks(wrist, labels):
            client.publish(topic, json.dumps(payload), qos=0)
            sent += 1
            if sent % 300 == 0:
                print(f"  {sent}/{total} s streamed")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        client.publish(f"wearable/{device_id}/status",
                       json.dumps({"online": False}), qos=1, retain=True)
        time.sleep(0.3)
        client.loop_stop()
        client.disconnect()
        print(f"done - {sent} chunks published to {topic}")


if __name__ == "__main__":
    main()
