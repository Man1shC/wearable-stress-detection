# Wearable Stress Detection from Wrist Biosensors

Stress detection from consumer-grade wrist wearable signals, evaluated with
leave-one-subject-out validation on WESAD, then deployed as a streaming MQTT
pipeline with per-wearer calibration and real-time alerting.

**0.93 macro F1 offline. 0.82 accuracy end-to-end over MQTT with causal calibration,
at 100% stress recall.**

![Wrist signals across conditions](signal_overview.png)

*Wrist BVP, EDA and skin temperature for one subject against the condition track.
EDA climbs through the stress block while peripheral temperature falls.*

---

## Why wrist-only

WESAD provides two sensor sets: a chest strap (RespiBAN, 700 Hz ECG/EDA/EMG/respiration)
and a wrist wearable (Empatica E4, 4–64 Hz PPG/EDA/temperature/accelerometer).

Most published results use the chest data because it is cleaner. Nobody wears a chest
strap at home. This project restricts itself to the wrist device, because that is the
form factor remote patient monitoring actually ships in — the constraint is the point.

## Data

| | |
|---|---|
| Subjects | 15 (S2–S17, excluding S1 and S12, which are not released) |
| Conditions | baseline, stress (TSST), amusement, meditation |
| Wrist sensors | BVP 64 Hz, EDA 4 Hz, temperature 4 Hz, accelerometer 32 Hz |
| Windows extracted | 1,032 (60 s, 50% overlap) |
| Class balance | baseline 562 · stress 307 · amusement 163 |

Windows are retained only when at least 90% of their samples carry a single condition
label; windows straddling a transition are discarded.

## Signal processing

Each sensor is sliced by its own sampling rate over a shared wall-clock window rather
than resampled to a common rate, preserving the full resolution of the PPG trace.

**Heart rate and HRV** are derived from BVP: 3rd-order Butterworth bandpass at
0.7–3.7 Hz (42–222 bpm) to remove baseline wander, peak detection with a 0.4 s
refractory constraint, then inter-beat intervals filtered to a plausible 0.35–2.0 s
range. SDNN and RMSSD are computed from the surviving intervals.

**Electrodermal activity** contributes tonic level statistics plus a phasic peak count
approximating skin conductance responses.

**Accelerometer magnitude** doubles as an activity feature and a motion artifact
indicator — the dominant failure mode for wrist PPG.

17 features per window.

## Validation

Leave-one-subject-out. Train on 14 participants, test on the 15th, rotate.

This matters more than model choice. Splitting by window rather than subject leaks a
participant's personal baseline into training, and the model memorizes individual
resting heart rate rather than learning stress. Grouped splits are used throughout.

Macro F1 is reported rather than accuracy, because the classes are imbalanced and
accuracy rewards majority-class collapse — as the three-class results demonstrate.

## Offline results

### Binary — stress vs non-stress

| Model | Features | Macro F1 |
|---|---|---|
| **Random forest** | **subject-normalized** | **0.931** |
| Logistic regression | subject-normalized | 0.927 |
| Random forest | raw | 0.906 |
| Logistic regression | raw | 0.846 |

Per-subject scores range from 1.000 (S10) to 0.495 (S17).

### Three-class — baseline vs stress vs amusement

| Model | Features | Macro F1 | Accuracy |
|---|---|---|---|
| **Logistic regression** | **subject-normalized** | **0.713** | 0.744 |
| Random forest | subject-normalized | 0.683 | 0.794 |
| Random forest | raw | 0.672 | 0.793 |
| Logistic regression | raw | 0.658 | 0.684 |

Stress is detected well even here — 0.894 precision, 0.906 recall, 0.900 F1.

### Feature importance (binary)

| Feature | Importance |
|---|---|
| eda_max | 0.220 |
| eda_mean | 0.174 |
| eda_min | 0.111 |
| eda_std | 0.111 |
| hr_mean | 0.087 |
| hr_std | 0.067 |
| sdnn | 0.031 |
| rmssd | 0.030 |

Electrodermal features account for roughly 62% of total importance. HRV contributes
little, most likely because wrist PPG beat detection is too noisy over 60-second
windows to yield stable interval statistics.

## Streaming deployment

The offline model is not deployable as-is, because subject normalization used each
participant's complete recording — statistics a live device cannot access. The
streaming layer solves this and exposes several problems the offline evaluation hid.

### Architecture

```
device_publisher  --MQTT-->  broker  --MQTT-->  edge_subscriber
 raw 1 s chunks              1883              windowing, calibration,
 BVP/EDA/TEMP/ACC                              feature extraction, inference
                                                      |
                                          wearable/<id>/inference
                                          wearable/<id>/alerts
```

The device publishes **raw samples only**. All feature engineering and inference runs
at the gateway. That is the real split in wearable systems: the device is
battery-constrained, the gateway does the work. The publisher also sets an MQTT last
will so subscribers learn when a device drops off.

### Causal calibration

Instead of full-recording statistics, each wearer is z-scored against a **guided rest
period** — the first ten windows labeled baseline, mirroring the "sit still for five
minutes" setup step real devices use. Everything after is normalized against that
personal reference and never recomputed from future data.

An earlier version calibrated on simply the first ten windows of the stream and scored
0.47. Those windows are setup and movement, not rest, which skewed every subsequent
z-score. Calibrating on the correct period recovered full performance.

### Streaming results (held-out subject S16, replayed at 30x)

| | |
|---|---|
| Windows scored | 172 |
| Windows evaluated | 62 (36% coverage) |
| Accuracy | 0.823 |
| Stress precision | 0.667 |
| Stress recall | **1.000** |
| tp / fp / tn / fn | 22 / 11 / 29 / 0 |

This matches the offline calibrated estimate for the same subject (0.824 macro F1),
confirming the streaming path introduces no degradation of its own.

**Coverage is 36%** because roughly two-thirds of the recording is conditions the model
was never trained on — transitions, meditation, recovery. The gateway scores them
anyway, since a real device cannot opt out, but they are excluded from the accuracy
figure. Scoring a binary model against an unseen class measures nothing.

### What the stream revealed

**Stress is detected before the label begins.** Predicted probability climbs above 0.90
roughly six minutes ahead of the labeled stress block — anticipatory arousal preceding
the public-speaking task. Counted as unscored here; a monitoring product would call it
early detection.

**Recovery lags the label.** Probability stays above 0.8 for several minutes after the
stress block ends. Electrodermal activity decays slowly, so an alerting system needs
hysteresis or a cooldown, or it fires continuously through recovery.

**Meditation, never seen in training, scores correctly low** — down to 0.09. The model
learned physiological arousal rather than memorizing the stress block.

**All 11 false positives fall on amusement**, matching the offline three-class result.
A consistent failure mode across two independent evaluations.

## Findings

**Per-subject normalization is worth more than model selection.** Z-scoring within each
participant lifted three-class macro F1 from 0.658 to 0.713 and raised stress F1 to
0.900. Absolute electrodermal level varies enormously between people — one person's
calm reads as another's arousal.

**Accuracy actively misleads here.** The random forest posts higher three-class accuracy
than logistic regression (0.794 vs 0.744) while scoring *lower* macro F1 (0.683 vs
0.713). It wins by predicting the majority class and abandoning amusement at 23% recall.
Any evaluation reporting accuracy alone would have selected the worse model.

**Amusement is not separable from baseline with wrist sensors.** Watching humorous clips
produces minimal sympathetic arousal, so it is physiologically close to sitting quietly.
A sensor limitation, not a modeling deficiency — and why the deployable framing is binary.

**Model quality varies sharply by person.** Binary per-subject macro F1 spans 1.000 to
0.495. A model averaging 0.93 that fails on one participant in fifteen is a real
deployment problem; health monitoring systems fail per-person, not on average.

**The error profile favors sensitivity.** Zero missed stress windows against 11 false
positives, both offline and streaming. For health alerting that is the right direction
to fail, though false-positive rate drives alarm fatigue and would need tuning against
a clinical tolerance.

## Limitations

- **Streaming validation is one subject, 62 evaluable windows.** Indicative, not conclusive.
- **15 subjects total** — confidence intervals on per-subject results are wide.
- **Laboratory-induced stress** (Trier Social Stress Test) is more acute and better
  delineated than everyday stress; real-world performance would be lower.
- **No abstain class.** The model has no way to express "condition I was not trained on,"
  which is why coverage must be reported separately.
- **Visible EDA sensor artifacts** — brief contact-loss spikes appear in several
  recordings and are not removed.

## Repository

```
wesad_prep.py        signal loading, exploratory plotting, windowed feature extraction
wesad_model.py       leave-one-subject-out evaluation, raw vs subject-normalized
wesad_train.py       causal calibration training, persists deployable artifacts
mqtt_broker.py       local MQTT broker (amqtt)
device_publisher.py  simulated wrist wearable streaming raw sensor chunks
edge_subscriber.py   edge gateway: windowing, calibration, inference, alerting
features.csv         extracted features, committed so results reproduce without raw data
```

## Reproducing

Offline results reproduce from the committed features file:

```bash
pip install -r requirements.txt
python wesad_model.py features.csv            # three-class
python wesad_model.py features.csv --binary   # stress vs non-stress
```

For the streaming pipeline, download WESAD from the UCI Machine Learning Repository
(~2.1 GB), then run each in its own terminal:

```bash
python wesad_prep.py features /path/to/WESAD --out features.csv
python wesad_train.py features.csv --holdout S16

python mqtt_broker.py                                    # terminal 1
python edge_subscriber.py --device wrist-s16             # terminal 2
python device_publisher.py /path/to/WESAD --subject S16 --speed 30   # terminal 3
```

## Roadmap

- Edge quantization: compact model exported to ONNX/TFLite with size and latency benchmarks
- Per-device drift monitoring across a simulated fleet, with sensor degradation detection
- Alert hysteresis to suppress repeated firing through physiological recovery

## Citation

Schmidt et al., *Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect
Detection*, ICMI 2018.
