# ABEN Dual RGB Field Imaging System — User Guide

This is the full guide to the operator console: every tab, every
safety system, the training workflow, and what to do when something
goes wrong. For a quick project overview, see the top-level
[`README.md`](../README.md). For safety systems specifically, see
[`SAFETY.md`](SAFETY.md) — read that one first if you haven't.

## Contents

1. [What this system is](#what-this-system-is)
2. [Installation](#installation)
3. [Hardware setup checklist](#hardware-setup-checklist)
4. [First launch](#first-launch)
5. [The header bar](#the-header-bar)
6. [Tab 1 — Data Collection](#tab-1--data-collection)
7. [Tab 2 — Detection](#tab-2--detection)
8. [Tab 3 — Session Analysis](#tab-3--session-analysis)
9. [A typical field session, start to finish](#a-typical-field-session-start-to-finish)
10. [YOLO model training & deployment](#yolo-model-training--deployment)
11. [Research data output](#research-data-output)
12. [Theming](#theming)
13. [Troubleshooting](#troubleshooting)
14. [Architecture reference](#architecture-reference)

---

## What this system is

A Husky A200 ground robot carries a fixed dual-camera gantry (two
eMeet SmartCam C960 4K cameras, left and right) over a crop row. A
Jetson AGX Orin runs YOLO instance-segmentation inference on both
camera feeds in real time, maps detections to one of four pixel
zones, and fires the corresponding one of three physical spray
nozzles when a weed or disease target is confirmed. An Arduino
handles the actual pump/nozzle/light hardware over serial.

Two detection modes exist:
- **WEED mode** — sprays herbicide on detected weed species
- **CLS mode** — sprays fungicide and flags detections for disease
  (Cercospora Leaf Spot) research mapping; every CLS spray is logged
  as a research data point regardless of outcome

Every spray event — zone, nozzle, detected class, confidence, GPS,
robot pose, timestamp — is recorded to disk for research use. See
[Research data output](#research-data-output).

## Installation

```bash
git clone https://github.com/bmensah2/data_spray_gui.git
cd data_spray_gui
pip install -r requirements.txt --break-system-packages
npm install   # for generate_report_rgb.js
```

Python dependencies: PyQt5, opencv-python, numpy, pyserial, tifffile,
PyYAML, ultralytics. The Husky-side scripts in `navigation/` run in a
separate ROS Noetic environment on the Husky's own onboard PC — they
are not part of this pip install. See
[`navigation/README.md`](../navigation/README.md).

Before training any YOLO model, also run:
```bash
python3 check_yolo_env.py
```
This confirms `ultralytics`/`torch`/CUDA are actually working — see
[YOLO model training & deployment](#yolo-model-training--deployment).

## Hardware setup checklist

Before launching the GUI for a real session:

- [ ] Husky powered on, reachable at `192.168.131.1`, `husky_odom_pub.py`
      running on its onboard PC (should auto-start via systemd)
- [ ] Arduino running `gantry_detection/gantry_detection.ino`, connected
      via USB (default port `/dev/ttyACM0`)
- [ ] Both eMeet cameras connected and visible to the Jetson
- [ ] Spray tank filled, lines primed (use the **PRIME PUMP** control
      in Detection tab before a real run)
- [ ] Physical wireless E-stop remote (if your Husky has one) within
      reach and tested

## First launch

```bash
cd /media/pagsun/Transcend/phd_project/emeet_dual_cam
python3 main_gui_rgb.py
```

The app opens on **Data Collection**, the only tab with active
navigation controls at launch (see
[Cross-tab motion lock](SAFETY.md#cross-tab-motion-lock)). The theme
that loads is whatever was last selected via `View > Theme` — the
choice persists across restarts.

---

## The header bar

Two rows, visible on every tab:

**Row 1** — title, four status LEDs (GANTRY / CAMERA / DETECT / NAV,
live connection indicators), a **Depth Cam** button (opens the
RealSense viewer as a separate process, if `pyrealsense2` is
installed), and the **E-STOP** button.

**Row 2** — Arduino connection bar: a warning message when
disconnected, a serial port dropdown, a refresh-ports button, and
**CONNECT ARDUINO** / **DISCONNECT**. This bar is deliberately always
visible regardless of which tab you're on, since nothing hardware-side
works until the Arduino is connected.

**Menu bar** — `File > Quit`, `View > Theme` (7 themes, see
[Theming](#theming)), `Help > About`.

---

## Tab 1 — Data Collection

Purpose: capture and organize labeled training data. This tab's
navigation controls are active by default; Detection's are locked
until you arm it (see [Cross-tab motion lock](SAFETY.md#cross-tab-motion-lock)).

**Left panel** has two sub-tabs:
- **Gantry** — camera servo angle control, movement sequences
  (`m <inches> a <degrees>` token syntax), AUX light/motor PSU
  toggles, and system status (limit switch, move/home speed).
- **Data Collection** — camera exposure/white-balance presets
  (Outdoor/Cloudy/Indoor — same controls also appear inside Detection
  tab so you don't need to switch tabs mid-session to adjust
  lighting), device status, auto-capture, and video recording.

**Middle** — live camera feed (side-by-side or other view modes),
shared with Detection tab (same underlying camera connection either
way — connecting once from either tab connects both).

**Right panel** — Navigation: manual movement (distance/speed
spinboxes, Forward/Backward/rotate), quick-move presets, mission
list (load/run/pause/stop a saved YAML mission), and the always-live
**STOP** button.

**Bottom** — system log, filtered to GANTRY/CAMERA/NAV/SYS sources
for this tab.

## Tab 2 — Detection

Purpose: arm weed/disease detection and let the system spray
automatically. This is the tab you'll spend the most time in during
an active field run.

**Top bar** (always visible): **ARM DETECTION** / **STOP** / **E-STOP**.
Arming here is what unlocks this tab's navigation controls and locks
Data Collection's — see [Cross-tab motion lock](SAFETY.md#cross-tab-motion-lock).
It's also what starts the real-time detection loop, the actuation
controller, and (if `cfg.logging.log_spray_events` is enabled, which
it is by default) the event logger that records every spray to disk.

**Left panel** has three sub-tabs:
- **Spray** — spray mission controls (drives the Husky + fires
  nozzles automatically per a mission profile: model path, distance,
  speed, field ID, dry-run/dummy-detect flags for bench testing) and
  a manual **HOLD TO PURGE** button for nozzle/line testing.
- **Detect** — mode selector (WEED/CLS), detection threshold, model
  loading, per-zone status (three color-coded zone groups showing
  live detection counts and confirmed detections), per-nozzle LEDs,
  pump enable/disable (a manual safety gate — auto-spray cannot fire
  until you explicitly enable the pump here), pump prime control.
- **Camera** — the same lighting presets (Outdoor/Cloudy/Indoor)
  available in Data Collection, so you can adjust exposure without
  switching tabs mid-session.

**Middle** — live camera feed with detection overlay.

**Right** — Navigation (same shared panel concept as Tab 1, but this
copy's controls only work while armed).

**Bottom** — system log for this tab.

### Arming sequence

1. Connect Arduino (header bar) — without it, detection runs in
   **dry-run mode**: full detection/zone/logging pipeline runs
   normally, but no real hardware commands go out. Useful for
   bench-testing the software without risking a real spray.
2. Load a model, or leave it in **stub mode** (no real detections,
   for testing the rest of the pipeline).
3. Enable the pump (Detection sub-tab) — required before auto-spray
   can fire, independent of arming.
4. Click **ARM DETECTION**.
5. When done: **STOP** to disarm cleanly (flushes the event log and
   reports the session's total event count), or **E-STOP** if
   something needs to stop immediately.

## Tab 3 — Session Analysis

Purpose: a live dashboard of what's happening in the *current*
detection session — not a camera view (that's already on Tabs 1/2).

- **Live Spray Event Feed** — every spray event, newest first: time,
  zone, nozzle, detected class, confidence, pose, GPS. Capped at 300
  displayed rows for responsiveness on a long session; the underlying
  data used for stats and the map is never capped.
- **Session Stats** — running totals: event count, events/minute,
  confidence mean/range, GPS coverage %, CLS-flagged count.
- **Events by Zone / Class** — quick breakdown.
- **System Status** — live Arduino/pump/nozzle state, Husky link
  status, armed/mode/EStop state — everything that's otherwise spread
  across different tabs, in one place.
- **Spray Location Map** — a simple 2D top-down scatter of where
  sprays happened this session, relative to session start, colored by
  mode (weed/CLS), with CLS-flagged events ringed and a faint robot
  path trail.

The feed/stats/map all reset automatically when a new session starts
(ARM). A **Clear Feed** button is also available if you want to reset
the view mid-session without re-arming.

---

## A typical field session, start to finish

1. Run through the [hardware setup checklist](#hardware-setup-checklist).
2. Launch the GUI, connect the Arduino from the header.
3. On Data Collection, connect the cameras and confirm the feed looks
   right.
4. If you need fresh training images first: use Data Collection's
   capture controls, then label them separately and retrain (see
   [YOLO model training](#yolo-model-training--deployment)) before
   your next session.
5. Switch to Detection. Load your trained model. Choose WEED or CLS
   mode.
6. Enable the pump.
7. **ARM DETECTION.**
8. Drive the field pass (manual controls or a saved mission).
9. Watch Session Analysis if you want live feedback on what's being
   sprayed and where.
10. **STOP** to end the session cleanly. Note the reported event count.
11. Check `logs/events/<session_id>/` for the JSONL/CSV/summary/GeoJSON
    output (see [Research data output](#research-data-output)).

---

## YOLO model training & deployment

The system supports **YOLOv8, YOLO11, and YOLO26** through one shared
pipeline. The runtime detection engine
(`core/detection_engine_rgb.py`) doesn't care which family produced a
`.pt` file — Ultralytics auto-detects the architecture from the
checkpoint. Model choice only matters at training time.

### 1. Check your environment

```bash
python3 check_yolo_env.py
```

Confirms `ultralytics` is installed and recent enough to recognize
YOLO26, and — critically — that CUDA is actually available to
`torch`. **On a Jetson, a plain `pip install torch` pulls a generic
build with no Jetson GPU support.** It imports fine and "works," but
silently trains on CPU only, which is 10–50× slower. If this script
reports CUDA unavailable, fix that before training anything — it
explains the correct NVIDIA JetPack-matched wheel to install instead.

### 2. Prepare your dataset

Expected structure (standard YOLO format):
```
yolo_dataset/
├── images/{train,val}/*.jpg
├── labels/{train,val}/*.txt
└── data_rgb.yaml
```

If your dataset came from an external export tool, it may already
have a `data_rgb.yaml` and `classes.json` written — worth checking
(`ls`, `cat`) before assuming you need to generate one. If the paths
and class list in an existing `data_rgb.yaml` already look correct,
use `--validate-only` below rather than regenerating it.

A quick manual sanity check before running anything is also worth
doing on a dataset you haven't used before:
```bash
find <dataset>/images/train -type f | wc -l
find <dataset>/labels/train -type f -name "*.txt" | wc -l
```
Image and label counts should match on both splits. `prepare_rgb_dataset.py`
below checks this properly (including which specific files don't
pair up), but a quick count first catches an obviously wrong path
immediately.

```bash
python3 prepare_rgb_dataset.py --dataset /path/to/yolo_dataset \
    --classes <your classes in class-id order> \
    --validate-only   # if data_rgb.yaml already exists and looks right
```
This checks every image has a matching label and vice versa, prints
per-class annotation counts for train/val (a good moment to notice a
badly underrepresented class before spending hours training on it),
and writes `data_rgb.yaml` unless `--validate-only` is given. Aim for
100% pairing (paired count == image count == label count on both
splits) before training — any mismatch means something is wrong with
the export, not something training will work around.

### 3. Train

**One model:**
```bash
python3 train_yolo_rgb.py --dataset /path/to/data_rgb.yaml --model yolo26n
```
`--model` accepts `yolov8n`, `yolov8s`, `yolo11n`, `yolo11s`,
`yolo26n`, `yolo26s` and auto-sets both the pretrained checkpoint and
a run name (`<model>_rgb_v1`) so training different families back to
back doesn't overwrite the same output directory. `--weights` and
`--name` are still available directly if you want full manual control.

**All three, compared:**
```bash
python3 compare_yolo_models_rgb.py --dataset /path/to/data_rgb.yaml --epochs 100
```
Trains (or, with `--skip-training`, evaluates already-trained weights
for) every requested family with identical hyperparameters, then
prints a side-by-side table: mAP50, mAP50-95, mask mAP50, inference
speed (ms/image), training time. It deliberately does **not**
auto-pick a winner — accuracy and inference speed trade off, and on a
real-time spray robot a slower-but-marginally-more-accurate model can
be the wrong deployment choice. The summary names the fastest and the
most accurate separately so you decide with your own priorities.

Recommended before a long real run: do a 2-epoch smoke test first
(`--epochs 2`) to confirm the whole pipeline works on your actual
hardware/dataset before committing hours. Consider running the real
comparison inside `tmux`/`screen` so it survives a disconnected
terminal.

**Reading a smoke-test result correctly** — a short run (2 epochs) is
for confirming the pipeline *works*, not for judging which model is
best. Specifically, don't read too much into:
- **Low mAP at 2 epochs** — none of these models have converged yet;
  0.1–0.2 mAP50-95 after 2 epochs is normal and not predictive of the
  100-epoch result.
- **The training-time column** — the script trains each family in the
  same process, one after another. The *first* model trained absorbs
  most of the one-time CUDA/cuDNN warm-up cost (visible in the raw
  per-epoch logs as a much slower epoch 1 than epoch 2), which makes
  it look artificially slower in a short smoke test. Over a full
  100-epoch run this bias becomes negligible, but don't conclude
  "model X is slower to train" from a 2-epoch smoke test's timing
  alone.
- **YOLO26 in particular** may show a different early-training curve
  than v8/v11 — it uses a different optimizer (MuSGD) and has an
  extra loss term (visible as `sem_loss` in `results.csv`) that v8/v11
  don't have. A lower mAP at epoch 2 doesn't mean it's worse; check
  whether it's still improving epoch to epoch before drawing any
  conclusion, and only trust the comparison once all three have run
  their full epoch count.

### 4. Evaluate a specific model

```bash
python3 evaluate_model_rgb.py --weights runs/aben_rgb/<run>/weights/best.pt \
    --dataset /path/to/data_rgb.yaml
```
Writes a JSON with box/mask mAP, precision/recall, confusion matrix,
and speed — the same function `compare_yolo_models_rgb.py` calls
internally, useful standalone if you just want numbers for one model.

### 5. Deploy

Copy the winning `best.pt` to `models/weed_rgb.pt` (for WEED mode) or
`models/cls_rgb.pt` (for CLS mode) — the GUI loads these automatically
when Detection is armed. `train_yolo_rgb.py` does this copy
automatically unless you override `--models-dest-name` (useful during
a comparison run, so intermediate models don't clobber what's
currently deployed).

Class names don't need any manual configuration after training —
Ultralytics stores the trained class list inside the `.pt` checkpoint
itself, and `RGBDetectionEngine` reads `model.names` from it
automatically on load. The class names hardcoded in
`core/detection_engine_rgb.py` (`DEFAULT_CLASS_NAMES`) are only a
stub-mode fallback shown before any real model is loaded — if you
retrain with a different class list than what's currently in that
fallback, the GUI will still show the *correct* classes once armed
with a real model; only the pre-training stub-mode display would be
stale. Worth updating that fallback dict to match your latest dataset
anyway, purely so stub-mode testing (no model loaded) shows accurate
class names too.

---

## Research data output

While Detection is armed, every real spray event is written to
`logs/events/<session_id>/`:

- **`<session_id>_events.jsonl`** — one JSON object per line,
  append-only, written incrementally (survives a crash mid-session —
  writes are retried on failure rather than silently dropped)
- **`<session_id>_events.csv`** — same data, flat, for R/Excel/MATLAB
- **`<session_id>_event_summary.json`** — written on disarm: totals,
  per-class/per-zone breakdown, GPS coverage, confidence stats
- **`<session_id>_map.geojson`** — spray locations for QGIS/ArcGIS,
  via `EventLogger.export_geojson()`

A lightweight `<session_id>_session.json` (mode, growth stage, field
ID, zone calibration) is also written immediately at ARM, independent
of whether any spray events occur — so a record exists even if the
session ends unexpectedly before the first spray.

---

## Theming

`View > Theme` in the menu bar. Seven themes, applied live to the
whole app:

| Theme | Look |
|---|---|
| ABEN Dark (default) | the original field-tested dark theme |
| Dark Professional | enhanced dark, blue accents |
| Dark Blue | sophisticated blue-tinted dark |
| Midnight | deep, code-editor style |
| Forest | green, vegetation-friendly |
| Scientific | bold pink/navy |
| Charcoal | elegant charcoal, orange accents |

Your choice persists across restarts (`gui/theme_config.json`).

---

## Troubleshooting

**Arduino won't connect** — check the port dropdown matches your
actual device (`/dev/ttyACM0` by default), confirm nothing else has
the port open, try the refresh-ports button.

**Detection stays in stub mode / no detections** — no model is
loaded, or the model file isn't at the expected path
(`models/weed_rgb.pt` / `models/cls_rgb.pt`). Load a model explicitly
from the Detection sub-tab, or check the system log for the exact
path it tried.

**"5/6 checks passed" (or similar) after Run Field Checks** — this is
the pre-flight hardware checklist in the Spray panel (Arduino, camera,
SSH-to-Husky, odom UDP, pump cycle, nozzle cycle). It tells you
exactly which check failed — a failed camera check usually just means
the camera wasn't connected when you ran it.

**Navigation controls greyed out on a tab** — expected behavior, not
a bug — see [Cross-tab motion lock](SAFETY.md#cross-tab-motion-lock).
Only one tab's movement controls are ever active at a time.

**Nozzle stays open longer than expected** — check the log for a
continuous-spray warning (see
[SAFETY.md](SAFETY.md#continuous-spray-sanity-guard)) — this is
advisory, not a bug, but worth checking what the camera is actually
seeing.

**CUDA unavailable during training** — see
[step 1 of the training workflow](#yolo-model-training--deployment)
above; this is a Jetson-specific PyTorch wheel issue, not a code bug.

**Husky navigation not responding** — confirm `husky_odom_pub.py` is
running on the Husky's onboard PC (should auto-start via systemd —
see [`navigation/README.md`](../navigation/README.md)), and that the
Jetson can reach `192.168.131.1`.

---

## Architecture reference

For anyone extending this codebase — a brief map of what lives where:

| Path | What |
|---|---|
| `main_gui_rgb.py` | App entry point, `MainWindow`, header, menu, theme wiring |
| `gui/tabs/` | The three tab assemblers (`tab_collection.py`, `tab_detection.py`, `tab_analysis_rgb.py`) |
| `gui/panels/` | Individual panel widgets (gantry, navigation, spray, detection, acquisition, camera) |
| `gui/theme_manager.py` | The 7-theme registry-based theming engine |
| `gui/style.py` | Shared style helpers (`LED`, `_muted`, `_sec`, `_divider`) built on top of `theme_manager` |
| `core/detection_engine_rgb.py` | YOLO inference wrapper — model-family-agnostic |
| `core/zone_manager_rgb.py` | Maps detections to the 4 pixel zones / 3 nozzles, with debounce filtering |
| `core/actuation_controller.py` | Hardware actuation: nozzle/pump sequencing, minimum hold floor, E-stop handling |
| `core/gantry_controller.py` | Serial protocol to the Arduino |
| `core/ros_bridge.py` | UDP telemetry receiver from the Husky (pose, heartbeat, estop) |
| `core/event_logger.py` | Research data logging (JSONL/CSV/summary/GeoJSON) |
| `core/detection_config_rgb.py` | All configuration dataclasses (`RGBConfig` and friends) |
| `gantry_detection/gantry_detection.ino` | Arduino firmware, including the comms watchdog |
| `navigation/` | Scripts that run on the Husky's own onboard PC (separate ROS environment) |
| `train_yolo_rgb.py`, `compare_yolo_models_rgb.py`, `evaluate_model_rgb.py`, `prepare_rgb_dataset.py`, `check_yolo_env.py` | The training pipeline |

Most core modules have a `python3 <module>.py` self-test at the
bottom — run them directly to sanity-check that module in isolation
(no hardware required for most; they run in dry-run/stub mode).
