# ABEN Dual RGB Field Imaging & Spray System

A precision weed-detection and spray robot: a **Clearpath Husky A200**
ground platform carrying a dual eMeet RGB camera rig, a **Jetson AGX
Orin** running YOLO-based weed detection, and an Arduino-driven
3-nozzle solenoid sprayer. A PyQt5 operator console (this repo) runs
on the Jetson, giving a field operator live camera feeds, spray
control, navigation, and a session analysis dashboard from one
5-tab interface.

Built for herbicide-resistance and disease (CLS) research at
NDSU — every spray event is logged with class, confidence, GPS, and
robot pose for downstream analysis.

> **This system controls physical hardware that sprays chemical and
> drives a robot.** Read [`docs/SAFETY.md`](docs/SAFETY.md) before
> operating it in the field, even if you're already familiar with the
> GUI.

---

## What's here

| Piece | What it does |
|---|---|
| **`main_gui_rgb.py`** | The operator console — 5-tab PyQt5 GUI, the main entry point |
| **`core/`** | Detection engine, zone→nozzle mapping, actuation control, event logging, ROS bridge to the Husky |
| **`gui/`** | All GUI panels, tabs, and the 7-theme theming system |
| **`gantry_detection/`** | Arduino firmware for the nozzle/pump/light controller |
| **`navigation/`** | Scripts that run on the Husky's own onboard PC |
| **`tools/`** | Standalone field utilities (image capture, spray demo) |
| **`train_yolo_rgb.py`**, **`compare_yolo_models_rgb.py`**, **`evaluate_model_rgb.py`**, **`prepare_rgb_dataset.py`**, **`check_yolo_env.py`** | The full YOLO training pipeline — see [Training a detection model](#training-a-detection-model) below |

Full details on every tab, every safety system, and the training
workflow are in **[`docs/USER_GUIDE.md`](docs/USER_GUIDE.md)**.

---

## Hardware this expects

- **Jetson AGX Orin** (this GUI runs here)
- **Clearpath Husky A200**, reachable at `192.168.131.1`, running its
  own onboard PC with the scripts in `navigation/`
- **Dual eMeet SmartCam C960 4K** cameras (left/right, mounted on a
  fixed gantry arm)
- **Arduino** running `gantry_detection/gantry_detection.ino`, driving:
  - 3× solenoid nozzle valves (TeeJet-style)
  - 1× SHURflo spray pump
  - 1× AL295W grow light
  - Camera tilt servo
- USB serial connection Jetson ↔ Arduino (`/dev/ttyACM0` by default)

## Software requirements

```bash
pip install -r requirements.txt --break-system-packages
```

Covers the Jetson-side Python stack (PyQt5, OpenCV, NumPy, pyserial,
tifffile, PyYAML, ultralytics). The Husky-side ROS scripts in
`navigation/` need a ROS Noetic environment instead — see
[`navigation/README.md`](navigation/README.md).

`generate_report_rgb.js` needs Node dependencies:
```bash
npm install
```

Before training any model, run the environment check —
Jetson's PyTorch/CUDA setup has a well-known gotcha (see
[Training a detection model](#training-a-detection-model)):
```bash
python3 check_yolo_env.py
```

## Quick start

```bash
cd /media/pagsun/Transcend/phd_project/emeet_dual_cam
python3 main_gui_rgb.py
```

The app opens on **Data Collection**. Connect the Arduino from the
header bar, connect the cameras, then switch to **Detection** to arm
weed detection and spraying. See
**[`docs/USER_GUIDE.md`](docs/USER_GUIDE.md)** for a full tab-by-tab
walkthrough — it's worth reading before your first real session,
especially the parts on arming/disarming and the cross-tab motion
lock.

## Training a detection model

The system supports **YOLOv8, YOLO11, and YOLO26** through the same
pipeline — the runtime detection engine auto-detects whichever
architecture trained the `.pt` file, so nothing else needs to know
which family you used.

```bash
# 1. Confirm your environment is actually ready (especially CUDA —
#    a plain `pip install torch` on a Jetson silently falls back to
#    CPU-only training, which can turn a 2-hour run into a 2-day one)
python3 check_yolo_env.py

# 2. Validate your labeled dataset (YOLO format: images/{train,val},
#    labels/{train,val})
python3 prepare_rgb_dataset.py --dataset /path/to/yolo_dataset \
    --classes sugarbeet kochia waterhemp common_ragweed common_lambsquaters unknown_weed

# 3. Train one model
python3 train_yolo_rgb.py --dataset /path/to/yolo_dataset/data_rgb.yaml --model yolo26n

# ...or train and compare all three on the same dataset
python3 compare_yolo_models_rgb.py --dataset /path/to/yolo_dataset/data_rgb.yaml --epochs 100
```

`compare_yolo_models_rgb.py` reports mAP, mask AP, and inference
speed side by side and deliberately does **not** auto-pick a winner —
on a real-time spray robot, a slower but marginally more accurate
model isn't necessarily the better choice. See
[`docs/USER_GUIDE.md`](docs/USER_GUIDE.md#yolo-model-training--deployment)
for the full workflow, including how to deploy the model you pick.

## Theming

`View > Theme` in the menu bar switches between 7 themes live. See
[`docs/USER_GUIDE.md`](docs/USER_GUIDE.md#theming) for the list.

## Documentation

- **[`docs/USER_GUIDE.md`](docs/USER_GUIDE.md)** — the full guide:
  every tab, every safety system, training, theming, troubleshooting
- **[`docs/SAFETY.md`](docs/SAFETY.md)** — safety systems reference,
  read this first
- [`navigation/README.md`](navigation/README.md) — Husky-side scripts
- [`tools/README.md`](tools/README.md) — standalone field utilities

## Author

Nana | NDSU / PhD Imaging System
