#!/usr/bin/env python3
"""
train_yolo_rgb.py
ABEN Dual RGB Detection System — YOLO Segmentation Training

Trains a YOLO instance segmentation model on standard 3-channel BGR
JPEGs captured by the dual eMeet camera system. Supports any
Ultralytics YOLO segmentation family through the same pipeline --
YOLOv8, YOLO11, and YOLO26 are all just different starting checkpoints;
nothing else about training, evaluation, or the runtime inference
engine (core/detection_engine_rgb.py) cares which family produced a
given .pt file.

Differences from train_yolo_4ch.py
------------------------------------
  - Standard 3-channel BGR input — no custom architecture YAML needed
  - Uses pretrained COCO weights (yolov8n-seg.pt / yolo11n-seg.pt /
    yolo26n-seg.pt) — much faster convergence vs 4-channel training
    from scratch
  - No prepare step needed — labelme_converter_rgb.py already produces
    the correct images/train|val + labels/train|val structure
  - Input size: 640×640 (YOLO default) vs 512×512 for multispec
  - data_rgb.yaml already written by labelme_converter_rgb.py

Dataset structure expected (from labelme_converter_rgb.py export):
    yolo_dataset/
    ├── images/
    │   ├── train/*.jpg
    │   └── val/*.jpg
    ├── labels/
    │   ├── train/*.txt   ← YOLO segment polygon format
    │   └── val/*.txt
    └── data_rgb.yaml

Usage
-----
    # Simplest — reads everything from data_rgb.yaml, trains YOLOv8n
    python train_yolo_rgb.py \\
        --dataset /path/to/yolo_dataset/data_rgb.yaml

    # Pick a model family with --model (auto-sets weights + run name)
    python train_yolo_rgb.py \\
        --dataset /path/to/yolo_dataset/data_rgb.yaml --model yolo11n

    python train_yolo_rgb.py \\
        --dataset /path/to/yolo_dataset/data_rgb.yaml --model yolo26n

    # Full options
    python train_yolo_rgb.py \\
        --dataset /media/pagsun/Transcend/phd_project/emeet_dual_cam/training_data/yolo_dataset/data_rgb.yaml \\
        --model   yolo26n \\
        --epochs  100 \\
        --imgsz   640 \\
        --batch   8 \\
        --device  0 \\
        --project runs/aben_rgb \\
        --name    weed_rgb_yolo26_v1

    # CPU-only (for testing)
    python train_yolo_rgb.py \\
        --dataset /path/to/data_rgb.yaml \\
        --device  cpu --batch 4 --epochs 10

    # Comparing multiple families side-by-side? See
    # compare_yolo_models_rgb.py instead — it calls this script's
    # run_training() once per family and produces a comparison report.

Output
------
    runs/aben_rgb/<name>/weights/best.pt   ← use this for detection
    runs/aben_rgb/<name>/weights/last.pt

After training:
    1. best.pt is auto-copied to models/weed_rgb.pt (or --models-dest-name)
    2. Run evaluate_model_rgb.py to measure mAP / mask AP
    3. The GUI will load models/weed_rgb.pt automatically on ARM

Before training for the first time, run check_yolo_env.py to confirm
ultralytics/torch/CUDA are actually set up correctly -- especially on
a Jetson, where a generic pip-installed torch silently falls back to
CPU-only training instead of failing loudly.

Author : Nana | NDSU / PhD Imaging System
Path   : /media/pagsun/Transcend/phd_project/emeet_dual_cam/
"""

from __future__ import annotations

import argparse
import json
import sys
import shutil
from pathlib import Path

import yaml

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    print("ultralytics not installed. Run: pip install ultralytics",
          file=sys.stderr)
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
#  DEFAULTS
# ─────────────────────────────────────────────────────────────

# Pretrained nano segmentation model from Ultralytics — default family
# Downloaded automatically on first run (~6MB)
PRETRAINED_WEIGHTS = "yolov8n-seg.pt"

# Convenience shorthand -> actual pretrained checkpoint filename.
# All three families use the same training pipeline/dataset format --
# Ultralytics' YOLO() class auto-detects the architecture from the
# checkpoint, so nothing else in this script (or in the runtime
# inference engine, detection_engine_rgb.py) needs to know which
# family is in use. This dict only exists so `--model yolo26n` is
# easier to type/remember than `--weights yolo26n-seg.pt`.
MODEL_FAMILIES = {
    "yolov8n": "yolov8n-seg.pt",
    "yolov8s": "yolov8s-seg.pt",
    "yolo11n": "yolo11n-seg.pt",
    "yolo11s": "yolo11s-seg.pt",
    "yolo26n": "yolo26n-seg.pt",
    "yolo26s": "yolo26s-seg.pt",
}

# Output root
DEFAULT_PROJECT = "runs/aben_rgb"
DEFAULT_NAME    = "weed_rgb_v1"

# Project root on Transcend drive
PROJECT_ROOT = Path(
    "/media/pagsun/Transcend/phd_project/emeet_dual_cam"
)


# ─────────────────────────────────────────────────────────────
#  DATASET VALIDATION
# ─────────────────────────────────────────────────────────────

def validate_dataset(yaml_path: Path) -> dict:
    """
    Load and validate data_rgb.yaml.
    Returns the parsed config dict.
    Raises ValueError if the dataset is not usable.
    """
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {yaml_path}\n"
            f"Run the ABEN Annotator → Export YOLO Dataset first."
        )

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    base = Path(cfg.get("path", yaml_path.parent))

    # Check train and val splits exist
    for split in ("train", "val"):
        split_dir = base / cfg.get(split, f"images/{split}")
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}\n"
                f"Expected layout: images/train/*.jpg  images/val/*.jpg"
            )
        images = list(split_dir.glob("*.jpg")) + \
                 list(split_dir.glob("*.jpeg")) + \
                 list(split_dir.glob("*.png"))
        if not images:
            raise ValueError(
                f"No images found in {split_dir}\n"
                f"Run the annotator export first."
            )
        print(f"  {split:5s}: {len(images)} images")

    names = cfg.get("names", [])
    nc    = cfg.get("nc",    len(names))
    print(f"  Classes ({nc}): {names}")

    # Warn if 4ch channels key is present — wrong yaml for RGB training
    if cfg.get("channels", 3) == 4:
        print(
            "\n  ⚠ WARNING: data_rgb.yaml has 'channels: 4' — "
            "this looks like a multispectral config.\n"
            "  Use data_rgb.yaml (written by labelme_converter_rgb.py), "
            "not data_4ch.yaml."
        )

    return cfg


# ─────────────────────────────────────────────────────────────
#  AUGMENTATION SETTINGS
# ─────────────────────────────────────────────────────────────

def get_augmentation_args(n_images: int) -> dict:
    """
    Return augmentation parameters scaled to dataset size.
    With small datasets (test/prototype), lighter augmentation avoids
    overfitting on augmented copies.

    n_images : total training images across all classes
    """
    if n_images < 50:
        # Very small dataset — light augmentation
        return dict(
            hsv_h=0.015,   # hue shift ±1.5%
            hsv_s=0.4,     # saturation ±40%
            hsv_v=0.3,     # value ±30%
            degrees=10.0,  # rotation ±10°
            translate=0.1,
            scale=0.3,
            flipud=0.0,
            fliplr=0.5,
            mosaic=0.5,    # mosaic 50% — helps with small plants
            mixup=0.0,
        )
    elif n_images < 200:
        # Medium dataset
        return dict(
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=15.0,
            translate=0.1,
            scale=0.5,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
        )
    else:
        # Full augmentation for large datasets
        return dict(
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=20.0,
            translate=0.1,
            scale=0.5,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.15,
        )


# ─────────────────────────────────────────────────────────────
#  CORE TRAINING (importable — used by both this script's CLI
#  and compare_yolo_models_rgb.py)
# ─────────────────────────────────────────────────────────────

def run_training(
    dataset_yaml: Path,
    weights:      str = PRETRAINED_WEIGHTS,
    epochs:       int = 100,
    imgsz:        int = 640,
    batch:        int = 8,
    device:       str = "0",
    project:      str = DEFAULT_PROJECT,
    name:         str = DEFAULT_NAME,
    patience:     int = 30,
    lr0:          float = 0.01,
    freeze:       int = 0,
    no_pretrained: bool = False,
    copy_to_models_dir: bool = True,
    models_dest_name: str = "weed_rgb.pt",
) -> dict:
    """
    Run one full training job and return a summary dict. This is the
    function both `python train_yolo_rgb.py ...` and
    compare_yolo_models_rgb.py call -- keeping the actual training
    logic in exactly one place rather than duplicated between a CLI
    script and a comparison harness.

    Raises FileNotFoundError/ValueError if the dataset is invalid
    (via validate_dataset) -- callers should catch these per-family
    when training multiple models in a loop, so one bad config
    doesn't take down the whole comparison run.
    """
    dataset_yaml = Path(dataset_yaml).resolve()
    cfg = validate_dataset(dataset_yaml)

    base      = Path(cfg.get("path", dataset_yaml.parent))
    train_dir = base / cfg.get("train", "images/train")
    n_train   = len(list(train_dir.glob("*.jpg")) +
                    list(train_dir.glob("*.jpeg")) +
                    list(train_dir.glob("*.png")))

    resolved_weights = "" if no_pretrained else weights
    model = YOLO(resolved_weights if resolved_weights else "yolov8n-seg.yaml")

    aug_args = get_augmentation_args(n_train)
    train_args = dict(
        data      = str(dataset_yaml),
        task      = "segment",
        epochs    = epochs,
        imgsz     = imgsz,
        batch     = batch,
        device    = device,
        project   = project,
        name      = name,
        patience  = patience,
        lr0       = lr0,
        freeze    = freeze if freeze > 0 else None,
        save      = True,
        plots     = True,
        verbose   = True,
        **aug_args,
    )

    results = model.train(**train_args)

    save_dir = Path(model.trainer.save_dir)
    best_pt  = save_dir / "weights" / "best.pt"
    last_pt  = save_dir / "weights" / "last.pt"

    dest = None
    if copy_to_models_dir and best_pt.exists():
        models_dir = PROJECT_ROOT / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        dest = models_dir / models_dest_name
        shutil.copy2(best_pt, dest)

    summary = {
        "dataset_yaml":  str(dataset_yaml),
        "weights_used":  resolved_weights or "scratch",
        "epochs":        epochs,
        "imgsz":         imgsz,
        "n_train":       n_train,
        "best_pt":       str(best_pt) if best_pt.exists() else None,
        "last_pt":       str(last_pt) if last_pt.exists() else None,
        "save_dir":      str(save_dir),
        "model_dest":    str(dest) if dest else None,
        "classes":       cfg.get("names", []),
        "nc":            cfg.get("nc", 0),
    }
    summary_path = save_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--dataset", required=True,
        help="Path to data_rgb.yaml (written by labelme_converter_rgb.py)"
    )
    ap.add_argument(
        "--model", choices=sorted(MODEL_FAMILIES.keys()), default=None,
        help="Convenience shorthand for --weights, e.g. 'yolo26n' -> "
             "'yolo26n-seg.pt'. If --name is not also given, the run "
             "name defaults to '<model>_rgb_v1' so training different "
             "families doesn't overwrite the same output directory. "
             "Overridden by --weights if both are given."
    )
    ap.add_argument(
        "--weights", default=None,
        help=f"Pretrained weights to start from (default: {PRETRAINED_WEIGHTS} "
             f"unless --model is given). Accepts any Ultralytics checkpoint "
             f"name, e.g. 'yolov8n-seg.pt', 'yolo11n-seg.pt', 'yolo26n-seg.pt'. "
             f"Pass '' to train from scratch."
    )
    ap.add_argument("--epochs",  type=int,   default=100)
    ap.add_argument("--imgsz",   type=int,   default=640)
    ap.add_argument("--batch",   type=int,   default=8)
    ap.add_argument("--device",  default="0",
                    help="'0' for cuda:0 (Jetson), 'cpu' for CPU")
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--name",    default=None)
    ap.add_argument(
        "--patience", type=int, default=30,
        help="Early stopping patience in epochs (default: 30)"
    )
    ap.add_argument(
        "--lr0", type=float, default=0.01,
        help="Initial learning rate (default: 0.01)"
    )
    ap.add_argument(
        "--freeze", type=int, default=0,
        help="Number of backbone layers to freeze (0=none, 10=freeze backbone). "
             "Useful when fine-tuning on a small dataset."
    )
    ap.add_argument(
        "--no-pretrained", action="store_true",
        help="Train from scratch (no COCO pretrained weights)"
    )
    ap.add_argument(
        "--models-dest-name", default="weed_rgb.pt",
        help="Filename to copy best.pt to under models/ (default: "
             "weed_rgb.pt, matching what the GUI loads on ARM). Change "
             "this if you're training a side-by-side comparison model "
             "you don't want the live GUI to pick up automatically."
    )
    args = ap.parse_args()

    dataset_yaml = Path(args.dataset).resolve()

    # Resolve --model shorthand -> --weights / --name, without
    # overriding an explicit --weights/--name if the user gave one.
    weights = args.weights
    name    = args.name
    if args.model is not None:
        if weights is None:
            weights = MODEL_FAMILIES[args.model]
        if name is None:
            name = f"{args.model}_rgb_v1"
    if weights is None:
        weights = PRETRAINED_WEIGHTS
    if name is None:
        name = DEFAULT_NAME

    print("=" * 60)
    print("ABEN Dual RGB — YOLO Segmentation Training")
    print("=" * 60)
    print(f"\nDataset YAML : {dataset_yaml}")

    # ── Validate dataset (also done inside run_training, but do it
    # here too so we can print the friendly per-split summary before
    # committing to a (possibly long) training run) ────────────────
    print("\nValidating dataset ...")
    try:
        cfg = validate_dataset(dataset_yaml)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n✗ Dataset error: {e}", file=sys.stderr)
        sys.exit(1)

    resolved_weights = "" if args.no_pretrained else weights
    if resolved_weights:
        print(f"\nWeights      : {resolved_weights} (pretrained COCO)")
    else:
        print(f"\nWeights      : none (training from scratch)")

    print(f"\nTraining config:")
    print(f"  Epochs   : {args.epochs}  (patience={args.patience})")
    print(f"  Image sz : {args.imgsz}×{args.imgsz}")
    print(f"  Batch    : {args.batch}")
    print(f"  Device   : {args.device}")
    print(f"  Run name : {name}")
    if args.freeze:
        print(f"  Freeze   : {args.freeze} backbone layers")
    print()

    # ── Train ─────────────────────────────────────────────────
    summary = run_training(
        dataset_yaml   = dataset_yaml,
        weights        = weights,
        epochs         = args.epochs,
        imgsz          = args.imgsz,
        batch          = args.batch,
        device         = args.device,
        project        = args.project,
        name           = name,
        patience       = args.patience,
        lr0            = args.lr0,
        freeze         = args.freeze,
        no_pretrained  = args.no_pretrained,
        models_dest_name = args.models_dest_name,
    )

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best weights : {summary['best_pt']}")
    print(f"  Last weights : {summary['last_pt']}")
    print(f"  Run dir      : {summary['save_dir']}")

    if summary["model_dest"]:
        print(f"\n✓ Copied best.pt → {summary['model_dest']}")
        print(f"  GUI will load this automatically on ARM "
              f"(if named weed_rgb.pt or cls_rgb.pt)")
    else:
        print(f"\n⚠ best.pt not found or not copied — check training logs")

    print(f"\nNext steps:")
    print(f"  1. Evaluate: python evaluate_model_rgb.py \\")
    print(f"                 --weights {summary['best_pt']} \\")
    print(f"                 --dataset {dataset_yaml}")
    print(f"  2. Launch GUI and ARM detection — model loads automatically")
    print(f"  3. Test purge and live detection with plants in view")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()