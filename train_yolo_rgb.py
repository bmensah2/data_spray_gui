#!/usr/bin/env python3
"""
train_yolo_rgb.py
ABEN Dual RGB Detection System — YOLOv8n-Seg Training

Trains a YOLOv8n instance segmentation model on standard 3-channel BGR
JPEGs captured by the dual eMeet camera system.

Differences from train_yolo_4ch.py
------------------------------------
  - Standard 3-channel BGR input — no custom architecture YAML needed
  - Uses pretrained COCO weights (yolov8n-seg.pt) — much faster convergence
    vs 4-channel training from scratch
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
    # Simplest — reads everything from data_rgb.yaml
    python train_yolo_rgb.py \\
        --dataset /path/to/yolo_dataset/data_rgb.yaml

    # Full options
    python train_yolo_rgb.py \\
        --dataset /media/pagsun/Transcend/phd_project/emeet_dual_cam/training_data/yolo_dataset/data_rgb.yaml \\
        --epochs  100 \\
        --imgsz   640 \\
        --batch   8 \\
        --device  0 \\
        --project runs/aben_rgb \\
        --name    weed_rgb_v1

    # CPU-only (for testing)
    python train_yolo_rgb.py \\
        --dataset /path/to/data_rgb.yaml \\
        --device  cpu --batch 4 --epochs 10

Output
------
    runs/aben_rgb/weed_rgb_v1/weights/best.pt   ← use this for detection
    runs/aben_rgb/weed_rgb_v1/weights/last.pt

After training:
    1. Copy best.pt to:
       /media/pagsun/Transcend/phd_project/emeet_dual_cam/models/weed_rgb.pt
    2. Run evaluate_model_rgb.py to measure mAP / mask AP
    3. The GUI will load it automatically on ARM

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

# Pretrained nano segmentation model from Ultralytics
# Downloaded automatically on first run (~6MB)
PRETRAINED_WEIGHTS = "yolov8n-seg.pt"

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
        "--weights", default=PRETRAINED_WEIGHTS,
        help=f"Pretrained weights to start from (default: {PRETRAINED_WEIGHTS}). "
             "Use 'yolov8n-seg.pt' for nano, 'yolov8s-seg.pt' for small. "
             "Pass '' to train from scratch."
    )
    ap.add_argument("--epochs",  type=int,   default=100)
    ap.add_argument("--imgsz",   type=int,   default=640)
    ap.add_argument("--batch",   type=int,   default=8)
    ap.add_argument("--device",  default="0",
                    help="'0' for cuda:0 (Jetson), 'cpu' for CPU")
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--name",    default=DEFAULT_NAME)
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
    args = ap.parse_args()

    dataset_yaml = Path(args.dataset).resolve()

    print("=" * 60)
    print("ABEN Dual RGB — YOLOv8n-Seg Training")
    print("=" * 60)
    print(f"\nDataset YAML : {dataset_yaml}")

    # ── Validate dataset ───────────────────────────────────────
    print("\nValidating dataset ...")
    try:
        cfg = validate_dataset(dataset_yaml)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n✗ Dataset error: {e}", file=sys.stderr)
        sys.exit(1)

    # Count training images for augmentation scaling
    base      = Path(cfg.get("path", dataset_yaml.parent))
    train_dir = base / cfg.get("train", "images/train")
    n_train   = len(list(train_dir.glob("*.jpg")) +
                    list(train_dir.glob("*.jpeg")) +
                    list(train_dir.glob("*.png")))

    # ── Model ─────────────────────────────────────────────────
    weights = "" if args.no_pretrained else args.weights
    if weights:
        print(f"\nWeights      : {weights} (pretrained COCO)")
    else:
        print(f"\nWeights      : none (training from scratch)")

    model = YOLO(weights if weights else "yolov8n-seg.yaml")

    # ── Training args ─────────────────────────────────────────
    aug_args = get_augmentation_args(n_train)

    train_args = dict(
        data      = str(dataset_yaml),
        task      = "segment",
        epochs    = args.epochs,
        imgsz     = args.imgsz,
        batch     = args.batch,
        device    = args.device,
        project   = args.project,
        name      = args.name,
        patience  = args.patience,
        lr0       = args.lr0,
        freeze    = args.freeze if args.freeze > 0 else None,
        save      = True,
        plots     = True,
        verbose   = True,
        **aug_args,
    )

    print(f"\nTraining config:")
    print(f"  Epochs   : {args.epochs}  (patience={args.patience})")
    print(f"  Image sz : {args.imgsz}×{args.imgsz}")
    print(f"  Batch    : {args.batch}")
    print(f"  Device   : {args.device}")
    print(f"  Augment  : {'light' if n_train < 50 else 'medium' if n_train < 200 else 'full'} "
          f"({n_train} training images)")
    if args.freeze:
        print(f"  Freeze   : {args.freeze} backbone layers")
    print()

    # ── Train ─────────────────────────────────────────────────
    results = model.train(**train_args)

    # ── Post-training ─────────────────────────────────────────
    save_dir = Path(model.trainer.save_dir)
    best_pt  = save_dir / "weights" / "best.pt"
    last_pt  = save_dir / "weights" / "last.pt"

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best weights : {best_pt}")
    print(f"  Last weights : {last_pt}")
    print(f"  Run dir      : {save_dir}")

    # Auto-copy best.pt to project models/ folder
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if best_pt.exists():
        dest = models_dir / "weed_rgb.pt"
        shutil.copy2(best_pt, dest)
        print(f"\n✓ Copied best.pt → {dest}")
        print(f"  GUI will load this automatically on ARM")
    else:
        print(f"\n⚠ best.pt not found — check training logs")

    # Save training summary JSON
    summary = {
        "dataset_yaml":  str(dataset_yaml),
        "weights_used":  weights or "scratch",
        "epochs":        args.epochs,
        "imgsz":         args.imgsz,
        "n_train":       n_train,
        "best_pt":       str(best_pt),
        "model_dest":    str(models_dir / "weed_rgb.pt"),
        "classes":       cfg.get("names", []),
        "nc":            cfg.get("nc", 0),
    }
    summary_path = save_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nNext steps:")
    print(f"  1. Evaluate: python evaluate_model_rgb.py \\")
    print(f"                 --weights {dest} \\")
    print(f"                 --dataset {dataset_yaml}")
    print(f"  2. Launch GUI and ARM detection — model loads automatically")
    print(f"  3. Test purge and live detection with plants in view")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()