#!/usr/bin/env python3
"""
evaluate_model_rgb.py
Dual RGB Detection System — YOLO Model Validation Metrics

RGB fork of evaluate_model.py.

Computes bounding-box AP and segmentation mask AP against the labeled
validation set produced by labelme_converter_rgb.py.
Output JSON feeds into generate_report.js for the final Word report

Usage
-----
    python evaluate_model_rgb.py \\
        --weights /media/pagsun/Transcend/phd_project/emeet_dual_cam/models/weed_rgb.pt \\
        --dataset /path/to/yolo_dataset/data_rgb.yaml \\
        --out     model_validation_rgb.json

Author : Bright Mensah | NDSU / Dual RGB Detection System
Path   : /media/pagsun/Transcend/phd_project/emeet_dual_cam/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics not installed. Run: pip install ultralytics",
          file=sys.stderr)
    sys.exit(1)


def evaluate(
    weights:      str,
    dataset_yaml: str,
    device:       str = "0",
    imgsz:        int = 640,
) -> dict:
    """
    Run YOLO validation and return a metrics dict.

    Returns the same structure as the multispec evaluate_model.py
    so generate_report.js can consume both without changes.
    """
    import yaml as _yaml

    model   = YOLO(weights)
    results = model.val(
        data    = dataset_yaml,
        device  = device,
        imgsz   = imgsz,
        verbose = False,
        task    = "segment",
    )

    # Count validation images
    num_val_images = None
    try:
        with open(dataset_yaml) as f:
            ds_cfg = _yaml.safe_load(f)
        base    = Path(ds_cfg.get("path", Path(dataset_yaml).parent))
        val_rel = ds_cfg.get("val", "images/val")
        val_dir = base / val_rel
        if val_dir.exists():
            num_val_images = sum(
                1 for p in val_dir.iterdir()
                if p.suffix.lower() in
                   (".jpg", ".jpeg", ".png", ".bmp")
            )
    except Exception:
        pass

    names     = results.names
    class_ids = sorted(names.keys())

    # ── Bounding-box metrics ──────────────────────────────────
    box = results.box
    per_class_box = []
    for i, cid in enumerate(class_ids):
        per_class_box.append({
            "class_id":   cid,
            "class_name": names[cid],
            "precision":  float(box.p[i])    if i < len(box.p)    else None,
            "recall":     float(box.r[i])    if i < len(box.r)    else None,
            "map50_95":   float(box.maps[i]) if i < len(box.maps) else None,
        })

    # ── Segmentation mask metrics ─────────────────────────────
    seg_overall   = None
    per_class_seg = []
    seg = getattr(results, "seg", None)
    if seg is not None:
        seg_overall = {
            "mask_mAP50":     float(seg.map50),
            "mask_mAP50_95":  float(seg.map),
            "mask_precision": float(seg.mp),
            "mask_recall":    float(seg.mr),
        }
        for i, cid in enumerate(class_ids):
            per_class_seg.append({
                "class_id":       cid,
                "class_name":     names[cid],
                "mask_precision": float(seg.p[i])    if i < len(seg.p)    else None,
                "mask_recall":    float(seg.r[i])    if i < len(seg.r)    else None,
                "mask_map50_95":  float(seg.maps[i]) if i < len(seg.maps) else None,
            })

    # ── Confusion matrix ──────────────────────────────────────
    confusion_matrix = None
    cm = getattr(results, "confusion_matrix", None)
    if cm is not None:
        confusion_matrix = {
            "matrix": cm.matrix.tolist(),
            "labels": [names[c] for c in class_ids] + ["background"],
        }

    return {
        "weights":          str(weights),
        "dataset":          str(dataset_yaml),
        "camera_system":    "eMeet C960 4K (Dual RGB)",
        "input_channels":   3,
        "input_size":       imgsz,
        "num_val_images":   num_val_images,
        "overall": {
            "mAP50":          float(box.map50),
            "mAP50_95":       float(box.map),
            "mean_precision": float(box.mp),
            "mean_recall":    float(box.mr),
        },
        "per_class":        per_class_box,
        "seg_overall":      seg_overall,
        "per_class_seg":    per_class_seg,
        "confusion_matrix": confusion_matrix,
        "speed_ms":         dict(results.speed),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--weights", required=True,
        help="Path to trained .pt weights (from train_yolo_rgb.py)"
    )
    ap.add_argument(
        "--dataset", required=True,
        help="Path to data_rgb.yaml"
    )
    ap.add_argument(
        "--device", default="0",
        help="'0' for cuda:0, 'cpu' for CPU (default: 0)"
    )
    ap.add_argument(
        "--imgsz", type=int, default=640,
        help="Inference image size — must match training imgsz (default: 640)"
    )
    ap.add_argument(
        "--out", default=None,
        help="Output JSON path. Default: <run_dir>/model_validation_rgb.json "
             "-- i.e. next to the weights being evaluated (in the same "
             "directory as training_summary.json), not the current "
             "working directory. Pass an explicit path to override."
    )
    args = ap.parse_args()

    if not Path(args.weights).exists():
        print(f"✗ Weights not found: {args.weights}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.dataset).exists():
        print(f"✗ Dataset YAML not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    if args.out is None:
        weights_path = Path(args.weights).resolve()
        # Standard train_yolo_rgb.py layout is <run_dir>/weights/best.pt --
        # put the report in <run_dir> itself, alongside training_summary.json.
        # If weights don't follow that layout, fall back to their own folder
        # rather than guessing further.
        run_dir = (weights_path.parent.parent
                   if weights_path.parent.name == "weights"
                   else weights_path.parent)
        out_path = run_dir / "model_validation_rgb.json"
    else:
        out_path = Path(args.out)

    print(f"Evaluating: {args.weights}")
    print(f"Dataset   : {args.dataset}")
    print(f"Device    : {args.device}  |  imgsz: {args.imgsz}")
    print()

    metrics = evaluate(args.weights, args.dataset, args.device, args.imgsz)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Console summary ───────────────────────────────────────
    ov = metrics["overall"]
    print(f"\n{'='*55}")
    print(f"  RGB MODEL VALIDATION — {Path(args.weights).name}")
    print(f"{'='*55}")
    print(f"\nBounding-box metrics:")
    print(f"  mAP50          : {ov['mAP50']:.3f}")
    print(f"  mAP50-95       : {ov['mAP50_95']:.3f}")
    print(f"  Mean precision : {ov['mean_precision']:.3f}")
    print(f"  Mean recall    : {ov['mean_recall']:.3f}")

    if metrics.get("per_class"):
        print(f"\nPer-class (box):")
        for c in metrics["per_class"]:
            p  = c['precision'] or 0
            r  = c['recall']    or 0
            m  = c['map50_95']  or 0
            print(f"  {c['class_name']:<20} P={p:.3f}  R={r:.3f}  mAP={m:.3f}")

    if metrics.get("seg_overall"):
        seg = metrics["seg_overall"]
        print(f"\nSegmentation mask metrics:")
        print(f"  Mask mAP50     : {seg['mask_mAP50']:.3f}")
        print(f"  Mask mAP50-95  : {seg['mask_mAP50_95']:.3f}")
        print(f"  Mask precision : {seg['mask_precision']:.3f}")
        print(f"  Mask recall    : {seg['mask_recall']:.3f}")

    if metrics.get("per_class_seg"):
        print(f"\nPer-class (mask):")
        for c in metrics["per_class_seg"]:
            p = c['mask_precision'] or 0
            r = c['mask_recall']    or 0
            m = c['mask_map50_95']  or 0
            print(f"  {c['class_name']:<20} P={p:.3f}  R={r:.3f}  mAP={m:.3f}")

    print(f"\nSpeed: {metrics.get('speed_ms', {})}")
    print(f"\nResults written → {out_path}")
    print(f"{'='*55}")
    print(f"\nNext: feed {out_path} into generate_report.js")


if __name__ == "__main__":
    main()