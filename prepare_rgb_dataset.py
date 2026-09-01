#!/usr/bin/env python3
"""
prepare_rgb_dataset.py
ABEN Dual RGB Detection System — Dataset Preparation

Validates and prepares the RGB dataset exported by the ABEN Annotator
for YOLOv8 training. Unlike prepare_4ch_dataset.py, almost nothing needs
to happen here — the Annotator already exports standard YOLO-format JPEGs
and polygon .txt labels.

What this script does:
  1. Validates the dataset structure (images + labels present and paired)
  2. Writes (or rewrites) data_rgb.yaml with correct paths
  3. Prints a dataset summary (class counts, split sizes)
  4. Optionally resizes images to a target size (default: keep as-is)

What it does NOT do (unlike prepare_4ch_dataset.py):
  - No band extraction (already RGB)
  - No .npy stack building (not needed for 3ch)
  - No architecture YAML (standard yolov8n-seg.yaml works)

Usage
-----
    # Basic — just validate and write YAML
    python prepare_rgb_dataset.py \\
        --dataset /path/to/yolo_dataset \\
        --classes sugarbeet weed

    # Full path as used in this project
    python prepare_rgb_dataset.py \\
        --dataset /media/pagsun/Transcend/phd_project/emeet_dual_cam/training_data/yolo_dataset \\
        --classes sugarbeet weed

    # If labelme_converter_rgb already wrote data_rgb.yaml, just validate:
    python prepare_rgb_dataset.py \\
        --dataset /path/to/yolo_dataset \\
        --validate-only

Author : Nana | NDSU / PhD Imaging System
Path   : /media/pagsun/Transcend/phd_project/emeet_dual_cam/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml


# ─────────────────────────────────────────────────────────────
#  VALIDATION
# ─────────────────────────────────────────────────────────────

def validate_split(
    images_dir: Path,
    labels_dir: Path,
    split:      str,
    class_names: list,
) -> dict:
    """
    Check one split for image/label pairing and count annotations.
    Returns a stats dict.
    """
    img_exts  = {".jpg", ".jpeg", ".png"}
    images    = {p.stem: p for p in images_dir.iterdir()
                 if p.suffix.lower() in img_exts}
    labels    = {p.stem: p for p in labels_dir.iterdir()
                 if p.suffix == ".txt"} if labels_dir.exists() else {}

    missing_labels  = [s for s in images if s not in labels]
    missing_images  = [s for s in labels if s not in images]
    paired          = [s for s in images if s in labels]

    per_class = defaultdict(int)
    total_annotations = 0

    for stem in paired:
        txt = labels[stem]
        lines = txt.read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            parts = line.split()
            if parts:
                cid = int(parts[0])
                name = (class_names[cid]
                        if cid < len(class_names) else f"class_{cid}")
                per_class[name] += 1
                total_annotations += 1

    return {
        "split":             split,
        "images":            len(images),
        "labels":            len(labels),
        "paired":            len(paired),
        "missing_labels":    missing_labels[:5],
        "missing_images":    missing_images[:5],
        "total_annotations": total_annotations,
        "per_class":         dict(per_class),
    }


# ─────────────────────────────────────────────────────────────
#  YAML WRITING
# ─────────────────────────────────────────────────────────────

def write_data_yaml(
    dataset_dir:  Path,
    class_names:  list,
    out_name:     str = "data_rgb.yaml",
) -> Path:
    """Write data_rgb.yaml for YOLOv8 RGB training."""
    cfg = {
        "# ABEN YOLO dataset — RGB (eMeet dual camera)": None,
        "path":  str(dataset_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(class_names),
        "names": class_names,
    }
    # Write manually to keep comment at top
    lines = [
        "# ABEN YOLO dataset — RGB (eMeet dual camera)",
        f"path:  {dataset_dir.resolve()}",
        "train: images/train",
        "val:   images/val",
        f"nc:    {len(class_names)}",
        f"names: {class_names}",
        "",
        "# Camera: eMeet SmartCam C960 4K (Dual)",
        "# Resolution: 1920×1080  |  3-channel BGR JPEG",
        f"# Zone calibration: N1=600px  N2=1700px/400px  N3=1400px",
        "# Train with: python train_yolo_rgb.py --dataset <this file>",
    ]
    out_path = dataset_dir / out_name
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


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
        help="Path to YOLO dataset root (contains images/ and labels/)"
    )
    ap.add_argument(
        "--classes", nargs="+",
        default=["sugarbeet", "weed"],
        help="Class names in class-id order (default: sugarbeet weed)"
    )
    ap.add_argument(
        "--validate-only", action="store_true",
        help="Only validate — do not rewrite data_rgb.yaml"
    )
    ap.add_argument(
        "--out-yaml", default="data_rgb.yaml",
        help="Output YAML filename (default: data_rgb.yaml)"
    )
    args = ap.parse_args()

    dataset_dir  = Path(args.dataset)
    class_names  = args.classes

    if not dataset_dir.exists():
        print(f"✗ Dataset directory not found: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    print("=" * 55)
    print("  ABEN RGB Dataset Preparation")
    print("=" * 55)
    print(f"\nDataset : {dataset_dir}")
    print(f"Classes : {class_names}")
    print()

    # ── Validate both splits ───────────────────────────────────
    all_stats = {}
    errors    = []

    for split in ("train", "val"):
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split

        if not img_dir.exists():
            errors.append(f"images/{split}/ not found")
            continue

        stats = validate_split(img_dir, lbl_dir, split, class_names)
        all_stats[split] = stats

        print(f"  {split.upper()}:")
        print(f"    Images     : {stats['images']}")
        print(f"    Labels     : {stats['labels']}")
        print(f"    Paired     : {stats['paired']}")
        print(f"    Annotations: {stats['total_annotations']}")
        if stats["per_class"]:
            for cls, cnt in sorted(stats["per_class"].items()):
                print(f"      {cls:<20}: {cnt}")
        if stats["missing_labels"]:
            print(f"    ⚠ Missing labels: {stats['missing_labels']}")
        if stats["missing_images"]:
            print(f"    ⚠ Missing images: {stats['missing_images']}")
        print()

    if errors:
        for e in errors:
            print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Write / update YAML ────────────────────────────────────
    if not args.validate_only:
        yaml_path = write_data_yaml(dataset_dir, class_names, args.out_yaml)
        print(f"✓ Written: {yaml_path}")
    else:
        existing = dataset_dir / args.out_yaml
        if existing.exists():
            print(f"  YAML exists: {existing} (--validate-only, not rewriting)")
        else:
            print(f"  ⚠ {args.out_yaml} not found — run without --validate-only to create it")

    # ── Save stats JSON ────────────────────────────────────────
    stats_path = dataset_dir / "dataset_stats_rgb.json"
    total_ann = sum(s["total_annotations"] for s in all_stats.values())
    summary = {
        "total_frames":      sum(s["paired"] for s in all_stats.values()),
        "total_annotations": total_ann,
        "classes":           class_names,
        "splits":            all_stats,
    }
    with open(stats_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Stats  : {stats_path}")

    # ── Training command ───────────────────────────────────────
    yaml_path = dataset_dir / args.out_yaml
    print(f"\nReady to train:")
    print(f"  python train_yolo_rgb.py \\")
    print(f"    --dataset {yaml_path} \\")
    print(f"    --epochs 100 --batch 8 --device 0")
    print(f"\n{'='*55}")


if __name__ == "__main__":
    main()