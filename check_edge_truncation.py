#!/usr/bin/env python3
"""
check_edge_truncation.py
ABEN Dual RGB Detection System — Per-class Edge-Truncation Diagnostic

Checks whether one class's training annotations are disproportionately
cut off at the image edge compared to others -- a common, purely
structural reason a class detects worse than the rest, independent of
label quality or how much training data it has.

Motivating hypothesis (operator's field layout): sugarbeet is planted
in rows ~22in apart; weeds are mostly inter-row, with nozzles (and
likely camera framing) positioned/tuned to cover that inter-row space.
If so, sugarbeet plants would tend to sit nearer the frame edges than
inter-row weeds, which are more likely centered -- and a partially
cut-off object is intrinsically harder for a detector to learn well,
independent of label quality or how much training data exists for it.

This reads your existing YOLO segmentation label files directly (no
image inspection needed) -- a polygon vertex sitting at or very near
0.0 or 1.0 on either normalized axis is a strong signal that object
was cut off by the frame boundary, not that it naturally ends there.

Usage
-----
    python3 check_edge_truncation.py --dataset /path/to/yolo_dataset

    # Looser/tighter edge threshold (default 0.02 = ~2% of frame)
    python3 check_edge_truncation.py --dataset /path/to/yolo_dataset --edge-threshold 0.03

    # Only one split
    python3 check_edge_truncation.py --dataset /path/to/yolo_dataset --split train

Output: a per-class table of annotation count, edge-touch count and
rate, and mean normalized bounding-box area (smaller = typically
either a genuinely small/young plant, or a partially-cut-off one --
read alongside the edge-touch rate, not alone).

Author : Nana | NDSU / PhD Imaging System
Path   : /media/pagsun/Transcend/phd_project/emeet_dual_cam/
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML required: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


def load_class_names(dataset_dir: Path) -> list:
    yaml_path = dataset_dir / "data_rgb.yaml"
    if not yaml_path.exists():
        # Fall back to any *.yaml in the dataset dir
        candidates = list(dataset_dir.glob("*.yaml"))
        if not candidates:
            raise FileNotFoundError(
                f"No data_rgb.yaml (or any .yaml) found in {dataset_dir}"
            )
        yaml_path = candidates[0]
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("names", [])


def analyze_split(labels_dir: Path, class_names: list, edge_threshold: float) -> dict:
    """
    Returns {class_name: {"count": int, "edge_touch": int,
                           "sum_area": float}}
    """
    stats = defaultdict(lambda: {"count": 0, "edge_touch": 0, "sum_area": 0.0})

    label_files = sorted(labels_dir.glob("*.txt"))
    for txt in label_files:
        try:
            lines = txt.read_text(encoding="utf-8").strip().splitlines()
        except Exception:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 7:  # need at least class_id + 3 (x,y) pairs
                continue
            try:
                cid = int(parts[0])
                coords = [float(v) for v in parts[1:]]
            except ValueError:
                continue

            xs = coords[0::2]
            ys = coords[1::2]
            if not xs or not ys:
                continue

            name = (class_names[cid] if cid < len(class_names)
                    else f"class_{cid}")

            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            touches_edge = (
                min_x <= edge_threshold or max_x >= 1.0 - edge_threshold or
                min_y <= edge_threshold or max_y >= 1.0 - edge_threshold
            )

            area = (max_x - min_x) * (max_y - min_y)

            stats[name]["count"] += 1
            stats[name]["sum_area"] += area
            if touches_edge:
                stats[name]["edge_touch"] += 1

    return dict(stats)


def merge_stats(a: dict, b: dict) -> dict:
    merged = defaultdict(lambda: {"count": 0, "edge_touch": 0, "sum_area": 0.0})
    for src in (a, b):
        for name, s in src.items():
            merged[name]["count"] += s["count"]
            merged[name]["edge_touch"] += s["edge_touch"]
            merged[name]["sum_area"] += s["sum_area"]
    return dict(merged)


def print_table(stats: dict, title: str):
    print(f"\n{title}")
    print("-" * 78)
    header = f"{'Class':<22} {'Count':>8} {'Edge-touch':>12} {'Edge %':>8} {'Mean bbox area':>16}"
    print(header)
    print("-" * 78)

    rows = []
    for name, s in stats.items():
        if s["count"] == 0:
            continue
        pct = 100.0 * s["edge_touch"] / s["count"]
        mean_area = s["sum_area"] / s["count"]
        rows.append((name, s["count"], s["edge_touch"], pct, mean_area))

    # Sort by edge-touch rate descending -- the classes most likely
    # suffering from this effect float to the top.
    rows.sort(key=lambda r: -r[3])

    for name, count, edge_touch, pct, mean_area in rows:
        flag = "  ⚠ notably high" if pct > 30 else ""
        print(f"{name:<22} {count:>8} {edge_touch:>12} {pct:>7.1f}% "
              f"{mean_area:>16.4f}{flag}")
    print("-" * 78)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                     help="Path to the yolo_dataset directory (containing "
                          "images/, labels/, data_rgb.yaml)")
    ap.add_argument("--edge-threshold", type=float, default=0.02,
                     help="Normalized distance from 0.0/1.0 that counts as "
                          "'touching the edge' (default: 0.02, i.e. ~2%% "
                          "of frame width/height)")
    ap.add_argument("--split", choices=["train", "val", "both"], default="both")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset).resolve()
    class_names = load_class_names(dataset_dir)
    print(f"Dataset: {dataset_dir}")
    print(f"Classes: {class_names}")
    print(f"Edge threshold: {args.edge_threshold} "
          f"({args.edge_threshold*100:.0f}% of frame)")

    splits_to_run = ["train", "val"] if args.split == "both" else [args.split]
    all_stats = {}
    combined = {}

    for split in splits_to_run:
        labels_dir = dataset_dir / "labels" / split
        if not labels_dir.exists():
            print(f"\n⚠ {labels_dir} not found, skipping {split}")
            continue
        stats = analyze_split(labels_dir, class_names, args.edge_threshold)
        all_stats[split] = stats
        print_table(stats, f"{split.upper()} split")
        combined = merge_stats(combined, stats)

    if len(splits_to_run) > 1 and combined:
        print_table(combined, "COMBINED (train + val)")

        # Explicit sugarbeet-vs-weeds comparison, since that's the
        # actual question motivating this script.
        if "sugarbeet" in combined:
            sb = combined["sugarbeet"]
            sb_pct = 100.0 * sb["edge_touch"] / sb["count"] if sb["count"] else 0
            weed_pcts = []
            for name, s in combined.items():
                if name != "sugarbeet" and s["count"] > 0:
                    weed_pcts.append(100.0 * s["edge_touch"] / s["count"])
            if weed_pcts:
                avg_weed_pct = sum(weed_pcts) / len(weed_pcts)
                print(f"\nsugarbeet edge-touch rate : {sb_pct:.1f}%")
                print(f"average weed edge-touch rate : {avg_weed_pct:.1f}%")
                if sb_pct > avg_weed_pct * 1.5 and sb_pct > 15:
                    print(f"\n→ sugarbeet IS disproportionately edge-truncated "
                          f"vs the weed classes ({sb_pct:.1f}% vs "
                          f"{avg_weed_pct:.1f}%). This supports the "
                          f"row-vs-interrow framing hypothesis as a real "
                          f"contributor to its lower recall.")
                elif sb_pct > 15:
                    print(f"\n→ sugarbeet has a meaningful edge-touch rate "
                          f"({sb_pct:.1f}%) but it's not clearly worse than "
                          f"the weed classes' average ({avg_weed_pct:.1f}%) "
                          f"-- edge truncation alone may not fully explain "
                          f"the recall gap; worth checking label "
                          f"quality/consistency too.")
                else:
                    print(f"\n→ sugarbeet's edge-touch rate ({sb_pct:.1f}%) "
                          f"is low -- edge truncation is probably NOT the "
                          f"main driver of its lower recall. Look at label "
                          f"quality, growth-stage variability, or "
                          f"occlusion by weeds instead.")


if __name__ == "__main__":
    main()
