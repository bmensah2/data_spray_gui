#!/usr/bin/env python3
"""
compare_yolo_models_rgb.py
ABEN Dual RGB Detection System — Multi-Family YOLO Comparison

Trains (or reuses existing) YOLOv8, YOLO11, and YOLO26 checkpoints on
the SAME dataset with the SAME hyperparameters, evaluates each with
evaluate_model_rgb.py's evaluate(), and writes a side-by-side
comparison report -- so you can actually pick a model for deployment
based on real numbers instead of guessing.

Reuses train_yolo_rgb.run_training() and evaluate_model_rgb.evaluate()
directly (no subprocess, no duplicated training logic) -- both of
those already work with any Ultralytics segmentation checkpoint, so
this script is really just a loop plus a report.

Deliberately does NOT pick a single "winner" for you: accuracy
(mAP) and inference speed trade off against each other, and which
one matters more depends on your deployment constraints (this is a
real-time spray robot, so speed is not a minor detail here). The
report shows both clearly; the report's summary section highlights
the fastest and the most accurate so you can decide with your own
priorities in mind.

If one family's training or evaluation fails (e.g. an old ultralytics
version that doesn't yet recognize yolo26n-seg.pt), that family is
recorded with its error and the comparison continues with the rest --
one bad model shouldn't block seeing results for the other two.

Usage
-----
    # Train + evaluate all three "nano" families on the same dataset
    python compare_yolo_models_rgb.py \\
        --dataset /path/to/yolo_dataset/data_rgb.yaml \\
        --epochs 100 --batch 8 --device 0

    # Compare a specific set of families/sizes
    python compare_yolo_models_rgb.py \\
        --dataset /path/to/data_rgb.yaml \\
        --models yolov8n yolo11n yolo26n yolo26s

    # Already trained all three separately? Just re-run the
    # evaluation/comparison against existing best.pt files:
    python compare_yolo_models_rgb.py \\
        --dataset /path/to/data_rgb.yaml \\
        --skip-training

    # Quick smoke test before committing to a long real run
    python compare_yolo_models_rgb.py \\
        --dataset /path/to/data_rgb.yaml \\
        --epochs 2 --device cpu

Before running for real, run check_yolo_env.py first -- this script
will train three full models back to back, which is a lot of wasted
time to discover partway through that CUDA wasn't actually being used.

Author : Nana | NDSU / PhD Imaging System
Path   : /media/pagsun/Transcend/phd_project/emeet_dual_cam/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    from train_yolo_rgb import (
        run_training, validate_dataset, MODEL_FAMILIES,
        DEFAULT_PROJECT, PROJECT_ROOT,
    )
    from evaluate_model_rgb import evaluate
except ImportError as e:
    print(f"Could not import train_yolo_rgb / evaluate_model_rgb: {e}",
          file=sys.stderr)
    print("Run this script from the same directory as those two files.",
          file=sys.stderr)
    sys.exit(1)

# The three families the operator actually asked to compare, as a
# sensible default -- --models can override with any MODEL_FAMILIES key.
DEFAULT_MODELS = ["yolov8n", "yolo11n", "yolo26n"]


# ─────────────────────────────────────────────────────────────
#  COMPARISON
# ─────────────────────────────────────────────────────────────

def find_existing_best_pt(project: str, model_key: str) -> Path | None:
    """For --skip-training: locate an already-trained best.pt for
    this family, matching the run-name convention run_training()/
    train_yolo_rgb.py's --model shorthand uses (<model>_rgb_v1)."""
    candidate = Path(project) / f"{model_key}_rgb_v1" / "weights" / "best.pt"
    return candidate if candidate.exists() else None


def compare_models(
    dataset_yaml:   str,
    model_keys:     list,
    epochs:         int   = 100,
    imgsz:          int   = 640,
    batch:          int   = 8,
    device:         str   = "0",
    project:        str   = DEFAULT_PROJECT,
    patience:       int   = 30,
    lr0:            float = 0.01,
    skip_training:  bool  = False,
) -> dict:
    """
    Train (or locate) + evaluate every requested model family.
    Returns {model_key: {"best_pt": str|None, "train_summary": dict|None,
                          "metrics": dict|None, "error": str|None,
                          "train_seconds": float|None}}
    """
    dataset_yaml = str(Path(dataset_yaml).resolve())

    # Fail fast on a bad dataset rather than partway through model 1 of 3.
    validate_dataset(Path(dataset_yaml))

    results = {}
    for model_key in model_keys:
        if model_key not in MODEL_FAMILIES:
            results[model_key] = {
                "best_pt": None, "train_summary": None, "metrics": None,
                "error": f"Unknown model family '{model_key}'. "
                         f"Valid: {sorted(MODEL_FAMILIES.keys())}",
                "train_seconds": None,
            }
            continue

        print(f"\n{'='*60}")
        print(f"  {model_key}  ({MODEL_FAMILIES[model_key]})")
        print(f"{'='*60}")

        entry = {
            "best_pt": None, "train_summary": None, "metrics": None,
            "error": None, "train_seconds": None,
        }

        try:
            if skip_training:
                best_pt = find_existing_best_pt(project, model_key)
                if best_pt is None:
                    raise FileNotFoundError(
                        f"--skip-training given but no existing best.pt "
                        f"found at {project}/{model_key}_rgb_v1/weights/"
                        f"best.pt -- train it first, or drop "
                        f"--skip-training."
                    )
                entry["best_pt"] = str(best_pt)
                print(f"Using existing weights: {best_pt}")
            else:
                t0 = time.time()
                summary = run_training(
                    dataset_yaml   = dataset_yaml,
                    weights        = MODEL_FAMILIES[model_key],
                    epochs         = epochs,
                    imgsz          = imgsz,
                    batch          = batch,
                    device         = device,
                    project        = project,
                    name           = f"{model_key}_rgb_v1",
                    patience       = patience,
                    lr0            = lr0,
                    copy_to_models_dir = False,  # don't clobber deployed
                                                   # model until a choice
                                                   # is made
                )
                entry["train_seconds"] = round(time.time() - t0, 1)
                entry["train_summary"] = summary
                if not summary.get("best_pt"):
                    raise RuntimeError(
                        "training finished but no best.pt was produced -- "
                        "check the training logs above"
                    )
                entry["best_pt"] = summary["best_pt"]

            print(f"\nEvaluating {model_key} ...")
            metrics = evaluate(
                weights      = entry["best_pt"],
                dataset_yaml = dataset_yaml,
                device       = device,
                imgsz        = imgsz,
            )
            entry["metrics"] = metrics
            ov = metrics["overall"]
            print(f"  mAP50={ov['mAP50']:.3f}  mAP50-95={ov['mAP50_95']:.3f}  "
                  f"speed={metrics['speed_ms'].get('inference', '?')}ms/img")

        except Exception as e:
            entry["error"] = str(e)
            print(f"\n✗ {model_key} failed: {e}", file=sys.stderr)

        results[model_key] = entry

    return results


def print_comparison_table(results: dict):
    print(f"\n{'='*78}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*78}")

    rows = []
    for key, entry in results.items():
        if entry["error"] or not entry["metrics"]:
            rows.append((key, None))
            continue
        ov = entry["metrics"]["overall"]
        seg = entry["metrics"].get("seg_overall") or {}
        speed = entry["metrics"].get("speed_ms", {})
        inf_ms = speed.get("inference")
        rows.append((key, {
            "mAP50":      ov["mAP50"],
            "mAP50_95":   ov["mAP50_95"],
            "mask_mAP50": seg.get("mask_mAP50"),
            "inf_ms":     inf_ms,
            "train_s":    entry.get("train_seconds"),
        }))

    header = (f"{'Model':<10} {'mAP50':>8} {'mAP50-95':>10} "
              f"{'maskAP50':>10} {'inf(ms)':>9} {'train(s)':>10}")
    print(f"\n{header}")
    print("-" * len(header))
    for key, stats in rows:
        if stats is None:
            print(f"{key:<10} {'FAILED — see error above':>49}")
            continue
        def _fmt(v, width, prec=3):
            return f"{v:>{width}.{prec}f}" if v is not None else f"{'—':>{width}}"
        print(
            f"{key:<10} "
            f"{_fmt(stats['mAP50'], 8)} "
            f"{_fmt(stats['mAP50_95'], 10)} "
            f"{_fmt(stats['mask_mAP50'], 10)} "
            f"{_fmt(stats['inf_ms'], 9, 1)} "
            f"{_fmt(stats['train_s'], 10, 0)}"
        )

    ok_rows = [(k, s) for k, s in rows if s is not None]
    if ok_rows:
        fastest = min(ok_rows, key=lambda ks: ks[1]["inf_ms"] if ks[1]["inf_ms"] is not None else float("inf"))
        most_accurate = max(ok_rows, key=lambda ks: ks[1]["mAP50_95"] if ks[1]["mAP50_95"] is not None else -1)
        print(f"\nFastest inference   : {fastest[0]}  "
              f"({fastest[1]['inf_ms']:.1f} ms/img)" if fastest[1]["inf_ms"] is not None else "")
        print(f"Highest mAP50-95    : {most_accurate[0]}  "
              f"({most_accurate[1]['mAP50_95']:.3f})")
        if fastest[0] == most_accurate[0]:
            print(f"\n→ {fastest[0]} wins on both speed and accuracy here.")
        else:
            print(f"\n→ Trade-off: {fastest[0]} is faster, "
                  f"{most_accurate[0]} is more accurate. For a real-time "
                  f"spray robot, weigh inference speed carefully against "
                  f"the accuracy gain -- a slower model that misses the "
                  f"spray window is not actually more useful in the field.")
    print(f"{'='*78}")


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dataset", required=True,
                     help="Path to data_rgb.yaml")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                     choices=sorted(MODEL_FAMILIES.keys()),
                     help=f"Model families to compare (default: {DEFAULT_MODELS})")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--skip-training", action="store_true",
                     help="Don't train -- just evaluate existing "
                          "<project>/<model>_rgb_v1/weights/best.pt "
                          "for each requested family")
    ap.add_argument("--out", default="model_comparison_rgb.json",
                     help="Output JSON report path")
    args = ap.parse_args()

    print("=" * 60)
    print("  ABEN Dual RGB — Multi-Family YOLO Comparison")
    print("=" * 60)
    print(f"\nDataset : {args.dataset}")
    print(f"Models  : {args.models}")
    print(f"Mode    : {'evaluate existing weights only' if args.skip_training else 'train + evaluate'}")

    try:
        results = compare_models(
            dataset_yaml  = args.dataset,
            model_keys    = args.models,
            epochs        = args.epochs,
            imgsz         = args.imgsz,
            batch         = args.batch,
            device        = args.device,
            project       = args.project,
            patience      = args.patience,
            lr0           = args.lr0,
            skip_training = args.skip_training,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"\n✗ Dataset error: {e}", file=sys.stderr)
        sys.exit(1)

    print_comparison_table(results)

    out_path = Path(args.out)
    with open(out_path, "w") as f:
        json.dump({
            "dataset": args.dataset,
            "models_compared": args.models,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "results": results,
        }, f, indent=2)
    print(f"\nFull report written → {out_path}")
    print(f"\nOnce you've picked a model:")
    print(f"  cp <chosen best.pt> "
          f"{PROJECT_ROOT / 'models' / 'weed_rgb.pt'}")
    print(f"  (or re-run train_yolo_rgb.py for the winning family with "
          f"--models-dest-name weed_rgb.pt to do this automatically)")


if __name__ == "__main__":
    main()
