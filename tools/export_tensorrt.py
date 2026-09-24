#!/usr/bin/env python3
"""
tools/export_tensorrt.py
ABEN Triple RGB Imaging System — Export trained model(s) to TensorRT

Confirmed by tools/diagnose_performance.py on real hardware: inference
is currently running on plain PyTorch (weed_rgb.pt exists,
weed_rgb.engine doesn't) at ~45ms/frame -- reasonable, but not fast.
core/detection_config_rgb.py already defaults to preferring a
TensorRT .engine over .pt (use_tensorrt=True), with automatic
fallback if the .engine file doesn't exist -- so producing that file
is the entire fix. No code changes anywhere else in the app are
needed; get_model_path() already does the right thing once the file
exists.

This script reads the SAME config the live app uses
(core/detection_config_rgb.py's get_weed_config()) to resolve the
exact .pt path and export settings, rather than hardcoding them
separately -- so the exported engine is guaranteed to match what the
app actually expects:
  - imgsz:  cfg.model.input_size (640 by default) -- the app resizes
            every frame to this exact size before inference
            (core/detection_engine_rgb.py's _preprocess()); a
            mismatched export size would silently produce wrong
            results, not just an error.
  - batch:  1 -- TripleDetectionEngine.run_triple() calls
            _infer_one() once per camera, sequentially, never as a
            batched call across cameras (unlike a from-scratch
            DeepStream pipeline design, which would batch all 3
            together -- this app's actual architecture doesn't).
  - device: cfg.model.device (cuda:0 by default) -- inference is
            already running on GPU via PyTorch; TensorRT compiles a
            GPU-specific engine for that same device.
  - half:   True (FP16) by default -- the standard, well-supported
            speed/precision tradeoff for Jetson inference; --no-half
            to export FP32 instead if precision is a concern.

Usage (run ON THE JETSON, where the real weights and TensorRT/CUDA
libraries live -- this cannot run in a sandbox without a GPU):
    python3 tools/export_tensorrt.py                 # exports weed model
    python3 tools/export_tensorrt.py --mode cls       # exports cls model
    python3 tools/export_tensorrt.py --mode both      # exports both
    python3 tools/export_tensorrt.py --no-half        # FP32 instead of FP16

What to expect: TensorRT engine compilation genuinely takes several
minutes (the exact time depends on the model and Jetson power mode --
budget 2-10 minutes, not a hang). This only needs to run once per
model version; the app picks up the resulting .engine file
automatically on its next start, no further action needed.
"""

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def export_one(pt_path: Path, engine_path: Path, imgsz: int, device: str,
               half: bool) -> bool:
    """
    Export one .pt file to TensorRT. Returns True on success.
    Ultralytics' own export() writes the resulting .engine file next
    to the input .pt file, same basename -- since pt_path is already
    the exact path get_model_path() will later look for the .pt at,
    the output naturally lands at engine_path too (same directory,
    same stem, .engine extension) without needing to move anything
    afterward.
    """
    if not pt_path.exists():
        print(f"✗ {pt_path} does not exist -- nothing to export.")
        return False

    try:
        from ultralytics import YOLO
    except ImportError:
        print("✗ ultralytics is not installed in this environment -- "
              "run this on the Jetson where the app's own model "
              "loading already works.")
        return False

    print(f"\nExporting {pt_path.name} → {engine_path.name}")
    print(f"  imgsz={imgsz}  batch=1  device={device}  "
          f"half={'FP16' if half else 'FP32'}")
    print("  This genuinely takes a few minutes (TensorRT is "
          "compiling/optimizing the engine for THIS specific Jetson "
          "-- not a hang).")

    t0 = time.time()
    model = YOLO(str(pt_path))
    try:
        result_path = model.export(
            format="engine",
            imgsz=imgsz,
            batch=1,
            device=device,
            half=half,
        )
    except Exception as e:
        print(f"✗ Export failed: {e}")
        return False
    elapsed = time.time() - t0

    result_path = Path(result_path) if result_path else None
    if result_path and result_path.exists():
        size_mb = result_path.stat().st_size / (1024 * 1024)
        print(f"✓ Exported in {elapsed:.0f}s → {result_path} "
              f"({size_mb:.1f} MB)")
        if result_path != engine_path:
            print(f"  Note: ultralytics wrote to {result_path}, which "
                  f"differs from the expected {engine_path} -- the app "
                  f"won't find it there automatically. Move/rename it "
                  f"to match, or check why the output path differs.")
        return True
    else:
        print(f"✗ export() completed but no .engine file was found "
              f"afterward -- check the ultralytics output above for "
              f"errors.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Export trained YOLO .pt weights to TensorRT .engine, "
                    "matching the exact settings the live app uses.")
    parser.add_argument(
        "--mode", choices=["weed", "cls", "both"], default="weed",
        help="Which model to export (default: weed -- the actively-used "
             "mode for the triple-camera spray system)")
    parser.add_argument(
        "--no-half", action="store_true",
        help="Export FP32 instead of the default FP16")
    args = parser.parse_args()

    try:
        from core.detection_config_rgb import get_weed_config, DetectionMode
    except ImportError as e:
        print(f"Could not import config: {e}")
        sys.exit(1)

    cfg = get_weed_config()
    m = cfg.model
    half = not args.no_half

    targets = []
    if args.mode in ("weed", "both"):
        targets.append(("weed", m.weed_rgb_pt, m.weed_rgb_engine))
    if args.mode in ("cls", "both"):
        targets.append(("cls", m.cls_rgb_pt, m.cls_rgb_engine))

    print("ABEN Triple RGB — TensorRT Export")
    print(f"input_size={m.input_size}  device={m.device}  "
         f"precision={'FP16' if half else 'FP32'}")

    all_ok = True
    for label, pt_path, engine_path in targets:
        ok = export_one(pt_path, engine_path, m.input_size, m.device, half)
        all_ok = all_ok and ok

    print("\n" + "=" * 60)
    if all_ok:
        print("✓ Done. The app will automatically use the new .engine "
              "file(s) on its next start -- core/detection_config_rgb.py's "
              "get_model_path() already prefers .engine over .pt, no "
              "code or config changes needed.")
        print("  Re-run tools/diagnose_performance.py to confirm section 1 "
              "now shows the .engine file in use, and to re-measure "
              "inference timing against the earlier ~45ms/frame baseline.")
    else:
        print("⚠ At least one export did not succeed -- see the errors "
              "above. The app will keep using the .pt file(s) it already "
              "has until a valid .engine file exists; nothing is broken "
              "by a failed export.")
    print("=" * 60)


if __name__ == "__main__":
    main()
