#!/usr/bin/env python3
"""
tools/diagnose_performance.py
ABEN Triple RGB Imaging System — Frame-drop / performance diagnostic

Built to answer one question before committing to any fix (a full
DeepStream rewrite, a TensorRT export, a USB hardware change, or
nothing at all): WHERE is the actual bottleneck? "Frame drops from
some cameras" could mean a USB-level hardware issue (confirmed once
already this session with Cam 1 -- errno=19, "No such device", a real
disconnect, not a software slowdown), a genuine capture/inference
bottleneck, or both. This script gathers concrete evidence for each
possibility rather than guessing.

Usage (run from the project root, on the Jetson, with all 3 cameras
connected):
    python3 tools/diagnose_performance.py
    python3 tools/diagnose_performance.py --duration 30   # longer capture test

Checks performed, each independent -- if one section can't run
(missing binary, no camera hardware, no trained model yet), it says so
and the rest still proceeds:

1. MODEL FORMAT — is TensorRT actually being used? core/detection_
   config_rgb.py already prefers models/weed_rgb.engine over .pt
   (use_tensorrt=True is the default), with automatic fallback to .pt
   if the .engine file doesn't exist. If it's silently falling back,
   that alone could explain slow, GIL/CPU-bound inference -- a
   Jetson-side plain-PyTorch YOLOv8n forward pass is meaningfully
   slower than the same model exported to a TensorRT engine, and
   fixing it needs no code changes at all, just the export step.
2. USB TOPOLOGY — lsusb -t output, plus each of the 3 cameras' actual
   USB bus/port via udevadm, so it's visible whether all three share
   one hub/controller (a real bandwidth-contention risk for 3x 4K
   streams) or are spread across separate ones.
3. PER-CAMERA CAPTURE TEST — runs the REAL TripleEMEETCamera (the
   exact class the live app uses, not a simplified substitute) for a
   fixed duration and reports each camera's measured FPS and dropped-
   frame count individually. A drop concentrated on one camera points
   at that camera's specific connection; drops spread evenly across
   all three point at a shared bottleneck (USB bandwidth or CPU).
4. INFERENCE TIMING — loads whichever model core/detection_config_rgb.py's
   get_model_path() actually resolves to right now (.engine or .pt,
   whichever is really in use) and times several real forward passes,
   separate from and after the capture test, so capture-side and
   inference-side timing are never conflated.

Prints a plain-language summary at the end pointing at what the
numbers actually suggest, not just raw output.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check_model_format():
    _section("1. MODEL FORMAT — is TensorRT actually being used?")
    try:
        from core.detection_config_rgb import get_weed_config, DetectionMode
        cfg = get_weed_config()
        m = cfg.model
        print(f"use_tensorrt setting: {m.use_tensorrt}")
        print(f"weed_rgb.engine exists: {m.weed_rgb_engine.exists()} "
              f"({m.weed_rgb_engine})")
        print(f"weed_rgb.pt exists:     {m.weed_rgb_pt.exists()} "
              f"({m.weed_rgb_pt})")
        resolved = cfg.model.get_model_path(DetectionMode.WEED)
        print(f"\n→ get_model_path() currently resolves to: {resolved}")
        if resolved.suffix == ".engine":
            print("✓ TensorRT engine IS being used — inference should "
                  "already be running at TensorRT speed.")
        elif resolved.suffix == ".pt":
            if m.weed_rgb_engine.exists():
                print("⚠ A .engine file exists but ISN'T what's being "
                      "loaded -- check use_tensorrt and the exact path.")
            else:
                print("⚠ Falling back to plain PyTorch (.pt) because no "
                      ".engine file exists yet. This is a likely source "
                      "of slow inference on Jetson hardware -- exporting "
                      "the trained .pt to TensorRT (`model.export("
                      "format='engine', half=True)`) needs ZERO code "
                      "changes here, since get_model_path() already "
                      "prefers .engine automatically once it exists.")
    except Exception as e:
        print(f"Could not check model config: {e}")


def check_usb_topology():
    _section("2. USB TOPOLOGY — do the 3 cameras share a hub/controller?")
    try:
        result = subprocess.run(
            ["lsusb", "-t"], capture_output=True, text=True, timeout=5)
        print(result.stdout or "(no output)")
    except FileNotFoundError:
        print("lsusb not found — install usbutils to see USB topology.")
    except Exception as e:
        print(f"lsusb failed: {e}")

    try:
        from core.triple_emeet_camera import CAM1_DEVICE, CAM2_DEVICE, CAM3_DEVICE
    except ImportError:
        print("Could not import camera device paths.")
        return

    print("\nPer-camera USB bus/device (via udevadm):")
    for label, dev in [("Cam 1", CAM1_DEVICE), ("Cam 2", CAM2_DEVICE),
                       ("Cam 3", CAM3_DEVICE)]:
        try:
            result = subprocess.run(
                ["udevadm", "info", "-q", "path", "-n", dev],
                capture_output=True, text=True, timeout=5)
            path = result.stdout.strip()
            # The sysfs path contains the USB bus/hub topology
            # (e.g. .../usb1/1-2/1-2.3/...) -- same leading numeric
            # segment before a camera's own interface number means
            # they're on the same physical hub.
            print(f"  {label}: {path if path else '(not found -- is it connected?)'}")
        except FileNotFoundError:
            print("udevadm not found — skipping per-camera bus mapping.")
            break
        except Exception as e:
            print(f"  {label}: error ({e})")

    print("\n→ If Cam 1/2/3's paths share the same early bus segment "
          "(e.g. all start with the same '.../usb1/1-2/...' prefix), "
          "they're on the same physical USB hub/controller and "
          "genuinely competing for the same bandwidth -- 3x 4K MJPEG "
          "streams can exceed a single USB 3.0 hub's real throughput "
          "even though each camera alone works fine. Different early "
          "segments mean they're already spread across separate "
          "controllers and this isn't a shared-bandwidth issue.")


def run_capture_test(duration_s: float):
    _section(f"3. PER-CAMERA CAPTURE TEST ({duration_s:.0f}s, real hardware)")
    try:
        from core.triple_emeet_camera import TripleEMEETCamera
    except ImportError as e:
        print(f"Could not import TripleEMEETCamera: {e}")
        return

    try:
        cam = TripleEMEETCamera()
    except Exception as e:
        print(f"Could not open cameras: {e}")
        print("(Check they're connected and not held open by another "
              "process -- e.g. main_gui_triple.py already running.)")
        return

    try:
        cam.start()
        print("Capturing... (this reads the REAL, live cameras through "
              "the exact same TripleEMEETCamera class the app uses)")
        frame_counts = [0, 0, 0]
        t_start = time.time()
        last_frame_id = -1
        while time.time() - t_start < duration_s:
            triple = cam.read_triple()
            if triple is not None and triple.frame_id != last_frame_id:
                last_frame_id = triple.frame_id
                for i, f in enumerate(triple.frames):
                    if f is not None:
                        frame_counts[i] += 1
            time.sleep(0.01)
        elapsed = time.time() - t_start

        status = cam.get_status()
        print(f"\nElapsed: {elapsed:.1f}s")
        for i in range(3):
            fps = frame_counts[i] / elapsed if elapsed > 0 else 0
            print(f"  Cam {i+1}: {frame_counts[i]} frames delivered "
                  f"(~{fps:.1f} fps) | drop_count={status['drop_counts'][i]}")

        max_drops = max(status['drop_counts'])
        min_drops = min(status['drop_counts'])
        if max_drops > 0 and max_drops > min_drops * 3 and max_drops > 20:
            worst = status['drop_counts'].index(max_drops) + 1
            print(f"\n→ Drops are heavily concentrated on Cam {worst} "
                  f"({max_drops} vs the others' {min_drops}-ish) -- this "
                  f"points at THAT camera's specific USB connection "
                  f"(cable/port/power), not a shared software bottleneck. "
                  f"A DeepStream/TensorRT rewrite would not fix a single "
                  f"camera's flaky physical connection.")
        elif max_drops > 20:
            print(f"\n→ Drops are roughly even across all three cameras -- "
                  f"more consistent with a SHARED bottleneck (USB "
                  f"bandwidth if they're on one hub — see section 2 — "
                  f"or CPU contention from the capture threads "
                  f"themselves).")
        else:
            print(f"\n→ Drop counts are low across the board -- capture "
                  f"itself doesn't look like the bottleneck right now. "
                  f"If frame drops are still visible in the live app, "
                  f"the gap is more likely downstream (inference timing, "
                  f"or GUI-thread display work) -- see section 4.")
    finally:
        cam.stop()


def run_inference_benchmark():
    _section("4. INFERENCE TIMING (separate from capture)")
    try:
        import numpy as np
        from core.detection_config_rgb import get_weed_config
        from core.detection_engine_rgb import RGBDetectionEngine
    except ImportError as e:
        print(f"Could not import detection engine: {e}")
        return

    cfg = get_weed_config()
    engine = RGBDetectionEngine(cfg)
    if engine.stub_mode:
        print("Engine is in STUB MODE (no trained weights found) -- "
              "can't benchmark real inference. Place weed_rgb.pt or "
              "weed_rgb.engine in models/ first.")
        return

    dummy_frame = np.random.randint(
        0, 255, (1080, 1920, 3), dtype=np.uint8)

    print("Warming up (first inference includes model JIT/graph setup, "
          "not representative)...")
    engine._infer_one(dummy_frame, "warmup")

    n = 10
    times = []
    for _ in range(n):
        t0 = time.time()
        engine._infer_one(dummy_frame, "bench")
        times.append((time.time() - t0) * 1000)

    mean_t = sum(times) / len(times)
    print(f"\n{n} timed inference passes (single 1920x1080 frame each):")
    print(f"  mean: {mean_t:.1f}ms   min: {min(times):.1f}ms   "
          f"max: {max(times):.1f}ms")
    print(f"  → for 3 cameras sequentially, that's roughly "
          f"{mean_t*3:.0f}ms/cycle just for inference "
          f"(~{1000/(mean_t*3):.1f} fps ceiling from inference alone, "
          f"before capture/display overhead)")
    if mean_t > 60:
        print(f"\n→ {mean_t:.0f}ms per frame is slow for a small "
              f"detection model on Orin hardware -- if this is running "
              f"on plain PyTorch (.pt) rather than a TensorRT .engine "
              f"(see section 1), that's the single most likely, lowest-"
              f"risk fix to try before anything else: export the "
              f"trained model to TensorRT and this number should drop "
              f"substantially with no other code changes.")
    else:
        print(f"\n→ {mean_t:.0f}ms/frame is reasonable -- inference "
              f"doesn't look like the primary bottleneck here.")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose triple-camera frame-drop/performance issues")
    parser.add_argument(
        "--duration", type=float, default=15.0,
        help="Capture test duration in seconds (default: 15)")
    parser.add_argument(
        "--skip-capture", action="store_true",
        help="Skip the live capture test (e.g. if the main app is "
             "already running and holding the cameras open)")
    args = parser.parse_args()

    print("ABEN Triple RGB — Performance Diagnostic")
    print("Run this with the main app CLOSED (it needs exclusive access "
          "to the cameras for the capture test).")

    check_model_format()
    check_usb_topology()
    if not args.skip_capture:
        run_capture_test(args.duration)
    else:
        print("\n(Skipping live capture test per --skip-capture)")
    run_inference_benchmark()

    _section("SUMMARY")
    print("Review each section's '→' conclusion above. In short:")
    print("  - Drops concentrated on ONE camera → hardware/USB issue for "
          "that camera specifically (check its cable/port/power) — no "
          "software change fixes this.")
    print("  - Drops spread evenly + cameras share a USB hub → real "
          "bandwidth contention — could justify spreading cameras "
          "across separate USB controllers, still short of DeepStream.")
    print("  - Inference time is high AND it's running .pt not .engine "
          "→ export to TensorRT first (near-zero risk, no rewrite, "
          "code already supports it) before considering anything "
          "bigger.")
    print("  - Only if inference is already TensRT-fast AND capture "
          "drops are genuinely broad/systemic does a heavier pipeline "
          "change like DeepStream become worth its cost.")


if __name__ == "__main__":
    main()
