#!/usr/bin/env python3
"""
tools/check_camera.py
ABEN Dual/Triple RGB Imaging System — Simple camera verification tool

Standalone check for ONE camera at a time (e.g. the new Cam 2 / middle
camera) before it's wired into the full pipeline. Doesn't touch
core/dual_emeet_camera.py or anything else -- just OpenCV + v4l2-ctl,
so it's safe to run any time without affecting the live system.

Usage (run from the project root):
    python3 tools/check_camera.py --list
        Lists every /dev/v4l/by-id/ camera path found, flagging which
        ones are already known as Cam 1 / Cam 3 (from
        core/dual_emeet_camera.py's LEFT_CAMERA/RIGHT_CAMERA
        constants) so the new one is easy to spot.

    python3 tools/check_camera.py /dev/v4l/by-id/usb-EMEET_..._index0
        Opens that device, prints its actual resolution/FPS, confirms
        frames are genuinely readable (not just "opened"), then shows
        a live preview window.

    python3 tools/check_camera.py /dev/video2
        Same, but by the (less stable -- can change across reboots/
        replugs) /dev/videoN path, useful for a first look before
        finding the by-id path via --list.

Controls while the preview window is open:
    s        save a snapshot (cam_check_snapshot_N.jpg) to the
             current directory
    q / Esc  quit
"""

import sys
import subprocess
from pathlib import Path

import cv2

# The two ALREADY-KNOWN camera paths, for reference only -- run
# --list and whichever by-id entry ISN'T one of these two is the new
# Cam 2. Imported as plain strings, not from core/dual_emeet_camera.py,
# so this script has zero dependency on the rest of the project and
# still works even if that module can't import for some reason.
KNOWN_CAMERAS = {
    "usb-EMEET_EMEET_SmartCam_C960_4K_A241213000400860-video-index0": "Cam 1 (LEFT_CAMERA, N1)",
    "usb-EMEET_EMEET_SmartCam_C960_4K_A241217000804000-video-index0": "Cam 3 (RIGHT_CAMERA, N3)",
}


def list_devices():
    print("=" * 64)
    print("Plain /dev/video* devices")
    print("=" * 64)
    video_devs = sorted(Path("/dev").glob("video*"))
    if video_devs:
        for p in video_devs:
            print(f"  {p}")
    else:
        print("  (none found)")

    print()
    print("=" * 64)
    print("Stable /dev/v4l/by-id/ paths (use THESE for anything")
    print("permanent -- /dev/videoN numbering can change across")
    print("reboots or replugging a camera)")
    print("=" * 64)
    by_id_dir = Path("/dev/v4l/by-id")
    if not by_id_dir.exists():
        print("  (no /dev/v4l/by-id directory -- is v4l-utils installed "
              "and at least one camera plugged in?)")
        return

    entries = sorted(p for p in by_id_dir.iterdir() if "video-index0" in p.name)
    if not entries:
        print("  (directory exists but no *-video-index0 entries found)")
        return

    for p in entries:
        label = KNOWN_CAMERAS.get(p.name)
        if label:
            print(f"  [{label}] {p}")
        else:
            print(f"  [NEW / UNRECOGNIZED -- likely Cam 2] {p}")

    print()
    unknown = [p for p in entries if p.name not in KNOWN_CAMERAS]
    if len(unknown) == 1:
        print(f"→ Test the new one with:")
        print(f"    python3 tools/check_camera.py {unknown[0]}")
    elif len(unknown) > 1:
        print(f"→ Found {len(unknown)} unrecognized cameras, expected "
              f"exactly 1 (the new Cam 2) -- check for duplicate/stale "
              f"entries below before picking one.")
    else:
        print("→ No new/unrecognized camera found -- confirm Cam 2 is "
              "plugged in and powered.")


def check_camera(device: str, width: int = 1920, height: int = 1080):
    print(f"\nOpening {device} ...")
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print(f"✗ Could not open {device}")
        print("  Check: is it plugged in, powered, and not already in "
              "use by another program (e.g. the main GUI)?")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"✓ Opened successfully")
    print(f"  Requested: {width}x{height}")
    print(f"  Actual:    {actual_w}x{actual_h} @ {fps:.1f} fps")

    # Confirm it's genuinely streaming, not just "opened" -- a camera
    # can report isOpened()==True while producing zero real frames if
    # there's a driver/power issue.
    ok_count = 0
    for _ in range(10):
        ok, frame = cap.read()
        if ok and frame is not None:
            ok_count += 1
    print(f"  Frame read test: {ok_count}/10 frames read successfully")
    if ok_count == 0:
        print("✗ Camera opened but produced NO readable frames -- "
              "check the physical connection/power, not a software issue.")
        cap.release()
        return False

    print(f"\nLive preview open. Press 's' to save a snapshot, "
          f"'q' or Esc to quit.")
    snap_count = 0
    win_name = f"Camera Check: {device}"
    while True:
        ok, frame = cap.read()
        if not ok:
            print("⚠ Lost a frame mid-preview")
            continue
        h, w = frame.shape[:2]
        cv2.putText(frame, f"{w}x{h} @ {fps:.1f}fps  |  's' save, 'q' quit",
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow(win_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            snap_count += 1
            fname = f"cam_check_snapshot_{snap_count}.jpg"
            cv2.imwrite(fname, frame)
            print(f"  Saved {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✓ Camera check complete ({snap_count} snapshot(s) saved).")
    return True


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("--list", "-l"):
        list_devices()
        if not args:
            print("\nRun again with a device path to actually test it "
                  "(see the command printed above once a new camera is "
                  "found).")
        return

    check_camera(args[0])


if __name__ == "__main__":
    main()
