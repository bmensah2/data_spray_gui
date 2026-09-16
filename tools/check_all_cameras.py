#!/usr/bin/env python3
"""
tools/check_all_cameras.py
ABEN Triple RGB Imaging System — Combined 3-camera overlap check

Opens all three eMeet C960 4K cameras (Cam 1/N1, Cam 2/N2, Cam 3/N3)
SIMULTANEOUSLY and shows them side by side in one window, so a shared
reference object (e.g. a tape measure laid across the ground, in view
of more than one camera at once) is visible across all three at the
same time -- the fastest way to see the REAL overlap between adjacent
cameras' fields of view, rather than inferring it from three
separately-measured numbers taken one camera at a time.

Usage (run from the project root):
    python3 tools/check_all_cameras.py

Controls while the preview window is open:
    s        save each camera's FULL-RESOLUTION frame individually,
             plus one downscaled combined side-by-side image for quick
             reference (4 files total per press) to the current
             directory
    q / Esc  quit

Note: running all three cameras at once, each requesting 1920x1080
MJPG @ 30fps, is meaningfully more USB bandwidth than running just one
(as tools/check_camera.py does). If they're all on the same USB hub/
controller, this can occasionally hit a bandwidth ceiling even with
MJPG compression -- if a camera fails to open here but worked fine
individually with check_camera.py, that's the most likely explanation,
not a camera fault.
"""

import sys

import cv2
import numpy as np

# Same three cameras + known serials as tools/check_camera.py --
# duplicated here rather than imported, to keep this script equally
# dependency-free and runnable standalone. Cam 2's serial confirmed
# working at full 1920x1080 via check_camera.py before this was written.
CAMERAS = [
    ("Cam 1 / N1", "/dev/v4l/by-id/usb-EMEET_EMEET_SmartCam_C960_4K_A241213000400860-video-index0"),
    ("Cam 2 / N2", "/dev/v4l/by-id/usb-EMEET_EMEET_SmartCam_C960_4K_A250106000208079-video-index0"),
    ("Cam 3 / N3", "/dev/v4l/by-id/usb-EMEET_EMEET_SmartCam_C960_4K_A241217000804000-video-index0"),
]


def open_camera(device: str, width: int = 1920, height: int = 1080):
    """Open one camera with the same MJPG-first setup that fixed Cam 2's
    resolution in tools/check_camera.py. Returns None on failure rather
    than raising, so the caller can report which specific camera failed."""
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def build_combined_display(frames, labels, disp_width: int = 640):
    """
    Downscale each frame to disp_width (preserving aspect ratio) and
    hstack them with a thin red divider between each -- display only;
    saved snapshots always use the original full-resolution frames,
    never this downscaled copy.
    """
    disp_frames = []
    for frame, label in zip(frames, labels):
        h, w = frame.shape[:2]
        disp_h = max(1, int(disp_width * h / w))
        small = cv2.resize(frame, (disp_width, disp_h))
        cv2.putText(small, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (0, 255, 0), 2)
        disp_frames.append(small)

    divider = np.zeros((disp_frames[0].shape[0], 4, 3), dtype=np.uint8)
    divider[:, :] = (0, 0, 255)

    parts = [disp_frames[0]]
    for d in disp_frames[1:]:
        parts.append(divider)
        parts.append(d)
    combined = cv2.hconcat(parts)
    cv2.putText(combined, "'s' save all frames, 'q' quit",
               (10, combined.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
               0.7, (0, 255, 255), 2)
    return combined


def main():
    print("Opening all 3 cameras simultaneously ...")
    caps = []
    for label, device in CAMERAS:
        cap = open_camera(device)
        if cap is None:
            print(f"✗ Could not open {label} at {device}")
            print("  Check: plugged in, powered, not already in use by "
                  "another program (e.g. the main GUI), and see the USB "
                  "bandwidth note in this script's docstring if the other "
                  "two opened fine.")
            for c, _ in caps:
                c.release()
            sys.exit(1)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✓ {label}: {actual_w}x{actual_h}")
        caps.append((cap, label))

    print(f"\nAll 3 cameras open. Live combined preview starting.")
    print(f"Lay a shared reference object (e.g. a tape measure) across "
          f"the ground so it's visible spanning the overlap regions "
          f"between adjacent cameras.")
    print(f"Press 's' to save each camera's full-resolution frame plus "
          f"one combined reference image, 'q' or Esc to quit.\n")

    snap_count = 0
    win_name = "All 3 Cameras — Overlap Check"
    try:
        while True:
            frames = []
            ok_all = True
            for cap, label in caps:
                ok, frame = cap.read()
                if not ok or frame is None:
                    ok_all = False
                    break
                frames.append(frame)

            if not ok_all:
                print("⚠ Lost a frame from one camera this cycle, retrying…")
                continue

            labels = [label for _, label in caps]
            combined = build_combined_display(frames, labels)
            cv2.imshow(win_name, combined)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                snap_count += 1
                for (cap, label), frame in zip(caps, frames):
                    safe_label = label.replace(" / ", "_").replace(" ", "_")
                    fname = f"overlap_check_{snap_count}_{safe_label}.jpg"
                    cv2.imwrite(fname, frame)
                    print(f"  Saved {fname} "
                         f"({frame.shape[1]}x{frame.shape[0]})")
                combo_fname = f"overlap_check_{snap_count}_combined.jpg"
                cv2.imwrite(combo_fname, combined)
                print(f"  Saved {combo_fname} (downscaled, for quick "
                     f"reference only)")
    finally:
        for cap, _ in caps:
            cap.release()
        cv2.destroyAllWindows()

    print(f"\n✓ Done. {snap_count} snapshot set(s) saved "
          f"({snap_count * 4} files total).")


if __name__ == "__main__":
    main()
