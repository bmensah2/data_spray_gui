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

Each camera is read in its OWN background thread (same pattern as
core/dual_emeet_camera.py's proven-working 2-camera capture, just
extended to 3) rather than reading all three sequentially in one loop.
A sequential read blocks on cap.read() for camera 1 before even
attempting camera 2 or 3 -- if any single camera stalls for a moment
(startup, USB contention, a slow frame), the whole loop stalls with
it, which is what produced repeated "Lost a frame... retrying" with
no progress on the very first version of this script. Independent
threads mean one camera's momentary stall never blocks the others or
the display loop; the display just shows each camera's most recently
completed frame.
"""

import sys
import threading
import time

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

# How long the main loop waits for EVERY camera to have delivered at
# least one frame before giving up and reporting which one(s) never
# did -- a real, specific diagnostic instead of hanging silently.
STARTUP_TIMEOUT_S = 8.0


class CameraStream:
    """
    One camera, read continuously in its own background thread. The
    main/display loop only ever reads self.frame (the latest
    successfully-read frame, or None until the first one arrives) --
    it never calls cap.read() itself and so can never be blocked by
    this camera specifically.
    """

    def __init__(self, label: str, device: str):
        self.label   = label
        self.device  = device
        self.frame   = None
        self.ts      = 0.0
        self.opened  = False
        self.actual_w = 0
        self.actual_h = 0
        self._lock    = threading.Lock()
        self._running = False
        self._cap     = None
        self._thread  = None

    def start(self, width: int = 1920, height: int = 1080) -> bool:
        cap = cv2.VideoCapture(self.device)
        if not cap.isOpened():
            return False
        # MJPG MUST be set before width/height -- same fix that
        # resolved Cam 2 reporting 640x480 in tools/check_camera.py.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._cap     = cap
        self.opened   = True
        self._running = True
        self._thread  = threading.Thread(
            target=self._capture_loop, daemon=True,
            name=f"cam-{self.label}")
        self._thread.start()
        return True

    def _capture_loop(self):
        while self._running:
            ok, frame = self._cap.read()
            if ok and frame is not None:
                with self._lock:
                    self.frame = frame
                    self.ts    = time.time()
            # No sleep/backoff on failure -- this is a short-lived
            # interactive diagnostic tool, not a long-running service;
            # a brief burst of fast retries during startup is fine.

    def latest(self):
        with self._lock:
            return self.frame, self.ts

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()


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
    print("Opening all 3 cameras (each in its own capture thread) ...")
    streams = []
    for label, device in CAMERAS:
        s = CameraStream(label, device)
        if not s.start():
            print(f"✗ Could not open {label} at {device}")
            print("  Check: plugged in, powered, not already in use by "
                  "another program (e.g. the main GUI).")
            for done in streams:
                done.stop()
            sys.exit(1)
        print(f"✓ {label}: opened, requested {s.actual_w}x{s.actual_h}")
        streams.append(s)

    print(f"\nWaiting for a first frame from all 3 cameras "
          f"(up to {STARTUP_TIMEOUT_S:.0f}s) ...")
    deadline = time.time() + STARTUP_TIMEOUT_S
    while time.time() < deadline:
        if all(s.latest()[0] is not None for s in streams):
            break
        time.sleep(0.1)
    missing = [s.label for s in streams if s.latest()[0] is None]
    if missing:
        print(f"✗ Timed out waiting for a first frame from: "
              f"{', '.join(missing)}")
        print(f"  The other camera(s) delivered frames fine, so this "
              f"is most likely a USB bandwidth/hub contention issue "
              f"with 3 simultaneous 1080p MJPG streams -- try moving "
              f"one camera to a different USB port/hub if available, "
              f"or test that specific camera alone first with "
              f"tools/check_camera.py to rule out a camera-specific "
              f"problem.")
        for s in streams:
            s.stop()
        sys.exit(1)
    print("✓ All 3 cameras delivering frames.\n")

    print("Live combined preview starting.")
    print("Lay a shared reference object (e.g. a tape measure) across "
          "the ground so it's visible spanning the overlap regions "
          "between adjacent cameras.")
    print("Press 's' to save each camera's full-resolution frame plus "
          "one combined reference image, 'q' or Esc to quit.\n")

    snap_count = 0
    win_name = "All 3 Cameras — Overlap Check"
    try:
        while True:
            frames = [s.latest()[0] for s in streams]
            labels = [s.label for s in streams]
            combined = build_combined_display(frames, labels)
            cv2.imshow(win_name, combined)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                snap_count += 1
                for s, frame in zip(streams, frames):
                    safe_label = s.label.replace(" / ", "_").replace(" ", "_")
                    fname = f"overlap_check_{snap_count}_{safe_label}.jpg"
                    cv2.imwrite(fname, frame)
                    print(f"  Saved {fname} "
                         f"({frame.shape[1]}x{frame.shape[0]})")
                combo_fname = f"overlap_check_{snap_count}_combined.jpg"
                cv2.imwrite(combo_fname, combined)
                print(f"  Saved {combo_fname} (downscaled, for quick "
                     f"reference only)")
    finally:
        for s in streams:
            s.stop()
        cv2.destroyAllWindows()

    print(f"\n✓ Done. {snap_count} snapshot set(s) saved "
          f"({snap_count * 4} files total).")


if __name__ == "__main__":
    main()
