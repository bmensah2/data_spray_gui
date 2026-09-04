#!/usr/bin/env python3
"""
dual_emeet_camera.py
eMeet Dual RGB Camera Driver

Manages two eMeet SmartCam C960 4K cameras for the ABEN dual-RGB
detection system.  Each camera runs in its own capture thread so
the main loop always gets the latest frame from both cameras with
minimal latency.

Camera identification (stable symlinks, not /dev/videoX):
  LEFT  — usb-EMEET_EMEET_SmartCam_C960_4K_A241213000400860-video-index0
  RIGHT — usb-EMEET_EMEET_SmartCam_C960_4K_A241217000804000-video-index0

Settings validated on Jetson AGX Orin (v4l2-ctl):
  Resolution : 1920×1080 @ 30fps (MJPG)
  Focus      : Manual, absolute=460
  Exposure   : Manual, time_absolute=300
  White bal  : Manual, temperature=5000

Author : Nana | NDSU / PhD Imaging System
Path   : /media/pagsun/Transcend/phd_project/emeet_dual_cam/
"""

import cv2
import time
import logging
import threading
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict

# ─────────────────────────────────────────────────────────────
#  CAMERA DEVICE PATHS  (stable symlinks — don't use /dev/videoX)
# ─────────────────────────────────────────────────────────────

LEFT_CAMERA  = (
    "/dev/v4l/by-id/"
    "usb-EMEET_EMEET_SmartCam_C960_4K_A241213000400860-video-index0"
)
RIGHT_CAMERA = (
    "/dev/v4l/by-id/"
    "usb-EMEET_EMEET_SmartCam_C960_4K_A241217000804000-video-index0"
)


# ─────────────────────────────────────────────────────────────
#  FRAME PAIR
# ─────────────────────────────────────────────────────────────

@dataclass
class FramePair:
    """
    One synchronized read from both cameras.
    All consumers work with FramePair objects, not raw ndarray tuples.
    """
    left:           object          # np.ndarray BGR 1920×1080
    right:          object          # np.ndarray BGR 1920×1080
    frame_id:       int
    left_ts:        float           # time.time() of left capture
    right_ts:       float           # time.time() of right capture
    sync_error_ms:  float           # |left_ts - right_ts| × 1000

    @property
    def sync_ok(self) -> bool:
        """True if cameras are in sync within 50 ms."""
        return self.sync_error_ms < 50.0

    def to_meta(self) -> Dict:
        return {
            'frame_id':      self.frame_id,
            'left_ts':       self.left_ts,
            'right_ts':      self.right_ts,
            'sync_error_ms': self.sync_error_ms,
            'sync_ok':       self.sync_ok,
        }


# ─────────────────────────────────────────────────────────────
#  DUAL EMEET CAMERA
# ─────────────────────────────────────────────────────────────

class DualEMEETCamera:
    """
    Threaded dual-camera manager for two eMeet SmartCam C960 4K cameras.

    Usage:
        cam = DualEMEETCamera()
        cam.start()

        while True:
            pair = cam.read_pair()
            if pair is None:
                continue
            # pair.left, pair.right → np.ndarray BGR

        cam.stop()
    """

    # ── Camera setting presets ───────────────────────────────
    PRESET_INDOOR = {
        "autofocus":  1,
        "exposure":   300,
        "brightness": 0,
        "contrast":   57,
        "saturation": 80,
        "gamma":      214,
        "gain":       0,
        "sharpness":  32,
        "wb_temp":    5000,
        "focus":      460,
        "backlight":  0,
        "freq":       2,
    }
    PRESET_OUTDOOR = {
        "autofocus":  1,
        "exposure":   5,
        "brightness": -10,
        "contrast":   60,
        "saturation": 80,
        "gamma":      150,
        "gain":       0,
        "sharpness":  40,
        "wb_temp":    5500,
        "focus":      460,
        "backlight":  0,
        "freq":       2,
    }
    PRESET_CLOUDY = {
        "autofocus":  1,
        "exposure":   80,
        "brightness": 0,
        "contrast":   58,
        "saturation": 80,
        "gamma":      180,
        "gain":       0,
        "sharpness":  36,
        "wb_temp":    6000,
        "focus":      460,
        "backlight":  0,
        "freq":       2,
    }
    # Default — indoor/lab
    DEFAULTS = PRESET_INDOOR

    def __init__(
        self,
        left_src:       str  = LEFT_CAMERA,
        right_src:      str  = RIGHT_CAMERA,
        width:          int  = 1920,
        height:         int  = 1080,
        fps:            int  = 30,
        save_dir:       str  = "captures",
        save_images:    bool = False,
        settings:       dict = None,
        skip_configure: bool = False,
    ):
        """
        settings       : dict of v4l2 values to apply instead of DEFAULTS.
                         Keys: autofocus, exposure, brightness, contrast, saturation,
                               gamma, gain, sharpness, wb_temp, focus,
                               backlight, freq.
                         Pass PRESET_OUTDOOR / PRESET_CLOUDY / PRESET_INDOOR
                         or a custom dict. Missing keys fall back to DEFAULTS.
        skip_configure : if True, skip all v4l2 configuration.
                         Use when camera is already configured correctly
                         (e.g. spray mission launched after GUI has set values).
        """
        self.left_src    = left_src
        self.right_src   = right_src
        self.width       = width
        self.height      = height
        self.fps         = fps
        self.save_images = save_images
        self.save_dir    = Path(save_dir)

        # Per-camera save dirs (created only if save_images=True)
        self.left_dir  = self.save_dir / "left"
        self.right_dir = self.save_dir / "right"
        if self.save_images:
            self.left_dir.mkdir(parents=True, exist_ok=True)
            self.right_dir.mkdir(parents=True, exist_ok=True)

        # Shared frame buffers (written by capture threads, read by main)
        self._left_frame:  Optional[object] = None
        self._right_frame: Optional[object] = None
        self._left_ts:     Optional[float]  = None
        self._right_ts:    Optional[float]  = None

        self._left_lock  = threading.Lock()
        self._right_lock = threading.Lock()

        self._running    = False
        self._frame_id   = 0

        # Stats
        self._left_drop  = 0
        self._right_drop = 0

        self._settings       = {**self.DEFAULTS, **(settings or {})}
        self._skip_configure = skip_configure

        if skip_configure:
            logging.info(
                "DualEMEETCamera: skipping v4l2 configuration "
                "(using existing camera settings)")
        else:
            s = self._settings
            logging.info(
                f"DualEMEETCamera: configuring cameras via v4l2-ctl "
                f"(exp={s['exposure']} gamma={s['gamma']} "
                f"wb={s['wb_temp']}) …")
            self._configure(self.left_src)
            self._configure(self.right_src)

        logging.info("DualEMEETCamera: opening VideoCapture …")
        self._left_cap  = self._open(self.left_src)
        self._right_cap = self._open(self.right_src)
        logging.info("DualEMEETCamera: ready")

    # ── v4l2 configuration ────────────────────────────────────

    def _v4l2(self, device: str, control: str, value) -> bool:
        """Set one v4l2 control. Returns True on success."""
        cmd = ["v4l2-ctl", "-d", device, "-c", f"{control}={value}"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logging.warning(
                f"v4l2-ctl: could not set {control}={value} on {device} "
                f"— {result.stderr.strip()}"
            )
            return False
        return True

    def _configure(self, device: str):
        """
        Apply camera settings via v4l2-ctl using self._settings.
        Callers can pass settings=DualEMEETCamera.PRESET_OUTDOOR etc.
        to the constructor to configure for field conditions.
        """
        s = self._settings
        logging.info(f"  Configuring {device}")

        # Image quality
        self._v4l2(device, "brightness",             s["brightness"])
        self._v4l2(device, "contrast",               s["contrast"])
        self._v4l2(device, "saturation",             s["saturation"])
        self._v4l2(device, "hue",                    0)
        self._v4l2(device, "gamma",                  s["gamma"])
        self._v4l2(device, "gain",                   s["gain"])
        self._v4l2(device, "sharpness",              s["sharpness"])
        self._v4l2(device, "backlight_compensation", s["backlight"])
        self._v4l2(device, "power_line_frequency",   s["freq"])

        # Focus — autofocus is enabled by default; manual focus remains available.
        autofocus = s.get("autofocus", 1)
        self._v4l2(device, "focus_automatic_continuous", autofocus)
        if not autofocus:
            time.sleep(0.1)
            self._v4l2(device, "focus_absolute", s["focus"])

        # Exposure — disable auto first, then set absolute
        self._v4l2(device, "auto_exposure", 1)
        time.sleep(0.1)
        self._v4l2(device, "exposure_time_absolute", s["exposure"])

        # White balance — disable auto first, then set temperature
        self._v4l2(device, "white_balance_automatic", 0)
        time.sleep(0.1)
        self._v4l2(device, "white_balance_temperature", s["wb_temp"])

    # ── VideoCapture setup ────────────────────────────────────

    def _open(self, src: str) -> cv2.VideoCapture:
        """Open one camera and configure resolution/codec."""
        cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS,          self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # always latest frame

        if not cap.isOpened():
            raise RuntimeError(f"DualEMEETCamera: could not open {src}")

        # Confirm actual resolution
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        logging.info(
            f"  Opened {src} → {actual_w}×{actual_h} @ {actual_fps:.0f}fps"
        )
        return cap

    # ── Capture threads ───────────────────────────────────────

    def start(self):
        """Start background capture threads."""
        if self._running:
            logging.warning("DualEMEETCamera.start() called while already running")
            return

        self._running = True

        self._left_thread = threading.Thread(
            target=self._capture_loop,
            args=(self._left_cap, self._left_lock,
                  '_left_frame', '_left_ts', 'LEFT'),
            daemon=True,
            name="emeet-left"
        )
        self._right_thread = threading.Thread(
            target=self._capture_loop,
            args=(self._right_cap, self._right_lock,
                  '_right_frame', '_right_ts', 'RIGHT'),
            daemon=True,
            name="emeet-right"
        )

        self._left_thread.start()
        self._right_thread.start()
        logging.info("DualEMEETCamera: capture threads started")

    def _capture_loop(self, cap, lock, frame_attr, ts_attr, label):
        """
        Per-camera capture loop — runs in its own thread.

        Backs off on persistent failures instead of spinning at full
        CPU re-attempting cap.read() as fast as possible -- confirmed
        in practice to reach tens of millions of failed reads and log
        lines in a single session when a camera physically disconnects
        (errno=19 ENODEV) and nothing throttles the retry rate. Log
        warnings are time-based (at most once per 5s) rather than
        count-based, so log volume stays bounded regardless of how
        fast failures happen -- a count-based "every 30 drops" throttle
        does nothing useful once drops are happening thousands of
        times per second.
        """
        drop_count            = 0
        consecutive_failures  = 0
        last_warn_time        = 0.0
        device_declared_dead  = False

        while self._running:
            ret, frame = cap.read()
            if ret:
                if consecutive_failures > 0:
                    logging.info(
                        f"DualEMEETCamera [{label}]: recovered after "
                        f"{consecutive_failures} consecutive failed reads")
                consecutive_failures = 0
                device_declared_dead = False
                with lock:
                    setattr(self, frame_attr, frame)
                    setattr(self, ts_attr, time.time())
            else:
                drop_count           += 1
                consecutive_failures += 1

                now = time.time()
                if now - last_warn_time >= 5.0:
                    logging.warning(
                        f"DualEMEETCamera [{label}]: dropped frame "
                        f"(total drops: {drop_count}, "
                        f"{consecutive_failures} consecutive)"
                    )
                    last_warn_time = now

                if consecutive_failures >= 100 and not device_declared_dead:
                    logging.error(
                        f"DualEMEETCamera [{label}]: {consecutive_failures} "
                        f"consecutive failed reads -- camera appears to "
                        f"have disconnected. Backing off retries (checking "
                        f"every 2s) instead of spinning at full CPU. Check "
                        f"the USB connection/power to this camera."
                    )
                    device_declared_dead = True

                # Backoff so a genuinely dead/disconnected camera doesn't
                # peg this thread's CPU forever. First few failures might
                # just be a normal transient single-frame drop -- retry
                # immediately, same as before, with no sleep at all.
                if device_declared_dead:
                    time.sleep(2.0)
                elif consecutive_failures >= 10:
                    time.sleep(0.1)

    # ── Public API ────────────────────────────────────────────

    def read_pair(self) -> Optional[FramePair]:
        """
        Return the latest synchronized frame pair.
        Returns None if either camera hasn't produced a frame yet.

        This is the primary method called every loop iteration
        by detection_engine_rgb.py and main_gui_rgb.py.
        """
        with self._left_lock:
            left    = None if self._left_frame  is None else self._left_frame.copy()
            left_ts = self._left_ts

        with self._right_lock:
            right    = None if self._right_frame is None else self._right_frame.copy()
            right_ts = self._right_ts

        if left is None or right is None:
            return None

        sync_error_ms = abs(left_ts - right_ts) * 1000
        pair = FramePair(
            left          = left,
            right         = right,
            frame_id      = self._frame_id,
            left_ts       = left_ts,
            right_ts      = right_ts,
            sync_error_ms = sync_error_ms,
        )
        self._frame_id += 1
        return pair

    def save_pair(self, pair: FramePair) -> Tuple[Path, Path]:
        """
        Save left and right frames to disk as JPEG.
        Only usable when save_images=True.
        """
        if not self.save_images:
            raise RuntimeError("save_pair() called but save_images=False")

        left_path  = self.left_dir  / f"left_{pair.frame_id:06d}.jpg"
        right_path = self.right_dir / f"right_{pair.frame_id:06d}.jpg"

        cv2.imwrite(str(left_path),  pair.left)
        cv2.imwrite(str(right_path), pair.right)

        logging.info(
            f"Saved pair {pair.frame_id:06d} "
            f"(sync error: {pair.sync_error_ms:.1f} ms)"
        )
        return left_path, right_path

    def get_status(self) -> Dict:
        """Return camera health status dict."""
        return {
            'running':       self._running,
            'frame_id':      self._frame_id,
            'left_ok':       self._left_frame  is not None,
            'right_ok':      self._right_frame is not None,
            'resolution':    f"{self.width}×{self.height}",
            'fps_target':    self.fps,
        }

    def stop(self):
        """Stop capture threads and release cameras."""
        if not self._running:
            return

        logging.info("DualEMEETCamera: stopping …")
        self._running = False

        # Explicitly join the capture threads before releasing the
        # VideoCapture objects -- a fixed sleep() here is a guess, not
        # a guarantee, and calling .release() on a cv2.VideoCapture
        # while another thread is still blocked inside .read() on that
        # SAME object is a real race in OpenCV's C++ backend. This can
        # surface as a low-level "terminate called without an active
        # exception" crash during process teardown rather than a
        # catchable Python exception -- confirmed via tools/demo_spray.py,
        # which was (accidentally, via this session's earlier fixes)
        # the first caller to actually exercise this stop() path in a
        # way that reliably hit the race. join(timeout=...) rather than
        # an unbounded join() so a genuinely stuck capture thread can't
        # hang shutdown forever -- .release() still runs either way,
        # same as before, just after actually confirming (not guessing)
        # the threads are done or timed out.
        for t in (self._left_thread, self._right_thread):
            if t is not None and t.is_alive():
                t.join(timeout=1.0)
                if t.is_alive():
                    logging.warning(
                        f"DualEMEETCamera: {t.name} did not exit within "
                        f"1.0s -- releasing camera anyway")

        self._left_cap.release()
        self._right_cap.release()
        logging.info("DualEMEETCamera: stopped")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


# ─────────────────────────────────────────────────────────────
#  QUICK TEST  (python3 dual_emeet_camera.py)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import time as _time
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    print("=" * 60)
    print("DualEMEETCamera — Headless Test")
    print("Captures 5 frame pairs, saves pair 0 to captures/")
    print("=" * 60)

    cam = DualEMEETCamera(
        width=1920, height=1080, fps=30,
        save_dir="captures", save_images=True,
    )
    cam.start()

    _time.sleep(2.0)

    sync_errors = []
    saved = False

    for i in range(5):
        pair = None
        for _ in range(60):
            pair = cam.read_pair()
            if pair is not None:
                break
            _time.sleep(0.05)

        if pair is None:
            print(f"  [{i}] No frame received")
            continue

        sync_errors.append(pair.sync_error_ms)
        print(
            f"  [{pair.frame_id:04d}]  "
            f"sync={pair.sync_error_ms:.1f}ms  "
            f"{'OK' if pair.sync_ok else 'WARN'}"
        )

        if not saved:
            l, r = cam.save_pair(pair)
            print(f"           Saved -> {l}  {r}")
            saved = True

        _time.sleep(0.2)

    cam.stop()

    if sync_errors:
        avg = sum(sync_errors) / len(sync_errors)
        print(f"\nAvg sync error: {avg:.1f}ms over {len(sync_errors)} pairs")
    print("Done.")