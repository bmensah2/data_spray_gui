#!/usr/bin/env python3
"""
triple_emeet_camera.py
ABEN Triple RGB Camera Driver

Phase 2 of the triple-camera software redesign: the camera capture
layer. Manages three eMeet SmartCam C960 4K cameras (Cam 1/N1, Cam
2/N2, Cam 3/N3, one per nozzle -- see the physical layout in
core/zone_manager_triple.py's module docstring) for the ABEN triple-
RGB detection system. Built as a new, self-contained module alongside
the existing, still-live core/dual_emeet_camera.py (DualEMEETCamera)
rather than modifying it in place -- zero import dependency between
the two, so this file cannot affect the currently-working 2-camera
system. The two coexist until every phase of this redesign is
reviewed and the operator is ready to cut over.

This generalizes DualEMEETCamera's proven threading design (one
capture thread per camera, always-latest-frame semantics, no shared
blocking read) from exactly 2 cameras to a list of N (3 for now) --
see that file's own docstring/history for why this specific design
(MJPG-first fourcc, CAP_V4L2 backend, BUFFERSIZE=1, per-camera
backoff on persistent read failures, and joining capture threads
before releasing VideoCapture objects in stop()) is what it is; every
one of those fixes is preserved here unchanged, just parameterized by
camera index instead of hardcoded left/right.

Camera identification (stable symlinks, not /dev/videoX) — confirmed
working at full 1920x1080 via tools/check_camera.py:
  Cam 1 / N1 — usb-EMEET_EMEET_SmartCam_C960_4K_A241213000400860-video-index0
  Cam 2 / N2 — usb-EMEET_EMEET_SmartCam_C960_4K_A250106000208079-video-index0
  Cam 3 / N3 — usb-EMEET_EMEET_SmartCam_C960_4K_A241217000804000-video-index0

Author : Bright Mensah | NDSU / Imaging System
Path   : /media/pagsun/Transcend/phd_project/emeet_dual_cam/
"""

import cv2
import time
import logging
import threading
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict

# ─────────────────────────────────────────────────────────────
#  CAMERA DEVICE PATHS  (stable symlinks — don't use /dev/videoX)
# ─────────────────────────────────────────────────────────────

CAM1_DEVICE = (
    "/dev/v4l/by-id/"
    "usb-EMEET_EMEET_SmartCam_C960_4K_A241213000400860-video-index0"
)
CAM2_DEVICE = (
    "/dev/v4l/by-id/"
    "usb-EMEET_EMEET_SmartCam_C960_4K_A250106000208079-video-index0"
)
CAM3_DEVICE = (
    "/dev/v4l/by-id/"
    "usb-EMEET_EMEET_SmartCam_C960_4K_A241217000804000-video-index0"
)

CAMERA_DEVICES = [CAM1_DEVICE, CAM2_DEVICE, CAM3_DEVICE]
CAMERA_LABELS  = ["CAM1", "CAM2", "CAM3"]


# ─────────────────────────────────────────────────────────────
#  FRAME TRIPLE
# ─────────────────────────────────────────────────────────────

@dataclass
class FrameTriple:
    """
    One synchronized read from all three cameras. All consumers work
    with FrameTriple objects, not raw ndarray lists -- same design
    intent as DualEMEETCamera's FramePair, generalized to 3 frames.
    """
    frames:         List[object]    # 3x np.ndarray BGR 1920×1080, index = camera index
    frame_id:       int
    timestamps:     List[float]     # time.time() of each camera's capture
    sync_error_ms:  float           # max(timestamps) - min(timestamps), ×1000

    @property
    def sync_ok(self) -> bool:
        """True if all three cameras are in sync within 50 ms."""
        return self.sync_error_ms < 50.0

    def to_meta(self) -> Dict:
        return {
            "frame_id":      self.frame_id,
            "timestamps":    self.timestamps,
            "sync_error_ms": self.sync_error_ms,
            "sync_ok":       self.sync_ok,
        }


# ─────────────────────────────────────────────────────────────
#  TRIPLE EMEET CAMERA
# ─────────────────────────────────────────────────────────────

class TripleEMEETCamera:
    """
    Threaded triple-camera manager for three eMeet SmartCam C960 4K
    cameras -- same usage shape as DualEMEETCamera:

        cam = TripleEMEETCamera()
        cam.start()

        while True:
            triple = cam.read_triple()
            if triple is None:
                continue
            # triple.frames[0], [1], [2] → np.ndarray BGR, Cam1/2/3

        cam.stop()
    """

    # ── Camera setting presets — identical values to
    # DualEMEETCamera's, since it's the same camera model and the
    # same lighting conditions apply regardless of camera count ──
    PRESET_INDOOR = {
        "autofocus":  1,
        "auto_wb":    0,
        "auto_exposure": 1,
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
        "auto_wb":    0,
        "auto_exposure": 1,
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
        "auto_wb":    0,
        "auto_exposure": 1,
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
    DEFAULTS = PRESET_INDOOR

    def __init__(
        self,
        devices:        List[str] = None,
        width:          int  = 1920,
        height:         int  = 1080,
        fps:            int  = 30,
        save_dir:       str  = "captures",
        save_images:    bool = False,
        settings:       dict = None,
        skip_configure: bool = False,
    ):
        """
        devices        : list of 3 device paths, defaults to
                         CAMERA_DEVICES (Cam1/Cam2/Cam3 by-id paths).
        settings       : dict of v4l2 values to apply instead of
                         DEFAULTS. Pass PRESET_OUTDOOR / PRESET_CLOUDY
                         / PRESET_INDOOR or a custom dict.
        skip_configure : if True, skip all v4l2 configuration -- use
                         when cameras are already configured correctly.
        save_images    : if True, save_triple() writes ALL THREE
                         cameras' frames into ONE SHARED save_dir
                         (not per-camera subfolders -- deliberately
                         different from DualEMEETCamera's left/right
                         subfolder split, per the operator's explicit
                         request that training data from all cameras
                         live together in one folder).
        """
        self.devices = list(devices) if devices else list(CAMERA_DEVICES)
        if len(self.devices) != 3:
            raise ValueError(
                f"TripleEMEETCamera requires exactly 3 device paths, "
                f"got {len(self.devices)}")

        self.width       = width
        self.height      = height
        self.fps         = fps
        self.save_images = save_images
        self.save_dir    = Path(save_dir)

        # ONE shared folder for all three cameras -- see save_images
        # docstring note above. Filenames carry the camera identity
        # instead of a subfolder.
        if self.save_images:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Per-camera shared state, indexed by camera index (0,1,2) --
        # lists rather than DualEMEETCamera's named _left_frame/
        # _right_frame attributes, so this generalizes to N cameras
        # without setattr/string-attribute tricks.
        self._frames     : List[Optional[object]] = [None, None, None]
        self._timestamps : List[Optional[float]]  = [None, None, None]
        self._locks       = [threading.Lock() for _ in range(3)]

        self._running    = False
        self._frame_id   = 0
        self._drop_counts = [0, 0, 0]

        self._settings       = {**self.DEFAULTS, **(settings or {})}
        self._skip_configure = skip_configure

        if skip_configure:
            logging.info(
                "TripleEMEETCamera: skipping v4l2 configuration "
                "(using existing camera settings)")
        else:
            s = self._settings
            logging.info(
                f"TripleEMEETCamera: configuring cameras via v4l2-ctl "
                f"(exp={s['exposure']} gamma={s['gamma']} "
                f"wb={s['wb_temp']}) …")
            for device in self.devices:
                self._configure(device)

        logging.info("TripleEMEETCamera: opening VideoCapture ×3 …")
        self._caps = [self._open(d) for d in self.devices]
        self._threads = [None, None, None]
        logging.info("TripleEMEETCamera: ready")

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
        """Apply camera settings via v4l2-ctl using self._settings --
        identical logic to DualEMEETCamera._configure(), just called
        once per camera in a loop instead of twice by name."""
        s = self._settings
        logging.info(f"  Configuring {device}")

        self._v4l2(device, "brightness",             s["brightness"])
        self._v4l2(device, "contrast",               s["contrast"])
        self._v4l2(device, "saturation",              s["saturation"])
        self._v4l2(device, "hue",                    0)
        self._v4l2(device, "gamma",                  s["gamma"])
        self._v4l2(device, "gain",                   s["gain"])
        self._v4l2(device, "sharpness",               s["sharpness"])
        self._v4l2(device, "backlight_compensation", s["backlight"])
        self._v4l2(device, "power_line_frequency",   s["freq"])

        autofocus = s.get("autofocus", 1)
        self._v4l2(device, "focus_automatic_continuous", autofocus)
        if not autofocus:
            time.sleep(0.1)
            self._v4l2(device, "focus_absolute", s["focus"])

        # auto_exposure is a MODE enum (1=manual, 3=auto), not a
        # boolean -- only push an absolute exposure when actually in
        # manual mode. Same rationale as DualEMEETCamera._configure().
        auto_exp = s.get("auto_exposure", 1)
        self._v4l2(device, "auto_exposure", auto_exp)
        if auto_exp == 1:
            time.sleep(0.1)
            self._v4l2(device, "exposure_time_absolute", s["exposure"])

        # White balance -- only force a fixed temperature when auto WB
        # is off, never unconditionally override it. Directly relevant
        # here: three independently-converging auto-WB cameras is what
        # produced the visible warm/neutral color mismatch across Cam
        # 1/2/3 before all three were manually locked to the same
        # temperature.
        auto_wb = s.get("auto_wb", 0)
        self._v4l2(device, "white_balance_automatic", auto_wb)
        if not auto_wb:
            time.sleep(0.1)
            self._v4l2(device, "white_balance_temperature", s["wb_temp"])

    # ── VideoCapture setup ────────────────────────────────────

    def _open(self, src: str) -> cv2.VideoCapture:
        """Open one camera and configure resolution/codec -- identical
        to DualEMEETCamera._open(), including the explicit CAP_V4L2
        backend (avoids OpenCV probing GStreamer first and failing,
        which is harmless but noisy) and BUFFERSIZE=1 (always the
        latest frame, not a queued stale one)."""
        cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS,          self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        if not cap.isOpened():
            raise RuntimeError(f"TripleEMEETCamera: could not open {src}")

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        logging.info(
            f"  Opened {src} → {actual_w}×{actual_h} @ {actual_fps:.0f}fps"
        )
        return cap

    # ── Capture threads ───────────────────────────────────────

    def start(self):
        """Start background capture threads, one per camera."""
        if self._running:
            logging.warning("TripleEMEETCamera.start() called while already running")
            return

        self._running = True
        for i in range(3):
            t = threading.Thread(
                target=self._capture_loop,
                args=(i,),
                daemon=True,
                name=f"emeet-{CAMERA_LABELS[i].lower()}",
            )
            self._threads[i] = t
            t.start()
        logging.info("TripleEMEETCamera: capture threads started (×3)")

    def _capture_loop(self, cam_idx: int):
        """
        Per-camera capture loop — runs in its own thread. Identical
        backoff/logging behavior to DualEMEETCamera._capture_loop(),
        parameterized by camera index instead of hardcoded left/right
        -- see that method's docstring for why this specific backoff
        shape (time-based warning throttle, declare-dead after 100
        consecutive failures, 2s backoff once declared dead) exists:
        confirmed in practice to reach tens of millions of failed
        reads/log lines in a single session without it, when a camera
        physically disconnects.
        """
        cap   = self._caps[cam_idx]
        lock  = self._locks[cam_idx]
        label = CAMERA_LABELS[cam_idx]

        drop_count           = 0
        consecutive_failures = 0
        last_warn_time       = 0.0
        device_declared_dead = False

        while self._running:
            ret, frame = cap.read()
            if ret:
                if consecutive_failures > 0:
                    logging.info(
                        f"TripleEMEETCamera [{label}]: recovered after "
                        f"{consecutive_failures} consecutive failed reads")
                consecutive_failures = 0
                device_declared_dead = False
                with lock:
                    self._frames[cam_idx]     = frame
                    self._timestamps[cam_idx] = time.time()
            else:
                drop_count           += 1
                consecutive_failures += 1
                self._drop_counts[cam_idx] = drop_count

                now = time.time()
                if now - last_warn_time >= 5.0:
                    logging.warning(
                        f"TripleEMEETCamera [{label}]: dropped frame "
                        f"(total drops: {drop_count}, "
                        f"{consecutive_failures} consecutive)"
                    )
                    last_warn_time = now

                if consecutive_failures >= 100 and not device_declared_dead:
                    logging.error(
                        f"TripleEMEETCamera [{label}]: {consecutive_failures} "
                        f"consecutive failed reads -- camera appears to "
                        f"have disconnected. Backing off retries (checking "
                        f"every 2s) instead of spinning at full CPU. Check "
                        f"the USB connection/power to this camera."
                    )
                    device_declared_dead = True

                if device_declared_dead:
                    time.sleep(2.0)
                elif consecutive_failures >= 10:
                    time.sleep(0.1)

    # ── Public API ────────────────────────────────────────────

    def read_triple(self) -> Optional[FrameTriple]:
        """
        Return the latest synchronized frame triple. Returns None if
        any camera hasn't produced a frame yet -- same
        all-or-nothing semantics as DualEMEETCamera.read_pair().
        """
        frames     = [None, None, None]
        timestamps = [None, None, None]
        for i in range(3):
            with self._locks[i]:
                f = self._frames[i]
                frames[i]     = None if f is None else f.copy()
                timestamps[i] = self._timestamps[i]

        if any(f is None for f in frames):
            return None

        sync_error_ms = (max(timestamps) - min(timestamps)) * 1000
        triple = FrameTriple(
            frames        = frames,
            frame_id      = self._frame_id,
            timestamps    = timestamps,
            sync_error_ms = sync_error_ms,
        )
        self._frame_id += 1
        return triple

    def save_triple(self, triple: FrameTriple) -> List[Path]:
        """
        Save all three frames as JPEG into ONE SHARED save_dir --
        filenames carry the frame id first (so all three cameras'
        images from the same capture moment sort together) and the
        camera label, e.g. "000123_CAM1.jpg", "000123_CAM2.jpg",
        "000123_CAM3.jpg" -- deliberately not split into per-camera
        subfolders like DualEMEETCamera.save_pair()'s left/right, per
        the operator's explicit request that training data from all
        cameras live together in one folder this time.

        Only usable when save_images=True.
        """
        if not self.save_images:
            raise RuntimeError("save_triple() called but save_images=False")

        paths = []
        for i in range(3):
            path = self.save_dir / f"{triple.frame_id:06d}_{CAMERA_LABELS[i]}.jpg"
            cv2.imwrite(str(path), triple.frames[i])
            paths.append(path)

        logging.info(
            f"Saved triple {triple.frame_id:06d} "
            f"(sync error: {triple.sync_error_ms:.1f} ms)"
        )
        return paths

    def get_status(self) -> Dict:
        """Return camera health status dict for all three cameras."""
        return {
            "running":    self._running,
            "frame_id":   self._frame_id,
            "cam_ok":     [f is not None for f in self._frames],
            "drop_counts": list(self._drop_counts),
            "resolution": f"{self.width}×{self.height}",
            "fps_target": self.fps,
        }

    def stop(self):
        """
        Stop capture threads and release cameras. Joins each capture
        thread (bounded by a 1.0s timeout) BEFORE releasing its
        VideoCapture object -- calling .release() while another
        thread is still blocked inside .read() on that SAME object is
        a real race in OpenCV's C++ backend that can surface as an
        uncatchable low-level crash during process teardown. Confirmed
        the hard way in DualEMEETCamera's own history; preserved here
        unchanged for all three cameras.
        """
        if not self._running:
            return

        logging.info("TripleEMEETCamera: stopping …")
        self._running = False

        for i, t in enumerate(self._threads):
            if t is not None and t.is_alive():
                t.join(timeout=1.0)
                if t.is_alive():
                    logging.warning(
                        f"TripleEMEETCamera: {t.name} did not exit within "
                        f"1.0s -- releasing camera anyway")

        for cap in self._caps:
            cap.release()
        logging.info("TripleEMEETCamera: stopped")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


# ─────────────────────────────────────────────────────────────
#  QUICK TEST  (python3 triple_emeet_camera.py)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time as _time
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    print("=" * 60)
    print("TripleEMEETCamera — Headless Test")
    print("Captures 5 frame triples, saves triple 0 to captures/")
    print("=" * 60)

    cam = TripleEMEETCamera(
        width=1920, height=1080, fps=30,
        save_dir="captures", save_images=True,
    )
    cam.start()

    _time.sleep(2.0)

    for i in range(5):
        triple = cam.read_triple()
        if triple is None:
            print(f"  Frame {i}: not ready yet")
            _time.sleep(0.2)
            continue
        print(f"  Frame {triple.frame_id}: sync_error={triple.sync_error_ms:.1f}ms "
              f"sync_ok={triple.sync_ok}")
        if i == 0:
            paths = cam.save_triple(triple)
            print(f"    Saved: {[str(p) for p in paths]}")
        _time.sleep(0.2)

    print(cam.get_status())
    cam.stop()
    print("Done.")
