#!/usr/bin/env python3
"""
realsense_camera.py
Field Imaging System — Intel RealSense Depth Camera Module

Provides:
  RealSenseCamera   — producer-consumer backend (mirrors CameraPanel design)
  RealSensePanel    — PyQt5 display widget for embedding in GUI
  run_standalone()  — standalone viewer (replaces the basic script)

Features:
  - Color stream   : 640×480 BGR @ 30fps
  - Depth stream   : 640×480 Z16 @ 30fps
  - Depth colormap : JET visualization
  - Aligned frames : depth aligned to color frame
  - Distance query : metric depth at any pixel (x, y) → metres
  - Snapshot save  : color + depth + metadata → timestamped files
  - Thread-safe    : producer thread → GUI consumer timer pattern
  - Simulation mode: random frames when no camera connected

Usage — standalone:
    python realsense_camera.py

Usage — embedded in GUI:
    from realsense_camera import RealSensePanel
    panel = RealSensePanel(shared_log)
    layout.addWidget(panel.display_widget())

Author : Bright Mensah  |  Field Imaging System
"""

import time
import threading
import warnings
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Tuple

warnings.filterwarnings("ignore", category=ResourceWarning)

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel,
        QPushButton, QComboBox, QGroupBox, QGridLayout,
        QSizePolicy, QApplication, QLineEdit, QDoubleSpinBox,
        QSpinBox, QProgressBar
    )
    from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject
    from PyQt5.QtGui import QImage, QPixmap
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False

# Capture-session and video-recording support is fully self-contained
# in this file (see _get_session_dir/_save_capture and
# _SimpleDualStreamRecorder below) -- it used to import
# core.acquisition_manager / core.acquisition_config / core.video_recorder,
# none of which exist anywhere in this repo (leftover references from
# a pre-RGB-pivot architecture). Rebuilt using the same proven
# approach gui/panels/acquisition_panel_rgb.py already uses
# successfully for the eMeet cameras: plain cv2.imwrite/cv2.VideoWriter
# and a simple session-directory convention, rather than depending on
# a more elaborate multispectral-oriented manager class that doesn't
# even fit a single RGB-D camera well.
import json

# Reuse the SAME theme system used by the main GUI
from gui.theme_manager import theme_manager
from gui.style import _muted


# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────
# ── Standalone mode (full quality) ───────────────────────────
COLOR_W_FULL, COLOR_H_FULL, COLOR_FPS_FULL = 640, 480, 30
DEPTH_W_FULL, DEPTH_H_FULL, DEPTH_FPS_FULL = 640, 480, 30

# ── Embedded mode ──────────────────
COLOR_W_EMB, COLOR_H_EMB, COLOR_FPS_EMB = 480, 270, 15
DEPTH_W_EMB, DEPTH_H_EMB, DEPTH_FPS_EMB = 480, 270, 15

DEPTH_ALPHA  = 0.03    # convertScaleAbs factor for JET colormap
DISPLAY_FPS  = 6       # GUI update rate when embedded
SAVE_PATH    = Path("recorded_videos/realsense")

# ── Structured capture / recording output  ──
CAPTURE_BASE = Path("acquired_data/realsense")   # manual + auto captures
VIDEO_BASE   = SAVE_PATH                          # video sessions
VIDEO_FPS_DEFAULT = 15.0


# ─────────────────────────────────────────────────────────────
#  FRAME CONTAINER
# ─────────────────────────────────────────────────────────────
@dataclass
class RealSenseFrame:
    """Thread-safe container for one captured frame pair."""
    color:        np.ndarray = field(default_factory=lambda: np.zeros((480,640,3), np.uint8))
    depth_raw:    np.ndarray = field(default_factory=lambda: np.zeros((480,640), np.uint16))
    depth_colormap: np.ndarray = field(default_factory=lambda: np.zeros((480,640,3), np.uint8))
    timestamp:    float = 0.0

    def distance_at(self, x: int, y: int,
                    radius: int = 3) -> float:
        """
        Return metric distance (metres) at pixel (x, y).
        Averages over a small radius to reduce noise.
        Returns 0.0 if invalid.
        """
        h, w = self.depth_raw.shape
        x1 = max(0, x - radius)
        x2 = min(w, x + radius + 1)
        y1 = max(0, y - radius)
        y2 = min(h, y + radius + 1)
        patch = self.depth_raw[y1:y2, x1:x2].astype(np.float32)
        valid = patch[patch > 0]
        if len(valid) == 0:
            return 0.0
        # RealSense depth unit is 0.001m (1mm per unit) by default
        return float(np.median(valid)) * 0.001


# ─────────────────────────────────────────────────────────────
#  REALSENSE CAMERA BACKEND
# ─────────────────────────────────────────────────────────────
class RealSenseCamera:
    """
    Producer-consumer camera backend.
    Producer thread fetches frames from RealSense SDK.
    Consumer (GUI timer) reads the safe buffer.
    """

    def __init__(self):
        self._lock             = threading.Lock()
        self._frame            = RealSenseFrame()
        self._safe_frame       = RealSenseFrame()
        self._producer_running = False
        self._producer_thread  = None
        self._pipeline         = None
        self._align            = None
        self.is_acquiring      = False
        self.camera_model      = None
        self._frame_count      = 0
        self._fps              = 0.0
        self._last_fps_t       = time.time()
        self._sim_mode         = False
        # Default dimensions (overridden in start())
        self._cw, self._ch, self._cfps = COLOR_W_FULL, COLOR_H_FULL, COLOR_FPS_FULL
        self._dw, self._dh, self._dfps = DEPTH_W_FULL, DEPTH_H_FULL, DEPTH_FPS_FULL

    # ── Lifecycle ─────────────────────────────────────────────

    def start(self, embedded: bool = False) -> Tuple[bool, str]:
        """
        Start camera acquisition.
        embedded=True  → 424×240 @ 6fps  (msCAM also running)
        embedded=False → 640×480 @ 30fps  (standalone)
        Returns (ok, message).
        """
        if self.is_acquiring:
            return True, "Already running"

        self._embedded = embedded
        if embedded:
            cw, ch, cfps = COLOR_W_EMB, COLOR_H_EMB, COLOR_FPS_EMB
            dw, dh, dfps = DEPTH_W_EMB, DEPTH_H_EMB, DEPTH_FPS_EMB
        else:
            cw, ch, cfps = COLOR_W_FULL, COLOR_H_FULL, COLOR_FPS_FULL
            dw, dh, dfps = DEPTH_W_FULL, DEPTH_H_FULL, DEPTH_FPS_FULL

        # Store for use in _producer_loop and _sim_frame
        self._cw, self._ch, self._cfps = cw, ch, cfps
        self._dw, self._dh, self._dfps = dw, dh, dfps

        if not REALSENSE_AVAILABLE:
            self._start_sim()
            mode = "embedded" if embedded else "standalone"
            return True, f"Simulation mode ({mode})"

        try:
            self._pipeline = rs.pipeline()
            cfg            = rs.config()
            cfg.enable_stream(
                rs.stream.color,
                cw, ch, rs.format.bgr8, cfps)
            cfg.enable_stream(
                rs.stream.depth,
                dw, dh, rs.format.z16, dfps)

            profile = self._pipeline.start(cfg)

            # Get device info
            dev = profile.get_device()
            self.camera_model = dev.get_info(rs.camera_info.name)

            # Align depth to color frame
            self._align = rs.align(rs.stream.color)

            self._start_producer()
            self.is_acquiring = True
            return True, f"Connected: {self.camera_model}"

        except Exception as e:
            self._pipeline = None
            self._start_sim()
            return False, f"Camera error: {e} — simulation active"

    def stop(self):
        """Stop acquisition and release hardware."""
        self._producer_running = False
        time.sleep(0.3)
        if self._pipeline:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
        self.is_acquiring = False

    # ── Producer thread ───────────────────────────────────────

    def _start_producer(self):
        self._producer_running = True
        self._producer_thread  = threading.Thread(
            target=self._producer_loop, daemon=True)
        self._producer_thread.start()

    def _producer_loop(self):
        while self._producer_running:
            try:
                if self._sim_mode:
                    self._sim_frame()
                    time.sleep(1.0 / self._cfps)
                    continue

                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                aligned = self._align.process(frames)

                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                color = np.asanyarray(color_frame.get_data())
                depth = np.asanyarray(depth_frame.get_data())

                # Depth colormap for visualization
                depth_cm = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth, alpha=DEPTH_ALPHA),
                    cv2.COLORMAP_JET)

                frame = RealSenseFrame(
                    color=color.copy(),
                    depth_raw=depth.copy(),
                    depth_colormap=depth_cm,
                    timestamp=time.time()
                )

                # Update FPS
                self._frame_count += 1
                now = time.time()
                dt  = now - self._last_fps_t
                if dt >= 1.0:
                    self._fps = self._frame_count / dt
                    self._frame_count = 0
                    self._last_fps_t  = now

                with self._lock:
                    self._safe_frame = frame

            except Exception as e:
                if self._producer_running:
                    time.sleep(0.05)

    def _start_sim(self):
        self._sim_mode        = True
        self.camera_model     = "Simulation"
        self._start_producer()
        self.is_acquiring     = True

    def _sim_frame(self):
        """Generate synthetic color + depth for testing."""
        h, w = self._ch, self._cw
        # Synthetic color — gradient + noise
        color = np.zeros((h, w, 3), np.uint8)
        color[:, :, 0] = np.linspace(40, 80, w, dtype=np.uint8)
        color[:, :, 2] = np.linspace(20, 60, h, dtype=np.uint8).reshape(-1, 1)
        noise = np.random.randint(-15, 15, (h, w, 3), dtype=np.int8)
        color = np.clip(color.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Synthetic depth — smooth gradient (500–2000mm)
        depth = np.zeros((h, w), np.uint16)
        for y in range(h):
            depth[y, :] = np.linspace(500, 2000, w, dtype=np.uint16) + \
                          np.random.randint(0, 50, w, dtype=np.uint16)

        depth_cm = cv2.applyColorMap(
            cv2.convertScaleAbs(depth, alpha=DEPTH_ALPHA),
            cv2.COLORMAP_JET)

        frame = RealSenseFrame(
            color=color, depth_raw=depth,
            depth_colormap=depth_cm,
            timestamp=time.time()
        )
        with self._lock:
            self._safe_frame = frame

    # ── Thread-safe access ────────────────────────────────────

    def get_frame(self) -> RealSenseFrame:
        """Get a thread-safe copy of the latest frame."""
        with self._lock:
            f = self._safe_frame
            return RealSenseFrame(
                color=f.color.copy(),
                depth_raw=f.depth_raw.copy(),
                depth_colormap=f.depth_colormap.copy(),
                timestamp=f.timestamp
            )

    def get_distance(self, x: int, y: int,
                     radius: int = 3) -> float:
        """Get metric distance (m) at pixel (x, y)."""
        with self._lock:
            return self._safe_frame.distance_at(x, y, radius)

    @property
    def fps(self) -> float:
        return self._fps

    # ── Snapshot save ─────────────────────────────────────────

    def save_snapshot(self, prefix: str = "snap") -> Path:
        """
        Save color + depth + metadata to disk.
        Returns path to saved files.
        """
        frame = self.get_frame()
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        SAVE_PATH.mkdir(parents=True, exist_ok=True)
        base  = SAVE_PATH / f"{prefix}_{ts}"

        cv2.imwrite(str(base) + "_color.jpg",
                    frame.color,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        np.save(str(base) + "_depth.npy", frame.depth_raw)
        cv2.imwrite(str(base) + "_depth_colormap.jpg",
                    frame.depth_colormap,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Metadata text
        with open(str(base) + "_meta.txt", "w") as mf:
            mf.write(f"timestamp:    {ts}\n")
            mf.write(f"camera:       {self.camera_model}\n")
            mf.write(f"color_shape:  {frame.color.shape}\n")
            mf.write(f"depth_shape:  {frame.depth_raw.shape}\n")
            mf.write(f"depth_min_m:  "
                     f"{frame.depth_raw[frame.depth_raw>0].min()*0.001:.3f}\n"
                     if frame.depth_raw.any() else "depth_min_m: N/A\n")
            mf.write(f"depth_max_m:  "
                     f"{frame.depth_raw.max()*0.001:.3f}\n")

        return base


# ─────────────────────────────────────────────────────────────
#  SIMPLE DUAL-STREAM VIDEO RECORDER
# ─────────────────────────────────────────────────────────────

class _SimpleDualStreamRecorder:
    """
    Self-contained video recorder for RealSense color + depth streams.
    """

    def __init__(self, output_dir: Path, fps: float, resolution):
        self.session_id  = datetime.now().strftime("rs_vid_%Y%m%d_%H%M%S")
        self.session_dir = output_dir / self.session_id
        self.fps         = fps
        self.resolution  = resolution   # (width, height)
        self.total_frames = 0
        self._color_writer = None
        self._depth_writer = None
        self._start_time   = None

    def start_recording(self, labels: dict = None) -> bool:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w, h   = self.resolution
        self._color_writer = cv2.VideoWriter(
            str(self.session_dir / "color.mp4"), fourcc, self.fps, (w, h))
        self._depth_writer = cv2.VideoWriter(
            str(self.session_dir / "depth_colormap.mp4"),
            fourcc, self.fps, (w, h))

        if not self._color_writer.isOpened() or not self._depth_writer.isOpened():
            self._color_writer = None
            self._depth_writer = None
            return False

        if labels:
            with open(self.session_dir / "labels.json", "w") as f:
                json.dump(labels, f, indent=2)

        self._start_time = time.time()
        return True

    def write_frame(self, color: np.ndarray, depth_colormap: np.ndarray):
        if self._color_writer is not None:
            self._color_writer.write(color)
        if self._depth_writer is not None:
            self._depth_writer.write(depth_colormap)
        self.total_frames += 1

    def stop_recording(self) -> dict:
        duration = (time.time() - self._start_time
                    if self._start_time else 0.0)
        if self._color_writer is not None:
            self._color_writer.release()
            self._color_writer = None
        if self._depth_writer is not None:
            self._depth_writer.release()
            self._depth_writer = None
        return {
            "session_id":       self.session_id,
            "total_frames":     self.total_frames,
            "duration_seconds": duration,
        }


# ─────────────────────────────────────────────────────────────
#  REALSENSE PANEL (PyQt5 widget)
# ─────────────────────────────────────────────────────────────
if QT_AVAILABLE:

    class _FrameWorker(QObject):
        """
        Runs in a QThread — fetches frames from RealSenseCamera,
        pre-renders them to QImage, and emits a signal.
        The main thread only does setPixmap() — zero scaling work.
        """
        frame_ready = pyqtSignal(QImage, float, float, str)
        # QImage, fps, center_dist, view_name

        def __init__(self, camera, view_ref, click_ref):
            super().__init__()
            self.camera    = camera
            self.view_ref  = view_ref  # callable → str
            self.click_ref = click_ref # callable → (x,y) or None
            self._running  = True

        def run(self):
            while self._running:
                t0 = time.time()
                try:
                    frame = self.camera.get_frame()
                    if frame.color is None:
                        time.sleep(0.033)
                        continue

                    view = self.view_ref()

                    # Build display image (off main thread)
                    if view == "Color":
                        img = frame.color.copy()
                    elif view == "Depth (JET)":
                        img = frame.depth_colormap.copy()
                    elif view == "Side by Side":
                        img = np.hstack(
                            (frame.color, frame.depth_colormap))
                    elif view == "Depth Overlay":
                        mask = (frame.depth_raw > 0).astype(np.uint8)
                        over = cv2.bitwise_and(
                            frame.depth_colormap,
                            frame.depth_colormap, mask=mask)
                        img = cv2.addWeighted(
                            frame.color, 0.6, over, 0.4, 0)
                    else:
                        img = frame.color.copy()

                    # Draw click crosshair (off main thread)
                    click = self.click_ref()
                    if click:
                        cx, cy, cd = click
                        cv2.drawMarker(img, (cx, cy),
                            (0,255,255), cv2.MARKER_CROSS, 20, 2)
                        cv2.putText(img, f"{cd:.3f}m",
                            (cx+8, cy-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0,255,255), 1, cv2.LINE_AA)

                    # Center marker
                    ch, cw = img.shape[:2]
                    cv2.drawMarker(img, (cw//2, ch//2),
                        (255,255,255), cv2.MARKER_CROSS, 12, 1)
                    center_dist = frame.distance_at(
                        frame.color.shape[1]//2,
                        frame.color.shape[0]//2)

                    # Convert BGR → RGB QImage (off main thread)
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    rgb = np.ascontiguousarray(rgb)
                    h, w = rgb.shape[:2]
                    qimg = QImage(
                        rgb.tobytes(), w, h, w*3,
                        QImage.Format_RGB888).copy()
                    # .copy() detaches from numpy buffer

                    self.frame_ready.emit(
                        qimg, self.camera.fps,
                        center_dist, view)

                except Exception:
                    pass

                # Target fps — subtract processing time
                target_fps = getattr(
                    self.camera, '_embedded', False) and 15 or 30
                elapsed = time.time() - t0
                sleep   = max(0.001, (1.0/target_fps) - elapsed)
                time.sleep(sleep)

        def stop(self):
            self._running = False


    class RealSensePanel:
        """
        PyQt5 display panel for RealSense camera.
        Mirrors CameraPanel architecture — call display_widget()
        to get an embeddable QWidget.
        """

        VIEWS = [
            "Color",
            "Depth (JET)",
            "Side by Side",
            "Depth Overlay",
        ]

        def __init__(self, shared_log=None, embedded: bool = False):
            self.shared_log  = shared_log
            self._embedded   = embedded
            self.camera      = RealSenseCamera()
            self._view       = "Color"
            self._click_dist = 0.0
            self._click_x    = -1
            self._click_y    = -1
            self._snapshot_count = 0

            # ── Structured capture state (mirrors AcquisitionPanelRGB) ──
            self._session_id       = None
            self._session_labels   = None   # (subject, notes) key
            self._session_dir_path = None
            self._capture_count    = 0
            self._img_count        = 0

            # ── Auto capture state ────────────────────────────────
            self._auto_timer = None
            self._auto_count = 0
            self._auto_max   = 0

            # ── Video recording state ─────────────────────────────
            self._video_rec       = None
            self._recording        = False
            self._video_timer      = None   # drives frame writes at config.fps
            self._depth_raw_dir    = None
            self._depth_raw_idx    = 0

            self._build_widget()
            self._start()

        # ── Widget builder ────────────────────────────────────

        def _build_widget(self):
            from PyQt5.QtWidgets import QSplitter

            self._widget = QWidget()
            outer = QHBoxLayout(self._widget)
            outer.setContentsMargins(4, 4, 4, 4)
            outer.setSpacing(0)

            splitter = QSplitter(Qt.Horizontal)
            splitter.setHandleWidth(3)
            theme_manager.register_widget(
                splitter, lambda p: (
                    f"QSplitter::handle{{background-color:{p['border2']};}}"))
            outer.addWidget(splitter)

            # ── LEFT: capture / recording controls ──────────────
            left_w = QWidget()
            left_w.setMinimumWidth(300)
            left_w.setMaximumWidth(360)
            llay = QVBoxLayout(left_w)
            llay.setContentsMargins(0, 0, 4, 0)
            llay.setSpacing(6)
            llay.addWidget(self._info_grp())
            llay.addWidget(self._capture_grp())
            llay.addStretch()
            splitter.addWidget(left_w)

            # ── MIDDLE: camera feed ──────────────────────────────
            mid_w = QWidget()
            mid_w.setMinimumWidth(420)
            mlay = QVBoxLayout(mid_w)
            mlay.setContentsMargins(4, 0, 4, 0)
            mlay.setSpacing(4)

            # Header bar
            hdr = QHBoxLayout()
            self._status_lbl = _muted("Initializing...", size=10)
            hdr.addWidget(self._status_lbl, stretch=1)

            hdr.addWidget(QLabel("View:"))
            self._view_combo = QComboBox()
            self._view_combo.addItems(self.VIEWS)
            self._view_combo.currentTextChanged.connect(
                lambda v: setattr(self, '_view', v))
            hdr.addWidget(self._view_combo)

            self._btn_snap = QPushButton("📷 Snapshot")
            self._btn_snap.setFixedWidth(95)
            self._btn_snap.clicked.connect(self._take_snapshot)
            hdr.addWidget(self._btn_snap)

            self._btn_toggle = QPushButton("Stop")
            self._btn_toggle.setFixedWidth(55)
            self._btn_toggle.clicked.connect(self._toggle)
            hdr.addWidget(self._btn_toggle)

            mlay.addLayout(hdr)

            # Image display
            self._img_lbl = QLabel()
            self._img_lbl.setAlignment(Qt.AlignCenter)
            self._img_lbl.setSizePolicy(
                QSizePolicy.Expanding, QSizePolicy.Expanding)
            theme_manager.register_widget(
                self._img_lbl, lambda p: (
                    f"border:2px solid {p['border']};"
                    f"background-color:{p['bg0']};border-radius:4px;"))
            self._img_lbl.mousePressEvent = self._on_click
            mlay.addWidget(self._img_lbl, stretch=1)

            # Stats row
            stats = QHBoxLayout()
            self._fps_lbl   = _muted("FPS: --", size=9)
            self._dist_lbl  = _muted("Distance: -- m", size=9)
            self._snap_lbl  = _muted("Snapshots: 0", size=9)
            for lbl in [self._fps_lbl, self._dist_lbl, self._snap_lbl]:
                stats.addWidget(lbl)
            stats.addStretch()
            mlay.addLayout(stats)

            splitter.addWidget(mid_w)

            # ── RIGHT: reserved for future use ───────────────────
            right_w = QWidget()
            right_w.setMinimumWidth(200)
            right_w.setMaximumWidth(280)
            rlay = QVBoxLayout(right_w)
            rlay.setContentsMargins(4, 0, 0, 0)
            future_grp = QGroupBox("Future")
            fg = QVBoxLayout(future_grp)
            future_lbl = QLabel("Reserved for\nfuture application")
            theme_manager.register_widget(
                future_lbl, lambda p: (
                    f"color:{p['border']};font-size:11px;"
                    f"font-family:'Noto Sans',Arial,sans-serif;"))
            future_lbl.setAlignment(Qt.AlignCenter)
            fg.addStretch()
            fg.addWidget(future_lbl)
            fg.addStretch()
            rlay.addWidget(future_grp, stretch=1)
            splitter.addWidget(right_w)

            splitter.setStretchFactor(0, 0)
            splitter.setStretchFactor(1, 1)
            splitter.setStretchFactor(2, 0)

        # ── Depth Info widget ───────────────────────────────────

        def _info_grp(self) -> QGroupBox:
            info_grp = QGroupBox("Depth Info")
            ig = QGridLayout(info_grp)
            ig.setSpacing(3)

            self._lbl_model   = QLabel("—")
            self._lbl_click   = QLabel("Click image to measure distance")
            self._lbl_center  = QLabel("—")

            for row, (lb, attr) in enumerate([
                ("Camera:", "_lbl_model"),
                ("Click point:", "_lbl_click"),
                ("Center depth:", "_lbl_center"),
            ]):
                l = QLabel(lb)
                theme_manager.register_widget(
                    l, lambda p: (
                        f"color:{p['muted']};font-family:'Noto Sans',Arial,sans-serif;"
                        f"font-size:9px;"))
                v = getattr(self, attr)
                theme_manager.register_widget(
                    v, lambda p: (
                        f"color:{p['muted2']};font-family:'Noto Sans',Arial,sans-serif;"
                        f"font-size:9px;"))
                ig.addWidget(l, row, 0)
                ig.addWidget(v, row, 1)

            return info_grp

        # ── Capture & Recording widget ─────────────────────────

        def _capture_grp(self) -> QGroupBox:
            grp = QGroupBox("Capture & Recording")
            lay = QVBoxLayout(grp)
            lay.setSpacing(5)

            # Labels row
            lbl_row = QHBoxLayout()
            lbl_row.addWidget(QLabel("Subject:"))
            self._entry_subject = QLineEdit()
            self._entry_subject.setPlaceholderText(
                "e.g. canopy height row 5")
            lbl_row.addWidget(self._entry_subject, stretch=1)
            lay.addLayout(lbl_row)

            notes_row = QHBoxLayout()
            notes_row.addWidget(QLabel("Notes:"))
            self._entry_notes = QLineEdit()
            self._entry_notes.setPlaceholderText("optional notes")
            notes_row.addWidget(self._entry_notes, stretch=1)
            lay.addLayout(notes_row)

            # ── Manual capture ──────────────────────────────────
            man_row = QHBoxLayout()
            btn_capture = QPushButton("📷  CAPTURE")
            theme_manager.register_button(btn_capture, "green")
            btn_capture.clicked.connect(self._capture_image)
            man_row.addWidget(btn_capture)
            self._lbl_img_count = _muted("Captures: 0", size=9)
            man_row.addWidget(self._lbl_img_count)
            lay.addLayout(man_row)

            # ── Auto capture ─────────────────────────────────────
            auto_grp = QGroupBox("Auto Capture")
            ag = QGridLayout(auto_grp)
            ag.addWidget(QLabel("Interval (s):"), 0, 0)
            self._spn_interval = QDoubleSpinBox()
            self._spn_interval.setRange(0.5, 60.0)
            self._spn_interval.setValue(2.0)
            self._spn_interval.setSingleStep(0.5)
            ag.addWidget(self._spn_interval, 0, 1)
            ag.addWidget(QLabel("Max:"), 0, 2)
            self._spn_max_cap = QSpinBox()
            self._spn_max_cap.setRange(1, 9999)
            self._spn_max_cap.setValue(50)
            ag.addWidget(self._spn_max_cap, 0, 3)

            self._auto_progress = QProgressBar()
            self._auto_progress.setRange(0, 100)
            self._auto_progress.setValue(0)
            self._auto_progress.setFormat("Ready")
            ag.addWidget(self._auto_progress, 1, 0, 1, 4)

            auto_btn_row = QHBoxLayout()
            self._btn_auto_start = QPushButton("▶ START")
            theme_manager.register_button(self._btn_auto_start, "green")
            self._btn_auto_start.clicked.connect(self._auto_start)
            auto_btn_row.addWidget(self._btn_auto_start)
            self._btn_auto_pause = QPushButton("⏸ PAUSE")
            theme_manager.register_button(self._btn_auto_pause, "amber")
            self._btn_auto_pause.setEnabled(False)
            self._btn_auto_pause.clicked.connect(self._auto_pause)
            auto_btn_row.addWidget(self._btn_auto_pause)
            self._btn_auto_stop = QPushButton("⏹ STOP")
            theme_manager.register_button(self._btn_auto_stop, "dim_red")
            self._btn_auto_stop.setEnabled(False)
            self._btn_auto_stop.clicked.connect(self._auto_stop)
            auto_btn_row.addWidget(self._btn_auto_stop)
            ag.addLayout(auto_btn_row, 2, 0, 1, 4)
            lay.addWidget(auto_grp)

            # ── Video recording ───────────────────────────────────
            vid_grp = QGroupBox("Video Recording  (color + JET depth + raw 16-bit depth)")
            vg = QGridLayout(vid_grp)
            vg.addWidget(QLabel("FPS:"), 0, 0)
            self._spn_vid_fps = QDoubleSpinBox()
            self._spn_vid_fps.setRange(1.0, 30.0)
            self._spn_vid_fps.setValue(VIDEO_FPS_DEFAULT)
            vg.addWidget(self._spn_vid_fps, 0, 1)

            self._lbl_vid_status = _muted("Not recording", size=9)
            vg.addWidget(self._lbl_vid_status, 1, 0, 1, 2)

            vid_btn_row = QHBoxLayout()
            self._btn_vid_start = QPushButton("⏺ START REC")
            theme_manager.register_button(self._btn_vid_start, "red")
            self._btn_vid_start.clicked.connect(self._vid_start)
            vid_btn_row.addWidget(self._btn_vid_start)
            self._btn_vid_stop = QPushButton("⏹ STOP REC")
            theme_manager.register_button(self._btn_vid_stop, "dim_red")
            self._btn_vid_stop.setEnabled(False)
            self._btn_vid_stop.clicked.connect(self._vid_stop)
            vid_btn_row.addWidget(self._btn_vid_stop)
            vg.addLayout(vid_btn_row, 2, 0, 1, 2)
            lay.addWidget(vid_grp)

            return grp

        def display_widget(self) -> QWidget:
            """Return embeddable QWidget."""
            return self._widget

        # ── Start / stop ──────────────────────────────────────

        def _start(self):
            ok, msg = self.camera.start(embedded=self._embedded)
            mode_tag = " [6fps embedded]" if self._embedded else " [30fps]"
            self._status_lbl.setText(msg + mode_tag)
            if self.camera.camera_model:
                self._lbl_model.setText(
                    self.camera.camera_model + mode_tag)
            if ok:
                self._btn_toggle.setText("Stop")
                self._start_worker()
            self._log(msg + mode_tag)

        def _start_worker(self):
            """Start the off-thread frame worker."""
            self._worker = _FrameWorker(
                self.camera,
                view_ref=lambda: self._view,
                click_ref=self._get_click
            )
            self._thread = QThread()
            self._worker.moveToThread(self._thread)
            self._thread.started.connect(self._worker.run)
            self._worker.frame_ready.connect(self._on_frame_ready)
            self._thread.start()

        def _stop_worker(self):
            """Stop the off-thread frame worker cleanly."""
            if hasattr(self, '_worker'):
                self._worker.stop()
            if hasattr(self, '_thread'):
                self._thread.quit()
                self._thread.wait(2000)

        def _toggle(self):
            if self.camera.is_acquiring:
                self._stop_worker()
                self.camera.stop()
                self._btn_toggle.setText("Start")
                self._status_lbl.setText("Stopped")
            else:
                self._start()

        # ── Labels ──────────────────────────────────────────────

        def _get_labels(self) -> dict:
            return {
                "subject": self._entry_subject.text().strip(),
                "notes":   self._entry_notes.text().strip(),
                "source":  "realsense",
            }

        # ── Structured capture session (mirrors AcquisitionPanelRGB) ─

        def _get_session_dir(self, labels: dict) -> Path:
            """
            (Re)creates/returns the session directory for the current
            labels. If the labels have changed since the last capture, a new
            session directory is created. The directory structure is:
            CAPTURE_BASE / session_id / color
            CAPTURE_BASE / session_id / depth_colormap
            CAPTURE_BASE / session_id / depth_raw
            CAPTURE_BASE / session_id / metadata
            """
            key = (labels.get("subject", ""), labels.get("notes", ""))
            if (self._session_labels == key
                    and self._session_dir_path is not None
                    and self._session_dir_path.exists()):
                return self._session_dir_path

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = f"rs_{ts}"
            base = CAPTURE_BASE / session_id
            for sub in ("color", "depth_colormap", "depth_raw", "metadata"):
                (base / sub).mkdir(parents=True, exist_ok=True)

            self._session_id       = session_id
            self._session_labels   = key
            self._session_dir_path = base
            self._capture_count    = 0
            self._log(f"New capture session: {session_id}")
            return base

        def reset_session(self):
            """Force a new capture session on next capture."""
            self._session_labels   = None
            self._session_dir_path = None

        def _save_capture(self, frame, labels: dict) -> dict:
            """
            Save one capture (color + JET depth + full-precision 16-bit
            raw depth + metadata) to the current session directory.
            Self-contained replacement for the old
            core.acquisition_manager.DataAcquisitionManager.acquire()
            call -- same file layout, no external dependency.
            cv2.imwrite supports 16-bit single-channel PNG directly,
            so the raw depth save is fully lossless.
            """
            session_dir = self._get_session_dir(labels)
            cid = f"{self._session_id}_{self._capture_count:04d}"
            ts  = datetime.now().isoformat()

            cv2.imwrite(str(session_dir / "color" / f"{cid}_color.jpg"),
                        frame.color, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(session_dir / "depth_colormap" /
                             f"{cid}_depth_cm.jpg"),
                        frame.depth_colormap, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(session_dir / "depth_raw" / f"{cid}_depth16.png"),
                        frame.depth_raw)

            meta = {
                "capture_id":   cid,
                "timestamp":    ts,
                "labels":       labels,
                "camera_model": self.camera.camera_model or "RealSense",
                "color_shape":  list(frame.color.shape),
                "depth_shape":  list(frame.depth_raw.shape),
            }
            with open(session_dir / "metadata" / f"{cid}_meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            self._capture_count += 1
            return meta

        # ── Manual capture ────────────────────────────────────────

        def _capture_image(self):
            frame  = self.camera.get_frame()
            labels = self._get_labels()
            try:
                meta = self._save_capture(frame, labels)
                self._img_count += 1
                self._lbl_img_count.setText(
                    f"Captures: {self._img_count}")
                self._log(f"Captured: {meta['capture_id']}")
            except Exception as e:
                self._log(f"Capture error: {e}")

        # ── Auto capture ──────────────────────────────────────────

        def _auto_start(self):
            labels = self._get_labels()
            self._auto_count = 0
            self._auto_max   = self._spn_max_cap.value()
            interval_ms       = int(self._spn_interval.value() * 1000)

            def _do_capture():
                if self._auto_count >= self._auto_max:
                    self._auto_stop()
                    self._log(
                        f"Auto complete: {self._auto_count}/{self._auto_max}")
                    return
                frame = self.camera.get_frame()
                lbl = labels.copy()
                lbl["auto_n"] = self._auto_count
                try:
                    self._save_capture(frame, lbl)
                    self._auto_count += 1
                    pct = int(self._auto_count / self._auto_max * 100)
                    self._auto_progress.setValue(pct)
                    self._auto_progress.setFormat(
                        f"{self._auto_count}/{self._auto_max}")
                except Exception as e:
                    self._log(f"Auto capture error: {e}")

            self._auto_timer = QTimer()
            self._auto_timer.timeout.connect(_do_capture)
            self._auto_timer.start(interval_ms)
            self._btn_auto_start.setEnabled(False)
            self._btn_auto_pause.setEnabled(True)
            self._btn_auto_stop.setEnabled(True)
            self._log(
                f"Auto capture started: {self._auto_max} @ "
                f"{self._spn_interval.value()}s")

        def _auto_pause(self):
            if self._auto_timer:
                if self._auto_timer.isActive():
                    self._auto_timer.stop()
                    self._log("Auto capture paused")
                else:
                    self._auto_timer.start()
                    self._log("Auto capture resumed")

        def _auto_stop(self):
            if self._auto_timer:
                self._auto_timer.stop()
                self._auto_timer = None
            self._btn_auto_start.setEnabled(True)
            self._btn_auto_pause.setEnabled(False)
            self._btn_auto_stop.setEnabled(False)
            self._auto_progress.setValue(0)
            self._auto_progress.setFormat("Ready")

        # ── Video recording ──────────────────────────────────────
        # Uses _SimpleDualStreamRecorder for the two standard 8-bit
        # streams (color, JET depth). Raw 16-bit depth (which no
        # standard video codec can hold) is separately dumped as a
        # per-frame PNG sequence alongside the video -- unchanged from
        # before, this part never depended on the missing modules.

        def _vid_start(self):
            frame = self.camera.get_frame()
            try:
                fps = self._spn_vid_fps.value()
                h, w = frame.color.shape[:2]
                labels = self._get_labels()
                self._video_rec = _SimpleDualStreamRecorder(
                    output_dir=VIDEO_BASE, fps=fps, resolution=(w, h))
                if self._video_rec.start_recording(labels=labels):
                    session_dir = self._video_rec.session_dir
                    self._depth_raw_dir = session_dir / "depth_raw"
                    self._depth_raw_dir.mkdir(
                        parents=True, exist_ok=True)
                    self._depth_raw_idx = 0

                    self._recording = True
                    self._video_timer = QTimer()
                    self._video_timer.timeout.connect(
                        self._write_video_frame)
                    self._video_timer.start(int(1000 / fps))

                    self._btn_vid_start.setEnabled(False)
                    self._btn_vid_stop.setEnabled(True)
                    self._lbl_vid_status.setText(
                        f"Recording: {self._video_rec.session_id}")
                    theme_manager.register_widget(
                        self._lbl_vid_status, lambda p: (
                            f"color:{p['red']};font-family:'Noto Sans',Arial,sans-serif;"
                            f"font-size:9px;"))
                    self._log(
                        f"Video started: {self._video_rec.session_id} "
                        f"@ {fps:.0f}fps (color + JET depth + raw16 depth)")
                else:
                    self._log("Video error: VideoWriter failed to open "
                               "(check codec availability / disk space)")
                    self._video_rec = None
            except Exception as e:
                self._log(f"Video error: {e}")

        def _write_video_frame(self):
            if not self._recording or self._video_rec is None:
                return
            frame = self.camera.get_frame()
            self._video_rec.write_frame(
                color=frame.color, depth_colormap=frame.depth_colormap)
            try:
                path = (self._depth_raw_dir /
                       f"frame_{self._depth_raw_idx:06d}.png")
                cv2.imwrite(str(path), frame.depth_raw)
                self._depth_raw_idx += 1
            except Exception as e:
                self._log(f"Depth raw frame error: {e}")

        def _vid_stop(self):
            if not self._recording:
                return
            if self._video_timer:
                self._video_timer.stop()
                self._video_timer = None
            try:
                meta = self._video_rec.stop_recording()
                self._recording = False
                self._video_rec = None
                self._btn_vid_start.setEnabled(True)
                self._btn_vid_stop.setEnabled(False)
                self._lbl_vid_status.setText("Not recording")
                theme_manager.register_widget(
                    self._lbl_vid_status, lambda p: (
                        f"color:{p['muted']};font-family:'Noto Sans',Arial,sans-serif;"
                        f"font-size:9px;"))
                if meta:
                    self._log(
                        f"Video saved: {meta['total_frames']}fr "
                        f"{meta['duration_seconds']:.1f}s  "
                        f"+ {self._depth_raw_idx} raw16 depth frames")
            except Exception as e:
                self._log(f"Stop error: {e}")

        def _get_click(self):
            """Return click info for worker thread."""
            if self._click_x >= 0:
                return (self._click_x,
                        self._click_y,
                        self._click_dist)
            return None

        # ── Frame receiver (main thread — only setPixmap) ─────

        def _on_frame_ready(self, qimg: QImage,
                            fps: float, center_dist: float,
                            view: str):
            """
            Called in main thread via signal/slot.
            Heavy work (cv2, numpy) is already done in worker.
            Only pixmap scaling + label update here.
            """
            try:
                if not self._img_lbl.isVisible():
                    return
                self._img_lbl.setPixmap(
                    QPixmap.fromImage(qimg).scaled(
                        self._img_lbl.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation))
                self._fps_lbl.setText(f"FPS: {fps:.1f}")
                self._lbl_center.setText(f"{center_dist:.3f} m")
            except Exception:
                pass

        def _show(self, arr: np.ndarray):
            """Legacy helper — kept for compatibility."""
            try:
                rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                rgb  = np.ascontiguousarray(rgb)
                q    = QImage(rgb.tobytes(), w, h, w*3,
                              QImage.Format_RGB888)
                self._img_lbl.setPixmap(
                    QPixmap.fromImage(q).scaled(
                        self._img_lbl.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation))
            except Exception:
                pass

        # ── Click to measure ──────────────────────────────────

        def _on_click(self, event):
            """Convert click position → camera pixel → distance."""
            lbl_w = self._img_lbl.width()
            lbl_h = self._img_lbl.height()

            # Determine actual image dimensions in the label
            frame = self.camera.get_frame()
            if self._view == "Side by Side":
                img_w = frame.color.shape[1] * 2
            else:
                img_w = frame.color.shape[1]
            img_h = frame.color.shape[0]

            # Scale factor (keep aspect ratio)
            scale = min(lbl_w / img_w, lbl_h / img_h)
            disp_w = int(img_w * scale)
            disp_h = int(img_h * scale)
            off_x  = (lbl_w - disp_w) // 2
            off_y  = (lbl_h - disp_h) // 2

            px = int((event.x() - off_x) / scale)
            py = int((event.y() - off_y) / scale)

            # For side-by-side, only left (color) half is meaningful
            if self._view == "Side by Side":
                if px > frame.color.shape[1]:
                    return
            px = max(0, min(px, frame.color.shape[1]-1))
            py = max(0, min(py, frame.color.shape[0]-1))

            dist = frame.distance_at(px, py)
            self._click_x    = px
            self._click_y    = py
            self._click_dist = dist
            self._dist_lbl.setText(f"Distance: {dist:.3f} m")
            self._lbl_click.setText(
                f"({px}, {py})  →  {dist:.3f} m")
            self._log(f"Depth at ({px},{py}): {dist:.3f}m")

        # ── Snapshot ──────────────────────────────────────────

        def _take_snapshot(self):
            try:
                base = self.camera.save_snapshot(prefix="aben_rs")
                self._snapshot_count += 1
                self._snap_lbl.setText(
                    f"Snapshots: {self._snapshot_count}")
                self._log(f"Snapshot saved: {base.name}")
            except Exception as e:
                self._log(f"Snapshot error: {e}")

        # ── Log ───────────────────────────────────────────────

        def _log(self, msg: str):
            if self.shared_log:
                self.shared_log.log("CAMERA", msg, "info")
            else:
                print(f"[REALSENSE] {msg}")

        # ── Cleanup ───────────────────────────────────────────

        def cleanup(self):
            if self._auto_timer:
                self._auto_timer.stop()
                self._auto_timer = None
            if self._recording:
                self._vid_stop()
            self._stop_worker()
            self.camera.stop()


# ─────────────────────────────────────────────────────────────
#  POPUP WINDOW (for embedding in main GUI)
# ─────────────────────────────────────────────────────────────
if QT_AVAILABLE:
    from PyQt5.QtWidgets import QDialog, QVBoxLayout

    class RealSenseWindow(QDialog):
        """
        Popup window for RealSense viewer.
        Launched from main GUI header — floats independently.

        Usage:
            win = RealSenseWindow(parent=self)
            win.show()   # non-blocking — stays open
        """

        def __init__(self, parent=None, embedded: bool = True):
            super().__init__(parent)
            self._embedded = embedded
            self.setWindowTitle(
                "ABEN — RealSense Depth Camera"
                + (" [Embedded 424×240@6fps]" if embedded else
                   " [Standalone 640×480@30fps]"))
            self.setMinimumSize(1100, 680)
            self.resize(1300, 750)
            self.setWindowFlags(
                Qt.Window |
                Qt.WindowMinimizeButtonHint |
                Qt.WindowMaximizeButtonHint |
                Qt.WindowCloseButtonHint)

            # Same theme as the main GUI (loaded from the shared
            # theme_config.json — apply() handles load() at the
            # QApplication level in run_standalone()).
            self.setStyleSheet(theme_manager.style())

            lay = QVBoxLayout(self)
            lay.setContentsMargins(6, 6, 6, 6)

            # Mode info bar
            from PyQt5.QtWidgets import QHBoxLayout as _HBox
            mode_bar = QWidget()
            mb = _HBox(mode_bar)
            mb.setContentsMargins(4, 2, 4, 2)
            if self._embedded:
                mode_lbl = QLabel(
                    "⚡ Embedded mode: 480×270 @ 15fps "
                    "(Bus 02 cameras-only layout — optimized)")
                theme_manager.register_widget(
                    mode_lbl, lambda p: (
                        f"color:{p['amber']};font-family:'Noto Sans',Arial,sans-serif;"
                        f"font-size:9px;"))
                mb.addWidget(mode_lbl)
                mb.addStretch()
                # Button to restart in full quality
                btn_full = QPushButton("Switch to Full 640×480@30fps")
                theme_manager.register_button(btn_full, "blue")
                btn_full.clicked.connect(self._switch_full)
                mb.addWidget(btn_full)
            else:
                mode_lbl = QLabel(
                    "✓ Standalone mode: 640×480 @ 30fps")
                theme_manager.register_widget(
                    mode_lbl, lambda p: (
                        f"color:{p['green']};font-family:'Noto Sans',Arial,sans-serif;"
                        f"font-size:9px;"))
                mb.addWidget(mode_lbl)
            lay.addWidget(mode_bar)

            self.panel = RealSensePanel(embedded=self._embedded)
            lay.addWidget(self.panel.display_widget())

        def _switch_full(self):
            """Restart in full quality mode — user accepts lag risk."""
            self.panel.cleanup()
            self._embedded = False
            self.setWindowTitle("ABEN — RealSense Depth Camera [Standalone 640×480@30fps]")
            self.panel._embedded = False
            self.panel.camera = RealSenseCamera()
            self.panel._start()

        def closeEvent(self, event):
            self.panel.cleanup()
            event.accept()


# ─────────────────────────────────────────────────────────────
#  STANDALONE VIEWER
# ─────────────────────────────────────────────────────────────
def run_standalone():
    """
    Standalone RealSense viewer.
    Replaces the basic script with a proper Qt-based viewer.
    """
    if not QT_AVAILABLE:
        # Fallback: basic OpenCV viewer (original script style)
        print("[REALSENSE] PyQt5 not available — running basic viewer")
        print("Press Q to quit")
        cam = RealSenseCamera()
        ok, msg = cam.start()
        print(f"[REALSENSE] {msg}")
        try:
            while True:
                frame = cam.get_frame()
                combined = np.hstack(
                    (frame.color, frame.depth_colormap))
                cv2.putText(
                    combined,
                    f"FPS:{cam.fps:.1f}  "
                    f"Center:{frame.distance_at(320,240):.2f}m",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255,255,255), 1)
                cv2.imshow("ABEN RealSense — Color + Depth", combined)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cam.stop()
            cv2.destroyAllWindows()
        return

    import sys
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")

    # Load the theme last chosen in the main GUI (shared via
    # gui/theme_config.json on disk) and apply it BEFORE building
    # any widgets, so theme_manager.palette() reflects it.
    theme_manager.load()
    theme_manager.apply(theme_manager.current, app=app, save=False)

    from PyQt5.QtWidgets import QMainWindow
    win = QMainWindow()
    win.setWindowTitle("ABEN RealSense Depth Camera Viewer")
    win.setMinimumSize(1100, 680)
    win.resize(1300, 750)

    panel = RealSensePanel()
    win.setCentralWidget(panel.display_widget())
    win.show()

    def on_close(event):
        panel.cleanup()
        event.accept()
    win.closeEvent = on_close

    sys.exit(app.exec_())


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_standalone()