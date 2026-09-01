#!/usr/bin/env python3
"""
realsense_camera.py
ABEN Field Imaging System — Intel RealSense Depth Camera Module

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

Author : Nana  |  ABEN PhD Imaging System
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
        QSizePolicy, QApplication
    )
    from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject
    from PyQt5.QtGui import QImage, QPixmap
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────
# ── Standalone mode (full quality) ───────────────────────────
COLOR_W_FULL, COLOR_H_FULL, COLOR_FPS_FULL = 640, 480, 30
DEPTH_W_FULL, DEPTH_H_FULL, DEPTH_FPS_FULL = 640, 480, 30

# ── Embedded mode (when msCAM also running) ──────────────────
# USB layout (confirmed optimal 2026-04-30):
#   Bus 02 → Realtek hub → Port 1: msCAM  (~630 MB/s)
#   Bus 02 → Realtek hub → Port 4: nested hub → RealSense
#   SSD moved to Bus 01 — Bus 02 is cameras only
#
# Bandwidth budget:
#   msCAM        : ~630 MB/s  (continuous)
#   RealSense 15fps: ~87 MB/s (480×270 RGB+Depth)
#   Total        : ~717 MB/s ← within hub capacity
COLOR_W_EMB, COLOR_H_EMB, COLOR_FPS_EMB = 480, 270, 15
DEPTH_W_EMB, DEPTH_H_EMB, DEPTH_FPS_EMB = 480, 270, 15

DEPTH_ALPHA  = 0.03    # convertScaleAbs factor for JET colormap
DISPLAY_FPS  = 6       # GUI update rate when embedded
SAVE_PATH    = Path("recorded_videos/realsense")


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
            self._build_widget()
            self._start()

        # ── Widget builder ────────────────────────────────────

        def _build_widget(self):
            self._widget = QWidget()
            lay = QVBoxLayout(self._widget)
            lay.setContentsMargins(4, 4, 4, 4)
            lay.setSpacing(4)

            # ── Header bar ────────────────────────────────────
            hdr = QHBoxLayout()

            self._status_lbl = QLabel("Initializing...")
            self._status_lbl.setStyleSheet(
                "color:#8090a8;font-family:'Noto Sans',Arial,sans-serif;font-size:10px;")
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

            lay.addLayout(hdr)

            # ── Image display ─────────────────────────────────
            self._img_lbl = QLabel()
            self._img_lbl.setAlignment(Qt.AlignCenter)
            self._img_lbl.setSizePolicy(
                QSizePolicy.Expanding, QSizePolicy.Expanding)
            self._img_lbl.setStyleSheet(
                "border:2px solid #3a4055;"
                "background-color:#0a0d14;border-radius:4px;")
            self._img_lbl.mousePressEvent = self._on_click
            lay.addWidget(self._img_lbl, stretch=1)

            # ── Stats row ─────────────────────────────────────
            stats = QHBoxLayout()
            self._fps_lbl   = QLabel("FPS: --")
            self._dist_lbl  = QLabel("Distance: -- m")
            self._snap_lbl  = QLabel("Snapshots: 0")
            for lbl in [self._fps_lbl, self._dist_lbl, self._snap_lbl]:
                lbl.setStyleSheet(
                    "color:#8090a8;font-family:'Noto Sans',Arial,sans-serif;"
                    "font-size:9px;")
                stats.addWidget(lbl)
            stats.addStretch()
            lay.addLayout(stats)

            # ── Info group ────────────────────────────────────
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
                l.setStyleSheet(
                    "color:#6070a0;font-family:'Noto Sans',Arial,sans-serif;"
                    "font-size:9px;")
                v = getattr(self, attr)
                v.setStyleSheet(
                    "color:#a0b8d0;font-family:'Noto Sans',Arial,sans-serif;"
                    "font-size:9px;")
                ig.addWidget(l, row, 0)
                ig.addWidget(v, row, 1)

            lay.addWidget(info_grp)

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
            self.setMinimumSize(820, 640)
            self.resize(900, 680)
            self.setWindowFlags(
                Qt.Window |
                Qt.WindowMinimizeButtonHint |
                Qt.WindowMaximizeButtonHint |
                Qt.WindowCloseButtonHint)

            # Match main GUI dark theme
            self.setStyleSheet("""
                QDialog  { background-color:#0f1117; color:#e8eaf0; }
                QGroupBox{
                    border:1px solid #2a2f3d; border-radius:4px;
                    margin-top:8px; color:#a0b0c0;
                    font-family:'Noto Sans',Arial,sans-serif; font-size:10px; }
                QGroupBox::title{
                    subcontrol-origin:margin; padding:0 4px; }
                QLabel  { font-family:'Noto Sans',Arial,sans-serif; }
                QPushButton{
                    background:#1a2030; color:#e8eaf0;
                    border:1px solid #2a3050; border-radius:4px;
                    padding:4px 8px; font-family:'Noto Sans',Arial,sans-serif; }
                QPushButton:hover{ background:#2a3040; }
                QComboBox{
                    background:#1a2030; color:#e8eaf0;
                    border:1px solid #3a4055; border-radius:3px;
                    font-family:'Noto Sans',Arial,sans-serif; font-size:10px; }
            """)

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
                mode_lbl.setStyleSheet(
                    "color:#f5a623;font-family:'Noto Sans',Arial,sans-serif;"
                    "font-size:9px;")
                mb.addWidget(mode_lbl)
                mb.addStretch()
                # Button to restart in full quality
                btn_full = QPushButton("Switch to Full 640×480@30fps")
                btn_full.setStyleSheet(
                    "QPushButton{background:#1a2030;color:#60c0ff;"
                    "border:1px solid #2a5080;border-radius:3px;"
                    "padding:2px 8px;font-family:'Noto Sans',Arial,sans-serif;"
                    "font-size:9px;}"
                    "QPushButton:hover{background:#2a3040;}")
                btn_full.clicked.connect(self._switch_full)
                mb.addWidget(btn_full)
            else:
                mode_lbl = QLabel(
                    "✓ Standalone mode: 640×480 @ 30fps")
                mode_lbl.setStyleSheet(
                    "color:#00c896;font-family:'Noto Sans',Arial,sans-serif;"
                    "font-size:9px;")
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
    app.setStyleSheet("""
        QWidget  { background-color: #0f1117; color: #e8eaf0; }
        QGroupBox{ border:1px solid #2a2f3d; border-radius:4px;
                   margin-top:8px; color:#a0b0c0;
                   font-family:'Noto Sans',Arial,sans-serif; font-size:10px; }
        QGroupBox::title { subcontrol-origin:margin; padding:0 4px; }
        QLabel   { font-family: 'Noto Sans', Arial, sans-serif; }
        QPushButton {
            background:#1a2030; color:#e8eaf0;
            border:1px solid #2a3050; border-radius:4px;
            padding:4px 8px; font-family:'Noto Sans',Arial,sans-serif; }
        QPushButton:hover { background:#2a3040; }
        QComboBox {
            background:#1a2030; color:#e8eaf0;
            border:1px solid #3a4055; border-radius:3px;
            font-family:'Noto Sans',Arial,sans-serif; font-size:10px; }
    """)

    from PyQt5.QtWidgets import QMainWindow
    win = QMainWindow()
    win.setWindowTitle("ABEN RealSense Depth Camera Viewer")
    win.setMinimumSize(800, 620)

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