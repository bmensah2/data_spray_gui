"""
tab_analysis_rgb.py
eMeet Dual RGB Detection System — Tab 3: Data Analysis

Drop-in replacement for tab_analysis.py.
Replaces spectral band imagery + NDVI/GNDVI with:
  - Left and right camera live frames (full colour)
  - Per-channel RGB histograms (left camera)
  - Frame sync error + basic statistics

Registered as camera.on_frame_ready callback — same mechanism
as the multispectral AnalysisTab, so wiring in main_gui_rgb.py
is identical.

band_data received = {"left": np.ndarray, "right": np.ndarray}

Author : Nana | NDSU / PhD Imaging System
Path   : /media/pagsun/Transcend/phd_project/emeet_dual_cam/
"""

import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QGroupBox, QCheckBox, QSizePolicy, QPushButton
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

from gui.style import _divider, _muted, _sec, BTN_BLUE
from gui.shared_log import UnifiedLog, LogPanel


# ─────────────────────────────────────────────────────────────
#  ANALYSIS TAB RGB
# ─────────────────────────────────────────────────────────────

class AnalysisTabRGB(QWidget):
    """
    Tab 3 — Data Analysis (RGB version).

    Receives frames via camera.on_frame_ready(frame, bands)
    where bands = {"left": np.ndarray, "right": np.ndarray}.

    Displays:
      Row 1 — Left frame | Right frame  (live, full colour)
      Row 2 — RGB histogram (left)      | Frame statistics
    """

    def __init__(self, camera, parent=None):
        super().__init__(parent)
        self.camera    = camera
        self.log       = UnifiedLog()
        self.band_data = {}

        # Histogram toggle
        self._show_hist  = True
        self._freeze     = False   # freeze display for inspection

        # Stats
        self._frame_count = 0
        self._last_sync_ms = 0.0

        self._build_ui()

        # Register frame callback — same pattern as multispectral AnalysisTab
        camera.on_frame_ready = self._on_frame

        self.log.log("SYS", "Analysis tab (RGB) ready", "ok")

    # ── UI construction ───────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        root.addWidget(_sec("DATA ANALYSIS  —  Dual RGB Camera"))
        root.addWidget(_divider())

        # ── Row 1: Left / Right live frames ───────────────────
        frames_grp = QGroupBox("Live Camera Frames")
        fg = QHBoxLayout(frames_grp)
        fg.setSpacing(8)

        # Left frame
        left_cell = QGroupBox("LEFT Camera")
        ll = QVBoxLayout(left_cell)
        ll.setContentsMargins(4, 4, 4, 4)
        self.lbl_left = QLabel()
        self.lbl_left.setAlignment(Qt.AlignCenter)
        self.lbl_left.setMinimumSize(420, 240)
        self.lbl_left.setStyleSheet(
            "border:1px solid #3a4055;"
            "background-color:#0a0d14;"
            "border-radius:3px;")
        self.lbl_left.setScaledContents(True)
        ll.addWidget(self.lbl_left)
        fg.addWidget(left_cell)

        # Right frame
        right_cell = QGroupBox("RIGHT Camera")
        rl = QVBoxLayout(right_cell)
        rl.setContentsMargins(4, 4, 4, 4)
        self.lbl_right = QLabel()
        self.lbl_right.setAlignment(Qt.AlignCenter)
        self.lbl_right.setMinimumSize(420, 240)
        self.lbl_right.setStyleSheet(
            "border:1px solid #3a4055;"
            "background-color:#0a0d14;"
            "border-radius:3px;")
        self.lbl_right.setScaledContents(True)
        rl.addWidget(self.lbl_right)
        fg.addWidget(right_cell)

        root.addWidget(frames_grp)

        # ── Row 2: Histogram + Statistics ─────────────────────
        bottom = QHBoxLayout()
        bottom.setSpacing(8)

        # ── Histogram (left camera) ────────────────────────────
        hist_grp = QGroupBox("RGB Histogram  (Left Camera)")
        hl = QVBoxLayout(hist_grp)
        hl.setContentsMargins(4, 4, 4, 4)

        hist_ctrl = QHBoxLayout()
        self.chk_hist = QCheckBox("Live Histogram")
        self.chk_hist.setChecked(True)
        self.chk_hist.stateChanged.connect(self._on_hist_toggle)
        hist_ctrl.addWidget(self.chk_hist)
        hist_ctrl.addStretch()
        hl.addLayout(hist_ctrl)

        self.lbl_hist = QLabel()
        self.lbl_hist.setAlignment(Qt.AlignCenter)
        self.lbl_hist.setMinimumSize(340, 160)
        self.lbl_hist.setStyleSheet(
            "border:1px solid #2a2f3d;"
            "background-color:#050810;"
            "border-radius:3px;")
        self.lbl_hist.setScaledContents(True)
        hl.addWidget(self.lbl_hist)

        bottom.addWidget(hist_grp, stretch=2)

        # ── Statistics panel ───────────────────────────────────
        stats_grp = QGroupBox("Frame Statistics")
        sl = QGridLayout(stats_grp)
        sl.setSpacing(4)
        sl.setContentsMargins(8, 8, 8, 8)

        def _stat_row(label, attr, row):
            sl.addWidget(_muted(label), row, 0)
            lbl = QLabel("—")
            lbl.setStyleSheet(
                "color:#60c0a0;font-size:10px;"
                "font-family:Courier New;")
            sl.addWidget(lbl, row, 1)
            setattr(self, attr, lbl)

        _stat_row("Frames:",         "stat_frames",    0)
        _stat_row("Resolution:",     "stat_res",       1)
        _stat_row("Sync error:",     "stat_sync",      2)
        _stat_row("Left mean R:",    "stat_left_r",    3)
        _stat_row("Left mean G:",    "stat_left_g",    4)
        _stat_row("Left mean B:",    "stat_left_b",    5)
        _stat_row("Right mean R:",   "stat_right_r",   6)
        _stat_row("Right mean G:",   "stat_right_g",   7)
        _stat_row("Right mean B:",   "stat_right_b",   8)

        # Freeze button
        self.btn_freeze = QPushButton("❄  Freeze")
        self.btn_freeze.setStyleSheet(BTN_BLUE)
        self.btn_freeze.setFixedHeight(28)
        self.btn_freeze.clicked.connect(self._on_freeze)
        sl.addWidget(self.btn_freeze, 9, 0, 1, 2)

        bottom.addWidget(stats_grp, stretch=1)

        root.addLayout(bottom)
        root.addStretch()

        # ── Log ───────────────────────────────────────────────
        root.addWidget(_divider())
        root.addWidget(
            LogPanel(self.log,
                     sources=["ANALYSIS", "CAMERA", "SYS"],
                     height=90))

    # ── Frame callback ────────────────────────────────────────

    def _on_frame(self, frame, bands):
        """
        Called each frame by DualCameraPanel.
        frame = left camera frame (np.ndarray BGR)
        bands = {"left": np.ndarray, "right": np.ndarray}
        """
        if self._freeze:
            return

        self.band_data = bands
        self._frame_count += 1

        left  = bands.get("left")
        right = bands.get("right")

        if left is None:
            return

        # ── Update left frame display ──────────────────────────
        self._show(self.lbl_left,
                   cv2.cvtColor(left, cv2.COLOR_BGR2RGB))

        # ── Update right frame display ─────────────────────────
        if right is not None:
            self._show(self.lbl_right,
                       cv2.cvtColor(right, cv2.COLOR_BGR2RGB))

        # ── Update histogram ───────────────────────────────────
        if self._show_hist:
            hist_img = self._draw_histogram(left)
            self._show(self.lbl_hist, hist_img)

        # ── Update statistics ──────────────────────────────────
        self._update_stats(left, right)

    # ── Histogram drawing ─────────────────────────────────────

    def _draw_histogram(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw per-channel RGB histogram on a dark background.
        Returns RGB numpy array (for QImage display).
        """
        H, W = 160, 340
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        canvas[:] = (8, 10, 18)   # dark background

        # Grid lines
        for x in range(0, W, W // 4):
            cv2.line(canvas, (x, 0), (x, H),
                     (30, 35, 50), 1)
        for y in range(0, H, H // 4):
            cv2.line(canvas, (0, y), (W, y),
                     (30, 35, 50), 1)

        colors_bgr = [
            (255, 60, 60),    # Blue channel  → blue line
            (60, 200, 60),    # Green channel → green line
            (60, 60, 255),    # Red channel   → red line
        ]
        channel_names = ["B", "G", "R"]

        for ch_idx in range(3):
            hist = cv2.calcHist([frame], [ch_idx], None,
                                [256], [0, 256])
            cv2.normalize(hist, hist, 0, H - 10, cv2.NORM_MINMAX)
            pts = []
            for i in range(256):
                px = int(i * W / 256)
                py = H - 1 - int(hist[i][0])
                pts.append((px, py))

            for i in range(len(pts) - 1):
                cv2.line(canvas, pts[i], pts[i+1],
                         colors_bgr[ch_idx], 1)

            # Channel label
            col = colors_bgr[ch_idx]
            cv2.putText(canvas, channel_names[ch_idx],
                        (W - 30 + ch_idx * 8 - 16, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    # ── Statistics update ─────────────────────────────────────

    def _update_stats(self, left: np.ndarray,
                      right: np.ndarray = None):
        """Update the statistics panel labels."""
        self.stat_frames.setText(str(self._frame_count))

        if left is not None:
            h, w = left.shape[:2]
            self.stat_res.setText(f"{w} × {h}")
            b, g, r = cv2.mean(left)[:3]
            self.stat_left_r.setText(f"{r:.1f}")
            self.stat_left_g.setText(f"{g:.1f}")
            self.stat_left_b.setText(f"{b:.1f}")

            # Colour warnings
            for lbl, val in [(self.stat_left_r, r),
                             (self.stat_left_g, g),
                             (self.stat_left_b, b)]:
                col = "#f5a623" if val < 20 or val > 235 else "#60c0a0"
                lbl.setStyleSheet(
                    f"color:{col};font-size:10px;"
                    "font-family:Courier New;")

        if right is not None:
            b, g, r = cv2.mean(right)[:3]
            self.stat_right_r.setText(f"{r:.1f}")
            self.stat_right_g.setText(f"{g:.1f}")
            self.stat_right_b.setText(f"{b:.1f}")

    # ── Controls ──────────────────────────────────────────────

    def _on_hist_toggle(self, state):
        self._show_hist = (state == Qt.Checked)
        if not self._show_hist:
            self.lbl_hist.clear()
            self.lbl_hist.setText("Histogram disabled")

    def _on_freeze(self):
        self._freeze = not self._freeze
        if self._freeze:
            self.btn_freeze.setText("▶  Unfreeze")
            self.log.log("ANALYSIS", "Display frozen", "info")
        else:
            self.btn_freeze.setText("❄  Freeze")
            self.log.log("ANALYSIS", "Display live", "info")

    # ── Shared image renderer ─────────────────────────────────

    def _show(self, lbl: QLabel, arr: np.ndarray):
        """Render RGB numpy array to a QLabel."""
        try:
            if not lbl.isVisible():
                return
            h, w = arr.shape[:2]
            arr  = np.ascontiguousarray(arr)
            q    = QImage(arr.tobytes(), w, h, w * 3,
                          QImage.Format_RGB888)
            lbl.setPixmap(
                QPixmap.fromImage(q).scaled(
                    lbl.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation))
        except Exception:
            pass

    # ── Cleanup ───────────────────────────────────────────────

    def cleanup(self):
        self.camera.on_frame_ready = None
