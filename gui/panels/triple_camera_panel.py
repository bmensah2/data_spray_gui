"""
gui/panels/triple_camera_panel.py
ABEN Triple RGB Camera Panel

Integration step 1 of 3 for the triple-camera standalone app (see
project notes): the display/control widget wrapping
core/triple_emeet_camera.py's TripleEMEETCamera (Phase 2) and
gui/overlay_rendering_triple.py's stitching/overlay functions
(Phase 4) into a real Qt panel, following the SAME public API shape
as the existing, still-live gui/panels/dual_camera_panel.py
(DualCameraPanel) so a future tab built against this one reads
almost identically to the existing tab_collection.py/tab_detection.py.

Built as a new, self-contained module -- zero import dependency on
dual_camera_panel.py -- so the existing 2-camera GUI is completely
unaffected. This will be wired into a NEW, SEPARATE standalone app
(not into main_gui_rgb.py), so the operator can validate the full
triple-camera system on real hardware with zero risk to the working
2-camera app.

Deliberately scoped for this first pass -- ported the essential flow
(connect/start/stop, live 3-way side-by-side display with detection
overlay, status reporting, camera settings passthrough) and left out
for a later pass:
  - Fullscreen popout dialog (DualCameraPanel's
    open_fullscreen_view()/_FullscreenCameraDialog)
  - Per-channel display modes (Red/Green/Blue) and Left/Right-only
    single-camera views -- less clearly meaningful for 3 separate
    physical cameras than they were for 2 halves of one zone system
  - Zoom
None of these affect whether the core system (camera capture -> zone
decision -> nozzle firing) can be validated on real hardware, which is
the immediate goal.

Public API (mirrors DualCameraPanel where the shape carries over):
  panel.camera_model          → str   e.g. "eMeet C960 4K (Triple)"
  panel.is_acquiring          → bool
  panel.on_detection_overlay  → callable(display_img) → img -- set by
                                the triple-camera Detection tab
  panel.camera_control_bar()  → QWidget   start/stop + status bar
  panel.display_widget()      → QWidget   3-way side-by-side live feed
  panel.get_frame_snapshot()  → (frames: list of 3, meta: dict)
  panel.cleanup()
"""

import cv2
import threading
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

from gui.style import _muted
from gui.theme_manager import theme_manager
from gui.shared_log import UnifiedLog

try:
    from core.triple_emeet_camera import TripleEMEETCamera, FrameTriple
except ImportError:
    from core.triple_emeet_camera import TripleEMEETCamera, FrameTriple

try:
    from gui.overlay_rendering_triple import build_triple_side_by_side
except ImportError:
    from overlay_rendering_triple import build_triple_side_by_side


class TripleCameraPanel:
    """
    Shared triple-RGB camera backend + display widget. Instantiated
    ONCE in the standalone triple-camera app's main window, shared
    between whichever tabs need a live feed -- same intended usage as
    DualCameraPanel.

    Does NOT inherit QWidget — provides widgets via camera_control_bar()
    and display_widget(), same pattern as DualCameraPanel.
    """

    camera_model:        str  = "eMeet C960 4K (Triple)"
    is_acquiring:        bool = False
    on_frame_ready              = None   # callable(frames: list, meta: dict)
    on_detection_overlay        = None   # callable(img) → img

    # GenICam image acquisition object — None disables GenICam controls,
    # mirrors DualCameraPanel's same compatibility attribute.
    ia = None

    def __init__(self, shared_log: UnifiedLog):
        self.shared_log = shared_log

        # Latest frames — written by QTimer, read by display
        self._frames: list = [None, None, None]
        self._frame_lock = threading.Lock()

        self._camera: TripleEMEETCamera = None

        self._display_lbls:     list = []
        self._start_btns:       list = []
        self._status_lbls:      list = []

        self._display_timer = QTimer()
        self._display_timer.timeout.connect(self._refresh_display)

        self.shared_log.log("CAMERA", "TripleCameraPanel ready", "ok")

    # ── Connection ────────────────────────────────────────────

    def connect(self):
        """Open all three cameras and start capture threads."""
        if self._camera is not None:
            self.shared_log.log("CAMERA", "Already connected", "warn")
            return True
        try:
            self.shared_log.log(
                "CAMERA", "Connecting eMeet cameras (x3) …", "info")
            self._camera = TripleEMEETCamera()
            self.camera_model = "eMeet C960 4K (Triple)"
            self._update_ctrl_bar()
            self.shared_log.log(
                "CAMERA", "eMeet cameras connected ✓", "ok")
            return True
        except Exception as e:
            self.shared_log.log(
                "CAMERA", f"Connection failed: {e}", "error")
            self._camera = None
            return False

    def start(self):
        """Start live capture and display refresh."""
        if self._camera is None:
            ok = self.connect()
            if not ok:
                return
        try:
            self._camera.start()
            self.is_acquiring = True
            self._display_timer.start(33)   # ~30 fps display refresh
            self._update_ctrl_bar()
            self.shared_log.log("CAMERA", "Acquisition started", "ok")
        except Exception as e:
            self.shared_log.log("CAMERA", f"Start failed: {e}", "error")

    def stop(self):
        """Stop capture and display."""
        self._display_timer.stop()
        if self._camera is not None:
            self._camera.stop()
            self._camera = None
        self.is_acquiring = False
        self._update_ctrl_bar()
        for lbl in self._display_lbls:
            try:
                lbl.clear()
                lbl.setText("Camera stopped")
            except RuntimeError:
                pass
        self.shared_log.log("CAMERA", "Acquisition stopped", "ok")

    # ── Frame acquisition loop ────────────────────────────────

    def _refresh_display(self):
        """
        Called every 33ms by QTimer (GUI thread). Reads the latest
        triple, updates display, fires callbacks -- same overall flow
        as DualCameraPanel._refresh_display(), generalized to 3
        frames.
        """
        if self._camera is None:
            return

        triple = self._camera.read_triple()
        if triple is None:
            return

        with self._frame_lock:
            self._frames = [f.copy() for f in triple.frames]

        if self.on_frame_ready:
            try:
                self.on_frame_ready(self._frames, triple.to_meta())
            except Exception as e:
                self.shared_log.log(
                    "CAMERA", f"on_frame_ready error: {e}", "warn")

        disp_w = self._largest_display_width()
        disp_img, panel_w = build_triple_side_by_side(
            triple.frames, target_width=disp_w)
        # Stashed for on_detection_overlay callers that need to know
        # the per-panel width used, to convert zone pixel boundaries
        # correctly (see gui/overlay_rendering_triple.py's
        # draw_triple_detection_overlay() panel_width argument).
        self.last_panel_width = panel_w

        if self.on_detection_overlay:
            try:
                disp_img = self.on_detection_overlay(disp_img)
            except Exception:
                pass

        self._show(disp_img)

    def _largest_display_width(self, default=1280):
        """
        Largest width among all currently-visible registered display
        labels -- same rationale as DualCameraPanel's
        _largest_display_size(): build the combined frame at the size
        of whatever's actually showing it, rather than an arbitrary
        fixed size, so it looks sharp regardless of window size.
        """
        best_w = 0
        for lbl in self._display_lbls:
            try:
                if not lbl.isVisible():
                    continue
                w = lbl.size().width()
                if w > 10 and w > best_w:
                    best_w = w
            except RuntimeError:
                pass
        return best_w if best_w >= 10 else default

    def _show(self, img: np.ndarray):
        """Render BGR numpy array to ALL registered display QLabels."""
        if not self._display_lbls:
            return
        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            rgb  = np.ascontiguousarray(rgb)
            q    = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q)
            for lbl in self._display_lbls:
                try:
                    sz = lbl.size()
                    if sz.width() > 10 and sz.height() > 10:
                        scaled = pixmap.scaled(
                            sz, Qt.KeepAspectRatio, Qt.FastTransformation)
                    else:
                        scaled = pixmap
                    lbl.setPixmap(scaled)
                except RuntimeError:
                    pass
        except Exception:
            pass

    # ── Public API ────────────────────────────────────────────

    def get_frame_snapshot(self):
        """
        Thread-safe copy of the latest frame triple.
        Returns (frames: list of 3 np.ndarray or None).
        """
        with self._frame_lock:
            return [f.copy() if f is not None else None for f in self._frames]

    # ── Widgets ───────────────────────────────────────────────

    def camera_control_bar(self) -> QWidget:
        """
        Returns the camera toolbar widget (status only, for this
        first pass -- no fullscreen/view-mode controls yet, see
        module docstring). Start/Stop lives as a single global button
        elsewhere in the app, same convention as DualCameraPanel.
        """
        bar = QWidget()
        bar.setFixedHeight(44)
        theme_manager.register_widget(bar, lambda p: f"background-color:{p['bg0']};")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(6)

        lay.addWidget(_muted("Triple RGB View — Cam1 / Cam2 / Cam3"))
        lay.addStretch()

        status_lbl = QLabel("Disconnected")
        theme_manager.register_widget(
            status_lbl, lambda p: (
                f"color:{p['muted']};font-size:9px;"
                f"font-family:'Noto Sans',Arial,sans-serif;"))
        lay.addWidget(status_lbl)
        self._status_lbls.append(status_lbl)

        self._update_ctrl_bar()
        return bar

    def display_widget(self) -> QWidget:
        """Returns the live feed display widget."""
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)

        lbl = QLabel("Camera not started")
        lbl.setAlignment(Qt.AlignCenter)
        theme_manager.register_widget(
            lbl, lambda p: (
                f"background-color:{p['bg0']};"
                f"color:{p['border']};"
                f"font-family:'Noto Sans',Arial,sans-serif;"
                f"font-size:11px;"
                f"border:1px solid {p['input_bg']};"))
        lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lbl.setMinimumSize(1, 1)
        lay.addWidget(lbl)
        self._display_lbls.append(lbl)
        return w

    def remove_display_widget(self, container: QWidget):
        """Unregister a display widget so it stops receiving updates."""
        lbl = container.findChild(QLabel)
        if lbl is not None and lbl in self._display_lbls:
            self._display_lbls.remove(lbl)

    def toggle_start_stop(self):
        if self.is_acquiring:
            self.stop()
        else:
            self.start()

    def _update_ctrl_bar(self):
        """Refresh ALL registered buttons and status labels."""
        connected = self._camera is not None
        acquiring = self.is_acquiring

        for btn in self._start_btns:
            try:
                if acquiring:
                    btn.setText("⏹  STOP CAMERAS")
                    theme_manager.register_button(btn, "red")
                else:
                    btn.setText("▶  START CAMERAS")
                    theme_manager.register_button(btn, "green")
            except RuntimeError:
                pass

        for lbl in self._status_lbls:
            try:
                if acquiring:
                    lbl.setText(f"● LIVE  —  {self.camera_model}")
                    theme_manager.register_widget(
                        lbl, lambda p: (
                            f"color:{p['blue']};font-size:9px;"
                            f"font-family:'Noto Sans',Arial,sans-serif;"
                            f"font-weight:bold;"))
                elif connected:
                    lbl.setText(f"Connected  —  {self.camera_model}")
                    theme_manager.register_widget(
                        lbl, lambda p: (
                            f"color:{p['green']};font-size:9px;"
                            f"font-family:'Noto Sans',Arial,sans-serif;"))
                else:
                    lbl.setText("Disconnected")
                    theme_manager.register_widget(
                        lbl, lambda p: (
                            f"color:{p['muted']};font-size:9px;"
                            f"font-family:'Noto Sans',Arial,sans-serif;"))
            except RuntimeError:
                pass

    # ── Public v4l2 access (used by camera settings dialogs) ──

    def set_v4l2(self, device: str, control: str, value) -> bool:
        """Set a v4l2 control on one of the eMeet cameras."""
        if self._camera is not None:
            return self._camera._v4l2(device, control, value)
        import subprocess
        cmd = ["v4l2-ctl", "-d", device, "-c", f"{control}={value}"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return r.returncode == 0

    # ── Cleanup ───────────────────────────────────────────────

    def cleanup(self):
        """Called by the standalone app's MainWindow.closeEvent()."""
        self._display_timer.stop()
        if self._camera is not None:
            try:
                self._camera.stop()
            except Exception:
                pass
            self._camera = None
        self.is_acquiring   = False
        self.on_frame_ready = None
        self._start_btns    = []
        self._status_lbls   = []
        self._display_lbls  = []
