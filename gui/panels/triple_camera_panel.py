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
for a later pass: Zoom (note: DualCameraPanel itself has no zoom
feature either, so there was never anything to match here).
None of these affect whether the core system (camera capture -> zone
decision -> nozzle firing) can be validated on real hardware, which is
the immediate goal.

UPDATE (Group A pass): fullscreen popout and per-camera view modes
are now implemented, matching DualCameraPanel's own feature set as
closely as the physical difference allows -- see DISPLAY_MODES below
and open_fullscreen_view(). Per-channel (Red/Green/Blue) views are
deliberately NOT ported: they operated on a single camera's frame in
the 2-camera system, and don't have a clean equivalent across 3
separate physical cameras (which camera's channel would it show?).
"Cam 1/2/3 Only" single-camera views replace them instead.

Public API (mirrors DualCameraPanel where the shape carries over):
  panel.camera_model          → str   e.g. "eMeet C960 4K (Triple)"
  panel.is_acquiring          → bool
  panel.on_detection_overlay  → callable(display_img) → img -- set by
                                the triple-camera Detection tab
  panel.detection_tab_ref     → DetectionPanelTriple or None -- set by
                                the standalone app so the fullscreen
                                popout can show its own Arm/Stop/E-Stop
                                bar (see open_fullscreen_view())
  panel.camera_control_bar()  → QWidget   view mode + fullscreen + status
  panel.display_widget()      → QWidget   3-way side-by-side live feed
  panel.get_frame_snapshot()  → (frames: list of 3, meta: dict)
  panel.open_fullscreen_view(parent) → QDialog
  panel.cleanup()
"""

import cv2
import threading
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSizePolicy, QDialog, QComboBox, QPushButton,
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

# Per-camera single views replace DualCameraPanel's Left/Right Only
# (there are 3 physical cameras here, not 2 halves of one system) --
# Red/Green/Blue Channel views are dropped entirely, see module
# docstring for why.
DISPLAY_MODES = [
    "Side by Side",
    "Cam 1 Only",
    "Cam 2 Only",
    "Cam 3 Only",
]


class _FullscreenCameraDialogTriple(QDialog):
    """
    Fullscreen popout for the live triple-camera feed (see
    TripleCameraPanel.open_fullscreen_view()).

    Defined as a REAL QDialog subclass with keyPressEvent as a genuine
    class method -- not assigned as an instance-level function
    attribute. Overriding a Qt virtual event handler by setting
    `dlg.keyPressEvent = some_function` is a known PyQt5 pitfall:
    SIP's C++-to-Python virtual dispatch ("catcher") doesn't reliably
    recognize an instance-attribute override the way it does a
    genuine subclass method override, and can raise `TypeError:
    invalid argument to sipBadCatcherResult()` at runtime. A real
    subclass avoids that entirely (same lesson DualCameraPanel's own
    _FullscreenCameraDialog documents).
    """

    def __init__(self, camera_panel, parent=None):
        super().__init__(parent)
        self._camera_panel = camera_panel
        self._display_widget = None

    def register_cleanup(self, display_widget):
        self._display_widget = display_widget

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        if self._display_widget is not None:
            self._camera_panel.remove_display_widget(self._display_widget)
        event.accept()

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
        self._disp_mode_combos: list = []   # all view-mode combos, tracked
                                             # so a change from ANY one
                                             # (embedded tab or fullscreen)
                                             # is reflected everywhere,
                                             # same convention as
                                             # DualCameraPanel's own list
        # Set by the standalone app (main_gui_triple.py) so the
        # fullscreen popout can show its own Arm/Stop/E-Stop bar --
        # None until then means fullscreen just shows video with no
        # detection controls.
        self.detection_tab_ref = None

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

        # View mode is applied AFTER the overlay, not instead of
        # building the combined image -- draw_triple_detection_
        # overlay() only knows how to draw zone boundaries/detections
        # across the full 3-panel layout (it needs panel_w to know
        # where each camera's section starts), so a single-camera
        # view is built by cropping the already-overlaid combined
        # image, not by skipping the overlay step. This means "Cam 2
        # Only" still shows N2's zone boundary and any live
        # detections, not a plain uncomposited feed.
        mode = "Side by Side"
        if self._disp_mode_combos:
            mode = self._disp_mode_combos[0].currentText()
        if mode in ("Cam 1 Only", "Cam 2 Only", "Cam 3 Only"):
            idx = int(mode.split()[1]) - 1
            x0 = idx * (panel_w + 4)
            x1 = min(x0 + panel_w, disp_img.shape[1])
            crop = disp_img[:, x0:x1]
            if crop.size > 0:
                target_h = int(disp_w / (1920 / 1080))
                disp_img = cv2.resize(crop, (disp_w, max(1, target_h)))

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

    @property
    def current_frame(self):
        """
        Compatibility property for gui/panels/spray_panel.py and any
        other shared panel that checks `camera.current_frame is not
        None` -- same role as DualCameraPanel's identically-named
        property. Returns the first camera's latest frame (Cam 1), or
        None if not yet acquired; any one frame being present is a
        reasonable "is the camera system acquiring" signal for that
        check, which doesn't need all 3 specifically.
        """
        with self._frame_lock:
            return self._frames[0].copy() if self._frames[0] is not None else None

    def get_frame_snapshot(self):
        """
        Thread-safe copy of the latest frame triple.
        Returns (frames: list of 3 np.ndarray or None).
        """
        with self._frame_lock:
            return [f.copy() if f is not None else None for f in self._frames]

    # ── Widgets ───────────────────────────────────────────────

    def camera_control_bar(self, fullscreen_dialog=None) -> QWidget:
        """
        Returns the camera toolbar widget (view mode + fullscreen +
        status). Start/Stop lives as a single global button elsewhere
        in the app, same convention as DualCameraPanel.

        fullscreen_dialog: pass the QDialog this bar is being
        embedded INSIDE (only from open_fullscreen_view() itself) so
        the Fullscreen button becomes a Restore button that closes
        that specific dialog, instead of opening another nested
        fullscreen dialog on top of the current one.
        """
        bar = QWidget()
        bar.setFixedHeight(44)
        theme_manager.register_widget(bar, lambda p: f"background-color:{p['bg0']};")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(6)

        lay.addWidget(_muted("View:"))
        disp_combo = QComboBox()
        disp_combo.addItems(DISPLAY_MODES)
        disp_combo.setFixedHeight(28)
        disp_combo.setMinimumWidth(120)
        theme_manager.register_widget(
            disp_combo, lambda p: (
                f"QComboBox{{background:{p['input_bg']};color:{p['text']};"
                f"border:1px solid {p['border']};border-radius:3px;"
                f"padding:2px 6px;font-family:'Noto Sans',Arial,sans-serif;"
                f"font-size:9px;}}"
                f"QComboBox::drop-down{{border:none;}}"
                f"QComboBox QAbstractItemView{{background:{p['input_bg']};"
                f"color:{p['text']};"
                f"selection-background-color:{p['btn_bg']};}}"))
        lay.addWidget(disp_combo)
        self._disp_mode_combos.append(disp_combo)

        if fullscreen_dialog is not None:
            fs_btn = QPushButton("📷 Restore")
            fs_btn.clicked.connect(fullscreen_dialog.close)
        else:
            fs_btn = QPushButton("📷 Fullscreen")
            fs_btn.clicked.connect(lambda: self.open_fullscreen_view(bar.window()))
        theme_manager.register_widget(
            fs_btn, lambda p: (
                f"QPushButton{{background:{p['input_bg']};color:{p['text']};"
                f"border:1px solid {p['border']};border-radius:3px;"
                f"padding:4px 8px;font-family:'Noto Sans',Arial,sans-serif;"
                f"font-size:9px;}}"
                f"QPushButton:hover{{background:{p['btn_hover']};}}"))
        fs_btn.setFixedHeight(28)
        lay.addWidget(fs_btn)

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

    def open_fullscreen_view(self, parent=None) -> QDialog:
        """
        Open the live triple-camera feed in a fullscreen popup, with
        its own view-mode/fullscreen control bar and (if
        detection_tab_ref is set -- see main_gui_triple.py) an Arm/
        Stop/E-Stop bar, so the operator doesn't need to exit
        fullscreen to control detection.

        Uses the same multi-display-widget mechanism display_widget()
        already provides for embedding the feed in more than one
        place at once: the popup gets its OWN QLabel via a fresh
        display_widget() call, automatically kept in sync with live
        frames in parallel with whatever's already embedded in the
        main window -- no extra routing logic needed here.

        Deliberately narrower than DualCameraPanel's own
        open_fullscreen_view(): no embedded live stats/spray-event
        sub-tables in this pass, since those depend on
        DetectionPanelRGB's stats_updated/spray_event_signal Qt
        signals, which DetectionPanelTriple doesn't emit (yet) --
        those tables are already visible in the main window itself,
        just not inside the fullscreen popup.
        """
        dlg = _FullscreenCameraDialogTriple(self, parent)
        dlg.setWindowTitle("Live Triple Camera Feed — Fullscreen")
        theme_manager.register_widget(
            dlg, lambda p: f"background-color:{p['bg0']};")
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        hint = QPushButton("Press Esc or click here to exit fullscreen")
        hint.setFlat(True)
        hint.setFixedHeight(22)
        hint.setCursor(Qt.PointingHandCursor)
        theme_manager.register_widget(
            hint, lambda p: (
                f"QPushButton{{background-color:{p['bg2']};"
                f"color:{p['muted']};border:none;"
                f"font-family:'Noto Sans',Arial,sans-serif;font-size:10px;}}"
                f"QPushButton:hover{{color:{p['text']};}}"))
        hint.clicked.connect(dlg.close)
        lay.addWidget(hint)

        if self.detection_tab_ref is not None:
            det = self.detection_tab_ref

            arm_bar = QWidget()
            theme_manager.register_widget(
                arm_bar, lambda p: f"background-color:{p['bg0']};")
            arm_lay = QHBoxLayout(arm_bar)
            arm_lay.setContentsMargins(8, 6, 8, 6)
            arm_lay.setSpacing(8)

            arm_lay.addWidget(_muted("DETECTION:"))

            btn_arm = QPushButton("▶  ARM DETECTION")
            theme_manager.register_button(btn_arm, "green")
            btn_arm.setMinimumHeight(32)
            btn_arm.setMinimumWidth(160)
            btn_arm.clicked.connect(det._det_start)
            arm_lay.addWidget(btn_arm)

            btn_stop = QPushButton("⏹  STOP")
            theme_manager.register_button(btn_stop, "dim_red")
            btn_stop.setMinimumHeight(32)
            btn_stop.clicked.connect(det._det_stop)
            arm_lay.addWidget(btn_stop)

            btn_estop = QPushButton("⚡  E-STOP")
            theme_manager.register_button(btn_estop, "estop")
            btn_estop.setMinimumHeight(32)
            btn_estop.setMinimumWidth(100)
            btn_estop.clicked.connect(det._det_estop)
            arm_lay.addWidget(btn_estop)

            arm_lay.addStretch()
            arm_status = _muted("DISARMED")
            arm_lay.addWidget(arm_status)

            def _sync_arm_state(armed, _btn_arm=btn_arm, _btn_stop=btn_stop,
                                _status=arm_status):
                try:
                    _btn_arm.setEnabled(not armed)
                    theme_manager.register_button(
                        _btn_arm, "dim_green" if armed else "green")
                    _btn_stop.setEnabled(armed)
                    theme_manager.register_button(
                        _btn_stop, "red" if armed else "dim_red")
                    _status.setText("ARMED" if armed else "DISARMED")
                    theme_manager.register_widget(
                        _status, lambda p, _armed=armed: (
                            f"color:{p['amber'] if _armed else p['muted']};"
                            f"font-size:10px;"
                            f"font-family:'Noto Sans',Arial,sans-serif;"))
                except RuntimeError:
                    pass   # dialog/widgets already destroyed

            # Initialize to whatever the REAL current armed state
            # already is -- opening fullscreen while already armed
            # (a very normal thing to do) must not show a stale
            # "DISARMED" until the next explicit arm/disarm action.
            _sync_arm_state(det.is_armed())
            det.armed_changed.connect(_sync_arm_state)

            lay.addWidget(arm_bar)

        lay.addWidget(self.camera_control_bar(fullscreen_dialog=dlg))

        display = self.display_widget()
        lay.addWidget(display, stretch=1)

        dlg.register_cleanup(display)
        dlg.showFullScreen()
        return dlg

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
        self._disp_mode_combos = []
