"""
dual_camera_panel.py
eMeet Dual RGB Camera Panel

Drop-in replacement for CameraPanel (gui/panels/camera_panel.py).
Wraps DualEMEETCamera and exposes the IDENTICAL public API so that
tab_collection.py, tab_detection.py, and main_gui_v3.py require
zero changes — only the import line changes.

Public API (mirrors CameraPanel exactly):
  panel.camera_model          → str   e.g. "eMeet C960 4K (Dual)"
  panel.is_acquiring          → bool
  panel.band_data             → dict  {"left": np.ndarray, "right": np.ndarray}
  panel.on_frame_ready        → callable(frame, bands) — set by AnalysisTab
  panel.on_detection_overlay  → callable(display_img) → img — set by DetectionPanel
  panel.get_frame_snapshot()  → (frame, bands)   thread-safe copy
  panel.camera_control_bar()  → QWidget          connect / start / stop bar
  panel.display_widget()      → QWidget          side-by-side live feed
  panel.ia                    → None             disables GenICam controls in AcquisitionPanel
  panel.cleanup()

Display:
  Side-by-side LEFT | RIGHT frames with zone overlay lines
  (zone lines are drawn when detection is armed)

Author : Nana | NDSU / PhD Imaging System
Path   : /media/pagsun/Transcend/phd_project/emeet_dual_cam/
"""

import cv2
import time
import threading
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSizePolicy, QComboBox, QDialog
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

from gui.style import _muted, _sec
from gui.theme_manager import theme_manager
from gui.shared_log import UnifiedLog
from gui.frame_text import put_text, text_size
from gui.spray_event_table import (
    build_spray_event_table, insert_spray_event_row,
    build_stats_table, update_stats_row,
)

try:
    from core.dual_emeet_camera import DualEMEETCamera, FramePair
except ImportError:
    from core.dual_emeet_camera import DualEMEETCamera, FramePair


# ─────────────────────────────────────────────────────────────
#  DISPLAY MODES
# ─────────────────────────────────────────────────────────────

DISPLAY_MODES = [
    "Side by Side",   # LEFT | RIGHT
    "Left Only",
    "Right Only",
    "Red Channel",
    "Green Channel",
    "Blue Channel",
]


# ─────────────────────────────────────────────────────────────
#  FULLSCREEN CAMERA DIALOG
# ─────────────────────────────────────────────────────────────

class _FullscreenCameraDialog(QDialog):
    """
    Fullscreen popout for the live camera feed (see
    DualCameraPanel.open_fullscreen_view()).

    Defined as a REAL QDialog subclass with keyPressEvent/closeEvent
    as genuine class methods -- not assigned as instance-level
    function attributes. Overriding a Qt virtual event handler by
    setting `dlg.keyPressEvent = some_function` is a known PyQt5
    pitfall: SIP's C++-to-Python virtual dispatch ("catcher") doesn't
    reliably recognize an instance-attribute override the way it does
    a genuine subclass method override, and can raise
    `TypeError: invalid argument to sipBadCatcherResult()` at runtime.
    A real subclass avoids that entirely.
    """

    def __init__(self, camera_panel, parent=None):
        super().__init__(parent)
        self._camera_panel = camera_panel
        self._display_widget = None
        self._extra_widgets = []
        self._connections = []   # list of (signal, slot) to disconnect on close

    def register_cleanup(self, display_widget, extra_widgets=None,
                          connections=None):
        """Record what needs unregistering/disconnecting when this
        dialog closes. Called once by open_fullscreen_view() right
        after building the dialog's contents.
        connections, if given, is a list of (signal, slot) pairs to
        disconnect -- one per live-updating table (stats, spray
        events) subscribed to a DetectionPanelRGB signal."""
        self._display_widget = display_widget
        self._extra_widgets = extra_widgets or []
        self._connections = connections or []

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self._camera_panel._cleanup_fullscreen_widgets(self)
        for signal, slot in self._connections:
            try:
                signal.disconnect(slot)
            except Exception:
                pass   # already disconnected, or source object gone
        event.accept()


# ─────────────────────────────────────────────────────────────
#  DUAL CAMERA PANEL
# ─────────────────────────────────────────────────────────────

class DualCameraPanel:
    """
    Shared dual-RGB camera backend + display widget.
    Instantiated ONCE in MainWindow, shared between Tab 1 and Tab 2.

    Does NOT inherit QWidget — provides widgets via camera_control_bar()
    and display_widget(), exactly like CameraPanel.
    """

    # ── Public API attributes (mirroring CameraPanel) ─────────
    camera_model:           str   = "eMeet C960 4K (Dual)"
    is_acquiring:           bool  = False
    band_data:              dict  = None   # {"left": arr, "right": arr}
    on_frame_ready          = None         # callable(frame, bands)
    on_detection_overlay    = None         # callable(img) → img

    # GenICam image acquisition object — None disables GenICam controls
    # AcquisitionPanel checks `camera.ia is None` before enabling settings
    ia = None

    def __init__(self, shared_log: UnifiedLog):
        self.shared_log     = shared_log
        self.band_data      = {}

        # Latest frames — written by QTimer, read by display
        self._left_frame:  np.ndarray = None
        self._right_frame: np.ndarray = None
        self._frame_lock   = threading.Lock()

        # Camera instance — created on connect
        self._camera: DualEMEETCamera = None

        # Display widget references (created lazily)
        # Lists — both Tab1 and Tab2 each call camera_control_bar()
        # and display_widget(), so we track all instances and update all.
        self._display_lbls:    list = []   # all QLabel display widgets
        self._connect_btns:    list = []   # all connect buttons
        self._start_btns:      list = []   # all start/stop buttons
        self._status_lbls:     list = []   # all status labels
        self._disp_mode_combos: list = []  # all display mode combos

        # Optional: set by whichever tab has a live detection panel
        # (tab_detection.py sets this to its DetectionPanelRGB instance
        # after construction) so the fullscreen popout can show a live
        # spray-event mini-feed. None on Data Collection tab, where
        # there's nothing to show since detection isn't running there.
        self.spray_event_source = None

        # Display refresh timer
        self._display_timer = QTimer()
        self._display_timer.timeout.connect(self._refresh_display)

        # Zone overlay lines (set by detection_panel when armed)
        # List of (x_fraction, camera) tuples to draw vertical lines
        self._zone_lines: list = []

        self.shared_log.log("CAMERA", "DualCameraPanel ready", "ok")

    # ── Connection ────────────────────────────────────────────

    def connect(self):
        """Open both cameras and start capture threads."""
        if self._camera is not None:
            self.shared_log.log(
                "CAMERA", "Already connected", "warn")
            return True
        try:
            self.shared_log.log(
                "CAMERA", "Connecting eMeet cameras …", "info")
            self._camera = DualEMEETCamera()
            self.camera_model = "eMeet C960 4K (Dual)"
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
            self.shared_log.log(
                "CAMERA", "Acquisition started", "ok")
        except Exception as e:
            self.shared_log.log(
                "CAMERA", f"Start failed: {e}", "error")

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
        Called every 33ms by QTimer (GUI thread).
        Reads latest pair, updates display, fires callbacks.
        """
        if self._camera is None:
            return

        pair = self._camera.read_pair()
        if pair is None:
            return

        with self._frame_lock:
            self._left_frame  = pair.left.copy()
            self._right_frame = pair.right.copy()

        # Update band_data — {"left": arr, "right": arr}
        # This is the equivalent of CameraPanel.band_data
        self.band_data = {
            "left":  self._left_frame,
            "right": self._right_frame,
        }

        # Fire on_frame_ready (used by AnalysisTab)
        if self.on_frame_ready:
            try:
                self.on_frame_ready(self._left_frame, self.band_data)
            except Exception as e:
                self.shared_log.log(
                    "CAMERA", f"on_frame_ready error: {e}", "warn")

        # Build display image
        disp_img = self._build_display(pair)

        # Fire on_detection_overlay (used by DetectionPanel)
        if self.on_detection_overlay:
            try:
                disp_img = self.on_detection_overlay(disp_img)
            except Exception:
                pass

        # Render to QLabel
        self._show(disp_img)

    def _build_display(self, pair: FramePair) -> np.ndarray:
        """
        Build the display image based on selected display mode.
        Returns BGR numpy array.
        """
        mode = "Side by Side"
        if self._disp_mode_combos:
            mode = self._disp_mode_combos[0].currentText()

        left  = pair.left
        right = pair.right

        # Determine label display size for all single-camera modes
        _lbl_w = _lbl_h = 0
        for _lbl in self._display_lbls:
            try:
                _sz = _lbl.size()
                if _sz.width() > 10 and _sz.height() > 10:
                    _lbl_w = _sz.width()
                    _lbl_h = _sz.height()
                    break
            except RuntimeError:
                pass
        if _lbl_w < 10:
            _lbl_w, _lbl_h = 1280, 720
        _disp_h = max(1, int(_lbl_w / (1920 / 1080)))  # 16:9

        if mode == "Left Only":
            img = cv2.resize(left, (_lbl_w, _disp_h))

        elif mode == "Right Only":
            img = cv2.resize(right, (_lbl_w, _disp_h))

        elif mode in ("Red Channel", "Green Channel", "Blue Channel"):
            ch = {"Red Channel": 2,
                  "Green Channel": 1,
                  "Blue Channel": 0}[mode]
            gray = left[:, :, ch]
            img  = cv2.resize(
                cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                (_lbl_w, _disp_h))

        else:
            # Side by Side — fit each half into the display label
            # preserving the camera's native 16:9 aspect ratio.
            # Camera native: 1920×1080.  Display: label_w × label_h.
            # Each camera gets half the label width; height is derived
            # from the 16:9 ratio so there is no stretching.
            cam_w, cam_h = 1920, 1080
            cam_aspect   = cam_w / cam_h   # 16:9 = 1.777…

            # Determine target display size from the first valid label
            lbl_w = lbl_h = 0
            for lbl in self._display_lbls:
                try:
                    sz = lbl.size()
                    if sz.width() > 10 and sz.height() > 10:
                        lbl_w = sz.width()
                        lbl_h = sz.height()
                        break
                except RuntimeError:
                    pass

            if lbl_w < 10:
                # Label not yet rendered — fall back to a sensible default
                lbl_w, lbl_h = 1280, 360

            # Each camera half gets half the label width
            half_w   = (lbl_w - 4) // 2   # subtract 4px for centre divider
            # Height derived from aspect ratio — no vertical stretch
            disp_h   = max(1, int(half_w / cam_aspect))

            l_disp = cv2.resize(left,  (half_w, disp_h))
            r_disp = cv2.resize(right, (half_w, disp_h))
            h = disp_h   # use for all subsequent drawing

            # NOTE: zone-line/nozzle-crosshair/N1-N3/A-B1-B2-C drawing
            # deliberately does NOT happen here. DetectionPanelRGB's
            # own _draw_overlay() (wired via on_detection_overlay,
            # below) already draws a strictly richer version of the
            # exact same information -- active-zone highlighting, real
            # detection boxes, HUD -- whenever detection is armed. An
            # earlier version of this code drew a second, more basic
            # copy of the same zone/nozzle guides gated on a separate
            # show_zone_overlay flag that also happened to be True
            # exactly whenever on_detection_overlay was active,
            # producing two overlapping, slightly-offset copies of the
            # same lines and labels on screen at once. There's no
            # remaining scenario where drawing it here adds anything:
            # not armed means nothing should show at all, and armed
            # means on_detection_overlay's version already covers it.

            # ── Camera labels ─────────────────────────────────
            # (always shown — basic camera identification, useful
            # regardless of detection/armed state)
            LABEL_FONT_SIZE = 13
            put_text(l_disp, "LEFT",  (6, 4),
                     font_size=LABEL_FONT_SIZE, color_bgr=(0, 220, 80))
            put_text(r_disp, "RIGHT", (6, 4),
                     font_size=LABEL_FONT_SIZE, color_bgr=(0, 220, 80))

            # Sync error warning (always shown — a camera health
            # diagnostic, not a detection/zone concept)
            if not pair.sync_ok:
                put_text(l_disp,
                          f"SYNC {pair.sync_error_ms:.0f}ms",
                          (10, 42),
                          font_size=LABEL_FONT_SIZE, color_bgr=(0, 100, 255))

            # Centre divider
            divider = np.zeros((h, 4, 3), dtype=np.uint8)
            divider[:] = (60, 60, 60)
            img = np.hstack((l_disp, divider, r_disp))

        return img

    def _show(self, img: np.ndarray):
        """Render BGR numpy array to ALL registered display QLabels."""
        if not self._display_lbls:
            return
        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            rgb  = np.ascontiguousarray(rgb)
            q    = QImage(rgb.tobytes(), w, h, w * 3,
                          QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q)
            for lbl in self._display_lbls:
                try:
                    # _build_display already produces a correctly-sized
                    # image matched to the label dimensions.
                    # We still scale here as a safety net for resize events,
                    # using KeepAspectRatio so it never distorts.
                    sz = lbl.size()
                    if sz.width() > 10 and sz.height() > 10:
                        scaled = pixmap.scaled(
                            sz,
                            Qt.KeepAspectRatio,
                            Qt.FastTransformation)   # faster than Smooth
                    else:
                        scaled = pixmap
                    lbl.setPixmap(scaled)
                except RuntimeError:
                    pass   # widget destroyed
        except Exception:
            pass

    # ── Public API ────────────────────────────────────────────

    @property
    def current_frame(self):
        """
        Compatibility property for spray_panel.py and any other
        shared panel that checks `camera.current_frame is not None`.
        Returns the latest left frame, or None if not yet acquired.
        """
        return self._left_frame

    def get_frame_snapshot(self):
        """
        Thread-safe copy of the latest frame pair.
        Returns (left_frame, band_data) mirroring CameraPanel.
        band_data = {"left": arr, "right": arr}
        """
        with self._frame_lock:
            if self._left_frame is None:
                return None, {}
            frame  = self._left_frame.copy()
            bands  = {
                "left":  self._left_frame.copy(),
                "right": self._right_frame.copy()
                       if self._right_frame is not None else frame,
            }
        return frame, bands

    # ── Widgets ───────────────────────────────────────────────

    def camera_control_bar(self, fullscreen_dialog=None) -> QWidget:
        """
        Returns the camera toolbar widget (connect / start / stop).
        Mirrors CameraPanel.camera_control_bar().

        fullscreen_dialog: pass the QDialog this bar is being embedded
        INSIDE (only from open_fullscreen_view() itself) so the
        Fullscreen button becomes a Restore button that closes that
        specific dialog, instead of opening ANOTHER nested fullscreen
        dialog on top of the current one.
        """
        bar = QWidget()
        bar.setFixedHeight(44)
        theme_manager.register_widget(bar, lambda p: f"background-color:{p['bg0']};")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(6)

        # Connect button
        connect_btn = QPushButton("🔌  CONNECT")
        theme_manager.register_button(connect_btn, "green")
        connect_btn.setFixedHeight(32)
        connect_btn.setMinimumWidth(110)
        connect_btn.clicked.connect(self._on_connect_btn)
        lay.addWidget(connect_btn)
        self._connect_btns.append(connect_btn)

        # Start / Stop button
        start_btn = QPushButton("▶  START")
        theme_manager.register_button(start_btn, "green")
        start_btn.setFixedHeight(32)
        start_btn.setMinimumWidth(90)
        start_btn.setEnabled(False)
        start_btn.clicked.connect(self._on_start_stop_btn)
        lay.addWidget(start_btn)
        self._start_btns.append(start_btn)

        # Display mode selector
        lay.addWidget(_muted("View:"))
        disp_combo = QComboBox()
        disp_combo.addItems(DISPLAY_MODES)
        disp_combo.setFixedHeight(28)
        disp_combo.setMinimumWidth(130)
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

        # Fullscreen popout / Restore
        if fullscreen_dialog is not None:
            fs_btn = QPushButton("🗗 Restore")
            fs_btn.clicked.connect(fullscreen_dialog.close)
        else:
            fs_btn = QPushButton("⛶ Fullscreen")
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

        # Status label
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
        """
        Returns the live feed display widget.
        Mirrors CameraPanel.display_widget().
        """
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
        lbl.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Keep aspect ratio scaling — pixmap fills label on resize
        lbl.setMinimumSize(1, 1)
        lay.addWidget(lbl)
        self._display_lbls.append(lbl)
        return w

    def remove_display_widget(self, container: QWidget):
        """
        Unregister a display widget (from display_widget()) so it
        stops receiving frame updates. Superseded for the fullscreen
        popout by _cleanup_fullscreen_widgets() below (which also
        prunes control-bar widgets, not just the display label), kept
        as a standalone method for any other caller that only ever
        added a bare display_widget().
        """
        lbl = container.findChild(QLabel)
        if lbl is not None and lbl in self._display_lbls:
            self._display_lbls.remove(lbl)

    def _cleanup_fullscreen_widgets(self, dlg: QWidget):
        """
        Remove every widget created for a now-closing fullscreen
        dialog from every tracking list (display labels, connect/
        start buttons, status labels, view-mode combos), so they stop
        receiving frame/state updates and don't accumulate as dead
        references over repeated open/close of the fullscreen view.
        Uses widget-tree ancestry rather than tracking "which list did
        this widget go into" individually, so it stays correct if the
        fullscreen dialog's contents ever change.
        """
        for lst in (self._display_lbls, self._connect_btns,
                    self._start_btns, self._status_lbls,
                    self._disp_mode_combos):
            for w in list(lst):
                try:
                    is_descendant = dlg.isAncestorOf(w)
                except RuntimeError:
                    is_descendant = True   # widget already destroyed by Qt
                if is_descendant:
                    lst.remove(w)

    def open_fullscreen_view(self, parent=None) -> QDialog:
        """
        Open the live camera feed in a fullscreen popup window, with
        its own connect/stop/view controls and (if this panel's
        spray_event_source is set -- see tab_detection.py) a live
        spray-event mini-feed, so an operator doesn't need to exit
        fullscreen to control the camera or lose track of what's
        being sprayed.

        Uses the same multi-display-widget mechanism display_widget()
        already provides for embedding the feed in more than one tab
        at once: the popup gets its OWN QLabel via a fresh
        display_widget() call (and its own control-bar widgets via a
        fresh camera_control_bar() call), automatically kept in sync
        with live frames/state in parallel with whatever's already
        embedded in the tab -- no extra routing logic needed here.
        """
        dlg = _FullscreenCameraDialog(self, parent)
        dlg.setWindowTitle("Live Camera Feed — Fullscreen")
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

        # Full control bar — connect/stop/view/fullscreen, same as the
        # embedded one, kept in sync automatically like every other
        # tracked control-bar instance.
        lay.addWidget(self.camera_control_bar(fullscreen_dialog=dlg))

        body = QVBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        display = self.display_widget()
        body.addWidget(display, stretch=3)

        connections = []   # (signal, slot) pairs to disconnect on close

        # Live status table (mode/FPS/inference/detections/events) --
        # replaces the HUD text that used to be baked into the video
        # frame itself (see DetectionPanelRGB._draw_overlay()).
        stats_table = None
        if self.spray_event_source is not None:
            stats_grp = QWidget()
            sg_lay = QVBoxLayout(stats_grp)
            sg_lay.setContentsMargins(4, 4, 4, 0)
            sg_lay.addWidget(_muted("Status"))

            stats_table = build_stats_table(
                ["Mode", "FPS", "Inference", "Detections", "Events"])
            sg_lay.addWidget(stats_table)
            body.addWidget(stats_grp)

            def _on_stats(stats, tbl=stats_table):
                try:
                    update_stats_row(tbl, [
                        stats.get("mode", "--"),
                        f"{stats.get('fps', 0):.1f}",
                        f"{stats.get('inference_ms', 0):.1f}ms",
                        stats.get("detections", 0),
                        stats.get("events", 0),
                    ])
                except RuntimeError:
                    pass   # dialog/table already destroyed

            self.spray_event_source.stats_updated.connect(_on_stats)
            connections.append(
                (self.spray_event_source.stats_updated, _on_stats))

        # Live spray-event feed — same table format as Session
        # Analysis's "Live Spray Event Feed" (Time, Zone, Nozzle,
        # Class, Conf, Pose, GPS), via the shared gui/spray_event_table
        # module so both stay in sync.
        events_table = None
        if self.spray_event_source is not None:
            events_grp = QWidget()
            eg_lay = QVBoxLayout(events_grp)
            eg_lay.setContentsMargins(4, 4, 4, 4)
            eg_title = _muted("Live Spray Event Feed")
            eg_lay.addWidget(eg_title)

            events_table = build_spray_event_table()
            events_table.setMaximumHeight(220)
            eg_lay.addWidget(events_table)

            def _on_spray_event(event, tbl=events_table):
                try:
                    insert_spray_event_row(tbl, event)
                except RuntimeError:
                    pass   # dialog/table already destroyed

            self.spray_event_source.spray_event_signal.connect(_on_spray_event)
            connections.append(
                (self.spray_event_source.spray_event_signal, _on_spray_event))
            body.addWidget(events_grp, stretch=1)

        lay.addLayout(body, stretch=1)

        dlg.register_cleanup(display, [stats_table, events_table], connections)
        dlg.showFullScreen()
        return dlg

    # ── Control bar button logic ──────────────────────────────

    def _on_connect_btn(self):
        if self._camera is None:
            self.connect()
        else:
            self.stop()

    def _on_start_stop_btn(self):
        if self.is_acquiring:
            self.stop()
        else:
            self.start()

    def _update_ctrl_bar(self):
        """Refresh ALL registered buttons and status labels."""
        connected   = self._camera is not None
        acquiring   = self.is_acquiring

        for btn in self._connect_btns:
            try:
                if connected:
                    btn.setText("🔌  DISCONNECT")
                    theme_manager.register_button(btn, "red")
                else:
                    btn.setText("🔌  CONNECT")
                    theme_manager.register_button(btn, "green")
            except RuntimeError:
                pass   # widget deleted

        for btn in self._start_btns:
            try:
                btn.setEnabled(connected)
                if acquiring:
                    btn.setText("⏹  STOP")
                    theme_manager.register_button(btn, "red")
                else:
                    btn.setText("▶  START")
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

    # ── Public v4l2 access (used by AcquisitionPanelRGB) ──────

    def set_v4l2(self, device: str, control: str, value) -> bool:
        """
        Set a v4l2 control on one of the eMeet cameras.
        Delegates to the camera driver if connected,
        otherwise calls v4l2-ctl directly.
        Returns True on success.
        """
        if self._camera is not None:
            return self._camera._v4l2(device, control, value)
        # Camera not started yet — call v4l2-ctl directly
        import subprocess
        cmd = ["v4l2-ctl", "-d", device, "-c",
               f"{control}={value}"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return r.returncode == 0

    # ── Cleanup ───────────────────────────────────────────────

    def cleanup(self):
        """Called by MainWindow.closeEvent()."""
        self._display_timer.stop()
        if self._camera is not None:
            try:
                self._camera.stop()
            except Exception:
                pass
            self._camera = None
        self.is_acquiring      = False
        self.on_frame_ready    = None
        self._connect_btns     = []
        self._start_btns       = []
        self._status_lbls      = []
        self._display_lbls     = []
        self._disp_mode_combos = []