"""
gui/tabs/tab_detection.py
ABEN Field Imaging System — Tab 2: Detection

Layout:
  Top    — Detection ARM/STOP/E-STOP bar (always visible)
  Left   — Spray Panel (Spray tab | Detection tab)
  Middle — Camera Feed with detection overlay (shared)
  Right  — Navigation Panel (shared)
  Bottom — System Log
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTabWidget, QLabel, QPushButton
)
from PyQt5.QtCore import Qt, pyqtSignal

from gui.style import _divider, _sec, _muted, LED
from gui.theme_manager import theme_manager, _darken, _lighten
from gui.shared_log import UnifiedLog, LogPanel
from gui.panels.gantry_panel import GantryPanel
from gui.panels.spray_panel import SprayPanel
from gui.panels.dual_camera_panel import DualCameraPanel
from gui.panels.acquisition_panel_rgb import AcquisitionPanelRGB
from gui.style import _sec, _muted
from gui.panels.navigation_panel_rgb import NavigationPanelRGB
from gui.panels.detection_panel_rgb import DetectionPanelRGB
from gui.spray_event_table import (
    build_spray_event_table, insert_spray_event_row,
    build_stats_table, update_stats_row,
)


class DetectionTab(QWidget):
    """
    Tab 2 — Detection & Spray Deployment.

    Shared objects passed in from MainWindow:
      camera  : CameraPanel
      nav     : NavigationPanel
      gantry  : GantryPanel  (shared — same controller as Tab 1)
    """

    # Emitted whenever detection is armed/disarmed/estopped, so
    # MainWindow can lock/unlock Data Collection's navigation
    # controls — only one tab should ever be able to command Husky
    # motion at a time (see NavigationPanelRGB.set_movement_controls_enabled).
    armed_changed = pyqtSignal(bool)

    def __init__(self, camera: DualCameraPanel,
                 nav: NavigationPanelRGB,
                 gantry: GantryPanel,
                 acq=None,
                 parent=None):
        super().__init__(parent)
        self.camera = camera
        self.nav    = nav
        self.gantry = gantry

        # Tab-local log
        self.log = UnifiedLog()

        # Spray panel — shares gantry controller
        self.spray = SprayPanel(
            self.log,
            gantry_ctrl_ref=lambda: gantry.ctrl,
            camera_ref=lambda: camera
        )
        # Wire gantry state → spray panel
        gantry.state_signal.connect(self.spray.update_state)

        # Use shared acq if provided (from MainWindow) — guarantees
        # identical camera settings between Data Collection and Detection.
        if acq is not None:
            self.acq = acq
            self._owns_acq = False
        else:
            self.acq = AcquisitionPanelRGB(self.log, camera)
            self._owns_acq = True
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(
            1500,
            lambda: self.acq.enable_camera_controls(True))

        # Detection panel
        self.detect = DetectionPanelRGB(
            self.log,
            camera=camera,
            gantry_ctrl_ref=lambda: gantry.ctrl
        )

        # Let the shared camera's fullscreen popout show a live
        # spray-event mini-feed (see DualCameraPanel.open_fullscreen_view()).
        # Harmless that self.camera is the SAME shared instance used by
        # Data Collection tab too -- events only ever occur while
        # Detection is armed regardless of which tab's fullscreen view
        # is open, so this is a reasonable bonus there too, not a leak
        # of Detection-only state into an unrelated tab.
        self.camera.spray_event_source = self.detect

        # Wire detection ros_bridge → nav odom display
        # Updated when detection arms
        self._wire_ros_bridge()

        self._build_ui()
        self.log.log("SYS", "Detection tab ready", "ok")

    def _wire_ros_bridge(self):
        """
        After detection arms, update nav panel with live ros_bridge.
        Poll every second — lightweight check.
        """
        from PyQt5.QtCore import QTimer
        self._bridge_timer = QTimer()
        self._bridge_timer.timeout.connect(self._update_nav_bridge)
        self._bridge_timer.start(1000)

    def _update_nav_bridge(self):
        bridge = self.detect.ros_bridge
        if bridge is not None:
            self.nav.set_ros_bridge(lambda b=bridge: b)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── TOP: Detection ARM bar (always visible) ───────────
        arm_bar = self._detection_arm_bar()
        root.addWidget(arm_bar)
        root.addWidget(_divider())

        # ── 3-panel splitter ──────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(3)
        theme_manager.register_widget(
            splitter, lambda p: (
                f"QSplitter::handle{{background-color:{p['border2']};}}"))

        # ── LEFT: Spray + Detection tabs ─────────────────────
        left_tabs = QTabWidget()
        left_tabs.addTab(self.spray,  "💉 Spray")
        left_tabs.addTab(self.detect, "🎯 Detect")
        # Camera settings — use a dedicated widget that references
        # the shared acq but does NOT reparent its internal subtab widget.
        # Reparenting steals the widget from Tab 1 leaving it empty.
        cam_settings_w = self._build_cam_settings_tab()
        left_tabs.addTab(cam_settings_w, "⚙ Camera")

        left_w = QWidget()
        left_w.setMinimumWidth(400)
        left_w.setMaximumWidth(500)
        llay = QVBoxLayout(left_w)
        llay.setContentsMargins(0, 0, 0, 0)
        llay.addWidget(left_tabs)
        splitter.addWidget(left_w)

        # ── MIDDLE: Camera feed + overlay ─────────────────────
        mid_w = QWidget()
        mid_w.setMinimumWidth(400)
        mlay = QVBoxLayout(mid_w)
        mlay.setContentsMargins(0, 0, 0, 0)
        mlay.addWidget(self.camera.camera_control_bar())

        # Status table (mode/FPS/inference/detections/events) above
        # the camera view, and the Live Spray Event Feed table below
        # it -- same shared table format/widgets as the fullscreen
        # popout (gui/spray_event_table module), so switching between
        # this normal view and fullscreen looks and behaves
        # identically, just at a different size.
        status_grp = QWidget()
        stlay = QVBoxLayout(status_grp)
        stlay.setContentsMargins(4, 4, 4, 0)
        stlay.addWidget(_muted("Status"))
        stats_table = build_stats_table(
            ["Mode", "FPS", "Inference", "Detections", "Events"])
        stlay.addWidget(stats_table)
        mlay.addWidget(status_grp)

        def _on_stats(stats, tbl=stats_table):
            update_stats_row(tbl, [
                stats.get("mode", "--"),
                f"{stats.get('fps', 0):.1f}",
                f"{stats.get('inference_ms', 0):.1f}ms",
                stats.get("detections", 0),
                stats.get("events", 0),
            ])
        self.detect.stats_updated.connect(_on_stats)

        mlay.addWidget(self.camera.display_widget())

        events_grp = QWidget()
        evlay = QVBoxLayout(events_grp)
        evlay.setContentsMargins(4, 4, 4, 4)
        evlay.addWidget(_muted("Live Spray Event Feed"))
        events_table = build_spray_event_table()
        events_table.setMaximumHeight(180)
        evlay.addWidget(events_table)
        mlay.addWidget(events_grp)

        def _on_spray_event(event, tbl=events_table):
            insert_spray_event_row(tbl, event)
        self.detect.spray_event_signal.connect(_on_spray_event)

        splitter.addWidget(mid_w)

        # ── RIGHT: Navigation ─────────────────────────────────
        right_w = QWidget()
        right_w.setMinimumWidth(380)
        right_w.setMaximumWidth(460)
        rlay = QVBoxLayout(right_w)
        rlay.setContentsMargins(0, 0, 0, 0)
        rlay.addWidget(self.nav)
        splitter.addWidget(right_w)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        root.addWidget(splitter, stretch=1)

        # ── BOTTOM: Log ───────────────────────────────────────
        root.addWidget(_divider())
        root.addWidget(
            LogPanel(self.log,
                     sources=["DETECT","GANTRY","NAV","SYS"],
                     height=110))

    def _build_cam_settings_tab(self) -> QWidget:
        """
        Lightweight camera settings widget for the Detection tab.
        Mirrors the shared acq panel but is a separate widget instance
        so it does NOT steal the subtab from Data Collection tab.

        Shows preset buttons + Read/Apply controls that operate on
        the same physical cameras via v4l2-ctl.
        """
        from PyQt5.QtWidgets import (
            QScrollArea, QGroupBox, QGridLayout, QSpinBox
        )
        from gui.style import BTN_BLUE, BTN_GREEN, BTN_AMBER

        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(6)

        # ── Info label ────────────────────────────────────────
        info = QLabel(
            "Camera settings are shared with Data Collection tab. "
            "Changes here apply to both tabs immediately.")
        theme_manager.register_widget(
            info, lambda p: (
                f"color:{_lighten(p['green'],10)};font-size:9px;"
                f"font-family:'Noto Sans',Arial,sans-serif;"
                f"padding:4px;background:{_darken(p['green'],55)};"
                f"border-radius:3px;"))
        info.setWordWrap(True)
        outer.addWidget(info)

        # ── Preset buttons ────────────────────────────────────
        grp = QGroupBox("Lighting Presets")
        theme_manager.register_widget(
            grp, lambda p: (
                f"QGroupBox{{border:1px solid {_darken(p['green'],30)};"
                f"border-radius:4px;margin-top:8px;color:{p['green']};"
                f"font-size:10px;}}"
                f"QGroupBox::title{{subcontrol-origin:margin;padding:0 4px;}}"))
        glay = QVBoxLayout(grp)

        btn_outdoor = QPushButton("🌤  Outdoor / Field  (exp=5)")
        theme_manager.register_widget(
            btn_outdoor, lambda p: (
                f"QPushButton{{background:{_darken(p['green'],45)};"
                f"color:{_lighten(p['green'],10)};"
                f"border:1px solid {_darken(p['green'],25)};border-radius:4px;"
                f"padding:6px;font-family:'Noto Sans',Arial,sans-serif;"
                f"font-size:10px;}}"
                f"QPushButton:hover{{background:{_darken(p['green'],35)};}}"))
        btn_outdoor.clicked.connect(
            lambda: getattr(self.acq, "_preset_outdoor", lambda: None)())
        glay.addWidget(btn_outdoor)

        btn_cloudy = QPushButton("☁  Cloudy / Shade  (exp=80)")
        theme_manager.register_widget(
            btn_cloudy, lambda p: (
                f"QPushButton{{background:{_darken(p['blue'],45)};"
                f"color:{_lighten(p['blue'],10)};"
                f"border:1px solid {_darken(p['blue'],25)};border-radius:4px;"
                f"padding:6px;font-family:'Noto Sans',Arial,sans-serif;"
                f"font-size:10px;}}"
                f"QPushButton:hover{{background:{_darken(p['blue'],35)};}}"))
        btn_cloudy.clicked.connect(
            lambda: getattr(self.acq, "_preset_cloudy", lambda: None)())
        glay.addWidget(btn_cloudy)

        btn_indoor = QPushButton("💡  Indoor / Lab  (exp=300)")
        theme_manager.register_widget(
            btn_indoor, lambda p: (
                f"QPushButton{{background:{_darken(p['amber'],55)};"
                f"color:{_lighten(p['amber'],5)};"
                f"border:1px solid {_darken(p['amber'],30)};border-radius:4px;"
                f"padding:6px;font-family:'Noto Sans',Arial,sans-serif;"
                f"font-size:10px;}}"
                f"QPushButton:hover{{background:{_darken(p['amber'],45)};}}"))
        btn_indoor.clicked.connect(
            lambda: getattr(self.acq, "_preset_indoor", lambda: None)())
        glay.addWidget(btn_indoor)

        outer.addWidget(grp)

        # ── Read/Apply shortcuts ──────────────────────────────
        btn_read = QPushButton("↻  Read from camera")
        btn_read.setStyleSheet(BTN_BLUE)
        btn_read.clicked.connect(
            lambda: getattr(self.acq, "_refresh_all", lambda: None)())
        outer.addWidget(btn_read)

        btn_apply = QPushButton("✓  Apply current settings")
        btn_apply.setStyleSheet(BTN_GREEN)
        btn_apply.clicked.connect(
            lambda: getattr(self.acq, "_apply_all", lambda: None)())
        outer.addWidget(btn_apply)

        outer.addWidget(_muted(
            "Full camera settings available in Data Collection tab."))
        outer.addStretch()
        return w

    def _detection_arm_bar(self) -> QWidget:
        """
        Prominent START / STOP / E-STOP bar always visible
        at the top of Tab 2 regardless of left panel subtab.
        Mirrors DetectionPanel buttons but positioned globally.
        """
        w = QWidget()
        theme_manager.register_widget(
            w, lambda p: (
                f"background-color:{p['bg0']};"))
        lay = QHBoxLayout(w)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(8)

        lay.addWidget(_muted("DETECTION:"))

        self.btn_arm_start = QPushButton("▶  ARM DETECTION")
        theme_manager.register_button(self.btn_arm_start, "green")
        self.btn_arm_start.setMinimumHeight(32)
        self.btn_arm_start.setMinimumWidth(160)
        self.btn_arm_start.clicked.connect(self._on_arm)
        lay.addWidget(self.btn_arm_start)

        self.btn_arm_stop = QPushButton("⏹  STOP")
        theme_manager.register_button(self.btn_arm_stop, "dim_red")
        self.btn_arm_stop.setMinimumHeight(32)
        self.btn_arm_stop.setEnabled(False)
        self.btn_arm_stop.clicked.connect(self._on_stop)
        lay.addWidget(self.btn_arm_stop)

        self.btn_arm_estop = QPushButton("⚡  E-STOP")
        theme_manager.register_button(self.btn_arm_estop, "estop")
        self.btn_arm_estop.setMinimumHeight(32)
        self.btn_arm_estop.setMinimumWidth(100)
        self.btn_arm_estop.clicked.connect(self._on_estop)
        lay.addWidget(self.btn_arm_estop)

        lay.addStretch()

        # Status LED + label
        self.arm_led = LED(14)
        lay.addWidget(self.arm_led)
        self.arm_status = _muted("DISARMED")
        lay.addWidget(self.arm_status)

        return w

    def _on_arm(self):
        self.detect._det_start()
        if self.detect.is_armed:
            self.btn_arm_start.setEnabled(False)
            theme_manager.register_button(self.btn_arm_start, "dim_green")
            self.btn_arm_stop.setEnabled(True)
            theme_manager.register_button(self.btn_arm_stop, "red")
            self.arm_led.set_state(True, role="amber")
            self.arm_status.setText("ARMED")
            theme_manager.register_widget(
                self.arm_status, lambda p: (
                    f"color:{p['amber']};font-size:10px;"
                    f"font-family:'Noto Sans',Arial,sans-serif;"))
            self.nav.set_movement_controls_enabled(True)
            self.armed_changed.emit(True)

    def _on_stop(self):
        self.detect._det_stop()
        self.btn_arm_start.setEnabled(True)
        theme_manager.register_button(self.btn_arm_start, "green")
        self.btn_arm_stop.setEnabled(False)
        theme_manager.register_button(self.btn_arm_stop, "dim_red")
        self.arm_led.set_state(False, role="amber")
        self.arm_status.setText("DISARMED")
        theme_manager.register_widget(
            self.arm_status, lambda p: (
                f"color:{p['muted']};font-size:10px;"
                f"font-family:'Noto Sans',Arial,sans-serif;"))
        self.nav.set_movement_controls_enabled(False)
        self.armed_changed.emit(False)

    def _on_estop(self):
        self.detect._det_estop()
        self.spray.emergency_stop()
        self.nav.set_movement_controls_enabled(False)
        self.armed_changed.emit(False)
        self.log.log("SYS", "E-STOP activated", "error")

    def cleanup(self):
        self._bridge_timer.stop()
        self.detect.cleanup()
        # acq cleaned up by MainWindow when shared
        if getattr(self, "_owns_acq", True):
            pass  # nothing extra to cleanup for camera-settings-only use