#!/usr/bin/env python3
"""
main_gui_triple.py
ABEN Triple RGB Imaging System — Standalone Triple-Camera App

Integration step 3 of 3: the runnable entry point wiring together
integration step 1 (gui/panels/triple_camera_panel.py,
TripleCameraPanel) and step 2 (gui/panels/detection_panel_triple.py,
DetectionPanelTriple) into an actual window the operator can launch
and test against real hardware.

Runs as a NEW, SEPARATE app from the existing, still-live
main_gui_rgb.py (the 2-camera Imaging and Spraying Dashboard) -- zero
import dependency between the two beyond genuinely shared, camera-
count-agnostic infrastructure (GantryPanel/GantryController for the
Arduino/nozzle/pump connection, theme_manager, UnifiedLog). Both apps
can be run independently; nothing here modifies or risks the working
2-camera app.

Usage:
    python3 main_gui_triple.py

Deliberately scoped for this first pass, matching integration steps
1 and 2's own scoping notes -- ported the essential flow (connect
Arduino, start cameras, arm/stop/e-stop detection, live 3-way display
with overlay, real nozzle firing) and left out for a later pass: the
fullscreen popout, session metadata/provenance/report logging, manual
pump prime/purge controls, and Data Collection-style manual image
capture. None of these block validating the core camera -> zone ->
nozzle pipeline on real hardware, which is what this app exists to
let the operator do.
"""

import os
# Must be set BEFORE PyQt5/QApplication ever initializes -- see
# offline_review_gui.py's identical comment for the full explanation
# (Qt's GTK platform theme integration otherwise tries to reach a
# GVFS daemon at startup and prints a warning on every attempt).
os.environ.setdefault("GIO_USE_VFS", "local")

import sys

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QScrollArea, QPushButton, QComboBox, QLabel, QAction, QActionGroup,
    QMessageBox, QDialog, QTabWidget,
)
from PyQt5.QtCore import QTimer

from gui.theme_manager import theme_manager
from gui.shared_log import UnifiedLog
from gui.style import LED, _muted, _divider
from gui.panels.gantry_panel import GantryPanel, LightWidget, MotorPsuWidget
from gui.panels.triple_camera_panel import TripleCameraPanel
from gui.panels.detection_panel_triple import DetectionPanelTriple
from gui.panels.triple_capture_panel import TripleCapturePanel
from gui.panels.spray_panel import SprayPanel
from gui.panels.navigation_panel_rgb import NavigationPanelRGB
from gui.tabs.tab_analysis_triple import AnalysisTabTriple
from gui.panels.acquisition_panel_rgb import CameraSettingsWidget
from core.triple_emeet_camera import CAM1_DEVICE, CAM2_DEVICE, CAM3_DEVICE


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ABEN — Triple RGB Imaging & Spraying (Standalone)")
        # Matches main_gui_rgb.py's own setMinimumSize()+resize() pair
        # (that app: 1400x820 minimum, 1600x900 default) -- sized
        # somewhat larger here since this app's top toolbar row packs
        # in more controls (Arduino connect, Camera Settings, Start
        # Cameras, View+Fullscreen, the Arm/Stop/E-Stop bar, and AUX
        # Light+Motor PSU, all in that one row) and needs more width
        # to lay out without wrapping/overflowing.
        self.setMinimumSize(1600, 900)
        self.resize(1700, 1000)

        self._sys_log = UnifiedLog()

        self.gantry = GantryPanel(self._sys_log)
        self.camera = TripleCameraPanel(self._sys_log)
        self.detect = DetectionPanelTriple(
            self._sys_log, self.camera,
            gantry_ctrl_ref=lambda: self.gantry.ctrl)
        # Lets the fullscreen popout show its own Arm/Stop/E-Stop bar
        # (see TripleCameraPanel.open_fullscreen_view()).
        self.camera.detection_tab_ref = self.detect

        # Spray panel (system checks, manual spray, nozzle test, demo)
        # -- gui/panels/spray_panel.py's SprayPanel needed ZERO changes
        # to reuse here: it already has no left/right or any other
        # camera-count-specific dependency (its "Nozzles N1/N2/N3"
        # check was always 3-nozzle, even in the 2-camera system,
        # since there were always 3 physical nozzles). Shares the same
        # gantry controller and camera panel as the rest of the app.
        self.spray = SprayPanel(
            self._sys_log,
            gantry_ctrl_ref=lambda: self.gantry.ctrl,
            camera_ref=lambda: self.camera)
        self.gantry.state_signal.connect(self.spray.update_state)

        # Navigation panel -- gui/panels/navigation_panel_rgb.py's
        # NavigationPanelRGB, also reused unchanged. ros_bridge_ref
        # resolves to whatever ROSBridge DetectionPanelTriple
        # currently holds (None until armed, same as the 2-camera
        # system's own pattern of only having odometry while armed).
        # show_spray_mission=True: unlike the 2-camera system (which
        # needs two separate NavigationPanelRGB instances, one per tab,
        # and only shows Spray Mission on the one living in Detection),
        # this app has ONE shared instance and ONE shared self.detect,
        # so the Spray Mission section is always meaningful here.
        self.nav = NavigationPanelRGB(
            self._sys_log,
            ros_bridge_ref=lambda: self.detect._odom,
            show_spray_mission=True)

        self.capture = TripleCapturePanel(self._sys_log, self.camera)
        self.analysis = AnalysisTabTriple(self.gantry, self.detect)

        # Camera Settings -- global, one v4l2 configuration applied to
        # all 3 cameras (see gui/panels/acquisition_panel_rgb.py's
        # CameraSettingsWidget, already fully generic for any device
        # count -- confirmed in Phase 5 of the triple-camera redesign).
        # Reachable from the toolbar via a single dialog, same
        # convention as main_gui_rgb.py's own "Camera Settings" button
        # -- not embedded per-tab. The "Edit Presets" button and full
        # preset editor dialog are already built into
        # CameraSettingsWidget itself, so nothing extra is needed here
        # to get preset editing too.
        self.camera_settings = CameraSettingsWidget(
            [CAM1_DEVICE, CAM2_DEVICE, CAM3_DEVICE], self._sys_log)
        self._camera_settings_dialog = None

        self._build_ui()
        theme_manager.register_widget(
            self, lambda p: f"background-color:{p['bg']};")
        self._build_menu()

        # Periodic Arduino status/port-combo refresh -- lightweight,
        # matches main_gui_rgb.py's own header refresh cadence.
        self._status_timer = QTimer()
        self._status_timer.timeout.connect(self._refresh_header)
        self._status_timer.start(500)
        self._refresh_arduino_ports()

    # ── UI ─────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        # ── Row 0: System status (LEDs + checklist), matching
        # main_gui_rgb.py's own header design ──
        outer.addWidget(self._header_status())

        # ── Top toolbar: Arduino connect + Start Cameras ──
        toolbar = QHBoxLayout()
        toolbar.addWidget(_muted("Arduino:"))
        self.hdr_port_combo = QComboBox()
        self.hdr_port_combo.setMinimumWidth(140)
        toolbar.addWidget(self.hdr_port_combo)

        refresh_btn = QPushButton("↻")
        refresh_btn.setFixedWidth(28)
        refresh_btn.clicked.connect(self._refresh_arduino_ports)
        toolbar.addWidget(refresh_btn)

        self.hdr_btn_connect = QPushButton("🔌 CONNECT ARDUINO")
        theme_manager.register_button(self.hdr_btn_connect, "blue")
        self.hdr_btn_connect.clicked.connect(self._toggle_arduino)
        toolbar.addWidget(self.hdr_btn_connect)

        toolbar.addSpacing(20)

        self.hdr_btn_camera_settings = QPushButton("📷  Camera Settings")
        theme_manager.register_widget(
            self.hdr_btn_camera_settings, lambda p: (
                f"QPushButton{{background:{p['input_bg']};color:{p['text']};"
                f"border:1px solid {p['border']};border-radius:4px;"
                f"padding:4px 12px;font-family:'Noto Sans',Arial,sans-serif;"
                f"font-size:10px;}}"
                f"QPushButton:hover{{background:{p['btn_hover']};}}"))
        self.hdr_btn_camera_settings.setFixedHeight(26)
        self.hdr_btn_camera_settings.clicked.connect(
            self._open_camera_settings_dialog)
        toolbar.addWidget(self.hdr_btn_camera_settings)

        self.hdr_btn_start_camera = QPushButton("▶  START CAMERAS")
        theme_manager.register_button(self.hdr_btn_start_camera, "green")
        self.hdr_btn_start_camera.clicked.connect(self.camera.toggle_start_stop)
        self.camera._start_btns.append(self.hdr_btn_start_camera)
        toolbar.addWidget(self.hdr_btn_start_camera)

        # View mode + Fullscreen (+ live status) live in the top
        # toolbar, next to Start Cameras -- not in the Live Operation
        # tab's left column, so they're reachable regardless of which
        # sub-tab (Data Collection / Detection / Navigation) is active.
        toolbar.addWidget(self.camera.camera_control_bar())

        # Detection Arm/Stop/E-Stop bar + AUX (Light/Motor PSU) --
        # all in this SAME toolbar row, right after View/Fullscreen,
        # not a separate row below. Both need to stay visible
        # regardless of which sub-tab is active, and the Arm/Stop/
        # E-Stop bar was genuinely too cramped squeezed into the
        # 420px-wide left column (button text was truncating) --
        # the full toolbar row fixes both at once.
        toolbar.addWidget(self._detection_arm_bar())

        # AUX -- Light + Motor PSU side by side, matching the rest of
        # this toolbar's compact horizontal style. Deliberately NOT
        # GantryPanel._aux_group(): that wraps both in a QGroupBox
        # with a QVBoxLayout (one stacked ON TOP of the other, plus a
        # caption line), which reads as "vertical" the moment it's
        # dropped into a horizontal toolbar row -- not what was asked
        # for. LightWidget and MotorPsuWidget are each ALREADY a
        # single self-contained horizontal row internally (LED+label+
        # ON+OFF, see gantry_panel.py) -- constructing them directly
        # from self.gantry.ctrl and placing them as two ordinary
        # toolbar widgets, side by side, gets genuinely horizontal
        # placement with no GantryPanel changes needed.
        self.aux_light = LightWidget(self.gantry.ctrl)
        toolbar.addWidget(self.aux_light)
        self.aux_motor_psu = MotorPsuWidget(self.gantry.ctrl)
        toolbar.addWidget(self.aux_motor_psu)
        self.gantry.state_signal.connect(
            lambda s: self.aux_light.update_state(s.light_on))
        self.gantry.state_signal.connect(
            lambda s: self.aux_motor_psu.update_state(s.motor_psu_on))

        toolbar.addStretch()

        self.lbl_arduino_status = QLabel("Arduino: disconnected")
        theme_manager.register_widget(
            self.lbl_arduino_status, lambda p: (
                f"color:{p['muted']};font-size:10px;"
                f"font-family:'Noto Sans',Arial,sans-serif;"))
        toolbar.addWidget(self.lbl_arduino_status)
        outer.addLayout(toolbar)

        # ── Main area: "Live Operation" (controls + camera view) and
        # "Session Analysis" (full-width, needs the room for its
        # wide event-feed table) as top-level tabs, rather than
        # cramming the analysis table into the 420px-wide left
        # column alongside Detection/Data Collection. ──
        main_tabs = QTabWidget()

        live_tab = QWidget()
        split = QHBoxLayout(live_tab)
        split.setContentsMargins(0, 0, 0, 0)

        left_col = QWidget()
        left_col.setMaximumWidth(420)
        left_lay = QVBoxLayout(left_col)
        left_lay.setContentsMargins(0, 0, 0, 0)

        left_tabs = QTabWidget()
        left_tabs.addTab(self.capture, "💾 Data Collection")

        # Detection sub-tab: inner Spray | Detect sub-tabs. The Arm/
        # Stop/E-Stop bar itself no longer lives here -- moved to the
        # top-level header (see _build_ui()'s new Row 2) so it's
        # visible regardless of which top-level sub-tab is active,
        # not just while "Detection" happens to be selected, and has
        # the full window width to work with instead of being
        # squeezed into this 420px column.
        detection_tabs = QTabWidget()
        detection_tabs.addTab(self.spray,  "💉 Spray")
        detection_tabs.addTab(self.detect, "🎯 Detect")
        left_tabs.addTab(detection_tabs, "🎯 Detection")

        # Navigation sub-tab -- AUX (Light / Motor PSU) no longer
        # lives here either, moved to the top-level header for the
        # same always-visible reason (see _build_ui()'s new Row 2).
        left_tabs.addTab(self.nav, "🧭 Navigation")

        left_lay.addWidget(left_tabs)

        split.addWidget(left_col)
        split.addWidget(self.camera.display_widget(), stretch=1)

        main_tabs.addTab(live_tab, "🎥 Live Operation")
        main_tabs.addTab(self.analysis, "📊 Session Analysis")
        outer.addWidget(main_tabs, stretch=1)

        self._sys_log.log("SYS", "Triple-camera app ready", "ok")

    def _detection_arm_bar(self) -> QWidget:
        """
        Always-visible ARM/STOP/E-STOP bar above the Detection
        sub-tab's Spray | Detect inner tabs -- matches
        main_gui_rgb.py's DetectionTab._detection_arm_bar(), which
        exists for exactly this reason: so the bar doesn't disappear
        depending on which inner sub-tab is selected. Calls
        self.detect's private _det_start()/_det_stop()/_det_estop()
        directly (DetectionPanelTriple's own btn_arm/btn_stop/
        btn_estop are still constructed there for internal state
        tracking, just no longer added to its own visible layout --
        see detection_panel_triple.py's _build_ui() comment).
        """
        w = QWidget()
        theme_manager.register_widget(w, lambda p: f"background-color:{p['bg0']};")
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

        self.arm_led = LED(14)
        lay.addWidget(self.arm_led)
        self.arm_status = _muted("DISARMED")
        lay.addWidget(self.arm_status)

        # Sync to whatever the real armed state already is (in case
        # this bar is ever rebuilt after detection was already armed)
        # and keep it in sync going forward via the real signal.
        self._sync_arm_bar(self.detect.is_armed())
        self.detect.armed_changed.connect(self._sync_arm_bar)

        return w

    def _sync_arm_bar(self, armed: bool):
        self.btn_arm_start.setEnabled(not armed)
        theme_manager.register_button(
            self.btn_arm_start, "dim_green" if armed else "green")
        self.btn_arm_stop.setEnabled(armed)
        theme_manager.register_button(
            self.btn_arm_stop, "red" if armed else "dim_red")
        self.arm_led.set_state(armed, role="amber")
        self.arm_status.setText("ARMED" if armed else "DISARMED")
        theme_manager.register_widget(
            self.arm_status, lambda p, _armed=armed: (
                f"color:{p['amber'] if _armed else p['muted']};"
                f"font-size:10px;font-family:'Noto Sans',Arial,sans-serif;"))

    def _on_arm(self):
        self.detect._det_start()

    def _on_stop(self):
        self.detect._det_stop()

    def _on_estop(self):
        self.detect._det_estop()
        self.spray.emergency_stop()

    def _header_status(self) -> QWidget:
        """
        System status LEDs + checklist + summary line, matching
        main_gui_rgb.py's own header design (Gantry/Camera/Detect/Nav
        LEDs, a Camera/Arduino/Armed/Pump checklist, and an "ALL
        SYSTEMS ONLINE" / "SYSTEM NOT READY (n/4)" summary) -- all
        driven from _refresh_header(), the same 500ms timer already
        polling Arduino/port state, no new polling added.
        """
        row = QWidget()
        theme_manager.register_widget(row, lambda p: f"background-color:{p['bg0']};")
        lay = QHBoxLayout(row)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(10)

        for led_attr, lbl_text, palette_key in [
            ("led_gantry", "GANTRY", "green"),
            ("led_camera", "CAMERA", "blue"),
            ("led_detect", "DETECT", "amber"),
            ("led_nav",    "NAV",    "purple"),
        ]:
            led = LED(12)
            lbl = QLabel(lbl_text)
            theme_manager.register_widget(
                lbl, lambda p, k=palette_key: (
                    f"color:{p[k]};font-size:10px;"
                    f"font-family:'Noto Sans',Arial,sans-serif;font-weight:bold;"
                    f"margin-right:6px;"))
            lay.addWidget(led)
            lay.addWidget(lbl)
            setattr(self, led_attr, led)

        lay.addWidget(_divider())

        def _mk_check_label(initial_text):
            lbl = QLabel(initial_text)
            theme_manager.register_widget(
                lbl, lambda p: (
                    f"color:{p['amber']};font-size:10px;"
                    f"font-family:'Noto Sans',Arial,sans-serif;font-weight:bold;"))
            return lbl

        self.chk_camera  = _mk_check_label("⚠  Camera")
        self.chk_arduino = _mk_check_label("⚠  Arduino")
        self.chk_armed   = _mk_check_label("⚠  Armed")
        self.chk_pump    = _mk_check_label("⚠  Pump")
        for lbl in (self.chk_camera, self.chk_arduino,
                   self.chk_armed, self.chk_pump):
            lay.addWidget(lbl)

        lay.addStretch()

        self.system_summary = QLabel("⚠  SYSTEM NOT READY")
        theme_manager.register_widget(
            self.system_summary, lambda p: (
                f"color:{p['amber']};font-size:11px;"
                f"font-family:'Noto Sans',Arial,sans-serif;font-weight:bold;"))
        lay.addWidget(self.system_summary)

        return row

    def _build_menu(self):
        mb = self.menuBar()

        fm = mb.addMenu("File")
        q = QAction("Quit", self)
        q.setShortcut("Ctrl+Q")
        q.triggered.connect(self.close)
        fm.addAction(q)

        vm = mb.addMenu("View")
        theme_menu = vm.addMenu("Theme")
        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)
        for key, label in theme_manager.list_themes():
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(key == theme_manager.current)
            action.triggered.connect(
                lambda checked, k=key: self._on_theme_selected(k))
            theme_group.addAction(action)
            theme_menu.addAction(action)

        hm = mb.addMenu("Help")
        ab = QAction("About", self)
        ab.triggered.connect(self._about)
        hm.addAction(ab)

    def _on_theme_selected(self, theme_key: str):
        theme_manager.apply(theme_key, app=QApplication.instance())
        self._sys_log.log("SYS", f"Theme changed: {theme_key}", "info")

    def _about(self):
        QMessageBox.information(
            self, "About",
            "ABEN Triple RGB Imaging & Spraying — Standalone v1.0\n\n"
            "Runs alongside the existing 2-camera dashboard "
            "(main_gui_rgb.py) with zero risk to it -- separate app, "
            "separate process.\n\n"
            "Connect the Arduino, start all 3 cameras, arm detection.\n"
            "Each camera fires its own dedicated nozzle "
            "(Cam 1→N1, Cam 2→N2, Cam 3→N3).\n\n"
            "Check the 'Static test' box in Detection to validate "
            "camera/nozzle alignment without driving the robot.")

    # ── Camera settings dialog ─────────────────────────────────

    def _open_camera_settings_dialog(self):
        """
        Open (or re-show) the global Camera Settings dialog, wrapping
        self.camera_settings -- created once and reused on subsequent
        clicks (show/raise/activate) rather than rebuilt every time,
        same convention as main_gui_rgb.py's own
        _open_camera_settings_dialog().
        """
        if self._camera_settings_dialog is None:
            dlg = QDialog(self)
            dlg.setWindowTitle("Camera Settings — applies to all 3 cameras")
            dlg.setMinimumSize(480, 640)
            theme_manager.register_widget(
                dlg, lambda p: f"background-color:{p['bg']};")
            lay = QVBoxLayout(dlg)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(self.camera_settings)
            self._camera_settings_dialog = dlg

        self._camera_settings_dialog.show()
        self._camera_settings_dialog.raise_()
        self._camera_settings_dialog.activateWindow()

    # ── Arduino connection ─────────────────────────────────────

    def _refresh_arduino_ports(self):
        ports = self.gantry.ctrl.list_ports()
        cur = self.hdr_port_combo.currentText()
        self.hdr_port_combo.blockSignals(True)
        self.hdr_port_combo.clear()
        self.hdr_port_combo.addItems(ports)
        if cur in ports:
            self.hdr_port_combo.setCurrentText(cur)
        self.hdr_port_combo.blockSignals(False)

    def _toggle_arduino(self):
        if self.gantry.ctrl.state.connected:
            self.gantry.ctrl.disconnect()
            self._sys_log.log("GANTRY", "Arduino disconnected", "info")
        else:
            port = self.hdr_port_combo.currentText()
            if not port:
                QMessageBox.warning(
                    self, "No Port",
                    "No serial port selected.\n"
                    "Connect the Arduino USB cable and click ↻ to refresh.")
                return
            self.hdr_btn_connect.setText("Connecting...")
            self.hdr_btn_connect.setEnabled(False)
            QTimer.singleShot(100, lambda: self._do_arduino_connect(port))

    def _do_arduino_connect(self, port: str):
        ok = self.gantry.ctrl.connect(port)
        self.hdr_btn_connect.setEnabled(True)
        if ok:
            self._sys_log.log("GANTRY", f"Arduino connected: {port}", "ok")
        else:
            self._sys_log.log(
                "GANTRY", f"Arduino connection failed: {port}", "error")

    def _refresh_header(self):
        connected = self.gantry.ctrl.state.connected
        self.hdr_btn_connect.setText(
            "🔌 DISCONNECT ARDUINO" if connected else "🔌 CONNECT ARDUINO")
        theme_manager.register_button(
            self.hdr_btn_connect, "red" if connected else "blue")
        self.lbl_arduino_status.setText(
            f"Arduino: {'connected — ' + port if connected else 'disconnected'}"
            if connected and (port := self.hdr_port_combo.currentText())
            else f"Arduino: {'connected' if connected else 'disconnected'}")

        # ── Status LEDs ───────────────────────────────────
        self.led_gantry.set_state(connected)
        self.led_camera.set_state(self.camera.is_acquiring, role="blue")
        armed = self.detect.is_armed()
        self.led_detect.set_state(armed, role="amber")
        bridge = self.detect._odom
        nav_connected = bridge is not None and bridge.is_connected()
        self.led_nav.set_state(nav_connected, role="purple")

        # ── System checklist ───────────────────────────────
        camera_on = self.camera.is_acquiring
        # Real Arduino-confirmed pump state, not a locally-set
        # optimistic flag -- same source main_gui_rgb.py's own
        # checklist deliberately uses (see its own comment on this
        # exact point).
        pump_on = self.gantry.ctrl.state.pump_on

        def _set_check(lbl, ok, label_text):
            lbl.setText(f"{'✓' if ok else '⚠'}  {label_text}")
            theme_manager.register_widget(
                lbl, lambda p, _ok=ok: (
                    f"color:{p['green'] if _ok else p['amber']};"
                    f"font-size:10px;"
                    f"font-family:'Noto Sans',Arial,sans-serif;"
                    f"font-weight:bold;"))

        _set_check(self.chk_camera, camera_on, "Camera")
        _set_check(self.chk_arduino, connected, "Arduino")
        _set_check(self.chk_armed, armed, "Armed")
        _set_check(self.chk_pump, pump_on, "Pump")

        all_ok = camera_on and connected and armed and pump_on
        n_ok = sum([camera_on, connected, armed, pump_on])
        if all_ok:
            self.system_summary.setText("✓  ALL SYSTEMS ONLINE")
            summary_key = "green"
        else:
            self.system_summary.setText(f"⚠  SYSTEM NOT READY  ({n_ok}/4)")
            summary_key = "amber"
        theme_manager.register_widget(
            self.system_summary, lambda p, k=summary_key: (
                f"color:{p[k]};font-size:11px;"
                f"font-family:'Noto Sans',Arial,sans-serif;font-weight:bold;"))

    # ── Cleanup ───────────────────────────────────────────────

    def closeEvent(self, event):
        self._status_timer.stop()
        self.detect.cleanup()
        self.capture.cleanup()
        self.analysis.cleanup()
        self.nav.cleanup()
        self.camera.cleanup()
        try:
            self.gantry.ctrl.disconnect()
        except Exception:
            pass
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    theme_manager.load()
    theme_manager.apply(theme_manager.current, app=app, save=False)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
