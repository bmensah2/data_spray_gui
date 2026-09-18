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
from gui.style import _muted
from gui.panels.gantry_panel import GantryPanel
from gui.panels.triple_camera_panel import TripleCameraPanel
from gui.panels.detection_panel_triple import DetectionPanelTriple
from gui.panels.triple_capture_panel import TripleCapturePanel
from gui.panels.acquisition_panel_rgb import CameraSettingsWidget
from core.triple_emeet_camera import CAM1_DEVICE, CAM2_DEVICE, CAM3_DEVICE


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ABEN — Triple RGB Imaging & Spraying (Standalone)")
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

        self.capture = TripleCapturePanel(self._sys_log, self.camera)

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

        toolbar.addStretch()

        self.lbl_arduino_status = QLabel("Arduino: disconnected")
        theme_manager.register_widget(
            self.lbl_arduino_status, lambda p: (
                f"color:{p['muted']};font-size:10px;"
                f"font-family:'Noto Sans',Arial,sans-serif;"))
        toolbar.addWidget(self.lbl_arduino_status)
        outer.addLayout(toolbar)

        # ── Main split: controls (left) | live camera view (right) ──
        split = QHBoxLayout()

        left_col = QWidget()
        left_col.setMaximumWidth(420)
        left_lay = QVBoxLayout(left_col)
        left_lay.setContentsMargins(0, 0, 0, 0)

        left_tabs = QTabWidget()
        left_tabs.addTab(self.detect,  "🎯 Detection")
        left_tabs.addTab(self.capture, "💾 Data Collection")
        left_lay.addWidget(left_tabs)

        left_lay.addWidget(self.camera.camera_control_bar())

        gantry_scroll = QScrollArea()
        gantry_scroll.setWidgetResizable(True)
        gantry_scroll.setWidget(self.gantry)
        left_lay.addWidget(gantry_scroll, stretch=1)

        split.addWidget(left_col)
        split.addWidget(self.camera.display_widget(), stretch=1)
        outer.addLayout(split, stretch=1)

        self._sys_log.log("SYS", "Triple-camera app ready", "ok")

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

    # ── Cleanup ───────────────────────────────────────────────

    def closeEvent(self, event):
        self._status_timer.stop()
        self.detect.cleanup()
        self.capture.cleanup()
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
