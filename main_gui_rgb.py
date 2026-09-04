"""
main_gui_v3.py
Imaging and Spraying Dashboard — v1.0

5-Tab modular architecture:
  Tab 1: Data Collection   — Gantry | Camera | Navigation | Data Acquisition
  Tab 2: Detection         — Spray  | Camera | Navigation | Detection and Spray Mode
  Tab 3: Session Analysis     — Spray Logs | Detection Logs | Data Analysis
  Tab 4: Future
  Tab 5: Future

Shared across tabs (single instance):
  CameraPanel     — one GenTL connection
  NavigationPanel — one SSH connection to Husky
  GantryPanel     — one Arduino serial connection

Launch:
    cd ~/phd_project/emeet_dual_rgb
    python main_gui_rgb.py
"""

import sys
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QPushButton, QAction, QActionGroup, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer

from gui.style import LED, _divider, _muted
from gui.theme_manager import theme_manager, _darken, _lighten
from gui.shared_log import UnifiedLog
from gui.keyboard_nav import KeyboardNav
from gui.panels.dual_camera_panel import DualCameraPanel as CameraPanel
from gui.panels.gantry_panel import GantryPanel
from gui.panels.navigation_panel_rgb import NavigationPanelRGB as NavigationPanel


import sys
import types
from gui.panels.acquisition_panel_rgb import AcquisitionPanelRGB
_acq_shim = types.ModuleType("gui.panels.acquisition_panel")
_acq_shim.AcquisitionPanel = AcquisitionPanelRGB
sys.modules["gui.panels.acquisition_panel"] = _acq_shim

# Patch DetectionPanel → DetectionPanelRGB
from gui.panels.detection_panel_rgb import DetectionPanelRGB
_det_shim = types.ModuleType("gui.panels.detection_panel")
_det_shim.DetectionPanel = DetectionPanelRGB
sys.modules["gui.panels.detection_panel"] = _det_shim


# ──────────────────────────────────────────────────────────────────

from gui.tabs.tab_collection import CollectionTab
from gui.tabs.tab_detection import DetectionTab
from gui.tabs.tab_analysis_rgb import AnalysisTabRGB as AnalysisTab
try:
    import realsense_camera as _rs_mod
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "IMAGING AND SPRAYING DASHBOARD V1.0")
        self.setMinimumSize(1400, 820)
        self.resize(1600, 900)
        self._build_shared()
        self._build_ui()
        self._build_menu()

    # ── Shared objects (created once) ─────────────────────────

    def _build_shared(self):
        """
        Instantiate shared panels before tabs.
        All tabs reference the same camera, nav, gantry.
        """
        # Single unified log for system-wide messages
        self._sys_log = UnifiedLog()

        # Camera — one GenTL connection, shared producer thread
        self.camera = CameraPanel(self._sys_log)
        self._rs_proc = None   # RealSense subprocess handle
        self.kb_nav  = KeyboardNav(self._sys_log)

        # Gantry — one Arduino serial connection
        self.gantry = GantryPanel(self._sys_log)

        # Navigation panels — one per tab (Qt widgets can't share parents)
        # Both connect to the same Husky via SSH
        self.nav_collection = NavigationPanel(self._sys_log)
        self.nav_detection  = NavigationPanel(self._sys_log)


        # Tab 1 owns the full widget; Tab 2 uses only the camera subtab.
        self.acq = AcquisitionPanelRGB(self._sys_log, self.camera)

    # ── Main UI ───────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        # ── Header bar ────────────────────────────────────────
        lay.addWidget(self._header())
        lay.addWidget(_divider())

        # ── Main tab widget ───────────────────────────────────
        self.main_tabs = QTabWidget()
        theme_manager.register_widget(
            self.main_tabs, lambda p: (
                "QTabBar::tab {"
                "padding: 8px 20px;"
                "font-size: 11px;"
                "min-width: 140px;"
                "}"
                f"QTabBar::tab:selected {{"
                f"color: {p['text']};"
                f"border-color: {p['blue']};"
                f"font-weight: bold;"
                f"}}"))

        # Build tabs — pass shared panels
        self.tab1 = CollectionTab(
            self.camera, self.nav_collection, self.gantry,
            acq=self.acq)
        self.tab2 = DetectionTab(
            self.camera, self.nav_detection, self.gantry,
            acq=self.acq)
        self.tab3 = AnalysisTab(self.gantry, self.tab2.detect)

        # ── Cross-tab movement lock ─────────────────────────────
        self.nav_detection.set_movement_controls_enabled(False)
        self.nav_collection.set_movement_controls_enabled(True)
        self.tab2.armed_changed.connect(self._on_detection_armed_changed)

        self.main_tabs.addTab(
            self.tab1, "📷  Data Collection")
        self.main_tabs.addTab(
            self.tab2, "🎯  Detection")
        self.main_tabs.addTab(
            self.tab3, "📊  Session Analysis")
        self.main_tabs.addTab(
            self._future_tab("Tab 4"), "🔬  Future")
        self.main_tabs.addTab(
            self._future_tab("Tab 5"), "🔬 Future")

        lay.addWidget(self.main_tabs, stretch=1)

        # Header LED refresh
        self._hdr_timer = QTimer()
        self._hdr_timer.timeout.connect(self._refresh_header)
        self._hdr_timer.start(500)

    def _on_detection_armed_changed(self, armed: bool):
        """Detection armed → lock Collection's movement controls.
        Detection disarmed/estopped → hand control back to Collection."""
        self.nav_collection.set_movement_controls_enabled(not armed)
        self._sys_log.log(
            "SYS",
            "Detection ARMED — Data Collection movement locked" if armed
            else "Detection DISARMED — Data Collection movement unlocked",
            "warn" if armed else "info")

    def _header(self) -> QWidget:
        """
        Two-row header:
          Row 1 — Title + Status LEDs + E-STOP
          Row 2 — Arduino connection bar (prominent, always visible)
        """
        container = QWidget()
        vlay = QVBoxLayout(container)
        vlay.setContentsMargins(0, 0, 0, 0)
        vlay.setSpacing(2)

        # ── Row 1: Title + LEDs + E-STOP ─────────────────────
        row1 = QWidget()
        r1lay = QHBoxLayout(row1)
        r1lay.setContentsMargins(6, 4, 6, 2)
        r1lay.setSpacing(10)

        title = QLabel("DUAL RGB IMAGING SYSTEM")
        theme_manager.register_widget(
            title, lambda p: (
                f"color:{p['text']};font-size:15px;font-weight:bold;"
                f"font-family:'Noto Sans',Arial,sans-serif;letter-spacing:3px;"))
        r1lay.addWidget(title)
        r1lay.addStretch()

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
            r1lay.addWidget(led)
            r1lay.addWidget(lbl)
            setattr(self, led_attr, led)

        # RealSense popup button
        self._rs_window = None
        rs_btn = QPushButton("📷  Depth Cam")
        theme_manager.register_widget(
            rs_btn, lambda p: (
                f"QPushButton{{background:{_darken(p['blue'],55)};"
                f"color:{_lighten(p['blue'],15)};"
                f"border:1px solid {_darken(p['blue'],30)};border-radius:4px;"
                f"padding:4px 8px;font-family:'Noto Sans',Arial,sans-serif;"
                f"font-size:10px;font-weight:bold;}}"
                f"QPushButton:hover{{background:{_darken(p['blue'],40)};}}"
                f"QPushButton:disabled{{background:{p['disabled_bg']};"
                f"color:{p['disabled_text']};border-color:{p['disabled_bg']};}}"))
        rs_btn.setMinimumHeight(30)
        rs_btn.setMinimumWidth(110)
        rs_btn.setEnabled(REALSENSE_AVAILABLE)
        rs_btn.setToolTip(
            "Open RealSense Depth Camera viewer"
            if REALSENSE_AVAILABLE
            else "Install pyrealsense2 to enable")
        rs_btn.clicked.connect(self._open_realsense)
        r1lay.addWidget(rs_btn)

        estop = QPushButton("⚡  E-STOP")
        theme_manager.register_button(estop, "estop")
        estop.setMinimumWidth(110)
        estop.setMinimumHeight(30)
        estop.clicked.connect(self._global_estop)
        r1lay.addWidget(estop)
        vlay.addWidget(row1)

        # ── Row 2: Arduino connection bar ─────────────────────
        row2 = QWidget()
        theme_manager.register_widget(
            row2, lambda p: f"background-color:{p['bg0']};")
        row2.setFixedHeight(80)
        r2lay = QHBoxLayout(row2)
        r2lay.setContentsMargins(10, 4, 10, 4)
        r2lay.setSpacing(10)

        # Warning icon + prompt
        self.arduino_warn = QLabel("⚠  Arduino not connected — connect to enable Gantry, Pump and Nozzles")
        theme_manager.register_widget(
            self.arduino_warn, lambda p: (
                f"color:{p['amber']};font-size:10px;"
                f"font-family:'Noto Sans',Arial,sans-serif;font-weight:bold;"))
        r2lay.addWidget(self.arduino_warn)
        r2lay.addStretch()

        # Port selector
        from PyQt5.QtWidgets import QComboBox
        self.hdr_port_combo = QComboBox()
        self.hdr_port_combo.setMinimumWidth(120)
        self.hdr_port_combo.setFixedHeight(26)
        theme_manager.register_widget(
            self.hdr_port_combo, lambda p: (
                f"QComboBox{{background:{p['input_bg']};color:{p['text']};"
                f"border:1px solid {p['border']};border-radius:3px;"
                f"font-family:'Noto Sans',Arial,sans-serif;font-size:10px;}}"
                f"QComboBox::drop-down{{border:none;width:16px;}}"
                f"QComboBox QAbstractItemView{{background:{p['input_bg']};"
                f"color:{p['text']};border:1px solid {p['border']};}}"))
        r2lay.addWidget(self.hdr_port_combo)

        # Refresh ports button
        btn_refresh = QPushButton("↻")
        btn_refresh.setFixedWidth(28)
        btn_refresh.setFixedHeight(26)
        theme_manager.register_widget(
            btn_refresh, lambda p: (
                f"QPushButton{{background:{p['input_bg']};color:{p['muted']};"
                f"border:1px solid {p['border']};border-radius:3px;}}"
                f"QPushButton:hover{{background:{p['btn_hover']};"
                f"color:{p['text']};}}"))
        btn_refresh.clicked.connect(self._refresh_arduino_ports)
        r2lay.addWidget(btn_refresh)

        # Connect/Disconnect button
        self.hdr_btn_connect = QPushButton("🔌  CONNECT ARDUINO")
        theme_manager.register_widget(
            self.hdr_btn_connect, lambda p: (
                f"QPushButton{{background:{_darken(p['green'],55)};"
                f"color:{_lighten(p['green'],20)};"
                f"border:1px solid {p['green']};border-radius:4px;"
                f"padding:4px 12px;font-family:'Noto Sans',Arial,sans-serif;"
                f"font-size:10px;font-weight:bold;}}"
                f"QPushButton:hover{{background:{_darken(p['green'],40)};}}"))
        self.hdr_btn_connect.setFixedHeight(26)
        self.hdr_btn_connect.clicked.connect(self._toggle_arduino)
        r2lay.addWidget(self.hdr_btn_connect)

        # Connected LED
        self.hdr_arduino_led = LED(12)
        r2lay.addWidget(self.hdr_arduino_led)

        vlay.addWidget(row2)

        # Add keyboard nav widget to row2
        r2lay.addWidget(self.kb_nav.widget())
        r2lay.addSpacing(8)

        # Install key handlers on main window (done after show())
        QTimer.singleShot(100, lambda: self.kb_nav.install(self))

        # Populate ports initially
        self._refresh_arduino_ports()

        return container

    def _open_realsense(self):
        """
        Launch RealSense viewer as a SEPARATE PROCESS.
        The viewer window appears independently and stays open until
        the user closes it. Closing main GUI does NOT kill the viewer.
        """
        if not REALSENSE_AVAILABLE:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "RealSense",
                "pyrealsense2 not installed.\n"
                "Install with: pip install pyrealsense2")
            return

        # Check if already running
        if (hasattr(self, '_rs_proc') and
                self._rs_proc is not None and
                self._rs_proc.poll() is None):
            self._sys_log.log(
                "CAMERA", "RealSense viewer already running", "info")
            return

        import subprocess, sys
        from pathlib import Path

        # Find realsense_camera.py relative to this script
        script = Path(__file__).parent / "realsense_camera.py"
        if not script.exists():
            # Try current working directory
            script = Path("realsense_camera.py")
        if not script.exists():
            self._sys_log.log(
                "CAMERA", "realsense_camera.py not found", "error")
            return

        # Launch as separate process — own GIL, own USB context
        self._rs_proc = subprocess.Popen(
            [sys.executable, str(script)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        self._sys_log.log(
            "CAMERA",
            f"RealSense viewer launched (PID {self._rs_proc.pid}) "
            f"— separate process, no GIL conflict", "ok")

        # Poll process state every 2s to update button
        self._rs_poll_timer = QTimer()
        self._rs_poll_timer.timeout.connect(self._poll_rs_proc)
        self._rs_poll_timer.start(2000)

    def _poll_rs_proc(self):
        """Check if RealSense process is still running."""
        if (hasattr(self, '_rs_proc') and
                self._rs_proc is not None and
                self._rs_proc.poll() is not None):
            # Process has exited
            self._rs_proc = None
            self._rs_poll_timer.stop()
            self._sys_log.log(
                "CAMERA", "RealSense viewer closed", "info")


    def _refresh_arduino_ports(self):
        """Populate port dropdown from GantryController.list_ports()."""
        ports = self.gantry.ctrl.list_ports()
        cur   = self.hdr_port_combo.currentText()
        self.hdr_port_combo.blockSignals(True)
        self.hdr_port_combo.clear()
        self.hdr_port_combo.addItems(ports)
        if cur in ports:
            self.hdr_port_combo.setCurrentText(cur)
        self.hdr_port_combo.blockSignals(False)

    def _toggle_arduino(self):
        """Connect or disconnect Arduino from header bar."""
        if self.gantry.ctrl.state.connected:
            # Disconnect
            self.gantry.ctrl.disconnect()
        else:
            port = self.hdr_port_combo.currentText()
            if not port:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self, "No Port",
                    "No serial port selected.\n"
                    "Connect the Arduino USB cable and click ↻ to refresh.")
                return
            self.hdr_btn_connect.setText("Connecting...")
            self.hdr_btn_connect.setEnabled(False)
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(
                100, lambda: self._do_arduino_connect(port))

    def _do_arduino_connect(self, port: str):
        ok = self.gantry.ctrl.connect(port)
        self.hdr_btn_connect.setEnabled(True)
        if ok:
            self._sys_log.log(
                "GANTRY", f"Arduino connected: {port}", "ok")
        else:
            self._sys_log.log(
                "GANTRY", f"Arduino connection failed: {port}", "error")

    def _refresh_header(self):
        try:
            arduino_connected = self.gantry.ctrl.state.connected

            # ── Status LEDs ───────────────────────────────────
            self.led_gantry.set_state(arduino_connected)
            self.led_camera.set_state(
                self.camera.is_acquiring, role="blue")
            self.led_detect.set_state(
                self.tab2.detect.is_armed, role="amber")
            bridge = self.tab2.detect.ros_bridge
            nav_connected = (bridge is not None and
                             bridge.is_connected())
            self.led_nav.set_state(nav_connected, role="purple")
            if nav_connected:
                self.nav_collection.set_ros_bridge(lambda: bridge)
                self.nav_detection.set_ros_bridge(lambda: bridge)

            # ── Arduino connection bar ────────────────────────
            self.hdr_arduino_led.set_state(
                arduino_connected, role="green")
            if arduino_connected:
                port = self.hdr_port_combo.currentText()
                mode = self.gantry.ctrl.state.firmware_mode
                if mode == "detection":
                    mode_str = "DETECTION firmware  —  Pump + Nozzles active  |  Stepper HOLDING"
                    palette_key = "blue"
                else:
                    mode_str = "UNIFIED firmware  —  Full Gantry + Pump + Nozzles active"
                    palette_key = "green"
                self.arduino_warn.setText(
                    f"✓  Arduino connected on {port}  —  {mode_str}")
                theme_manager.register_widget(
                    self.arduino_warn, lambda p, k=palette_key: (
                        f"color:{p[k]};font-size:10px;"
                        f"font-family:'Noto Sans',Arial,sans-serif;font-weight:bold;"))
                self.hdr_btn_connect.setText("🔌  DISCONNECT")
                theme_manager.register_widget(
                    self.hdr_btn_connect, lambda p: (
                        f"QPushButton{{background:{_darken(p['red'],55)};"
                        f"color:{_lighten(p['red'],10)};"
                        f"border:1px solid {p['red']};border-radius:4px;"
                        f"padding:4px 12px;font-family:'Noto Sans',Arial,sans-serif;"
                        f"font-size:10px;font-weight:bold;}}"
                        f"QPushButton:hover{{background:{_darken(p['red'],40)};}}"))
            else:
                self.arduino_warn.setText(
                    "⚠  Arduino not connected"
                    "  —  connect to enable Gantry, Pump and Nozzles")
                theme_manager.register_widget(
                    self.arduino_warn, lambda p: (
                        f"color:{p['amber']};font-size:10px;"
                        f"font-family:'Noto Sans',Arial,sans-serif;font-weight:bold;"))
                self.hdr_btn_connect.setText("🔌  CONNECT ARDUINO")
                theme_manager.register_widget(
                    self.hdr_btn_connect, lambda p: (
                        f"QPushButton{{background:{_darken(p['green'],55)};"
                        f"color:{_lighten(p['green'],20)};"
                        f"border:1px solid {p['green']};border-radius:4px;"
                        f"padding:4px 12px;font-family:'Noto Sans',Arial,sans-serif;"
                        f"font-size:10px;font-weight:bold;}}"
                        f"QPushButton:hover{{background:{_darken(p['green'],40)};}}"))
        except Exception:
            pass

    def _future_tab(self, name: str) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(20, 20, 20, 20)
        lbl = QLabel(f"{name} — Reserved for future expansion")
        theme_manager.register_widget(
            lbl, lambda p: (
                f"color:{p['dim']};font-size:14px;"
                f"font-family:'Noto Sans',Arial,sans-serif;"))
        lbl.setAlignment(Qt.AlignCenter)
        lay.addStretch()
        lay.addWidget(lbl)
        lay.addStretch()
        return w

    # ── Global E-STOP ─────────────────────────────────────────

    def _global_estop(self):
        """E-STOP accessible from any tab via header bar."""
        # Stop detection + spray
        self.tab2.detect.emergency_stop()
        self.tab2.spray.emergency_stop()
        # Stop gantry
        self.gantry.emergency_stop()
        # Stop navigation
        self.nav_collection._nav_stop()
        self.nav_detection._nav_stop()
        self._sys_log.log(
            "SYS", "GLOBAL E-STOP — all systems halted", "error")

    # ── Menu ──────────────────────────────────────────────────

    def _build_menu(self):
        mb = self.menuBar()
        # theme_manager's app-level QSS
        # (_STYLE_TEMPLATE in theme_manager.py)
        # QMenuBar/QMenu for the active theme.
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
        QMessageBox.information(self, "About",
            "ABEN Field Imaging System  v3.0\n\n"
            "Modular 5-tab architecture:\n"
            "  Tab 1 — Data Collection\n"
            "  Tab 2 — Detection & Spray\n"
            "  Tab 3 — Session Analysis\n"
            "  Tab 4 — Future\n"
            "  Tab 5 — Future\n\n"
            "Camera: eMeet C960 4K (Dual RGB)\n"
            "Robot:  Husky A200 (cpr-a200-0943)\n"
            "Author: Nana | NDSU PhD Imaging System")

    # ── Cleanup ───────────────────────────────────────────────

    def closeEvent(self, event):
        # ── Confirm close if detection is armed or pump is on ──
        armed   = self.tab2.detect.is_armed
        pump_on = self.gantry.ctrl.state.pump_on
        if armed or pump_on:
            from PyQt5.QtWidgets import QMessageBox
            what = []
            if armed:   what.append("Detection is ARMED")
            if pump_on: what.append("Pump is RUNNING")
            reply = QMessageBox.warning(
                self, "Confirm Close",
                f"{'  |  '.join(what)}\n\n"
                "Closing will stop all hardware safely.\n"
                "Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                event.ignore()
                return

        # ── Safe shutdown sequence ────────────────────────────
        self._hdr_timer.stop()

        # Stop hardware first — write directly to serial (bypass queue)
        # so commands are guaranteed sent before port closes
        if self.gantry.ctrl.state.connected:
            import time
            ser = self.gantry.ctrl._serial
            if ser and ser.is_open:
                for cmd in ["na off", "pump off", "light off", "stop"]:
                    try:
                        ser.write((cmd + "\n").encode())
                        ser.flush()
                        time.sleep(0.08)   # 80ms per command — Arduino processes at 9600 baud
                    except Exception:
                        pass
            time.sleep(0.2)   # final settle before disconnect

        # Stop detection pipeline
        self.tab2.detect.cleanup()

        # Stop nav SSH processes
        self.nav_collection._nav_kill()
        self.nav_detection._nav_kill()

        # Stop keyboard nav
        self.kb_nav.cleanup()

        # Stop RealSense subprocess if running
        if hasattr(self, '_rs_proc') and self._rs_proc is not None:
            if self._rs_proc.poll() is None:
                self._rs_proc.terminate()

        # Stop gantry worker thread BEFORE Qt destroys the panel widget
        self.gantry.ctrl.disconnect()

        # Now safe to clean up Qt panels
        self.tab1.cleanup()
        self.tab2.cleanup()
        self.tab3.cleanup()
        self.nav_collection.cleanup()
        self.nav_detection.cleanup()
        self.acq.cleanup()
        self.camera.cleanup()
        event.accept()


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # Load the operator's saved theme choice (falls back to the
    # aben_dark default if none was ever saved) and apply the
    # app-level QSS BEFORE any window/widget is constructed.
    theme_manager.load()
    theme_manager.apply(theme_manager.current, app=app, save=False)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())