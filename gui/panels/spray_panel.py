"""
gui/panels/spray_panel.py
ABEN Field Imaging System — Spray & System Check Panel

Used in: Tab 2 (Detection) → Spray subtab

Sections:
  1. System Checks     — 6-point pre-field verification
  2. Manual Controls   — pump + nozzles manual on/off
  3. Nozzle Test       — individual 0.8s pulse per nozzle
  4. Spray Demo        — drive + random fire demo
"""

import math
import json
import time
import random
import socket
import threading
import subprocess

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QGroupBox, QDoubleSpinBox,
    QProgressBar, QPlainTextEdit, QScrollArea, QFrame
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QTextCursor

from gui.style import LED, _divider, _muted
from gui.theme_manager import theme_manager
from gui.shared_log import UnifiedLog
from core.gantry_controller import GantryController, GantryState

HUSKY_IP   = "192.168.131.1"
HUSKY_USER = "administrator"
ODOM_PORT  = 5005


# ─────────────────────────────────────────────────────────────
#  CHECK ROW WIDGET
# ─────────────────────────────────────────────────────────────
class CheckRow(QWidget):
    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 1, 4, 1)
        lay.setSpacing(8)
        self.led = LED(12)
        lay.addWidget(self.led)
        self.name_lbl = QLabel(name)
        theme_manager.register_widget(
            self.name_lbl, lambda p: (
                f"color:{p['text']};font-family:Courier New;"
                f"font-size:10px;font-weight:bold;"))
        self.name_lbl.setMinimumWidth(140)
        lay.addWidget(self.name_lbl)
        self.status_lbl = QLabel("—")
        self.status_lbl.setStyleSheet(
            f"color:{theme_manager.palette()['muted']};"
            f"font-family:Courier New;font-size:10px;")
        lay.addWidget(self.status_lbl, stretch=1)

    def set_pending(self, msg="Checking..."):
        self.led.set_state(False, role="green")
        self.status_lbl.setText(msg)
        p = theme_manager.palette()
        self.status_lbl.setStyleSheet(
            f"color:{p['amber']};font-family:Courier New;font-size:10px;")

    def set_pass(self, msg="OK"):
        self.led.set_state(True, role="green")
        self.status_lbl.setText(f"✓  {msg}")
        p = theme_manager.palette()
        self.status_lbl.setStyleSheet(
            f"color:{p['green']};font-family:Courier New;font-size:10px;")

    def set_fail(self, msg="FAIL"):
        self.led.set_state(True, role="red")
        self.status_lbl.setText(f"✗  {msg}")
        p = theme_manager.palette()
        self.status_lbl.setStyleSheet(
            f"color:{p['red']};font-family:Courier New;font-size:10px;")

    def set_warn(self, msg="WARNING"):
        self.led.set_state(True, role="amber")
        self.status_lbl.setText(f"⚠  {msg}")
        p = theme_manager.palette()
        self.status_lbl.setStyleSheet(
            f"color:{p['amber']};font-family:Courier New;font-size:10px;")

    def reset(self):
        self.led.set_state(False, role="green")
        self.status_lbl.setText("—")
        p = theme_manager.palette()
        self.status_lbl.setStyleSheet(
            f"color:{p['muted']};font-family:Courier New;font-size:10px;")


# ─────────────────────────────────────────────────────────────
#  NOZZLE WIDGET (manual control row)
# ─────────────────────────────────────────────────────────────
class NozzleWidget(QFrame):
    def __init__(self, index: int, ctrl_ref, parent=None):
        super().__init__(parent)
        self.index    = index
        self.ctrl_ref = ctrl_ref
        theme_manager.register_widget(
            self, lambda p: f"background-color:{p['bg3']};border-radius:3px;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 3, 8, 3)
        lay.setSpacing(8)
        self.led = LED(12)
        lay.addWidget(self.led)
        lb = QLabel(f"N{index+1}")
        theme_manager.register_widget(
            lb, lambda p: (
                f"color:{p['text']};font-family:'Courier New';"
                f"font-weight:bold;font-size:11px;background:transparent;"))
        lay.addWidget(lb)
        lay.addStretch()
        self.btn_on = QPushButton("ON")
        theme_manager.register_button(self.btn_on, "green")
        self.btn_on.setFixedWidth(50)
        self.btn_on.clicked.connect(self._on)
        self.btn_off = QPushButton("OFF")
        theme_manager.register_button(self.btn_off, "red")
        self.btn_off.setFixedWidth(50)
        self.btn_off.clicked.connect(self._off)
        lay.addWidget(self.btn_on)
        lay.addWidget(self.btn_off)

    def _ctrl(self): return self.ctrl_ref()
    def _on(self):   self._ctrl().send_command(f"n{self.index+1} on")
    def _off(self):  self._ctrl().send_command(f"n{self.index+1} off")

    def update_state(self, on: bool):
        self.led.set_state(on, role="green")
        theme_manager.register_button(self.btn_on, "green" if on else "dim_green")
        theme_manager.register_button(self.btn_off, "dim_red" if on else "red")

    def set_enabled_controls(self, enabled: bool):
        self.btn_on.setEnabled(enabled)
        self.btn_off.setEnabled(enabled)


# ─────────────────────────────────────────────────────────────
#  SPRAY PANEL
# ─────────────────────────────────────────────────────────────
class SprayPanel(QWidget):
    """
    Combined hardware verification + manual spray control panel.
    gantry_ctrl_ref: callable → GantryController
    camera_ref:      callable → CameraPanel
    """

    state_signal   = pyqtSignal(object)
    _log_signal    = pyqtSignal(str)
    _check_signal  = pyqtSignal(str, str, str)
    _progress_signal = pyqtSignal(int, str)

    def __init__(self, shared_log: UnifiedLog,
                 gantry_ctrl_ref,
                 camera_ref=None,
                 parent=None):
        super().__init__(parent)
        self.shared_log      = shared_log
        self.gantry_ctrl_ref = gantry_ctrl_ref
        self.camera_ref      = camera_ref or (lambda: None)

        self._demo_running   = False
        self._demo_thread    = None

        self.state_signal.connect(self._apply_state)
        self._log_signal.connect(self._append_log)
        self._check_signal.connect(self._update_check)
        self._progress_signal.connect(self._update_progress)

        self._build_ui()

    def _ctrl(self) -> GantryController:
        return self.gantry_ctrl_ref()

    # ── Build UI ──────────────────────────────────────────────

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea{border:none;}")

        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        lay.addWidget(self._checks_grp())
        lay.addWidget(_divider())
        lay.addWidget(self._manual_grp())
        lay.addWidget(self._nozzle_test_grp())
        lay.addWidget(_divider())
        lay.addWidget(self._demo_grp())
        lay.addWidget(self._check_log_grp())
        lay.addStretch()

        scroll.setWidget(inner)
        outer.addWidget(scroll)

    # ── Section: System Checks ────────────────────────────────

    def _checks_grp(self):
        grp = QGroupBox("System Checks")
        lay = QVBoxLayout(grp)
        lay.setSpacing(2)

        self.checks = {}
        for key, name in [
            ("arduino",  "Arduino"),
            ("camera",   "Camera"),
            ("ssh",      "Husky SSH"),
            ("odom",     "Odometry UDP"),
            ("pump",     "Pump"),
            ("nozzles",  "Nozzles N1/N2/N3"),
        ]:
            row = CheckRow(name)
            self.checks[key] = row
            lay.addWidget(row)

        lay.addWidget(_divider())

        self.lbl_overall = QLabel("Run checks before arming")
        theme_manager.register_widget(
            self.lbl_overall, lambda p: (
                f"color:{p['muted']};font-size:10px;"
                f"font-family:Courier New;font-weight:bold;"))
        self.lbl_overall.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.lbl_overall)

        self.btn_run_checks = QPushButton("▶  RUN ALL CHECKS")
        theme_manager.register_button(self.btn_run_checks, "teal")
        self.btn_run_checks.setMinimumHeight(30)
        self.btn_run_checks.clicked.connect(self._run_checks)
        lay.addWidget(self.btn_run_checks)
        return grp

    # ── Section: Manual Controls ──────────────────────────────

    def _manual_grp(self):
        grp = QGroupBox("Manual Controls")
        lay = QVBoxLayout(grp)
        lay.setSpacing(4)

        # Pump
        pump_row = QHBoxLayout()
        self.led_pump = LED(14)
        pump_row.addWidget(self.led_pump)
        self.lbl_pump = QLabel("PUMP  OFF")
        self.lbl_pump.setStyleSheet(
            f"color:{theme_manager.palette()['red']};font-size:10px;"
            f"font-weight:bold;font-family:Courier New;")
        pump_row.addWidget(self.lbl_pump)
        pump_row.addStretch()
        self.btn_pump_on = QPushButton("PUMP ON")
        theme_manager.register_button(self.btn_pump_on, "dim_green")
        self.btn_pump_on.setMinimumHeight(26)
        self.btn_pump_on.clicked.connect(
            lambda: self._ctrl().send_command("pump on"))
        pump_row.addWidget(self.btn_pump_on)
        self.btn_pump_off = QPushButton("PUMP OFF")
        theme_manager.register_button(self.btn_pump_off, "red")
        self.btn_pump_off.setMinimumHeight(26)
        self.btn_pump_off.clicked.connect(
            lambda: self._ctrl().send_command("pump off"))
        pump_row.addWidget(self.btn_pump_off)
        lay.addLayout(pump_row)

        # Nozzles
        self.nozzle_widgets = []
        for i in range(3):
            nw = NozzleWidget(i, self.gantry_ctrl_ref)
            lay.addWidget(nw)
            self.nozzle_widgets.append(nw)

        # All on/off
        all_row = QHBoxLayout()
        aon = QPushButton("ALL  ON")
        theme_manager.register_button(aon, "green")
        aon.setMinimumHeight(26)
        aon.clicked.connect(
            lambda: self._ctrl().send_command("na on"))
        all_row.addWidget(aon)
        aoff = QPushButton("ALL  OFF")
        theme_manager.register_button(aoff, "red")
        aoff.setMinimumHeight(26)
        aoff.clicked.connect(
            lambda: self._ctrl().send_command("na off"))
        all_row.addWidget(aoff)
        lay.addLayout(all_row)
        return grp

    # ── Section: Nozzle Test ──────────────────────────────────

    def _nozzle_test_grp(self):
        grp = QGroupBox("Nozzle Pulse Test  (0.8s each)")
        lay = QVBoxLayout(grp)
        btn_row = QHBoxLayout()
        for n in [1, 2, 3]:
            btn = QPushButton(f"Test N{n}")
            theme_manager.register_button(btn, "blue")
            btn.setMinimumHeight(26)
            btn.clicked.connect(
                lambda _, nn=n: self._test_nozzle(nn))
            btn_row.addWidget(btn)
        lay.addLayout(btn_row)
        return grp

    # ── Section: Spray Demo ───────────────────────────────────

    def _demo_grp(self):
        grp = QGroupBox("Spray Demo — Drive & Fire")
        lay = QVBoxLayout(grp)

        cfg = QHBoxLayout()
        cfg.addWidget(_muted("Distance (m):"))
        self.spn_dist = QDoubleSpinBox()
        self.spn_dist.setRange(0.5, 10.0)
        self.spn_dist.setValue(1.0)
        self.spn_dist.setSingleStep(0.5)
        cfg.addWidget(self.spn_dist)
        cfg.addStretch()
        lay.addLayout(cfg)

        self.demo_progress = QProgressBar()
        self.demo_progress.setRange(0, 100)
        self.demo_progress.setValue(0)
        self.demo_progress.setFormat("Ready")
        lay.addWidget(self.demo_progress)

        btn_row = QHBoxLayout()
        self.btn_demo = QPushButton("▶  RUN SPRAY DEMO")
        theme_manager.register_button(self.btn_demo, "green")
        self.btn_demo.setMinimumHeight(30)
        self.btn_demo.clicked.connect(self._run_demo)
        btn_row.addWidget(self.btn_demo)
        self.btn_demo_stop = QPushButton("■  STOP")
        theme_manager.register_button(self.btn_demo_stop, "dim_red")
        self.btn_demo_stop.setMinimumHeight(30)
        self.btn_demo_stop.setEnabled(False)
        self.btn_demo_stop.clicked.connect(self._stop_demo)
        btn_row.addWidget(self.btn_demo_stop)
        lay.addLayout(btn_row)
        return grp

    # ── Section: Check log ────────────────────────────────────

    def _check_log_grp(self):
        grp = QGroupBox("Check / Demo Log")
        lay = QVBoxLayout(grp)
        self.check_log = QPlainTextEdit()
        self.check_log.setReadOnly(True)
        self.check_log.setMaximumBlockCount(200)
        self.check_log.setFixedHeight(110)
        theme_manager.register_widget(
            self.check_log, lambda p: (
                f"background:{p['bg0']};color:{p['muted']};"
                f"font-family:Courier New;font-size:9px;border:none;"))
        lay.addWidget(self.check_log)
        return grp

    # ── State update from GantryPanel ────────────────────────

    def update_state(self, s: GantryState):
        self.state_signal.emit(s)

    def _apply_state(self, s: GantryState):
        p = theme_manager.palette()
        # Pump
        self.led_pump.set_state(s.pump_on, role="green")
        self.lbl_pump.setText("PUMP  ON" if s.pump_on else "PUMP  OFF")
        c = p["green"] if s.pump_on else p["red"]
        self.lbl_pump.setStyleSheet(
            f"color:{c};font-size:10px;"
            f"font-weight:bold;font-family:Courier New;")
        theme_manager.register_button(
            self.btn_pump_on, "green" if s.pump_on else "dim_green")
        theme_manager.register_button(
            self.btn_pump_off, "dim_red" if s.pump_on else "red")
        # Nozzles
        for i, nw in enumerate(self.nozzle_widgets):
            on = s.nozzles[i] if i < len(s.nozzles) else False
            nw.update_state(on)

    # ── Signal receivers ──────────────────────────────────────

    def _update_check(self, key, state, msg):
        row = self.checks.get(key)
        if not row: return
        if state == "pass":   row.set_pass(msg)
        elif state == "fail": row.set_fail(msg)
        elif state == "warn": row.set_warn(msg)
        elif state == "pend": row.set_pending(msg)

    def _ck(self, key, state, msg):
        self._check_signal.emit(key, state, msg)

    def _update_progress(self, pct, msg):
        self.demo_progress.setValue(pct)
        self.demo_progress.setFormat(msg)

    def _append_log(self, msg):
        ts = time.strftime("%H:%M:%S")
        cur = self.check_log.textCursor()
        cur.movePosition(QTextCursor.End)
        cur.insertText(f"[{ts}] {msg}\n")
        self.check_log.setTextCursor(cur)
        self.check_log.ensureCursorVisible()

    def _log(self, msg):
        self._log_signal.emit(msg)

    # ── Nozzle test ───────────────────────────────────────────

    def _test_nozzle(self, n: int):
        if not self._ctrl().state.connected:
            self.shared_log.log(
                "GANTRY", "Connect Arduino first", "error")
            return
        self._ctrl().send_command(f"n{n} on")
        QTimer.singleShot(
            800, lambda: self._ctrl().send_command(f"n{n} off"))
        self.shared_log.log("GANTRY", f"N{n} pulse test", "info")

    # ── Run all checks ────────────────────────────────────────

    def _run_checks(self):
        for row in self.checks.values():
            row.reset()
        self.lbl_overall.setText("Checking...")
        self.lbl_overall.setStyleSheet(
            f"color:{theme_manager.palette()['amber']};font-size:10px;"
            f"font-family:Courier New;font-weight:bold;")
        self.btn_run_checks.setEnabled(False)
        self.check_log.clear()
        self._log("=== System check started ===")
        threading.Thread(
            target=self._check_thread, daemon=True).start()

    def _check_thread(self):
        passed = failed = 0

        # 1. Arduino
        self._ck("arduino", "pend", "Checking...")
        if self._ctrl().state.connected:
            self._ck("arduino", "pass", "Connected /dev/ttyACM0")
            self._log("✓ Arduino connected")
            passed += 1
        else:
            self._ck("arduino", "fail", "Not connected")
            self._log("✗ Arduino not connected")
            failed += 1

        # 2. Camera
        self._ck("camera", "pend", "Checking...")
        cam = self.camera_ref()
        if cam and cam.is_acquiring and cam.current_frame is not None:
            model = cam.camera_model or "Camera"
            self._ck("camera", "pass", f"{model}")
            self._log(f"✓ {model} acquiring")
            passed += 1
        else:
            self._ck("camera", "fail", "Not acquiring")
            self._log("✗ Camera not acquiring")
            failed += 1

        # 3. SSH
        self._ck("ssh", "pend", "Pinging Husky...")
        try:
            r = subprocess.run(
                ['ssh', '-o', 'StrictHostKeyChecking=no',
                 '-o', 'ConnectTimeout=3',
                 f'{HUSKY_USER}@{HUSKY_IP}', 'hostname'],
                capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                self._ck("ssh", "pass", r.stdout.strip())
                self._log(f"✓ SSH OK — {r.stdout.strip()}")
                passed += 1
            else:
                self._ck("ssh", "fail", "SSH failed")
                self._log("✗ SSH failed")
                failed += 1
        except Exception as e:
            self._ck("ssh", "fail", str(e)[:35])
            self._log(f"✗ SSH error: {e}")
            failed += 1

        # 4. Odom UDP
        self._ck("odom", "pend", "Listening...")
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('', ODOM_PORT))
            sock.settimeout(3.0)
            data, _ = sock.recvfrom(1024)
            sock.close()
            msg = json.loads(data.decode())
            if msg.get('type') == 'odom':
                x, y = msg.get('x', 0), msg.get('y', 0)
                self._ck("odom", "pass", f"x={x:.2f}  y={y:.2f}")
                self._log(f"✓ Odom UDP: x={x:.3f}  y={y:.3f}")
                passed += 1
        except socket.timeout:
            self._ck("odom", "fail", "No UDP — check odom bridge")
            self._log("✗ No odom received (3s timeout)")
            failed += 1
        except OSError as e:
            if "Address already in use" in str(e):
                self._ck("odom", "pass", "Bridge active")
                self._log("✓ Odom port in use (bridge running)")
                passed += 1
            else:
                self._ck("odom", "fail", str(e)[:35])
                failed += 1

        # 5. Pump
        self._ck("pump", "pend", "Testing...")
        if self._ctrl().state.connected:
            self._ctrl().send_command("pump on")
            time.sleep(1.0)
            self._ctrl().send_command("pump off")
            self._ck("pump", "pass", "ON/OFF cycle OK")
            self._log("✓ Pump cycled (1s)")
            passed += 1
        else:
            self._ck("pump", "fail", "Arduino not connected")
            self._log("✗ Pump skipped")
            failed += 1

        # 6. Nozzles
        self._ck("nozzles", "pend", "Testing N1/N2/N3...")
        if self._ctrl().state.connected:
            self._ctrl().send_command("pump on")
            time.sleep(0.3)
            for n in [1, 2, 3]:
                self._log(f"  → N{n} open...")
                self._ctrl().send_command(f"n{n} on")
                time.sleep(0.5)
                self._ctrl().send_command(f"n{n} off")
                time.sleep(0.2)
            self._ctrl().send_command("pump off")
            self._ck("nozzles", "pass", "N1/N2/N3 cycled (3/3)")
            self._log("✓ All 3 nozzles cycled")
            passed += 1
        else:
            self._ck("nozzles", "fail", "Arduino not connected")
            self._log("✗ Nozzles skipped")
            failed += 1

        # Result
        self._log(
            f"\n=== {passed}/{passed+failed} passed ===")
        p = theme_manager.palette()
        if failed == 0:
            txt   = f"✓  ALL {passed} CHECKS PASSED — Field ready"
            style = (f"color:{p['green']};font-size:10px;"
                     f"font-family:Courier New;font-weight:bold;")
            self.shared_log.log("SYS", "All checks passed", "ok")
        elif passed > failed:
            txt   = f"⚠  {passed}/{passed+failed} — Fix {failed} issue(s)"
            style = (f"color:{p['amber']};font-size:10px;"
                     f"font-family:Courier New;font-weight:bold;")
            self.shared_log.log(
                "SYS", f"{passed}/{passed+failed} checks passed", "warn")
        else:
            txt   = f"✗  {failed} FAILURES — Not field ready"
            style = (f"color:{p['red']};font-size:10px;"
                     f"font-family:Courier New;font-weight:bold;")
            self.shared_log.log("SYS", f"{failed} checks FAILED", "error")

        self.lbl_overall.setText(txt)
        self.lbl_overall.setStyleSheet(style)
        QTimer.singleShot(
            0, lambda: self.btn_run_checks.setEnabled(True))

    # ── Spray demo ────────────────────────────────────────────

    def _run_demo(self):
        if self._demo_running:
            return
        if not self._ctrl().state.connected:
            self.shared_log.log(
                "GANTRY", "Connect Arduino first", "error")
            return
        self._demo_running = True
        self.btn_demo.setEnabled(False)
        self.btn_demo_stop.setEnabled(True)
        theme_manager.register_button(self.btn_demo_stop, "red")
        self.check_log.clear()
        self._log("=== Spray demo starting ===")
        self._demo_thread = threading.Thread(
            target=self._demo_fn,
            args=(self.spn_dist.value(),),
            daemon=True)
        self._demo_thread.start()

    def _stop_demo(self):
        self._demo_running = False
        subprocess.run(
            ['ssh', '-o', 'StrictHostKeyChecking=no',
             f'{HUSKY_USER}@{HUSKY_IP}',
             'pkill -SIGINT -f test_nav_v3.py 2>/dev/null'],
            timeout=3, capture_output=True)
        self._ctrl().send_command("na off")
        self._ctrl().send_command("pump off")
        self._progress_signal.emit(0, "Stopped")
        self._log("=== Stopped by user ===")
        self.shared_log.log("SYS", "Spray demo stopped", "warn")
        QTimer.singleShot(0, lambda: self.btn_demo.setEnabled(True))
        QTimer.singleShot(
            0, lambda: self.btn_demo_stop.setEnabled(False))
        QTimer.singleShot(
            0, lambda: theme_manager.register_button(self.btn_demo_stop, "dim_red"))

    def _demo_fn(self, dist: float):
        DRIVE_SPEED    = 0.1
        LOOK_AHEAD_M   = 0.4064    # 16 inches (camera footprint → nozzle bar)
        SPRAY_WINDOW_M = 0.15
        DETECT_INT     = (1.0, 2.5)
        STALL_TIMEOUT  = 5.0
        ZONE_NAMES     = ["Zone A (N1)", "Zone B (N2)", "Zone C (N3)"]

        # Inline odom receiver
        class _Odom:
            def __init__(self):
                self.x = self.y = 0.0
                self.home_x = self.home_y = None
                self._lock = threading.Lock()
                self._run  = True
                self._sock = socket.socket(
                    socket.AF_INET, socket.SOCK_DGRAM)
                self._sock.setsockopt(
                    socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self._sock.settimeout(0.5)
                try:
                    self._sock.bind(('', ODOM_PORT))
                except OSError:
                    pass
                threading.Thread(target=self._loop,
                                 daemon=True).start()

            def _loop(self):
                while self._run:
                    try:
                        d, _ = self._sock.recvfrom(1024)
                        m = json.loads(d.decode())
                        if m.get('type') == 'odom':
                            with self._lock:
                                if self.home_x is None:
                                    self.home_x = m['x']
                                    self.home_y = m['y']
                                self.x = m['x']
                                self.y = m['y']
                    except Exception:
                        pass

            def traveled(self):
                with self._lock:
                    if self.home_x is None: return 0.0
                    return math.sqrt(
                        (self.x-self.home_x)**2 +
                        (self.y-self.home_y)**2)

            def dist_from(self, x, y):
                with self._lock:
                    return math.sqrt(
                        (self.x-x)**2+(self.y-y)**2)

            def stop(self):
                self._run = False
                try: self._sock.close()
                except Exception: pass

        odom = _Odom()

        # Wait for odom
        self._log(f"Distance: {dist:.1f}m  Speed: {DRIVE_SPEED}m/s")
        self._log("Waiting for odometry...")
        deadline = time.time() + 8.0
        while odom.home_x is None and self._demo_running:
            if time.time() > deadline:
                self._log("✗ No odometry — check odom bridge")
                self._progress_signal.emit(0, "No odometry")
                odom.stop()
                self._demo_running = False
                QTimer.singleShot(
                    0, lambda: self.btn_demo.setEnabled(True))
                QTimer.singleShot(
                    0, lambda: self.btn_demo_stop.setEnabled(False))
                return
            time.sleep(0.1)

        self._log(f"Odom OK: ({odom.x:.3f}, {odom.y:.3f})")

        # Pump on + drive
        self._ctrl().send_command("pump on")
        time.sleep(0.5)
        self._log(f"Driving forward {dist}m...")

        nav_proc = subprocess.Popen([
            'ssh', '-o', 'StrictHostKeyChecking=no',
            f'{HUSKY_USER}@{HUSKY_IP}',
            f'source /opt/ros/noetic/setup.bash && '
            f'python3 ~/test_nav_v3.py forward {dist} '
            f'--speed {DRIVE_SPEED}'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        pending    = []
        events     = []
        start      = time.time()
        stall_t    = time.time()
        stall_last = 0.0
        last_det   = time.time()
        det_int    = random.uniform(*DETECT_INT)

        while self._demo_running:
            traveled = odom.traveled()
            pct = int(min(traveled / dist * 100, 100))
            self._progress_signal.emit(
                pct, f"{traveled:.2f}/{dist:.1f}m")

            if traveled >= dist:
                self._log(f"✓ Target: {traveled:.3f}m")
                break

            if traveled > stall_last + 0.005:
                stall_last = traveled
                stall_t    = time.time()
            elif time.time() - stall_t > STALL_TIMEOUT:
                self._log("⚠ Stall detected — aborting")
                break

            if time.time() - last_det > det_int:
                last_det = time.time()
                det_int  = random.uniform(*DETECT_INT)
                n = random.randint(1, 3)
                pending.append({
                    'nozzle': n, 'x': odom.x, 'y': odom.y,
                    'firing': False, 'done': False
                })
                self._log(
                    f"⚡ {ZONE_NAMES[n-1]}  "
                    f"travel={traveled:.2f}m  N{n} in 30cm")

            for ev in pending:
                if ev['done']: continue
                d = odom.dist_from(ev['x'], ev['y'])
                if not ev['firing'] and d >= LOOK_AHEAD_M:
                    ev['firing'] = True
                    self._ctrl().send_command(f"n{ev['nozzle']} on")
                    self._log(
                        f"✓ N{ev['nozzle']} OPEN  dist={d:.3f}m")
                    events.append({
                        'nozzle': ev['nozzle'],
                        'dist': round(d, 3),
                        'time': time.strftime("%H:%M:%S")
                    })
                elif ev['firing'] and d >= LOOK_AHEAD_M + SPRAY_WINDOW_M:
                    ev['done'] = True
                    self._ctrl().send_command(f"n{ev['nozzle']} off")
                    self._log(f"✗ N{ev['nozzle']} CLOSED")
            pending = [e for e in pending if not e['done']]
            time.sleep(0.1)

        nav_proc.terminate()
        self._ctrl().send_command("na off")
        self._ctrl().send_command("pump off")
        odom.stop()

        traveled = odom.traveled()
        duration = time.time() - start
        self._progress_signal.emit(100, f"Done — {traveled:.2f}m")
        self._log(
            f"=== Done: {traveled:.3f}m  {duration:.1f}s  "
            f"{len(events)} spray events ===")
        for i, ev in enumerate(events, 1):
            self._log(
                f"  {i}. N{ev['nozzle']}  "
                f"+{ev['dist']:.3f}m  @ {ev['time']}")

        self.shared_log.log(
            "SYS",
            f"Spray demo — {len(events)} events  "
            f"{traveled:.3f}m", "ok")

        self._demo_running = False
        QTimer.singleShot(0, lambda: self.btn_demo.setEnabled(True))
        QTimer.singleShot(
            0, lambda: self.btn_demo_stop.setEnabled(False))
        QTimer.singleShot(
            0, lambda: theme_manager.register_button(self.btn_demo_stop, "dim_red"))

    # ── Public API ────────────────────────────────────────────

    def emergency_stop(self):
        ctrl = self._ctrl()
        if ctrl:
            ctrl.send_command("na off")
            ctrl.send_command("pump off")
        self.shared_log.log(
            "GANTRY", "E-STOP — pump + nozzles off", "error")