"""
gui/panels/gantry_panel.py
ABEN Field Imaging System — Gantry Control Panel

Controls:
  - Stepper arm (position, speed, homing)
  - Camera servo (angle)
  - Sequence runner
  - Aux: Grow light (AL295W), Motor PSU
  - System status

Note: Pump and nozzles are in spray_panel.py (Detection tab only).

Usage:
    from gui.panels.gantry_panel import GantryPanel
    panel = GantryPanel(shared_log)
"""

import threading
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QSlider, QLineEdit,
    QGroupBox, QFrame, QScrollArea, QComboBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal

from gui.style import LED, _divider, _muted
from gui.theme_manager import theme_manager
from gui.shared_log import UnifiedLog
from core.gantry_controller import GantryController, GantryState


# ─────────────────────────────────────────────────────────────
#  SUB-WIDGETS
# ─────────────────────────────────────────────────────────────
class LightWidget(QFrame):
    """AL295W grow light ON/OFF control."""

    def __init__(self, ctrl: GantryController, parent=None):
        super().__init__(parent)
        self.ctrl = ctrl
        theme_manager.register_widget(
            self, lambda p: f"background-color:{p['bg3']};border-radius:3px;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 3, 8, 3)
        lay.setSpacing(8)
        self.led = LED(12)
        self.led.set_state(False, role="yellow")
        lay.addWidget(self.led)
        lb = QLabel("AL295W")
        theme_manager.register_widget(
            lb, lambda p: (
                f"color:{p['text']};font-family:'Noto Sans',Arial,sans-serif;"
                f"font-weight:bold;font-size:11px;background:transparent;"))
        lay.addWidget(lb)
        lay.addStretch()
        self.btn_on = QPushButton("ON")
        theme_manager.register_button(self.btn_on, "green")
        self.btn_on.setFixedWidth(50)
        self.btn_on.clicked.connect(lambda: ctrl.send_light(True))
        self.btn_off = QPushButton("OFF")
        theme_manager.register_button(self.btn_off, "red")
        self.btn_off.setFixedWidth(50)
        self.btn_off.clicked.connect(lambda: ctrl.send_light(False))
        lay.addWidget(self.btn_on)
        lay.addWidget(self.btn_off)

    def update_state(self, on: bool):
        self.led.set_state(on, role="yellow")
        theme_manager.register_button(self.btn_on, "green" if on else "dim_green")
        theme_manager.register_button(self.btn_off, "dim_red" if on else "red")


class MotorPsuWidget(QFrame):
    """Motor PSU ON/OFF with double-confirm for OFF."""

    def __init__(self, ctrl: GantryController, parent=None):
        super().__init__(parent)
        self.ctrl = ctrl
        self._confirming = False
        theme_manager.register_widget(
            self, lambda p: f"background-color:{p['bg3']};border-radius:3px;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 3, 8, 3)
        lay.setSpacing(8)
        self.led = LED(12)
        self.led.set_state(True, role="blue")
        lay.addWidget(self.led)
        lb = QLabel("12V PSU")
        theme_manager.register_widget(
            lb, lambda p: (
                f"color:{p['text']};font-family:'Noto Sans',Arial,sans-serif;"
                f"font-weight:bold;font-size:11px;background:transparent;"))
        lay.addWidget(lb)
        lay.addStretch()
        self.btn_on = QPushButton("ON")
        theme_manager.register_button(self.btn_on, "green")
        self.btn_on.setFixedWidth(50)
        self.btn_on.clicked.connect(lambda: ctrl.send_motor_psu(True))
        self.btn_off = QPushButton("OFF")
        theme_manager.register_button(self.btn_off, "dim_red")
        self.btn_off.setFixedWidth(50)
        self.btn_off.clicked.connect(self._confirm_off)
        lay.addWidget(self.btn_on)
        lay.addWidget(self.btn_off)
        self._reset_timer = QTimer()
        self._reset_timer.setSingleShot(True)
        self._reset_timer.timeout.connect(self._reset_btn)

    def _confirm_off(self):
        if self._confirming:
            self._do_off()
            return
        self._confirming = True
        self.btn_off.setText("SURE?")
        theme_manager.register_button(self.btn_off, "red")
        self._reset_timer.start(2000)

    def _do_off(self):
        self._reset_timer.stop()
        self._confirming = False
        self.ctrl.send_motor_psu(False)
        self._reset_btn()

    def _reset_btn(self):
        self._confirming = False
        self.btn_off.setText("OFF")

    def update_state(self, on: bool):
        self._confirming = False
        self._reset_timer.stop()
        self.btn_off.setText("OFF")
        self.led.set_state(on, role="blue")
        theme_manager.register_button(self.btn_on, "green" if on else "dim_green")
        theme_manager.register_button(self.btn_off, "dim_red" if on else "red")


# ─────────────────────────────────────────────────────────────
#  GANTRY PANEL
# ─────────────────────────────────────────────────────────────
class GantryPanel(QWidget):
    """
    Full gantry control panel.
    Contains: arm, servo, sequence, aux (light + PSU), status.
    Pump and nozzles are NOT included (see spray_panel.py).
    """

    state_signal = pyqtSignal(object)

    def __init__(self, shared_log: UnifiedLog, parent=None):
        super().__init__(parent)
        self.shared_log = shared_log
        self.ctrl = GantryController()
        self.ctrl.set_log_callback(self._on_log)
        self.ctrl.set_state_callback(self._on_state)
        self.state_signal.connect(self._apply_state)
        self._drag_arm   = False
        self._drag_servo = False
        self._build_ui()

    # ── Build UI ──────────────────────────────────────────────
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea{border:none;}")

        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        lay.addWidget(self._arm_group())
        lay.addWidget(self._servo_group())
        lay.addWidget(self._seq_group())
        lay.addWidget(_divider())
        lay.addWidget(self._aux_group())
        lay.addWidget(self._status_group())
        lay.addStretch()

        scroll.setWidget(inner)
        outer.addWidget(scroll)

    def _arm_group(self):
        grp = QGroupBox("STEPPER ARM")
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)

        sr = QHBoxLayout()
        self.led_homed = LED(12)
        sr.addWidget(self.led_homed)
        self.lbl_homed = QLabel("NOT HOMED")
        self.lbl_homed.setStyleSheet(
            f"color:{theme_manager.palette()['red']};font-size:10px;"
            f"font-family:'Noto Sans',Arial,sans-serif;font-weight:bold;"
        )
        sr.addWidget(self.lbl_homed)
        sr.addStretch()
        self.lbl_arm_pos = QLabel("0.00 in")
        theme_manager.register_widget(
            self.lbl_arm_pos, lambda p: (
                f"color:{p['blue']};font-size:13px;"
                f"font-weight:bold;font-family:'Noto Sans',Arial,sans-serif;"))
        sr.addWidget(self.lbl_arm_pos)
        lay.addLayout(sr)

        self.slider_arm = QSlider(Qt.Horizontal)
        self.slider_arm.setRange(0, 3700)
        self.slider_arm.sliderPressed.connect(
            lambda: setattr(self, '_drag_arm', True))
        self.slider_arm.sliderReleased.connect(self._arm_released)
        self.slider_arm.valueChanged.connect(
            lambda v: self.lbl_arm_pos.setText(f"{v/100.0:.2f} in"))
        lay.addWidget(self.slider_arm)

        rr = QHBoxLayout()
        rr.addWidget(_muted("0.00"))
        rr.addStretch()
        rr.addWidget(_muted("29.75 in"))
        lay.addLayout(rr)

        cr = QHBoxLayout()
        cr.setSpacing(4)
        self.entry_arm = QLineEdit()
        self.entry_arm.setPlaceholderText("inches")
        self.entry_arm.setFixedWidth(58)
        self.entry_arm.returnPressed.connect(self._send_arm)
        cr.addWidget(self.entry_arm)
        go = QPushButton("GO")
        theme_manager.register_button(go, "blue")
        go.setFixedWidth(38)
        go.clicked.connect(self._send_arm)
        cr.addWidget(go)
        cr.addSpacing(4)
        self.btn_home = QPushButton("⌂ HOME")
        theme_manager.register_button(self.btn_home, "blue")
        self.btn_home.clicked.connect(
            lambda: self.ctrl.send_command("h"))
        cr.addWidget(self.btn_home)
        self.btn_return = QPushButton("↩ RETURN")
        theme_manager.register_button(self.btn_return, "dim_blue")
        self.btn_return.clicked.connect(
            lambda: self.ctrl.send_command("r"))
        cr.addWidget(self.btn_return)
        cr.addStretch()
        self.btn_enable = QPushButton("⚡ EN")
        theme_manager.register_button(self.btn_enable, "dim_green")
        self.btn_enable.clicked.connect(
            lambda: self.ctrl.send_command("e"))
        cr.addWidget(self.btn_enable)
        self.btn_disable = QPushButton("✕ DIS")
        theme_manager.register_button(self.btn_disable, "red")
        self.btn_disable.clicked.connect(
            lambda: self.ctrl.send_command("d"))
        cr.addWidget(self.btn_disable)
        lay.addLayout(cr)

        spd = QHBoxLayout()
        spd.setSpacing(4)
        spd.addWidget(_muted("Mv µs:"))
        self.entry_speed = QLineEdit("500")
        self.entry_speed.setFixedWidth(44)
        self.entry_speed.returnPressed.connect(self._send_speed)
        spd.addWidget(self.entry_speed)
        s1 = QPushButton("SET")
        theme_manager.register_button(s1, "blue")
        s1.setFixedWidth(36)
        s1.clicked.connect(self._send_speed)
        spd.addWidget(s1)
        spd.addSpacing(6)
        spd.addWidget(_muted("Hm µs:"))
        self.entry_hspeed = QLineEdit("800")
        self.entry_hspeed.setFixedWidth(44)
        self.entry_hspeed.returnPressed.connect(self._send_hspeed)
        spd.addWidget(self.entry_hspeed)
        s2 = QPushButton("SET")
        theme_manager.register_button(s2, "blue")
        s2.setFixedWidth(36)
        s2.clicked.connect(self._send_hspeed)
        spd.addWidget(s2)
        spd.addStretch()
        lay.addLayout(spd)
        return grp

    def _servo_group(self):
        grp = QGroupBox("CAMERA SERVO")
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)

        ar = QHBoxLayout()
        self.lbl_servo = QLabel("150 °")
        theme_manager.register_widget(
            self.lbl_servo, lambda p: (
                f"color:{p['amber']};font-size:13px;"
                f"font-weight:bold;font-family:'Noto Sans',Arial,sans-serif;"))
        ar.addWidget(self.lbl_servo)
        ar.addStretch()
        lay.addLayout(ar)

        self.slider_servo = QSlider(Qt.Horizontal)
        self.slider_servo.setObjectName("servo_slider")
        self.slider_servo.setRange(0, 170)
        self.slider_servo.setValue(150)
        self.slider_servo.sliderPressed.connect(
            lambda: setattr(self, '_drag_servo', True))
        self.slider_servo.sliderReleased.connect(self._servo_released)
        self.slider_servo.valueChanged.connect(
            lambda v: self.lbl_servo.setText(f"{v} °"))
        lay.addWidget(self.slider_servo)

        rr = QHBoxLayout()
        rr.addWidget(_muted("0°"))
        rr.addStretch()
        rr.addWidget(_muted("170°"))
        lay.addLayout(rr)

        er = QHBoxLayout()
        er.setSpacing(4)
        self.entry_angle = QLineEdit()
        self.entry_angle.setPlaceholderText("deg")
        self.entry_angle.setFixedWidth(58)
        self.entry_angle.returnPressed.connect(self._send_angle)
        er.addWidget(self.entry_angle)
        sb = QPushButton("SET")
        theme_manager.register_button(sb, "blue")
        sb.setFixedWidth(38)
        sb.clicked.connect(self._send_angle)
        er.addWidget(sb)
        er.addStretch()
        lay.addLayout(er)

        pr = QHBoxLayout()
        pr.setSpacing(4)
        pr.addWidget(_muted("Quick:"))
        for ang, lbl in [(0,"0°"),(50,"50°"),(130,"130°"),
                         (150,"150°"),(170,"170°")]:
            b = QPushButton(lbl)
            theme_manager.register_button(b, "blue")
            b.setFixedWidth(44)
            b.setFixedHeight(24)
            b.clicked.connect(
                lambda _, a=ang: self.ctrl.send_command(f"a {a}"))
            pr.addWidget(b)
        pr.addStretch()
        lay.addLayout(pr)
        return grp

    def _seq_group(self):
        grp = QGroupBox("SEQUENCE")
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)

        sr = QHBoxLayout()
        self.led_seq = LED(12)
        sr.addWidget(self.led_seq)
        self.lbl_seq_st = QLabel("IDLE")
        self.lbl_seq_st.setStyleSheet(
            f"color:{theme_manager.palette()['muted']};"
            f"font-size:10px;font-family:'Noto Sans',Arial,sans-serif;")
        sr.addWidget(self.lbl_seq_st)
        sr.addStretch()
        self.lbl_seq_pr = QLabel("")
        theme_manager.register_widget(
            self.lbl_seq_pr, lambda p: (
                f"color:{p['blue']};font-size:10px;"
                f"font-family:'Noto Sans',Arial,sans-serif;font-weight:bold;"))
        sr.addWidget(self.lbl_seq_pr)
        lay.addLayout(sr)

        self.entry_seq = QLineEdit()
        self.entry_seq.setPlaceholderText(
            "e.g.  m 0, a 90, m 10, a 145")
        self.entry_seq.returnPressed.connect(self._run_once)
        lay.addWidget(self.entry_seq)
        lay.addWidget(_muted(
            "Tokens: m <in>  a <deg>  light on/off"))

        br = QHBoxLayout()
        br.setSpacing(4)
        self.btn_run_once = QPushButton("▶ RUN")
        theme_manager.register_button(self.btn_run_once, "teal")
        self.btn_run_once.clicked.connect(self._run_once)
        br.addWidget(self.btn_run_once)
        self.btn_run_loop = QPushButton("↺ LOOP")
        theme_manager.register_button(self.btn_run_loop, "amber")
        self.btn_run_loop.clicked.connect(self._run_loop)
        br.addWidget(self.btn_run_loop)
        br.addSpacing(4)
        self.btn_pause = QPushButton("⏸ PAUSE")
        theme_manager.register_button(self.btn_pause, "dim_amber")
        self.btn_pause.clicked.connect(
            lambda: self.ctrl.send_command("pause"))
        br.addWidget(self.btn_pause)
        self.btn_resume = QPushButton("▶ RESUME")
        theme_manager.register_button(self.btn_resume, "dim_green")
        self.btn_resume.clicked.connect(
            lambda: self.ctrl.send_command("resume"))
        br.addWidget(self.btn_resume)
        self.btn_seq_stop = QPushButton("⏹ STOP")
        theme_manager.register_button(self.btn_seq_stop, "dim_red")
        self.btn_seq_stop.clicked.connect(
            lambda: self.ctrl.send_command("stop"))
        br.addWidget(self.btn_seq_stop)
        br.addStretch()
        lay.addLayout(br)
        return grp

    def _aux_group(self):
        grp = QGroupBox("AUX  —  Light  ·  Motor PSU")
        lay = QVBoxLayout(grp)
        lay.setSpacing(4)
        self.light_widget = LightWidget(self.ctrl)
        lay.addWidget(self.light_widget)
        self.motor_psu_widget = MotorPsuWidget(self.ctrl)
        lay.addWidget(self.motor_psu_widget)
        lay.addWidget(
            _muted("motor psu off clears home — re-home after restore"))
        return grp

    def _status_group(self):
        grp = QGroupBox("SYSTEM STATUS")
        lay = QVBoxLayout(grp)
        lay.setSpacing(4)
        grid = QGridLayout()
        grid.setSpacing(3)

        def _row(r, lbl, attr, default="—", role=None):
            l = QLabel(lbl)
            theme_manager.register_widget(
                l, lambda p: (
                    f"color:{p['muted']};font-size:10px;"
                    f"font-family:'Noto Sans',Arial,sans-serif;"))
            v = QLabel(default)
            if role:
                theme_manager.register_widget(
                    v, lambda p, role=role: (
                        f"color:{p[role]};font-size:10px;"
                        f"font-family:'Noto Sans',Arial,sans-serif;font-weight:bold;"))
            grid.addWidget(l, r, 0)
            grid.addWidget(v, r, 1)
            setattr(self, attr, v)

        _row(0, "Limit switch:", "lbl_limit",  "—")
        self.lbl_limit.setStyleSheet(
            f"color:{theme_manager.palette()['dim']};font-size:10px;"
            f"font-family:'Noto Sans',Arial,sans-serif;font-weight:bold;")
        _row(1, "Move speed:",   "lbl_mspeed", "—", role="blue")
        _row(2, "Home speed:",   "lbl_hspeed", "—", role="blue")
        lay.addLayout(grid)

        br = QHBoxLayout()
        br.setSpacing(6)
        ref = QPushButton("↻ STATUS")
        theme_manager.register_button(ref, "blue")
        ref.setMinimumHeight(26)
        ref.clicked.connect(lambda: self.ctrl.send_command("p"))
        br.addWidget(ref)
        inf = QPushButton("ℹ INFO")
        theme_manager.register_button(inf, "blue")
        inf.setMinimumHeight(26)
        inf.clicked.connect(lambda: self.ctrl.send_command("i"))
        br.addWidget(inf)
        lay.addLayout(br)
        return grp

    # ── Command senders ───────────────────────────────────────
    def _arm_released(self):
        self._drag_arm = False
        self.ctrl.send_move(self.slider_arm.value() / 100.0)

    def _servo_released(self):
        self._drag_servo = False
        self.ctrl.send_angle(self.slider_servo.value())

    def _send_arm(self):
        try:
            self.ctrl.send_move(float(self.entry_arm.text()))
            self.entry_arm.clear()
        except Exception:
            self.shared_log.log("GANTRY","Invalid position","error")

    def _send_angle(self):
        try:
            self.ctrl.send_angle(int(self.entry_angle.text()))
            self.entry_angle.clear()
        except Exception:
            self.shared_log.log("GANTRY","Invalid angle","error")

    def _send_speed(self):
        try:
            self.ctrl.send_speed(int(self.entry_speed.text()))
        except Exception:
            self.shared_log.log("GANTRY","Invalid speed","error")

    def _send_hspeed(self):
        try:
            self.ctrl.send_home_speed(int(self.entry_hspeed.text()))
        except Exception:
            self.shared_log.log("GANTRY","Invalid speed","error")

    def _run_once(self):
        seq = self.entry_seq.text().strip()
        if seq:
            self.ctrl.send_sequence(seq, loop=False)

    def _run_loop(self):
        seq = self.entry_seq.text().strip()
        if seq:
            self.ctrl.send_sequence(seq, loop=True)

    # ── Callbacks ─────────────────────────────────────────────
    def _on_log(self, msg, tag):
        self.shared_log.log("GANTRY", msg, tag)

    def _on_state(self, s):
        try:
            self.state_signal.emit(s)
        except RuntimeError:
            pass  # Qt widget already deleted during shutdown

    def _apply_state(self, s: GantryState):
        p = theme_manager.palette()

        # Homed — grey out all stepper controls in detection mode
        detection_mode = (getattr(s, 'firmware_mode', 'unified')
                          == 'detection')
        stepper_widgets = [
            self.slider_arm, self.entry_arm,
            self.btn_home, self.btn_return,
            self.btn_enable, self.btn_disable,
            self.entry_speed, self.entry_hspeed,
        ]
        for w in stepper_widgets:
            w.setEnabled(not detection_mode)

        if detection_mode:
            self.lbl_homed.setText("DETECTION MODE — arm locked")
            self.lbl_homed.setStyleSheet(
                f"color:{p['blue']};font-size:10px;"
                f"font-family:'Noto Sans',Arial,sans-serif;font-weight:bold;")
            self.led_homed.set_state(True, role="blue")
        else:
            self.led_homed.set_state(s.homed, role="green")
            c = p["green"] if s.homed else p["red"]
            self.lbl_homed.setText("HOMED" if s.homed else "NOT HOMED")
            self.lbl_homed.setStyleSheet(
                f"color:{c};font-size:10px;"
                f"font-family:'Noto Sans',Arial,sans-serif;font-weight:bold;")
            theme_manager.register_button(
                self.btn_home, "dim_blue" if s.homed else "blue")
            theme_manager.register_button(
                self.btn_return, "blue" if s.homed else "dim_blue")

        # Arm slider
        if not self._drag_arm:
            self.lbl_arm_pos.setText(f"{s.arm_pos:.2f} in")
            self.slider_arm.blockSignals(True)
            self.slider_arm.setValue(int(s.arm_pos * 100))
            self.slider_arm.blockSignals(False)

        # Servo slider
        if not self._drag_servo:
            self.lbl_servo.setText(f"{s.cam_angle} °")
            self.slider_servo.blockSignals(True)
            self.slider_servo.setValue(s.cam_angle)
            self.slider_servo.blockSignals(False)

        # Aux
        self.light_widget.update_state(s.light_on)
        self.motor_psu_widget.update_state(s.motor_psu_on)

        # Sequence
        self.led_seq.set_state(
            s.seq_running and not s.seq_paused, role="green")
        if s.seq_running:
            st   = "PAUSED" if s.seq_paused else "RUNNING"
            mode = "LOOP"   if s.seq_looping else "ONCE"
            self.lbl_seq_st.setText(f"{st} [{mode}]")
            c = p["amber"] if s.seq_paused else p["green"]
            self.lbl_seq_st.setStyleSheet(
                f"color:{c};font-size:10px;font-family:'Noto Sans',Arial,sans-serif;")
            self.lbl_seq_pr.setText(
                f"step {s.seq_step}/{s.seq_total}")
        else:
            self.lbl_seq_st.setText("IDLE")
            self.lbl_seq_st.setStyleSheet(
                f"color:{p['muted']};font-size:10px;font-family:'Noto Sans',Arial,sans-serif;")
            self.lbl_seq_pr.setText("")

        # Status
        self.lbl_limit.setText("OK" if s.limit_ok else "⚠ TRIGGERED")
        c = p["green"] if s.limit_ok else p["red"]
        self.lbl_limit.setStyleSheet(
            f"color:{c};font-size:10px;"
            f"font-family:'Noto Sans',Arial,sans-serif;font-weight:bold;")
        self.lbl_mspeed.setText(f"{s.move_speed} µs")
        self.lbl_hspeed.setText(f"{s.home_speed} µs")

    # ── Public API ────────────────────────────────────────────
    @property
    def is_connected(self) -> bool:
        return self.ctrl.state.connected

    def emergency_stop(self):
        """Called from global E-STOP."""
        self.ctrl.send_command("stop")
        self.ctrl.send_command("light off")

    def cleanup(self):
        # Hardware shutdown and disconnect handled by MainWindow
        # to ensure correct order before Qt widget destruction
        pass