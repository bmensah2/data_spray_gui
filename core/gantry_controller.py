"""
gantry_controller.py
ABEN Imaging Gantry — Serial Communication Backend
Handles all Arduino communication in a dedicated thread.
Thread-safe command queue, response parsing, state tracking.
"""

try:
    import serial
    import serial.tools.list_ports
except ImportError:  # pragma: no cover - optional hardware dependency
    serial = None
    serial_tools = None

import threading
import queue
import time
import re
from dataclasses import dataclass, field
from typing import Optional, Callable


# ─────────────────────────────────────────────────────────────
#  GANTRY STATE  (mirrors Arduino state)
# ─────────────────────────────────────────────────────────────
@dataclass
class GantryState:
    connected:    bool  = False
    firmware_mode: str   = 'unified'   # 'unified' or 'detection'
    homed:        bool  = False
    arm_pos:      float = 0.0        # inches
    cam_angle:    int   = 80         # degrees
    pump_on:      bool  = False
    nozzles:      list  = field(default_factory=lambda: [False, False, False])
    light_on:     bool  = False
    motor_psu_on: bool  = True    # starts True — firmware powers it on at boot
    seq_running:  bool  = False
    seq_paused:   bool  = False
    seq_looping:  bool  = False
    seq_step:     int   = 0
    seq_total:    int   = 0
    move_speed:   int   = 500
    home_speed:   int   = 800
    limit_ok:     bool  = True


# ─────────────────────────────────────────────────────────────
#  GANTRY CONTROLLER
# ─────────────────────────────────────────────────────────────
class GantryController:
    BAUD = 9600
    POLL_INTERVAL = 1.5       # seconds between status polls
    CONNECT_TIMEOUT = 8.0     # seconds to wait for Arduino boot

    def __init__(self):
        self.state          = GantryState()
        self._serial        = None
        self._cmd_queue     = queue.Queue()
        self._worker        = None
        self._running       = False
        self._lock          = threading.Lock()
        self._log_cb: Optional[Callable] = None   # GUI log callback
        self._state_cb: Optional[Callable] = None  # GUI state-update callback

    # ── Callbacks ────────────────────────────────────────────
    def set_log_callback(self, cb: Callable):
        self._log_cb = cb

    def set_state_callback(self, cb: Callable):
        self._state_cb = cb

    def _log(self, msg: str, tag: str = "info"):
        if self._log_cb:
            self._log_cb(msg, tag)

    def _notify_state(self):
        if self._state_cb:
            self._state_cb(self.state)

    # ── Port utilities ────────────────────────────────────────
    @staticmethod
    def list_ports() -> list[str]:
        if serial is None or serial.tools.list_ports is None:
            return []
        ports = serial.tools.list_ports.comports()
        return [p.device for p in sorted(ports)]

    # ── Connection ────────────────────────────────────────────
    def connect(self, port: str) -> bool:
        if serial is None:
            self._log("pyserial not installed — gantry unavailable", "error")
            self.state.connected = False
            return False

        try:
            self._serial = serial.Serial(
                port=port,
                baudrate=self.BAUD,
                timeout=1.0
            )
            time.sleep(0.5)
            self._serial.reset_input_buffer()

            self._running = True
            self.state.connected = True
            self._worker = threading.Thread(
                target=self._worker_loop, daemon=True
            )
            self._worker.start()

            self._log(f"Connected to {port} @ {self.BAUD} baud", "ok")
            self._notify_state()

            # Wait for Arduino boot banner then request status
            time.sleep(2.5)
            self.send_command("p")
            return True

        except Exception as e:
            self._log(f"Connection failed: {e}", "error")
            self.state.connected = False
            return False

    def disconnect(self):
        self._running = False
        self.state.connected = False
        if self._serial and self._serial.is_open:
            try:
                self._serial.close()
            except Exception:
                pass
        self._log("Disconnected", "warn")
        self._notify_state()

    # ── Command sending ────────────────────────────────────────
    def send_command(self, cmd: str):
        """Queue a command for sending. Thread-safe."""
        self._cmd_queue.put(cmd.strip())

    def send_move(self, inches: float):
        self.send_command(f"m {inches:.2f}")

    def send_angle(self, angle: int):
        self.send_command(f"a {angle}")

    def send_speed(self, speed: int):
        self.send_command(f"s {speed}")

    def send_home_speed(self, speed: int):
        self.send_command(f"hs {speed}")

    def send_sequence(self, seq_str: str, loop: bool = False):
        if loop:
            self.send_command(f"loop {seq_str}")
        else:
            self.send_command(seq_str)

    def send_light(self, on: bool):
        self.send_command("light on" if on else "light off")

    def send_motor_psu(self, on: bool):
        self.send_command("mpsu on" if on else "mpsu off")

    # ── Worker thread ─────────────────────────────────────────
    def _worker_loop(self):
        last_poll = time.time()

        while self._running:
            # Send queued commands
            try:
                cmd = self._cmd_queue.get_nowait()
                self._write(cmd)
                self._log(f"► {cmd}", "send")
            except queue.Empty:
                pass

            # Read all available lines
            try:
                if self._serial and self._serial.in_waiting:
                    try:
                        raw = self._serial.readline()
                        line = raw.decode("utf-8", errors="replace").strip()
                        if line:
                            self._parse_line(line)
                            self._log(line, "recv")
                    except Exception as e:
                        self._log(f"Read error: {e}", "error")
                        self.disconnect()
                        break
            except OSError:
                # Serial port closed — exit cleanly
                break

            # Periodic status poll
            if time.time() - last_poll > self.POLL_INTERVAL:
                self._write("p")
                last_poll = time.time()

            time.sleep(0.02)

    def _write(self, cmd: str):
        if self._serial and self._serial.is_open:
            try:
                self._serial.write((cmd + "\n").encode("utf-8"))
            except Exception as e:
                self._log(f"Write error: {e}", "error")

    # ── Response parser ───────────────────────────────────────
    def _parse_line(self, line: str):
        changed = False

        # Homed confirmation
        if "[OK] Home set" in line:
            self.state.homed = True;  changed = True

        # Not homed
        elif "Not homed" in line:
            self.state.homed = False; changed = True

        # Arm position  →  "    Arm pos  : 12.50 in  (1667 steps)"
        m = re.search(r"Arm pos\s*:\s*([\d.]+)\s*in", line)
        if m:
            self.state.arm_pos = float(m.group(1)); changed = True

        # Camera angle  →  "    Cam angle: 90 deg"
        m = re.search(r"Cam angle\s*:\s*(\d+)\s*deg", line)
        if m:
            self.state.cam_angle = int(m.group(1)); changed = True

        # Pump state
        if "[PMP] Pump ON" in line:
            self.state.pump_on = True;  changed = True
        elif "[PMP] Pump OFF" in line:
            self.state.pump_on = False; changed = True

        # Pump status line  →  "    Pump     : ON  [RUNNING]"
        m = re.search(r"Pump\s*:\s*(ON|OFF)", line)
        if m:
            self.state.pump_on = (m.group(1) == "ON"); changed = True

        # Light state  →  "[LT] Light ON" / "[LT] Light OFF"
        if "[LT] Light ON" in line:
            self.state.light_on = True;  changed = True
        elif "[LT] Light OFF" in line:
            self.state.light_on = False; changed = True

        # Light status line  →  "    Light    : ON  [ACTIVE]"
        m = re.search(r"Light\s*:\s*(ON|OFF)", line)
        if m:
            self.state.light_on = (m.group(1) == "ON"); changed = True

        # Motor PSU state  →  "[PSU] Motor PSU ON" / "[PSU] Motor PSU OFF"
        if "[PSU] Motor PSU ON" in line:
            self.state.motor_psu_on = True;  changed = True
        elif "[PSU] Motor PSU OFF" in line:
            self.state.motor_psu_on = False; changed = True
            self.state.homed        = False  # PSU off clears homed — mirror firmware behaviour

        # Motor PSU status line  →  "    Motor PSU: ON  [POWERED]"
        m = re.search(r"Motor PSU\s*:\s*(ON|OFF)", line)
        if m:
            self.state.motor_psu_on = (m.group(1) == "ON"); changed = True
        
        # Nozzle state  →  "[NZ] Nozzle 1 ON" / "[NZ] Nozzle 1 OFF"
        m = re.search(r"\[NZ\] Nozzle (\d) (ON|OFF)", line)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < 3:
                self.state.nozzles[idx] = (m.group(2) == "ON")
                changed = True

        # Nozzle status line  →  "    Nozzle 1  : ON  [OPEN]"
        m = re.search(r"Nozzle (\d)\s*:\s*(ON|OFF)", line)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < 3:
                self.state.nozzles[idx] = (m.group(2) == "ON")
                changed = True

        # Sequence state  →  "    Sequence : RUNNING [once]  step 3 / 8"
        if "Sequence" in line and ("RUNNING" in line or "PAUSED" in line):
            self.state.seq_running = True
            self.state.seq_paused  = "PAUSED" in line
            self.state.seq_looping = "[loop]" in line
            m = re.search(r"step (\d+) / (\d+)", line)
            if m:
                self.state.seq_step  = int(m.group(1))
                self.state.seq_total = int(m.group(2))
            changed = True
        elif "Sequence" in line and "IDLE" in line:
            self.state.seq_running = False
            self.state.seq_paused  = False
            changed = True

        # Sequence complete/stopped
        if "[OK] Sequence complete" in line or "[X] Sequence STOPPED" in line:
            self.state.seq_running = False
            self.state.seq_paused  = False
            changed = True

        # Sequence started  →  "[>>] Sequence started - 6 steps (looping)"
        if "[>>] Sequence started" in line:
            self.state.seq_running = True
            self.state.seq_paused  = False
            self.state.seq_looping = "looping" in line
            m = re.search(r"(\d+) steps", line)
            if m:
                self.state.seq_total = int(m.group(1))
                self.state.seq_step  = 1
            changed = True

        # Limit switch
        if "LIMIT SWITCH TRIGGERED" in line:
            self.state.limit_ok = False; changed = True
        elif "Home set" in line:
            self.state.limit_ok = True;  changed = True

        # Move speed  →  "[OK] Move speed = 500 us"
        m = re.search(r"Move speed = (\d+)", line)
        if m:
            self.state.move_speed = int(m.group(1)); changed = True

        # Firmware mode detection
        if "[MODE] DETECTION" in line:
            self.state.firmware_mode = "detection"
            self.state.homed         = False   # no homing in detection mode
            self.state.limit_ok      = True    # no limit switch to worry about
            changed = True
        elif "[MODE] UNIFIED" in line or "[OK] Home set" in line:
            self.state.firmware_mode = "unified"
            changed = True

        if changed:
            self._notify_state()