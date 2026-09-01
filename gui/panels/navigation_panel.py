"""
gui/panels/navigation_panel.py
ABEN Field Imaging System — Navigation Panel

Shared panel used in both Tab 1 (Data Collection) and Tab 2 (Detection).
Same widget instance — switching tabs preserves state.

Features:
  - Connection status + live odometry display
  - Manual movement: forward / backward / left / right
  - Quick move buttons
  - Mission loader: select YAML, sync to Husky, dry run, run, progress
  - STOP button: pkill + zero velocity

Requires:
  - SSH key auth: Jetson → Husky (passwordless)
  - test_nav_v3.py on Husky PC at ~/test_nav_v3.py
  - field_nav.py  on Husky PC at ~/field_nav.py
  - missions/     on Husky PC at ~/missions/

Usage:
    from gui.panels.navigation_panel import NavigationPanel
    nav = NavigationPanel(shared_log, ros_bridge_ref)
    layout.addWidget(nav)
"""

import subprocess
import warnings
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QGroupBox, QDoubleSpinBox,
    QComboBox, QProgressBar, QPlainTextEdit,
    QFileDialog, QScrollArea, QTabWidget,
    QDialog, QTextEdit, QSplitter, QMessageBox,
    QLineEdit, QDialogButtonBox, QInputDialog
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QTextCursor

from gui.style import _divider, _muted, _sec, _scroll
from gui.theme_manager import theme_manager, _lighten, _darken, _dim_pair, _btn
from gui.shared_log import UnifiedLog
from gui.mission_editor import MissionEditorDialog, TEMPLATE_YAML

warnings.filterwarnings("ignore", category=ResourceWarning)

# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────
HUSKY_IP      = "192.168.131.1"
HUSKY_USER    = "administrator"
ROS_SOURCE    = "source /opt/ros/noetic/setup.bash && "
# Nav scripts live in navigation/ on Jetson
# Copies on Husky PC remain at ~/ (synced separately)
NAV_SCRIPT    = "~/test_nav_v3.py"    # on Husky PC
FIELD_NAV     = "~/field_nav.py"      # on Husky PC
MISSIONS_HUSKY = "~/missions"

# Try Jetson home path first, fall back to SSD
_MISSIONS_HOME = Path("/home/pagsun/phd_project/multispec_camera/missions")
_MISSIONS_SSD  = Path("/media/pagsun/Transcend/phd_project/multispec_camera/missions")
MISSIONS_LOCAL = _MISSIONS_HOME if _MISSIONS_HOME.parent.exists() else _MISSIONS_SSD
# Nav scripts on Jetson (for sync reference)
NAV_SCRIPTS_LOCAL = Path(
    "/media/pagsun/Transcend/phd_project/multispec_camera/navigation"
)


def _custom_btn(aben_qss: str, build_fn):
    """
    style_fn(palette) -> str for a one-off custom-colored button.
    Uses the exact original aben_dark QSS when that theme is active
    (pixel-identical to the pre-theming look); otherwise computes a
    themed equivalent via build_fn(palette).
    """
    def _style(p):
        if theme_manager.current == "aben_dark":
            return aben_qss
        return build_fn(p)
    return _style


# ─────────────────────────────────────────────────────────────
#  NAVIGATION PANEL
# ─────────────────────────────────────────────────────────────
class NavigationPanel(QWidget):
    """
    Shared navigation panel — manual moves + mission control.
    ros_bridge_ref: callable → ROSBridge or None
    """

    def __init__(self, shared_log: UnifiedLog,
                 ros_bridge_ref=None, parent=None):
        super().__init__(parent)
        self.shared_log    = shared_log
        self.ros_bridge_ref = ros_bridge_ref or (lambda: None)

        self._nav_proc     = None
        self._mission_proc = None
        self._log_timer    = None
        self._mission_running = False

        MISSIONS_LOCAL.mkdir(parents=True, exist_ok=True)
        self._build_ui()

        # Odom refresh timer — 500ms
        self._odom_timer = QTimer()
        self._odom_timer.timeout.connect(self._update_odom)
        self._odom_timer.start(500)

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

        lay.addWidget(self._connection_grp())
        lay.addWidget(_divider())
        lay.addWidget(self._manual_grp())
        lay.addWidget(self._quick_grp())
        lay.addWidget(_divider())
        lay.addWidget(self._mission_grp())
        lay.addWidget(self._stop_grp())
        lay.addStretch()

        scroll.setWidget(inner)
        outer.addWidget(scroll)

    # ── Connection status ─────────────────────────────────────

    def _connection_grp(self):
        grp = QGroupBox("Husky Connection")
        lay = QVBoxLayout(grp)
        lay.setSpacing(4)

        self.lbl_conn_status = QLabel("● ROS bridge not armed")
        theme_manager.register_widget(
            self.lbl_conn_status, lambda p: (
                f"color:{_darken(p['amber'], 40)};"
                f"font-family:Courier New;font-size:10px;"))
        lay.addWidget(self.lbl_conn_status)

        self.lbl_odom = QLabel(
            "x=--  y=--  yaw=--  spd=--")
        theme_manager.register_widget(
            self.lbl_odom, lambda p: (
                f"color:{p['muted']};font-family:Courier New;font-size:9px;"))
        lay.addWidget(self.lbl_odom)

        self.lbl_husky_ip = _muted(f"Husky: {HUSKY_USER}@{HUSKY_IP}")
        lay.addWidget(self.lbl_husky_ip)
        return grp

    # ── Manual movement ───────────────────────────────────────

    def _manual_grp(self):
        grp = QGroupBox("Manual Movement")
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)

        # ── Linear ───────────────────────────────────────────
        lin_grp = QGroupBox("Linear")
        ll = QVBoxLayout(lin_grp)
        lr1 = QHBoxLayout()
        lr1.addWidget(_muted("Distance (m):"))
        self.spn_dist = QDoubleSpinBox()
        self.spn_dist.setRange(0.05, 100.0)
        self.spn_dist.setValue(1.0)
        self.spn_dist.setSingleStep(0.1)
        self.spn_dist.setDecimals(2)
        lr1.addWidget(self.spn_dist)
        lr1.addWidget(_muted("Speed (m/s):"))
        self.spn_lin_spd = QDoubleSpinBox()
        self.spn_lin_spd.setRange(0.05, 1.0)
        self.spn_lin_spd.setValue(0.3)
        self.spn_lin_spd.setSingleStep(0.05)
        self.spn_lin_spd.setDecimals(2)
        lr1.addWidget(self.spn_lin_spd)
        ll.addLayout(lr1)

        lr2 = QHBoxLayout()
        btn_back = QPushButton("◄ Backward")
        theme_manager.register_widget(btn_back, _custom_btn(
            "QPushButton{background:#1a2a3a;color:#60a0d0;"
            "border:1px solid #2a4a6a;border-radius:4px;"
            "padding:6px;font-weight:bold;}"
            "QPushButton:hover{background:#2a3a4a;}",
            lambda p: (
                f"QPushButton{{background:{_darken(p['blue'],55)};"
                f"color:{_lighten(p['blue'],10)};"
                f"border:1px solid {_darken(p['blue'],30)};border-radius:4px;"
                f"padding:6px;font-weight:bold;}}"
                f"QPushButton:hover{{background:{_darken(p['blue'],40)};}}")))
        btn_back.clicked.connect(
            lambda: self._nav_run('backward',
                                  self.spn_dist.value(),
                                  self.spn_lin_spd.value()))
        lr2.addWidget(btn_back)

        btn_fwd = QPushButton("Forward ►")
        theme_manager.register_widget(btn_fwd, _custom_btn(
            "QPushButton{background:#1a3a1a;color:#60d060;"
            "border:1px solid #2a6a2a;border-radius:4px;"
            "padding:6px;font-weight:bold;}"
            "QPushButton:hover{background:#2a4a2a;}",
            lambda p: (
                f"QPushButton{{background:{_darken(p['green'],55)};"
                f"color:{_lighten(p['green'],10)};"
                f"border:1px solid {_darken(p['green'],30)};border-radius:4px;"
                f"padding:6px;font-weight:bold;}}"
                f"QPushButton:hover{{background:{_darken(p['green'],40)};}}")))
        btn_fwd.clicked.connect(
            lambda: self._nav_run('forward',
                                  self.spn_dist.value(),
                                  self.spn_lin_spd.value()))
        lr2.addWidget(btn_fwd)
        ll.addLayout(lr2)
        lay.addWidget(lin_grp)

        # ── Rotation ─────────────────────────────────────────
        rot_grp = QGroupBox("Rotation")
        rl = QVBoxLayout(rot_grp)
        rr1 = QHBoxLayout()
        rr1.addWidget(_muted("Angle (°):"))
        self.spn_angle = QDoubleSpinBox()
        self.spn_angle.setRange(1.0, 360.0)
        self.spn_angle.setValue(90.0)
        self.spn_angle.setSingleStep(5.0)
        self.spn_angle.setDecimals(1)
        rr1.addWidget(self.spn_angle)
        rr1.addWidget(_muted("Speed (rad/s):"))
        self.spn_ang_spd = QDoubleSpinBox()
        self.spn_ang_spd.setRange(0.1, 1.5)
        self.spn_ang_spd.setValue(0.5)
        self.spn_ang_spd.setSingleStep(0.1)
        self.spn_ang_spd.setDecimals(1)
        rr1.addWidget(self.spn_ang_spd)
        rl.addLayout(rr1)

        rr2 = QHBoxLayout()
        btn_left = QPushButton("↺ Left")
        theme_manager.register_widget(btn_left, _custom_btn(
            "QPushButton{background:#2a1a3a;color:#b060d0;"
            "border:1px solid #5a2a6a;border-radius:4px;"
            "padding:6px;font-weight:bold;}"
            "QPushButton:hover{background:#3a2a4a;}",
            lambda p: (
                f"QPushButton{{background:{_darken(p['purple'],55)};"
                f"color:{p['purple']};"
                f"border:1px solid {_darken(p['purple'],30)};border-radius:4px;"
                f"padding:6px;font-weight:bold;}}"
                f"QPushButton:hover{{background:{_darken(p['purple'],40)};}}")))
        btn_left.clicked.connect(
            lambda: self._nav_run('left',
                                  self.spn_angle.value(),
                                  self.spn_ang_spd.value()))
        rr2.addWidget(btn_left)

        btn_right = QPushButton("Right ↻")
        theme_manager.register_widget(btn_right, _custom_btn(
            "QPushButton{background:#2a1a3a;color:#b060d0;"
            "border:1px solid #5a2a6a;border-radius:4px;"
            "padding:6px;font-weight:bold;}"
            "QPushButton:hover{background:#3a2a4a;}",
            lambda p: (
                f"QPushButton{{background:{_darken(p['purple'],55)};"
                f"color:{p['purple']};"
                f"border:1px solid {_darken(p['purple'],30)};border-radius:4px;"
                f"padding:6px;font-weight:bold;}}"
                f"QPushButton:hover{{background:{_darken(p['purple'],40)};}}")))
        btn_right.clicked.connect(
            lambda: self._nav_run('right',
                                  self.spn_angle.value(),
                                  self.spn_ang_spd.value()))
        rr2.addWidget(btn_right)
        rl.addLayout(rr2)
        lay.addWidget(rot_grp)
        return grp

    # ── Quick moves ───────────────────────────────────────────

    def _quick_grp(self):
        grp = QGroupBox("Quick Moves")
        lay = QVBoxLayout(grp)

        def _qbtn(label, mode, val, speed):
            b = QPushButton(label)
            theme_manager.register_widget(b, _custom_btn(
                "QPushButton{background:#1a2030;color:#8090c0;"
                "border:1px solid #2a3050;border-radius:3px;"
                "padding:4px;font-size:10px;}"
                "QPushButton:hover{background:#2a3040;}",
                lambda p: (
                    f"QPushButton{{background:{p['input_bg']};"
                    f"color:{_lighten(p['blue'],10)};"
                    f"border:1px solid {p['border']};border-radius:3px;"
                    f"padding:4px;font-size:10px;}}"
                    f"QPushButton:hover{{background:{p['btn_hover']};}}")))
            b.clicked.connect(
                lambda: self._nav_run(mode, val, speed))
            return b

        r1 = QHBoxLayout()
        r1.addWidget(_qbtn("◄ 0.5m",  "backward", 0.5, 0.3))
        r1.addWidget(_qbtn("0.5m ►",  "forward",  0.5, 0.3))
        r1.addWidget(_qbtn("◄ 1.0m",  "backward", 1.0, 0.3))
        r1.addWidget(_qbtn("1.0m ►",  "forward",  1.0, 0.3))
        lay.addLayout(r1)

        r2 = QHBoxLayout()
        r2.addWidget(_qbtn("↺ 45°",  "left",  45,  0.5))
        r2.addWidget(_qbtn("45° ↻",  "right", 45,  0.5))
        r2.addWidget(_qbtn("↺ 90°",  "left",  90,  0.5))
        r2.addWidget(_qbtn("90° ↻",  "right", 90,  0.5))
        lay.addLayout(r2)

        r3 = QHBoxLayout()
        r3.addWidget(_qbtn("↺ 180°", "left",  180, 0.5))
        r3.addWidget(_qbtn("180° ↻", "right", 180, 0.5))
        r3.addWidget(_qbtn("◄ 2.0m", "backward", 2.0, 0.3))
        r3.addWidget(_qbtn("2.0m ►", "forward",  2.0, 0.3))
        lay.addLayout(r3)
        return grp

    # ── Mission control ───────────────────────────────────────

    def _mission_grp(self):
        grp = QGroupBox("Mission Control")
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)

        # Mission selector
        sel_row = QHBoxLayout()
        self.cmb_mission = QComboBox()
        self.cmb_mission.setMinimumWidth(160)
        self.cmb_mission.currentTextChanged.connect(
            self._on_mission_selected)
        sel_row.addWidget(self.cmb_mission, stretch=1)

        btn_refresh = QPushButton("↻")
        btn_refresh.setFixedWidth(28)
        btn_refresh.setToolTip("Refresh mission list")
        btn_refresh.clicked.connect(self._refresh_missions)
        sel_row.addWidget(btn_refresh)

        btn_browse = QPushButton("Browse")
        btn_browse.setFixedWidth(60)
        theme_manager.register_button(btn_browse, "blue")
        btn_browse.clicked.connect(self._browse_mission)
        sel_row.addWidget(btn_browse)
        lay.addLayout(sel_row)

        # Edit / New buttons row
        edit_row = QHBoxLayout()
        btn_new = QPushButton("＋ New Mission")
        theme_manager.register_button(btn_new, "blue")
        btn_new.setToolTip("Create a new mission YAML file")
        btn_new.clicked.connect(self._new_mission)
        edit_row.addWidget(btn_new)

        self.btn_edit = QPushButton("✏ Edit Mission")
        theme_manager.register_widget(self.btn_edit, _custom_btn(
            "QPushButton{background:#1a2a1a;color:#60c060;"
            "border:1px solid #2a5a2a;border-radius:4px;padding:5px;}"
            "QPushButton:hover{background:#2a3a2a;}",
            lambda p: (
                f"QPushButton{{background:{_darken(p['green'],55)};"
                f"color:{_lighten(p['green'],10)};"
                f"border:1px solid {_darken(p['green'],30)};"
                f"border-radius:4px;padding:5px;}}"
                f"QPushButton:hover{{background:{_darken(p['green'],40)};}}")))
        self.btn_edit.setToolTip("Edit selected mission in built-in editor")
        self.btn_edit.clicked.connect(self._edit_mission)
        edit_row.addWidget(self.btn_edit)

        btn_delete = QPushButton("🗑 Delete")
        theme_manager.register_widget(btn_delete, _custom_btn(
            "QPushButton{background:#2a0000;color:#c06060;"
            "border:1px solid #5a1010;border-radius:4px;padding:5px;}"
            "QPushButton:hover{background:#3a1010;}",
            lambda p: (
                f"QPushButton{{background:{_darken(p['red'],55)};"
                f"color:{_lighten(p['red'],10)};"
                f"border:1px solid {_darken(p['red'],30)};"
                f"border-radius:4px;padding:5px;}}"
                f"QPushButton:hover{{background:{_darken(p['red'],40)};}}")))
        btn_delete.setToolTip("Delete selected mission file")
        btn_delete.clicked.connect(self._delete_mission)
        edit_row.addWidget(btn_delete)
        lay.addLayout(edit_row)

        # Mission description
        self.lbl_mission_desc = QLabel("—")
        theme_manager.register_widget(
            self.lbl_mission_desc, lambda p: (
                f"color:{p['muted']};font-size:9px;"
                f"font-family:Courier New;"))
        self.lbl_mission_desc.setWordWrap(True)
        lay.addWidget(self.lbl_mission_desc)

        # Progress
        self.mission_progress = QProgressBar()
        self.mission_progress.setRange(0, 100)
        self.mission_progress.setValue(0)
        self.mission_progress.setFormat("Ready")
        lay.addWidget(self.mission_progress)

        self.lbl_mission_step = _muted("Step: —")
        lay.addWidget(self.lbl_mission_step)

        # Mission log
        self.mission_log = QPlainTextEdit()
        self.mission_log.setReadOnly(True)
        self.mission_log.setMaximumBlockCount(100)
        self.mission_log.setFixedHeight(70)
        theme_manager.register_widget(
            self.mission_log, lambda p: (
                f"background:{p['bg0']};color:{p['muted']};"
                f"font-family:Courier New;font-size:9px;border:none;"))
        lay.addWidget(self.mission_log)

        # Control buttons
        r1 = QHBoxLayout()
        self.btn_sync = QPushButton("⇅ Sync to Husky")
        theme_manager.register_widget(self.btn_sync, _custom_btn(
            "QPushButton{background:#1a2a3a;color:#60a0d0;"
            "border:1px solid #2a4a6a;border-radius:4px;padding:5px;}"
            "QPushButton:hover{background:#2a3a4a;}",
            lambda p: (
                f"QPushButton{{background:{_darken(p['blue'],55)};"
                f"color:{_lighten(p['blue'],10)};"
                f"border:1px solid {_darken(p['blue'],30)};"
                f"border-radius:4px;padding:5px;}}"
                f"QPushButton:hover{{background:{_darken(p['blue'],40)};}}")))
        self.btn_sync.clicked.connect(self._sync_mission)
        r1.addWidget(self.btn_sync)

        self.btn_dryrun = QPushButton("▷ Dry Run")
        theme_manager.register_widget(self.btn_dryrun, _custom_btn(
            "QPushButton{background:#1a2030;color:#8090c0;"
            "border:1px solid #2a3050;border-radius:4px;padding:5px;}"
            "QPushButton:hover{background:#2a3040;}",
            lambda p: (
                f"QPushButton{{background:{p['input_bg']};"
                f"color:{_lighten(p['blue'],10)};"
                f"border:1px solid {p['border']};border-radius:4px;padding:5px;}}"
                f"QPushButton:hover{{background:{p['btn_hover']};}}")))
        self.btn_dryrun.clicked.connect(self._dry_run)
        r1.addWidget(self.btn_dryrun)
        lay.addLayout(r1)

        r2 = QHBoxLayout()
        self.btn_run_mission = QPushButton("▶  RUN MISSION")
        theme_manager.register_button(self.btn_run_mission, "green")
        self.btn_run_mission.setMinimumHeight(30)
        self.btn_run_mission.clicked.connect(self._run_mission)
        r2.addWidget(self.btn_run_mission)

        self.btn_stop_mission = QPushButton("■  STOP MISSION")
        theme_manager.register_widget(self.btn_stop_mission, _custom_btn(
            "QPushButton{background:#5a0000;color:#ff4040;"
            "border:2px solid #ff0000;border-radius:4px;"
            "padding:6px;font-weight:bold;}"
            "QPushButton:hover{background:#7a0000;}"
            "QPushButton:disabled{background:#1a0000;"
            "color:#3a1010;border-color:#3a0000;}",
            lambda p: (
                f"QPushButton{{background:{_darken(p['red'],50)};color:{p['red']};"
                f"border:2px solid {p['red']};border-radius:4px;"
                f"padding:6px;font-weight:bold;}}"
                f"QPushButton:hover{{background:{_darken(p['red'],35)};}}"
                f"QPushButton:disabled{{background:{p['disabled_bg']};"
                f"color:{p['disabled_text']};border-color:{p['disabled_bg']};}}")))
        self.btn_stop_mission.setEnabled(False)
        self.btn_stop_mission.clicked.connect(self._stop_mission)
        r2.addWidget(self.btn_stop_mission)
        lay.addLayout(r2)

        # Populate mission list
        self._refresh_missions()
        return grp

    # ── STOP button ───────────────────────────────────────────

    def _stop_grp(self):
        grp = QGroupBox("Safety")
        lay = QVBoxLayout(grp)
        btn = QPushButton("■  STOP")
        theme_manager.register_widget(btn, _custom_btn(
            "QPushButton{background:#5a0000;color:#ff4040;"
            "border:2px solid #ff0000;border-radius:6px;"
            "padding:10px;font-size:14px;font-weight:bold;}"
            "QPushButton:hover{background:#7a0000;color:#ff6060;}",
            lambda p: (
                f"QPushButton{{background:{_darken(p['red'],50)};color:{p['red']};"
                f"border:2px solid {p['red']};border-radius:6px;"
                f"padding:10px;font-size:14px;font-weight:bold;}}"
                f"QPushButton:hover{{background:{_darken(p['red'],35)};"
                f"color:{_lighten(p['red'],15)};}}")))
        btn.clicked.connect(self._nav_stop)
        lay.addWidget(btn)
        return grp

    # ── Odom display ──────────────────────────────────────────

    def _update_odom(self):
        try:
            bridge = self.ros_bridge_ref()
            p = theme_manager.palette()
            if bridge is None:
                self.lbl_conn_status.setText("● ROS bridge not armed")
                theme_manager.register_widget(
                    self.lbl_conn_status, lambda p: (
                        f"color:{_darken(p['amber'], 40)};"
                        f"font-family:Courier New;font-size:10px;"))
                return

            connected = bridge.is_connected()
            pose      = bridge.get_pose()

            if connected:
                self.lbl_conn_status.setText(
                    "● Connected  cpr-a200-0943")
                theme_manager.register_widget(
                    self.lbl_conn_status, lambda p: (
                        f"color:{p['green']};font-family:Courier New;"
                        f"font-size:10px;"))
            else:
                self.lbl_conn_status.setText("● Disconnected")
                theme_manager.register_widget(
                    self.lbl_conn_status, lambda p: (
                        f"color:{p['red']};font-family:Courier New;"
                        f"font-size:10px;"))

            if pose:
                self.lbl_odom.setText(
                    f"x={pose['x']:.2f}m  "
                    f"y={pose['y']:.2f}m  "
                    f"yaw={pose['heading']:.1f}°  "
                    f"spd={pose['speed']:.2f}m/s")
        except Exception:
            pass

    # ── SSH helpers ───────────────────────────────────────────

    def _ssh_cmd(self, cmd: str,
                 wait: bool = False,
                 capture: bool = False) -> subprocess.Popen:
        """Run a command on Husky via SSH."""
        proc = subprocess.Popen(
            ['ssh', '-o', 'StrictHostKeyChecking=no',
             f'{HUSKY_USER}@{HUSKY_IP}',
             f'{ROS_SOURCE}{cmd}'],
            stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
            stderr=subprocess.STDOUT if capture else subprocess.DEVNULL,
            text=True
        )
        if wait:
            proc.wait()
        return proc

    # ── Manual navigation ─────────────────────────────────────

    def _nav_run(self, mode: str, val: float, speed: float):
        """Run test_nav_v3.py on Husky — non-blocking."""
        self._nav_kill()
        self._nav_proc = self._ssh_cmd(
            f'python3 {NAV_SCRIPT} {mode} {val} --speed {speed}'
        )
        self.shared_log.log(
            "NAV",
            f"{mode} {val} @ {speed}  →  Husky", "info")

    def _nav_kill(self):
        """Kill local SSH subprocess."""
        if self._nav_proc and self._nav_proc.poll() is None:
            self._nav_proc.terminate()
            self._nav_proc = None

    def _nav_stop(self):
        """
        Stop all robot motion immediately.
        Sequence:
          1. Kill local SSH subprocess
          2. SIGINT nav scripts on Husky (triggers clean ROS shutdown)
          3. Wait 0.5s for process to die and motors to coast down
          4. Publish zero velocity 5 times to guarantee stop
        Runs kill+stop in background thread so GUI stays responsive.
        """
        self._nav_kill()

        stop_cmd = (
            # SIGINT triggers rospy shutdown handler in nav scripts
            "pkill -SIGINT -f field_nav.py 2>/dev/null; "
            "pkill -SIGINT -f test_nav_v3.py 2>/dev/null; "
            "sleep 0.5; "
            # Source ROS then publish zero velocity multiple times
            "source /opt/ros/noetic/setup.bash && "
            "python3 -c 'import rospy,time;"
            "from geometry_msgs.msg import Twist;"
            "rospy.init_node(\"e_stop\",anonymous=True);"
            "p=rospy.Publisher(\"/joy_teleop/cmd_vel\",Twist,queue_size=1);"
            "time.sleep(0.3);"
            "[p.publish(Twist()) or time.sleep(0.1) for _ in range(5)]'"
        )

        import threading
        def _do_stop():
            try:
                subprocess.run(
                    ['ssh', '-o', 'StrictHostKeyChecking=no',
                     f'{HUSKY_USER}@{HUSKY_IP}', stop_cmd],
                    timeout=8
                )
            except Exception:
                pass
        threading.Thread(target=_do_stop, daemon=True).start()

        self.shared_log.log(
            "NAV", "STOP — SIGINT + zero velocity sent", "warn")

    # ── Mission management ────────────────────────────────────

    # ── Mission editor ────────────────────────────────────────

    def _new_mission(self):
        """Open editor with a blank template mission."""
        self._open_editor(
            filename="new_mission.yaml",
            content=TEMPLATE_YAML)

    def _edit_mission(self):
        """Open selected mission YAML in built-in editor."""
        name = self.cmb_mission.currentText()
        if not name:
            return
        path = MISSIONS_LOCAL / name
        if not path.exists():
            self.shared_log.log("NAV", f"Mission file not found: {name}", "error")
            return
        try:
            with open(path) as f:
                content = f.read()
            self._open_editor(filename=name, content=content, path=path)
        except Exception as e:
            self.shared_log.log("NAV", f"Cannot open {name}: {e}", "error")

    def _open_editor(self, filename: str, content: str, path=None):
        """Open built-in YAML editor dialog."""
        dlg = MissionEditorDialog(
            parent=self,
            filename=filename,
            content=content,
            save_path=path,
            missions_dir=MISSIONS_LOCAL,
            log_fn=lambda msg, tag="ok": self.shared_log.log("NAV", msg, tag),
            refresh_fn=self._refresh_missions,
            sync_fn=self._sync_mission,
        )
        dlg.exec_()

    def _delete_mission(self):
        """Delete selected mission after confirmation."""
        name = self.cmb_mission.currentText()
        if not name:
            return
        path = MISSIONS_LOCAL / name
        reply = QMessageBox.question(
            self, "Delete Mission",
            f"Delete '{name}'?\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            try:
                path.unlink(missing_ok=True)
                self.shared_log.log("NAV", f"Deleted: {name}", "warn")
                self._refresh_missions()
            except Exception as e:
                self.shared_log.log("NAV", f"Delete failed: {e}", "error")

    def _refresh_missions(self):
        self.cmb_mission.blockSignals(True)
        cur = self.cmb_mission.currentText()
        self.cmb_mission.clear()
        yamls = sorted(MISSIONS_LOCAL.glob("*.yaml"))
        for y in yamls:
            self.cmb_mission.addItem(y.name, str(y))
        if cur:
            idx = self.cmb_mission.findText(cur)
            if idx >= 0:
                self.cmb_mission.setCurrentIndex(idx)
        self.cmb_mission.blockSignals(False)
        if self.cmb_mission.count() > 0:
            self._on_mission_selected(
                self.cmb_mission.currentText())

    def _browse_mission(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Mission File",
            str(MISSIONS_LOCAL),
            "YAML files (*.yaml *.yml);;All files (*)")
        if path:
            import shutil
            p = Path(path)
            if p.parent != MISSIONS_LOCAL:
                shutil.copy(p, MISSIONS_LOCAL / p.name)
            self._refresh_missions()
            idx = self.cmb_mission.findText(p.name)
            if idx >= 0:
                self.cmb_mission.setCurrentIndex(idx)

    def _on_mission_selected(self, name: str):
        path = self.cmb_mission.currentData()
        if not path:
            return
        try:
            import yaml
            with open(path) as f:
                m = yaml.safe_load(f)
            n_steps = len(m.get('steps', []))
            ret     = m.get('return_home', True)
            lin     = m.get('default_linear_speed', 0.1)
            ang     = m.get('default_angular_speed', 0.1)
            self.lbl_mission_desc.setText(
                f"{m.get('description','—')}  |  "
                f"{n_steps} steps  |  "
                f"return={ret}  |  "
                f"lin={lin}m/s  ang={ang}rad/s"
            )
        except Exception as e:
            self.lbl_mission_desc.setText(f"Parse error: {e}")

    def _sync_mission(self):
        path = self.cmb_mission.currentData()
        if not path:
            return
        name = Path(path).name
        self._mission_log(f"Syncing {name} to Husky...")
        result = subprocess.run([
            'scp', '-o', 'StrictHostKeyChecking=no',
            path,
            f'{HUSKY_USER}@{HUSKY_IP}:{MISSIONS_HUSKY}/{name}'
        ], capture_output=True, text=True)
        if result.returncode == 0:
            self._mission_log(f"✓ Synced: {name}")
            self.shared_log.log("NAV", f"Mission synced: {name}", "ok")
        else:
            self._mission_log(f"✗ Sync failed")
            self.shared_log.log(
                "NAV", f"Sync failed: {result.stderr}", "error")

    def _dry_run(self):
        path = self.cmb_mission.currentData()
        if not path:
            return
        name = Path(path).name
        self._mission_log(f"Dry run: {name}")
        proc = subprocess.Popen([
            'ssh', '-o', 'StrictHostKeyChecking=no',
            f'{HUSKY_USER}@{HUSKY_IP}',
            f'{ROS_SOURCE}python3 {FIELD_NAV} '
            f'{MISSIONS_HUSKY}/{name} --dry-run'
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out, _ = proc.communicate(timeout=15)
        for line in out.strip().split('\n'):
            if line.strip():
                self._mission_log(line)

    def _run_mission(self):
        path = self.cmb_mission.currentData()
        if not path:
            return
        name = Path(path).name
        self._sync_mission()
        self._mission_log(f"▶ Starting: {name}")
        self.shared_log.log("NAV", f"Mission START: {name}", "ok")

        self._mission_proc = subprocess.Popen([
            'ssh', '-o', 'StrictHostKeyChecking=no',
            f'{HUSKY_USER}@{HUSKY_IP}',
            f'{ROS_SOURCE}python3 {FIELD_NAV} '
            f'{MISSIONS_HUSKY}/{name}'
        ], stdout=subprocess.PIPE,
           stderr=subprocess.STDOUT,
           text=True)

        self._mission_running = True
        self.mission_progress.setValue(0)
        self.mission_progress.setFormat("Running...")
        self.btn_run_mission.setEnabled(False)
        self.btn_stop_mission.setEnabled(True)

        # Poll output every 500ms
        self._log_timer = QTimer()
        self._log_timer.timeout.connect(self._poll_mission)
        self._log_timer.start(500)

    def _poll_mission(self):
        if self._mission_proc is None:
            return
        if self._mission_proc.poll() is not None:
            remaining = self._mission_proc.stdout.read()
            for line in remaining.strip().split('\n'):
                if line.strip():
                    self._parse_mission_line(line)
            self._mission_done(self._mission_proc.returncode == 0)
            return
        import select
        readable, _, _ = select.select(
            [self._mission_proc.stdout], [], [], 0)
        if readable:
            line = self._mission_proc.stdout.readline()
            if line:
                self._parse_mission_line(line.rstrip())

    def _parse_mission_line(self, line: str):
        self._mission_log(line)
        import re
        m = re.search(r'\[STEP (\d+)/(\d+)\]', line)
        if m:
            cur   = int(m.group(1))
            total = int(m.group(2))
            pct   = int(cur / total * 100)
            self.mission_progress.setValue(pct)
            self.mission_progress.setFormat(
                f"Step {cur}/{total}")
            label = line.split(']', 1)[-1].strip()
            self.lbl_mission_step.setText(
                f"Step {cur}/{total}: {label}")
            self.shared_log.log(
                "NAV", f"Mission step {cur}/{total}", "info")
        elif '[RETURN HOME]' in line:
            self.lbl_mission_step.setText("Returning home...")
        elif 'MISSION COMPLETE' in line:
            self.mission_progress.setValue(100)
            self.mission_progress.setFormat("Complete ✓")

    def _mission_done(self, success: bool):
        self._mission_running = False
        if self._log_timer:
            self._log_timer.stop()
            self._log_timer = None
        self._mission_proc = None
        self.btn_run_mission.setEnabled(True)
        self.btn_stop_mission.setEnabled(False)
        if success:
            self.mission_progress.setValue(100)
            self.mission_progress.setFormat("Complete ✓")
            self._mission_log("✓ Mission complete")
            self.shared_log.log("NAV", "Mission COMPLETE", "ok")
        else:
            self.mission_progress.setFormat("Stopped")
            self._mission_log("✗ Mission stopped")
            self.shared_log.log("NAV", "Mission STOPPED", "warn")

    def _stop_mission(self):
        if self._log_timer:
            self._log_timer.stop()
        if self._mission_proc:
            self._mission_proc.terminate()
            self._mission_proc = None
        self._nav_stop()
        self._mission_done(False)

    def _mission_log(self, msg: str):
        import time
        ts  = time.strftime("%H:%M:%S")
        cur = self.mission_log.textCursor()
        cur.movePosition(QTextCursor.End)
        cur.insertText(f"[{ts}] {msg}\n")
        self.mission_log.setTextCursor(cur)
        self.mission_log.ensureCursorVisible()

    # ── Public API ────────────────────────────────────────────

    def set_ros_bridge(self, bridge_ref):
        """Update ros_bridge reference after detection arms."""
        self.ros_bridge_ref = bridge_ref

    def cleanup(self):
        self._odom_timer.stop()
        if self._log_timer:
            self._log_timer.stop()
        self._nav_kill()