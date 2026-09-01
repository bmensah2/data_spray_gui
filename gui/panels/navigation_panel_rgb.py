"""
gui/panels/navigation_panel_rgb.py
ABEN Dual RGB Imaging System — Navigation Panel

Fork of navigation_panel.py.
Identical to the multispec version plus a Spray Mission group
at the bottom: Start / Pause / End spray_mission_rgb.py,
progress bar, and Generate Report button.

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
import threading
import warnings
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QGroupBox, QDoubleSpinBox,
    QComboBox, QProgressBar, QPlainTextEdit,
    QFileDialog, QScrollArea, QTabWidget,
    QDialog, QTextEdit, QSplitter, QMessageBox,
    QLineEdit, QDialogButtonBox, QInputDialog,
    QCheckBox, QSpinBox
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

# Spray mission script (RGB) — on Jetson
SPRAY_MISSION_SCRIPT = (
    "/media/pagsun/Transcend/phd_project/emeet_dual_cam/spray_mission_rgb.py"
)
REPORT_SCRIPT = (
    "/media/pagsun/Transcend/phd_project/emeet_dual_cam/generate_report_rgb.js"
)

# Try Jetson home path first, fall back to SSD
_MISSIONS_HOME = Path("/home/pagsun/phd_project/emeet_dual_cam/missions")
_MISSIONS_SSD  = Path("/media/pagsun/Transcend/phd_project/emeet_dual_cam/missions")
MISSIONS_LOCAL = _MISSIONS_HOME if _MISSIONS_HOME.parent.exists() else _MISSIONS_SSD
# Nav scripts on Jetson (for sync reference)
NAV_SCRIPTS_LOCAL = Path(
    "/media/pagsun/Transcend/phd_project/emeet_dual_cam/navigation"
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
class NavigationPanelRGB(QWidget):
    """
    Shared navigation panel — manual moves + mission control.
    ros_bridge_ref: callable → ROSBridge or None
    """

    def __init__(self, shared_log: UnifiedLog,
                 ros_bridge_ref=None, parent=None):
        super().__init__(parent)
        self.shared_log    = shared_log
        self.ros_bridge_ref = ros_bridge_ref or (lambda: None)

        self._nav_proc        = None
        self._mission_proc    = None
        self._log_timer       = None
        self._mission_running = False
        # Spray mission state
        self._spray_proc:    subprocess.Popen = None
        self._spray_paused:  bool = False
        self._spray_running: bool = False
        self._spray_timer:   QTimer = None
        self._last_session_json: str = None

        MISSIONS_LOCAL.mkdir(parents=True, exist_ok=True)
        self._manual_grp_widget        = None
        self._quick_grp_widget         = None
        self._mission_grp_widget       = None
        self._spray_mission_grp_widget = None
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
        self._manual_grp_widget = self._manual_grp()
        lay.addWidget(self._manual_grp_widget)
        self._quick_grp_widget = self._quick_grp()
        lay.addWidget(self._quick_grp_widget)
        lay.addWidget(_divider())
        self._mission_grp_widget = self._mission_grp()
        lay.addWidget(self._mission_grp_widget)
        lay.addWidget(self._stop_grp())
        lay.addWidget(_divider())
        self._spray_mission_grp_widget = self._spray_mission_grp()
        lay.addWidget(self._spray_mission_grp_widget)
        lay.addStretch()

        scroll.setWidget(inner)
        outer.addWidget(scroll)

    # ── Cross-tab movement lock ─────────────────────────────────
    # Data Collection and Detection each get their own NavigationPanelRGB
    # instance (Qt widgets can't share a parent), but both talk to the
    # SAME physical Husky over independent SSH subprocesses -- with no
    # coordination between them, both tabs could issue conflicting
    # motion commands at once. Design: only one tab's controls are ever
    # enabled at a time. Data Collection starts enabled; Detection starts
    # locked until explicitly armed (see tab_detection.py _on_arm /
    # _on_stop / _on_estop, wired to this in main_gui_rgb.py).
    def set_movement_controls_enabled(self, enabled: bool):
        """
        Enable/disable everything that can command robot motion —
        manual moves, quick moves, mission start, spray mission start.
        The STOP button is intentionally NOT part of this group and
        always stays enabled/clickable.

        When disabling, also actively stops any motion this panel may
        have already started — locking the controls only prevents NEW
        commands, it doesn't interrupt one already in flight, so we
        stop explicitly to guarantee a clean handoff to the other tab.
        """
        for grp in (self._manual_grp_widget, self._quick_grp_widget,
                    self._mission_grp_widget, self._spray_mission_grp_widget):
            if grp is not None:
                grp.setEnabled(enabled)

        if not enabled:
            self._nav_stop()  # kill anything this panel may have started
            self.shared_log.log(
                "NAV", "Movement controls LOCKED — other tab is active",
                "warn")
        else:
            self.shared_log.log(
                "NAV", "Movement controls unlocked", "info")

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


    # ─────────────────────────────────────────────────────────
    #  SPRAY MISSION GROUP  (RGB-only addition)
    # ─────────────────────────────────────────────────────────

    def _spray_mission_grp(self) -> QGroupBox:
        """
        Spray Mission control panel.
        Runs spray_mission_rgb.py as a local subprocess on the Jetson.
        The script handles camera, detection, and Arduino via its own
        threads — the GUI just starts/pauses/stops it and shows progress.
        """
        grp = QGroupBox("Spray Mission  (RGB)")
        grp.setStyleSheet(
            "QGroupBox{border:1px solid #2a5a3a;"
            "border-radius:4px;margin-top:8px;"
            "color:#60c090;font-size:10px;font-weight:bold;}"
            "QGroupBox::title{subcontrol-origin:margin;padding:0 4px;}")
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)
        lay.setContentsMargins(8, 10, 8, 8)

        # ── Config row ─────────────────────────────────────────
        cfg_grid = QGridLayout()
        cfg_grid.setSpacing(4)

        cfg_grid.addWidget(_muted("Model:"), 0, 0)
        self.sm_model = QLineEdit(
            "/media/pagsun/Transcend/phd_project/"
            "emeet_dual_cam/models/weed_rgb.pt")
        self.sm_model.setStyleSheet(
            "background:#1a1e2e;color:#e8eaf0;"
            "border:1px solid #3a4055;border-radius:3px;"
            "font-family:Courier New;font-size:9px;padding:2px;")
        cfg_grid.addWidget(self.sm_model, 0, 1, 1, 2)
        btn_browse_model = QPushButton("Browse")
        btn_browse_model.setFixedWidth(60)
        theme_manager.register_button(btn_browse_model, "blue")
        btn_browse_model.clicked.connect(self._sm_browse_model)
        cfg_grid.addWidget(btn_browse_model, 0, 3)

        cfg_grid.addWidget(_muted("Dist (m):"), 1, 0)
        self.sm_dist = QDoubleSpinBox()
        self.sm_dist.setRange(0.1, 100.0)
        self.sm_dist.setValue(3.0)
        self.sm_dist.setSingleStep(0.5)
        self.sm_dist.setDecimals(1)
        cfg_grid.addWidget(self.sm_dist, 1, 1)

        cfg_grid.addWidget(_muted("Speed (m/s):"), 1, 2)
        self.sm_speed = QDoubleSpinBox()
        self.sm_speed.setRange(0.05, 0.5)
        self.sm_speed.setValue(0.3)
        self.sm_speed.setSingleStep(0.05)
        self.sm_speed.setDecimals(2)
        cfg_grid.addWidget(self.sm_speed, 1, 3)

        cfg_grid.addWidget(_muted("Field ID:"), 2, 0)
        self.sm_field_id = QLineEdit()
        self.sm_field_id.setPlaceholderText("e.g. Wilkin_Plot_A")
        self.sm_field_id.setStyleSheet(
            "background:#1a1e2e;color:#e8eaf0;"
            "border:1px solid #3a4055;border-radius:3px;"
            "font-family:Courier New;font-size:9px;padding:2px;")
        cfg_grid.addWidget(self.sm_field_id, 2, 1)

        cfg_grid.addWidget(_muted("Port:"), 2, 2)
        self.sm_port = QLineEdit("/dev/ttyACM0")
        self.sm_port.setStyleSheet(
            "background:#1a1e2e;color:#e8eaf0;"
            "border:1px solid #3a4055;border-radius:3px;"
            "font-family:Courier New;font-size:9px;padding:2px;")
        cfg_grid.addWidget(self.sm_port, 2, 3)

        lay.addLayout(cfg_grid)

        # ── Flags ─────────────────────────────────────────────
        flag_row = QHBoxLayout()
        self.sm_dry_run = QCheckBox("Dry run (log only)")
        self.sm_dry_run.setStyleSheet(
            "color:#f5a623;font-family:Courier New;font-size:9px;")
        flag_row.addWidget(self.sm_dry_run)
        self.sm_dummy = QCheckBox("Dummy detect (timing test)")
        self.sm_dummy.setStyleSheet(
            "color:#f5a623;font-family:Courier New;font-size:9px;")
        flag_row.addWidget(self.sm_dummy)
        flag_row.addStretch()
        lay.addLayout(flag_row)

        lay.addWidget(_divider())

        # ── Control buttons ────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self.btn_sm_start = QPushButton("▶  START MISSION")
        theme_manager.register_button(self.btn_sm_start, "green")
        self.btn_sm_start.setMinimumHeight(32)
        self.btn_sm_start.clicked.connect(self._sm_start)
        btn_row.addWidget(self.btn_sm_start)

        self.btn_sm_pause = QPushButton("⏸  PAUSE")
        theme_manager.register_button(self.btn_sm_pause, "amber")
        self.btn_sm_pause.setMinimumHeight(32)
        self.btn_sm_pause.setEnabled(False)
        self.btn_sm_pause.clicked.connect(self._sm_pause)
        btn_row.addWidget(self.btn_sm_pause)

        self.btn_sm_end = QPushButton("⏹  END MISSION")
        theme_manager.register_widget(self.btn_sm_end, _custom_btn(
            "QPushButton{background:#5a0000;color:#ff4040;"
            "border:2px solid #ff0000;border-radius:4px;"
            "padding:6px;font-weight:bold;}"
            "QPushButton:hover{background:#7a0000;}"
            "QPushButton:disabled{background:#1a0000;"
            "color:#3a1010;border-color:#3a0000;}",
            lambda p: (
                f"QPushButton{{background:{_darken(p['red'],55)};"
                f"color:{_lighten(p['red'],10)};"
                f"border:2px solid {p['red']};border-radius:4px;"
                f"padding:6px;font-weight:bold;}}"
                f"QPushButton:hover{{background:{_darken(p['red'],40)};}}"
                f"QPushButton:disabled{{background:{_darken(p['red'],70)};"
                f"color:{_darken(p['red'],40)};border-color:{_darken(p['red'],60)};}}"))
        )
        self.btn_sm_end.setMinimumHeight(32)
        self.btn_sm_end.setEnabled(False)
        self.btn_sm_end.clicked.connect(self._sm_end)
        btn_row.addWidget(self.btn_sm_end)

        lay.addLayout(btn_row)

        # ── Status + progress ──────────────────────────────────
        self.lbl_sm_status = QLabel("Ready")
        self.lbl_sm_status.setStyleSheet(
            "color:#8090a8;font-family:Courier New;"
            "font-size:9px;")
        lay.addWidget(self.lbl_sm_status)

        self.sm_progress = QProgressBar()
        self.sm_progress.setRange(0, 100)
        self.sm_progress.setValue(0)
        self.sm_progress.setFormat("Ready")
        lay.addWidget(self.sm_progress)

        # Mission log (compact)
        self.sm_log = QPlainTextEdit()
        self.sm_log.setReadOnly(True)
        self.sm_log.setMaximumBlockCount(80)
        self.sm_log.setFixedHeight(80)
        self.sm_log.setStyleSheet(
            "background:#050810;color:#6070a0;"
            "font-family:Courier New;font-size:9px;border:none;")
        lay.addWidget(self.sm_log)

        lay.addWidget(_divider())

        # ── Generate report ────────────────────────────────────
        rep_row = QHBoxLayout()
        rep_row.addWidget(_muted("Session JSON:"))
        self.sm_session_path = QLineEdit()
        self.sm_session_path.setPlaceholderText(
            "auto-filled after mission ends")
        self.sm_session_path.setStyleSheet(
            "background:#1a1e2e;color:#e8eaf0;"
            "border:1px solid #3a4055;border-radius:3px;"
            "font-family:Courier New;font-size:9px;padding:2px;")
        rep_row.addWidget(self.sm_session_path, stretch=1)
        btn_browse_session = QPushButton("Browse")
        btn_browse_session.setFixedWidth(60)
        theme_manager.register_button(btn_browse_session, "blue")
        btn_browse_session.clicked.connect(self._sm_browse_session)
        rep_row.addWidget(btn_browse_session)
        lay.addLayout(rep_row)

        self.btn_sm_report = QPushButton("📄  GENERATE REPORT")
        theme_manager.register_button(self.btn_sm_report, "blue")
        self.btn_sm_report.setMinimumHeight(30)
        self.btn_sm_report.clicked.connect(self._sm_generate_report)
        lay.addWidget(self.btn_sm_report)

        return grp

    # ── Spray mission actions ──────────────────────────────────

    def _sm_browse_model(self):
        dlg = QFileDialog(self, "Select RGB model weights")
        dlg.setFileMode(QFileDialog.ExistingFile)
        dlg.setNameFilter("Model files (*.pt *.engine)")
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)
        if dlg.exec_():
            files = dlg.selectedFiles()
            if files:
                self.sm_model.setText(files[0])

    def _sm_browse_session(self):
        dlg = QFileDialog(self, "Select session JSON")
        dlg.setFileMode(QFileDialog.ExistingFile)
        dlg.setNameFilter("JSON files (*.json)")
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)
        # Default to emeet_dual_cam directory
        import os
        default_dir = "/media/pagsun/Transcend/phd_project/emeet_dual_cam"
        if os.path.exists(default_dir):
            dlg.setDirectory(default_dir)
        if dlg.exec_():
            files = dlg.selectedFiles()
            if files:
                self.sm_session_path.setText(files[0])

    def _sm_log(self, msg: str):
        """Append a line to the spray mission log."""
        import time as _t
        ts  = _t.strftime("%H:%M:%S")
        cur = self.sm_log.textCursor()
        cur.movePosition(QTextCursor.End)
        cur.insertText(f"[{ts}] {msg}\n")
        self.sm_log.setTextCursor(cur)
        self.sm_log.ensureCursorVisible()
        self.shared_log.log("SPRAY", msg, "info")

    def _sm_start(self):
        """Build args and launch spray_mission_rgb.py as subprocess."""
        if self._spray_running:
            self.shared_log.log(
                "SPRAY", "Mission already running", "warn")
            return

        import shutil
        if not shutil.which("python3"):
            self.shared_log.log(
                "SPRAY", "python3 not found", "error")
            return

        script = SPRAY_MISSION_SCRIPT
        if not Path(script).exists():
            self.shared_log.log(
                "SPRAY",
                f"spray_mission_rgb.py not found at:\n{script}",
                "error")
            return

        # Build command
        cmd = [
            "python3", script,
            "--model",  self.sm_model.text().strip(),
            "--dist",   str(self.sm_dist.value()),
            "--speed",  str(self.sm_speed.value()),
            "--port",   self.sm_port.text().strip(),
        ]
        fid = self.sm_field_id.text().strip()
        if fid:
            cmd += ["--field-id", fid]
        # Use port 5006 for spray mission odom — port 5005 is used by GUI ROSBridge
        cmd += ["--odom-port", "5006"]
        if self.sm_dry_run.isChecked():
            cmd.append("--dry-run")
        if self.sm_dummy.isChecked():
            cmd.append("--dummy-detect")

        self._sm_log(f"Starting: {' '.join(cmd[2:])}")

        try:
            self._spray_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(Path(script).parent),
            )
        except Exception as e:
            self.shared_log.log(
                "SPRAY", f"Launch failed: {e}", "error")
            return

        self._spray_running = True
        self._spray_paused  = False

        # Update UI
        self.btn_sm_start.setEnabled(False)
        self.btn_sm_pause.setEnabled(True)
        self.btn_sm_end.setEnabled(True)
        self.sm_progress.setValue(0)
        self.sm_progress.setFormat("Running…")
        self.lbl_sm_status.setText("● RUNNING")
        self.lbl_sm_status.setStyleSheet(
            "color:#00c896;font-family:Courier New;"
            "font-size:9px;font-weight:bold;")

        # Poll output every 400ms
        self._spray_timer = QTimer()
        self._spray_timer.timeout.connect(self._sm_poll)
        self._spray_timer.start(400)

        self.shared_log.log(
            "SPRAY",
            f"Spray mission started — dist={self.sm_dist.value()}m "
            f"speed={self.sm_speed.value()}m/s",
            "ok")

    def _sm_pause(self):
        """Send SIGSTOP/SIGCONT to pause/resume the subprocess."""
        import signal
        if not self._spray_proc or not self._spray_running:
            return
        try:
            if not self._spray_paused:
                self._spray_proc.send_signal(signal.SIGSTOP)
                self._spray_paused = True
                self.btn_sm_pause.setText("▶  CONTINUE")
                self.lbl_sm_status.setText("⏸  PAUSED")
                self.lbl_sm_status.setStyleSheet(
                    "color:#f5a623;font-family:Courier New;"
                    "font-size:9px;font-weight:bold;")
                self.shared_log.log("SPRAY", "Mission PAUSED", "warn")
            else:
                self._spray_proc.send_signal(signal.SIGCONT)
                self._spray_paused = False
                self.btn_sm_pause.setText("⏸  PAUSE")
                self.lbl_sm_status.setText("● RUNNING")
                self.lbl_sm_status.setStyleSheet(
                    "color:#00c896;font-family:Courier New;"
                    "font-size:9px;font-weight:bold;")
                self.shared_log.log("SPRAY", "Mission RESUMED", "info")
        except Exception as e:
            self.shared_log.log(
                "SPRAY", f"Pause/resume failed: {e}", "warn")

    def _sm_end(self):
        """Terminate the spray mission subprocess."""
        if self._spray_timer:
            self._spray_timer.stop()
            self._spray_timer = None
        if self._spray_proc:
            try:
                self._spray_proc.terminate()
                self._spray_proc.wait(timeout=3)
            except Exception:
                try:
                    self._spray_proc.kill()
                except Exception:
                    pass
            self._spray_proc = None

        self._spray_running = False
        self._spray_paused  = False
        self._sm_done(success=False)

    def _sm_poll(self):
        """
        Called every 400ms — reads subprocess stdout and updates UI.
        Detects completion and session JSON path.
        """
        if self._spray_proc is None:
            return

        # Check if process finished
        if self._spray_proc.poll() is not None:
            # Drain remaining output
            try:
                remaining = self._spray_proc.stdout.read()
                for line in remaining.strip().split("\n"):
                    if line.strip():
                        self._sm_parse_line(line.strip())
            except Exception:
                pass
            self._sm_done(success=self._spray_proc.returncode == 0)
            return

        # Read available lines non-blocking
        import select
        try:
            readable, _, _ = select.select(
                [self._spray_proc.stdout], [], [], 0)
            if readable:
                line = self._spray_proc.stdout.readline()
                if line:
                    self._sm_parse_line(line.rstrip())
        except Exception:
            pass

    def _sm_parse_line(self, line: str):
        """Parse a line from spray_mission_rgb.py stdout."""
        import re
        self._sm_log(line)

        # Distance progress: "[MISSION] 1.23m / 3.0m"
        m = re.search(r"(\d+\.\d+)m\s*/\s*(\d+\.\d+)m", line)
        if m:
            traveled = float(m.group(1))
            target   = float(m.group(2))
            pct = min(100, int(traveled / target * 100)) if target > 0 else 0
            self.sm_progress.setValue(pct)
            self.sm_progress.setFormat(
                f"{traveled:.2f}m / {target:.1f}m")

        # Session JSON path written by spray_mission_rgb.py
        m2 = re.search(r"[Ss]ession\s+JSON.*?:\s*(\S+\.json)", line)
        if m2:
            self._last_session_json = m2.group(1)
            self.sm_session_path.setText(self._last_session_json)

        # Target reached
        if "target reached" in line.lower() or "COMPLETE" in line:
            self.sm_progress.setValue(100)
            self.sm_progress.setFormat("Complete ✓")

    def _sm_done(self, success: bool):
        """Clean up after mission ends."""
        if self._spray_timer:
            self._spray_timer.stop()
            self._spray_timer = None
        self._spray_proc    = None
        self._spray_running = False
        self._spray_paused  = False

        self.btn_sm_start.setEnabled(True)
        self.btn_sm_pause.setEnabled(False)
        self.btn_sm_pause.setText("⏸  PAUSE")
        self.btn_sm_end.setEnabled(False)

        if success:
            self.sm_progress.setValue(100)
            self.sm_progress.setFormat("Complete ✓")
            self.lbl_sm_status.setText("✓ Mission complete")
            self.lbl_sm_status.setStyleSheet(
                "color:#00c896;font-family:Courier New;"
                "font-size:9px;font-weight:bold;")
            self.shared_log.log("SPRAY", "Mission COMPLETE", "ok")
        else:
            self.sm_progress.setFormat("Stopped")
            self.lbl_sm_status.setText("⏹ Stopped")
            self.lbl_sm_status.setStyleSheet(
                "color:#8090a8;font-family:Courier New;"
                "font-size:9px;")
            self.shared_log.log("SPRAY", "Mission STOPPED", "warn")

        # Auto-fill session path if found during polling
        if self._last_session_json:
            self.sm_session_path.setText(self._last_session_json)
            self._sm_log(
                f"Session JSON: {self._last_session_json}")
            self._sm_log(
                "Click GENERATE REPORT to build the Word document.")

    def _sm_generate_report(self):
        """Run generate_report_rgb.js with node."""
        import shutil
        session_path = self.sm_session_path.text().strip()

        if not session_path:
            self.shared_log.log(
                "SPRAY",
                "No session JSON — run a mission first or browse to file",
                "warn")
            return

        if not Path(session_path).exists():
            self.shared_log.log(
                "SPRAY",
                f"Session file not found: {session_path}",
                "error")
            return

        if not shutil.which("node"):
            self.shared_log.log(
                "SPRAY",
                "node not found — install Node.js to generate reports",
                "error")
            return

        import time as _t
        out_path = (
            Path(session_path).parent /
            f"ABEN_RGB_Report_{_t.strftime('%Y%m%d_%H%M%S')}.docx"
        )

        # Check for validation JSON alongside session
        val_path = Path(session_path).parent / "model_validation_rgb.json"
        cmd = [
            "node", REPORT_SCRIPT,
            "--session", session_path,
            "--out",     str(out_path),
        ]
        if val_path.exists():
            cmd += ["--validation", str(val_path)]

        self._sm_log(f"Generating report → {out_path.name} …")
        self.shared_log.log("SPRAY", "Generating report…", "info")

        from PyQt5.QtCore import QMetaObject, Q_ARG
        from PyQt5.QtCore import Qt as _Qt

        def _log_safe(msg: str):
            """Route log call safely to main thread."""
            QMetaObject.invokeMethod(
                self, "_sm_log_safe",
                _Qt.QueuedConnection,
                Q_ARG(str, msg),
            )

        def _run():
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True,
                    cwd=str(Path(REPORT_SCRIPT).parent),
                    timeout=60,
                )
                if result.returncode == 0:
                    _log_safe(f"✓ Report: {out_path.name}")
                    QMetaObject.invokeMethod(
                        self, "_sm_report_done",
                        _Qt.QueuedConnection,
                        Q_ARG(str, str(out_path)),
                        Q_ARG(bool, True),
                        Q_ARG(str, ""),
                    )
                else:
                    err = result.stderr.strip()[:200]
                    _log_safe(f"✗ Report failed: {err}")
                    QMetaObject.invokeMethod(
                        self, "_sm_report_done",
                        _Qt.QueuedConnection,
                        Q_ARG(str, str(out_path)),
                        Q_ARG(bool, False),
                        Q_ARG(str, err),
                    )
            except subprocess.TimeoutExpired:
                _log_safe("✗ Report timed out (>60s)")
                QMetaObject.invokeMethod(
                    self, "_sm_report_done",
                    _Qt.QueuedConnection,
                    Q_ARG(str, ""),
                    Q_ARG(bool, False),
                    Q_ARG(str, "Process timed out after 60s"),
                )
            except Exception as e:
                _log_safe(f"✗ Report error: {e}")

        threading.Thread(target=_run, daemon=True).start()

    from PyQt5.QtCore import pyqtSlot

    @pyqtSlot(str)
    def _sm_log_safe(self, msg: str):
        """Thread-safe log — always called on main thread via invokeMethod."""
        self._sm_log(msg)

    @pyqtSlot(str, bool, str)
    def _sm_report_done(self, out_path: str, success: bool, err: str):
        """Called on the main thread when report generation finishes."""
        from PyQt5.QtWidgets import QMessageBox
        if success:
            msg = QMessageBox(self)
            msg.setWindowTitle("Report Generated")
            msg.setIcon(QMessageBox.Information)
            msg.setText(
                f"<b>Report generation complete!</b>")
            msg.setInformativeText(
                f"Saved to:<br><tt>{out_path}</tt>")
            msg.setDetailedText(
                f"Full path: {out_path}\n\n"
                f"Open with Microsoft Word or LibreOffice."
            )
            msg.setStandardButtons(QMessageBox.Ok)
            msg.setOption(QMessageBox.StandardButton.Ok)
            msg.exec_()
        else:
            msg = QMessageBox(self)
            msg.setWindowTitle("Report Failed")
            msg.setIcon(QMessageBox.Warning)
            msg.setText("<b>Report generation failed.</b>")
            msg.setInformativeText(
                "Check that Node.js is installed and "
                "generate_report_rgb.js is present.")
            if err:
                msg.setDetailedText(f"Error:\n{err}")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

    # ── Public API ────────────────────────────────────────────

    def set_ros_bridge(self, bridge_ref):
        """Update ros_bridge reference after detection arms."""
        self.ros_bridge_ref = bridge_ref

    def cleanup(self):
        self._odom_timer.stop()
        if self._log_timer:
            self._log_timer.stop()
        # Stop any running spray mission
        if self._spray_running:
            self._sm_end()
        self._nav_kill()