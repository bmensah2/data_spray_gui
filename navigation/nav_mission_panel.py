"""
nav_mission_panel.py
ABEN Field Imaging System — Mission Loader Panel

Add to Navigate tab in main_gui_v2.py.
Import and instantiate MissionPanel inside _tab_navigate().
"""

import subprocess
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QComboBox, QProgressBar,
    QFileDialog, QPlainTextEdit
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QTextCharFormat, QTextCursor

# ─────────────────────────────────────────────────────────────
#  MISSION PANEL
# ─────────────────────────────────────────────────────────────
class MissionPanel(QWidget):
    """
    Mission loader and executor panel for the Navigate tab.

    Workflow:
      1. Select mission YAML from Jetson ~/missions/
      2. Sync to Husky PC via scp
      3. Run field_nav.py on Husky via SSH
      4. Monitor progress via log polling
      5. STOP MISSION = pkill field_nav.py on Husky
    """

    # ── Config ────────────────────────────────────────────────
    HUSKY_IP       = "192.168.131.1"
    HUSKY_USER     = "administrator"
    MISSIONS_LOCAL = Path.home() / "phd_project/multispec_camera/missions"
    MISSIONS_HUSKY = "~/missions"
    FIELD_NAV_PY   = "~/field_nav.py"
    LOG_DIR_HUSKY  = "~/missions/logs"

    def __init__(self, shared_log, parent=None):
        super().__init__(parent)
        self.shared_log    = shared_log
        self._mission_proc = None
        self._log_timer    = None
        self._step_count   = 0
        self._total_steps  = 0
        self._running      = False
        self.MISSIONS_LOCAL.mkdir(parents=True, exist_ok=True)
        self._build_ui()
        self._refresh_missions()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        # ── Mission selector ──────────────────────────────────
        sel_grp = QGroupBox("Mission File")
        sl = QVBoxLayout(sel_grp)
        sr = QHBoxLayout()
        self.mission_combo = QComboBox()
        self.mission_combo.setMinimumWidth(180)
        self.mission_combo.currentTextChanged.connect(self._on_mission_selected)
        sr.addWidget(self.mission_combo, stretch=1)
        btn_refresh = QPushButton("↻")
        btn_refresh.setFixedWidth(28)
        btn_refresh.setToolTip("Refresh mission list")
        btn_refresh.clicked.connect(self._refresh_missions)
        sr.addWidget(btn_refresh)
        btn_browse = QPushButton("Browse")
        btn_browse.setFixedWidth(60)
        btn_browse.clicked.connect(self._browse_mission)
        sr.addWidget(btn_browse)
        sl.addLayout(sr)

        self.mission_desc = QLabel("—")
        self.mission_desc.setStyleSheet(
            "color:#8090a8;font-size:9px;font-family:Courier New;")
        self.mission_desc.setWordWrap(True)
        sl.addWidget(self.mission_desc)
        lay.addWidget(sel_grp)

        # ── Mission info ──────────────────────────────────────
        info_grp = QGroupBox("Mission Info")
        il = QVBoxLayout(info_grp)
        self.mission_info = QLabel("No mission loaded")
        self.mission_info.setStyleSheet(
            "color:#a0b0c0;font-size:9px;font-family:Courier New;")
        self.mission_info.setWordWrap(True)
        il.addWidget(self.mission_info)
        lay.addWidget(info_grp)

        # ── Progress ──────────────────────────────────────────
        prog_grp = QGroupBox("Mission Progress")
        pl = QVBoxLayout(prog_grp)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Ready")
        pl.addWidget(self.progress_bar)
        self.step_label = QLabel("Step: —")
        self.step_label.setStyleSheet(
            "color:#8090a8;font-size:9px;font-family:Courier New;")
        pl.addWidget(self.step_label)
        lay.addWidget(prog_grp)

        # ── Mission log ───────────────────────────────────────
        log_grp = QGroupBox("Mission Log")
        ll = QVBoxLayout(log_grp)
        self.mission_log = QPlainTextEdit()
        self.mission_log.setReadOnly(True)
        self.mission_log.setMaximumBlockCount(200)
        self.mission_log.setFixedHeight(80)
        self.mission_log.setStyleSheet(
            "background:#0a0d14;color:#8090a8;"
            "font-family:Courier New;font-size:9px;border:none;")
        ll.addWidget(self.mission_log)
        lay.addWidget(log_grp)

        # ── Controls ──────────────────────────────────────────
        ctrl_grp = QGroupBox("Mission Control")
        cl = QVBoxLayout(ctrl_grp)

        # Sync + dry run row
        r1 = QHBoxLayout()
        self.btn_sync = QPushButton("⇅ Sync to Husky")
        self.btn_sync.setStyleSheet(
            "QPushButton{background:#1a2a3a;color:#60a0d0;"
            "border:1px solid #2a4a6a;border-radius:4px;padding:5px;}"
            "QPushButton:hover{background:#2a3a4a;}")
        self.btn_sync.clicked.connect(self._sync_mission)
        r1.addWidget(self.btn_sync)

        self.btn_dryrun = QPushButton("▷ Dry Run")
        self.btn_dryrun.setStyleSheet(
            "QPushButton{background:#1a2030;color:#8090c0;"
            "border:1px solid #2a3050;border-radius:4px;padding:5px;}"
            "QPushButton:hover{background:#2a3040;}")
        self.btn_dryrun.clicked.connect(self._dry_run)
        r1.addWidget(self.btn_dryrun)
        cl.addLayout(r1)

        # Run + Stop row
        r2 = QHBoxLayout()
        self.btn_run = QPushButton("▶  RUN MISSION")
        self.btn_run.setStyleSheet(
            "QPushButton{background:#007a50;color:#ffffff;"
            "border:none;border-radius:4px;padding:8px;"
            "font-size:12px;font-weight:bold;}"
            "QPushButton:hover{background:#009060;}"
            "QPushButton:disabled{background:#1a3028;color:#3a6040;}")
        self.btn_run.clicked.connect(self._run_mission)
        r2.addWidget(self.btn_run)

        self.btn_stop_mission = QPushButton("■  STOP MISSION")
        self.btn_stop_mission.setStyleSheet(
            "QPushButton{background:#5a0000;color:#ff4040;"
            "border:2px solid #ff0000;border-radius:4px;padding:8px;"
            "font-size:12px;font-weight:bold;}"
            "QPushButton:hover{background:#7a0000;}"
            "QPushButton:disabled{background:#1a0000;color:#3a1010;"
            "border-color:#3a0000;}")
        self.btn_stop_mission.setEnabled(False)
        self.btn_stop_mission.clicked.connect(self._stop_mission)
        r2.addWidget(self.btn_stop_mission)
        cl.addLayout(r2)
        lay.addWidget(ctrl_grp)

    # ── Mission file management ───────────────────────────────

    def _refresh_missions(self):
        """Scan local missions/ folder and populate combo."""
        self.mission_combo.blockSignals(True)
        cur = self.mission_combo.currentText()
        self.mission_combo.clear()
        yamls = sorted(self.MISSIONS_LOCAL.glob("*.yaml"))
        for y in yamls:
            self.mission_combo.addItem(y.name, str(y))
        if cur:
            idx = self.mission_combo.findText(cur)
            if idx >= 0:
                self.mission_combo.setCurrentIndex(idx)
        self.mission_combo.blockSignals(False)
        if self.mission_combo.count() > 0:
            self._on_mission_selected(self.mission_combo.currentText())

    def _browse_mission(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Mission File",
            str(self.MISSIONS_LOCAL),
            "YAML files (*.yaml *.yml);;All files (*)"
        )
        if path:
            p = Path(path)
            # Copy to local missions folder if not already there
            if p.parent != self.MISSIONS_LOCAL:
                import shutil
                shutil.copy(p, self.MISSIONS_LOCAL / p.name)
            self._refresh_missions()
            idx = self.mission_combo.findText(p.name)
            if idx >= 0:
                self.mission_combo.setCurrentIndex(idx)

    def _on_mission_selected(self, name: str):
        """Parse and display mission info when selection changes."""
        if not name:
            return
        path = self.mission_combo.currentData()
        if not path:
            return
        try:
            import yaml
            with open(path) as f:
                m = yaml.safe_load(f)
            n_steps = len(m.get('steps', []))
            ret = m.get('return_home', True)
            lin = m.get('default_linear_speed', 0.1)
            ang = m.get('default_angular_speed', 0.1)
            self.mission_desc.setText(
                m.get('description', '—'))
            self.mission_info.setText(
                f"Steps: {n_steps}  |  "
                f"Return home: {ret}  |  "
                f"Lin: {lin}m/s  |  "
                f"Ang: {ang}rad/s"
            )
            self._total_steps = n_steps + (1 if ret else 0)
        except Exception as e:
            self.mission_desc.setText(f"Parse error: {e}")

    # ── SSH helpers ───────────────────────────────────────────

    def _ssh(self, cmd: str, wait: bool = False) -> subprocess.Popen:
        """Run command on Husky via SSH."""
        proc = subprocess.Popen([
            'ssh', '-o', 'StrictHostKeyChecking=no',
            f'{self.HUSKY_USER}@{self.HUSKY_IP}',
            f'source /opt/ros/noetic/setup.bash && {cmd}'
        ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        if wait:
            proc.wait()
        return proc

    def _sync_mission(self):
        """scp current mission YAML to Husky ~/missions/."""
        path = self.mission_combo.currentData()
        if not path:
            self._log("No mission selected")
            return
        name = Path(path).name
        self._log(f"Syncing {name} to Husky...")
        result = subprocess.run([
            'scp', '-o', 'StrictHostKeyChecking=no',
            path,
            f'{self.HUSKY_USER}@{self.HUSKY_IP}:{self.MISSIONS_HUSKY}/{name}'
        ], capture_output=True, text=True)
        if result.returncode == 0:
            self._log(f"✓ Synced: {name}")
            self.shared_log.log("NAV", f"Mission synced: {name}", "ok")
        else:
            self._log(f"✗ Sync failed: {result.stderr}")
            self.shared_log.log("NAV", f"Sync failed: {result.stderr}", "error")

    def _dry_run(self):
        """Run field_nav.py --dry-run on Husky and show output."""
        path = self.mission_combo.currentData()
        if not path:
            return
        name = Path(path).name
        self._log(f"Dry run: {name}")
        proc = self._ssh(
            f'python3 {self.FIELD_NAV_PY} '
            f'{self.MISSIONS_HUSKY}/{name} --dry-run',
            wait=False
        )
        # Read output
        out, _ = proc.communicate(timeout=15)
        for line in out.strip().split('\n'):
            self._log(line)

    def _run_mission(self):
        """Execute mission on Husky — non-blocking."""
        path = self.mission_combo.currentData()
        if not path:
            self._log("No mission selected")
            return
        name = Path(path).name

        # Auto-sync first
        self._sync_mission()

        self._log(f"▶ Starting mission: {name}")
        self.shared_log.log("NAV", f"Mission START: {name}", "ok")

        self._mission_proc = subprocess.Popen([
            'ssh', '-o', 'StrictHostKeyChecking=no',
            f'{self.HUSKY_USER}@{self.HUSKY_IP}',
            f'source /opt/ros/noetic/setup.bash && '
            f'python3 {self.FIELD_NAV_PY} '
            f'{self.MISSIONS_HUSKY}/{name}'
        ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        self._running      = True
        self._step_count   = 0
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Running...")
        self.step_label.setText("Step: starting...")
        self.btn_run.setEnabled(False)
        self.btn_stop_mission.setEnabled(True)

        # Poll output every 500ms
        self._log_timer = QTimer()
        self._log_timer.timeout.connect(self._poll_mission)
        self._log_timer.start(500)

    def _poll_mission(self):
        """Read mission output line by line and update progress."""
        if self._mission_proc is None:
            return

        # Check if done
        if self._mission_proc.poll() is not None:
            # Read remaining output
            remaining = self._mission_proc.stdout.read()
            for line in remaining.strip().split('\n'):
                if line.strip():
                    self._parse_output_line(line)
            self._mission_finished(
                self._mission_proc.returncode == 0
            )
            return

        # Read available output without blocking
        import select
        readable, _, _ = select.select(
            [self._mission_proc.stdout], [], [], 0
        )
        if readable:
            line = self._mission_proc.stdout.readline()
            if line:
                self._parse_output_line(line.rstrip())

    def _parse_output_line(self, line: str):
        """Parse field_nav.py output to update progress."""
        self._log(line)

        # Detect step progress
        if '[STEP' in line and '/' in line:
            # e.g. [STEP 3/9]
            import re
            m = re.search(r'\[STEP (\d+)/(\d+)\]', line)
            if m:
                cur   = int(m.group(1))
                total = int(m.group(2))
                self._step_count   = cur
                self._total_steps  = total
                pct = int(cur / total * 100)
                self.progress_bar.setValue(pct)
                self.progress_bar.setFormat(f"Step {cur}/{total}")
                # Extract label if present
                label_part = line.split(']', 1)[-1].strip()
                self.step_label.setText(f"Step {cur}/{total}: {label_part}")
                self.shared_log.log(
                    "NAV", f"Mission step {cur}/{total}", "info")

        elif '[RETURN HOME]' in line:
            self.step_label.setText("Returning home...")
            self.progress_bar.setFormat("Returning home...")

        elif '[MISSION COMPLETE]' in line:
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("Complete ✓")

    def _mission_finished(self, success: bool):
        """Called when mission process exits."""
        self._running = False
        if self._log_timer:
            self._log_timer.stop()
            self._log_timer = None
        self._mission_proc = None

        self.btn_run.setEnabled(True)
        self.btn_stop_mission.setEnabled(False)

        if success:
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("Mission complete ✓")
            self._log("✓ Mission complete")
            self.shared_log.log("NAV", "Mission COMPLETE", "ok")
        else:
            self.progress_bar.setFormat("Stopped / Failed")
            self._log("✗ Mission stopped or failed")
            self.shared_log.log("NAV", "Mission STOPPED", "warn")

    def _stop_mission(self):
        """Kill field_nav.py on Husky + send zero velocity."""
        if self._log_timer:
            self._log_timer.stop()
        if self._mission_proc:
            self._mission_proc.terminate()
            self._mission_proc = None

        # Kill on Husky and send zero velocity
        subprocess.Popen([
            'ssh', '-o', 'StrictHostKeyChecking=no',
            f'{self.HUSKY_USER}@{self.HUSKY_IP}',
            (
                'pkill -f field_nav.py; '
                'source /opt/ros/noetic/setup.bash && '
                'python3 -c '
                "'import rospy, time;"
                'from geometry_msgs.msg import Twist;'
                "rospy.init_node('stop_node',anonymous=True);"
                "pub=rospy.Publisher('/cmd_vel',Twist,queue_size=1);"
                "time.sleep(0.2);pub.publish(Twist());time.sleep(0.2)'"
            )
        ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        self._mission_finished(False)
        self._log("■ Mission STOPPED")
        self.shared_log.log("NAV", "Mission STOP — zero velocity sent", "warn")

    def _log(self, msg: str):
        """Append to mission log display."""
        import time
        ts = time.strftime("%H:%M:%S")
        cur = self.mission_log.textCursor()
        cur.movePosition(QTextCursor.End)
        cur.insertText(f"[{ts}] {msg}\n")
        self.mission_log.setTextCursor(cur)
        self.mission_log.ensureCursorVisible()

    def cleanup(self):
        self._stop_mission()
