"""
gui/tabs/tab_analysis_triple.py
ABEN Triple RGB Session Analysis Tab

Group E of the triple-camera standalone app's capability pass: the
Session Analysis tab, which didn't exist at all for the triple-camera
system before this. Ported from gui/tabs/tab_analysis_rgb.py's
AnalysisTabRGB, reusing its shared, already camera-count-agnostic
building blocks directly:
  - gui/spray_event_table.py's build_spray_event_table()/
    insert_spray_event_row() (Time/Zone/Nozzle/Class/Conf/Pose/GPS
    columns) -- the same live-feed table the Group A fullscreen popout
    would have used had DetectionPanelTriple had the right signals at
    the time (it does now, see below).
  - Word report generation (_on_generate_report()) is reused nearly
    verbatim: it already reads self.detect._last_report_path (built
    by Group C's _write_session_report()) and shells out to
    generate_gui_session_report.js, neither of which has any
    camera-count dependency at all.

Required adding two Qt signals DetectionPanelTriple didn't have yet
(spray_event_signal, session_started) plus get_actuation_status()/
get_husky_status() wrapper methods -- all now added there.

Deliberately different from the 2-camera version: "Events by Zone"
groups directly by nozzle (N1/N2/N3), since the triple-camera system
already has a strict 1:1 camera-to-nozzle mapping (Phase 1) -- no
more A/B1/B2/C-style zone names to translate.

Deliberately scoped for this first pass -- ported the live event
feed, session stats, system status, and report generation, and left
out for a later pass (none of which block the tab's core purpose,
reviewing what happened this session and producing a report):
  - The spray location map (_map_grp()/_redraw_map(), an OpenCV-drawn
    2D scatter of GPS/pose positions) -- a real feature, but a
    separate, self-contained visual piece that doesn't block anything
    else here.
"""

import time
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QPushButton,
)
from PyQt5.QtCore import QTimer, pyqtSlot

from gui.style import _muted, _sec, _divider
from gui.theme_manager import theme_manager
from gui.shared_log import UnifiedLog, LogPanel

try:
    from gui.spray_event_table import (
        build_spray_event_table, insert_spray_event_row)
except ImportError:
    from spray_event_table import (
        build_spray_event_table, insert_spray_event_row)

MAX_FEED_ROWS = 500


class AnalysisTabTriple(QWidget):
    """
    Session Analysis tab for the triple-camera system: live spray
    event feed, session stats, system status, and report generation.

    Args:
        gantry: GantryPanel (shared with the other tabs) -- pump/
                nozzle/Arduino connection status.
        detect: DetectionPanelTriple -- the source of
                spray_event_signal/session_started, and of
                get_actuation_status()/get_husky_status().
    """

    def __init__(self, gantry, detect, parent=None):
        super().__init__(parent)
        self.gantry = gantry
        self.detect = detect
        self.log    = UnifiedLog()

        self._session_id    = None
        self._session_start = None
        self._events        = []

        self._build_ui()

        self.detect.spray_event_signal.connect(self._on_spray_event)
        self.detect.session_started.connect(self._on_session_started)

        self._status_timer = QTimer()
        self._status_timer.timeout.connect(self._refresh_status)
        self._status_timer.start(500)

        self.log.log("SYS", "Analysis tab ready — live session dashboard", "ok")

    # ── UI construction ───────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        root.addWidget(_sec("SESSION ANALYSIS  —  Live Event Feed & System Status"))
        root.addWidget(_divider())

        hdr = QHBoxLayout()
        self.lbl_session = _muted("No session — ARM detection to start")
        hdr.addWidget(self.lbl_session)
        hdr.addStretch()
        self.lbl_duration = _muted("--:--")
        hdr.addWidget(self.lbl_duration)
        self.btn_clear = QPushButton("Clear Feed")
        theme_manager.register_button(self.btn_clear, "blue")
        self.btn_clear.setFixedHeight(24)
        self.btn_clear.clicked.connect(self._on_clear)
        hdr.addWidget(self.btn_clear)
        self.btn_report = QPushButton("📄  Generate Word Report")
        theme_manager.register_button(self.btn_report, "green")
        self.btn_report.setFixedHeight(24)
        self.btn_report.setToolTip(
            "Builds a publication-quality .docx from the most recently "
            "completed ARM DETECTION session (events, camera settings, "
            "model info, software versions).")
        self.btn_report.clicked.connect(self._on_generate_report)
        hdr.addWidget(self.btn_report)
        root.addLayout(hdr)

        main_split = QHBoxLayout()
        main_split.setSpacing(8)
        main_split.addWidget(self._event_feed_grp(), stretch=2)

        right_col = QVBoxLayout()
        right_col.setSpacing(8)
        right_col.addWidget(self._stats_grp())
        right_col.addWidget(self._zone_grp())
        right_col.addWidget(self._status_grp())
        right_col.addStretch()
        main_split.addLayout(right_col, stretch=1)
        root.addLayout(main_split, stretch=2)

        root.addWidget(_divider())
        root.addWidget(
            LogPanel(self.log, sources=["ANALYSIS", "DETECT", "SYS"], height=90))

    def _event_feed_grp(self):
        grp = QGroupBox("Live Spray Event Feed")
        lay = QVBoxLayout(grp)
        lay.setContentsMargins(4, 4, 4, 4)
        self.tbl_events = build_spray_event_table()
        lay.addWidget(self.tbl_events)
        return grp

    def _stats_grp(self):
        grp = QGroupBox("Session Stats")
        gl = QGridLayout(grp)
        gl.setSpacing(4)
        gl.setContentsMargins(8, 8, 8, 8)

        def _row(label, attr, row):
            gl.addWidget(_muted(label), row, 0)
            lbl = QLabel("—")
            theme_manager.register_widget(
                lbl, lambda p: (
                    f"color:{p['green']};font-size:10px;"
                    f"font-family:'Noto Sans',Arial,sans-serif;"))
            gl.addWidget(lbl, row, 1)
            setattr(self, attr, lbl)

        _row("Total events:",     "stat_total",      0)
        _row("Events/min:",       "stat_rate",       1)
        _row("Confidence mean:",  "stat_conf_mean",  2)
        _row("Confidence range:", "stat_conf_range", 3)
        _row("GPS coverage:",     "stat_gps",        4)
        _row("CLS flagged:",      "stat_cls",        5)
        return grp

    def _zone_grp(self):
        grp = QGroupBox("Events by Nozzle / Class")
        gl = QGridLayout(grp)
        gl.setSpacing(4)
        gl.setContentsMargins(8, 8, 8, 8)

        gl.addWidget(_muted("By nozzle:"), 0, 0)
        self.lbl_by_zone = QLabel("—")
        self.lbl_by_zone.setWordWrap(True)
        theme_manager.register_widget(
            self.lbl_by_zone, lambda p: (
                f"color:{p['text_dim']};font-size:10px;"
                f"font-family:'Noto Sans',Arial,sans-serif;"))
        gl.addWidget(self.lbl_by_zone, 0, 1)

        gl.addWidget(_muted("By class:"), 1, 0)
        self.lbl_by_class = QLabel("—")
        self.lbl_by_class.setWordWrap(True)
        theme_manager.register_widget(
            self.lbl_by_class, lambda p: (
                f"color:{p['text_dim']};font-size:10px;"
                f"font-family:'Noto Sans',Arial,sans-serif;"))
        gl.addWidget(self.lbl_by_class, 1, 1)
        return grp

    def _status_grp(self):
        grp = QGroupBox("System Status")
        gl = QGridLayout(grp)
        gl.setSpacing(4)
        gl.setContentsMargins(8, 8, 8, 8)

        def _row(label, attr, row):
            gl.addWidget(_muted(label), row, 0)
            lbl = QLabel("—")
            theme_manager.register_widget(
                lbl, lambda p: (
                    f"color:{p['text_dim']};font-size:10px;"
                    f"font-family:'Noto Sans',Arial,sans-serif;"))
            gl.addWidget(lbl, row, 1)
            setattr(self, attr, lbl)

        _row("Detection:",  "stat_armed",   0)
        _row("Mode:",       "stat_mode",    1)
        _row("Arduino:",    "stat_arduino", 2)
        _row("Pump:",       "stat_pump",    3)
        _row("Nozzles:",    "stat_nozzles", 4)
        _row("Husky link:", "stat_husky",   5)
        _row("EStop:",      "stat_estop",   6)
        return grp

    def _style_label(self, lbl, palette_key, extra=""):
        theme_manager.register_widget(
            lbl, lambda p: (
                f"color:{p.get(palette_key, p['text'])};font-size:10px;"
                f"font-family:'Noto Sans',Arial,sans-serif;{extra}"))

    # ── Session lifecycle ──────────────────────────────────────

    def _on_session_started(self, session_id):
        self._session_id    = session_id
        self._session_start = time.time()
        self._events        = []
        self.tbl_events.setRowCount(0)
        self.lbl_session.setText(f"Session: {session_id}")
        self._update_stats_labels()
        self.log.log("ANALYSIS", f"New session: {session_id} — feed cleared", "info")

    def _on_clear(self):
        self._events = []
        self.tbl_events.setRowCount(0)
        self._update_stats_labels()
        self.log.log("ANALYSIS", "Feed cleared manually", "info")

    # ── Spray event handling ──────────────────────────────────

    def _on_spray_event(self, event):
        self._events.append(event)
        insert_spray_event_row(self.tbl_events, event, max_rows=MAX_FEED_ROWS)
        self._update_stats_labels()

    # ── Stats ─────────────────────────────────────────────────

    def _update_stats_labels(self):
        n = len(self._events)
        self.stat_total.setText(str(n))

        if n == 0 or self._session_start is None:
            for lbl in (self.stat_rate, self.stat_conf_mean,
                       self.stat_conf_range, self.stat_gps, self.stat_cls,
                       self.lbl_by_zone, self.lbl_by_class):
                lbl.setText("—")
            return

        elapsed_min = max((time.time() - self._session_start) / 60.0, 1e-6)
        self.stat_rate.setText(f"{n / elapsed_min:.1f}")

        confs = []
        for e in self._events:
            if e.detections:
                confs.append(max(d.get('confidence', 0.0) for d in e.detections))
        if confs:
            self.stat_conf_mean.setText(f"{sum(confs)/len(confs):.2f}")
            self.stat_conf_range.setText(f"{min(confs):.2f} – {max(confs):.2f}")
        else:
            self.stat_conf_mean.setText("—")
            self.stat_conf_range.setText("—")

        gps_valid = sum(1 for e in self._events
                        if e.gps and e.gps.get('fix_valid'))
        self.stat_gps.setText(f"{gps_valid}/{n} ({gps_valid/n*100:.0f}%)")

        cls_count = sum(1 for e in self._events if getattr(e, 'flagged_cls', False))
        self.stat_cls.setText(str(cls_count))

        zone_counts  = {}
        class_counts = {}
        for e in self._events:
            zone_counts[e.zone_name] = zone_counts.get(e.zone_name, 0) + 1
            for d in e.detections:
                name = d.get('class_name', '?')
                class_counts[name] = class_counts.get(name, 0) + 1

        self.lbl_by_zone.setText(
            "  ".join(f"{z}:{c}" for z, c in sorted(zone_counts.items())))
        self.lbl_by_class.setText(
            "  ".join(f"{c}:{n}" for c, n in
                      sorted(class_counts.items(), key=lambda kv: -kv[1])))

    # ── Periodic system status refresh ────────────────────────

    def _refresh_status(self):
        if self._session_start is not None:
            elapsed = int(time.time() - self._session_start)
            self.lbl_duration.setText(f"{elapsed // 60:02d}:{elapsed % 60:02d}")

        gctrl  = getattr(self.gantry, 'ctrl', None)
        gstate = gctrl.state if gctrl is not None else None
        if gstate is not None and gstate.connected:
            self.stat_arduino.setText("Connected")
            self._style_label(self.stat_arduino, 'green')
            self.stat_pump.setText("ON" if gstate.pump_on else "off")
            self._style_label(self.stat_pump,
                              'green' if gstate.pump_on else 'muted')
            noz = gstate.nozzles or [False, False, False]
            self.stat_nozzles.setText(
                "  ".join(f"N{i+1}:{'ON' if v else 'off'}"
                         for i, v in enumerate(noz)))
            self._style_label(self.stat_nozzles, 'text_dim')
        else:
            self.stat_arduino.setText("Not connected")
            self._style_label(self.stat_arduino, 'amber')
            self.stat_pump.setText("—")
            self.stat_nozzles.setText("—")
            self._style_label(self.stat_pump, 'muted')
            self._style_label(self.stat_nozzles, 'muted')

        act = self.detect.get_actuation_status()
        if act is not None:
            self.stat_armed.setText("ARMED" if act['armed'] else "disarmed")
            self._style_label(self.stat_armed,
                              'amber' if act['armed'] else 'muted',
                              extra="font-weight:bold;")
            self.stat_mode.setText(
                f"{act['mode'].upper()}"
                f"{'  [DRY RUN]' if act['dry_run'] else ''}")
            self._style_label(self.stat_mode, 'text_dim')

            estop = act.get('estop_active') or act.get('manual_estop_active')
            self.stat_estop.setText("ACTIVE" if estop else "clear")
            self._style_label(self.stat_estop,
                              'red' if estop else 'green',
                              extra="font-weight:bold;")
        else:
            self.stat_armed.setText("disarmed")
            self._style_label(self.stat_armed, 'muted')
            self.stat_mode.setText("—")
            self.stat_estop.setText("—")
            self._style_label(self.stat_mode, 'muted')
            self._style_label(self.stat_estop, 'muted')

        husky = self.detect.get_husky_status()
        if husky is not None:
            if husky['connected']:
                self.stat_husky.setText(
                    f"Connected (hb {husky['heartbeat_age_s']:.1f}s ago)")
                self._style_label(self.stat_husky, 'green')
            else:
                self.stat_husky.setText("No connection")
                self._style_label(self.stat_husky, 'amber')
        else:
            self.stat_husky.setText("—")
            self._style_label(self.stat_husky, 'muted')

    # ── Report generation ──────────────────────────────────────

    def _on_generate_report(self):
        """
        Build a .docx from the most recently completed ARM DETECTION
        session (self.detect._last_report_path, written by
        DetectionPanelTriple._write_session_report() on disarm).
        Reused nearly verbatim from AnalysisTabRGB's own
        _on_generate_report() -- reading the report JSON and shelling
        out to generate_gui_session_report.js has no camera-count
        dependency at all.
        """
        import shutil, subprocess, threading
        from PyQt5.QtCore import QMetaObject, Q_ARG, Qt as _Qt

        report_path = getattr(self.detect, "_last_report_path", None)
        if not report_path or not Path(report_path).exists():
            self.log.log(
                "ANALYSIS",
                "No completed session report yet — ARM detection, spray "
                "at least once, then STOP to generate one", "warn")
            return

        if not shutil.which("node"):
            self.log.log(
                "ANALYSIS",
                "node not found — install Node.js to generate reports",
                "error")
            return

        script = Path(__file__).resolve().parent.parent.parent / \
            "generate_gui_session_report.js"
        if not script.exists():
            self.log.log(
                "ANALYSIS",
                f"generate_gui_session_report.js not found at {script}",
                "error")
            return

        out_path = (Path(report_path).parent /
                   f"ABEN_Session_Report_"
                   f"{time.strftime('%Y%m%d_%H%M%S')}.docx")
        cmd = ["node", str(script),
              "--session", str(report_path), "--out", str(out_path)]

        self.log.log("ANALYSIS", f"Generating report → {out_path.name} …", "info")
        self.btn_report.setEnabled(False)

        def _log_safe(msg):
            QMetaObject.invokeMethod(
                self, "_report_log_safe", _Qt.QueuedConnection,
                Q_ARG(str, msg))

        def _run():
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True,
                    cwd=str(script.parent), timeout=60)
                if result.returncode == 0:
                    _log_safe(f"✓ Report: {out_path.name}")
                    QMetaObject.invokeMethod(
                        self, "_report_done", _Qt.QueuedConnection,
                        Q_ARG(str, str(out_path)),
                        Q_ARG(bool, True), Q_ARG(str, ""))
                else:
                    err = result.stderr.strip()[:200]
                    _log_safe(f"✗ Report failed: {err}")
                    QMetaObject.invokeMethod(
                        self, "_report_done", _Qt.QueuedConnection,
                        Q_ARG(str, str(out_path)),
                        Q_ARG(bool, False), Q_ARG(str, err))
            except subprocess.TimeoutExpired:
                _log_safe("✗ Report timed out (>60s)")
                QMetaObject.invokeMethod(
                    self, "_report_done", _Qt.QueuedConnection,
                    Q_ARG(str, ""), Q_ARG(bool, False),
                    Q_ARG(str, "Process timed out after 60s"))
            except Exception as e:
                _log_safe(f"✗ Report error: {e}")
                QMetaObject.invokeMethod(
                    self, "_report_done", _Qt.QueuedConnection,
                    Q_ARG(str, ""), Q_ARG(bool, False), Q_ARG(str, str(e)))

        threading.Thread(target=_run, daemon=True).start()

    @pyqtSlot(str)
    def _report_log_safe(self, msg: str):
        self.log.log("ANALYSIS", msg, "info")

    @pyqtSlot(str, bool, str)
    def _report_done(self, out_path: str, success: bool, err: str):
        from PyQt5.QtWidgets import QMessageBox
        self.btn_report.setEnabled(True)
        if success:
            msg = QMessageBox(self)
            msg.setWindowTitle("Report Generated")
            msg.setIcon(QMessageBox.Information)
            msg.setText("<b>Session report generated.</b>")
            msg.setInformativeText(f"Saved to:<br><tt>{out_path}</tt>")
            msg.setStandardButtons(QMessageBox.Ok)
        else:
            msg = QMessageBox(self)
            msg.setWindowTitle("Report Failed")
            msg.setIcon(QMessageBox.Warning)
            msg.setText("<b>Report generation failed.</b>")
            msg.setInformativeText(
                "Check that Node.js and its docx package are installed.")
            if err:
                msg.setDetailedText(f"Error:\n{err}")
            msg.setStandardButtons(QMessageBox.Ok)
        self._report_msgbox = msg
        msg.show()

    # ── Cleanup ───────────────────────────────────────────────

    def cleanup(self):
        self._status_timer.stop()
