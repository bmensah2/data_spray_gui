"""
tab_analysis_rgb.py
ABEN Field Imaging System — Tab 3: Session Analysis

Live dashboard for the current detection session:
  - Live Spray Event Feed  — every SprayEvent, newest first
  - Session Stats          — running totals, by-zone/by-class breakdown
  - System Status          — Arduino/gantry, Husky, EStop, armed state
  - Spray Location Map     — 2D scatter of where sprays happened

Camera feeds live on Tabs 1/2 already, so this tab is intentionally
NOT a camera view -- it's what happened this session, not what the
camera currently sees.

Data sources (all already exist elsewhere in the app):
  - DetectionPanelRGB.spray_event_signal  — emitted per SprayEvent
  - DetectionPanelRGB.session_started     — emitted on ARM
  - DetectionPanelRGB.get_actuation_status() / get_husky_status()
  - GantryPanel.ctrl.state                — GantryState (pump/nozzles/etc)

Author : Nana | NDSU / PhD Imaging System
Path   : /media/pagsun/Transcend/phd_project/emeet_dual_cam/
"""

import time
import datetime

import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QGroupBox, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

from gui.style import _divider, _muted, _sec, BTN_BLUE
from gui.shared_log import UnifiedLog, LogPanel


# Cap how many rows the live feed table keeps on screen -- stats and
# the map still use the FULL session event list regardless of this,
# only the table view is capped for UI responsiveness over a long
# session with many events.
MAX_FEED_ROWS = 300

# Cap how many recent poses we keep for the map's light "path trail"
# line. Spray-event markers are never capped -- every real spray this
# session shows up on the map. Only the trail (a nice-to-have visual
# aid, not research data) is bounded.
MAX_TRAIL_POINTS = 2000


class AnalysisTabRGB(QWidget):
    """
    Tab 3 — Session Analysis (live event feed + stats + system status).

    Args:
        gantry:     GantryPanel (shared with Tabs 1/2) — for pump/
                    nozzle/Arduino connection status.
        detect:     DetectionPanelRGB (Tab 2's detection panel) — the
                    source of spray_event_signal / session_started,
                    and of get_actuation_status()/get_husky_status().
    """

    def __init__(self, gantry, detect, parent=None):
        super().__init__(parent)
        self.gantry = gantry
        self.detect = detect
        self.log    = UnifiedLog()

        # ── Session state ───────────────────────────────────────
        self._session_id    = None
        self._session_start = None
        self._events        = []   # full list of SprayEvent this session
        self._trail          = []  # recent (x, y) poses for the map line

        self._build_ui()

        # Wire live data sources
        self.detect.spray_event_signal.connect(self._on_spray_event)
        self.detect.session_started.connect(self._on_session_started)

        # Periodic refresh — system status + duration ticker + map redraw.
        # Spray events themselves are event-driven (the signal above),
        # this timer is only for things that change without a discrete
        # event: connection status, pump/nozzle state, elapsed time.
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

        # ── Session header row ─────────────────────────────────
        hdr = QHBoxLayout()
        self.lbl_session = _muted("No session — ARM detection to start")
        hdr.addWidget(self.lbl_session)
        hdr.addStretch()
        self.lbl_duration = _muted("--:--")
        hdr.addWidget(self.lbl_duration)
        self.btn_clear = QPushButton("Clear Feed")
        self.btn_clear.setStyleSheet(BTN_BLUE)
        self.btn_clear.setFixedHeight(24)
        self.btn_clear.clicked.connect(self._on_clear)
        hdr.addWidget(self.btn_clear)
        root.addLayout(hdr)

        # ── Main split: event feed (left) | stats+status (right) ──
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

        # ── Spray location map ─────────────────────────────────
        root.addWidget(self._map_grp(), stretch=1)

        # ── Log ───────────────────────────────────────────────
        root.addWidget(_divider())
        root.addWidget(
            LogPanel(self.log,
                     sources=["ANALYSIS", "DETECT", "SYS"],
                     height=90))

    def _event_feed_grp(self):
        grp = QGroupBox("Live Spray Event Feed")
        lay = QVBoxLayout(grp)
        lay.setContentsMargins(4, 4, 4, 4)

        cols = ["Time", "Zone", "Nozzle", "Class", "Conf", "Pose (x,y)", "GPS"]
        self.tbl_events = QTableWidget(0, len(cols))
        self.tbl_events.setHorizontalHeaderLabels(cols)
        self.tbl_events.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.tbl_events.verticalHeader().setVisible(False)
        self.tbl_events.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_events.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_events.setAlternatingRowColors(True)
        self.tbl_events.setStyleSheet(
            "QTableWidget{background-color:#0a0d14;color:#c0c4d0;"
            "font-family:Courier New;font-size:10px;"
            "gridline-color:#2a2f3d;border:1px solid #2a2f3d;}"
            "QHeaderView::section{background-color:#151a26;color:#8090c0;"
            "padding:4px;border:1px solid #2a2f3d;font-weight:bold;}"
            "QTableWidget::item:alternate{background-color:#0d1119;}")
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
            lbl.setStyleSheet(
                "color:#60c0a0;font-size:10px;font-family:Courier New;")
            gl.addWidget(lbl, row, 1)
            setattr(self, attr, lbl)

        _row("Total events:",   "stat_total",     0)
        _row("Events/min:",     "stat_rate",      1)
        _row("Confidence mean:", "stat_conf_mean", 2)
        _row("Confidence range:", "stat_conf_range", 3)
        _row("GPS coverage:",   "stat_gps",       4)
        _row("CLS flagged:",    "stat_cls",       5)
        return grp

    def _zone_grp(self):
        grp = QGroupBox("Events by Zone / Class")
        gl = QGridLayout(grp)
        gl.setSpacing(4)
        gl.setContentsMargins(8, 8, 8, 8)

        gl.addWidget(_muted("By zone:"), 0, 0)
        self.lbl_by_zone = QLabel("—")
        self.lbl_by_zone.setWordWrap(True)
        self.lbl_by_zone.setStyleSheet(
            "color:#c0c4d0;font-size:10px;font-family:Courier New;")
        gl.addWidget(self.lbl_by_zone, 0, 1)

        gl.addWidget(_muted("By class:"), 1, 0)
        self.lbl_by_class = QLabel("—")
        self.lbl_by_class.setWordWrap(True)
        self.lbl_by_class.setStyleSheet(
            "color:#c0c4d0;font-size:10px;font-family:Courier New;")
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
            lbl.setStyleSheet(
                "color:#c0c4d0;font-size:10px;font-family:Courier New;")
            gl.addWidget(lbl, row, 1)
            setattr(self, attr, lbl)

        _row("Detection:",   "stat_armed",       0)
        _row("Mode:",        "stat_mode",        1)
        _row("Arduino:",     "stat_arduino",     2)
        _row("Pump:",        "stat_pump",        3)
        _row("Nozzles:",     "stat_nozzles",     4)
        _row("Husky link:",  "stat_husky",       5)
        _row("EStop:",       "stat_estop",       6)
        return grp

    def _map_grp(self):
        grp = QGroupBox("Spray Location Map (session, relative to start)")
        lay = QVBoxLayout(grp)
        lay.setContentsMargins(4, 4, 4, 4)
        self.lbl_map = QLabel()
        self.lbl_map.setAlignment(Qt.AlignCenter)
        self.lbl_map.setMinimumSize(300, 220)
        self.lbl_map.setStyleSheet(
            "border:1px solid #2a2f3d;background-color:#050810;"
            "border-radius:3px;")
        self.lbl_map.setScaledContents(True)
        lay.addWidget(self.lbl_map)
        self._redraw_map()  # draw an empty map immediately
        return grp

    # ── Session lifecycle ─────────────────────────────────────

    def _on_session_started(self, session_id):
        self._session_id    = session_id
        self._session_start = time.time()
        self._events        = []
        self._trail          = []
        self.tbl_events.setRowCount(0)
        self.lbl_session.setText(f"Session: {session_id}")
        self._update_stats_labels()
        self._redraw_map()
        self.log.log("ANALYSIS", f"New session: {session_id} — feed cleared", "info")

    def _on_clear(self):
        """Manual clear -- e.g. operator wants a clean feed mid-session
        without re-arming. Session stats/id are left alone; only the
        displayed feed and accumulated event list reset."""
        self._events = []
        self._trail   = []
        self.tbl_events.setRowCount(0)
        self._update_stats_labels()
        self._redraw_map()
        self.log.log("ANALYSIS", "Feed cleared manually", "info")

    # ── Spray event handling ──────────────────────────────────

    def _on_spray_event(self, event):
        self._events.append(event)

        # Track pose for the map trail (not just spray points)
        if event.pose:
            x, y = event.pose.get('x'), event.pose.get('y')
            if x is not None and y is not None:
                self._trail.append((x, y))
                if len(self._trail) > MAX_TRAIL_POINTS:
                    self._trail = self._trail[-MAX_TRAIL_POINTS:]

        self._insert_feed_row(event)
        self._update_stats_labels()
        self._redraw_map()

    def _insert_feed_row(self, event):
        names = [d.get('class_name', '?') for d in event.detections]
        conf  = max((d.get('confidence', 0.0) for d in event.detections),
                    default=0.0)
        ts = datetime.datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
        pose_str = "—"
        if event.pose:
            x, y = event.pose.get('x'), event.pose.get('y')
            if x is not None and y is not None:
                pose_str = f"{x:.2f}, {y:.2f}"
        gps_str = "—"
        if event.gps and event.gps.get('fix_valid'):
            lat, lon = event.gps.get('lat'), event.gps.get('lon')
            if lat is not None and lon is not None:
                gps_str = f"{lat:.5f}, {lon:.5f}"

        self.tbl_events.insertRow(0)  # newest first
        row = [
            ts, event.zone_name, f"N{event.nozzle_id + 1}",
            ", ".join(names) or "—", f"{conf:.2f}", pose_str, gps_str,
        ]
        for col, text in enumerate(row):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignCenter)
            if event.flagged_cls:
                item.setForeground(Qt.yellow)
            self.tbl_events.setItem(0, col, item)

        # Cap displayed rows — full data stays in self._events regardless
        while self.tbl_events.rowCount() > MAX_FEED_ROWS:
            self.tbl_events.removeRow(self.tbl_events.rowCount() - 1)

    # ── Stats ─────────────────────────────────────────────────

    def _update_stats_labels(self):
        n = len(self._events)
        self.stat_total.setText(str(n))

        if n == 0 or self._session_start is None:
            self.stat_rate.setText("—")
            self.stat_conf_mean.setText("—")
            self.stat_conf_range.setText("—")
            self.stat_gps.setText("—")
            self.stat_cls.setText("—")
            self.lbl_by_zone.setText("—")
            self.lbl_by_class.setText("—")
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

        cls_count = sum(1 for e in self._events if e.flagged_cls)
        self.stat_cls.setText(str(cls_count))

        zone_counts = {}
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
        # Duration ticker
        if self._session_start is not None:
            elapsed = int(time.time() - self._session_start)
            self.lbl_duration.setText(f"{elapsed // 60:02d}:{elapsed % 60:02d}")

        # Gantry / Arduino / pump / nozzles
        gctrl = getattr(self.gantry, 'ctrl', None)
        gstate = gctrl.state if gctrl is not None else None
        if gstate is not None and gstate.connected:
            self.stat_arduino.setText("Connected")
            self.stat_arduino.setStyleSheet(
                "color:#00c896;font-size:10px;font-family:Courier New;")
            self.stat_pump.setText("ON" if gstate.pump_on else "off")
            self.stat_pump.setStyleSheet(
                f"color:{'#00c896' if gstate.pump_on else '#8090c0'};"
                f"font-size:10px;font-family:Courier New;")
            noz = gstate.nozzles or [False, False, False]
            self.stat_nozzles.setText(
                "  ".join(f"N{i+1}:{'ON' if v else 'off'}"
                          for i, v in enumerate(noz)))
        else:
            self.stat_arduino.setText("Not connected")
            self.stat_arduino.setStyleSheet(
                "color:#f5a623;font-size:10px;font-family:Courier New;")
            self.stat_pump.setText("—")
            self.stat_nozzles.setText("—")

        # ActuationController (only present while armed)
        act = self.detect.get_actuation_status()
        if act is not None:
            self.stat_armed.setText("ARMED" if act['armed'] else "disarmed")
            self.stat_armed.setStyleSheet(
                f"color:{'#f5a623' if act['armed'] else '#8090c0'};"
                f"font-size:10px;font-family:Courier New;font-weight:bold;")
            self.stat_mode.setText(
                f"{act['mode'].upper()}"
                f"{'  [DRY RUN]' if act['dry_run'] else ''}")

            estop = act.get('estop_active') or act.get('manual_estop_active')
            self.stat_estop.setText("ACTIVE" if estop else "clear")
            self.stat_estop.setStyleSheet(
                f"color:{'#ff4040' if estop else '#00c896'};"
                f"font-size:10px;font-family:Courier New;font-weight:bold;")
        else:
            self.stat_armed.setText("disarmed")
            self.stat_armed.setStyleSheet(
                "color:#8090c0;font-size:10px;font-family:Courier New;")
            self.stat_mode.setText("—")
            self.stat_estop.setText("—")

        # Husky / ROSBridge (only present while armed)
        husky = self.detect.get_husky_status()
        if husky is not None:
            if husky['connected']:
                self.stat_husky.setText(
                    f"Connected (hb {husky['heartbeat_age_s']:.1f}s ago)")
                self.stat_husky.setStyleSheet(
                    "color:#00c896;font-size:10px;font-family:Courier New;")
            else:
                self.stat_husky.setText("No connection")
                self.stat_husky.setStyleSheet(
                    "color:#f5a623;font-size:10px;font-family:Courier New;")
        else:
            self.stat_husky.setText("—")
            self.stat_husky.setStyleSheet(
                "color:#8090c0;font-size:10px;font-family:Courier New;")

    # ── Spray location map ─────────────────────────────────────

    def _redraw_map(self):
        """
        Draw a simple 2D top-down scatter of spray locations relative
        to session start (0, 0), plus a light trail of recent poses.
        Reuses the same OpenCV-canvas-to-QLabel technique as the
        histogram in the original version of this tab -- no new
        dependencies.
        """
        H, W = 220, 460
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        canvas[:] = (8, 10, 18)

        cx, cy = W // 2, H // 2
        # Grid
        for x in range(0, W, 40):
            cv2.line(canvas, (x, 0), (x, H), (25, 28, 40), 1)
        for y in range(0, H, 40):
            cv2.line(canvas, (0, y), (W, y), (25, 28, 40), 1)
        cv2.line(canvas, (cx, 0), (cx, H), (40, 45, 60), 1)
        cv2.line(canvas, (0, cy), (W, cy), (40, 45, 60), 1)

        pts = [e.pose for e in self._events if e.pose]
        pts = [(p.get('x'), p.get('y')) for p in pts
               if p.get('x') is not None and p.get('y') is not None]

        if not pts and not self._trail:
            cv2.putText(canvas, "No GPS/pose data yet this session",
                        (20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (120, 120, 140), 1, cv2.LINE_AA)
            self._show(self.lbl_map, cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            return

        all_x = [p[0] for p in pts] + [p[0] for p in self._trail]
        all_y = [p[1] for p in pts] + [p[1] for p in self._trail]
        span = max(max(abs(v) for v in all_x + all_y), 0.5) * 1.2
        px_per_m = min(W, H) / 2 / span

        def to_px(x, y):
            return (int(cx + x * px_per_m), int(cy - y * px_per_m))

        # Trail (robot path) — faint line
        if len(self._trail) > 1:
            trail_pts = [to_px(x, y) for x, y in self._trail]
            for i in range(len(trail_pts) - 1):
                cv2.line(canvas, trail_pts[i], trail_pts[i + 1],
                          (60, 60, 90), 1, cv2.LINE_AA)

        # Session start marker
        cv2.drawMarker(canvas, (cx, cy), (100, 200, 255),
                        cv2.MARKER_CROSS, 10, 1)

        # Spray points — color by mode, ring for CLS-flagged
        for e in self._events:
            if not e.pose:
                continue
            x, y = e.pose.get('x'), e.pose.get('y')
            if x is None or y is None:
                continue
            px, py = to_px(x, y)
            color = (80, 200, 255) if e.mode == 'weed' else (255, 160, 60)
            cv2.circle(canvas, (px, py), 4, color, -1, cv2.LINE_AA)
            if e.flagged_cls:
                cv2.circle(canvas, (px, py), 7, (0, 220, 220), 1, cv2.LINE_AA)

        cv2.putText(canvas, f"{len(pts)} located spray(s)  |  scale: "
                              f"{span:.1f}m half-width",
                    (8, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (120, 120, 140), 1, cv2.LINE_AA)

        self._show(self.lbl_map, cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

    # ── Shared image renderer ─────────────────────────────────

    def _show(self, lbl, arr):
        try:
            if not lbl.isVisible():
                return
            h, w = arr.shape[:2]
            arr  = np.ascontiguousarray(arr)
            q    = QImage(arr.tobytes(), w, h, w * 3, QImage.Format_RGB888)
            lbl.setPixmap(
                QPixmap.fromImage(q).scaled(
                    lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception:
            pass

    # ── Cleanup ───────────────────────────────────────────────

    def cleanup(self):
        self._status_timer.stop()
