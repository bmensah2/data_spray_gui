"""
gui/spray_event_table.py
ABEN Field Imaging System — Shared Live Spray Event Feed table

Builds the exact "Live Spray Event Feed" QTableWidget (Time, Zone,
Nozzle, Class, Conf, Pose, GPS columns) originally built inline in
gui/tabs/tab_analysis_rgb.py, extracted here so both Session Analysis
and the camera fullscreen popout (gui/panels/dual_camera_panel.py)
build and update an identical table from one place instead of two
copies quietly drifting apart over time.
"""

import datetime

from PyQt5.QtWidgets import (
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

from gui.theme_manager import theme_manager

# Cap how many rows a live feed table keeps on screen. Callers that
# also keep a full, uncapped list of events for stats/export purposes
# (like AnalysisTabRGB) are unaffected -- this only caps the table
# widget's displayed rows, for UI responsiveness over a long session.
MAX_FEED_ROWS = 300

COLUMNS = ["Time", "Zone", "Nozzle", "Class", "Conf", "Pose (x,y)", "GPS"]


def build_spray_event_table() -> QTableWidget:
    """Construct the themed, empty Live Spray Event Feed table."""
    table = QTableWidget(0, len(COLUMNS))
    table.setHorizontalHeaderLabels(COLUMNS)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setAlternatingRowColors(True)
    theme_manager.register_widget(
        table, lambda p: (
            f"QTableWidget{{background-color:{p['bg0']};color:{p['text_dim']};"
            f"font-family:'Noto Sans',Arial,sans-serif;font-size:10px;"
            f"gridline-color:{p['border2']};border:1px solid {p['border2']};}}"
            f"QHeaderView::section{{background-color:{p['bg3']};"
            f"color:{p['muted']};padding:4px;border:1px solid {p['border2']};"
            f"font-weight:bold;}}"
            f"QTableWidget::item:alternate{{background-color:{p['bg2']};}}"
            f"QTableWidget::item:selected{{background-color:{p['btn_bg']};}}"
        ))
    return table


def insert_spray_event_row(table: QTableWidget, event,
                            max_rows: int = MAX_FEED_ROWS):
    """
    Insert one SprayEvent as the newest (top) row of `table`, and trim
    to `max_rows` displayed rows. Identical formatting to
    AnalysisTabRGB's original _insert_feed_row().
    """
    names = [d.get('class_name', '?') for d in event.detections]
    conf  = max(
        (d.get('confidence', 0.0) for d in event.detections), default=0.0)
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

    table.insertRow(0)  # newest first
    row = [
        ts, event.zone_name, f"N{event.nozzle_id + 1}",
        ", ".join(names) or "—", f"{conf:.2f}", pose_str, gps_str,
    ]
    flag_color = QColor(theme_manager.palette()['amber'])
    for col, text in enumerate(row):
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignCenter)
        if event.flagged_cls:
            item.setForeground(flag_color)
        table.setItem(0, col, item)

    while table.rowCount() > max_rows:
        table.removeRow(table.rowCount() - 1)
