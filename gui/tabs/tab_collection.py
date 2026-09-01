"""
gui/tabs/tab_collection.py
ABEN Field Imaging System — Tab 1: Data Collection

Layout:
  Left   — System Panel (Gantry tab | Data Collection tab)
  Middle — Camera Feed (shared)
  Right  — Navigation Panel (shared)
  Bottom — System Log
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTabWidget, QLabel
)
from PyQt5.QtCore import Qt

from gui.style import STYLE, _divider, _muted
from gui.shared_log import UnifiedLog, LogPanel
from gui.panels.gantry_panel import GantryPanel
from gui.panels.dual_camera_panel import DualCameraPanel
from gui.panels.acquisition_panel_rgb import AcquisitionPanelRGB
from gui.panels.navigation_panel_rgb import NavigationPanelRGB as NavigationPanel


class CollectionTab(QWidget):
    """
    Tab 1 — Data Collection.

    Shared objects passed in from MainWindow:
      camera : CameraPanel     (shared with Tab 2)
      nav    : NavigationPanel (shared with Tab 2)
      gantry : GantryPanel     (shared with Tab 2 — SAME controller)
    """

    def __init__(self, camera: DualCameraPanel,
                 nav: NavigationPanel,
                 gantry: GantryPanel,
                 acq=None,
                 parent=None):
        super().__init__(parent)
        self.camera = camera
        self.nav    = nav
        self.gantry = gantry   # shared — same Arduino connection

        # Tab-local log
        self.log = UnifiedLog()
        # Point gantry log to this tab's log
        self.gantry.shared_log = self.log

        # Use shared acq if provided (from MainWindow) — guarantees
        # identical camera settings between Data Collection and Detection.
        if acq is not None:
            self.acq = acq
            self.acq.shared_log = self.log
            self._owns_acq = False
        else:
            self.acq = AcquisitionPanelRGB(self.log, camera)
            self._owns_acq = True

        # Enable camera controls after short delay
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(
            1500,
            lambda: self.acq.enable_camera_controls(True))

        self._build_ui()
        self.log.log("SYS", "Data Collection tab ready", "ok")

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── 3-panel splitter ──────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(3)
        splitter.setStyleSheet(
            "QSplitter::handle{background-color:#2a2f3d;}")

        # ── LEFT: System panel (Gantry + Data Collection tabs) ──
        left_tabs = QTabWidget()

        # Gantry tab
        left_tabs.addTab(self.gantry, "⚙ Gantry")

        # Data Collection tab
        left_tabs.addTab(self.acq, "📷 Data Collection")

        left_w = QWidget()
        left_w.setMinimumWidth(400)
        left_w.setMaximumWidth(500)
        llay = QVBoxLayout(left_w)
        llay.setContentsMargins(0, 0, 0, 0)
        llay.addWidget(left_tabs)
        splitter.addWidget(left_w)

        # ── MIDDLE: Camera feed ───────────────────────────────
        mid_w = QWidget()
        mid_w.setMinimumWidth(400)
        mlay = QVBoxLayout(mid_w)
        mlay.setContentsMargins(0, 0, 0, 0)
        mlay.addWidget(self.camera.camera_control_bar())
        mlay.addWidget(self.camera.display_widget())
        splitter.addWidget(mid_w)

        # ── RIGHT: Navigation ─────────────────────────────────
        right_w = QWidget()
        right_w.setMinimumWidth(380)
        right_w.setMaximumWidth(460)
        rlay = QVBoxLayout(right_w)
        rlay.setContentsMargins(0, 0, 0, 0)
        rlay.addWidget(self.nav)
        splitter.addWidget(right_w)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        root.addWidget(splitter, stretch=1)

        # ── BOTTOM: Log ───────────────────────────────────────
        root.addWidget(_divider())
        root.addWidget(
            LogPanel(self.log,
                     sources=["GANTRY","CAMERA","NAV","SYS"],
                     height=110))

    def cleanup(self):
        # gantry is shared — cleaned up by MainWindow
        # acq cleaned up by MainWindow when shared
        if getattr(self, "_owns_acq", True):
            self.acq.cleanup()