#!/usr/bin/env python3
"""
offline_review_gui.py
ABEN Dual RGB Imaging System — Standalone Offline Review

Launches the Offline Review tab (recorded video/image inference
review + model evaluation launcher) as its OWN lightweight
application, separate from the main Imaging and Spraying Dashboard
(main_gui_rgb.py). Offline review is fundamentally a post-processing/
analysis tool -- it never needs the camera/gantry/Arduino hardware
connections the main dashboard requires, so bundling it into that
dashboard meant launching the whole thing (with its hardware-connect
warnings, Husky SSH state, etc.) just to review old footage. This
runs standalone: no hardware required, works on a machine without the
robot connected at all, needs only the recorded videos/images and a
model file.

Usage:
    python3 offline_review_gui.py

Shares theme_manager's saved theme preference with the main
dashboard (same on-disk config), so switching themes in either app is
picked up by the other on next launch.
"""

import sys

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QAction, QActionGroup, QMessageBox,
)

from gui.theme_manager import theme_manager
from gui.shared_log import UnifiedLog
from gui.tabs.tab_offline_review import OfflineReviewTab


class OfflineReviewWindow(QMainWindow):
    """
    Thin QMainWindow wrapping OfflineReviewTab as the sole central
    widget -- no other tabs, no hardware panels, no cross-tab state.
    detect_ref is None here (there is no live Detection tab to pull
    an armed model from in a standalone app); the tab's own "Use
    Armed Model" button already handles that gracefully with a clear
    log warning rather than crashing.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ABEN — Offline Video/Image Review")
        self.resize(1500, 950)

        self.log = UnifiedLog()
        self.tab = OfflineReviewTab(self.log, detect_ref=None)
        self.setCentralWidget(self.tab)

        theme_manager.register_widget(
            self, lambda p: f"background-color:{p['bg']};")

        self._build_menu()

    def _build_menu(self):
        mb = self.menuBar()

        fm = mb.addMenu("File")
        q = QAction("Quit", self)
        q.setShortcut("Ctrl+Q")
        q.triggered.connect(self.close)
        fm.addAction(q)

        vm = mb.addMenu("View")
        theme_menu = vm.addMenu("Theme")
        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)
        for key, label in theme_manager.list_themes():
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(key == theme_manager.current)
            action.triggered.connect(
                lambda checked, k=key: self._on_theme_selected(k))
            theme_group.addAction(action)
            theme_menu.addAction(action)

        hm = mb.addMenu("Help")
        ab = QAction("About", self)
        ab.triggered.connect(self._about)
        hm.addAction(ab)

    def _on_theme_selected(self, theme_key: str):
        theme_manager.apply(theme_key, app=QApplication.instance())
        self.log.log("SYS", f"Theme changed: {theme_key}", "info")

    def _about(self):
        QMessageBox.information(
            self, "About",
            "ABEN Offline Video/Image Review  v1.0\n\n"
            "Standalone review tool -- no camera/gantry/Arduino "
            "hardware required.\n\n"
            "Load recorded dual Left/Right field video (or paired "
            "image folders), run it through the same detection model "
            "used live, review results frame by frame, export an "
            "annotated video, and (with labeled data) launch model "
            "evaluation.\n\n"
            "Author: Nana | NDSU PhD Imaging System")

    def closeEvent(self, event):
        self.tab.cleanup()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # Load the operator's saved theme choice (shared with the main
    # dashboard's on-disk config) and apply the app-level QSS BEFORE
    # any window/widget is constructed -- same startup order
    # main_gui_rgb.py uses.
    theme_manager.load()
    theme_manager.apply(theme_manager.current, app=app, save=False)
    win = OfflineReviewWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
