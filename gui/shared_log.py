"""
gui/shared_log.py
Field Imaging System — Unified Log Widget

Each tab gets its own UnifiedLog instance.
Sources: GANTRY, CAMERA, DETECT, NAV, SYS, ANALYSIS

Usage:
    from gui.shared_log import UnifiedLog, LogPanel

    # In a tab:
    self.log = UnifiedLog()
    log_panel = LogPanel(self.log)   # styled container with header
"""

import time

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QPlainTextEdit
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QTextCharFormat, QTextCursor

from gui.theme_manager import theme_manager


# ─────────────────────────────────────────────────────────────
#  UNIFIED LOG
# ─────────────────────────────────────────────────────────────
class UnifiedLog(QPlainTextEdit):
    """
    Thread-safe log widget. Accepts messages from any thread
    via append_signal → _do_append (runs in GUI thread).

    Sources and their palette roles:
        GANTRY   → green
        CAMERA   → blue
        DETECT   → amber
        NAV      → purple
        SYS      → dim
        ANALYSIS → teal
    """

    append_signal = pyqtSignal(str, str, str)

    # Palette role keys (not hex) -- resolved fresh against the active
    # theme in _do_append(), same pattern as _SRC_ROLE below.
    _TAG_ROLE = {
        "send":  "blue",
        "recv":  "muted",
        "ok":    "green",
        "error": "red",
        "warn":  "amber",
        "info":  "dim",
    }
    _PFX = {
        "send":  "►",
        "recv":  "◄",
        "ok":    "✓",
        "error": "✗",
        "warn":  "⚠",
        "info":  "·",
    }

    _SRC_ROLE = {
        "GANTRY":   "green",
        "CAMERA":   "blue",
        "DETECT":   "amber",
        "NAV":      "purple",
        "SYS":      "dim",
        "ANALYSIS": "teal",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(500)
        self.append_signal.connect(self._do_append)

    @staticmethod
    def source_color(source: str) -> str:
        """Theme-aware color for a log source name."""
        p = theme_manager.palette()
        role = UnifiedLog._SRC_ROLE.get(source)
        if role is None:
            return p["muted"]
        return p.get(role, p["muted"])

    def log(self, source: str, msg: str, tag: str = "info"):
        """
        Thread-safe log entry.
        source: GANTRY | CAMERA | DETECT | NAV | SYS | ANALYSIS
        tag:    ok | error | warn | info | send | recv
        """
        self.append_signal.emit(source, msg, tag)

    def _do_append(self, source: str, msg: str, tag: str):
        p     = theme_manager.palette()
        src_c = self.source_color(source)
        msg_c = p.get(self._TAG_ROLE.get(tag), p["muted"])
        pfx   = self._PFX.get(tag,    "·")
        ts    = time.strftime("%H:%M:%S")

        cur = self.textCursor()
        cur.movePosition(QTextCursor.End)

        def _w(text, color):
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(color))
            cur.setCharFormat(fmt)
            cur.insertText(text)

        _w(f"[{ts}] ", p["dim"])
        _w(f"[{source}] ", src_c)
        _w(f"{pfx} {msg}\n", msg_c)

        self.setTextCursor(cur)
        self.ensureCursorVisible()

        # Trim if over block limit
        doc = self.document()
        while doc.blockCount() > 500:
            c2 = QTextCursor(doc.begin())
            c2.select(QTextCursor.BlockUnderCursor)
            c2.movePosition(
                QTextCursor.NextCharacter, QTextCursor.KeepAnchor
            )
            c2.removeSelectedText()


# ─────────────────────────────────────────────────────────────
#  LOG PANEL  —  styled container used at bottom of each tab
# ─────────────────────────────────────────────────────────────
class LogPanel(QWidget):
    """
    Styled log container with source legend and CLEAR button.
    Fixed height — sits at the bottom of each tab.

    Usage:
        self.log = UnifiedLog()
        log_panel = LogPanel(self.log, sources=["GANTRY","CAMERA","SYS"])
        tab_layout.addWidget(log_panel)
    """

    def __init__(self, log: UnifiedLog,
                 sources: list = None,
                 height:  int  = 110,
                 parent=None):
        super().__init__(parent)
        self.setFixedHeight(height)
        self._log = log

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)

        # ── Header row ────────────────────────────────────────
        hdr = QHBoxLayout()
        title = QLabel("SYSTEM LOG")
        theme_manager.register_widget(
            title, lambda p: (
                f"color:{p['muted2']};font-size:10px;font-weight:bold;"
                f"font-family:'Noto Sans',Arial,sans-serif;letter-spacing:2px;"))
        hdr.addWidget(title)

        # Source legend dots
        shown = sources or list(UnifiedLog._SRC_ROLE.keys())
        for src in shown:
            role = UnifiedLog._SRC_ROLE.get(src)
            dot = QLabel("●")
            theme_manager.register_widget(
                dot, lambda p, role=role: (
                    f"color:{p.get(role, p['muted'])};"
                    f"font-size:11px;margin-left:6px;"))
            hdr.addWidget(dot)
            lbl = QLabel(src)
            theme_manager.register_widget(
                lbl, lambda p, role=role: (
                    f"color:{p.get(role, p['muted'])};"
                    f"font-size:9px;font-family:'Noto Sans',Arial,sans-serif;"))
            hdr.addWidget(lbl)

        hdr.addStretch()
        clr = QPushButton("CLEAR")
        theme_manager.register_button(clr, "amber")
        clr.setFixedWidth(65)
        clr.clicked.connect(log.clear)
        hdr.addWidget(clr)
        lay.addLayout(hdr)

        # ── Log widget ────────────────────────────────────────
        lay.addWidget(log)