"""
gui/style.py
Field Imaging System — Styling, Colors, Shared Widgets

All styling constants, button helpers, and small reusable
widgets live here. Import from any panel or tab.

Usage (unchanged from earlier versions):
    from gui.style import STYLE, BTN_GREEN, LED, _btn, _muted, _divider, _sec

THEMING
-------
This module is now backed by gui.theme_manager.

"""

import weakref

from PyQt5.QtWidgets import QLabel, QFrame, QScrollArea, QWidget
from PyQt5.QtCore import Qt

from gui.theme_manager import theme_manager, _lighten, _darken, _btn


# ─────────────────────────────────────────────────────────────
#  THEME-DRIVEN MODULE CONSTANTS
# ─────────────────────────────────────────────────────────────
def refresh():
    """
    Refresh every module-level constant from theme_manager's current
    theme. Called automatically by theme_manager.apply() — you do not
    normally need to call this yourself.
    """
    g = globals()
    p = theme_manager.palette()

    g["C_BG"]     = p["bg"]
    g["C_BG2"]    = p["bg2"]
    g["C_BG3"]    = p["bg3"]
    g["C_BORDER"] = p["border"]
    g["C_TEXT"]   = p["text"]
    g["C_MUTED"]  = p["muted"]
    g["C_GREEN"]  = p["green"]
    g["C_BLUE"]   = p["blue"]
    g["C_AMBER"]  = p["amber"]
    g["C_RED"]    = p["red"]
    g["C_TEAL"]   = p["teal"]
    g["C_DIM"]    = p["dim"]

    g["BTN_BLUE"]      = theme_manager.button_style("blue")
    g["BTN_GREEN"]     = theme_manager.button_style("green")
    g["BTN_RED"]       = theme_manager.button_style("red")
    g["BTN_AMBER"]     = theme_manager.button_style("amber")
    g["BTN_TEAL"]      = theme_manager.button_style("teal")
    g["BTN_CONNECT"]   = theme_manager.button_style("connect")
    g["BTN_ESTOP"]     = theme_manager.button_style("estop")
    g["BTN_DIM_GREEN"] = theme_manager.button_style("dim_green")
    g["BTN_DIM_RED"]   = theme_manager.button_style("dim_red")
    g["BTN_DIM_BLUE"]  = theme_manager.button_style("dim_blue")
    g["BTN_DIM_AMBER"] = theme_manager.button_style("dim_amber")

    g["STYLE"] = theme_manager.style()


# Populate the constants above for the default ("field_dark") theme at
# import time, so `from gui.style import BTN_GREEN, STYLE, C_TEXT, ...`
# works exactly as before.
refresh()


# ─────────────────────────────────────────────────────────────
#  SHARED UTILITY WIDGETS
# ─────────────────────────────────────────────────────────────
class LED(QLabel):
    """
    Small circular LED indicator widget.

    Theme-aware (preferred):
        led.set_state(connected, role="green")
        led.set_state(armed, role="amber")
        # roles: "green","blue","amber","purple","red","yellow"

    Legacy explicit-color override (sticky; does NOT re-theme):
        led.set_state(connected, color_on="#00c896")

    `color_on`/`color_off`, once given, override the role for that
    LED until cleared by passing role-only calls again.
    """

    _instances = weakref.WeakSet()

    def __init__(self, size: int = 10, parent=None):
        super().__init__(parent)
        self._size = size
        self._on = False
        self._role = "green"
        self._color_on = None
        self._color_off = None
        self.setFixedSize(size, size)
        LED._instances.add(self)
        self._paint()

    def set_state(self, on: bool,
                   role: str = None,
                   color_on: str = None,
                   color_off: str = None):
        self._on = bool(on)
        if role is not None:
            self._role = role
        if color_on is not None:
            self._color_on = color_on
        if color_off is not None:
            self._color_off = color_off
        self._paint()

    def _paint(self):
        leds = theme_manager.led_colors()
        c_on = self._color_on or leds.get(self._role, leds["green"])
        c_off = self._color_off or leds["off"]
        c = c_on if self._on else c_off
        border = _lighten(c_on, 35) if self._on else leds["off_border"]
        r = self._size // 2
        self.setStyleSheet(
            f"background-color:{c};"
            f"border-radius:{r}px;"
            f"border:1px solid {border};"
        )

    def refresh(self):
        """Called by theme_manager.apply() on every theme change."""
        self._paint()


# ─────────────────────────────────────────────────────────────
#  LAYOUT HELPERS  (theme-aware: re-painted on theme change)
# ─────────────────────────────────────────────────────────────
def _divider() -> QFrame:
    """Horizontal 1px divider line."""
    f = QFrame()
    f.setObjectName("divider")
    f.setFrameShape(QFrame.HLine)
    f.setFixedHeight(1)

    def _style(p):
        return f"background-color:{p['border2']};"

    theme_manager.register_widget(f, _style)
    return f


def _muted(text: str, size: int = 10) -> QLabel:
    """Muted grey label for secondary information."""
    l = QLabel(text)

    def _style(p):
        return (
            f"color:{p['muted']};font-size:{size}px;"
            f"font-family:'Noto Sans',Arial,sans-serif;background:transparent;"
        )

    theme_manager.register_widget(l, _style)
    return l


def _sec(text: str, size: int = 11) -> QLabel:
    """Section header label in teal/accent color."""
    l = QLabel(text)

    def _style(p):
        if theme_manager.current == "field_dark":
            color = "#00a0c0"
        else:
            color = _lighten(p["teal"], 25)
        return (
            f"color:{color};font-size:{size}px;font-weight:bold;"
            f"font-family:'Noto Sans',Arial,sans-serif;letter-spacing:1px;padding:1px 0;"
        )

    theme_manager.register_widget(l, _style)
    return l


def _scroll(widget: QWidget) -> QScrollArea:
    """Wrap widget in a vertical scroll area."""
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    sa.setStyleSheet("QScrollArea{border:none;background:transparent;}")
    sa.setWidget(widget)
    return sa