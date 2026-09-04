"""
gui/theme_manager.py
Field Imaging System — Theme Manager

Provides a registry-based theming engine on top of the existing
gui/style.py look-and-feel. Seven themes are available:

    field_dark           — the original field-tested dark theme (DEFAULT)
    dark_professional   — enhanced dark professional
    dark_blue           — sophisticated dark blue
    midnight            — deep midnight / code-editor style
    forest              — green, vegetation-friendly
    scientific          — bold pink/navy research theme
    charcoal            — elegant charcoal with orange accents

ARCHITECTURE
------------
Each theme is described by a small "palette" dict (~24 colors). From a
palette we can generate:

    build_style(theme_key)        -> full QMainWindow/QWidget QSS string
                                      (the global "chrome")
    build_buttons(theme_key)      -> dict[role] -> QPushButton QSS string
                                      roles: blue, green, red, amber, teal,
                                      connect, estop, dim_green, dim_red,
                                      dim_blue, dim_amber
    build_led_colors(theme_key)   -> dict[role] -> hex color
                                      roles: green, blue, amber, purple,
                                      red, off, off_border

For "field_dark" the generated STYLE / button styles / LED colors are
byte-identical (or visually identical) to the original field-tested
gui/style.py — selecting it does not change anything.

USAGE
-----
    from gui.theme_manager import theme_manager

    # at startup (after QApplication is created):
    theme_manager.load()
    theme_manager.apply(theme_manager.current, app=app)

    # register a themed button (applies style immediately + tracks it
    # so future theme switches restyle it too):
    theme_manager.register_button(my_button, "green")

    # LEDs self-register — just create them and call set_state(on, role=...)

    # switch theme (e.g. from a menu action):
    theme_manager.apply("forest", app=QApplication.instance())
"""

import json
import logging
import weakref
from pathlib import Path

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  COLOR MATH HELPERS
# ─────────────────────────────────────────────────────────────
def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(rgb) -> str:
    r, g, b = (max(0, min(255, round(c))) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _lighten(h: str, amount: int = 25) -> str:
    r, g, b = _hex_to_rgb(h)
    return _rgb_to_hex((r + amount, g + amount, b + amount))


def _darken(h: str, amount: int = 25) -> str:
    r, g, b = _hex_to_rgb(h)
    return _rgb_to_hex((r - amount, g - amount, b - amount))


def _blend(h1: str, h2: str, t: float = 0.5) -> str:
    r1, g1, b1 = _hex_to_rgb(h1)
    r2, g2, b2 = _hex_to_rgb(h2)
    return _rgb_to_hex((r1 + (r2 - r1) * t,
                         g1 + (g2 - g1) * t,
                         b1 + (b2 - b1) * t))


def _dim_pair(c: str):
    """Generate (dim_bg, dim_fg) for 'inactive/armed-but-off' button states."""
    bg = _darken(_darken(_darken(c)))
    fg = _darken(c)
    return bg, fg


def _btn(bg: str, fg: str = "#ffffff",
         hover: str = None, pressed: str = None) -> str:
    """Generate a complete QPushButton stylesheet (same shape as the
    original gui/style.py._btn)."""
    hv = hover or _lighten(bg)
    pr = pressed or _darken(bg)
    return (
        f"QPushButton{{background-color:{bg};color:{fg};border:none;"
        f"border-radius:4px;padding:4px 9px;"
        f"font-family:'Noto Sans',Arial,sans-serif;font-size:10px;"
        f"font-weight:bold;min-height:24px;}}"
        f"QPushButton:hover{{background-color:{hv};}}"
        f"QPushButton:pressed{{background-color:{pr};}}"
        f"QPushButton:disabled{{background-color:#1a2030;color:#4a5068;}}"
    )


# ─────────────────────────────────────────────────────────────
#  PALETTES
# ─────────────────────────────────────────────────────────────
# Palette keys used by every theme:
#   bg, bg0, bg2, bg3          backgrounds (darkest -> groupbox/surface)
#   border, border2           borders (primary, secondary/divider)
#   text, text_dim, muted,
#   muted2, dim                text tones
#   btn_bg, btn_hover,
#   btn_pressed                generic/default QPushButton
#   input_bg, disabled_text,
#   disabled_bg                inputs / disabled state
#   green, blue, amber, red,
#   teal, purple, connect      semantic accent colors

# ── Field Dark (DEFAULT) — exact values from the original gui/style.py ──
_FIELD_DARK = dict(
    bg="#0f1117", bg0="#0a0d14", bg2="#131720", bg3="#181c27",
    border="#3a4055", border2="#2a2f3d",
    text="#e8eaf0", text_dim="#b0b8c8", muted="#8090a8",
    muted2="#a0a8b8", dim="#6b7280",
    btn_bg="#2e3d55", btn_hover="#3a4e6a", btn_pressed="#1e2d45",
    input_bg="#1a1e2e", disabled_text="#4a5a70", disabled_bg="#1a2030",
    green="#00c896", blue="#4a9eff", amber="#f5a623", red="#e84545",
    teal="#1a6888", purple="#b060d0", connect="#00a070",
)

# ── Base colors for the 6 new themes (from styles_new.py StyleManager) ──
# (bg, surface, text, muted, primary, success, warning, danger)
_NEW_THEME_BASES = {
    "dark_professional": dict(
        bg="#2c3e50", surface="#34495e", text="#ecf0f1", muted="#bdc3c7",
        primary="#3498db", success="#27ae60", warning="#f39c12", danger="#e74c3c",
    ),
    "dark_blue": dict(
        bg="#1e2a3a", surface="#243447", text="#e8eaed", muted="#90a4ae",
        primary="#1976d2", success="#388e3c", warning="#f57c00", danger="#d32f2f",
    ),
    "midnight": dict(
        bg="#0d1117", surface="#161b22", text="#f0f6fc", muted="#8b949e",
        primary="#58a6ff", success="#238636", warning="#d29922", danger="#da3633",
    ),
    "forest": dict(
        bg="#1b2f1b", surface="#243824", text="#e8f5e8", muted="#a5d6a7",
        primary="#81c784", success="#4caf50", warning="#ff9800", danger="#d84315",
    ),
    "scientific": dict(
        bg="#1a1a2e", surface="#0f3460", text="#eeeeee", muted="#a0a0a0",
        primary="#e94560", success="#3282b8", warning="#ff6b35", danger="#e94560",
    ),
    "charcoal": dict(
        bg="#2b2b2b", surface="#363636", text="#f5f5f5", muted="#cccccc",
        primary="#ff6b35", success="#4caf50", warning="#ff9800", danger="#f44336",
    ),
}


def _derive_palette(base: dict) -> dict:
    """Derive a full 24-key palette from a theme's base colors."""
    bg, surface = base["bg"], base["surface"]
    text, muted = base["text"], base["muted"]
    primary, success = base["primary"], base["success"]
    warning, danger = base["warning"], base["danger"]

    bg2 = _blend(bg, surface, 0.5)
    bg0 = _darken(_darken(bg))
    bg3 = surface
    border = _lighten(_lighten(surface))
    border2 = _darken(border)
    text_dim = _blend(text, muted, 0.5)
    muted2 = _lighten(muted)
    dim = _blend(muted, bg2, 0.5)
    btn_bg = _lighten(surface)
    btn_hover = _lighten(btn_bg)
    btn_pressed = _darken(btn_bg)
    input_bg = _darken(bg2)
    disabled_bg = _darken(input_bg)
    disabled_text = _blend(muted, bg, 0.5)
    teal = _darken(_darken(primary))
    connect = _lighten(success)

    return dict(
        bg=bg, bg0=bg0, bg2=bg2, bg3=bg3,
        border=border, border2=border2,
        text=text, text_dim=text_dim, muted=muted, muted2=muted2, dim=dim,
        btn_bg=btn_bg, btn_hover=btn_hover, btn_pressed=btn_pressed,
        input_bg=input_bg, disabled_text=disabled_text, disabled_bg=disabled_bg,
        green=success, blue=primary, amber=warning, red=danger,
        teal=teal, purple="#b060d0", connect=connect,
    )


# ── Build the final PALETTES dict (lazy, computed once at import) ──
PALETTES = {
    "field_dark": dict(_FIELD_DARK),
    "aben_dark": dict(_FIELD_DARK),
}
for _name, _base in _NEW_THEME_BASES.items():
    PALETTES[_name] = _derive_palette(_base)


# Display names + ordering for the View > Theme menu
THEME_LABELS = {
    "aben_dark":         "ABEN Dark (Default)",
    "dark_professional": "Dark Professional",
    "dark_blue":         "Dark Blue",
    "midnight":          "Midnight",
    "forest":            "Forest",
    "scientific":        "Scientific",
    "charcoal":          "Charcoal",
}
THEME_ORDER = list(THEME_LABELS.keys())

DEFAULT_THEME = "aben_dark"


# ─────────────────────────────────────────────────────────────
#  BUTTON ROLE GENERATION
# ─────────────────────────────────────────────────────────────
# Roles map 1:1 onto the original BTN_* constants:
#   blue       -> BTN_BLUE   (neutral / general action)
#   green      -> BTN_GREEN  (positive action e.g. start/save)
#   red        -> BTN_RED    (stop / negative action)
#   amber      -> BTN_AMBER  (warning / caution action)
#   teal       -> BTN_TEAL   (informational action)
#   connect    -> BTN_CONNECT (bright green — connect/online)
#   estop      -> BTN_ESTOP  (bright red — emergency stop)
#   dim_green/red/blue/amber -> BTN_DIM_* (inactive status indicators)

_ABEN_DARK_BUTTONS = {
    "blue":      _btn("#525558"),
    "green":     _btn("#007a50"),
    "red":       _btn("#971c1c"),
    "amber":     _btn("#a06000"),
    "teal":      _btn("#1a6888"),
    "connect":   _btn("#00a070"),
    "estop":     _btn("#cc1515"),
    "dim_green": _btn("#1a3828", "#3a8860"),
    "dim_red":   _btn("#2a1515", "#884040"),
    "dim_blue":  _btn("#111d30", "#3a6090"),
    "dim_amber": _btn("#2a1e00", "#907040"),
}


def build_buttons(theme_key: str) -> dict:
    """Return dict[role] -> QPushButton QSS string for a theme."""
    if theme_key == "aben_dark":
        return dict(_ABEN_DARK_BUTTONS)

    p = PALETTES[theme_key]
    return {
        "blue":      _btn(p["btn_bg"]),
        "green":     _btn(_darken(p["green"])),
        "red":       _btn(_darken(_darken(p["red"]))),
        "amber":     _btn(_darken(p["amber"])),
        "teal":      _btn(p["teal"]),
        "connect":   _btn(p["connect"]),
        "estop":     _btn(p["red"]),
        "dim_green": _btn(*_dim_pair(p["green"])),
        "dim_red":   _btn(*_dim_pair(p["red"])),
        "dim_blue":  _btn(*_dim_pair(p["blue"])),
        "dim_amber": _btn(*_dim_pair(p["amber"])),
    }


BUTTON_STYLES = {key: build_buttons(key) for key in THEME_ORDER}


# ─────────────────────────────────────────────────────────────
#  LED COLOR GENERATION
# ─────────────────────────────────────────────────────────────
# Roles: green (gantry), blue (camera), amber (detect), purple (nav),
#        red (error/general), yellow (stepper/limit status),
#        off / off_border (inactive LED)

_ABEN_DARK_LEDS = {
    "green": "#00c896", "blue": "#4a9eff", "amber": "#f5a623",
    "purple": "#b060d0", "red": "#e84545", "yellow": "#ffe066",
    "off": "#2a2f3d", "off_border": "#1a1f2d",
}


def build_led_colors(theme_key: str) -> dict:
    if theme_key == "aben_dark":
        return dict(_ABEN_DARK_LEDS)
    p = PALETTES[theme_key]
    return {
        "green": p["green"], "blue": p["blue"], "amber": p["amber"],
        "purple": p["purple"], "red": p["red"],
        "yellow": _lighten(p["amber"], 40),
        "off": p["border2"], "off_border": _darken(p["border2"]),
    }


LED_COLORS = {key: build_led_colors(key) for key in THEME_ORDER}


# ─────────────────────────────────────────────────────────────
#  GLOBAL QSS ("chrome") GENERATION
# ─────────────────────────────────────────────────────────────
# The original gui/style.py STYLE string, parameterized by palette.
# For "aben_dark" this reproduces the original almost exactly (a couple
# of derived shades — slider sub-pages, focus backgrounds — are computed
# rather than hand-copied, but are visually equivalent).

_STYLE_TEMPLATE = """
QMainWindow, QWidget {{
    background-color: {bg};
    color: {text};
    font-family: 'Noto Sans', Arial, sans-serif;
    font-size: 11px;
}}
QGroupBox {{
    background-color: {bg3};
    border: 1px solid {border};
    border-radius: 5px;
    margin-top: 10px;
    padding: 8px 6px 6px 6px;
    font-size: 10px;
    font-weight: bold;
    color: {muted2};
    letter-spacing: 1px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 1px 5px;
    color: {muted2};
}}
QPushButton {{
    background-color: {btn_bg}; color: #ffffff; border: none;
    border-radius: 4px; padding: 4px 9px;
    font-family: 'Noto Sans', Arial, sans-serif; font-size: 10px;
    font-weight: bold; min-height: 24px;
}}
QPushButton:hover   {{ background-color: {btn_hover}; }}
QPushButton:pressed {{ background-color: {btn_pressed}; }}
QPushButton:disabled {{ color: {disabled_text}; background-color: {disabled_bg}; }}
QSlider::groove:horizontal {{
    height: 5px; background: {btn_bg}; border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {blue}; width: 14px; height: 14px;
    margin: -5px 0; border-radius: 7px; border: 2px solid {blue_light};
}}
QSlider::sub-page:horizontal {{
    background: {blue_dark}; border-radius: 3px;
}}
QSlider#servo_slider::handle:horizontal {{
    background: {amber}; border: 2px solid {amber_light};
}}
QSlider#servo_slider::sub-page:horizontal {{
    background: {amber_dark}; border-radius: 3px;
}}
QLineEdit {{
    background-color: {input_bg}; color: {text};
    border: 1px solid {border}; border-radius: 4px;
    padding: 3px 6px; font-family: 'Noto Sans', Arial, sans-serif; font-size: 11px;
    min-height: 22px;
}}
QLineEdit:focus {{ border-color: {blue}; background-color: {input_focus}; }}
QComboBox {{
    background-color: {input_bg}; color: {text};
    border: 1px solid {border}; border-radius: 4px;
    padding: 3px 6px; font-family: 'Noto Sans', Arial, sans-serif; font-size: 11px;
    min-height: 24px;
}}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background-color: {input_bg}; color: {text};
    selection-background-color: {btn_bg};
    border: 1px solid {border}; font-size: 11px;
}}
QPlainTextEdit, QTextEdit {{
    background-color: {bg0}; color: {text_dim};
    border: 1px solid {border2}; border-radius: 4px;
    font-family: 'Noto Sans', Arial, sans-serif; font-size: 10px;
}}
QTabWidget::pane {{
    border: 1px solid {border}; border-radius: 4px;
    background-color: {bg2};
}}
QTabBar::tab {{
    background-color: {input_bg}; color: {muted};
    padding: 4px 12px; border: 1px solid {border2};
    border-bottom: none; border-radius: 4px 4px 0 0;
    font-family: 'Noto Sans', Arial, sans-serif; font-size: 10px; min-width: 52px;
}}
QTabBar::tab:selected {{
    background-color: {bg2}; color: {text};
    border-color: {blue};
}}
QTabBar::tab:hover {{ background-color: {tab_hover_bg}; color: {tab_hover_text}; }}
QScrollBar:vertical {{
    background: {bg}; width: 7px; margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {border}; border-radius: 3px; min-height: 14px;
}}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{
    background: {bg}; height: 7px; margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {border}; border-radius: 3px; min-width: 14px;
}}
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {{ width: 0; }}
QFrame#divider {{
    background-color: {border2}; max-height: 1px;
}}
QCheckBox {{ color: {text}; font-size: 10px; }}
QCheckBox::indicator {{
    width: 12px; height: 12px;
    background-color: {input_bg}; border: 1px solid {border};
    border-radius: 2px;
}}
QCheckBox::indicator:checked {{ background-color: {connect}; }}
QSpinBox, QDoubleSpinBox {{
    background-color: {input_bg}; color: {text};
    border: 1px solid {border}; border-radius: 4px;
    padding: 2px 5px; font-family: 'Noto Sans', Arial, sans-serif;
    min-height: 22px; font-size: 11px;
}}
QProgressBar {{
    background-color: {input_bg}; border: 1px solid {border};
    border-radius: 4px; color: {text}; text-align: center;
    font-size: 10px; min-height: 14px;
}}
QProgressBar::chunk {{ background-color: {connect}; border-radius: 3px; }}
QScrollArea {{ border: none; background: transparent; }}
QMenuBar {{
    background-color: {bg}; color: {text};
    font-family: 'Noto Sans', Arial, sans-serif; font-size: 10px;
}}
QMenuBar::item:selected {{ background-color: {input_bg}; }}
QMenu {{
    background-color: {input_bg}; color: {text};
    border: 1px solid {border};
}}
QMenu::item:selected {{ background-color: {btn_bg}; }}
QMessageBox {{ background-color: {bg3}; color: {text}; }}
QToolTip {{
    background-color: {bg3}; color: {text};
    border: 1px solid {border}; padding: 4px; border-radius: 3px;
}}
"""


def build_style(theme_key: str) -> str:
    """Return the full global QSS ('chrome') string for a theme."""
    p = PALETTES[theme_key]
    derived = dict(
        blue_light=_lighten(p["blue"], 35),
        blue_dark=_darken(p["blue"], 40),
        amber_light=_lighten(p["amber"], 35),
        amber_dark=_darken(p["amber"], 45),
        input_focus=_lighten(p["input_bg"], 20) if theme_key not in ("field_dark", "aben_dark")
        else "#1e2238",
        tab_hover_bg=_lighten(p["input_bg"], 12) if theme_key not in ("field_dark", "aben_dark")
        else "#202535",
        tab_hover_text=_lighten(p["muted"], 30) if theme_key not in ("field_dark", "aben_dark")
        else "#c0c8d8",
    )
    fmt = dict(p)
    fmt.update(derived)
    return _STYLE_TEMPLATE.format(**fmt)


STYLES = {key: build_style(key) for key in THEME_ORDER}


# ─────────────────────────────────────────────────────────────
#  THEME MANAGER
# ─────────────────────────────────────────────────────────────
_CONFIG_PATH = Path(__file__).resolve().parent / "theme_config.json"


class ThemeManager:
    """
    Singleton-style registry. Tracks the active theme plus every
    button / LED / callback that needs restyling when the theme
    changes, so a single `apply()` call performs a *full* re-theme.
    """

    def __init__(self):
        self.current = DEFAULT_THEME
        self._buttons = []     # list of (weakref, role)
        self._widgets = []     # list of (weakref, style_fn(palette)->str)
        self._callbacks = []   # list of zero-arg callables

    # ── Lookups for the active theme ──────────────────────────
    def palette(self) -> dict:
        return PALETTES[self.current]

    def style(self) -> str:
        return STYLES[self.current]

    def button_style(self, role: str) -> str:
        return BUTTON_STYLES[self.current].get(
            role, BUTTON_STYLES[self.current]["blue"])

    def led_colors(self) -> dict:
        return LED_COLORS[self.current]

    def list_themes(self):
        """Return [(key, label), ...] in display order."""
        return [(k, THEME_LABELS[k]) for k in THEME_ORDER]

    # ── Registration ───────────────────────────────────────────
    def register_button(self, widget, role: str):
        """Apply `role` styling to `widget` now, and keep it themed.

        Safe to call repeatedly on the same widget (e.g. when a
        toggle button switches role between 'green' and 'dim_green')
        — the registry entry is updated in place rather than
        duplicated.
        """
        for i, (ref, _) in enumerate(self._buttons):
            w = ref()
            if w is widget:
                self._buttons[i] = (ref, role)
                break
        else:
            self._buttons.append((weakref.ref(widget), role))
        try:
            widget.setStyleSheet(self.button_style(role))
        except RuntimeError:
            pass

    def register_widget(self, widget, style_fn):
        """Register a non-button widget (label, divider, frame, ...)
        for re-theming. `style_fn(palette_dict) -> stylesheet string`
        is called immediately and again on every theme change.

        Safe to call repeatedly on the same widget (e.g. a status
        label that swaps style_fn between "idle" and "active" looks)
        — the registry entry is updated in place rather than
        duplicated.
        """
        for i, (ref, _) in enumerate(self._widgets):
            w = ref()
            if w is widget:
                self._widgets[i] = (ref, style_fn)
                break
        else:
            self._widgets.append((weakref.ref(widget), style_fn))
        try:
            widget.setStyleSheet(style_fn(self.palette()))
        except RuntimeError:
            pass

    def on_change(self, callback):
        """Register a zero-arg callable invoked on every theme change
        (e.g. to re-apply a widget-level stylesheet built from
        gui.style.STYLE)."""
        self._callbacks.append(callback)

    # ── Apply / persist ─────────────────────────────────────────
    def apply(self, theme_key: str, app=None, save: bool = True):
        if theme_key not in PALETTES:
            theme_key = DEFAULT_THEME
        self.current = theme_key

        # Keep gui.style's module-level constants in sync for any code
        # that accesses them fresh (e.g. `from gui import style` then
        # `style.STYLE` / `style.BTN_GREEN` at call time).
        try:
            from gui import style as _style
            _style.refresh()
        except Exception:
            pass

        if app is not None:
            app.setStyleSheet(self.style())

        # Re-apply every registered button
        alive = []
        for ref, role in self._buttons:
            w = ref()
            if w is None:
                continue
            alive.append((ref, role))
            try:
                w.setStyleSheet(self.button_style(role))
            except RuntimeError:
                pass
        self._buttons = alive

        # Re-apply every registered non-button widget (labels, dividers, ...)
        alive_w = []
        p = self.palette()
        for ref, style_fn in self._widgets:
            w = ref()
            if w is None:
                continue
            alive_w.append((ref, style_fn))
            try:
                w.setStyleSheet(style_fn(p))
            except RuntimeError:
                pass
        self._widgets = alive_w

        # Refresh every LED instance
        try:
            from gui.style import LED
            for led in list(LED._instances):
                led.refresh()
        except Exception:
            pass

        # Run any extra registered callbacks (panel-specific re-theming)
        for cb in list(self._callbacks):
            try:
                cb()
            except Exception:
                logger.exception("theme on_change callback failed")

        if save:
            self.save()

        logger.info("Applied theme '%s'", theme_key)

    def load(self):
        """Load the persisted theme choice (call once at startup,
        before apply())."""
        try:
            if _CONFIG_PATH.exists():
                data = json.loads(_CONFIG_PATH.read_text())
                key = data.get("theme")
                if key in PALETTES:
                    self.current = key
        except Exception:
            logger.exception("Failed to load theme config")
        return self.current

    def save(self):
        try:
            _CONFIG_PATH.write_text(json.dumps({"theme": self.current}, indent=2))
        except Exception:
            logger.exception("Failed to save theme config")


# Module-level singleton — import this everywhere
theme_manager = ThemeManager()