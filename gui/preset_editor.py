"""
gui/preset_editor.py
ABEN Dual RGB Imaging System — Camera Preset Editor

Dialog for viewing, editing and saving the three lighting presets
(Outdoor / Cloudy / Indoor) that used to be hardcoded in
core/dual_emeet_camera.py.

Two ways to set a preset, per operator request:
  - "Save current camera settings" -- one click, captures whatever
    the camera is actually running right now (the practical path:
    tune live until detection looks good, then store it)
  - Direct editing of every value in a table (the precise path, and
    also just so the operator can SEE what each preset actually is)

Factory defaults are never overwritten (see core/camera_presets.py),
so "Restore factory defaults" always works regardless of how a preset
has been tuned.
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTabWidget, QWidget, QSpinBox, QGridLayout, QMessageBox,
)
from PyQt5.QtCore import Qt

from gui.theme_manager import theme_manager
from core import camera_presets as cp


class PresetEditorDialog(QDialog):
    """
    read_current_fn: optional callable returning a settings dict of
                     the camera's CURRENT values (used by the "save
                     current camera settings" button). If None, that
                     button is disabled with an explanatory tooltip
                     rather than hidden, so it's clear the feature
                     exists but needs a connected camera.
    """

    def __init__(self, parent=None, read_current_fn=None):
        super().__init__(parent)
        self._read_current_fn = read_current_fn
        self._spins = {}       # {preset_name: {key: QSpinBox}}
        self._presets = cp.load_presets()

        self.setWindowTitle("Camera Lighting Presets")
        self.setMinimumWidth(520)
        theme_manager.register_widget(
            self, lambda p: (
                f"QDialog{{background:{p['bg']};color:{p['text']};}}"
                f"QLabel{{color:{p['text']};"
                f"font-family:'Noto Sans',Arial,sans-serif;font-size:11px;}}"
                f"QSpinBox{{background:{p['input_bg']};color:{p['text']};"
                f"border:1px solid {p['border2']};border-radius:3px;"
                f"padding:2px 4px;"
                f"font-family:'Noto Sans',Arial,sans-serif;font-size:11px;}}"
                f"QTabWidget::pane{{border:1px solid {p['border2']};}}"
                f"QTabBar::tab{{background:{p['bg2']};color:{p['muted']};"
                f"padding:6px 14px;"
                f"font-family:'Noto Sans',Arial,sans-serif;font-size:11px;}}"
                f"QTabBar::tab:selected{{background:{p['bg3']};"
                f"color:{p['text']};}}"))

        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        blurb = QLabel(
            "Edit the camera settings used by each lighting preset. "
            "Changes are saved to camera_presets.json; factory defaults "
            "are always recoverable.")
        blurb.setWordWrap(True)
        theme_manager.register_widget(
            blurb, lambda p: (
                f"color:{p['muted']};font-size:10px;"
                f"font-family:'Noto Sans',Arial,sans-serif;"))
        lay.addWidget(blurb)

        self.tabs = QTabWidget()
        for name in cp.PRESET_NAMES:
            self.tabs.addTab(self._build_preset_tab(name),
                             cp.DISPLAY_NAMES[name])
        lay.addWidget(self.tabs)

        self.lbl_status = QLabel("")
        theme_manager.register_widget(
            self.lbl_status, lambda p: (
                f"color:{p['green']};font-size:10px;"
                f"font-family:'Noto Sans',Arial,sans-serif;"))
        lay.addWidget(self.lbl_status)

        btn_row = QHBoxLayout()

        self.btn_from_camera = QPushButton("📷  Save current camera settings")
        theme_manager.register_button(self.btn_from_camera, "blue")
        self.btn_from_camera.clicked.connect(self._load_from_camera)
        if self._read_current_fn is None:
            self.btn_from_camera.setEnabled(False)
            self.btn_from_camera.setToolTip(
                "Needs a connected camera — open this from the Data "
                "Collection tab with the cameras connected.")
        else:
            self.btn_from_camera.setToolTip(
                "Read the camera's current values into this preset's "
                "fields (does not save until you press Save)")
        btn_row.addWidget(self.btn_from_camera)

        btn_restore = QPushButton("↺  Restore factory defaults")
        theme_manager.register_button(btn_restore, "amber")
        btn_restore.clicked.connect(self._restore_defaults)
        btn_row.addWidget(btn_restore)

        btn_row.addStretch()

        btn_save = QPushButton("💾  Save preset")
        theme_manager.register_button(btn_save, "green")
        btn_save.clicked.connect(self._save_current_tab)
        btn_row.addWidget(btn_save)

        btn_close = QPushButton("Close")
        theme_manager.register_widget(
            btn_close, lambda p: (
                f"QPushButton{{background:{p['bg2']};color:{p['muted']};"
                f"border:1px solid {p['border2']};border-radius:4px;"
                f"padding:6px 14px;"
                f"font-family:'Noto Sans',Arial,sans-serif;font-size:11px;}}"
                f"QPushButton:hover{{background:{p['btn_hover']};"
                f"color:{p['text']};}}"))
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)

        lay.addLayout(btn_row)

    def _build_preset_tab(self, name: str) -> QWidget:
        w = QWidget()
        grid = QGridLayout(w)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setSpacing(6)

        values = self._presets[name]
        self._spins[name] = {}

        # Two columns of (label, spinbox) so all 14 settings fit
        # without scrolling.
        half = (len(cp.SETTING_SPECS) + 1) // 2
        for i, (key, label, lo, hi) in enumerate(cp.SETTING_SPECS):
            col = 0 if i < half else 2
            row = i if i < half else i - half

            lbl = QLabel(label)
            grid.addWidget(lbl, row, col)

            spin = QSpinBox()
            spin.setRange(lo, hi)
            spin.setValue(int(values.get(key, lo)))
            spin.setFixedWidth(90)
            grid.addWidget(spin, row, col + 1)
            self._spins[name][key] = spin

        return w

    def _current_name(self) -> str:
        return cp.PRESET_NAMES[self.tabs.currentIndex()]

    def _status(self, msg: str, ok: bool = True):
        self.lbl_status.setText(msg)
        theme_manager.register_widget(
            self.lbl_status, lambda p, _ok=ok: (
                f"color:{p['green'] if _ok else p['red']};font-size:10px;"
                f"font-family:'Noto Sans',Arial,sans-serif;"))

    def _load_from_camera(self):
        """Populate the current tab's fields from the live camera."""
        if self._read_current_fn is None:
            return
        try:
            current = self._read_current_fn()
        except Exception as e:
            self._status(f"Could not read camera settings: {e}", ok=False)
            return
        if not current:
            self._status(
                "Camera returned no settings — is it connected?", ok=False)
            return

        name = self._current_name()
        applied = 0
        for key, spin in self._spins[name].items():
            if key in current:
                try:
                    spin.setValue(int(current[key]))
                    applied += 1
                except Exception:
                    pass
        self._status(
            f"Loaded {applied} live value(s) into "
            f"{cp.DISPLAY_NAMES[name]} — press Save preset to keep them.")

    def _save_current_tab(self):
        name = self._current_name()
        settings = {k: s.value() for k, s in self._spins[name].items()}
        if cp.save_preset(name, settings):
            self._presets[name] = settings
            self._status(f"Saved {cp.DISPLAY_NAMES[name]} preset.")
        else:
            self._status(
                f"Could not save {cp.DISPLAY_NAMES[name]} — check file "
                f"permissions on {cp.PRESET_FILE}", ok=False)

    def _restore_defaults(self):
        name = self._current_name()
        reply = QMessageBox.question(
            self, "Restore factory defaults",
            f"Discard your saved values for "
            f"{cp.DISPLAY_NAMES[name]} and restore the original "
            f"factory defaults?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        if cp.reset_preset(name):
            factory = cp.factory_preset(name)
            for key, spin in self._spins[name].items():
                if key in factory:
                    spin.setValue(int(factory[key]))
            self._presets[name] = dict(factory)
            self._status(
                f"Restored {cp.DISPLAY_NAMES[name]} to factory defaults.")
        else:
            self._status(
                f"Could not restore {cp.DISPLAY_NAMES[name]}.", ok=False)
