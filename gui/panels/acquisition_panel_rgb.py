"""
acquisition_panel_rgb.py
eMeet Dual RGB Detection System — Data Acquisition Panel

Fork of acquisition_panel.py for the dual RGB camera system.

Two subtabs — same structure as the multispectral panel:
  1. Camera Settings  — per-camera v4l2-ctl controls (LEFT / RIGHT tabs)
  2. Capture          — Image / Auto / Video subtabs

Key differences from the multispectral panel:
  - Camera settings use v4l2-ctl instead of GenICam node_map
  - Per-camera: LEFT and RIGHT each have independent controls
  - Capture source selector: Both / Left Only / Right Only
  - Saves left.jpg + right.jpg + metadata JSON + session log
    (no band TIFs, no PseudoRGB — it's already RGB)
  - Video records left and right as separate MP4 files

Everything else is identical:
  - Session labels (crop, target, stage, disease, weed type, notes)
  - Auto capture (interval, max captures, pause/resume/stop, progress)
  - Browse/save path, session key, capture counter

Author : Nana | NDSU / PhD Imaging System
Path   : /media/pagsun/Transcend/phd_project/emeet_dual_cam/
"""

import re
import cv2
import json
import time
import subprocess
import numpy as np
from datetime import datetime
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QSlider, QLineEdit,
    QGroupBox, QTabWidget, QCheckBox, QDoubleSpinBox,
    QSpinBox, QComboBox, QFileDialog, QScrollArea,
    QProgressBar, QButtonGroup, QRadioButton, QFrame
)
from PyQt5.QtCore import Qt, QTimer

from gui.style import _divider, _muted, _sec, _scroll
from gui.theme_manager import theme_manager, _darken, _lighten
from gui.shared_log import UnifiedLog


# ─────────────────────────────────────────────────────────────
#  CONSTANTS  (identical to multispectral panel)
# ─────────────────────────────────────────────────────────────

CROPS = [
    "sugarbeet", "wheat", "corn", "sorghum",
    "soybean", "barley", "other"
]
STAGES = [
    "emergence", "cotyledon", "2_leaf", "4_leaf",
    "6_leaf", "8_leaf", "canopy_closure",
    "vegetative", "maturity"
]
TARGETS = [
    "cls_infected", "cls_healthy", "kochia", "waterhemp",
    "common_ragweed", "common_lambsquarters", "redroot_pigweed",
    "kochia_res", "kochia_sus", "waterhemp_res", "waterhemp_sus",
    "common_ragweed_res", "common_ragweed_sus",
    "common_lambsquarters_res", "common_lambsquarters_sus", "weed"
]

BASE_PATH = Path(
    "/media/pagsun/Transcend/phd_project/emeet_dual_cam/training_data"
)

# eMeet device paths
LEFT_DEVICE  = (
    "/dev/v4l/by-id/"
    "usb-EMEET_EMEET_SmartCam_C960_4K_A241213000400860-video-index0"
)
RIGHT_DEVICE = (
    "/dev/v4l/by-id/"
    "usb-EMEET_EMEET_SmartCam_C960_4K_A241217000804000-video-index0"
)

# Capture source options
SRC_BOTH  = "Both"
SRC_LEFT  = "Left only"
SRC_RIGHT = "Right only"


# ─────────────────────────────────────────────────────────────
#  v4l2 HELPER
# ─────────────────────────────────────────────────────────────

def _v4l2_set(device: str, control: str, value) -> bool:
    """Set one v4l2 control. Returns True on success."""
    cmd = ["v4l2-ctl", "-d", device, "-c", f"{control}={value}"]
    r   = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0

def _v4l2_get(device: str, control: str):
    """
    Read one v4l2 control value.
    Returns int value or None on failure.
    """
    cmd = ["v4l2-ctl", "-d", device, "--get-ctrl", control]
    r   = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None
    # Output: "control_name: value"
    parts = r.stdout.strip().split(":")
    if len(parts) >= 2:
        try:
            return int(parts[-1].strip())
        except ValueError:
            return None
    return None


# ─────────────────────────────────────────────────────────────
#  SINGLE-CAMERA SETTINGS WIDGET
# ─────────────────────────────────────────────────────────────

class CameraSettingsWidget(QWidget):
    """
    Settings panel for ONE eMeet camera via v4l2-ctl.
    Instantiated twice — once for LEFT, once for RIGHT.
    All controls mirror the validated settings from dual_emeet_camera.py.
    """

    def __init__(self, device: str, label: str,
                 shared_log: UnifiedLog, parent=None):
        super().__init__(parent)
        self.device     = device
        self.label      = label   # "LEFT" or "RIGHT"
        self.shared_log = shared_log
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        lay.addWidget(_sec(f"{self.label}  —  {self.device.split('/')[-1][:32]}"))
        lay.addWidget(_divider())

        # ── Focus ─────────────────────────────────────────────
        foc_grp = QGroupBox("Focus")
        fg = QVBoxLayout(foc_grp)

        self.chk_autofocus = QCheckBox("Auto Focus (continuous)")
        self.chk_autofocus.setChecked(True)
        self.chk_autofocus.stateChanged.connect(self._on_autofocus_toggle)
        fg.addWidget(self.chk_autofocus)

        frow = QHBoxLayout()
        frow.addWidget(_muted("Absolute (0–1023):"))
        self.spn_focus = QSpinBox()
        self.spn_focus.setRange(0, 1023)
        self.spn_focus.setValue(460)
        self.spn_focus.setSingleStep(10)
        self.spn_focus.setMinimumWidth(90)
        frow.addWidget(self.spn_focus)
        frow.addStretch()
        fg.addLayout(frow)
        lay.addWidget(foc_grp)

        # ── Exposure ──────────────────────────────────────────
        exp_grp = QGroupBox("Exposure")
        eg = QVBoxLayout(exp_grp)

        self.chk_auto_exp = QCheckBox("Auto Exposure")
        self.chk_auto_exp.stateChanged.connect(self._on_auto_exp_toggle)
        eg.addWidget(self.chk_auto_exp)

        erow = QHBoxLayout()
        erow.addWidget(_muted("Time absolute (1–10000):"))
        self.spn_exposure = QSpinBox()
        self.spn_exposure.setRange(1, 10000)
        self.spn_exposure.setValue(300)
        self.spn_exposure.setSingleStep(10)
        self.spn_exposure.setMinimumWidth(90)
        erow.addWidget(self.spn_exposure)
        erow.addStretch()
        eg.addLayout(erow)
        lay.addWidget(exp_grp)

        # ── White Balance ─────────────────────────────────────
        wb_grp = QGroupBox("White Balance")
        wg = QVBoxLayout(wb_grp)

        self.chk_auto_wb = QCheckBox("Auto White Balance")
        self.chk_auto_wb.stateChanged.connect(self._on_auto_wb_toggle)
        wg.addWidget(self.chk_auto_wb)

        wrow = QHBoxLayout()
        wrow.addWidget(_muted("Temperature (2800–6500 K):"))
        self.spn_wb_temp = QSpinBox()
        self.spn_wb_temp.setRange(2800, 6500)
        self.spn_wb_temp.setValue(5000)
        self.spn_wb_temp.setSingleStep(100)
        self.spn_wb_temp.setMinimumWidth(90)
        wrow.addWidget(self.spn_wb_temp)
        wrow.addStretch()
        wg.addLayout(wrow)
        lay.addWidget(wb_grp)

        # ── Image Controls ────────────────────────────────────
        img_grp = QGroupBox("Image Controls")
        ig = QGridLayout(img_grp)
        ig.setSpacing(6)

        controls = [
            ("Brightness (-64–64):", "spn_brightness",  -64,  64,  1,   0),
            ("Contrast (0–64):",     "spn_contrast",      0,  64,  1,  57),
            ("Saturation (0–128):", "spn_saturation",     0, 128,  1,  80),
            ("Hue (-40–40):",       "spn_hue",          -40,  40,  1,   0),
            ("Gamma (72–500):",     "spn_gamma",         72, 500,  1, 214),
            ("Gain (0–100):",       "spn_gain",           0, 100,  1,   0),
            ("Sharpness (0–6):",    "spn_sharpness",      0,   6,  1,   3),
        ]
        for row, (label, attr, mn, mx, step, default) in \
                enumerate(controls):
            ig.addWidget(_muted(label), row, 0)
            spn = QSpinBox()
            spn.setRange(mn, mx)
            spn.setValue(default)
            spn.setSingleStep(step)
            spn.setMinimumWidth(80)
            ig.addWidget(spn, row, 1)
            setattr(self, attr, spn)

        # Backlight compensation
        ig.addWidget(_muted("Backlight comp (0–1):"),
                     len(controls), 0)
        self.spn_backlight = QSpinBox()
        self.spn_backlight.setRange(0, 1)
        self.spn_backlight.setValue(0)
        ig.addWidget(self.spn_backlight, len(controls), 1)

        # Power line frequency
        ig.addWidget(_muted("Power line freq:"),
                     len(controls) + 1, 0)
        self.cmb_freq = QComboBox()
        self.cmb_freq.addItems(["Disabled (0)", "50 Hz (1)", "60 Hz (2)"])
        self.cmb_freq.setCurrentIndex(2)  # default 60 Hz
        ig.addWidget(self.cmb_freq, len(controls) + 1, 1)

        lay.addWidget(img_grp)

        # ── Apply / Refresh buttons ───────────────────────────
        lay.addWidget(_divider())
        btn_row = QHBoxLayout()

        self.btn_refresh = QPushButton("↻  Read from camera")
        theme_manager.register_button(self.btn_refresh, "blue")
        self.btn_refresh.clicked.connect(self._refresh_all)
        btn_row.addWidget(self.btn_refresh)

        self.btn_apply = QPushButton("✓  Apply to camera")
        theme_manager.register_button(self.btn_apply, "green")
        self.btn_apply.setMinimumHeight(30)
        self.btn_apply.clicked.connect(self._apply_all)
        btn_row.addWidget(self.btn_apply)

        self.btn_reset = QPushButton("↺  Reset defaults")
        theme_manager.register_button(self.btn_reset, "amber")
        self.btn_reset.clicked.connect(self._reset_defaults)
        btn_row.addWidget(self.btn_reset)
        btn_row.addStretch()

        lay.addLayout(btn_row)

        # ── Lighting presets ──────────────────────────────────
        lay.addWidget(_divider())
        preset_lbl = QLabel("Quick Presets:")
        theme_manager.register_widget(
            preset_lbl, lambda p: (
                f"color:{p['muted']};font-size:9px;"
                f"font-family:'Noto Sans',Arial,sans-serif;"))
        lay.addWidget(preset_lbl)

        preset_row = QHBoxLayout()

        btn_outdoor = QPushButton("🌤  Outdoor / Field")
        theme_manager.register_widget(
            btn_outdoor, lambda p: (
                f"QPushButton{{background:{_darken(p['green'],45)};"
                f"color:{_lighten(p['green'],10)};"
                f"border:1px solid {_darken(p['green'],25)};border-radius:4px;"
                f"padding:5px 10px;font-family:'Noto Sans',Arial,sans-serif;"
                f"font-size:10px;}}"
                f"QPushButton:hover{{background:{_darken(p['green'],35)};}}"))
        btn_outdoor.setToolTip(
            "Outdoor / direct sunlight\n"
            "exposure=5  brightness=-10  gamma=150  wb=5500")
        btn_outdoor.clicked.connect(self._preset_outdoor)
        preset_row.addWidget(btn_outdoor)

        btn_cloudy = QPushButton("☁  Cloudy / Shade")
        theme_manager.register_widget(
            btn_cloudy, lambda p: (
                f"QPushButton{{background:{_darken(p['blue'],45)};"
                f"color:{_lighten(p['blue'],10)};"
                f"border:1px solid {_darken(p['blue'],25)};border-radius:4px;"
                f"padding:5px 10px;font-family:'Noto Sans',Arial,sans-serif;"
                f"font-size:10px;}}"
                f"QPushButton:hover{{background:{_darken(p['blue'],35)};}}"))
        btn_cloudy.setToolTip(
            "Overcast / cloudy / shade\n"
            "exposure=80  brightness=0  gamma=180  wb=6000")
        btn_cloudy.clicked.connect(self._preset_cloudy)
        preset_row.addWidget(btn_cloudy)

        btn_indoor = QPushButton("💡  Indoor / Lab")
        theme_manager.register_widget(
            btn_indoor, lambda p: (
                f"QPushButton{{background:{_darken(p['amber'],55)};"
                f"color:{_lighten(p['amber'],5)};"
                f"border:1px solid {_darken(p['amber'],30)};border-radius:4px;"
                f"padding:5px 10px;font-family:'Noto Sans',Arial,sans-serif;"
                f"font-size:10px;}}"
                f"QPushButton:hover{{background:{_darken(p['amber'],45)};}}"))
        btn_indoor.setToolTip(
            "Indoor lab / bench testing\n"
            "exposure=300  brightness=0  gamma=214  wb=5000")
        btn_indoor.clicked.connect(self._preset_indoor)
        preset_row.addWidget(btn_indoor)

        preset_row.addStretch()
        lay.addLayout(preset_row)
        lay.addStretch()

    # ── Auto-toggle helpers ───────────────────────────────────

    def _on_autofocus_toggle(self, state):
        self.spn_focus.setEnabled(state != Qt.Checked)

    def _on_auto_exp_toggle(self, state):
        self.spn_exposure.setEnabled(state != Qt.Checked)

    def _on_auto_wb_toggle(self, state):
        self.spn_wb_temp.setEnabled(state != Qt.Checked)

    # ── Lighting presets ─────────────────────────────────────

    def _apply_preset(self, preset: dict, name: str):
        """
        Apply a preset dict to all UI spinboxes then push to camera.
        preset keys match DualEMEETCamera.PRESET_* dicts.
        """
        mapping = {
            "exposure":   "spn_exposure",
            "brightness": "spn_brightness",
            "contrast":   "spn_contrast",
            "saturation": "spn_saturation",
            "gamma":      "spn_gamma",
            "gain":       "spn_gain",
            "sharpness":  "spn_sharpness",
            "wb_temp":    "spn_wb_temp",
            "backlight":  "spn_backlight",
            "focus":      "spn_focus",
        }
        for key, attr in mapping.items():
            if key in preset and hasattr(self, attr):
                getattr(self, attr).setValue(preset[key])
        self._apply_all()
        self.shared_log.log(
            "CAMERA", f"Preset applied: {name}", "ok")

    def _preset_outdoor(self):
        """Outdoor / direct sunlight — exposure=5."""
        try:
            from core.dual_emeet_camera import DualEMEETCamera
        except ImportError:
            from core.dual_emeet_camera import DualEMEETCamera
        self._apply_preset(DualEMEETCamera.PRESET_OUTDOOR,
                           "Outdoor / Field (exp=5)")

    def _preset_cloudy(self):
        """Overcast / cloudy / shade — exposure=80."""
        try:
            from core.dual_emeet_camera import DualEMEETCamera
        except ImportError:
            from core.dual_emeet_camera import DualEMEETCamera
        self._apply_preset(DualEMEETCamera.PRESET_CLOUDY,
                           "Cloudy / Shade (exp=80)")

    def _preset_indoor(self):
        """Indoor lab — exposure=300 (factory default)."""
        try:
            from core.dual_emeet_camera import DualEMEETCamera
        except ImportError:
            from core.dual_emeet_camera import DualEMEETCamera
        self._apply_preset(DualEMEETCamera.PRESET_INDOOR,
                           "Indoor / Lab (exp=300)")

    # ── Read from camera ──────────────────────────────────────

    def _refresh_all(self):
        """Read current v4l2 values and populate UI."""
        controls_map = {
            "focus_absolute":          self.spn_focus,
            "exposure_time_absolute":  self.spn_exposure,
            "white_balance_temperature": self.spn_wb_temp,
            "brightness":              self.spn_brightness,
            "contrast":                self.spn_contrast,
            "saturation":              self.spn_saturation,
            "hue":                     self.spn_hue,
            "gamma":                   self.spn_gamma,
            "gain":                    self.spn_gain,
            "sharpness":               self.spn_sharpness,
            "backlight_compensation":  self.spn_backlight,
        }
        errors = []
        for ctrl, spn in controls_map.items():
            val = _v4l2_get(self.device, ctrl)
            if val is not None:
                spn.blockSignals(True)
                spn.setValue(val)
                spn.blockSignals(False)
            else:
                errors.append(ctrl)

        # Bool controls
        af = _v4l2_get(self.device, "focus_automatic_continuous")
        if af is not None:
            self.chk_autofocus.blockSignals(True)
            self.chk_autofocus.setChecked(bool(af))
            self.chk_autofocus.blockSignals(False)
            self.spn_focus.setEnabled(not bool(af))

        ae = _v4l2_get(self.device, "auto_exposure")
        if ae is not None:
            # auto_exposure: 1=manual, 3=auto (v4l2 convention)
            is_auto = (ae == 3)
            self.chk_auto_exp.blockSignals(True)
            self.chk_auto_exp.setChecked(is_auto)
            self.chk_auto_exp.blockSignals(False)
            self.spn_exposure.setEnabled(not is_auto)

        awb = _v4l2_get(self.device, "white_balance_automatic")
        if awb is not None:
            self.chk_auto_wb.blockSignals(True)
            self.chk_auto_wb.setChecked(bool(awb))
            self.chk_auto_wb.blockSignals(False)
            self.spn_wb_temp.setEnabled(not bool(awb))

        freq = _v4l2_get(self.device, "power_line_frequency")
        if freq is not None:
            self.cmb_freq.setCurrentIndex(
                min(freq, self.cmb_freq.count() - 1))

        msg = f"{self.label}: settings read"
        if errors:
            msg += f" (skipped: {', '.join(errors[:3])})"
        self.shared_log.log("CAMERA", msg,
                            "ok" if not errors else "warn")

    # ── Apply to camera ───────────────────────────────────────

    def _apply_all(self):
        """Write all UI values to camera via v4l2-ctl."""
        errors = []

        def _set(ctrl, value):
            if not _v4l2_set(self.device, ctrl, value):
                errors.append(ctrl)

        # Focus
        af = 1 if self.chk_autofocus.isChecked() else 0
        _set("focus_automatic_continuous", af)
        if not af:
            time.sleep(0.05)
            _set("focus_absolute", self.spn_focus.value())

        # Exposure
        if self.chk_auto_exp.isChecked():
            _set("auto_exposure", 3)   # 3=auto
        else:
            _set("auto_exposure", 1)   # 1=manual
            time.sleep(0.05)
            _set("exposure_time_absolute", self.spn_exposure.value())

        # White balance
        awb = 1 if self.chk_auto_wb.isChecked() else 0
        _set("white_balance_automatic", awb)
        if not awb:
            time.sleep(0.05)
            _set("white_balance_temperature",
                 self.spn_wb_temp.value())

        # Image controls
        _set("brightness",           self.spn_brightness.value())
        _set("contrast",             self.spn_contrast.value())
        _set("saturation",           self.spn_saturation.value())
        _set("hue",                  self.spn_hue.value())
        _set("gamma",                self.spn_gamma.value())
        _set("gain",                 self.spn_gain.value())
        _set("sharpness",            self.spn_sharpness.value())
        _set("backlight_compensation", self.spn_backlight.value())
        _set("power_line_frequency",
             self.cmb_freq.currentIndex())

        msg = f"{self.label}: settings applied"
        if errors:
            msg += f" (failed: {', '.join(errors[:4])})"
        self.shared_log.log("CAMERA", msg,
                            "ok" if not errors else "warn")

    # ── Reset to validated defaults ───────────────────────────

    def _reset_defaults(self):
        """Restore the validated field-ready default values."""
        self.chk_autofocus.setChecked(True)
        self.spn_focus.setValue(460)
        self.chk_auto_exp.setChecked(False)
        self.spn_exposure.setValue(300)
        self.chk_auto_wb.setChecked(False)
        self.spn_wb_temp.setValue(5000)
        self.spn_brightness.setValue(0)
        self.spn_contrast.setValue(57)
        self.spn_saturation.setValue(80)
        self.spn_hue.setValue(0)
        self.spn_gamma.setValue(214)
        self.spn_gain.setValue(0)
        self.spn_sharpness.setValue(3)
        self.spn_backlight.setValue(0)
        self.cmb_freq.setCurrentIndex(2)   # 60 Hz
        self.shared_log.log(
            "CAMERA",
            f"{self.label}: defaults restored (not yet applied)",
            "info")


# ─────────────────────────────────────────────────────────────
#  ACQUISITION PANEL RGB
# ─────────────────────────────────────────────────────────────

class AcquisitionPanelRGB(QWidget):
    """
    Camera Settings + Capture panel for the dual eMeet RGB system.

    Drop-in replacement for AcquisitionPanel used in tab_collection
    and tab_detection. Public API is identical:
      panel.subtabs              → QTabWidget (for tab_detection camera tab)
      panel.enable_camera_controls(bool)
      panel.cleanup()
      panel.reset_session()
    """

    def __init__(self, shared_log: UnifiedLog,
                 camera, parent=None):
        super().__init__(parent)
        self.shared_log = shared_log
        self.camera     = camera   # DualCameraPanel

        # Capture state
        self._auto_timer  = None
        self._auto_count  = 0
        self._auto_max    = 0
        self._img_count   = 0
        self._recording   = False
        self._vid_writers = {}    # {"left": cv2.VideoWriter,
                                  #  "right": cv2.VideoWriter}

        # Session state
        self._session_path   = None
        self._session_labels = None
        self._capture_count  = 0
        self._session_id     = None

        self._build_ui()

        # Device info refresh
        self._info_timer = QTimer()
        self._info_timer.timeout.connect(self._refresh_device_status)
        self._info_timer.start(5000)

    # ── Build UI ──────────────────────────────────────────────

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.subtabs = QTabWidget()
        self.subtabs.addTab(
            _scroll(self._tab_cam_settings()), "⚙ Camera")
        self.subtabs.addTab(
            _scroll(self._tab_capture()),      "💾 Capture")
        lay.addWidget(self.subtabs)

    # ─────────────────────────────────────────────────────────
    #  SUBTAB 1: CAMERA SETTINGS
    # ─────────────────────────────────────────────────────────

    def _tab_cam_settings(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        lay.addWidget(_sec("CAMERA SETTINGS  —  eMeet C960 4K"))
        lay.addWidget(_divider())

        # ── Device status bar ─────────────────────────────────
        status_grp = QGroupBox("Device Status")
        sg = QGridLayout(status_grp)
        sg.setSpacing(4)

        sg.addWidget(_muted("Left device:"),  0, 0)
        self.lbl_left_dev = QLabel("—")
        theme_manager.register_widget(
            self.lbl_left_dev, lambda p: (
                f"color:{p['green']};font-size:9px;"
                f"font-family:'Noto Sans',Arial,sans-serif;"))
        sg.addWidget(self.lbl_left_dev, 0, 1)

        sg.addWidget(_muted("Right device:"), 1, 0)
        self.lbl_right_dev = QLabel("—")
        theme_manager.register_widget(
            self.lbl_right_dev, lambda p: (
                f"color:{p['green']};font-size:9px;"
                f"font-family:'Noto Sans',Arial,sans-serif;"))
        sg.addWidget(self.lbl_right_dev, 1, 1)

        sg.addWidget(_muted("Resolution:"),   2, 0)
        self.lbl_res = QLabel("1920 × 1080  @  30fps  MJPG")
        theme_manager.register_widget(
            self.lbl_res, lambda p: (
                f"color:{p['muted']};font-size:9px;"
                f"font-family:'Noto Sans',Arial,sans-serif;"))
        sg.addWidget(self.lbl_res, 2, 1)

        lay.addWidget(status_grp)

        # ── Apply to both button ──────────────────────────────
        sync_row = QHBoxLayout()
        self.btn_sync_lr = QPushButton(
            "⇄  Copy LEFT settings → RIGHT")
        theme_manager.register_button(self.btn_sync_lr, "blue")
        self.btn_sync_lr.clicked.connect(self._sync_left_to_right)
        sync_row.addWidget(self.btn_sync_lr)
        sync_row.addStretch()
        lay.addLayout(sync_row)

        # ── Per-camera tabs ───────────────────────────────────
        cam_tabs = QTabWidget()

        self.left_settings = CameraSettingsWidget(
            LEFT_DEVICE, "LEFT", self.shared_log)
        self.right_settings = CameraSettingsWidget(
            RIGHT_DEVICE, "RIGHT", self.shared_log)

        cam_tabs.addTab(
            _scroll(self.left_settings),  "📷 LEFT camera")
        cam_tabs.addTab(
            _scroll(self.right_settings), "📷 RIGHT camera")

        lay.addWidget(cam_tabs)
        return w

    # ─────────────────────────────────────────────────────────
    #  SUBTAB 2: CAPTURE
    # ─────────────────────────────────────────────────────────

    def _tab_capture(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        lay.addWidget(_sec("DATA CAPTURE"))

        # ── Capture source ────────────────────────────────────
        src_grp = QGroupBox("Capture Source")
        sg = QHBoxLayout(src_grp)
        sg.setSpacing(16)

        self._src_group = QButtonGroup(self)
        for label in [SRC_BOTH, SRC_LEFT, SRC_RIGHT]:
            rb = QRadioButton(label)
            if label == SRC_BOTH:
                rb.setChecked(True)
            self._src_group.addButton(rb)
            sg.addWidget(rb)
        sg.addStretch()
        lay.addWidget(src_grp)

        # ── Session Labels ────────────────────────────────────
        meta_grp = QGroupBox("Session Labels")
        mg = QGridLayout(meta_grp)
        mg.setSpacing(6)

        mg.addWidget(_muted("Crop:"),   0, 0)
        self.cmb_crop = QComboBox()
        self.cmb_crop.addItems(CROPS)
        self.cmb_crop.currentTextChanged.connect(self.reset_session)
        mg.addWidget(self.cmb_crop, 0, 1)

        mg.addWidget(_muted("Target:"), 1, 0)
        self.cmb_target = QComboBox()
        self.cmb_target.addItems(TARGETS)
        self.cmb_target.currentTextChanged.connect(self.reset_session)
        mg.addWidget(self.cmb_target, 1, 1)

        mg.addWidget(_muted("Stage:"),  2, 0)
        self.cmb_stage = QComboBox()
        self.cmb_stage.addItems(STAGES)
        self.cmb_stage.currentTextChanged.connect(self.reset_session)
        mg.addWidget(self.cmb_stage, 2, 1)

        mg.addWidget(_muted("Save to:"), 3, 0)
        folder_row = QHBoxLayout()
        self.lbl_folder = QLabel(str(BASE_PATH))
        theme_manager.register_widget(
            self.lbl_folder, lambda p: (
                f"color:{p['dim']};font-size:9px;"
                f"font-family:'Noto Sans',Arial,sans-serif;"))
        self.lbl_folder.setWordWrap(True)
        folder_row.addWidget(self.lbl_folder, stretch=1)
        btn_browse = QPushButton("Browse")
        theme_manager.register_button(btn_browse, "blue")
        btn_browse.setFixedWidth(60)
        btn_browse.clicked.connect(self._browse_folder)
        folder_row.addWidget(btn_browse)
        mg.addLayout(folder_row, 3, 1)
        lay.addWidget(meta_grp)

        # ── Additional Labels ─────────────────────────────────
        extra_grp = QGroupBox("Additional Labels")
        xg = QGridLayout(extra_grp)
        xg.setSpacing(6)

        xg.addWidget(_muted("Disease present:"), 0, 0)
        self.chk_disease = QCheckBox()
        self.chk_disease.stateChanged.connect(
            self._on_disease_toggle)
        xg.addWidget(self.chk_disease, 0, 1)

        xg.addWidget(_muted("Disease type:"), 1, 0)
        self.cmb_disease_type = QComboBox()
        self.cmb_disease_type.addItems([
            "none", "cercospora", "powdery_mildew",
            "rust", "blight", "root_rot", "other"
        ])
        self.cmb_disease_type.setEnabled(False)
        xg.addWidget(self.cmb_disease_type, 1, 1)

        xg.addWidget(_muted("Weed type:"), 2, 0)
        self.cmb_weed_type = QComboBox()
        self.cmb_weed_type.addItems([
            "none", "kochia", "waterhemp", "ragweed",
            "wild_oat", "pigweed", "lambsquarters",
            "foxtail", "bindweed", "mixed", "unknown"
        ])
        xg.addWidget(self.cmb_weed_type, 2, 1)

        xg.addWidget(_muted("Notes:"), 3, 0)
        self.entry_notes = QLineEdit()
        self.entry_notes.setPlaceholderText(
            "e.g. heavy canopy, wet, 3 DAE …")
        xg.addWidget(self.entry_notes, 3, 1)

        lay.addWidget(extra_grp)

        # ── Capture sub-tabs ──────────────────────────────────
        cap_tabs = QTabWidget()
        cap_tabs.addTab(self._tab_image(),  "Image")
        cap_tabs.addTab(self._tab_auto(),   "Auto")
        cap_tabs.addTab(self._tab_video(),  "Video")
        lay.addWidget(cap_tabs, stretch=1)
        return w

    # ── Image sub-tab ─────────────────────────────────────────

    def _tab_image(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        self.lbl_img_count = _muted("Captures this session: 0")
        lay.addWidget(self.lbl_img_count)

        self.lbl_last_saved = QLabel("—")
        theme_manager.register_widget(
            self.lbl_last_saved, lambda p: (
                f"color:{p['dim']};font-size:9px;"
                f"font-family:'Noto Sans',Arial,sans-serif;"))
        self.lbl_last_saved.setWordWrap(True)
        lay.addWidget(self.lbl_last_saved)

        btn_cap = QPushButton("📷  CAPTURE IMAGE")
        theme_manager.register_button(btn_cap, "green")
        btn_cap.setMinimumHeight(36)
        btn_cap.clicked.connect(self._capture_image)
        lay.addWidget(btn_cap)

        lay.addStretch()
        return w

    # ── Auto sub-tab ──────────────────────────────────────────

    def _tab_auto(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        cfg_grp = QGroupBox("Auto Capture Config")
        cg = QGridLayout(cfg_grp)

        cg.addWidget(_muted("Interval (s):"), 0, 0)
        self.spn_interval = QDoubleSpinBox()
        self.spn_interval.setRange(0.5, 60.0)
        self.spn_interval.setValue(2.0)
        self.spn_interval.setSingleStep(0.5)
        cg.addWidget(self.spn_interval, 0, 1)

        cg.addWidget(_muted("Max captures:"), 1, 0)
        self.spn_max_cap = QSpinBox()
        self.spn_max_cap.setRange(1, 9999)
        self.spn_max_cap.setValue(50)
        cg.addWidget(self.spn_max_cap, 1, 1)

        lay.addWidget(cfg_grp)

        self.auto_progress = QProgressBar()
        self.auto_progress.setRange(0, 100)
        self.auto_progress.setValue(0)
        self.auto_progress.setFormat("Ready")
        lay.addWidget(self.auto_progress)

        btn_row = QHBoxLayout()
        self.btn_auto_start = QPushButton("▶  START")
        theme_manager.register_button(self.btn_auto_start, "green")
        self.btn_auto_start.clicked.connect(self._auto_start)
        btn_row.addWidget(self.btn_auto_start)

        self.btn_auto_pause = QPushButton("⏸  PAUSE")
        theme_manager.register_button(self.btn_auto_pause, "amber")
        self.btn_auto_pause.setEnabled(False)
        self.btn_auto_pause.clicked.connect(self._auto_pause)
        btn_row.addWidget(self.btn_auto_pause)

        self.btn_auto_stop = QPushButton("⏹  STOP")
        theme_manager.register_button(self.btn_auto_stop, "dim_red")
        self.btn_auto_stop.setEnabled(False)
        self.btn_auto_stop.clicked.connect(self._auto_stop)
        btn_row.addWidget(self.btn_auto_stop)

        lay.addLayout(btn_row)
        lay.addStretch()
        return w

    # ── Video sub-tab ─────────────────────────────────────────

    def _tab_video(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        vid_cfg = QGroupBox("Video Config")
        vc = QGridLayout(vid_cfg)

        vc.addWidget(_muted("FPS:"), 0, 0)
        self.spn_vid_fps = QDoubleSpinBox()
        self.spn_vid_fps.setRange(1.0, 30.0)
        self.spn_vid_fps.setValue(15.0)
        vc.addWidget(self.spn_vid_fps, 0, 1)

        vc.addWidget(_muted("Codec:"), 1, 0)
        self.cmb_vid_codec = QComboBox()
        self.cmb_vid_codec.addItems(["mp4v", "XVID"])
        vc.addWidget(self.cmb_vid_codec, 1, 1)

        self.chk_rec_left  = QCheckBox("Record LEFT")
        self.chk_rec_left.setChecked(True)
        self.chk_rec_right = QCheckBox("Record RIGHT")
        self.chk_rec_right.setChecked(True)
        vc.addWidget(self.chk_rec_left,  2, 0)
        vc.addWidget(self.chk_rec_right, 2, 1)
        lay.addWidget(vid_cfg)

        vid_lbl_grp = QGroupBox("Video Labels")
        vl_g = QGridLayout(vid_lbl_grp)
        vl_g.setSpacing(4)
        vl_g.addWidget(_muted("Subject:"), 0, 0)
        self.entry_vid_subject = QLineEdit()
        self.entry_vid_subject.setPlaceholderText(
            "e.g. Row 5 kochia infestation")
        vl_g.addWidget(self.entry_vid_subject, 0, 1)
        vl_g.addWidget(_muted("Notes:"), 1, 0)
        self.entry_vid_notes = QLineEdit()
        self.entry_vid_notes.setPlaceholderText(
            "e.g. overcast, 10:30am, post-irrigation")
        vl_g.addWidget(self.entry_vid_notes, 1, 1)
        lay.addWidget(vid_lbl_grp)

        self.lbl_vid_status = QLabel("Not recording")
        theme_manager.register_widget(
            self.lbl_vid_status, lambda p: (
                f"color:{p['muted']};font-size:10px;"
                f"font-family:'Noto Sans',Arial,sans-serif;"))
        lay.addWidget(self.lbl_vid_status)

        vid_btn_row = QHBoxLayout()
        self.btn_vid_start = QPushButton("⏺  START REC")
        theme_manager.register_button(self.btn_vid_start, "red")
        self.btn_vid_start.clicked.connect(self._vid_start)
        vid_btn_row.addWidget(self.btn_vid_start)

        self.btn_vid_stop = QPushButton("⏹  STOP REC")
        theme_manager.register_button(self.btn_vid_stop, "dim_red")
        self.btn_vid_stop.setEnabled(False)
        self.btn_vid_stop.clicked.connect(self._vid_stop)
        vid_btn_row.addWidget(self.btn_vid_stop)

        lay.addLayout(vid_btn_row)
        lay.addStretch()
        return w

    # ─────────────────────────────────────────────────────────
    #  CAPTURE LOGIC
    # ─────────────────────────────────────────────────────────

    def _get_capture_source(self) -> str:
        """Return SRC_BOTH, SRC_LEFT, or SRC_RIGHT."""
        for btn in self._src_group.buttons():
            if btn.isChecked():
                return btn.text()
        return SRC_BOTH

    def _get_labels(self) -> dict:
        return {
            "crop_type":       self.cmb_crop.currentText(),
            "target":          self.cmb_target.currentText(),
            "growth_stage":    self.cmb_stage.currentText(),
            "disease_present": self.chk_disease.isChecked(),
            "disease_type":    self.cmb_disease_type.currentText(),
            "weed_type":       self.cmb_weed_type.currentText(),
            "notes":           self.entry_notes.text().strip(),
            "capture_source":  self._get_capture_source(),
        }

    def _get_session_dir(self, labels: dict) -> Path:
        """
        Return (and create) the session directory.
        Session changes when labels or date changes.
        """
        crop   = re.sub(r"[^a-z0-9_]", "_",
                        labels["crop_type"].lower())
        stage  = re.sub(r"[^a-z0-9_]", "_",
                        labels["growth_stage"].lower())
        target = re.sub(r"[^a-z0-9_]", "_",
                        labels["target"].lower())
        date   = datetime.now().strftime("%Y%m%d")
        key    = f"{crop}_{target}_{stage}_{date}"

        # Return cached path only if key matches AND folder still exists on disk.
        # User may have deleted it — always recheck.
        if (self._session_labels == key
                and self._session_path is not None
                and self._session_path.exists()
                and (self._session_path / "metadata").exists()):
            return self._session_path

        # (Re)create session directories
        base = Path(self.lbl_folder.text()) / key
        (base / "left").mkdir(parents=True, exist_ok=True)
        (base / "right").mkdir(parents=True, exist_ok=True)
        (base / "metadata").mkdir(parents=True, exist_ok=True)

        # Reset capture counter only for a genuinely new label key
        is_new_key = self._session_labels != key
        if is_new_key:
            self._capture_count = 0
            self._session_id    = key

        self._session_path   = base
        self._session_labels = key

        # Write/append session log
        log_path = base / "session.log"
        with open(log_path, "a") as f:
            reason = "started" if is_new_key else "recreated (folder was deleted)"
            f.write(
                f"\n=== Session {reason}: "
                f"{datetime.now().isoformat()} ===\n"
                f"Labels: {json.dumps(labels, indent=2)}\n"
            )

        action = "New session" if is_new_key else "Session folder recreated"
        self.shared_log.log("CAMERA", f"{action}: {key}", "info")
        return base

    def _save_pair(self, left, right, labels: dict,
                   session_dir: Path) -> dict:
        """
        Save left.jpg and/or right.jpg based on capture source.
        Write metadata JSON. Return metadata dict.
        """
        src    = self._get_capture_source()
        cid    = f"{self._session_id}_{self._capture_count:04d}"
        ts     = datetime.now().isoformat()
        saved  = []
        meta   = {
            "capture_id":    cid,
            "timestamp":     ts,
            "labels":        labels,
            "capture_source": src,
            "camera_model":  "eMeet C960 4K (Dual)",
            "resolution":    "1920x1080",
        }

        # Ensure subdirs exist — guards against partial deletion
        (session_dir / "left").mkdir(parents=True, exist_ok=True)
        (session_dir / "right").mkdir(parents=True, exist_ok=True)
        (session_dir / "metadata").mkdir(parents=True, exist_ok=True)

        if src in (SRC_BOTH, SRC_LEFT) and left is not None:
            p = session_dir / "left" / f"{cid}_left.jpg"
            cv2.imwrite(str(p), left,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            meta["left_image"] = p.name
            saved.append("left")

        if src in (SRC_BOTH, SRC_RIGHT) and right is not None:
            p = session_dir / "right" / f"{cid}_right.jpg"
            cv2.imwrite(str(p), right,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            meta["right_image"] = p.name
            saved.append("right")

        # Write metadata JSON
        mp = session_dir / "metadata" / f"{cid}_meta.json"
        with open(mp, "w") as f:
            json.dump(meta, f, indent=2)

        # Append to session log
        lp = session_dir / "session.log"
        with open(lp, "a") as f:
            f.write(
                f"{ts}  [{cid}]  "
                f"saved={saved}  "
                f"target={labels['target']}  "
                f"stage={labels['growth_stage']}\n"
            )

        return meta

    # ── Single image capture ──────────────────────────────────

    def _capture_image(self):
        frame, bands = self.camera.get_frame_snapshot()
        if frame is None:
            self.shared_log.log(
                "CAMERA", "No frame available", "warn")
            return

        labels      = self._get_labels()
        session_dir = self._get_session_dir(labels)
        left        = bands.get("left")
        right       = bands.get("right")

        meta = self._save_pair(left, right, labels, session_dir)
        self._capture_count += 1
        self._img_count     += 1

        self.lbl_img_count.setText(
            f"Captures this session: {self._img_count}")
        self.lbl_last_saved.setText(
            f"Saved: {meta['capture_id']}")

        src = self._get_capture_source()
        self.shared_log.log(
            "CAMERA",
            f"Captured {meta['capture_id']}  [{src}]",
            "ok")

    # ── Auto capture ──────────────────────────────────────────

    def _auto_start(self):
        labels      = self._get_labels()
        session_dir = self._get_session_dir(labels)
        self._auto_count = 0
        self._auto_max   = self.spn_max_cap.value()
        interval_ms      = int(self.spn_interval.value() * 1000)

        def _do():
            if self._auto_count >= self._auto_max:
                self._auto_stop()
                self.shared_log.log(
                    "CAMERA",
                    f"Auto complete: {self._auto_count}/"
                    f"{self._auto_max}", "ok")
                return

            frame, bands = self.camera.get_frame_snapshot()
            if frame is None:
                return

            lbl = self._get_labels()
            lbl["auto_n"] = self._auto_count
            self._save_pair(
                bands.get("left"), bands.get("right"),
                lbl, session_dir)

            self._capture_count += 1
            self._auto_count    += 1
            pct = int(self._auto_count / self._auto_max * 100)
            self.auto_progress.setValue(pct)
            self.auto_progress.setFormat(
                f"{self._auto_count}/{self._auto_max}")

        self._auto_timer = QTimer()
        self._auto_timer.timeout.connect(_do)
        self._auto_timer.start(interval_ms)

        self.btn_auto_start.setEnabled(False)
        self.btn_auto_pause.setEnabled(True)
        self.btn_auto_stop.setEnabled(True)
        self.shared_log.log(
            "CAMERA",
            f"Auto started: {self._auto_max} captures "
            f"@ {self.spn_interval.value()}s  "
            f"[{self._get_capture_source()}]",
            "info")

    def _auto_pause(self):
        if self._auto_timer:
            if self._auto_timer.isActive():
                self._auto_timer.stop()
                self.shared_log.log(
                    "CAMERA", "Auto paused", "info")
            else:
                self._auto_timer.start()
                self.shared_log.log(
                    "CAMERA", "Auto resumed", "info")

    def _auto_stop(self):
        if self._auto_timer:
            self._auto_timer.stop()
            self._auto_timer = None
        self.btn_auto_start.setEnabled(True)
        self.btn_auto_pause.setEnabled(False)
        self.btn_auto_stop.setEnabled(False)
        self.auto_progress.setValue(0)
        self.auto_progress.setFormat("Ready")

    # ── Video recording ───────────────────────────────────────

    def _vid_start(self):
        frame, bands = self.camera.get_frame_snapshot()
        if frame is None:
            self.shared_log.log(
                "CAMERA", "No frame", "warn")
            return

        labels      = self._get_labels()
        session_dir = self._get_session_dir(labels)
        vid_dir     = session_dir / "video"
        vid_dir.mkdir(parents=True, exist_ok=True)

        ts      = datetime.now().strftime("%H%M%S")
        fourcc  = cv2.VideoWriter_fourcc(
            *self.cmb_vid_codec.currentText())
        fps     = self.spn_vid_fps.value()
        h, w    = frame.shape[:2]

        self._vid_writers = {}

        if self.chk_rec_left.isChecked():
            p = vid_dir / f"left_{ts}.mp4"
            self._vid_writers["left"] = cv2.VideoWriter(
                str(p), fourcc, fps, (w, h))

        if self.chk_rec_right.isChecked():
            p = vid_dir / f"right_{ts}.mp4"
            self._vid_writers["right"] = cv2.VideoWriter(
                str(p), fourcc, fps, (w, h))

        if not self._vid_writers:
            self.shared_log.log(
                "CAMERA",
                "No camera selected for recording", "warn")
            return

        self._recording = True
        self.camera.on_frame_ready = self._write_video_frame
        self.btn_vid_start.setEnabled(False)
        self.btn_vid_stop.setEnabled(True)
        self.lbl_vid_status.setText(
            f"Recording  —  {ts}  [{', '.join(self._vid_writers)}]")
        theme_manager.register_widget(
            self.lbl_vid_status, lambda p: (
                f"color:{p['red']};font-size:10px;"
                f"font-family:'Noto Sans',Arial,sans-serif;font-weight:bold;"))
        self.shared_log.log(
            "CAMERA",
            f"Video recording started: {list(self._vid_writers)}",
            "ok")

    def _write_video_frame(self, frame, bands):
        """on_frame_ready callback during recording."""
        if not self._recording:
            return
        if "left" in self._vid_writers:
            left = bands.get("left")
            if left is not None:
                self._vid_writers["left"].write(left)
        if "right" in self._vid_writers:
            right = bands.get("right")
            if right is not None:
                self._vid_writers["right"].write(right)

    def _vid_stop(self):
        if not self._recording:
            return
        self.camera.on_frame_ready = None
        self._recording = False

        for key, writer in self._vid_writers.items():
            try:
                writer.release()
            except Exception:
                pass
        self._vid_writers = {}

        self.btn_vid_start.setEnabled(True)
        self.btn_vid_stop.setEnabled(False)
        self.lbl_vid_status.setText("Not recording")
        theme_manager.register_widget(
            self.lbl_vid_status, lambda p: (
                f"color:{p['muted']};font-size:10px;"
                f"font-family:'Noto Sans',Arial,sans-serif;"))
        self.shared_log.log(
            "CAMERA", "Video recording stopped", "ok")

    # ─────────────────────────────────────────────────────────
    #  HELPERS
    # ─────────────────────────────────────────────────────────

    def _sync_left_to_right(self):
        """Copy all LEFT settings values → RIGHT widget."""
        r = self.right_settings
        l = self.left_settings

        r.chk_autofocus.setChecked(l.chk_autofocus.isChecked())
        r.spn_focus.setValue(l.spn_focus.value())
        r.chk_auto_exp.setChecked(l.chk_auto_exp.isChecked())
        r.spn_exposure.setValue(l.spn_exposure.value())
        r.chk_auto_wb.setChecked(l.chk_auto_wb.isChecked())
        r.spn_wb_temp.setValue(l.spn_wb_temp.value())
        r.spn_brightness.setValue(l.spn_brightness.value())
        r.spn_contrast.setValue(l.spn_contrast.value())
        r.spn_saturation.setValue(l.spn_saturation.value())
        r.spn_hue.setValue(l.spn_hue.value())
        r.spn_gamma.setValue(l.spn_gamma.value())
        r.spn_gain.setValue(l.spn_gain.value())
        r.spn_sharpness.setValue(l.spn_sharpness.value())
        r.spn_backlight.setValue(l.spn_backlight.value())
        r.cmb_freq.setCurrentIndex(l.cmb_freq.currentIndex())

        self.shared_log.log(
            "CAMERA",
            "LEFT settings copied to RIGHT (not yet applied)",
            "info")

    def _refresh_device_status(self):
        """Update device path labels."""
        self.lbl_left_dev.setText(
            LEFT_DEVICE.split("/")[-1][:48])
        self.lbl_right_dev.setText(
            RIGHT_DEVICE.split("/")[-1][:48])

    def _on_disease_toggle(self, state):
        self.cmb_disease_type.setEnabled(state == Qt.Checked)
        if state != Qt.Checked:
            self.cmb_disease_type.setCurrentIndex(0)

    def _browse_folder(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Save Folder", str(BASE_PATH))
        if path:
            self.lbl_folder.setText(path)

    # ── Public API (called by tab_collection / tab_detection) ─

    def enable_camera_controls(self, enabled: bool):
        """
        Called after camera connects.
        For RGB the camera settings are always accessible
        (v4l2 doesn't require the camera to be streaming).
        We do a one-time settings refresh when enabled.
        """
        if enabled:
            self.left_settings._refresh_all()
            self.right_settings._refresh_all()

    def reset_session(self):
        """Force new session on next capture."""
        self._session_path   = None
        self._session_labels = None
        self._capture_count  = 0

    # ── Cleanup ───────────────────────────────────────────────

    def cleanup(self):
        self._info_timer.stop()
        self._auto_stop()
        if self._recording:
            self._vid_stop()