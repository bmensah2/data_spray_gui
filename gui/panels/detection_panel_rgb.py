"""
detection_panel_rgb.py
eMeet Dual RGB Detection System — Detection Panel

Fork of detection_panel.py adapted for the dual RGB camera system.

Key changes from multispectral version:
  - Imports from detection_config_rgb / detection_engine_rgb /
    zone_manager_rgb instead of multispectral equivalents
  - ModelInputFormat.BANDS_4CH removed — RGB uses standard BGR
  - Input selector replaced with camera source selector
    (Both / Left / Right)
  - _build_cfg() builds RGBConfig, not ABENConfig
  - _run_inference() receives band_data = {"left":arr,"right":arr}
    and runs DualInferenceResult through ZoneManagerRGB
  - _draw_overlay() uses calibrated dual-camera zone geometry
    (B1_SPLIT_X, B2_SPLIT_X) instead of hardcoded 512px band coords
  - DistanceBufferedZone logic identical — camera-agnostic

Everything else identical:
  - Zone LEDs (A/B/C), Nozzle LEDs (N1/N2/N3)
  - FPS / inference stats, spray event log
  - Manual purge, E-STOP, ARM/STOP UI
  - ActuationController, EventLogger, ROSBridge wiring

Author : Nana | NDSU / PhD Imaging System
Path   : /media/pagsun/Transcend/phd_project/emeet_dual_cam/
"""

import time
import logging
import math as _math
import datetime
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QGroupBox, QSpinBox,
    QComboBox, QLineEdit, QFileDialog, QScrollArea,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal

from gui.style import (
    LED, BTN_BLUE, BTN_GREEN, BTN_RED, BTN_AMBER,
    BTN_TEAL, BTN_DIM_GREEN, BTN_DIM_RED,
    _divider, _muted, _sec, _scroll, _btn
)
from gui.shared_log import UnifiedLog

# ── RGB detection pipeline ────────────────────────────────────
try:
    from core.detection_config_rgb import (
        RGBConfig, DetectionMode, GrowthStage,
        get_weed_config, get_cls_config,
    )
    from core.detection_engine_rgb import (
        RGBDetectionEngine, DualInferenceResult
    )
    from core.zone_manager_rgb import (
        ZoneManagerRGB, ZoneDecision, ZoneState,
        ZONE_A, ZONE_B1, ZONE_B2, ZONE_C
    )
    from core.actuation_controller import ActuationController
    from core.event_logger import EventLogger
    from core.ros_bridge import ROSBridge
    DETECTION_AVAILABLE = True
except ImportError as _e:
    DETECTION_AVAILABLE = False
    print(f"⚠  RGB Detection modules not found: {_e}")


# ─────────────────────────────────────────────────────────────
#  DISTANCE-BUFFERED ZONE CONTROLLER  (identical to multispec)
# ─────────────────────────────────────────────────────────────

class DistanceBufferedZone:
    """
    Per-nozzle spray trigger using exact plant-to-nozzle geometry.

    Replaces the fixed 16-inch look-ahead with per-detection
    distance calculation using GeometryConfig:

      trigger_dist = (nozzle_y_px - det_cy) × GSD_m
      spray_time   = plant_width_m / robot_speed

    Each detection carries its own trigger distance and spray
    duration derived from its pixel position and size.

    Pending dict structure:
      {
        'x': float,          robot x at detection time
        'y': float,          robot y at detection time
        'trigger_dist': float,  meters to travel before firing
        'spray_time_s': float,  seconds to hold nozzle open
        'fire_time':  None | float,  time.time() when nozzle opened
      }
    """
    def __init__(self, min_speed_mps: float = 0.05):
        self.min_speed    = min_speed_mps
        self.pending      = []
        self.spray_active = False
        self._spray_end   = 0.0   # time.time() when current spray ends

    def reset(self):
        self.pending      = []
        self.spray_active = False
        self._spray_end   = 0.0

    def add_detection(self, pose: dict, trigger_dist_m: float,
                      spray_time_s: float):
        """
        Register one detection with its computed trigger distance
        and spray duration.
        Called from the inference loop when a weed is confirmed
        in this zone.
        """
        self.pending.append({
            'x':            pose['x'],
            'y':            pose['y'],
            'trigger_dist': trigger_dist_m,
            'spray_time_s': spray_time_s,
            'fired':        False,
        })

    def update(self, has_detection: bool,
               pose: dict, speed: float,
               manual_purge: bool = False,
               # Legacy params — kept for compatibility
               trigger_dist_m: float = 0.0,
               spray_time_s:   float = 0.5) -> bool:
        import time as _time

        if manual_purge:
            self.spray_active = True
            return True

        now = _time.time()

        # Check if current spray window has expired
        if self.spray_active and now >= self._spray_end:
            self.spray_active = False

        if speed < self.min_speed and not self.spray_active:
            return self.spray_active

        # Add new detection with provided geometry
        if has_detection and trigger_dist_m >= 0:
            self.pending.append({
                'x':            pose['x'],
                'y':            pose['y'],
                'trigger_dist': trigger_dist_m,
                'spray_time_s': spray_time_s,
                'fired':        False,
            })

        # Check all pending detections
        keep = []
        for p in self.pending:
            if p['fired']:
                continue
            dist = _math.sqrt(
                (pose['x'] - p['x'])**2 +
                (pose['y'] - p['y'])**2
            )
            if dist >= p['trigger_dist']:
                # Fire — open nozzle for spray_time_s
                if not self.spray_active:
                    self.spray_active = True
                    self._spray_end   = now + p['spray_time_s']
                p['fired'] = True
            else:
                # Not yet reached — keep in queue
                # Discard if robot has traveled way past (stale)
                if dist < p['trigger_dist'] + 0.5:
                    keep.append(p)

        self.pending = keep
        return self.spray_active


# ─────────────────────────────────────────────────────────────
#  DETECTION PANEL RGB
# ─────────────────────────────────────────────────────────────

class DetectionPanelRGB(QWidget):
    """
    Full detection pipeline panel for Tab 2 (RGB system).
    Drop-in replacement for DetectionPanel.
    """

    overlay_ready = pyqtSignal(object)

    def __init__(self, shared_log: UnifiedLog,
                 camera,
                 gantry_ctrl_ref,
                 parent=None):
        super().__init__(parent)
        self.shared_log      = shared_log
        self.camera          = camera
        self.gantry_ctrl_ref = gantry_ctrl_ref

        # Detection state
        self._armed       = False
        self._cfg         = None
        self._engine      = None
        self._zones       = None
        self._actuation   = None
        self._logger      = None
        self._odom        = None
        self._fps         = 0.0
        self._last_t      = 0.0
        self._events      = 0
        # 3 distance-buffered zones: N1, N2, N3
        self._dist_zones  = [DistanceBufferedZone() for _ in range(3)]
        self._purge            = False
        self._prev_spray       = [False, False, False]
        self._session_log_path = None
        # Pump state
        self._pump_enabled     = False   # manual toggle gate
        self._prime_timer      = None    # QTimer for prime duration
        # Static test mode — bypasses min_speed check for bench testing
        self._static_test      = False

        self._build_ui()

    # ── Build UI ──────────────────────────────────────────────

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea{border:none;}")

        inner = QWidget()
        lay   = QVBoxLayout(inner)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        if not DETECTION_AVAILABLE:
            lbl = QLabel(
                "⚠ RGB Detection modules not found.\n"
                "Ensure detection_config_rgb.py, detection_engine_rgb.py,\n"
                "zone_manager_rgb.py are in core/."
            )
            lbl.setStyleSheet(
                "color:#f5a623;font-family:Courier New;"
                "font-size:10px;")
            lbl.setWordWrap(True)
            lay.addWidget(lbl)
        else:
            lay.addWidget(self._config_grp())
            lay.addWidget(self._model_grp())
            lay.addWidget(self._zones_grp())
            lay.addWidget(self._stats_grp())
            lay.addWidget(self._event_grp())
            lay.addWidget(self._pump_grp())
            lay.addWidget(self._purge_grp())

        lay.addStretch()
        scroll.setWidget(inner)
        outer.addWidget(scroll)

    def _config_grp(self):
        grp = QGroupBox("Configuration")
        g   = QGridLayout(grp)
        g.setSpacing(6)

        g.addWidget(_muted("Mode:"), 0, 0)
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["WEED — herbicide", "CLS — fungicide"])
        g.addWidget(self.cmb_mode, 0, 1)

        # Camera source — replaces multispec "Input format" selector
        g.addWidget(_muted("Camera:"), 0, 2)
        self.cmb_cam_src = QComboBox()
        self.cmb_cam_src.addItems(["Both", "Left only", "Right only"])
        g.addWidget(self.cmb_cam_src, 0, 3)

        g.addWidget(_muted("Threshold:"), 1, 0)
        self.spn_threshold = QSpinBox()
        self.spn_threshold.setRange(1, 20)
        self.spn_threshold.setValue(4)
        g.addWidget(self.spn_threshold, 1, 1)

        g.addWidget(_muted("Field ID:"), 1, 2)
        self.entry_field = QLineEdit()
        self.entry_field.setPlaceholderText("e.g. Field_01")
        g.addWidget(self.entry_field, 1, 3)

        # Static test mode row
        from PyQt5.QtWidgets import QCheckBox as _QCB
        g.addWidget(_muted("Static test:"), 2, 0)
        self.chk_static = _QCB("Bypass speed check (bench/indoor test)")
        self.chk_static.setStyleSheet(
            "color:#f5a623;font-family:Courier New;font-size:9px;")
        self.chk_static.setChecked(False)
        self.chk_static.stateChanged.connect(self._on_static_toggle)
        g.addWidget(self.chk_static, 2, 1, 1, 3)

        return grp

    def _model_grp(self):
        grp = QGroupBox("Model")
        lay = QHBoxLayout(grp)

        self.lbl_model = QLabel("No model loaded  (stub mode)")
        self.lbl_model.setStyleSheet(
            "color:#f5a623;font-family:Courier New;font-size:10px;")
        lay.addWidget(self.lbl_model, stretch=1)

        self.btn_load = QPushButton("Load Model")
        self.btn_load.setStyleSheet(BTN_BLUE)
        self.btn_load.setFixedWidth(110)
        self.btn_load.clicked.connect(self._load_model)
        lay.addWidget(self.btn_load)
        return grp

    def _zones_grp(self):
        grp = QGroupBox("Spray Zones  ·  Nozzles")
        lay = QVBoxLayout(grp)

        # Zone A / B / C LEDs
        zone_row = QHBoxLayout()
        self._zone_widgets = []
        for name, color in [
            ("Zone A", "#00c896"),
            ("Zone B", "#f5a623"),
            ("Zone C", "#4a9eff"),
        ]:
            zw  = QGroupBox(name)
            zw.setStyleSheet(
                f"QGroupBox{{border:1px solid {color};"
                f"border-radius:4px;margin-top:8px;"
                f"color:{color};font-size:9px;}}")
            zl  = QVBoxLayout(zw)
            zl.setSpacing(2)
            zl.setContentsMargins(4, 8, 4, 4)
            led = LED(16)
            cnt = QLabel("0/4")
            cnt.setAlignment(Qt.AlignCenter)
            cnt.setStyleSheet(
                "color:#8090a8;font-size:10px;"
                "font-family:Courier New;")
            dl  = QLabel("--")
            dl.setAlignment(Qt.AlignCenter)
            dl.setStyleSheet(
                "color:#6070a0;font-size:9px;"
                "font-family:Courier New;")
            dl.setWordWrap(True)
            zl.addWidget(led, alignment=Qt.AlignCenter)
            zl.addWidget(cnt)
            zl.addWidget(dl)
            zone_row.addWidget(zw)
            self._zone_widgets.append((led, cnt, dl))
        lay.addLayout(zone_row)

        # Nozzle LEDs
        noz_row = QHBoxLayout()
        self._nozzle_leds = []
        for i in range(3):
            nl = LED(14)
            lb = QLabel(f"N{i+1}")
            lb.setStyleSheet(
                "color:#8090a8;font-size:11px;"
                "font-family:Courier New;")
            noz_row.addWidget(lb)
            noz_row.addWidget(nl)
            self._nozzle_leds.append(nl)
        noz_row.addStretch()
        lay.addLayout(noz_row)
        return grp

    def _stats_grp(self):
        grp = QGroupBox("Statistics")
        g   = QGridLayout(grp)
        g.setSpacing(4)
        self.lbl_fps    = _muted("FPS: --")
        self.lbl_inf    = _muted("Inference: --ms")
        self.lbl_events = _muted("Spray events: 0")
        self.lbl_pose   = _muted("Pose: --")
        g.addWidget(self.lbl_fps,    0, 0)
        g.addWidget(self.lbl_inf,    0, 1)
        g.addWidget(self.lbl_events, 1, 0)
        g.addWidget(self.lbl_pose,   1, 1)
        return grp

    def _event_grp(self):
        grp = QGroupBox("Last Spray Event")
        lay = QVBoxLayout(grp)
        self.lbl_last_event = QLabel("No spray events yet")
        self.lbl_last_event.setStyleSheet(
            "color:#8090a8;font-size:10px;"
            "font-family:Courier New;")
        self.lbl_last_event.setWordWrap(True)
        lay.addWidget(self.lbl_last_event)
        return grp

    def _pump_grp(self):
        """
        Pump enable toggle + prime button.

        Pump ON/OFF toggle: safety gate — pump must be manually
        enabled before auto-spray can begin. When toggled OFF,
        any active spray is cancelled and the pump stops.

        Prime button: runs pump for a set duration (default 3s)
        to pressurize lines before a field run.
        """
        from PyQt5.QtWidgets import QSpinBox
        grp = QGroupBox("Pump Control")
        grp.setStyleSheet(
            "QGroupBox{border:1px solid #3a4055;border-radius:4px;"
            "margin-top:8px;color:#a0a8b8;font-size:10px;font-weight:bold;}"
            "QGroupBox::title{subcontrol-origin:margin;padding:0 4px;}")
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)
        lay.setContentsMargins(8, 10, 8, 8)

        # ── Pump enable toggle ────────────────────────────────
        toggle_row = QHBoxLayout()
        self.led_pump = LED(14)
        self.led_pump.set_state(False)
        toggle_row.addWidget(self.led_pump)

        self.lbl_pump_state = QLabel("PUMP  DISABLED")
        self.lbl_pump_state.setStyleSheet(
            "color:#e84545;font-family:Courier New;"
            "font-size:10px;font-weight:bold;")
        toggle_row.addWidget(self.lbl_pump_state)
        toggle_row.addStretch()

        self.btn_pump_enable = QPushButton("⚡  ENABLE PUMP")
        self.btn_pump_enable.setStyleSheet(BTN_GREEN)
        self.btn_pump_enable.setMinimumHeight(28)
        self.btn_pump_enable.setMinimumWidth(130)
        self.btn_pump_enable.clicked.connect(self._pump_toggle)
        toggle_row.addWidget(self.btn_pump_enable)
        lay.addLayout(toggle_row)

        lay.addWidget(_muted(
            "Pump must be ENABLED before auto-spray will fire"))

        lay.addWidget(_divider())

        # ── Prime button ──────────────────────────────────────
        prime_row = QHBoxLayout()
        prime_row.addWidget(_muted("Prime duration (s):"))

        self.spn_prime = QSpinBox()
        self.spn_prime.setRange(1, 30)
        self.spn_prime.setValue(3)
        self.spn_prime.setFixedWidth(60)
        self.spn_prime.setStyleSheet(
            "QSpinBox{background:#1a1e2e;color:#e8eaf0;"
            "border:1px solid #3a4055;border-radius:3px;"
            "font-family:Courier New;font-size:10px;padding:2px;}")
        prime_row.addWidget(self.spn_prime)
        prime_row.addStretch()

        self.btn_prime = QPushButton("💧  PRIME PUMP")
        self.btn_prime.setStyleSheet(BTN_BLUE)
        self.btn_prime.setMinimumHeight(28)
        self.btn_prime.setMinimumWidth(120)
        self.btn_prime.clicked.connect(self._prime_pump)
        prime_row.addWidget(self.btn_prime)
        lay.addLayout(prime_row)

        self.lbl_prime_status = QLabel("")
        self.lbl_prime_status.setStyleSheet(
            "color:#4a9eff;font-size:9px;font-family:Courier New;")
        lay.addWidget(self.lbl_prime_status)

        return grp

    def _purge_grp(self):
        grp = QGroupBox("Manual Purge / Nozzle Test")
        lay = QVBoxLayout(grp)
        lay.addWidget(_muted(
            "Hold to open all nozzles — robot must be moving"))
        self.btn_purge = QPushButton("⏺  HOLD TO PURGE")
        self.btn_purge.setStyleSheet(
            "QPushButton{background-color:#5a3a00;color:#f5a623;"
            "border:1px solid #f5a623;border-radius:4px;"
            "padding:6px;font-family:'Courier New';"
            "font-size:10px;font-weight:bold;}"
            "QPushButton:pressed{background-color:#f5a623;color:#000;}")
        self.btn_purge.setMinimumHeight(32)
        self.btn_purge.pressed.connect(self._purge_start)
        self.btn_purge.released.connect(self._purge_stop)
        lay.addWidget(self.btn_purge)
        return grp

    # ── Detection control ─────────────────────────────────────

    def _build_cfg(self) -> RGBConfig:
        """Build RGBConfig from current UI state."""
        field = self.entry_field.text().strip() or "gui_run"
        if self.cmb_mode.currentIndex() == 0:
            cfg = get_weed_config(
                field_id=field,
                growth_stage=GrowthStage.SIX_LEAF)
        else:
            cfg = get_cls_config(field_id=field)

        cfg.zones.detection_threshold = self.spn_threshold.value()
        return cfg

    def _det_start(self):
        if not DETECTION_AVAILABLE or self._armed:
            return
        self._armed = True
        cfg         = self._build_cfg()
        self._cfg   = cfg

        self._engine = RGBDetectionEngine(cfg)
        self._zones  = ZoneManagerRGB(cfg)

        # Update model label to reflect actual loaded state
        if self._engine.stub_mode:
            self.lbl_model.setText("No model loaded  (stub mode)")
            self.lbl_model.setStyleSheet(
                "color:#f5a623;font-family:Courier New;font-size:10px;")
        else:
            model_path = cfg.model.get_model_path(cfg.session.detection_mode)
            self.lbl_model.setText(f"✓  {model_path.name}")
            self.lbl_model.setStyleSheet(
                "color:#00c896;font-family:Courier New;font-size:10px;")

        # ROSBridge — receives Husky odometry via UDP.
        # Uses its own NetworkConfig (from detection_config, not RGBConfig).
        # Disabled gracefully if Husky is not connected.
        try:
            from core.detection_config_rgb import NetworkConfig as _NetCfg
            net_cfg          = _NetCfg()
            net_cfg.husky_ip = "192.168.131.1"
            self._odom       = ROSBridge(net_cfg)
            self._odom.start()
            self.shared_log.log("DETECT", "ROSBridge started (Husky odom)", "info")
        except Exception as e:
            self.shared_log.log(
                "DETECT", "ROSBridge disabled — pose unavailable", "info")
            self._odom = None

        session_id = (
            f"gui_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            f"_{cfg.session.detection_mode.value}"
        )

        # EventLogger — RGB session log written directly to project logs/.
        # The shared EventLogger expects ABENConfig.storage; instead we
        # write a lightweight session JSON here.
        try:
            log_dir = cfg.logging.base_dir
            log_dir.mkdir(parents=True, exist_ok=True)
            import json
            session_meta = {
                "session_id":     session_id,
                "mode":           cfg.session.detection_mode.value,
                "growth_stage":   cfg.session.growth_stage.value,
                "field_id":       cfg.session.field_id,
                "operator":       cfg.session.operator,
                "camera":         "eMeet C960 4K (Dual RGB)",
                "B1_SPLIT_X":     cfg.zones.B1_SPLIT_X,
                "B2_SPLIT_X":     cfg.zones.B2_SPLIT_X,
                "threshold":      cfg.zones.detection_threshold,
            }
            meta_path = log_dir / f"{session_id}_session.json"
            with open(meta_path, "w") as f:
                json.dump(session_meta, f, indent=2)
            self._session_log_path = log_dir / f"{session_id}_events.log"
            self._session_log_path.touch()
            self._logger = True   # flag — we handle logging directly
            self.shared_log.log(
                "DETECT", f"Session log: {meta_path.name}", "info")
        except Exception as e:
            self.shared_log.log(
                "DETECT", f"Session log unavailable: {e}", "warn")
            self._logger = None
            self._session_log_path = None

        try:
            gantry      = self.gantry_ctrl_ref()
            gantry_ctrl = (gantry
                           if (gantry and gantry.state.connected)
                           else None)
            # dry_run=False when Arduino is connected → real pump+nozzle commands
            # dry_run=True  when Arduino not connected → log only, no hardware
            hardware_connected = (gantry_ctrl is not None)
            self._actuation = ActuationController(
                cfg,
                gantry=gantry_ctrl,
                on_spray_event=self._on_spray_event
            )
            self._actuation._dry_run = not hardware_connected
            self._actuation.start()

            mode_str = "HARDWARE" if hardware_connected else "DRY RUN"
            self.shared_log.log(
                "DETECT",
                f"ActuationController: {mode_str} "
                f"({'Arduino connected' if hardware_connected else 'Arduino not connected'})",
                "ok" if hardware_connected else "warn")
        except Exception as e:
            self.shared_log.log(
                "DETECT", f"ActuationController: {e}", "warn")
            self._actuation = None

        # Wire camera overlay callback
        self.camera.on_detection_overlay = self._run_inference

        # Log geometry summary
        geo = cfg.geometry
        geo.__post_init__()   # ensure derived values computed
        self.shared_log.log("DETECT", geo.summary(), "info")
        self.shared_log.log(
            "DETECT",
            f"Armed — {cfg.session.detection_mode.value.upper()} "
            f"| threshold={cfg.zones.detection_threshold} "
            f"| stub={self._engine.stub_mode}",
            "ok")

    def _det_stop(self):
        if not self._armed:
            return
        self.camera.on_detection_overlay = None
        self._armed = False

        if self._actuation:
            try:
                self._actuation.stop()
            except Exception:
                pass
        if self._odom:
            try:
                self._odom.stop()
            except Exception:
                pass
        # _logger is True (lightweight) or None — no .stop() needed
        self._logger = None

        if self._zones:
            self._zones.reset()

        for z in self._dist_zones:
            z.reset()
        self._prev_spray = [False, False, False]

        self._engine = None
        self._zones  = None
        self.shared_log.log("DETECT", "Detection stopped", "info")

    def _det_estop(self):
        """Emergency stop — cuts actuation immediately."""
        if self._actuation:
            try:
                self._actuation.emergency_stop()
                self.shared_log.log(
                    "DETECT", "ActuationController E-STOP acknowledged", "error")
            except Exception as e:
                # This used to be a bare `except: pass` -- it silently
                # swallowed the fact that emergency_stop() didn't exist
                # on ActuationController at all, meaning this call site
                # never actually stopped anything for who knows how
                # long. Never swallow an E-STOP failure silently again.
                self.shared_log.log(
                    "DETECT",
                    f"⚠⚠ ActuationController.emergency_stop() FAILED: {e} "
                    f"-- relying on direct gantry E-STOP only",
                    "error")
                logging.error(
                    f"ActuationController.emergency_stop() failed during "
                    f"E-STOP: {e}", exc_info=True)
        else:
            self.shared_log.log(
                "DETECT",
                "⚠ E-STOP pressed but no ActuationController is active "
                "(not armed) -- nothing to stop on this path",
                "warn")
        self.shared_log.log("DETECT", "E-STOP", "error")

    def _load_model(self):
        if not self._armed:
            self.shared_log.log(
                "DETECT", "ARM detection first", "warn")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load RGB model weights", "",
            "Model files (*.pt *.engine)")
        if not path:
            return
        try:
            from pathlib import Path as _P
            self._engine.model = None
            self._engine.ready = False
            self._engine.stub_mode = True
            # Reload via config path update
            self._cfg.model.weed_rgb_pt = _P(path)
            self._engine._load_model()
            if self._engine.ready:
                self.lbl_model.setText(f"Model: {_P(path).name}")
                self.lbl_model.setStyleSheet(
                    "color:#00c896;font-family:Courier New;"
                    "font-size:10px;")
                self.shared_log.log(
                    "DETECT", f"Model loaded: {_P(path).name}", "ok")
            else:
                self.shared_log.log(
                    "DETECT", "Model load failed", "error")
        except Exception as e:
            self.shared_log.log(
                "DETECT", f"Load error: {e}", "error")

    # ── Inference ─────────────────────────────────────────────

    def _run_inference(self, display_img):
        """
        Called by DualCameraPanel.on_detection_overlay each frame.
        Receives combined display BGR image, returns overlay image.
        """
        if not self._armed or not DETECTION_AVAILABLE:
            return display_img
        try:
            import numpy as np, cv2

            # Get latest frame pair from camera band_data
            bands = self.camera.band_data
            if not bands:
                return display_img

            # Build a minimal pair-like object for the engine
            class _Pair:
                left     = bands.get("left")
                right    = bands.get("right")
                frame_id = 0
                left_ts  = time.time()

            if _Pair.left is None:
                return display_img

            # Run dual inference
            dual_result = self._engine.run(_Pair())

            # Update zone manager
            decision = self._zones.update(dual_result)

            # Pose + speed
            pose  = self._odom.get_pose() if self._odom else None
            speed = pose['speed'] if pose else 0.0
            if pose is None:
                pose = {'x': 0.0, 'y': 0.0,
                        'heading': 0.0, 'speed': 0.0}

            # Distance-buffered spray decisions (3 nozzles)
            # Map zone_manager zones to nozzles:
            #   ZONE_A(0)→N1, ZONE_B1(1)+ZONE_B2(2)→N2, ZONE_C(3)→N3
            zone_hits = [False, False, False]  # N1, N2, N3
            for zone in self._zones.zones:
                if zone.spray_active:
                    if zone.nozzle_id == 0:
                        zone_hits[0] = True
                    elif zone.nozzle_id == 1:
                        zone_hits[1] = True
                    elif zone.nozzle_id == 2:
                        zone_hits[2] = True

            # ── Geometry-based spray timing ───────────────────
            geo = self._cfg.geometry   # GeometryConfig
            effective_speed = 0.5 if self._static_test else speed

            # For each zone that has an active detection,
            # compute the exact trigger distance and spray duration
            # from the detection's pixel position and size.
            # Then pass these to DistanceBufferedZone.update().

            # Collect per-nozzle geometry from zone detections
            nozzle_trigger = [None, None, None]   # N1, N2, N3
            nozzle_spray_t = [0.5,  0.5,  0.5]   # default 0.5s

            for zone in self._zones.zones:
                if not zone.current_detections:
                    continue
                nid = zone.nozzle_id
                # Use the highest-confidence weed detection in zone
                weed_dets = [d for d in zone.current_detections
                             if d.class_name != "sugarbeet"]
                if not weed_dets:
                    continue
                det = max(weed_dets, key=lambda d: d.confidence)

                # Scale cy from display back to original 1920×1080
                # det.cy is already in original frame coordinates
                # (RGBDetectionEngine scales boxes back to original)
                trig = geo.trigger_distance_m(det.cy)
                sprt = geo.spray_time_s(
                    det.width, effective_speed)

                # Keep the closest (smallest trigger) per nozzle
                if nozzle_trigger[nid] is None or                         trig < nozzle_trigger[nid]:
                    nozzle_trigger[nid] = trig
                    nozzle_spray_t[nid] = sprt

            # Update each DistanceBufferedZone with computed geometry
            spray_states = []
            for i in range(3):
                trig = nozzle_trigger[i] if nozzle_trigger[i]                        is not None else 0.0
                state = self._dist_zones[i].update(
                    has_detection  = zone_hits[i] and self._pump_enabled,
                    pose           = pose,
                    speed          = effective_speed,
                    manual_purge   = (self._purge and self._pump_enabled),
                    trigger_dist_m = trig,
                    spray_time_s   = nozzle_spray_t[i],
                )
                spray_states.append(state)

            # Actuation
            _prev         = self._prev_spray
            _new_triggers = [i for i, (c, p) in
                             enumerate(zip(spray_states, _prev))
                             if c and not p]
            _new_releases = [i for i, (c, p) in
                             enumerate(zip(spray_states, _prev))
                             if not c and p]

            if self._actuation and (
                    _new_triggers or _new_releases):
                # Build a proper ZoneDecision so ActuationController
                # can access decision.zones[zone_id] without error
                act_decision = ZoneDecision(
                    zones            = self._zones.zones,
                    nozzles_to_fire  = [i for i, s in
                                        enumerate(spray_states) if s],
                    nozzles_to_stop  = [i for i, s in
                                        enumerate(spray_states) if not s],
                    new_triggers     = _new_triggers,
                    new_releases     = _new_releases,
                    total_detections = dual_result.total_detections,
                    frame_id         = dual_result.frame_id,
                    timestamp        = dual_result.timestamp,
                )
                self._actuation.actuate(act_decision, pose=pose)

            self._prev_spray = list(spray_states)

            # FPS
            now          = time.time()
            dt           = max(now - self._last_t, 1e-6)
            self._last_t = now
            self._fps    = 0.9 * self._fps + 0.1 * (1.0 / dt)

            # Update zone LEDs
            # Map zone_manager 4-zone list → 3 GUI zone widgets
            # Widget 0=A, 1=B (B1 OR B2), 2=C
            zone_active = [False, False, False]
            zone_counts = [(0, 4), (0, 4), (0, 4)]
            zone_dets   = [[], [], []]

            for zone in self._zones.zones:
                if zone.zone_id == ZONE_A:
                    wi = 0
                elif zone.zone_id in (ZONE_B1, ZONE_B2):
                    wi = 1
                else:
                    wi = 2
                if zone.spray_active:
                    zone_active[wi] = True
                if zone.current_detections:
                    zone_dets[wi].extend(zone.current_detections)
                zone_counts[wi] = (zone.counter, zone.threshold)

            for i, (led, cnt, dl) in enumerate(self._zone_widgets):
                led.set_state(spray_states[i],
                              color_on="#00c896",
                              color_off="#2a2f3d")
                c, th = zone_counts[i]
                cnt.setText(f"{c}/{th}")
                dets = zone_dets[i]
                if dets:
                    dl.setText(", ".join(
                        d.class_name for d in dets[:2]))
                    dl.setStyleSheet(
                        "color:#00c896;font-size:9px;"
                        "font-family:Courier New;")
                else:
                    dl.setText("--")
                    dl.setStyleSheet(
                        "color:#6070a0;font-size:9px;"
                        "font-family:Courier New;")

            for i, nl in enumerate(self._nozzle_leds):
                nl.set_state(spray_states[i],
                             color_on="#00c896",
                             color_off="#2a2f3d")

            self.lbl_fps.setText(f"FPS: {self._fps:.1f}")
            self.lbl_inf.setText(
                f"Inference: {dual_result.total_ms:.1f}ms")
            self.lbl_events.setText(
                f"Spray events: {self._events}")
            if pose:
                self.lbl_pose.setText(
                    f"pos({pose['x']:.2f},{pose['y']:.2f}) "
                    f"hdg={pose['heading']:.0f}° "
                    f"spd={pose['speed']:.2f}m/s")

            # Draw overlay on display image
            if display_img is not None:
                display_img = self._draw_overlay(
                    display_img, dual_result, spray_states)

        except Exception as e:
            self.shared_log.log(
                "DETECT", f"Inference error: {e}", "error")
        return display_img

    # ── Overlay drawing ───────────────────────────────────────

    def _draw_overlay(self, img, dual_result, spray_states):
        """
        Draw zone boundaries and detection boxes on the
        side-by-side display image.

        Style constants — all in one place for easy tuning:
          FONT        : FONT_HERSHEY_DUPLEX (cleaner, closer to Courier)
          FONT_SCALE  : 0.32  — small, unobtrusive
          FONT_THICK  : 1     — single pixel weight
          BOX_THICK   : 1     — thin detection boxes
          ZONE_THICK  : 1     — thin zone boundary lines (2 when active)
          DASH        : 6/12  — short dashes on nozzle centerline
        """
        import cv2, numpy as np

        if self._cfg is None:
            return img

        # ── Style constants ───────────────────────────────────
        FONT        = cv2.FONT_HERSHEY_DUPLEX   # crisp, close to monospace
        FONT_SCALE  = 0.32                       # small, readable
        FONT_THICK  = 1
        BOX_THICK   = 1                          # thin detection boxes
        ZONE_THICK  = 1                          # thin zone lines

        # Per-class colours
        CLASS_COLORS = {
            "sugarbeet": (100, 255, 100),   # green
            "weed":      (0,   200, 255),   # yellow-cyan
        }
        DEFAULT_DET_COLOR = (0, 220, 255)

        out    = img.copy()
        h, w   = out.shape[:2]
        half_w = w // 2

        zones_cfg = self._cfg.zones
        scale     = half_w / 1920

        # ── Zone boundaries ───────────────────────────────────
        b1_x    = int(zones_cfg.B1_SPLIT_X * scale)
        n1_cx   = int(zones_cfg.n1_center_cam1 * scale)
        n2_cx_l = int(zones_cfg.n2_center_cam1 * scale)

        off = half_w + 4
        b2_x    = int(zones_cfg.B2_SPLIT_X * scale)
        n2_cx_r = int(zones_cfg.n2_center_cam2 * scale)
        n3_cx   = int(zones_cfg.n3_center_cam2 * scale)

        left_zones = [
            (0,    b1_x,  "A",  n1_cx,   0, (0,   200, 100)),
            (b1_x, half_w,"B1", n2_cx_l, 1, (255, 180,   0)),
        ]
        right_zones = [
            (off,        off + b2_x, "B2", off + n2_cx_r, 1, (255, 180, 0)),
            (off + b2_x, w,          "C",  off + n3_cx,   2, (0,  140, 255)),
        ]

        for dx1, dx2, zlbl, noz_cx, noz_idx, color in \
                left_zones + right_zones:
            active = spray_states[noz_idx] if noz_idx < 3 else False

            # Zone boundary — thin line, thicker + brighter when active
            thickness = ZONE_THICK + 1 if active else ZONE_THICK
            cv2.rectangle(out, (dx1, 0), (dx2, h), color, thickness)

            # Zone label — small, bottom of frame
            (tw, th), _ = cv2.getTextSize(zlbl, FONT, FONT_SCALE, FONT_THICK)
            lx = dx1 + max(4, (dx2 - dx1 - tw) // 2)
            cv2.putText(out, zlbl, (lx, h - 6),
                        FONT, FONT_SCALE, color,
                        FONT_THICK, cv2.LINE_AA)

            # Nozzle centerline — short fine dashes
            DASH, GAP = 6, 10
            dash_color = (230, 230, 230) if active else color
            for y_s in range(0, h, DASH + GAP):
                cv2.line(out, (noz_cx, y_s),
                         (noz_cx, min(y_s + DASH, h)),
                         dash_color, 1)

            # Active zone — subtle fill
            if active:
                ovl = out.copy()
                cv2.rectangle(ovl, (dx1, 0), (dx2, h), color, -1)
                cv2.addWeighted(ovl, 0.10, out, 0.90, 0, out)

        # ── Detection boxes ───────────────────────────────────
        def _draw_detection(det, x_offset=0):
            bx1 = x_offset + int(det.x1 * scale)
            by1 = int(det.y1 * h / 1080)
            bx2 = x_offset + int(det.x2 * scale)
            by2 = int(det.y2 * h / 1080)
            col = CLASS_COLORS.get(det.class_name, DEFAULT_DET_COLOR)

            # Thin bounding box
            cv2.rectangle(out, (bx1, by1), (bx2, by2), col, BOX_THICK)

            # Label: "classname conf" — small text, dark background pill
            label = f"{det.class_name} {det.confidence:.2f}"
            (tw, th), bl = cv2.getTextSize(
                label, FONT, FONT_SCALE, FONT_THICK)
            ty = max(by1 - 2, th + 2)
            # Dark backing rectangle for readability
            cv2.rectangle(out,
                          (bx1, ty - th - 2),
                          (bx1 + tw + 4, ty + 1),
                          (15, 15, 15), -1)
            cv2.putText(out, label, (bx1 + 2, ty - 1),
                        FONT, FONT_SCALE, col,
                        FONT_THICK, cv2.LINE_AA)

        for det in dual_result.left.detections:
            _draw_detection(det, x_offset=0)

        for det in dual_result.right.detections:
            _draw_detection(det, x_offset=off)

        # ── HUD (top-left status line) ────────────────────────
        mode = (self._cfg.session.detection_mode.value.upper()
                if self._cfg else "DET")
        stub = " STUB" if (self._engine and self._engine.stub_mode) else ""
        hud  = (f"{mode}{stub}  FPS:{self._fps:.0f}"
                f"  Det:{dual_result.total_detections}"
                f"  Ev:{self._events}")
        (hw, hh), _ = cv2.getTextSize(hud, FONT, FONT_SCALE, FONT_THICK)
        cv2.rectangle(out, (0, 0), (hw + 8, hh + 6), (10, 10, 10), -1)
        cv2.putText(out, hud, (4, hh + 3),
                    FONT, FONT_SCALE, (180, 210, 255),
                    FONT_THICK, cv2.LINE_AA)
        return out

    # ── Callbacks ─────────────────────────────────────────────

    def _on_spray_event(self, event):
        self._events += 1
        names = [d["class_name"] for d in event.detections]
        conf  = max(
            (d["confidence"] for d in event.detections),
            default=0.0)
        ts = datetime.datetime.fromtimestamp(
            event.timestamp).strftime("%H:%M:%S")
        self.lbl_last_event.setText(
            f"{event.zone_name} | {names} | "
            f"conf={conf:.2f} | {ts}")
        self.lbl_last_event.setStyleSheet(
            "color:#00c896;font-size:10px;"
            "font-family:Courier New;")
        self.shared_log.log(
            "DETECT",
            f"Spray: {event.zone_name} | {names} | {conf:.2f}",
            "ok")

    def _on_static_toggle(self, state):
        from PyQt5.QtCore import Qt as _Qt
        self._static_test = (state == _Qt.Checked)
        msg = ("Static test ON — speed check bypassed"
               if self._static_test
               else "Static test OFF — speed check active")
        self.shared_log.log("DETECT", msg,
                            "warn" if self._static_test else "info")

    # ── Pump toggle ───────────────────────────────────────────

    def _pump_toggle(self):
        """Enable or disable the pump safety gate."""
        from PyQt5.QtCore import QTimer as _QTimer
        self._pump_enabled = not self._pump_enabled

        if self._pump_enabled:
            # Send pump on to hardware
            if self._actuation:
                try:
                    self._actuation.manual_pump(True)
                except Exception:
                    pass
            elif self.gantry_ctrl_ref:
                try:
                    g = self.gantry_ctrl_ref()
                    if g and g.state.connected:
                        g.send_command("pump on")
                except Exception:
                    pass

            self.lbl_pump_state.setText("PUMP  ENABLED ●")
            self.lbl_pump_state.setStyleSheet(
                "color:#00c896;font-family:Courier New;"
                "font-size:10px;font-weight:bold;")
            self.led_pump.set_state(True, color_on="#00c896")
            self.btn_pump_enable.setText("⛔  DISABLE PUMP")
            self.btn_pump_enable.setStyleSheet(BTN_RED)
            self.shared_log.log(
                "DETECT", "Pump ENABLED — auto-spray armed", "ok")
        else:
            # Send pump off
            if self._actuation:
                try:
                    self._actuation.manual_pump(False)
                except Exception:
                    pass
            elif self.gantry_ctrl_ref:
                try:
                    g = self.gantry_ctrl_ref()
                    if g and g.state.connected:
                        g.send_command("pump off")
                except Exception:
                    pass

            self.lbl_pump_state.setText("PUMP  DISABLED")
            self.lbl_pump_state.setStyleSheet(
                "color:#e84545;font-family:Courier New;"
                "font-size:10px;font-weight:bold;")
            self.led_pump.set_state(False)
            self.btn_pump_enable.setText("⚡  ENABLE PUMP")
            self.btn_pump_enable.setStyleSheet(BTN_GREEN)
            self.shared_log.log(
                "DETECT", "Pump DISABLED — auto-spray blocked", "warn")

    # ── Prime pump ────────────────────────────────────────────

    def _prime_pump(self):
        """Run pump for prime duration, then stop."""
        from PyQt5.QtCore import QTimer as _QTimer

        duration_s = self.spn_prime.value()

        # Turn pump on
        try:
            g = self.gantry_ctrl_ref()
            if g and g.state.connected:
                g.send_command("pump on")
                self.lbl_prime_status.setText(
                    f"Priming … {duration_s}s")
                self.btn_prime.setEnabled(False)
                self.shared_log.log(
                    "DETECT",
                    f"Priming pump for {duration_s}s", "info")

                def _stop_prime():
                    try:
                        g.send_command("pump off")
                        # Restore pump state to match toggle
                        if not self._pump_enabled:
                            pass  # already off
                        else:
                            g.send_command("pump on")
                    except Exception:
                        pass
                    self.lbl_prime_status.setText("Prime complete ✓")
                    self.btn_prime.setEnabled(True)
                    self.shared_log.log(
                        "DETECT", "Prime complete", "ok")

                self._prime_timer = _QTimer()
                self._prime_timer.setSingleShot(True)
                self._prime_timer.timeout.connect(_stop_prime)
                self._prime_timer.start(duration_s * 1000)
            else:
                self.lbl_prime_status.setText(
                    "Arduino not connected")
                self.shared_log.log(
                    "DETECT", "Prime failed — Arduino not connected",
                    "warn")
        except Exception as e:
            self.lbl_prime_status.setText(f"Prime error: {e}")

    def _purge_start(self):
        self._purge = True
        self.shared_log.log(
            "DETECT", "Purge START — nozzles open", "warn")

    def _purge_stop(self):
        self._purge = False
        for z in self._dist_zones:
            z.reset()
        self.shared_log.log("DETECT", "Purge STOP", "info")

    # ── Public API ────────────────────────────────────────────

    @property
    def is_armed(self) -> bool:
        return self._armed

    @property
    def ros_bridge(self):
        return self._odom

    def emergency_stop(self):
        self._det_estop()
        if self._armed:
            self._det_stop()

    def cleanup(self):
        if self._armed:
            self._det_stop()