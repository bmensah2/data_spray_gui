#!/usr/bin/env python3
"""
detection_config_rgb.py
eMeet Dual RGB Detection System — Centralized Configuration

RGB-only fork of the multispectral detection_config.py.
All multispectral / 4-channel / band-specific config removed.

Zone geometry measured 2026-06-26 using tape measure calibration.
measured in the field (see CALIBRATION NOTE below).

Author : Nana | NDSU / PhD Imaging System
Path   : /media/pagsun/Transcend/phd_project/emeet_dual_cam/
"""

from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from typing import Optional, List, Tuple


# ─────────────────────────────────────────────────────────────
#  ENUMERATIONS
# ─────────────────────────────────────────────────────────────

class DetectionMode(Enum):
    """
    WEED : Herbicide-resistant weed detection → spray herbicide
    CLS  : Cercospora Leaf Spot detection     → spray fungicide + log
    """
    WEED = "weed"
    CLS  = "cls"


class GrowthStage(Enum):
    """Sugarbeet growth stages for session metadata."""
    EMERGENCE      = "emergence"
    COTYLEDON      = "cotyledon"
    TWO_LEAF       = "2_leaf"
    FOUR_LEAF      = "4_leaf"
    SIX_LEAF       = "6_leaf"
    EIGHT_LEAF     = "8_leaf"
    CANOPY_CLOSURE = "canopy_closure"
    VEGETATIVE     = "vegetative"
    MATURITY       = "maturity"


class TargetClass(Enum):
    """Detection target classes — shared with multispectral system."""
    KOCHIA         = "kochia"
    WATERHEMP      = "waterhemp"
    COMMON_RAGWEED = "common_ragweed"
    WILD_OAT       = "wild_oat"
    PIGWEED        = "pigweed"
    UNKNOWN_WEED   = "unknown_weed"
    CLS_INFECTED   = "cls_infected"
    CLS_HEALTHY    = "cls_healthy"
    SUGARBEET      = "sugarbeet"
    SOIL           = "soil"

@dataclass
class NetworkConfig:
    """Minimal compatibility config used by the ROS bridge and GUI."""
    husky_ip: str = "192.168.131.1"
    odom_port: int = 5005
    estop_port: int = 5006
    heartbeat_port: int = 5007

# ─────────────────────────────────────────────────────────────
#  CAMERA CONFIGURATION
# ─────────────────────────────────────────────────────────────

@dataclass
class RGBCameraConfig:
    """
    eMeet SmartCam C960 4K — dual camera settings.
    Device paths use stable symlinks, not /dev/videoX.
    Settings validated on Jetson AGX Orin via v4l2-ctl.
    """
    left_src:  str = (
        "/dev/v4l/by-id/"
        "usb-EMEET_EMEET_SmartCam_C960_4K_A241213000400860-video-index0"
    )
    right_src: str = (
        "/dev/v4l/by-id/"
        "usb-EMEET_EMEET_SmartCam_C960_4K_A241217000804000-video-index0"
    )

    width:  int = 1920
    height: int = 1080
    fps:    int = 30

    # v4l2 validated values
    focus_absolute:          int = 460
    exposure_time_absolute:  int = 300
    white_balance_temperature: int = 5000


# ─────────────────────────────────────────────────────────────
#  MODEL CONFIGURATION
# ─────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """
    YOLO model paths and inference settings.
    Standard 3-channel BGR input — no custom yaml needed.
    Model weights will be added after field training.
    """

    # ── Model weights (populated after training) ──────────────
    weed_rgb_pt:     Path = Path("models/weed_rgb.pt")
    weed_rgb_engine: Path = Path("models/weed_rgb.engine")  # TensorRT export
    cls_rgb_pt:      Path = Path("models/cls_rgb.pt")
    cls_rgb_engine:  Path = Path("models/cls_rgb.engine")

    # ── Inference settings ────────────────────────────────────
    confidence_threshold: float = 0.45
    iou_threshold:        float = 0.45
    input_size:           int   = 640    # YOLO inference size (px)
    use_tensorrt:         bool  = True   # use .engine if available
    device:               str   = "cuda:0"

    def get_model_path(self, mode: DetectionMode) -> Path:
        """Return the correct model path, falling back .engine → .pt."""
        if mode == DetectionMode.WEED:
            p = self.weed_rgb_engine if self.use_tensorrt else self.weed_rgb_pt
            fallback = self.weed_rgb_pt
        else:
            p = self.cls_rgb_engine if self.use_tensorrt else self.cls_rgb_pt
            fallback = self.cls_rgb_pt

        if not p.exists():
            return fallback
        return p

    def model_ready(self, mode: DetectionMode) -> bool:
        """Return True if any weights exist for this mode."""
        if mode == DetectionMode.WEED:
            return self.weed_rgb_pt.exists() or self.weed_rgb_engine.exists()
        return self.cls_rgb_pt.exists() or self.cls_rgb_engine.exists()


# ─────────────────────────────────────────────────────────────
#  SPRAY ZONE CONFIGURATION
# ─────────────────────────────────────────────────────────────

@dataclass
class ZoneConfig:
    """
    Spray zone pixel boundaries for the dual eMeet camera setup.

    Physical layout:
    ─────────────────────────────────────────────────────────
    LEFT camera (cam1)           RIGHT camera (cam2)
    ┌─────────────┬──────┐       ┌──────┬─────────────┐
    │   Zone A    │  B1  │       │  B2  │   Zone C    │
    │  Nozzle 1   │  N2  │       │  N2  │  Nozzle 3   │
    └─────────────┴──────┘       └──────┴─────────────┘
    ─────────────────────────────────────────────────────────

    Zone B logic: N2 fires if detection in B1 OR B2.

    ╔══════════════════════════════════════════════════════╗
    ║  CALIBRATION NOTE — PLACEHOLDER VALUES              ║
    ║                                                      ║
    ║  B1/B2 split point has NOT been measured yet.        ║
    ║  Current values are estimates based on 65/35         ║
    ║  split (cam1) and 35/65 split (cam2).                ║
    ║                                                      ║
    ║  To calibrate:                                       ║
    ║    1. Place a marker directly above N2               ║
    ║    2. Run dual_emeet_camera.py                       ║
    ║    3. Note pixel X of marker in cam1 (→ b1_split_x) ║
    ║    4. Note pixel X of marker in cam2 (→ b2_split_x) ║
    ║    5. Update B1_SPLIT_X and B2_SPLIT_X below        ║
    ╚══════════════════════════════════════════════════════╝
    """

    # ── Calibration values — UPDATE AFTER MEASUREMENT ────────
    B1_SPLIT_X: int = 1150   # MEASURED 2026-06-26: midpoint N1(600px) and N2(1700px) in cam1
    B2_SPLIT_X: int = 900    # MEASURED 2026-06-26: midpoint N2(400px) and N3(1400px) in cam2

    # ── cam1 zones (left camera) ──────────────────────────────
    # Derived from B1_SPLIT_X at runtime — see RGBConfig.build_zones()
    # Zone A  : (0, 0, B1_SPLIT_X, 1080)   → Nozzle 1
    # Zone B1 : (B1_SPLIT_X, 0, 1920, 1080) → Nozzle 2 (shared with B2)

    # ── cam2 zones (right camera) ─────────────────────────────
    # Zone B2 : (0, 0, B2_SPLIT_X, 1080)    → Nozzle 2 (shared with B1)
    # Zone C  : (B2_SPLIT_X, 0, 1920, 1080) → Nozzle 3

    # Nozzle ID map: zone index → nozzle (0-indexed)
    # ZoneA=0, ZoneB1=1, ZoneB2=2, ZoneC=3
    # B1 and B2 both map to nozzle 1 (N2)
    zone_nozzle_map: List[int] = field(
        default_factory=lambda: [0, 1, 1, 2]
    )

    # Detection filter — frames needed before nozzle fires
    # Validated value from Test_High_Focus.py weedbot
    detection_threshold: int = 4
    drain_rate:          int = 1   # frames decremented per no-detection tick

    # Spray durations (seconds)
    weed_spray_duration: float = 0.5
    cls_spray_duration:  float = 1.0

    def cam1_zones(self) -> List[Tuple[int, int, int, int]]:
        """Return cam1 zone rects: [ZoneA, ZoneB1]."""
        return [
            (0,              0, self.B1_SPLIT_X, 1080),  # Zone A
            (self.B1_SPLIT_X, 0, 1920,           1080),  # Zone B1
        ]

    def cam2_zones(self) -> List[Tuple[int, int, int, int]]:
        """Return cam2 zone rects: [ZoneB2, ZoneC]."""
        return [
            (0,              0, self.B2_SPLIT_X, 1080),  # Zone B2
            (self.B2_SPLIT_X, 0, 1920,           1080),  # Zone C
        ]


    # ── Nozzle center X — MEASURED 2026-06-26 ───────────────
    # Physical nozzle positions measured with tape measure.
    # Black marker objects placed directly under each nozzle.
    # These are stored as offsets from B1/B2 splits so they
    # auto-scale correctly if the split is ever recalibrated.
    #
    # Measured absolute values:
    #   N1 cam1 = 600px  | N2 cam1 = 1700px
    #   N2 cam2 = 400px  | N3 cam2 = 1400px
    #
    # Stored as offset from split so they remain consistent
    # if B1_SPLIT_X / B2_SPLIT_X are updated.

    # Offsets (measured_center - split_value) — fixed at calibration
    _N1_OFFSET: int = 600  - 1150   # = -550  (N1 is 550px left of B1 split)
    _N2_OFFSET_CAM1: int = 1700 - 1150  # = +550  (N2 is 550px right of B1 split)
    _N2_OFFSET_CAM2: int = 400  - 900   # = -500  (N2 is 500px left of B2 split)
    _N3_OFFSET: int = 1400 - 900    # = +500  (N3 is 500px right of B2 split)

    @property
    def n1_center_cam1(self) -> int:
        """N1 center X in left camera (measured: 600px)."""
        return max(0, self.B1_SPLIT_X + self._N1_OFFSET)

    @property
    def n2_center_cam1(self) -> int:
        """N2 center X in left camera (measured: 1700px)."""
        return min(1919, self.B1_SPLIT_X + self._N2_OFFSET_CAM1)

    @property
    def n2_center_cam2(self) -> int:
        """N2 center X in right camera (measured: 400px)."""
        return max(0, self.B2_SPLIT_X + self._N2_OFFSET_CAM2)

    @property
    def n3_center_cam2(self) -> int:
        """N3 center X in right camera (measured: 1400px)."""
        return min(1919, self.B2_SPLIT_X + self._N3_OFFSET)

    def nozzle_centers(self) -> dict:
        """All nozzle center X values — pass to display/annotation functions."""
        return {
            'N1_cam1': self.n1_center_cam1,
            'N2_cam1': self.n2_center_cam1,
            'N2_cam2': self.n2_center_cam2,
            'N3_cam2': self.n3_center_cam2,
        }

    def spray_duration(self, mode: DetectionMode) -> float:
        return (self.weed_spray_duration
                if mode == DetectionMode.WEED
                else self.cls_spray_duration)


# ─────────────────────────────────────────────────────────────
#  CAMERA GEOMETRY  (calibrated 2026-06-26)
# ─────────────────────────────────────────────────────────────

@dataclass
class GeometryConfig:
    """
    Physical camera-to-nozzle geometry for spray timing.

    Coordinate system:
      - Image Y=0 is TOP of frame (far ahead of robot)
      - Image Y=1079 is BOTTOM of frame (behind nozzle line)
      - Robot moves forward → plants move UP the image (decreasing Y)
      - Nozzle line is BELOW the camera center by look_ahead_px

    Calibrated values (2026-06-26):
      Camera height  : 36.5 in = 0.9271m
      Look-ahead     : 7 in    = 0.1778m  (camera center ahead of nozzles)
      Camera HFOV    : ~78° (eMeet C960 4K)
      FOV width/cam  : 2 × 0.9271 × tan(39°) = 1.501m
      GSD            : 1.501 / 1920 = 0.000782 m/px

    Nozzle Y in image:
      image_center_y = 1080 / 2 = 540px
      look_ahead_px  = 0.1778 / 0.000782 = 227px
      nozzle_y_px    = 540 + 227 = 767px  (below center)

    Spray timing:
      trigger_dist_m = (nozzle_y_px - det_cy) × gsd_m_per_px
      spray_time_s   = spray_dist_m / robot_speed_mps
      spray_dist_m   = max(min_spray_m, plant_width_px × gsd_m_per_px)
    """

    # ── Physical measurements ─────────────────────────────────
    camera_height_m:    float = 0.9271   # 36.5 inches
    look_ahead_m:       float = 0.1778   # 7 inches (camera ahead of nozzles)
    camera_hfov_deg:    float = 78.0     # eMeet C960 4K horizontal FOV

    # ── Derived (computed in __post_init__) ───────────────────
    gsd_m_per_px:       float = 0.0      # ground sample distance m/px
    fov_width_m:        float = 0.0      # full FOV width at ground
    nozzle_y_px:        int   = 0        # nozzle line Y in image
    look_ahead_px:      int   = 0        # look-ahead in pixels

    # ── Image dimensions ──────────────────────────────────────
    image_width:        int   = 1920
    image_height:       int   = 1080

    # ── Spray window ─────────────────────────────────────────
    min_spray_dist_m:   float = 0.05     # minimum 5cm spray window
    max_spray_dist_m:   float = 0.30     # maximum 30cm spray window
    min_speed_mps:      float = 0.05     # minimum robot speed to spray

    def __post_init__(self):
        import math
        half_fov_rad     = math.radians(self.camera_hfov_deg / 2)
        self.fov_width_m = 2 * self.camera_height_m * math.tan(half_fov_rad)
        self.gsd_m_per_px = self.fov_width_m / self.image_width
        self.look_ahead_px = int(self.look_ahead_m / self.gsd_m_per_px)
        self.nozzle_y_px   = (self.image_height // 2) + self.look_ahead_px

    def trigger_distance_m(self, det_cy_px: int) -> float:
        """
        Distance robot must travel before nozzle fires.
        det_cy_px: Y pixel of detection center in original 1920×1080 frame.

        If plant is above nozzle line (cy < nozzle_y_px):
          → robot needs to travel that distance
        If plant is at or below nozzle line (cy >= nozzle_y_px):
          → fire immediately (distance = 0)
        """
        dist = (self.nozzle_y_px - det_cy_px) * self.gsd_m_per_px
        return max(0.0, dist)

    def spray_time_s(self, plant_width_px: int,
                     robot_speed_mps: float) -> float:
        """
        Duration to keep nozzle open.
        Based on plant width in pixels converted to meters,
        divided by robot speed.
        """
        if robot_speed_mps < self.min_speed_mps:
            robot_speed_mps = 0.3   # fallback default speed
        plant_w_m = plant_width_px * self.gsd_m_per_px
        spray_m   = max(self.min_spray_dist_m,
                        min(plant_w_m, self.max_spray_dist_m))
        return spray_m / robot_speed_mps

    def summary(self) -> str:
        return (
            f"GeometryConfig: height={self.camera_height_m:.4f}m "
            f"look_ahead={self.look_ahead_m:.4f}m "
            f"GSD={self.gsd_m_per_px*1000:.2f}mm/px "
            f"nozzle_y={self.nozzle_y_px}px "
            f"FOV={self.fov_width_m:.3f}m/cam"
        )


# ─────────────────────────────────────────────────────────────
#  CAPTURE CONFIGURATION
# ─────────────────────────────────────────────────────────────

@dataclass
class CaptureConfig:
    """Field data capture — saves paired RGB frames for annotation."""
    auto_interval:   float         = 2.0    # seconds between auto-saves
    max_frames:      Optional[int] = None   # None = unlimited
    save_left_jpg:   bool          = True
    save_right_jpg:  bool          = True
    save_dir:        str           = (
        "/media/pagsun/Transcend/phd_project/emeet_dual_cam/captures"
    )
    # Quality filter
    min_sharpness:   float = 50.0
    min_brightness:  float = 10.0


# ─────────────────────────────────────────────────────────────
#  LOGGING CONFIGURATION
# ─────────────────────────────────────────────────────────────

@dataclass
class LogConfig:
    """Session logging paths and verbosity."""
    base_dir: Path = Path(
        "/media/pagsun/Transcend/phd_project/emeet_dual_cam/logs"
    )
    log_spray_events:     bool = True
    log_sync_errors:      bool = True
    sync_error_threshold: float = 50.0   # ms — log warning above this


# ─────────────────────────────────────────────────────────────
#  SESSION CONFIGURATION
# ─────────────────────────────────────────────────────────────

@dataclass
class SessionConfig:
    """Per-session metadata written to every log file."""
    field_id:        str          = "wilkin_county"
    operator:        str          = "nana"
    growth_stage:    GrowthStage  = GrowthStage.SIX_LEAF
    detection_mode:  DetectionMode = DetectionMode.WEED
    notes:           str          = ""


# ─────────────────────────────────────────────────────────────
#  ROOT CONFIG
# ─────────────────────────────────────────────────────────────

@dataclass
class RGBConfig:
    """
    Root configuration object for the dual RGB detection system.
    Pass one RGBConfig instance to every module.
    """
    camera:    RGBCameraConfig  = field(default_factory=RGBCameraConfig)
    model:     ModelConfig      = field(default_factory=ModelConfig)
    zones:     ZoneConfig       = field(default_factory=ZoneConfig)
    capture:   CaptureConfig    = field(default_factory=CaptureConfig)
    logging:   LogConfig        = field(default_factory=LogConfig)
    session:   SessionConfig    = field(default_factory=SessionConfig)
    geometry:  GeometryConfig   = field(default_factory=GeometryConfig)

    # Parent project path
    project_root: Path = Path(
        "/media/pagsun/Transcend/phd_project/emeet_dual_cam"
    )

    def __post_init__(self):
        """Resolve model paths relative to project root."""
        self.model.weed_rgb_pt      = self.project_root / "models/weed_rgb.pt"
        self.model.weed_rgb_engine  = self.project_root / "models/weed_rgb.engine"
        self.model.cls_rgb_pt       = self.project_root / "models/cls_rgb.pt"
        self.model.cls_rgb_engine   = self.project_root / "models/cls_rgb.engine"
        self.logging.base_dir       = self.project_root / "logs"
        self.capture.save_dir       = str(self.project_root / "captures")


# ─────────────────────────────────────────────────────────────
#  FACTORY HELPERS
# ─────────────────────────────────────────────────────────────

def get_weed_config(
    field_id:     str         = "wilkin_county",
    growth_stage: GrowthStage = GrowthStage.SIX_LEAF,
    operator:     str         = "nana",
) -> RGBConfig:
    """Return a ready-to-use config for the weed resistance experiment."""
    cfg = RGBConfig()
    cfg.session.field_id       = field_id
    cfg.session.growth_stage   = growth_stage
    cfg.session.operator       = operator
    cfg.session.detection_mode = DetectionMode.WEED
    return cfg


def get_cls_config(
    field_id:  str = "wilkin_county",
    operator:  str = "nana",
) -> RGBConfig:
    """Return a ready-to-use config for the CLS disease experiment."""
    cfg = RGBConfig()
    cfg.session.field_id       = field_id
    cfg.session.operator       = operator
    cfg.session.detection_mode = DetectionMode.CLS
    return cfg


# ─────────────────────────────────────────────────────────────
#  SELF TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = get_weed_config()

    print("RGBConfig self-test")
    print(f"  Mode:    {cfg.session.detection_mode.value}")
    print(f"  Stage:   {cfg.session.growth_stage.value}")
    print(f"  Camera:  {cfg.camera.width}×{cfg.camera.height} @ {cfg.camera.fps}fps")
    print()
    print("  cam1 zones:")
    for i, r in enumerate(cfg.zones.cam1_zones()):
        nozzle = cfg.zones.zone_nozzle_map[i]
        label  = ["ZoneA", "ZoneB1"][i]
        print(f"    {label}: {r}  → N{nozzle + 1}")
    print()
    print("  cam2 zones:")
    for i, r in enumerate(cfg.zones.cam2_zones()):
        nozzle = cfg.zones.zone_nozzle_map[2 + i]
        label  = ["ZoneB2", "ZoneC"][i]
        print(f"    {label}: {r}  → N{nozzle + 1}")
    print()
    print("  ✓  B1/B2 split values MEASURED 2026-06-26  B1=1150px  B2=900px | N1=600 N2=1700/400 N3=1400")
    print()
    centers = cfg.zones.nozzle_centers()
    print("  Nozzle centers (computed):")
    print(f"    N1  cam1 = {centers['N1_cam1']}px  (Zone A midpoint)")
    print(f"    N2  cam1 = {centers['N2_cam1']}px  (Zone B1 midpoint)")
    print(f"    N2  cam2 = {centers['N2_cam2']}px  (Zone B2 midpoint)")
    print(f"    N3  cam2 = {centers['N3_cam2']}px  (Zone C midpoint)")
    print()
    print("  Model ready (weed):", cfg.model.model_ready(DetectionMode.WEED))
    print("  Model ready (cls): ", cfg.model.model_ready(DetectionMode.CLS))
    print()
    print("detection_config_rgb.py ✓")