#!/usr/bin/env python3
"""
detection_engine_rgb.py
eMeet Dual RGB Detection System — YOLO Inference Engine

Standard 3-channel BGR inference for both eMeet cameras.
If model weights are not available yet — engine runs in STUB MODE
until weed_rgb.pt / cls_rgb.pt are trained and placed in models/.

Key design:
  - Model loaded ONCE at startup
  - Each camera frame preprocessed independently (resize → BGR)
  - Returns clean Detection / DualInferenceResult objects
  - Completely decoupled from camera source and zone logic
  - Stub mode returns empty detections so the full pipeline can be
    tested end-to-end before model weights exist

Author : Bright Mensah | NDSU / Imaging System
Path   : /media/pagsun/Transcend/phd_project/emeet_dual_cam/
"""

import cv2
import time
import logging
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple

try:
    from core.detection_config_rgb import RGBConfig, DetectionMode
except ImportError:
    from detection_config_rgb import RGBConfig, DetectionMode

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logging.warning("ultralytics not installed — engine in stub mode")


# ─────────────────────────────────────────────────────────────
#  DETECTION  (single bounding box result)
# ─────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """
    Single object detection from one camera frame.
    Coordinates are in pixel space of the ORIGINAL camera image
    (scaled back from YOLO input size to full 1920×1080).
    """
    class_id:   int
    class_name: str
    confidence: float
    x1: int; y1: int   # top-left
    x2: int; y2: int   # bottom-right
    camera: str = "left"   # "left" or "right"

    @property
    def cx(self) -> int:
        return (self.x1 + self.x2) // 2

    @property
    def cy(self) -> int:
        return (self.y1 + self.y2) // 2

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_dict(self) -> Dict:
        return {
            'class_id':   self.class_id,
            'class_name': self.class_name,
            'confidence': round(self.confidence, 4),
            'bbox':       [self.x1, self.y1, self.x2, self.y2],
            'center':     [self.cx, self.cy],
            'camera':     self.camera,
        }


# ─────────────────────────────────────────────────────────────
#  INFERENCE RESULT  (per camera frame)
# ─────────────────────────────────────────────────────────────

@dataclass
class InferenceResult:
    """
    YOLO inference result for one camera frame.
    """
    detections:      List[Detection]
    inference_ms:    float
    preprocess_ms:   float
    total_ms:        float
    frame_shape:     Tuple[int, int]   # (H, W) of original frame
    camera:          str               # "left" or "right"
    stub_mode:       bool = False      # True if no model loaded

    @property
    def detection_count(self) -> int:
        return len(self.detections)

    @property
    def has_detections(self) -> bool:
        return len(self.detections) > 0

    def to_dict(self) -> Dict:
        return {
            'camera':         self.camera,
            'detections':     [d.to_dict() for d in self.detections],
            'count':          self.detection_count,
            'inference_ms':   round(self.inference_ms, 2),
            'total_ms':       round(self.total_ms, 2),
            'stub_mode':      self.stub_mode,
        }


# ─────────────────────────────────────────────────────────────
#  DUAL INFERENCE RESULT  (both cameras, one frame pair)
# ─────────────────────────────────────────────────────────────

@dataclass
class DualInferenceResult:
    """
    Combined inference results from one FramePair.
    This is the object passed to ZoneManagerRGB.update().
    """
    left:      InferenceResult
    right:     InferenceResult
    frame_id:  int
    timestamp: float = field(default_factory=time.time)

    @property
    def total_detections(self) -> int:
        return self.left.detection_count + self.right.detection_count

    @property
    def has_detections(self) -> bool:
        return self.left.has_detections or self.right.has_detections

    @property
    def total_ms(self) -> float:
        return self.left.total_ms + self.right.total_ms

    def all_detections(self) -> List[Detection]:
        """Flat list of all detections from both cameras."""
        return self.left.detections + self.right.detections

    def to_dict(self) -> Dict:
        return {
            'frame_id':          self.frame_id,
            'timestamp':         self.timestamp,
            'total_detections':  self.total_detections,
            'total_ms':          round(self.total_ms, 2),
            'left':              self.left.to_dict(),
            'right':             self.right.to_dict(),
        }


# ─────────────────────────────────────────────────────────────
#  RGB DETECTION ENGINE
# ─────────────────────────────────────────────────────────────

class RGBDetectionEngine:
    """
    Runs YOLOv8 inference on BGR frames from the dual eMeet cameras.

    Usage:
        engine = RGBDetectionEngine(cfg)
        # engine.ready → False until model weights exist

        dual_result = engine.run(pair)
        # dual_result.left.detections   → List[Detection] from left cam
        # dual_result.right.detections  → List[Detection] from right cam

    Stub mode (no weights):
        Returns empty InferenceResult so the rest of the pipeline
        (zone_manager, actuation) can be tested without a trained model.
    """

    # Class names — will be replaced by model.names after loading
    # (Ultralytics stores the trained class list inside the .pt
    # checkpoint itself, so once a real model loads this dict is
    # overwritten with the correct names/order automatically — this
    # is purely the stub-mode fallback shown before any model exists).
    # Matches yolo_dataset/classes.json as of the first real
    # multiweed_detection training dataset (2026 summer data).
    DEFAULT_CLASS_NAMES = {
        0: 'sugarbeet',
        1: 'kochia',
        2: 'waterhemp',
        3: 'common_ragweed',
        4: 'common_lambsquaters',
        5: 'unknown_weed',
    }

    def __init__(self, cfg: RGBConfig):
        self.cfg       = cfg
        self.model     = None
        self.ready     = False
        self.stub_mode = True
        self.class_names: Dict[int, str] = dict(self.DEFAULT_CLASS_NAMES)

        self._load_model()

    def _load_model(self):
        """Attempt to load YOLO weights. Falls back to stub if not found."""
        if not ULTRALYTICS_AVAILABLE:
            logging.warning(
                "RGBDetectionEngine: ultralytics not available — stub mode"
            )
            return

        mode = self.cfg.session.detection_mode
        model_path = self.cfg.model.get_model_path(mode)

        if not model_path.exists():
            logging.warning(
                f"RGBDetectionEngine: model weights not found at "
                f"{model_path} — running in STUB MODE\n"
                f"  Place weed_rgb.pt or cls_rgb.pt in "
                f"{self.cfg.project_root / 'models/'} to enable inference."
            )
            return

        try:
            logging.info(f"RGBDetectionEngine: loading {model_path} …")
            self.model = YOLO(str(model_path))
            self.model.to(self.cfg.model.device)

            # Override class names from model if available
            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = dict(self.model.names)

            self.ready     = True
            self.stub_mode = False
            logging.info(
                f"RGBDetectionEngine: loaded — "
                f"{len(self.class_names)} classes | "
                f"device={self.cfg.model.device}"
            )
        except Exception as e:
            logging.error(f"RGBDetectionEngine: failed to load model — {e}")

    # ── Preprocessing ─────────────────────────────────────────

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize BGR frame to YOLO input size.
        Returns the resized frame; YOLO handles normalization internally.
        """
        size = self.cfg.model.input_size
        return cv2.resize(frame, (size, size))

    def _scale_box(
        self,
        x1: float, y1: float, x2: float, y2: float,
        orig_h: int, orig_w: int
    ) -> Tuple[int, int, int, int]:
        """
        Scale YOLO output coordinates back to original frame resolution.
        YOLO input: input_size × input_size
        Original:   orig_w × orig_h
        """
        size = self.cfg.model.input_size
        sx = orig_w / size
        sy = orig_h / size
        return (
            int(x1 * sx), int(y1 * sy),
            int(x2 * sx), int(y2 * sy),
        )

    # ── Single-frame inference ─────────────────────────────────

    def _infer_one(
        self,
        frame: np.ndarray,
        camera: str
    ) -> InferenceResult:
        """Run YOLO on one BGR frame. Returns InferenceResult."""
        orig_h, orig_w = frame.shape[:2]
        t0 = time.perf_counter()

        # Preprocess
        resized = self._preprocess(frame)
        t1 = time.perf_counter()
        preprocess_ms = (t1 - t0) * 1000

        # Stub mode — no model
        if self.stub_mode or self.model is None:
            return InferenceResult(
                detections    = [],
                inference_ms  = 0.0,
                preprocess_ms = preprocess_ms,
                total_ms      = preprocess_ms,
                frame_shape   = (orig_h, orig_w),
                camera        = camera,
                stub_mode     = True,
            )

        # YOLO inference
        try:
            results = self.model.predict(
                source    = resized,
                conf      = self.cfg.model.confidence_threshold,
                iou       = self.cfg.model.iou_threshold,
                device    = self.cfg.model.device,
                verbose   = False,
            )
            t2 = time.perf_counter()
            inference_ms = (t2 - t1) * 1000

            detections = self._parse_results(
                results, orig_h, orig_w, camera
            )

            return InferenceResult(
                detections    = detections,
                inference_ms  = inference_ms,
                preprocess_ms = preprocess_ms,
                total_ms      = preprocess_ms + inference_ms,
                frame_shape   = (orig_h, orig_w),
                camera        = camera,
                stub_mode     = False,
            )
        except Exception as e:
            logging.error(f"RGBDetectionEngine [{camera}]: inference error — {e}")
            return InferenceResult(
                detections    = [],
                inference_ms  = 0.0,
                preprocess_ms = preprocess_ms,
                total_ms      = preprocess_ms,
                frame_shape   = (orig_h, orig_w),
                camera        = camera,
                stub_mode     = False,
            )

    def _parse_results(
        self,
        results,
        orig_h: int,
        orig_w: int,
        camera: str
    ) -> List[Detection]:
        """Extract Detection objects from raw YOLO results."""
        detections = []
        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            xyxy   = box.xyxy[0].cpu().numpy()

            x1, y1, x2, y2 = self._scale_box(
                xyxy[0], xyxy[1], xyxy[2], xyxy[3],
                orig_h, orig_w
            )

            detections.append(Detection(
                class_id   = cls_id,
                class_name = self.class_names.get(cls_id, f"cls_{cls_id}"),
                confidence = conf,
                x1=x1, y1=y1, x2=x2, y2=y2,
                camera     = camera,
            ))

        return detections

    # ── Dual-frame inference (primary public API) ─────────────

    def run(self, pair) -> DualInferenceResult:
        """
        Run inference on one FramePair from DualEMEETCamera.

        Args:
            pair: FramePair with .left and .right np.ndarray BGR frames

        Returns:
            DualInferenceResult containing detections for both cameras
        """
        left_result  = self._infer_one(pair.left,  camera="left")
        right_result = self._infer_one(pair.right, camera="right")

        return DualInferenceResult(
            left      = left_result,
            right     = right_result,
            frame_id  = pair.frame_id,
            timestamp = pair.left_ts,
        )

    def get_status(self) -> Dict:
        return {
            'ready':       self.ready,
            'stub_mode':   self.stub_mode,
            'class_count': len(self.class_names),
            'classes':     self.class_names,
            'device':      self.cfg.model.device,
            'input_size':  self.cfg.model.input_size,
        }


# ─────────────────────────────────────────────────────────────
#  SELF TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    from detection_config_rgb import get_weed_config

    print("=" * 55)
    print("RGBDetectionEngine — Self Test")
    print("=" * 55)

    cfg    = get_weed_config()
    engine = RGBDetectionEngine(cfg)

    print(f"\nEngine status:")
    status = engine.get_status()
    for k, v in status.items():
        if k != 'classes':
            print(f"  {k}: {v}")

    # Simulate a FramePair with random frames
    class FakePair:
        left     = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        right    = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        frame_id = 0
        left_ts  = time.time()

    print("\nRunning inference on fake 1920×1080 frame pair …")
    result = engine.run(FakePair())
    print(f"  Left  detections: {result.left.detection_count}  "
          f"(stub={result.left.stub_mode})")
    print(f"  Right detections: {result.right.detection_count}  "
          f"(stub={result.right.stub_mode})")
    print(f"  Total ms: {result.total_ms:.1f}")

    assert isinstance(result, DualInferenceResult)
    assert isinstance(result.left, InferenceResult)
    assert isinstance(result.right, InferenceResult)

    print()
    print("detection_engine_rgb.py ✓  ALL TESTS PASSED")
    print("(Stub mode — add model weights to enable real inference)")
    print("=" * 55)