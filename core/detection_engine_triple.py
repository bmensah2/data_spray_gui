#!/usr/bin/env python3
"""
detection_engine_triple.py
ABEN Triple RGB Detection System — YOLO Inference Engine (3 cameras)

Phase 3 of the triple-camera software redesign: the detection engine.
Runs the SAME YOLO model against all three eMeet cameras' frames
(Cam 1/N1, Cam 2/N2, Cam 3/N3) instead of two.

Built as a new, self-contained module alongside the existing, still-
live core/detection_engine_rgb.py (RGBDetectionEngine) -- same
isolation approach as Phases 1 and 2 -- but UNLIKE those two phases,
this one deliberately WRAPS the existing engine via composition rather
than duplicating its logic. Model loading, preprocessing, single-
frame inference, and result parsing (RGBDetectionEngine._load_model /
_preprocess / _infer_one / _parse_results) are already fully camera-
count-agnostic in the existing code -- they operate on one frame at a
time, parameterized by a camera label string, with no assumption
baked in about there being exactly two. Only the top-level run()
method is hardwired to pair.left/pair.right. Reusing that proven
logic via composition (an internal RGBDetectionEngine instance) means:
  - Zero duplication of ~150 lines of model-loading/inference/parsing
    logic that has nothing camera-count-specific about it.
  - Any future fix or improvement to that logic in
    core/detection_engine_rgb.py automatically applies here too,
    since this runs through the exact same code path, not a copy.
  - core/detection_engine_rgb.py itself is NOT modified at all -- the
    live 2-camera system remains completely untouched.

Author : Bright Mensah | NDSU / Imaging System
Path   : /media/pagsun/Transcend/phd_project/emeet_dual_cam/
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict

try:
    from core.detection_config_rgb import RGBConfig
    from core.detection_engine_rgb import (
        RGBDetectionEngine, Detection, InferenceResult,
    )
except ImportError:
    from detection_config_rgb import RGBConfig
    from detection_engine_rgb import (
        RGBDetectionEngine, Detection, InferenceResult,
    )


CAMERA_LABELS = ["cam1", "cam2", "cam3"]


# ─────────────────────────────────────────────────────────────
#  TRIPLE INFERENCE RESULT  (all 3 cameras, one FrameTriple)
# ─────────────────────────────────────────────────────────────

@dataclass
class TripleInferenceResult:
    """
    Combined inference results from one FrameTriple.

    This is the object that hands off to Phase 1's
    core/zone_manager_triple.py -- ZoneManagerTriple.update() takes a
    plain List[List[Detection]] (one list per camera) by design (see
    that module's own docstring), specifically so this handoff would
    be a thin adapter rather than a design change to either phase.
    all_detections_by_camera() below returns EXACTLY that shape.
    """
    results:   List[InferenceResult]   # length 3, index = camera index
    frame_id:  int
    timestamp: float = field(default_factory=time.time)

    @property
    def total_detections(self) -> int:
        return sum(r.detection_count for r in self.results)

    @property
    def has_detections(self) -> bool:
        return any(r.has_detections for r in self.results)

    @property
    def total_ms(self) -> float:
        return sum(r.total_ms for r in self.results)

    def all_detections(self) -> List[Detection]:
        """Flat list of all detections from all 3 cameras."""
        flat: List[Detection] = []
        for r in self.results:
            flat.extend(r.detections)
        return flat

    def all_detections_by_camera(self) -> List[List[Detection]]:
        """
        Per-camera detection lists, index = camera index (0=Cam1,
        1=Cam2, 2=Cam3) -- exactly the input shape
        ZoneManagerTriple.update() expects. Pass this straight
        through: zone_decision = zone_manager.update(
            triple_result.all_detections_by_camera())
        """
        return [r.detections for r in self.results]

    def to_dict(self) -> Dict:
        return {
            "frame_id":         self.frame_id,
            "timestamp":        self.timestamp,
            "total_detections": self.total_detections,
            "total_ms":         round(self.total_ms, 2),
            "cameras":          [r.to_dict() for r in self.results],
        }


# ─────────────────────────────────────────────────────────────
#  TRIPLE DETECTION ENGINE
# ─────────────────────────────────────────────────────────────

class TripleDetectionEngine:
    """
    Runs YOLO inference on BGR frames from all three eMeet cameras.

    Usage:
        engine = TripleDetectionEngine(cfg)
        # engine.ready → False until model weights exist (stub mode)

        triple_result = engine.run_triple(triple)  # triple: FrameTriple
        # triple_result.results[0/1/2].detections → per-camera detections
        # triple_result.all_detections_by_camera() → ready for
        #   ZoneManagerTriple.update()

    Stub mode (no weights) behaves identically to RGBDetectionEngine's
    own stub mode -- returns empty detections per camera so the rest
    of the pipeline (zone manager, actuation) can be exercised
    end-to-end before a trained model exists, exactly like the
    dual-camera system already does.
    """

    def __init__(self, cfg: RGBConfig):
        # Composition, not inheritance or duplication -- see module
        # docstring. All model loading, preprocessing, and parsing
        # happens through this single instance's already-generic
        # internals; this class only adds the "run on 3 frames instead
        # of 2" orchestration on top.
        self._engine = RGBDetectionEngine(cfg)

    @property
    def ready(self) -> bool:
        return self._engine.ready

    @property
    def stub_mode(self) -> bool:
        return self._engine.stub_mode

    @property
    def class_names(self) -> Dict[int, str]:
        return self._engine.class_names

    def run_triple(self, triple) -> TripleInferenceResult:
        """
        Run inference on one FrameTriple from TripleEMEETCamera.

        Args:
            triple: FrameTriple with .frames (list of 3 np.ndarray
                    BGR frames) and .frame_id / .timestamps.

        Returns:
            TripleInferenceResult containing detections for all 3
            cameras.
        """
        results = [
            self._engine._infer_one(triple.frames[i], camera=CAMERA_LABELS[i])
            for i in range(3)
        ]
        return TripleInferenceResult(
            results   = results,
            frame_id  = triple.frame_id,
            # Representative timestamp -- same convention as
            # DualInferenceResult.timestamp using pair.left_ts (the
            # first camera's timestamp, not an aggregate); sync
            # spread across cameras is already captured separately in
            # FrameTriple.sync_error_ms.
            timestamp = triple.timestamps[0],
        )

    def get_status(self) -> Dict:
        return self._engine.get_status()


# ─────────────────────────────────────────────────────────────
#  SELF TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys as _sys
    from pathlib import Path as _Path
    _ROOT = _Path(__file__).resolve().parent.parent
    if str(_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_ROOT))

    import numpy as np
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    from core.detection_config_rgb import get_weed_config
    from core.zone_manager_triple import ZoneManagerTriple, TripleZoneConfig

    print("=" * 60)
    print("TripleDetectionEngine — Self Test")
    print("=" * 60)

    cfg    = get_weed_config()
    engine = TripleDetectionEngine(cfg)

    print("\nEngine status:")
    status = engine.get_status()
    for k, v in status.items():
        if k != "classes":
            print(f"  {k}: {v}")

    class FakeTriple:
        frames     = [
            np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        frame_id   = 0
        timestamps = [time.time(), time.time(), time.time()]

    print("\nRunning inference on a fake 3x 1920x1080 frame triple …")
    result = engine.run_triple(FakeTriple())
    assert isinstance(result, TripleInferenceResult)
    assert len(result.results) == 3
    for i, r in enumerate(result.results):
        assert isinstance(r, InferenceResult)
        print(f"  Cam{i+1} detections: {r.detection_count} (stub={r.stub_mode})")
    print(f"  Total ms: {result.total_ms:.1f}")
    print("✓ run_triple() correctly produces a TripleInferenceResult "
          "with 3 real InferenceResult objects")

    # ── The decisive test: does this phase's output actually plug
    # straight into Phase 1's ZoneManagerTriple.update() with zero
    # adaptation, as both phases were designed to allow? ──
    by_camera = result.all_detections_by_camera()
    assert isinstance(by_camera, list) and len(by_camera) == 3
    assert all(isinstance(lst, list) for lst in by_camera)
    print(f"✓ all_detections_by_camera() returns exactly the "
          f"List[List[Detection]] shape ZoneManagerTriple.update() "
          f"expects: {[len(lst) for lst in by_camera]} detections/camera")

    zone_cfg = TripleZoneConfig()
    zone_mgr = ZoneManagerTriple(zone_cfg)
    decision = zone_mgr.update(by_camera, frame_id=result.frame_id,
                               timestamp=result.timestamp)
    print(f"✓ Phase 3's output was passed DIRECTLY into Phase 1's "
          f"ZoneManagerTriple.update() with no adapter code at all -- "
          f"decision: nozzles_to_fire={decision.nozzles_to_fire}")

    # In stub mode there are never any real detections, so nothing
    # should ever fire -- confirms the end-to-end wiring doesn't
    # spuriously trigger anything by accident.
    assert decision.nozzles_to_fire == []
    print("✓ Stub mode correctly produces zero detections end-to-end, "
          "so no nozzle spuriously fires")

    print()
    print("=" * 60)
    print("detection_engine_triple.py ✓  ALL TESTS PASSED")
    print("(Stub mode — add model weights to enable real inference)")
    print("=" * 60)
