#!/usr/bin/env python3
"""
zone_manager_rgb.py
eMeet Dual RGB Detection System — Spray Zone Manager

Maps YOLO detections from two cameras to three physical spray zones
and applies the validated passive debounce filter from Test_High_Focus.py.

Zone layout:
    LEFT camera (cam1)               RIGHT camera (cam2)
    ┌──────────────┬────────┐        ┌────────┬──────────────┐
    │    Zone A    │ Zone B1│        │ Zone B2│   Zone C     │
    │   Nozzle 1   │ Nozzle2│        │ Nozzle2│  Nozzle 3    │
    └──────────────┴────────┘        └────────┴──────────────┘

Key logic:
  - Zone B fires N2 if detection in B1 OR B2 (either camera)
  - Each zone has its own counter — independent debounce per zone
  - B1 and B2 share nozzle_id=1; nozzle_active OR logic handled here

Debounce filter (identical to validated weedbot Test_High_Focus.py):
  - Detection in zone  → counter += 1  (capped at threshold)
  - No detection       → counter -= drain_rate (floored at 0)
  - counter >= threshold → spray_active = True
  - counter == 0       → spray_active = False

Author : Nana | NDSU / PhD Imaging System
Path   : /media/pagsun/Transcend/phd_project/emeet_dual_cam/
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

try:
    from core.detection_config_rgb import RGBConfig, DetectionMode
    from core.detection_engine_rgb import Detection, InferenceResult, DualInferenceResult
except ImportError:
    from detection_config_rgb import RGBConfig, DetectionMode
    from detection_engine_rgb import Detection, InferenceResult, DualInferenceResult


# ─────────────────────────────────────────────────────────────
#  ZONE IDENTIFIERS
# ─────────────────────────────────────────────────────────────

ZONE_A  = 0   # cam1 left region   → Nozzle 1
ZONE_B1 = 1   # cam1 right region  → Nozzle 2 (shared with B2)
ZONE_B2 = 2   # cam2 left region   → Nozzle 2 (shared with B1)
ZONE_C  = 3   # cam2 right region  → Nozzle 3

ZONE_NAMES = {
    ZONE_A:  "ZoneA",
    ZONE_B1: "ZoneB1",
    ZONE_B2: "ZoneB2",
    ZONE_C:  "ZoneC",
}

ZONE_CAMERAS = {
    ZONE_A:  "left",
    ZONE_B1: "left",
    ZONE_B2: "right",
    ZONE_C:  "right",
}


# ─────────────────────────────────────────────────────────────
#  ZONE STATE
# ─────────────────────────────────────────────────────────────

@dataclass
class ZoneState:
    """
    State for one spray zone.
    Mirrors the detection_countX / detection_threshold logic
    from the validated weedbot script (Test_High_Focus.py).
    """
    zone_id:    int
    nozzle_id:  int                          # 0=N1, 1=N2, 2=N3
    pixel_rect: Tuple[int, int, int, int]    # (x1, y1, x2, y2)
    camera:     str                          # "left" or "right"

    # Debounce counter
    counter:    int  = 0
    threshold:  int  = 4     # frames to confirm
    drain_rate: int  = 1     # frames decremented per empty tick

    # Current state
    spray_active: bool = False

    # Audit trail
    total_triggers:     int   = 0
    total_detections:   int   = 0
    last_trigger_time:  float = 0.0

    # Detections in this zone this frame
    current_detections: List[Detection] = field(default_factory=list)

    @property
    def name(self) -> str:
        return ZONE_NAMES.get(self.zone_id, f"Zone{self.zone_id}")

    @property
    def x1(self) -> int: return self.pixel_rect[0]
    @property
    def y1(self) -> int: return self.pixel_rect[1]
    @property
    def x2(self) -> int: return self.pixel_rect[2]
    @property
    def y2(self) -> int: return self.pixel_rect[3]

    def contains(self, cx: int, cy: int) -> bool:
        """Return True if point (cx, cy) falls within this zone."""
        return self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2

    def to_dict(self) -> Dict:
        return {
            'zone_id':       self.zone_id,
            'name':          self.name,
            'nozzle_id':     self.nozzle_id,
            'camera':        self.camera,
            'counter':       self.counter,
            'threshold':     self.threshold,
            'spray_active':  self.spray_active,
            'total_triggers': self.total_triggers,
            'detections_now': len(self.current_detections),
            'pixel_rect':    list(self.pixel_rect),
        }


# ─────────────────────────────────────────────────────────────
#  ZONE DECISION
# ─────────────────────────────────────────────────────────────

@dataclass
class ZoneDecision:
    """
    Output from ZoneManagerRGB for one frame pair.
    Tells actuation_controller.py which nozzles to fire / stop.
    """
    zones:           List[ZoneState]
    nozzles_to_fire: List[int]   # nozzle IDs (0-indexed) that should be ON
    nozzles_to_stop: List[int]   # nozzle IDs that should turn OFF
    new_triggers:    List[int]   # zone IDs that just crossed threshold ↑
    new_releases:    List[int]   # zone IDs that just dropped to 0 ↓
    timestamp:       float = field(default_factory=time.time)
    total_detections: int  = 0
    frame_id:        int   = 0

    @property
    def any_spray_active(self) -> bool:
        return len(self.nozzles_to_fire) > 0

    def to_dict(self) -> Dict:
        return {
            'frame_id':        self.frame_id,
            'nozzles_to_fire': self.nozzles_to_fire,
            'nozzles_to_stop': self.nozzles_to_stop,
            'new_triggers':    self.new_triggers,
            'new_releases':    self.new_releases,
            'any_active':      self.any_spray_active,
            'total_detections': self.total_detections,
            'zones':           [z.to_dict() for z in self.zones],
            'timestamp':       self.timestamp,
        }


# ─────────────────────────────────────────────────────────────
#  ZONE MANAGER RGB
# ─────────────────────────────────────────────────────────────

class ZoneManagerRGB:
    """
    Maps dual-camera YOLO detections to spray zones with passive
    debounce filtering.

    Zone B OR logic:
        N2 fires if ZoneB1 (cam1) OR ZoneB2 (cam2) has an active
        confirmed detection.  Each sub-zone has its own counter so
        a sustained detection in either half is sufficient.

    Frame-by-frame usage:
        manager = ZoneManagerRGB(cfg)
        decision = manager.update(dual_result)
        # decision.nozzles_to_fire → fire these nozzles
        # decision.nozzles_to_stop → stop these nozzles
    """

    def __init__(self, cfg: RGBConfig):
        self.cfg    = cfg
        self.zones  = self._build_zones()
        self._prev_spray_states = [False] * len(self.zones)
        self._frame_count = 0

        logging.info(
            f"ZoneManagerRGB initialized: {len(self.zones)} zones | "
            f"threshold={cfg.zones.detection_threshold} | "
            f"drain_rate={cfg.zones.drain_rate}"
        )
        for z in self.zones:
            logging.info(
                f"  {z.name}: cam={z.camera} nozzle=N{z.nozzle_id+1} "
                f"rect={z.pixel_rect}"
            )

    # ── Zone construction ─────────────────────────────────────

    def _build_zones(self) -> List[ZoneState]:
        """Build the four-zone list from config."""
        cfg       = self.cfg
        threshold = cfg.zones.detection_threshold
        drain     = cfg.zones.drain_rate
        nozzle_map = cfg.zones.zone_nozzle_map  # [0, 1, 1, 2]

        cam1_rects = cfg.zones.cam1_zones()  # [ZoneA, ZoneB1]
        cam2_rects = cfg.zones.cam2_zones()  # [ZoneB2, ZoneC]

        zones = []

        # cam1 zones: ZoneA (id=0), ZoneB1 (id=1)
        for i, rect in enumerate(cam1_rects):
            zones.append(ZoneState(
                zone_id    = i,
                nozzle_id  = nozzle_map[i],
                pixel_rect = rect,
                camera     = "left",
                threshold  = threshold,
                drain_rate = drain,
            ))

        # cam2 zones: ZoneB2 (id=2), ZoneC (id=3)
        for i, rect in enumerate(cam2_rects):
            zones.append(ZoneState(
                zone_id    = 2 + i,
                nozzle_id  = nozzle_map[2 + i],
                pixel_rect = rect,
                camera     = "right",
                threshold  = threshold,
                drain_rate = drain,
            ))

        return zones

    # ── Detection assignment ──────────────────────────────────

    def _assign_detections(self, dual_result: DualInferenceResult):
        """
        Route each detection to the correct zone based on cx,cy.
        Left camera detections → ZoneA / ZoneB1
        Right camera detections → ZoneB2 / ZoneC

        Detections whose class is in cfg.zones.non_spray_classes (the
        crop itself, at minimum) never reach a zone's
        current_detections at all -- they're filtered out here, before
        the debounce counter or any spray decision ever sees them.
        This is the single point every spray-eligibility check flows
        through; filtering here means a non-spray class categorically
        cannot trigger a nozzle regardless of confidence or how long
        it stays in view.
        """
        non_spray = set(self.cfg.zones.non_spray_classes)

        # Clear previous frame's detections
        for zone in self.zones:
            zone.current_detections = []

        # Route left camera detections
        for det in dual_result.left.detections:
            if det.class_name in non_spray:
                continue
            for zone in self.zones:
                if zone.camera == "left" and zone.contains(det.cx, det.cy):
                    zone.current_detections.append(det)
                    break   # one zone per detection

        # Route right camera detections
        for det in dual_result.right.detections:
            if det.class_name in non_spray:
                continue
            for zone in self.zones:
                if zone.camera == "right" and zone.contains(det.cx, det.cy):
                    zone.current_detections.append(det)
                    break

    # ── Debounce update ───────────────────────────────────────

    def update(self, dual_result: DualInferenceResult) -> ZoneDecision:
        """
        Process one DualInferenceResult and update all zone states.

        This implements the validated weedbot passive filter:
          - Detection   → counter += 1  (up to threshold)
          - No detect   → counter -= drain_rate (down to 0)
          - active when counter >= threshold

        Zone B fires N2 if B1 OR B2 has confirmed detection.
        This is handled naturally by the nozzle_active OR logic below.

        Returns:
            ZoneDecision with nozzles_to_fire and nozzles_to_stop lists.
        """
        self._frame_count += 1
        self._assign_detections(dual_result)

        new_triggers = []
        new_releases = []

        for i, zone in enumerate(self.zones):
            prev_active    = self._prev_spray_states[i]
            has_detection  = len(zone.current_detections) > 0

            # ── Counter update ─────────────────────────────────
            if has_detection:
                zone.counter = min(zone.counter + 1, zone.threshold)
                zone.total_detections += 1
            else:
                zone.counter = max(zone.counter - zone.drain_rate, 0)

            # ── Spray state ────────────────────────────────────
            zone.spray_active = zone.counter >= zone.threshold

            # ── State change tracking ──────────────────────────
            if zone.spray_active and not prev_active:
                new_triggers.append(zone.zone_id)
                zone.total_triggers    += 1
                zone.last_trigger_time  = time.time()
                logging.info(
                    f"🟢 {zone.name} TRIGGERED (N{zone.nozzle_id+1}, {zone.camera}) | "
                    f"detections: {[d.class_name for d in zone.current_detections]}"
                )
            elif not zone.spray_active and prev_active:
                new_releases.append(zone.zone_id)
                logging.info(
                    f"🔴 {zone.name} RELEASED (N{zone.nozzle_id+1}, {zone.camera})"
                )

            self._prev_spray_states[i] = zone.spray_active

        # ── Nozzle fire/stop lists ─────────────────────────────
        # Zone B OR logic: N2 fires if ZoneB1 OR ZoneB2 is active.
        # Implemented naturally: both B1 and B2 have nozzle_id=1.
        # A nozzle fires if ANY zone with that nozzle_id is active.
        nozzle_active: Dict[int, bool] = {}
        for zone in self.zones:
            nid = zone.nozzle_id
            if nid not in nozzle_active:
                nozzle_active[nid] = False
            if zone.spray_active:
                nozzle_active[nid] = True

        nozzles_to_fire = [n for n, active in nozzle_active.items() if active]
        nozzles_to_stop = [n for n, active in nozzle_active.items() if not active]

        total_detections = sum(len(z.current_detections) for z in self.zones)

        if self._frame_count % 100 == 0:
            self._log_status()

        return ZoneDecision(
            zones            = self.zones,
            nozzles_to_fire  = sorted(nozzles_to_fire),
            nozzles_to_stop  = sorted(nozzles_to_stop),
            new_triggers     = new_triggers,
            new_releases     = new_releases,
            timestamp        = dual_result.timestamp,
            total_detections = total_detections,
            frame_id         = dual_result.frame_id,
        )

    # ── Configuration updates ─────────────────────────────────

    def update_b_split(self, b1_split_x: int, b2_split_x: int):
        """
        Update B1/B2 zone split after field measurement.
        Call this after you've measured the pixel X of N2 in each camera.

            b1_split_x : pixel X in cam1 where ZoneA ends / ZoneB1 starts
            b2_split_x : pixel X in cam2 where ZoneB2 ends / ZoneC starts
        """
        # Update config
        self.cfg.zones.B1_SPLIT_X = b1_split_x
        self.cfg.zones.B2_SPLIT_X = b2_split_x

        # Rebuild zone rects
        cam1_rects = self.cfg.zones.cam1_zones()
        cam2_rects = self.cfg.zones.cam2_zones()

        self.zones[ZONE_A].pixel_rect  = cam1_rects[0]
        self.zones[ZONE_B1].pixel_rect = cam1_rects[1]
        self.zones[ZONE_B2].pixel_rect = cam2_rects[0]
        self.zones[ZONE_C].pixel_rect  = cam2_rects[1]

        logging.info(
            f"ZoneManagerRGB: B split updated — "
            f"B1_SPLIT_X={b1_split_x} (cam1) | "
            f"B2_SPLIT_X={b2_split_x} (cam2)"
        )

    def set_threshold(self, threshold: int):
        """Update detection threshold for all zones."""
        for zone in self.zones:
            zone.threshold = threshold
        logging.info(f"ZoneManagerRGB: all zones threshold → {threshold}")

    def set_drain_rate(self, rate: int):
        """Update drain rate for all zones."""
        for zone in self.zones:
            zone.drain_rate = rate
        logging.info(f"ZoneManagerRGB: all zones drain rate → {rate}")

    def reset(self):
        """Reset all zone counters. Call at session start."""
        for zone in self.zones:
            zone.counter            = 0
            zone.spray_active       = False
            zone.current_detections = []
        self._prev_spray_states = [False] * len(self.zones)
        self._frame_count       = 0
        logging.info("ZoneManagerRGB: reset")

    # ── Diagnostics ───────────────────────────────────────────

    def _log_status(self):
        parts = []
        for z in self.zones:
            state = "ON" if z.spray_active else "--"
            parts.append(f"{z.name}[{state} {z.counter}/{z.threshold}]")
        logging.info(f"Zones: {' | '.join(parts)}")

    def get_status(self) -> Dict:
        return {
            'frame_count': self._frame_count,
            'zones':       [z.to_dict() for z in self.zones],
            'any_active':  any(z.spray_active for z in self.zones),
            'b_split':     {
                'B1_SPLIT_X': self.cfg.zones.B1_SPLIT_X,
                'B2_SPLIT_X': self.cfg.zones.B2_SPLIT_X,
                'calibrated': False,  # update to True after field verification,   # set to True after measurement
            },
        }


# ─────────────────────────────────────────────────────────────
#  SELF TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import time
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    from detection_config_rgb import get_weed_config
    from detection_engine_rgb import (
        Detection, InferenceResult, DualInferenceResult
    )

    print("=" * 60)
    print("ZoneManagerRGB — Self Test")
    print("=" * 60)

    cfg     = get_weed_config()
    manager = ZoneManagerRGB(cfg)
    print()

    # ── Helper: build DualInferenceResult from detection lists ─
    def make_dual(left_dets, right_dets, frame_id=0):
        def make_result(dets, camera):
            return InferenceResult(
                detections    = dets,
                inference_ms  = 5.0,
                preprocess_ms = 2.0,
                total_ms      = 7.0,
                frame_shape   = (1080, 1920),
                camera        = camera,
            )
        return DualInferenceResult(
            left      = make_result(left_dets,  "left"),
            right     = make_result(right_dets, "right"),
            frame_id  = frame_id,
            timestamp = time.time(),
        )

    def det(cx, cy, cam, name="kochia"):
        """Make a fake detection centered at (cx, cy)."""
        return Detection(
            class_id=1, class_name=name, confidence=0.85,
            x1=cx-20, y1=cy-20, x2=cx+20, y2=cy+20,
            camera=cam,
        )

    # Zone A boundary: 0 to B1_SPLIT_X = 1150 (measured 2026-06-26)
    # Zone B1: 1150 to 1920 on cam1
    # Zone B2: 0 to 900 on cam2
    # Zone C: 900 to 1920 on cam2

    # ── Test 1: Zone A fires after 4 frames ───────────────────
    print("Test 1: Zone A detection — 4 frames → should fire N1")
    manager.reset()
    det_a = det(cx=400, cy=540, cam="left")   # Zone A (0-1150), N1 center=600
    for _ in range(4):
        d = manager.update(make_dual([det_a], []))
    assert manager.zones[ZONE_A].spray_active,   "Zone A should be active"
    assert 0 in d.nozzles_to_fire,               "N1 should fire"
    assert not manager.zones[ZONE_B1].spray_active, "B1 should NOT fire"
    print(f"  ZoneA active={manager.zones[ZONE_A].spray_active} ✓")
    print(f"  Nozzles firing: {[n+1 for n in d.nozzles_to_fire]} ✓")
    print()

    # ── Test 2: Zone B1 alone fires N2 ────────────────────────
    print("Test 2: Zone B1 detection (cam1) — should fire N2 alone")
    manager.reset()
    det_b1 = det(cx=1700, cy=540, cam="left")  # Zone B1 (1150-1920), N2 center=1700
    for _ in range(4):
        d = manager.update(make_dual([det_b1], []))
    assert manager.zones[ZONE_B1].spray_active,  "Zone B1 should be active"
    assert 1 in d.nozzles_to_fire,               "N2 should fire"
    assert not manager.zones[ZONE_A].spray_active
    assert not manager.zones[ZONE_B2].spray_active
    print(f"  ZoneB1 active={manager.zones[ZONE_B1].spray_active} ✓")
    print(f"  Nozzles firing: {[n+1 for n in d.nozzles_to_fire]} ✓")
    print()

    # ── Test 3: Zone B2 alone fires N2 ────────────────────────
    print("Test 3: Zone B2 detection (cam2) — should fire N2 alone")
    manager.reset()
    det_b2 = det(cx=400, cy=540, cam="right")  # Zone B2 (0-900), N2 center=400
    for _ in range(4):
        d = manager.update(make_dual([], [det_b2]))
    assert manager.zones[ZONE_B2].spray_active,  "Zone B2 should be active"
    assert 1 in d.nozzles_to_fire,               "N2 should fire (B2)"
    print(f"  ZoneB2 active={manager.zones[ZONE_B2].spray_active} ✓")
    print(f"  Nozzles firing: {[n+1 for n in d.nozzles_to_fire]} ✓")
    print()

    # ── Test 4: Zone C fires N3 ───────────────────────────────
    print("Test 4: Zone C detection — should fire N3")
    manager.reset()
    det_c = det(cx=1400, cy=540, cam="right")  # Zone C (900-1920), N3 center=1400
    for _ in range(4):
        d = manager.update(make_dual([], [det_c]))
    assert manager.zones[ZONE_C].spray_active,   "Zone C should be active"
    assert 2 in d.nozzles_to_fire,               "N3 should fire"
    print(f"  ZoneC active={manager.zones[ZONE_C].spray_active} ✓")
    print(f"  Nozzles firing: {[n+1 for n in d.nozzles_to_fire]} ✓")
    print()

    # ── Test 5: B1 + B2 both active → still only N2 ──────────
    print("Test 5: B1 and B2 both detect → N2 fires (not doubled)")
    manager.reset()
    for _ in range(4):
        d = manager.update(make_dual([det_b1], [det_b2]))
    assert manager.zones[ZONE_B1].spray_active
    assert manager.zones[ZONE_B2].spray_active
    assert d.nozzles_to_fire.count(1) == 1,   "N2 should appear only once"
    print(f"  B1 active={manager.zones[ZONE_B1].spray_active} ✓")
    print(f"  B2 active={manager.zones[ZONE_B2].spray_active} ✓")
    print(f"  Nozzles firing: {[n+1 for n in d.nozzles_to_fire]} ✓  (N2 once)")
    print()

    # ── Test 6: Counter drains to 0 after detection stops ─────
    print("Test 6: Detection stops → counter drains → nozzle stops")
    # Zone A is still active from test 5? Reset was called, so start fresh
    manager.reset()
    for _ in range(4):
        manager.update(make_dual([det_a], []))
    assert manager.zones[ZONE_A].spray_active
    for _ in range(5):
        d = manager.update(make_dual([], []))
    assert not manager.zones[ZONE_A].spray_active, "Zone A should stop"
    assert manager.zones[ZONE_A].counter == 0
    print(f"  ZoneA active={manager.zones[ZONE_A].spray_active} ✓")
    print(f"  Counter={manager.zones[ZONE_A].counter} ✓")
    print()

    # ── Test 7: Threshold not crossed (3 frames) ──────────────
    print("Test 7: 3 frames only — threshold=4 not crossed")
    manager.reset()
    for _ in range(3):
        d = manager.update(make_dual([det_a], []))
    assert not manager.zones[ZONE_A].spray_active
    assert manager.zones[ZONE_A].counter == 3
    print(f"  ZoneA active={manager.zones[ZONE_A].spray_active} ✓")
    print(f"  Counter=3 (need 4) ✓")
    print()

    # ── Test 8: B split update ─────────────────────────────────
    print("Test 8: update_b_split() runtime recalibration")
    manager.update_b_split(b1_split_x=1100, b2_split_x=820)
    assert manager.zones[ZONE_A].pixel_rect  == (0, 0, 1100, 1080)
    assert manager.zones[ZONE_B1].pixel_rect == (1100, 0, 1920, 1080)
    assert manager.zones[ZONE_B2].pixel_rect == (0, 0, 820, 1080)
    assert manager.zones[ZONE_C].pixel_rect  == (820, 0, 1920, 1080)
    print(f"  ZoneA  rect: {manager.zones[ZONE_A].pixel_rect}  ✓")
    print(f"  ZoneB1 rect: {manager.zones[ZONE_B1].pixel_rect} ✓")
    print(f"  ZoneB2 rect: {manager.zones[ZONE_B2].pixel_rect} ✓")
    print(f"  ZoneC  rect: {manager.zones[ZONE_C].pixel_rect}  ✓")
    print()

    # ── Test 9: non_spray_classes (crop) never triggers a spray ──
    print("Test 9: sugarbeet (non_spray_classes) never triggers a nozzle")
    cfg9 = get_weed_config()
    assert "sugarbeet" in cfg9.zones.non_spray_classes, (
        "sugarbeet must be excluded by default -- this is a real "
        "safety requirement (don't spray herbicide on the crop), "
        "not just a config default"
    )
    manager9 = ZoneManagerRGB(cfg9)

    # Reset the split back to the default the earlier tests may have
    # changed via update_b_split(), so Zone A coordinates match det()'s
    # defaults again for this test.
    manager9.update_b_split(b1_split_x=1150, b2_split_x=900)

    # Feed MANY more frames than the trigger threshold, high confidence,
    # squarely inside Zone A -- if the exclusion filter didn't work,
    # this would trigger a spray almost immediately.
    for _ in range(20):
        decision = manager9.update(
            make_dual([det(500, 500, "left", name="sugarbeet")], []))
        assert decision.nozzles_to_fire == [], (
            "sugarbeet must NEVER appear in nozzles_to_fire, no matter "
            "how long it's been detected"
        )
        assert manager9.zones[ZONE_A].current_detections == [], (
            "sugarbeet detections must not even reach current_detections "
            "-- filtered before the debounce counter sees them"
        )
    assert manager9.zones[ZONE_A].counter == 0, (
        f"debounce counter must stay at 0 for an excluded class, "
        f"got {manager9.zones[ZONE_A].counter}"
    )
    print(f"  20 frames of sugarbeet in Zone A -- nozzle never fired ✓")
    print(f"  Zone A debounce counter stayed at 0 ✓")

    # A real weed class in the SAME zone, immediately after, still
    # works normally -- confirms the filter is class-specific, not a
    # broken zone.
    for _ in range(4):
        decision9b = manager9.update(
            make_dual([det(500, 500, "left", name="kochia")], []))
    assert 0 in decision9b.nozzles_to_fire, (
        "a real weed class in the same zone must still trigger normally "
        "-- the exclusion must be class-specific, not zone-wide"
    )
    print(f"  kochia in the same zone still fires normally ✓ "
          f"(exclusion is class-specific, not zone-wide)")
    print()

    print("=" * 60)
    print("zone_manager_rgb.py ✓  ALL TESTS PASSED")
    print("=" * 60)