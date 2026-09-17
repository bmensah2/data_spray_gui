#!/usr/bin/env python3
"""
zone_manager_triple.py
ABEN Triple RGB Imaging System — Spray Zone Manager (3 cameras, 1:1 nozzles)

Phase 1 of the triple-camera software redesign: zone geometry and the
detection-to-nozzle mapping logic, built as a new, self-contained
module rather than modifying the existing (still live, still
production) dual-camera core/zone_manager_rgb.py / ZoneConfig in
place. The two coexist until every phase of the redesign (camera
capture, detection engine, GUI display, camera settings) is reviewed
and the operator is ready to cut over -- this file has zero import
dependency on the 2-camera modules, so nothing here can affect the
currently-working live system.

── Why this is simpler than the old dual-camera zone system ─────────
The old ZoneConfig/ZoneManagerRGB (see core/detection_config_rgb.py,
core/zone_manager_rgb.py) needed a SHARED "Zone B" split within a
single camera's frame, because one camera covered TWO nozzles' worth
of physical ground (e.g. cam1 covered N1 fully plus part of N2's
territory, cam2 covered the rest of N2's territory plus N3 fully).
That's why N2 needed an OR-across-two-sub-zones rule.

The new hardware has one eMeet camera mounted physically inline with
each nozzle (Cam 1/N1, Cam 2/N2, Cam 3/N3) -- see
/projects/.../overview.md for the physical layout. So each camera now
maps to exactly ONE nozzle; there is no more shared zone to OR
together. The only remaining complexity is that adjacent cameras'
fields of view PHYSICALLY OVERLAP (confirmed by direct measurement
with a tape measure spanning the overlap regions, via
tools/check_all_cameras.py) -- so each camera is restricted to a
pixel range within its OWN frame ("owned territory"), and a detection
outside that range is ignored here because it belongs to a
neighboring camera's owned range instead (that neighbor will pick it
up in its own frame). This is the "split by midpoint" design the
operator chose over allowing double-spraying in the overlap.

── Measured calibration (see calibration block in TripleZoneConfig) ──
Physical camera views and owned-zone boundaries measured directly by
the operator with a tape measure spanning the ground, cross-checked
against a physical spray test (nozzle fired directly onto a pot
centered in its own camera's frame, confirmed to land accurately).
Pixel boundaries below assume a linear pixel-to-physical mapping
across each camera's 1920px frame width -- the same approximation the
old dual-camera B1_SPLIT_X/B2_SPLIT_X system used, and the only
practical option without a full lens-distortion calibration.

Author : Bright Mensah | NDSU / Imaging System
Path   : /media/pagsun/Transcend/phd_project/emeet_dual_cam/
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

try:
    from core.detection_engine_rgb import Detection
except ImportError:
    from detection_engine_rgb import Detection


# ─────────────────────────────────────────────────────────────
#  ZONE CONFIG — measured calibration
# ─────────────────────────────────────────────────────────────

@dataclass
class TripleZoneConfig:
    """
    Spray zone pixel boundaries for the triple eMeet camera setup.

    ╔══════════════════════════════════════════════════════════╗
    ║  CALIBRATION — MEASURED (tape measure + check_all_cameras.py) ║
    ║                                                            ║
    ║  Physical camera views (inches along the row):             ║
    ║    Cam 1 (N1): view  1–32in  |  owned  1–28in              ║
    ║    Cam 2 (N2): view 22–54in  |  owned 28–49in              ║
    ║    Cam 3 (N3): view 43–74in  |  owned 49–74in              ║
    ║                                                            ║
    ║  Boundary between Cam1/Cam2's owned zones: 28in            ║
    ║  Boundary between Cam2/Cam3's owned zones: 49in            ║
    ║                                                            ║
    ║  Pixel boundaries (linear map across each camera's own     ║
    ║  1920px frame width, from its own view span above):        ║
    ║    Cam 1 acts on pixel_x <  CAM1_MAX_X                     ║
    ║    Cam 2 acts on CAM2_MIN_X < pixel_x < CAM2_MAX_X          ║
    ║    Cam 3 acts on pixel_x >  CAM3_MIN_X                     ║
    ╚══════════════════════════════════════════════════════════╝
    """

    FRAME_WIDTH:  int = 1920
    FRAME_HEIGHT: int = 1080

    # ── Calibration values — UPDATE AFTER RE-MEASUREMENT ────────
    # (28in - 1in) / 31in * 1920  ≈ 1672
    CAM1_MAX_X: int = 1672
    # (28in - 22in) / 32in * 1920 = 360
    CAM2_MIN_X: int = 360
    # (49in - 22in) / 32in * 1920 ≈ 1620
    CAM2_MAX_X: int = 1620
    # (49in - 43in) / 31in * 1920 ≈ 372
    CAM3_MIN_X: int = 372

    # camera_index -> nozzle_id (0-indexed); one-to-one, unlike the
    # old dual-camera zone_nozzle_map which had two zones sharing one
    # nozzle_id.
    zone_nozzle_map: List[int] = field(default_factory=lambda: [0, 1, 2])

    camera_labels: List[str] = field(
        default_factory=lambda: ["Cam 1 / N1", "Cam 2 / N2", "Cam 3 / N3"]
    )

    # Detection filter — frames needed before nozzle fires. Same
    # validated value as the dual-camera system
    # (core/detection_config_rgb.py's ZoneConfig.detection_threshold).
    detection_threshold: int = 4
    drain_rate:          int = 1

    # Same non-spray exclude-list design as the dual-camera system:
    # the crop itself must never trigger a spray, regardless of
    # confidence, and this stays an EXCLUDE list (not an allow list)
    # so a newly-added weed class doesn't silently fail to spray-
    # qualify just because nobody added it to an allow list.
    non_spray_classes: List[str] = field(
        default_factory=lambda: ["sugarbeet"]
    )

    def owned_ranges(self) -> List[Tuple[Optional[int], Optional[int]]]:
        """
        Per-camera (min_x, max_x) pixel range this camera acts on.
        None means "no boundary on this side" (an edge camera acts
        all the way to its own frame's edge there — Cam 1 has no left
        boundary, Cam 3 has no right boundary, since there's no
        neighbor beyond them to hand detections off to).
        """
        return [
            (None,           self.CAM1_MAX_X),   # Cam 1
            (self.CAM2_MIN_X, self.CAM2_MAX_X),  # Cam 2
            (self.CAM3_MIN_X, None),             # Cam 3
        ]

    @property
    def camera_count(self) -> int:
        return len(self.zone_nozzle_map)


# ─────────────────────────────────────────────────────────────
#  ZONE STATE / DECISION — same shape as the dual-camera system,
#  redefined here (not imported) to keep this module fully
#  independent during the migration period.
# ─────────────────────────────────────────────────────────────

@dataclass
class ZoneState:
    """State for one camera's spray zone (one zone per camera now,
    not up to two sharing a nozzle as in the dual-camera system)."""
    zone_id:    int             # == camera_index, since it's 1:1 now
    nozzle_id:  int
    camera_index: int
    label:      str
    min_x:      Optional[int]   # None = no left boundary
    max_x:      Optional[int]   # None = no right boundary

    counter:    int  = 0
    threshold:  int  = 4
    drain_rate: int  = 1

    spray_active: bool = False

    total_triggers:    int   = 0
    total_detections:  int   = 0
    last_trigger_time: float = 0.0

    current_detections: List[Detection] = field(default_factory=list)

    def contains_x(self, cx: int) -> bool:
        """
        True if pixel x-coordinate cx falls within this camera's
        OWNED range -- i.e. this camera should act on a detection at
        this position rather than deferring to a neighbor.
        """
        if self.min_x is not None and cx < self.min_x:
            return False
        if self.max_x is not None and cx > self.max_x:
            return False
        return True

    def to_dict(self) -> Dict:
        return {
            "zone_id":        self.zone_id,
            "label":          self.label,
            "nozzle_id":      self.nozzle_id,
            "camera_index":   self.camera_index,
            "counter":        self.counter,
            "threshold":      self.threshold,
            "spray_active":   self.spray_active,
            "total_triggers": self.total_triggers,
            "detections_now": len(self.current_detections),
            "min_x":          self.min_x,
            "max_x":          self.max_x,
        }


@dataclass
class ZoneDecision:
    """Output from ZoneManagerTriple for one frame across all cameras."""
    zones:            List[ZoneState]
    nozzles_to_fire:  List[int]
    nozzles_to_stop:  List[int]
    new_triggers:     List[int]
    new_releases:     List[int]
    timestamp:        float = field(default_factory=time.time)
    total_detections: int   = 0
    frame_id:         int   = 0

    @property
    def any_spray_active(self) -> bool:
        return len(self.nozzles_to_fire) > 0

    def to_dict(self) -> Dict:
        return {
            "frame_id":         self.frame_id,
            "nozzles_to_fire":  self.nozzles_to_fire,
            "nozzles_to_stop":  self.nozzles_to_stop,
            "new_triggers":     self.new_triggers,
            "new_releases":     self.new_releases,
            "any_active":       self.any_spray_active,
            "total_detections": self.total_detections,
            "zones":            [z.to_dict() for z in self.zones],
            "timestamp":        self.timestamp,
        }


# ─────────────────────────────────────────────────────────────
#  ZONE MANAGER — TRIPLE CAMERA
# ─────────────────────────────────────────────────────────────

class ZoneManagerTriple:
    """
    Maps per-camera YOLO detections (one detection list per camera,
    N cameras) to N spray zones -- one zone per camera, one camera
    per nozzle -- with the same validated passive debounce filter as
    the dual-camera system.

    Input shape deliberately does NOT depend on the dual-camera
    DualInferenceResult/InferenceResult wrapper types (which are
    hardwired to exactly two cameras, .left/.right) -- update() takes
    a plain List[List[Detection]], one list per camera, so this
    module has no dependency on whatever the eventual triple-camera
    detection engine's result type turns out to be (a later phase of
    this redesign). Wiring that phase to this one should be a thin
    adapter, not a change to this file.
    """

    def __init__(self, cfg: TripleZoneConfig):
        self.cfg   = cfg
        self.zones = self._build_zones()
        self._prev_spray_states = [False] * len(self.zones)
        self._frame_count = 0

        logging.info(
            f"ZoneManagerTriple initialized: {len(self.zones)} zones | "
            f"threshold={cfg.detection_threshold} | "
            f"drain_rate={cfg.drain_rate}"
        )
        for z in self.zones:
            logging.info(
                f"  {z.label}: nozzle=N{z.nozzle_id + 1} "
                f"owned_range=({z.min_x}, {z.max_x})"
            )

    # ── Zone construction ─────────────────────────────────────

    def _build_zones(self) -> List[ZoneState]:
        cfg = self.cfg
        ranges = cfg.owned_ranges()
        zones = []
        for i in range(cfg.camera_count):
            min_x, max_x = ranges[i]
            zones.append(ZoneState(
                zone_id      = i,
                nozzle_id    = cfg.zone_nozzle_map[i],
                camera_index = i,
                label        = cfg.camera_labels[i],
                min_x        = min_x,
                max_x        = max_x,
                threshold    = cfg.detection_threshold,
                drain_rate   = cfg.drain_rate,
            ))
        return zones

    # ── Detection assignment ──────────────────────────────────

    def _assign_detections(self, detections_by_camera: List[List[Detection]]):
        """
        Route each camera's detections to that SAME camera's zone
        only, and only if the detection's cx falls inside the zone's
        owned pixel range -- a detection outside the owned range is
        deliberately dropped here (not routed to any zone at all),
        since it belongs to a neighboring camera's owned range and
        will be (or already was) picked up there instead.

        Non-spray classes (the crop itself, at minimum) are filtered
        out before ever reaching a zone's current_detections, exactly
        like the dual-camera system -- this is the single point every
        spray-eligibility check flows through.
        """
        non_spray = set(self.cfg.non_spray_classes)

        for zone in self.zones:
            zone.current_detections = []

        n = min(len(detections_by_camera), len(self.zones))
        for cam_idx in range(n):
            zone = self.zones[cam_idx]
            for det in detections_by_camera[cam_idx]:
                if det.class_name in non_spray:
                    continue
                if zone.contains_x(det.cx):
                    zone.current_detections.append(det)
                # else: outside this camera's owned range -- belongs
                # to a neighbor, deliberately dropped here.

    # ── Debounce update ───────────────────────────────────────

    def update(self, detections_by_camera: List[List[Detection]],
               frame_id: int = 0, timestamp: Optional[float] = None
               ) -> ZoneDecision:
        """
        Process one frame's detections (one list per camera) and
        update all zone states. Same passive debounce filter as the
        dual-camera system:
          - Detection   → counter += 1  (up to threshold)
          - No detect   → counter -= drain_rate (down to 0)
          - active when counter >= threshold

        Unlike the dual-camera system, no zone shares a nozzle_id
        with another zone here (one camera per nozzle), so the
        "nozzle fires if ANY of its zones is active" OR-logic is
        vacuously just "the one zone for that nozzle" -- kept as the
        same general form anyway, so adding a genuinely shared zone
        later (if the hardware ever changes again) would not require
        touching this method.
        """
        self._frame_count += 1
        self._assign_detections(detections_by_camera)
        if timestamp is None:
            timestamp = time.time()

        new_triggers = []
        new_releases = []

        for i, zone in enumerate(self.zones):
            prev_active   = self._prev_spray_states[i]
            has_detection = len(zone.current_detections) > 0

            if has_detection:
                zone.counter = min(zone.counter + 1, zone.threshold)
                zone.total_detections += 1
            else:
                zone.counter = max(zone.counter - zone.drain_rate, 0)

            zone.spray_active = zone.counter >= zone.threshold

            if zone.spray_active and not prev_active:
                new_triggers.append(zone.zone_id)
                zone.total_triggers   += 1
                zone.last_trigger_time = time.time()
                logging.info(
                    f"🟢 {zone.label} TRIGGERED (N{zone.nozzle_id + 1}) | "
                    f"detections: {[d.class_name for d in zone.current_detections]}"
                )
            elif not zone.spray_active and prev_active:
                new_releases.append(zone.zone_id)
                logging.info(
                    f"🔴 {zone.label} RELEASED (N{zone.nozzle_id + 1})"
                )

            self._prev_spray_states[i] = zone.spray_active

        nozzle_active: Dict[int, bool] = {}
        for zone in self.zones:
            nid = zone.nozzle_id
            nozzle_active.setdefault(nid, False)
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
            timestamp        = timestamp,
            total_detections = total_detections,
            frame_id         = frame_id,
        )

    # ── Configuration updates ─────────────────────────────────

    def update_boundaries(self, cam1_max_x: int, cam2_min_x: int,
                          cam2_max_x: int, cam3_min_x: int):
        """Update the four owned-range boundaries after re-measurement."""
        self.cfg.CAM1_MAX_X = cam1_max_x
        self.cfg.CAM2_MIN_X = cam2_min_x
        self.cfg.CAM2_MAX_X = cam2_max_x
        self.cfg.CAM3_MIN_X = cam3_min_x

        ranges = self.cfg.owned_ranges()
        for i, zone in enumerate(self.zones):
            zone.min_x, zone.max_x = ranges[i]

        logging.info(
            f"ZoneManagerTriple: boundaries updated — "
            f"CAM1_MAX_X={cam1_max_x} CAM2_MIN_X={cam2_min_x} "
            f"CAM2_MAX_X={cam2_max_x} CAM3_MIN_X={cam3_min_x}"
        )

    def set_threshold(self, threshold: int):
        for zone in self.zones:
            zone.threshold = threshold
        logging.info(f"ZoneManagerTriple: all zones threshold → {threshold}")

    def set_drain_rate(self, rate: int):
        for zone in self.zones:
            zone.drain_rate = rate
        logging.info(f"ZoneManagerTriple: all zones drain rate → {rate}")

    def reset(self):
        for zone in self.zones:
            zone.counter            = 0
            zone.spray_active       = False
            zone.current_detections = []
        self._prev_spray_states = [False] * len(self.zones)
        self._frame_count       = 0
        logging.info("ZoneManagerTriple: reset")

    def _log_status(self):
        parts = []
        for z in self.zones:
            parts.append(
                f"{z.label}: {z.counter}/{z.threshold} "
                f"{'ACTIVE' if z.spray_active else 'idle'}"
            )
        logging.info(f"ZoneManagerTriple status — {' | '.join(parts)}")


# ─────────────────────────────────────────────────────────────
#  SELF TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys as _sys
    from pathlib import Path as _Path
    _ROOT = _Path(__file__).resolve().parent.parent
    if str(_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_ROOT))

    logging.basicConfig(level=logging.WARNING)

    from core.detection_engine_rgb import Detection

    def det(cls_name, cx, conf=0.9, camera="test"):
        # Detection's cx/cy are derived properties from x1..y2, so
        # build a small box centered exactly at cx for a clean,
        # unambiguous test coordinate.
        return Detection(class_id=0, class_name=cls_name, confidence=conf,
                         x1=cx - 10, y1=500, x2=cx + 10, y2=520,
                         camera=camera)

    print("=" * 60)
    print("zone_manager_triple.py — self test")
    print("=" * 60)

    cfg = TripleZoneConfig()
    print(f"Owned ranges: {cfg.owned_ranges()}")
    assert cfg.owned_ranges() == [
        (None, 1672), (360, 1620), (372, None)
    ]
    print("✓ Default calibration matches the measured boundaries "
          "(Cam1 <1672, 360<Cam2<1620, Cam3 >372)")

    # ── Test 1: a detection well inside Cam 1's owned range fires N1 ──
    mgr = ZoneManagerTriple(TripleZoneConfig())
    for _ in range(4):
        decision = mgr.update([[det("kochia", 500)], [], []])
    assert decision.nozzles_to_fire == [0]
    print(f"✓ Detection at cx=500 in Cam 1 (well inside its owned "
          f"range) fires N1 only: {decision.nozzles_to_fire}")

    # ── Test 2: a detection in Cam 1's frame but PAST its owned
    # boundary (belongs to Cam 2) is ignored by Cam 1 -- and does NOT
    # fire N1, since it was never actually detected by Cam 2 in this
    # test (Cam 2's detection list is empty) ──
    mgr2 = ZoneManagerTriple(TripleZoneConfig())
    for _ in range(6):
        decision2 = mgr2.update([[det("kochia", 1800)], [], []])
    assert decision2.nozzles_to_fire == [], (
        f"a detection past Cam 1's owned boundary (1800 > 1672) must "
        f"NOT fire N1 -- got {decision2.nozzles_to_fire}")
    print(f"✓ Detection at cx=1800 in Cam 1's frame (PAST its 1672 "
          f"boundary, in the overlap with Cam 2) correctly does NOT "
          f"fire N1 -- it's outside Cam 1's owned range")

    # ── Test 3: the SAME physical overlap position, but now also
    # genuinely detected by Cam 2 in ITS owned range -- N2 fires,
    # confirming the overlap detection is picked up by the RIGHT
    # neighbor rather than being lost entirely ──
    mgr3 = ZoneManagerTriple(TripleZoneConfig())
    for _ in range(6):
        decision3 = mgr3.update([[], [det("kochia", 500)], []])
    assert decision3.nozzles_to_fire == [1]
    print(f"✓ The same physical plant, correctly detected by Cam 2 "
          f"within ITS OWN owned range (cx=500, well inside "
          f"360<x<1620), fires N2 -- confirms overlap-zone detections "
          f"are the neighbor's responsibility, not lost")

    # ── Test 4: exactly one nozzle fires per frame in a realistic
    # scenario where ALL THREE cameras see something simultaneously,
    # each within its own owned range -- no cross-firing ──
    mgr4 = ZoneManagerTriple(TripleZoneConfig())
    for _ in range(6):
        decision4 = mgr4.update([
            [det("kochia", 800)],           # Cam 1, well within range
            [det("common_ragweed", 1000)],  # Cam 2, well within range
            [det("kochia", 1000)],          # Cam 3, well within range
        ])
    assert decision4.nozzles_to_fire == [0, 1, 2]
    print(f"✓ All three cameras detecting simultaneously, each within "
          f"its own owned range, correctly fires all three nozzles "
          f"independently: {decision4.nozzles_to_fire}")

    # ── Test 5: the crop (sugarbeet) never fires a nozzle, regardless
    # of position -- same non-spray-class guarantee as the dual-camera
    # system ──
    mgr5 = ZoneManagerTriple(TripleZoneConfig())
    for _ in range(10):
        decision5 = mgr5.update([[det("sugarbeet", 500)], [], []])
    assert decision5.nozzles_to_fire == []
    print(f"✓ sugarbeet (the crop) never fires a nozzle regardless of "
          f"position or how many frames it's detected in")

    # ── Test 6: debounce genuinely requires sustained detection --
    # a single frame's detection does not fire, matching the
    # dual-camera system's validated threshold behavior ──
    mgr6 = ZoneManagerTriple(TripleZoneConfig())
    decision6 = mgr6.update([[det("kochia", 500)], [], []])
    assert decision6.nozzles_to_fire == []
    print(f"✓ A single frame's detection alone does not fire "
          f"(threshold={cfg.detection_threshold} frames required) — "
          f"same debounce guarantee as the dual-camera system")

    # ── Test 7: exactly at the boundary pixel is treated as OWNED
    # (inclusive), matching contains_x()'s <= semantics ──
    mgr7 = ZoneManagerTriple(TripleZoneConfig())
    for _ in range(6):
        decision7 = mgr7.update([[det("kochia", 1672)], [], []])
    assert decision7.nozzles_to_fire == [0], (
        "the boundary pixel itself (1672) should be inclusive to Cam 1")
    print(f"✓ Detection exactly AT the boundary pixel (1672) is "
          f"correctly treated as owned by Cam 1 (inclusive boundary)")

    print()
    print("=" * 60)
    print("zone_manager_triple.py ✓  ALL TESTS PASSED")
    print("=" * 60)
