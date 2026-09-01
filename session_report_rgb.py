#!/usr/bin/env python3
"""
session_report_rgb.py
ABEN Dual RGB Detection System — Session Telemetry & Reporting

RGB fork of session_report.py.

Key differences from multispectral version:
  - SystemConfigSnapshot: RGB camera fields (left/right device, geometry)
    instead of spectral bands / resistance classifier fields
  - SprayEventRecord: trigger_dist_m + spray_time_s (geometry-computed)
    instead of resistance_prob / resistance_decision
  - No resistance classifier breakdown in summary
  - Dual-camera sync stats added to camera section
  - CameraStats tracks left/right grabs separately + sync error

Usage (wired into spray_mission_rgb.py):
    report = SessionReportRGB(system_config)
    report.record_frame(...)
    report.record_camera_grab(success=True, sync_error_ms=12.3)
    ev_record = report.record_spray_trigger(nozzle, zone_name, x, y,
                    confirming_classes, confirming_confidences,
                    trigger_dist_m, spray_time_s)
    report.record_spray_fire(ev_record, x, y, distance)
    report.record_spray_close(ev_record, x, y)
    report.record_zone_counter(zone_name, counter, threshold)
    report.finalize(distance_traveled, target, abort_reason)
    report.print_console_summary()
    report.save_json("session_rgb.json")
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────────
#  DATA RECORDS
# ─────────────────────────────────────────────────────────────

@dataclass
class SystemConfigSnapshot:
    # Camera — eMeet dual RGB
    camera_model: str = "eMeet C960 4K (Dual RGB)"
    left_device:  str = "usb-EMEET_EMEET_SmartCam_C960_4K_A241213000400860-video-index0"
    right_device: str = "usb-EMEET_EMEET_SmartCam_C960_4K_A241217000804000-video-index0"
    resolution:   str = "1920x1080"
    fps:          int = 30
    # Camera geometry (calibrated 2026-06-26)
    camera_height_m:  float = 0.9271    # 36.5 inches
    look_ahead_m:     float = 0.1778    # 7 inches (camera ahead of nozzles)
    gsd_mm_per_px:    float = 0.782     # ground sample distance mm/px
    nozzle_y_px:      int   = 767       # nozzle line Y in 1080p frame
    # Zone calibration (measured 2026-06-26)
    b1_split_x:   int = 1150   # cam1 Zone A/B1 boundary px
    b2_split_x:   int = 900    # cam2 Zone B2/C boundary px
    n1_center_px: int = 600    # N1 center in cam1
    n2_center_cam1_px: int = 1700  # N2 center in cam1
    n2_center_cam2_px: int = 400   # N2 center in cam2
    n3_center_px: int = 1400   # N3 center in cam2
    # Model
    model_path:           str   = "none"
    imgsz:                int   = 640
    confidence_threshold: float = 0.30
    iou_threshold:        float = 0.45
    device:               str   = "cuda:0"
    detection_mode:       str   = "LIVE"   # LIVE or DUMMY
    # Zones
    num_zones:      int = 4    # A, B1, B2, C
    zone_threshold: int = 4    # debounce frames to confirm
    nozzle_count:   int = 3    # N1, N2, N3
    # Spray timing
    min_spray_dist_m: float = 0.05   # minimum spray window
    max_spray_dist_m: float = 0.30   # maximum spray window
    min_speed_mps:    float = 0.05   # minimum robot speed
    # Robot
    drive_speed_mps:   float = 0.3
    target_distance_m: float = 1.0
    # Session
    session_start_iso: str = ""
    field_id:          str = ""
    researcher:        str = ""


@dataclass
class FrameRecord:
    frame_index: int
    timestamp: float
    preprocess_ms: float
    inference_ms: float
    total_ms: float
    detection_count: int
    detections: List[Dict] = field(default_factory=list)  # [{class_name, confidence}, ...]
    robot_x: float = 0.0
    robot_y: float = 0.0
    robot_speed: float = 0.0
    distance_traveled: float = 0.0


@dataclass
class SprayEventRecord:
    event_id: int
    nozzle: int
    zone_name: str
    trigger_time: float
    trigger_x: float
    trigger_y: float
    confirming_classes: List[str] = field(default_factory=list)
    confirming_confidences: List[float] = field(default_factory=list)
    # Geometry-computed spray timing
    trigger_dist_m: float = 0.0    # distance robot travels before nozzle fires
    spray_time_s:   float = 0.0    # nozzle open duration
    plant_width_px: int   = 0      # plant width in pixels (used for spray_time)
    fire_time: Optional[float] = None
    fire_x: Optional[float] = None
    fire_y: Optional[float] = None
    fire_distance_m: Optional[float] = None
    close_time: Optional[float] = None
    close_x: Optional[float] = None
    close_y: Optional[float] = None

    @property
    def fired(self) -> bool:
        return self.fire_time is not None

    @property
    def closed(self) -> bool:
        return self.close_time is not None

    @property
    def missed(self) -> bool:
        """A confirmed detection that never actually got sprayed —
        the run ended (target reached / timeout / stall / interrupt)
        before the nozzle reached the look-ahead distance."""
        return not self.fired

    @property
    def duration_s(self) -> Optional[float]:
        if self.fire_time is not None and self.close_time is not None:
            return self.close_time - self.fire_time
        return None


@dataclass
class SubThresholdRecord:
    """A zone that saw detections building up but never confirmed —
    counter rose above 0 but drained back down before crossing
    threshold. Worth tracking: these are weeds the system 'almost'
    caught, useful for tuning confidence/debounce threshold."""
    zone_name: str
    first_seen_time: float
    max_counter_reached: int
    threshold: int


@dataclass
class CameraStats:
    grabs_attempted: int   = 0
    grabs_succeeded: int   = 0
    grabs_failed:    int   = 0
    # Dual-camera sync tracking
    sync_errors_over_50ms: int   = 0
    sync_error_ms_sum:     float = 0.0
    sync_error_ms_max:     float = 0.0

    @property
    def success_rate(self) -> float:
        if self.grabs_attempted == 0:
            return 0.0
        return self.grabs_succeeded / self.grabs_attempted

    @property
    def avg_sync_error_ms(self) -> float:
        if self.grabs_succeeded == 0:
            return 0.0
        return self.sync_error_ms_sum / self.grabs_succeeded


# ─────────────────────────────────────────────────────────────
#  SESSION REPORT
# ─────────────────────────────────────────────────────────────

class SessionReportRGB:
    def __init__(self, system_config: SystemConfigSnapshot):
        self.system_config = system_config
        # Per-frame data is AGGREGATED, not stored raw — a 10-min mission
        # at ~9fps would otherwise produce a multi-MB JSON of redundant
        # frame records. We keep:
        #   - running timing stats (count/sum/min/max + reservoir for p95)
        #   - per-class detection counts + confidence sums
        #   - detection TRANSITIONS (class set changes between frames)
        #   - a downsampled trajectory (1 point per ~2s) for path plots
        self._n_frames        = 0
        self._first_frame_ts: Optional[float] = None
        self._last_frame_ts:  Optional[float] = None
        self._t_inf  = {"sum": 0.0, "min": None, "max": 0.0}
        self._t_pre  = {"sum": 0.0, "min": None, "max": 0.0}
        self._t_tot  = {"sum": 0.0, "min": None, "max": 0.0}
        self._inf_samples: List[float] = []   # reservoir for p95 (cap 2000)
        self._det_class_counts: Dict[str, int]   = {}
        self._det_class_conf:   Dict[str, float] = {}
        self._det_conf_min: Optional[float] = None
        self._det_conf_max: Optional[float] = None
        self._total_detections = 0
        self._prev_class_set: frozenset = frozenset()
        self._pending_set:    frozenset = frozenset()
        self._pending_count:  int = 0
        self.detection_transitions: List[Dict] = []
        self.trajectory: List[Dict] = []
        self._last_traj_ts = 0.0
        self._speed_sum = 0.0
        self.spray_events: List[SprayEventRecord] = []
        self.subthreshold_events: List[SubThresholdRecord] = []
        self.camera_stats = CameraStats()

        self._start_time = time.time()
        self._end_time: Optional[float] = None
        self._next_event_id = 1

        # zone_name -> running max counter seen since it was last 0,
        # used to detect "rose above 0 but never crossed threshold"
        self._zone_peak_counter: Dict[str, int] = {}
        self._zone_peak_time: Dict[str, float] = {}

        self.distance_traveled_final: Optional[float] = None
        self.target_distance: Optional[float] = None
        self.duration_s: Optional[float] = None
        self.abort_reason: str = "completed"

    # ── Recording ────────────────────────────────────────────

    def record_frame(self, frame_index: int, preprocess_ms: float,
                      inference_ms: float, total_ms: float,
                      detections: List[Dict], robot_x: float, robot_y: float,
                      robot_speed: float, distance_traveled: float):
        """
        Same signature as before, but aggregates instead of storing
        every frame. JSON stays small regardless of mission length.
        """
        now = time.time()
        if self._first_frame_ts is None:
            self._first_frame_ts = now
        self._last_frame_ts = now
        self._n_frames += 1
        self._speed_sum += robot_speed

        # Timing aggregates
        for agg, val in ((self._t_inf, inference_ms),
                         (self._t_pre, preprocess_ms),
                         (self._t_tot, total_ms)):
            agg["sum"] += val
            agg["max"] = max(agg["max"], val)
            agg["min"] = val if agg["min"] is None else min(agg["min"], val)
        if len(self._inf_samples) < 2000:
            self._inf_samples.append(inference_ms)

        # Per-class detection aggregates
        for d in detections:
            name = d.get("class_name", "?")
            conf = float(d.get("confidence", 0.0))
            self._det_class_counts[name] = \
                self._det_class_counts.get(name, 0) + 1
            self._det_class_conf[name] = \
                self._det_class_conf.get(name, 0.0) + conf
            self._total_detections += 1
            self._det_conf_min = (conf if self._det_conf_min is None
                                  else min(self._det_conf_min, conf))
            self._det_conf_max = (conf if self._det_conf_max is None
                                  else max(self._det_conf_max, conf))

        # Detection TRANSITIONS — record only when the visible class
        # set changes AND stays changed for 3 consecutive frames.
        # Debouncing prevents a transition storm from YOLO flicker
        # (plants popping in/out near the confidence threshold).
        cur_set = frozenset(d.get("class_name", "?") for d in detections)
        if cur_set == self._pending_set:
            self._pending_count += 1
        else:
            self._pending_set   = cur_set
            self._pending_count = 1

        TRANSITION_DEBOUNCE = 3     # frames the new set must persist
        MAX_TRANSITIONS     = 500   # hard cap on stored transitions

        if (self._pending_count == TRANSITION_DEBOUNCE
                and cur_set != self._prev_class_set
                and len(self.detection_transitions) < MAX_TRANSITIONS):
            self.detection_transitions.append({
                "frame":     frame_index,
                "t":         round(now - (self._first_frame_ts or now), 2),
                "appeared":  sorted(cur_set - self._prev_class_set),
                "left":      sorted(self._prev_class_set - cur_set),
                "now":       sorted(cur_set),
                "x":         round(robot_x, 3),
                "dist":      round(distance_traveled, 3),
            })
            self._prev_class_set = cur_set

        # Trajectory — one point every 2 seconds max
        if now - self._last_traj_ts >= 2.0:
            self.trajectory.append({
                "t":     round(now - (self._first_frame_ts or now), 1),
                "x":     round(robot_x, 3),
                "y":     round(robot_y, 3),
                "speed": round(robot_speed, 3),
                "dist":  round(distance_traveled, 3),
            })
            self._last_traj_ts = now

    def record_camera_grab(self, success: bool,
                           sync_error_ms: float = 0.0):
        self.camera_stats.grabs_attempted += 1
        if success:
            self.camera_stats.grabs_succeeded += 1
            self.camera_stats.sync_error_ms_sum += sync_error_ms
            if sync_error_ms > self.camera_stats.sync_error_ms_max:
                self.camera_stats.sync_error_ms_max = sync_error_ms
            if sync_error_ms > 50.0:
                self.camera_stats.sync_errors_over_50ms += 1
        else:
            self.camera_stats.grabs_failed += 1

    def record_spray_trigger(self, nozzle: int, zone_name: str,
                              x: float, y: float,
                              confirming_classes: List[str],
                              confirming_confidences: List[float],
                              trigger_dist_m: float = 0.0,
                              spray_time_s: float = 0.0,
                              plant_width_px: int = 0,
                              ) -> SprayEventRecord:
        rec = SprayEventRecord(
            event_id=self._next_event_id, nozzle=nozzle, zone_name=zone_name,
            trigger_time=time.time(), trigger_x=x, trigger_y=y,
            confirming_classes=confirming_classes,
            confirming_confidences=confirming_confidences,
            trigger_dist_m=trigger_dist_m,
            spray_time_s=spray_time_s,
            plant_width_px=plant_width_px,
        )
        self._next_event_id += 1
        self.spray_events.append(rec)
        # Triggering resets the "near miss" tracking for this zone —
        # it wasn't a near miss, it confirmed.
        self._zone_peak_counter.pop(zone_name, None)
        return rec

    def record_spray_fire(self, rec: SprayEventRecord, x: float, y: float,
                           distance_m: float = None,
                           distance: float = None):
        """distance= is alias for distance_m= for compatibility."""
        rec.fire_time = time.time()
        rec.fire_x, rec.fire_y = x, y
        rec.fire_distance_m = distance_m if distance_m is not None else distance

    def record_spray_close(self, rec: SprayEventRecord, x: float, y: float):
        rec.close_time = time.time()
        rec.close_x, rec.close_y = x, y

    def record_zone_counter(self, zone_name: str, counter: int, threshold: int):
        """Call every frame for every zone, regardless of whether it
        triggered, to track 'almost triggered' near misses."""
        if counter <= 0:
            # Zone drained back to 0 — if it had peaked above 0 without
            # confirming, log it as a near miss.
            peak = self._zone_peak_counter.pop(zone_name, 0)
            if peak > 0:
                self.subthreshold_events.append(SubThresholdRecord(
                    zone_name=zone_name,
                    first_seen_time=self._zone_peak_time.get(zone_name, time.time()),
                    max_counter_reached=peak, threshold=threshold,
                ))
            self._zone_peak_time.pop(zone_name, None)
        else:
            prev_peak = self._zone_peak_counter.get(zone_name, 0)
            if counter > prev_peak:
                self._zone_peak_counter[zone_name] = counter
                if zone_name not in self._zone_peak_time:
                    self._zone_peak_time[zone_name] = time.time()

    def finalize(self, distance_traveled: float,
                 target_distance: float = None,
                 abort_reason: str = "completed",
                 target: float = None):
        """target= is an alias for target_distance= for compatibility."""
        if target is not None and target_distance is None:
            target_distance = target
        self._end_time = time.time()
        self.distance_traveled_final = distance_traveled
        self.target_distance = target_distance or 0.0
        self.duration_s = self._end_time - self._start_time
        self.abort_reason = abort_reason

    # ── Summary computation ─────────────────────────────────────

    def compute_summary(self) -> Dict:
        s = {}

        # Robot kinematics
        commanded_speed = self.system_config.drive_speed_mps
        actual_speed = (self.distance_traveled_final / self.duration_s
                         if self.duration_s else 0.0)
        s["robot"] = {
            "target_distance_m": self.target_distance,
            "distance_traveled_m": self.distance_traveled_final,
            "duration_s": self.duration_s,
            "commanded_speed_mps": commanded_speed,
            "actual_avg_speed_mps": actual_speed,
            "speed_accuracy_pct": (100.0 * actual_speed / commanded_speed
                                    if commanded_speed else None),
            "abort_reason": self.abort_reason,
        }

        # Camera
        s["camera"] = {
            "model":                 self.system_config.camera_model,
            "left_device":           self.system_config.left_device,
            "right_device":          self.system_config.right_device,
            "resolution":            self.system_config.resolution,
            "grabs_attempted":       self.camera_stats.grabs_attempted,
            "grabs_succeeded":       self.camera_stats.grabs_succeeded,
            "grabs_failed":          self.camera_stats.grabs_failed,
            "success_rate_pct":      round(100 * self.camera_stats.success_rate, 1),
            "avg_sync_error_ms":     round(self.camera_stats.avg_sync_error_ms, 2),
            "max_sync_error_ms":     round(self.camera_stats.sync_error_ms_max, 2),
            "sync_errors_over_50ms": self.camera_stats.sync_errors_over_50ms,
        }

        # Detection / inference timing — from running aggregates
        if self._n_frames > 0:
            n = self._n_frames
            span = ((self._last_frame_ts - self._first_frame_ts)
                    if (self._first_frame_ts and self._last_frame_ts
                        and n > 1) else None)
            achieved_fps = ((n - 1) / span if span and span > 0 else None)

            def _agg_stats(agg, count):
                if count == 0 or agg["min"] is None:
                    return None
                return {
                    "mean": round(agg["sum"] / count, 1),
                    "min":  round(agg["min"], 1),
                    "max":  round(agg["max"], 1),
                }

            inf_stats = _agg_stats(self._t_inf, n)
            if inf_stats and self._inf_samples:
                srt = sorted(self._inf_samples)
                inf_stats["p95"] = round(
                    srt[max(0, int(len(srt) * 0.95) - 1)], 1)

            by_class = {}
            for cls, cnt in self._det_class_counts.items():
                mean_conf = (self._det_class_conf.get(cls, 0.0) / cnt
                             if cnt else 0.0)
                by_class[cls] = {
                    "count": cnt,
                    "confidence": {"mean": round(mean_conf, 2)},
                }

            conf_overall = None
            if self._total_detections > 0:
                total_conf = sum(self._det_class_conf.values())
                conf_overall = {
                    "mean": round(total_conf / self._total_detections, 2),
                    "min":  round(self._det_conf_min or 0, 2),
                    "max":  round(self._det_conf_max or 0, 2),
                }

            s["detection"] = {
                "frames_processed":      n,
                "achieved_fps":          round(achieved_fps, 2) if achieved_fps else None,
                "inference_ms":          inf_stats,
                "preprocess_ms":         _agg_stats(self._t_pre, n),
                "total_ms":              _agg_stats(self._t_tot, n),
                "total_detections":      self._total_detections,
                "confidence_overall":    conf_overall,
                "by_class":              by_class,
                "detection_transitions": len(self.detection_transitions),
            }
        else:
            s["detection"] = {"frames_processed": 0}

        # Spray events
        fired = [e for e in self.spray_events if e.fired]
        missed = [e for e in self.spray_events if e.missed]
        by_nozzle: Dict[int, int] = {}
        for e in self.spray_events:
            by_nozzle[e.nozzle] = by_nozzle.get(e.nozzle, 0) + 1
        fire_distances = [e.fire_distance_m for e in fired if e.fire_distance_m is not None]
        durations = [e.duration_s for e in fired if e.duration_s is not None]

        # Spray geometry stats
        trigger_dists = [e.trigger_dist_m for e in fired]
        spray_times   = [e.spray_time_s   for e in fired]
        plant_widths  = [e.plant_width_px for e in fired if e.plant_width_px > 0]

        s["spray"] = {
            "total_triggers":    len(self.spray_events),
            "fired":             len(fired),
            "missed":            len(missed),
            "missed_events": [
                {"event_id":    e.event_id,
                 "nozzle":      e.nozzle,
                 "zone":        e.zone_name,
                 "trigger_pos": [round(e.trigger_x, 3), round(e.trigger_y, 3)]}
                for e in missed
            ],
            "triggers_by_nozzle": by_nozzle,
            "fire_distance_m":    _stats(fire_distances) if fire_distances else None,
            "spray_duration_s":   _stats(durations)      if durations      else None,
            "near_misses":        len(self.subthreshold_events),
            "near_miss_by_zone":  _count_by(
                [e.zone_name for e in self.subthreshold_events]),
            # Geometry-based spray timing stats
            "trigger_dist_m":     _stats(trigger_dists) if trigger_dists else None,
            "spray_time_s":       _stats(spray_times)   if spray_times   else None,
            "plant_width_px":     _stats(plant_widths)  if plant_widths  else None,
        }

        return s

    # ── Output ──────────────────────────────────────────────

    def to_dict(self) -> Dict:
        return {
            "system_config": asdict(self.system_config),
            "summary": self.compute_summary(),
            # Per-frame data replaced with compact representations:
            # transitions = when detected class set changed (weed appeared, etc.)
            # trajectory  = robot path downsampled to 1 point / 2s
            "detection_transitions": self.detection_transitions,
            "trajectory": self.trajectory,
            "spray_events": [asdict(e) for e in self.spray_events],
            "subthreshold_events": [asdict(e) for e in self.subthreshold_events],
        }

    def save_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        print(f"[REPORT] Session telemetry written to: {path}")

    def print_console_summary(self):
        s = self.compute_summary()
        print(f"\n{'='*60}")
        print(f"  SESSION REPORT — {self.system_config.detection_mode} mode")
        print(f"{'='*60}")

        r = s["robot"]
        print(f"\nROBOT")
        print(f"  Target distance:     {r['target_distance_m']:.3f} m")
        print(f"  Distance traveled:   {r['distance_traveled_m']:.3f} m")
        print(f"  Duration:            {r['duration_s']:.1f} s")
        print(f"  Commanded speed:     {r['commanded_speed_mps']:.3f} m/s")
        print(f"  Actual avg speed:    {r['actual_avg_speed_mps']:.3f} m/s"
              + (f"  ({r['speed_accuracy_pct']:.0f}% of commanded)"
                 if r['speed_accuracy_pct'] else ""))
        print(f"  Ended because:       {r['abort_reason']}")

        c = s["camera"]
        print(f"\nCAMERA")
        print(f"  Model:               {c['model']}")
        print(f"  Frame grabs:         {c['grabs_succeeded']}/{c['grabs_attempted']} "
              f"succeeded ({c['success_rate_pct']}%)")

        d = s["detection"]
        print(f"\nDETECTION")
        print(f"  Frames processed:    {d.get('frames_processed', 0)}")
        if d.get("achieved_fps"):
            print(f"  Achieved frame rate: {d['achieved_fps']} fps")
        if "inference_ms" in d and d["inference_ms"]:
            im = d["inference_ms"]
            print(f"  Inference time:      mean={im['mean']:.1f}ms  "
                  f"min={im['min']:.1f}ms  max={im['max']:.1f}ms  "
                  f"p95={im['p95']:.1f}ms")
        print(f"  Total detections:    {d.get('total_detections', 0)}")
        if d.get("confidence_overall"):
            co = d["confidence_overall"]
            print(f"  Confidence:          mean={co['mean']:.2f}  "
                  f"min={co['min']:.2f}  max={co['max']:.2f}")
        for cls, info in d.get("by_class", {}).items():
            print(f"    {cls}: {info['count']} detections, "
                  f"avg conf={info['confidence']['mean']:.2f}")

        sp = s["spray"]
        print(f"\nSPRAY")
        print(f"  Total triggers:      {sp['total_triggers']}")
        print(f"  Fired:               {sp['fired']}")
        print(f"  Missed:              {sp['missed']}"
              + ("  ⚠ confirmed detections never reached the nozzle"
                 if sp['missed'] else ""))
        print(f"  Near misses:         {sp['near_misses']} "
              f"(detected but never confirmed across {self.system_config.zone_threshold} frames)")
        for nozzle, count in sorted(sp["triggers_by_nozzle"].items()):
            print(f"    N{int(nozzle)+1}: {count} trigger(s)")
        if sp.get("fire_distance_m"):
            fd = sp["fire_distance_m"]
            print(f"  Fire distance:       mean={fd['mean']:.3f}m  "
                  f"(target look-ahead={self.system_config.look_ahead_m:.3f}m)")
        if sp.get("spray_duration_s"):
            dur = sp["spray_duration_s"]
            print(f"  Spray duration:      mean={dur['mean']:.2f}s")

        # Geometry stats
        if sp.get("trigger_dist_m"):
            td = sp["trigger_dist_m"]
            print(f"  Trigger dist (mean): {td['mean']:.3f}m  "
                  f"min={td['min']:.3f}m  max={td['max']:.3f}m")
        if sp.get("spray_time_s"):
            st = sp["spray_time_s"]
            print(f"  Spray time (mean):   {st['mean']:.3f}s  "
                  f"min={st['min']:.3f}s  max={st['max']:.3f}s")

        # Sync stats
        c = s["camera"]
        if c.get("avg_sync_error_ms") is not None:
            print(f"\nDUAL-CAMERA SYNC")
            print(f"  Avg sync error:      {c['avg_sync_error_ms']:.1f} ms")
            print(f"  Max sync error:      {c['max_sync_error_ms']:.1f} ms")
            print(f"  Sync errors >50ms:   {c['sync_errors_over_50ms']}")

        print(f"\n{'='*60}\n")


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def _stats(values: List[float]) -> Optional[Dict]:
    if not values:
        return None
    sorted_v = sorted(values)
    n = len(sorted_v)
    p95_idx = min(n - 1, int(round(0.95 * (n - 1))))
    return {
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "p95": sorted_v[p95_idx],
        "n": n,
    }


def _count_by(items: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out
