#!/usr/bin/env python3
"""
spray_mission.py
Dual RGB Detection System — Autonomous Spray Mission

Drives the Husky robot, captures dual RGB frames from the eMeet cameras,
runs YOLOv8n-Seg inference, and fires nozzles for all confirmed weed
detections using geometry-based look-ahead timing.

Pipeline per frame:
  eMeet C960 4K (Left + Right) → YOLOv8n-Seg → per-plant masks
       → ZoneManagerRGB (B1/B2 OR logic, debounce threshold)
       → GeometryConfig (trigger_dist_m + spray_time_s per detection)
       → DistanceBufferedZone → Arduino nozzle control

Spray decision:
  class == "weed"      → spray (geometry-timed)
  class == "sugarbeet" → skip (never spray)

Usage:
    cd /media/pagsun/Transcend/phd_project/emeet_dual_cam

    # Full autonomous mission:
    python spray_mission_rgb.py \
        --model   models/weed_rgb.pt \
        --dist    3.0 \
        --speed   0.3 \
        --field-id Wilkin_County_Plot_A

    # Dry run — log only, no hardware:
    python spray_mission_rgb.py --dry-run --dist 2.0

    # Dummy detect — random detections, no camera/model needed:
    python spray_mission_rgb.py --dummy-detect --dist 2.0

Author : Nana | NDSU / PhD Imaging System
Path   : /media/pagsun/Transcend/phd_project/emeet_dual_cam/
"""

from __future__ import annotations

import math
import time
import random
import argparse
import threading
import logging
import json
import subprocess
import statistics
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

# ─────────────────────────────────────────────────────────────
#  RGB detection pipeline
# ─────────────────────────────────────────────────────────────

DETECTION_AVAILABLE = False
try:
    from core.detection_engine_rgb import RGBDetectionEngine
    from core.zone_manager_rgb import (
        ZoneManagerRGB, ZONE_A, ZONE_B1, ZONE_B2, ZONE_C
    )
    from core.detection_config_rgb import (
        RGBConfig, get_weed_config, GrowthStage,
        DetectionMode, GeometryConfig
    )
    DETECTION_AVAILABLE = True
except ImportError:
    try:
        from core.detection_engine_rgb import RGBDetectionEngine
        from core.zone_manager_rgb import (
            ZoneManagerRGB, ZONE_A, ZONE_B1, ZONE_B2, ZONE_C
        )
        from core.detection_config_rgb import (
            RGBConfig, get_weed_config, GrowthStage,
            DetectionMode, GeometryConfig
        )
        DETECTION_AVAILABLE = True
    except ImportError as e:
        print(f"[MISSION] RGB detection stack not importable ({e}) "
              f"— use --dummy-detect")

# ─────────────────────────────────────────────────────────────
#  eMeet dual camera
# ─────────────────────────────────────────────────────────────

CAMERA_AVAILABLE = False
LEFT_CAMERA = None   # resolved when DualEMEETCamera imports
try:
    from core.dual_emeet_camera import DualEMEETCamera, LEFT_CAMERA
    CAMERA_AVAILABLE = True
except ImportError:
    try:
        from core.dual_emeet_camera import DualEMEETCamera, LEFT_CAMERA
        CAMERA_AVAILABLE = True
    except ImportError as e:
        print(f"[MISSION] DualEMEETCamera not importable ({e})")

# ─────────────────────────────────────────────────────────────
#  Arduino serial
# ─────────────────────────────────────────────────────────────

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
#  Session report
# ─────────────────────────────────────────────────────────────

REPORT_AVAILABLE = False
try:
    from session_report_rgb import (
        SystemConfigSnapshot, SessionReportRGB as SessionReport
    )
    REPORT_AVAILABLE = True
except ImportError as e:
    print(f"[MISSION] session_report_rgb.py not importable ({e}) "
          f"— continuing without telemetry")


# ─────────────────────────────────────────────────────────────
#  CAMERA GRABBER
# ─────────────────────────────────────────────────────────────

def _read_v4l2_settings(device: str) -> dict:
    """
    Read current v4l2 settings from device.
    Returns dict with keys matching DualEMEETCamera.DEFAULTS.
    Falls back silently — missing keys use DEFAULTS in DualEMEETCamera.
    """
    controls = {
        "exposure":   "exposure_time_absolute",
        "brightness": "brightness",
        "contrast":   "contrast",
        "saturation": "saturation",
        "gamma":      "gamma",
        "gain":       "gain",
        "sharpness":  "sharpness",
        "wb_temp":    "white_balance_temperature",
        "focus":      "focus_absolute",
    }
    settings = {}
    for key, ctrl in controls.items():
        try:
            result = subprocess.run(
                ["v4l2-ctl", "-d", device, "--get-ctrl", ctrl],
                capture_output=True, text=True, timeout=2
            )
            # Output format: "brightness: 0"
            val = int(result.stdout.strip().split(":")[-1].strip())
            settings[key] = val
        except Exception:
            pass
    return settings


class RGBCameraGrabber:
    """
    Minimal dual eMeet camera wrapper for spray_mission_rgb.py.
    No Qt — start, grab pairs, close.

    Reads current v4l2 settings from the camera before opening,
    so the spray mission uses the same settings as the GUI
    (outdoor/indoor preset applied by the operator before the run).
    """

    def __init__(self):
        self._cam: Optional[DualEMEETCamera] = None
        if not CAMERA_AVAILABLE:
            print("[CAMERA] DualEMEETCamera unavailable")
            return
        try:
            # Read what the camera is currently set to
            # (GUI may have applied outdoor/indoor preset already)
            # Read current v4l2 settings if device path is known
            current = (_read_v4l2_settings(LEFT_CAMERA)
                       if LEFT_CAMERA else {})
            if current:
                print(f"[CAMERA] Current v4l2 settings: "
                      f"exp={current.get('exposure','?')}  "
                      f"gamma={current.get('gamma','?')}  "
                      f"wb={current.get('wb_temp','?')}")
                # Pass to DualEMEETCamera so it doesn't reset to defaults
                self._cam = DualEMEETCamera(settings=current)
            else:
                print("[CAMERA] Could not read v4l2 settings — using defaults")
                self._cam = DualEMEETCamera()
            self._cam.start()
            print("[CAMERA] eMeet dual cameras started (1920x1080 @ 30fps)")
        except Exception as e:
            print(f"[CAMERA] Could not start eMeet cameras: {e}")
            self._cam = None

    @property
    def connected(self) -> bool:
        return self._cam is not None

    def grab_pair(self):
        if self._cam is None:
            return None
        try:
            return self._cam.read_pair()
        except Exception as e:
            logging.warning(f"[CAMERA] Grab failed: {e}")
            return None

    def close(self):
        if self._cam is not None:
            try:
                self._cam.stop()
            except Exception:
                pass
            self._cam = None


# ─────────────────────────────────────────────────────────────
#  ARDUINO CONTROLLER
# ─────────────────────────────────────────────────────────────

class ArduinoController:
    """Serial interface to Arduino Mega (TB6600 + solenoid nozzles)."""

    def __init__(self, dry_run: bool, port: str = "/dev/ttyACM0",
                 baud: int = 9600):
        self.dry_run = dry_run
        self._ser = None
        self._lock = threading.Lock()

        if dry_run:
            print("[ARDUINO] DRY RUN — no hardware commands")
            return
        if not SERIAL_AVAILABLE:
            print("[ARDUINO] pyserial not installed — dry run")
            self.dry_run = True
            return
        try:
            self._ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2.0)
            print(f"[ARDUINO] Connected: {port} @ {baud}")
        except Exception as e:
            print(f"[ARDUINO] Connect failed ({e}) — dry run")
            self.dry_run = True

    def send(self, cmd: str):
        if self.dry_run:
            print(f"[DRY]  {cmd}")
            return
        with self._lock:
            try:
                if self._ser and self._ser.is_open:
                    self._ser.write(f"{cmd}\n".encode())
            except Exception as e:
                logging.warning(f"[ARDUINO] Send failed: {e}")

    def pump_on(self):             self.send("pump on")
    def pump_off(self):            self.send("pump off")
    def nozzle_on(self, n: int):   self.send(f"n{n+1} on")
    def nozzle_off(self, n: int):  self.send(f"n{n+1} off")
    def all_off(self):
        self.send("na off")
        self.pump_off()

    def close(self):
        self.all_off()
        if self._ser:
            try:
                self._ser.close()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────
#  ODOMETRY RECEIVER
# ─────────────────────────────────────────────────────────────

class OdomReceiver:
    """Receives /odometry/filtered via UDP from husky_odom_pub.py."""

    def __init__(self, port: int = 5006):
        import socket
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("", port))
        self._sock.settimeout(0.01)
        self._pose = {'x': 0.0, 'y': 0.0, 'heading': 0.0, 'speed': 0.0}
        self._dist = 0.0
        self._start = time.time()
        self._running = True
        self._thread = threading.Thread(
            target=self._recv_loop, daemon=True)
        self._thread.start()
        print(f"[ODOM] UDP listener on :{port}  (spray mission port)")

    def _recv_loop(self):
        while self._running:
            try:
                data, _ = self._sock.recvfrom(512)
                msg  = json.loads(data.decode())
                prev = dict(self._pose)
                self._pose = {
                    'x':       float(msg.get('x', 0)),
                    'y':       float(msg.get('y', 0)),
                    'heading': float(msg.get('yaw', 0)),
                    'speed':   float(msg.get('speed', 0)),
                }
                dx = self._pose['x'] - prev['x']
                dy = self._pose['y'] - prev['y']
                self._dist += math.sqrt(dx*dx + dy*dy)
            except Exception:
                pass

    @property
    def distance_traveled(self) -> float:
        return self._dist

    def get_pose(self) -> dict:
        return dict(self._pose)

    def stop(self):
        self._running = False
        self._sock.close()


# ─────────────────────────────────────────────────────────────
#  HUSKY DRIVER
# ─────────────────────────────────────────────────────────────

HUSKY_IP   = "192.168.131.1"
HUSKY_USER = "administrator"
ROS_SOURCE = "source /opt/ros/noetic/setup.bash && "
NAV_SCRIPT = "~/test_nav_v3.py"


class TimebasedOdom:
    """
    Fallback when UDP odom is unavailable.
    Estimates distance from commanded_speed × elapsed_time.
    Also provides a synthetic pose so DistanceBufferedZone works.
    """
    def __init__(self, speed: float):
        self._speed    = speed
        self._start    = time.time()
        self._dist     = 0.0
        self._last_t   = time.time()

    @property
    def distance_traveled(self) -> float:
        now = time.time()
        dt  = now - self._last_t
        self._dist   += self._speed * dt
        self._last_t  = now
        return self._dist

    def get_pose(self) -> dict:
        d = self.distance_traveled
        return {'x': d, 'y': 0.0, 'heading': 0.0, 'speed': self._speed}

    def stop(self):
        pass


class HuskyDriver:
    def __init__(self, dry_run: bool):
        self.dry_run = dry_run

    def _ssh(self, cmd: str):
        full = (f"ssh -o StrictHostKeyChecking=no "
                f"{HUSKY_USER}@{HUSKY_IP} '{ROS_SOURCE}{cmd}'")
        if self.dry_run:
            print(f"[SSH] {cmd}")
            return
        subprocess.Popen(full, shell=True,
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)

    def start_forward(self, dist: float, speed: float):
        self._ssh(f"python3 {NAV_SCRIPT} forward {dist} --speed {speed}")
        print(f"[HUSKY] Forward {dist}m @ {speed}m/s")

    def stop(self):
        self._ssh(f"python3 {NAV_SCRIPT} stop")
        print("[HUSKY] Stop sent")


# ─────────────────────────────────────────────────────────────
#  PENDING SPRAY
# ─────────────────────────────────────────────────────────────

class PendingSpray:
    """One confirmed detection waiting to fire at trigger distance."""
    def __init__(self, nozzle_id: int, zone_name: str,
                 x: float, y: float,
                 trigger_dist_m: float, spray_time_s: float,
                 confirming_classes: List[str],
                 event_record=None):
        self.nozzle_id          = nozzle_id
        self.zone_name          = zone_name
        self.origin_x           = x
        self.origin_y           = y
        self.trigger_dist_m     = trigger_dist_m
        self.spray_time_s       = spray_time_s
        self.confirming_classes = confirming_classes
        self.event_record       = event_record
        self.fired              = False


# ─────────────────────────────────────────────────────────────
#  SPRAY MISSION RGB
# ─────────────────────────────────────────────────────────────

class SprayMissionRGB:
    """Autonomous weed detection and precision spray — dual RGB cameras."""

    def __init__(self,
                 model_path:   str   = "models/weed_rgb.pt",
                 dist:         float = 1.0,
                 speed:        float = 0.3,
                 arduino_port: str   = "/dev/ttyACM0",
                 odom_port:    int   = 5006,
                 dry_run:      bool  = False,
                 dummy_detect: bool  = False,
                 field_id:     str   = "",
                 researcher:   str   = "nana"):

        self.dist         = dist
        self.speed        = speed
        self.dry_run      = dry_run
        self.dummy_detect = dummy_detect or not DETECTION_AVAILABLE
        self.field_id     = field_id
        self.odom_port    = odom_port
        self._running     = False

        # ── RGB detection ─────────────────────────────────────
        self.engine  = None
        self.zones   = None
        self.rgb_cfg = None
        self.geo     = GeometryConfig()
        self.geo.__post_init__()

        if DETECTION_AVAILABLE and not self.dummy_detect:
            try:
                self.rgb_cfg = get_weed_config(field_id=field_id)
                self.rgb_cfg.model.weed_rgb_pt = Path(model_path)
                self.engine  = RGBDetectionEngine(self.rgb_cfg)
                self.zones   = ZoneManagerRGB(self.rgb_cfg)
                self.geo     = self.rgb_cfg.geometry
                self.geo.__post_init__()
                print(f"[MISSION] {self.geo.summary()}")
            except Exception as e:
                print(f"[MISSION] Detection init failed: {e}")
                self.dummy_detect = True

        # ── Camera ────────────────────────────────────────────
        self.camera = None
        if not dry_run and not self.dummy_detect:
            self.camera = RGBCameraGrabber()
            if not self.camera.connected:
                print("[MISSION] Camera unavailable — dummy detect")
                self.dummy_detect = True

        # ── Hardware ──────────────────────────────────────────
        self.arduino = ArduinoController(dry_run, arduino_port)
        self.husky   = HuskyDriver(dry_run)
        self.odom: Optional[OdomReceiver] = None

        # ── Pending sprays ────────────────────────────────────
        self._pending: List[PendingSpray] = []
        self._lock = threading.Lock()

        # ── Session report ────────────────────────────────────
        self.report = None
        if REPORT_AVAILABLE:
            snap = SystemConfigSnapshot(
                camera_model      = "eMeet C960 4K (Dual RGB)",
                camera_height_m   = self.geo.camera_height_m,
                look_ahead_m      = self.geo.look_ahead_m,
                gsd_mm_per_px     = self.geo.gsd_m_per_px * 1000,
                nozzle_y_px       = self.geo.nozzle_y_px,
                model_path        = model_path,
                drive_speed_mps   = speed,
                target_distance_m = dist,
                field_id          = field_id,
                researcher        = researcher,
                session_start_iso = datetime.utcnow().isoformat() + "Z",
                zone_threshold    = (self.rgb_cfg.zones.detection_threshold
                                     if self.rgb_cfg else 4),
                detection_mode    = "DUMMY" if self.dummy_detect else "LIVE",
            )
            self.report = SessionReport(snap)

        self._print_config()

    def _print_config(self):
        g = self.geo
        print(f"\n{'='*60}")
        print(f"  ABEN Dual RGB Spray Mission")
        print(f"{'='*60}")
        print(f"  Camera     :  eMeet C960 4K (Dual RGB)")
        print(f"  Mode       :  {'DUMMY' if self.dummy_detect else 'LIVE'}")
        print(f"  Dry run    :  {self.dry_run}")
        print(f"  Distance   :  {self.dist} m")
        print(f"  Speed      :  {self.speed} m/s")
        print(f"  Height     :  {g.camera_height_m:.4f} m  "
              f"({g.camera_height_m*39.37:.1f} in)")
        print(f"  Look-ahead :  {g.look_ahead_m:.4f} m  "
              f"({g.look_ahead_m*39.37:.1f} in)")
        print(f"  GSD        :  {g.gsd_m_per_px*1000:.2f} mm/px")
        print(f"  Nozzle Y   :  {g.nozzle_y_px} px  (in 1920x1080 frame)")
        print(f"  Field ID   :  {self.field_id or 'n/a'}")
        print(f"{'='*60}\n")

    # ── Dummy detections ──────────────────────────────────────

    def _dummy_detections(self) -> list:
        class FakeDet:
            def __init__(self):
                self.class_id   = random.choice([0, 1])
                self.class_name = "sugarbeet" if self.class_id==0 else "weed"
                self.confidence = round(random.uniform(0.55, 0.95), 2)
                self.cx  = random.randint(200, 1700)
                self.cy  = random.randint(300, 900)
                self.x1  = self.cx - 40
                self.x2  = self.cx + 40
                self.y1  = self.cy - 40
                self.y2  = self.cy + 40
                self.width  = 80
                self.height = 80
                self.camera = random.choice(["left", "right"])
        return [FakeDet() for _ in range(random.randint(0, 4))]

    # ── Detection loop ────────────────────────────────────────

    def _detection_loop(self):
        frame_idx = 0
        logging.info("[DETECT] Started")

        while self._running:
            t0 = time.perf_counter()

            # Grab
            pair = None
            if not self.dummy_detect and self.camera:
                pair    = self.camera.grab_pair()
                sync_ms = pair.sync_error_ms if pair else 0.0
                if self.report:
                    self.report.record_camera_grab(
                        success=pair is not None,
                        sync_error_ms=sync_ms)

            # Inference
            detections  = []
            dual_result = None

            if self.dummy_detect:
                detections = self._dummy_detections()
            elif pair is not None and self.engine:
                try:
                    dual_result = self.engine.run(pair)
                    detections  = dual_result.all_detections()
                except Exception as e:
                    logging.warning(f"[DETECT] Inference: {e}")

            # Zone update
            if self.zones and dual_result is not None:
                decision = self.zones.update(dual_result)
            else:
                decision = None

            # Pose
            pose  = (self.odom.get_pose() if self.odom
                     else {'x':0.0,'y':0.0,'heading':0.0,'speed':0.0})
            speed = pose.get('speed', self.speed)

            # Queue sprays for new zone triggers
            if decision and decision.new_triggers:
                for zone_id in decision.new_triggers:
                    zone = self.zones.zones[zone_id]

                    # Weed-only — never spray sugarbeet
                    weed_dets = [d for d in zone.current_detections
                                 if d.class_name != "sugarbeet"]
                    if not weed_dets:
                        logging.info(
                            f"[DETECT] {zone.name} triggered but "
                            f"no weeds — skipping (sugarbeet only)")
                        continue

                    best      = max(weed_dets, key=lambda d: d.confidence)
                    eff_speed = speed if speed > self.geo.min_speed_mps \
                                else self.speed
                    trig_dist = self.geo.trigger_distance_m(best.cy)
                    spray_t   = self.geo.spray_time_s(best.width, eff_speed)

                    ev = None
                    if self.report:
                        ev = self.report.record_spray_trigger(
                            nozzle=zone.nozzle_id,
                            zone_name=zone.name,
                            x=pose['x'], y=pose['y'],
                            confirming_classes=[d.class_name for d in weed_dets],
                            confirming_confidences=[d.confidence
                                                    for d in weed_dets],
                            trigger_dist_m=trig_dist,
                            spray_time_s=spray_t,
                            plant_width_px=best.width,
                        )

                    with self._lock:
                        self._pending.append(PendingSpray(
                            nozzle_id=zone.nozzle_id,
                            zone_name=zone.name,
                            x=pose['x'], y=pose['y'],
                            trigger_dist_m=trig_dist,
                            spray_time_s=spray_t,
                            confirming_classes=[d.class_name
                                                for d in weed_dets],
                            event_record=ev,
                        ))

                    logging.info(
                        f"[DETECT] Queued  {zone.name}  N{zone.nozzle_id+1}"
                        f"  trig={trig_dist:.3f}m  open={spray_t:.3f}s"
                        f"  classes={[d.class_name for d in weed_dets]}"
                    )

            # Telemetry
            if self.report:
                ms = (time.perf_counter() - t0) * 1000
                self.report.record_frame(
                    frame_index=frame_idx,
                    preprocess_ms=(dual_result.left.preprocess_ms
                                   if dual_result else 0.0),
                    inference_ms=(dual_result.total_ms
                                  if dual_result else 0.0),
                    total_ms=ms,
                    detections=[{'class_name': d.class_name,
                                 'confidence': d.confidence}
                                for d in detections],
                    robot_x=pose['x'], robot_y=pose['y'],
                    robot_speed=speed,
                    distance_traveled=(self.odom.distance_traveled
                                       if self.odom else 0.0),
                )
            frame_idx += 1

    # ── Spray actuator loop ────────────────────────────────────

    def _spray_loop(self):
        active: Dict[int, float] = {}   # nozzle_id → close_time
        logging.info("[SPRAY] Started")

        while self._running:
            now  = time.time()
            pose = (self.odom.get_pose() if self.odom
                    else {'x':0.0,'y':0.0,'heading':0.0,'speed':0.0})

            # Close expired nozzles
            for nid, close_t in list(active.items()):
                if now >= close_t:
                    self.arduino.nozzle_off(nid)
                    del active[nid]
                    logging.info(f"[SPRAY] N{nid+1} CLOSED")

            # Fire pending sprays that have reached trigger distance
            with self._lock:
                still = []
                for ps in self._pending:
                    if ps.fired:
                        continue
                    dist = math.sqrt(
                        (pose['x'] - ps.origin_x)**2 +
                        (pose['y'] - ps.origin_y)**2
                    )
                    if dist >= ps.trigger_dist_m:
                        self.arduino.nozzle_on(ps.nozzle_id)
                        ps.fired = True
                        active[ps.nozzle_id] = now + ps.spray_time_s

                        if self.report and ps.event_record:
                            self.report.record_spray_fire(
                                ps.event_record,
                                x=pose['x'], y=pose['y'],
                                distance=dist)

                        logging.info(
                            f"[SPRAY] N{ps.nozzle_id+1} FIRE"
                            f"  zone={ps.zone_name}"
                            f"  dist={dist:.3f}m"
                            f"  open={ps.spray_time_s:.3f}s"
                        )
                    else:
                        if dist < ps.trigger_dist_m + 0.5:
                            still.append(ps)
                self._pending = still

            time.sleep(0.02)   # 50 Hz

    # ── Zone monitor ──────────────────────────────────────────

    def _zone_monitor_loop(self):
        if not self.report or not self.zones:
            return
        while self._running:
            for zone in self.zones.zones:
                if zone.counter > 0:
                    self.report.record_zone_counter(
                        zone.name, zone.counter, zone.threshold)
            time.sleep(0.5)

    # ── Run ────────────────────────────────────────────────────

    def run(self):
        self._running = True

        # Odometry — try UDP first, fall back to time-based estimator
        try:
            self.odom = OdomReceiver(port=self.odom_port)
            time.sleep(0.3)
            print(f"[MISSION] Odom: UDP on port {self.odom_port}")
        except Exception as e:
            print(f"[MISSION] Odom UDP unavailable ({e}) "
                  f"— using time-based distance estimate")
            self.odom = TimebasedOdom(speed=self.speed)

        # Start hardware
        self.arduino.pump_on()
        print("[MISSION] Pump ON")
        self.husky.start_forward(self.dist, self.speed)

        # Start threads
        for name, fn in [
            ("detect",   self._detection_loop),
            ("spray",    self._spray_loop),
            ("zone-mon", self._zone_monitor_loop),
        ]:
            threading.Thread(target=fn, daemon=True, name=name).start()

        # Wait for target distance
        t_start      = time.time()
        abort_reason = "completed"
        try:
            while self._running:
                dist = (self.odom.distance_traveled if self.odom
                        else (time.time() - t_start) * self.speed)
                if dist >= self.dist:
                    print(f"\n[MISSION] Target reached: {dist:.3f}m")
                    abort_reason = "target_reached"
                    break
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[MISSION] Interrupted")
            abort_reason = "user_interrupt"
        finally:
            self._running = False
            self.husky.stop()
            self.arduino.all_off()
            print("[MISSION] All nozzles off, pump off")

        # Finalize
        dist_final = self.odom.distance_traveled if self.odom else 0.0
        if self.odom:
            self.odom.stop()
        if self.camera:
            self.camera.close()
        self.arduino.close()

        if self.report:
            self.report.finalize(
                distance_traveled=dist_final,
                target=self.dist,
                abort_reason=abort_reason,
            )
            self.report.print_console_summary()
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"session_rgb_{ts}.json"
            self.report.save_json(path)
            print(f"\nSession JSON  : {path}")
            print(f"Generate report:")
            print(f"  node generate_report_rgb.js "
                  f"--session {path} "
                  f"--validation model_validation_rgb.json "
                  f"--out ABEN_RGB_Report.docx")


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    ap = argparse.ArgumentParser(
        description="ABEN Dual RGB Autonomous Spray Mission",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model",      default="models/weed_rgb.pt",
                    help="YOLOv8n-seg RGB weights")
    ap.add_argument("--dist",       type=float, default=1.0,
                    help="Distance to travel (m)")
    ap.add_argument("--speed",      type=float, default=0.3,
                    help="Robot forward speed (m/s)")
    ap.add_argument("--port",       default="/dev/ttyACM0",
                    help="Arduino serial port")
    ap.add_argument("--odom-port",  type=int, default=5006,  # avoids GUI on 5005
                    help="UDP port for odometry (default: 5006 — "
                         "avoids conflict with GUI on 5005)")
    ap.add_argument("--dry-run",    action="store_true",
                    help="Log only — no hardware commands")
    ap.add_argument("--dummy-detect", action="store_true",
                    help="Random detections — no camera/model (timing test)")
    ap.add_argument("--field-id",   default="",
                    help="Field identifier for session report")
    ap.add_argument("--researcher", default="nana",
                    help="Researcher name for session report")
    args = ap.parse_args()

    SprayMissionRGB(
        model_path   = args.model,
        dist         = args.dist,
        speed        = args.speed,
        arduino_port = args.port,
        odom_port    = args.odom_port,
        dry_run      = args.dry_run,
        dummy_detect = args.dummy_detect,
        field_id     = args.field_id,
        researcher   = args.researcher,
    ).run()


if __name__ == "__main__":
    main()