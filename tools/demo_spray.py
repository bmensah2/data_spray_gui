#!/usr/bin/env python3
"""
demo_spray.py
Dual RGB Detection System — Spray Demo Script

Runs on JETSON — controls both robot (via SSH) and nozzles (via Arduino).

Architecture:
  Jetson runs this script:
    - Drives Husky via SSH → test_nav_v3.py
    - Receives odometry via UDP from husky_odom_pub.py
    - Fires nozzles directly via Arduino serial (/dev/ttyACM0)
    - Runs LIVE detection (dual eMeet RGB cameras + RGBDetectionEngine +
      ZoneManagerRGB) when a model is available, or a dummy random-
      trigger mode for spray-timing tests without camera/model
    - Applies 16-inch look-ahead spray timing logic

Usage (run from the project root, not from inside tools/ --
this script imports core/, spray_mission_rgb.py, and
session_report_rgb.py, which all live at the project root):
    cd /media/pagsun/Transcend/phd_project/emeet_dual_cam
    python3 tools/demo_spray.py --dummy-detect --dist 1.0   # no camera/model needed
    python3 tools/demo_spray.py --model models/weed_rgb.pt --dist 2.0  # live detection
    python3 tools/demo_spray.py --dry-run --dist 2.0        # no hardware at all — log only
"""

import sys
import math
import time
import random
import argparse
import threading
import socket
import json
import subprocess
import logging
import numpy as np
from pathlib import Path

# This script lives in tools/, but imports core/, spray_mission_rgb.py,
# and session_report_rgb.py from the project root one level up. Running
# `python3 tools/demo_spray.py` from the project root only puts tools/
# itself on sys.path (Python adds the SCRIPT's own directory, not the
# CWD or the project root) -- so those imports fail with "No module
# named 'core'" etc. unless the project root is added explicitly here,
# regardless of the current working directory the script is invoked from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Arduino serial ────────────────────────────────────────────
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# ── Detection pipeline (dual RGB eMeet cameras) ─────────────────
# Matches spray_mission_rgb.py's real, working import pattern --
# this used to reference DetectionEngine/ZoneManager (no RGB suffix,
# pre-RGB-pivot class names that no longer exist) and never actually
# constructed or called them anywhere, with live detection hardcoded
# off ("Live multispectral detection removed — using dummy mode").
# Revived to use the real, current classes and actually wire them up
# below, rather than leaving --dummy-detect as the only working mode.
DETECTION_AVAILABLE = False
try:
    from core.detection_engine_rgb import RGBDetectionEngine
    from core.zone_manager_rgb import ZoneManagerRGB
    from core.detection_config_rgb import get_weed_config
    DETECTION_AVAILABLE = True
except ImportError as e:
    print(f"[DEMO]  ⚠ RGB detection stack not importable ({e}) — "
          f"falling back to --dummy-detect mode only")

# ── Dual RGB camera grabber ─────────────────────────────────────
# Reuses spray_mission_rgb.py's RGBCameraGrabber (dual eMeet camera
# wrapper, no Qt dependency) rather than duplicating that ~50 lines
# of camera-init logic in a second script.
CAMERA_AVAILABLE = False
try:
    from spray_mission_rgb import RGBCameraGrabber
    CAMERA_AVAILABLE = True
except ImportError as e:
    print(f"[DEMO]  ⚠ RGBCameraGrabber not importable ({e}) — "
          f"live detection needs spray_mission_rgb.py on the path")

# ── Session telemetry / reporting ─────────────────────────────
try:
    from session_report_rgb import (
        SystemConfigSnapshot, SessionReportRGB as SessionReport
    )
    REPORT_AVAILABLE = True
except ImportError as e:
    REPORT_AVAILABLE = False
    print(f"[DEMO]  ⚠ session_report_rgb.py not importable ({e}) — "
          f"running without telemetry/reporting")

# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────
ARDUINO_PORT     = "/dev/ttyACM0"
ARDUINO_BAUD     = 9600

HUSKY_IP         = "192.168.131.1"
HUSKY_USER       = "administrator"
ROS_SOURCE       = "source /opt/ros/noetic/setup.bash && "
NAV_SCRIPT       = "~/test_nav_v3.py"

ODOM_PORT        = 5005        # UDP port for odometry from Husky

DRIVE_SPEED      = 0.1         # m/s
DRIVE_DIST       = 1.0         # m (overridden by --dist)
LOOK_AHEAD_M     = 0.4064      # 16 inches
SPRAY_WINDOW_M   = 0.15        # 6 inches
DETECT_INTERVAL  = (1.0, 2.5)  # seconds between fake detections (dummy mode only)
STALL_TIMEOUT    = 5.0         # seconds without movement = abort


# ─────────────────────────────────────────────────────────────
#  ARDUINO (nozzle control — on Jetson)
# ─────────────────────────────────────────────────────────────
class Arduino:
    def __init__(self, dry_run: bool):
        self.dry_run = dry_run
        self._ser    = None
        if not dry_run and SERIAL_AVAILABLE:
            try:
                self._ser = serial.Serial(
                    ARDUINO_PORT, ARDUINO_BAUD, timeout=1)
                time.sleep(2.0)
                print(f"[ARDUINO] Connected: {ARDUINO_PORT}")
            except Exception as e:
                print(f"[ARDUINO] Failed ({e}) — dry mode")
                self.dry_run = True
        elif dry_run:
            print("[ARDUINO] Dry run — no hardware commands")

    def send(self, cmd: str):
        tag = "(dry) " if self.dry_run else ""
        print(f"[ARDUINO] {tag}► {cmd}")
        if not self.dry_run and self._ser:
            try:
                self._ser.write((cmd + "\n").encode())
                time.sleep(0.05)
            except Exception as e:
                print(f"[ARDUINO] Error: {e}")

    def nozzle_on(self, n: int):  self.send(f"n{n} on")
    def nozzle_off(self, n: int): self.send(f"n{n} off")
    def all_off(self):
        self.send("na off")
        self.send("pump off")

    def close(self):
        if self._ser:
            self._ser.close()


# ─────────────────────────────────────────────────────────────
#  ODOMETRY RECEIVER (UDP from Husky)
# ─────────────────────────────────────────────────────────────
class OdomReceiver:
    def __init__(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(('', ODOM_PORT))
        self._sock.settimeout(0.5)
        self.x     = 0.0
        self.y     = 0.0
        self.speed = 0.0
        self._home_x = None
        self._home_y = None
        self._lock   = threading.Lock()
        self._running = True
        self._thread  = threading.Thread(
            target=self._recv_loop, daemon=True)
        self._thread.start()

    def _recv_loop(self):
        while self._running:
            try:
                data, _ = self._sock.recvfrom(1024)
                msg = json.loads(data.decode())
                if msg.get('type') == 'odom':
                    with self._lock:
                        if self._home_x is None:
                            self._home_x = msg['x']
                            self._home_y = msg['y']
                        self.x     = msg['x']
                        self.y     = msg['y']
                        self.speed = msg.get('speed', 0.0)
            except socket.timeout:
                pass
            except Exception:
                pass

    def distance_traveled(self) -> float:
        with self._lock:
            if self._home_x is None:
                return 0.0
            return math.sqrt(
                (self.x - self._home_x)**2 +
                (self.y - self._home_y)**2)

    def dist_from(self, x: float, y: float) -> float:
        with self._lock:
            return math.sqrt(
                (self.x - x)**2 + (self.y - y)**2)

    def get_pose(self):
        with self._lock:
            return self.x, self.y, self.speed

    def stop(self):
        self._running = False
        self._sock.close()


# ─────────────────────────────────────────────────────────────
#  HUSKY DRIVER (SSH → test_nav_v3.py)
# ─────────────────────────────────────────────────────────────
class HuskyDriver:
    def __init__(self, dry_run: bool):
        self.dry_run  = dry_run
        self._nav_proc = None

    def _ssh(self, cmd: str, block: bool = False):
        tag = "(dry) " if self.dry_run else ""
        print(f"[HUSKY]  {tag}SSH: {cmd}")
        if self.dry_run:
            return None
        proc = subprocess.Popen([
            'ssh', '-o', 'StrictHostKeyChecking=no',
            f'{HUSKY_USER}@{HUSKY_IP}',
            f'{ROS_SOURCE}{cmd}'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if block:
            proc.wait()
        return proc

    def start_forward(self, dist: float, speed: float):
        """Start driving forward non-blocking."""
        self._nav_proc = self._ssh(
            f'python3 {NAV_SCRIPT} forward {dist} --speed {speed}',
            block=False)

    def stop(self):
        """Kill nav process + send zero velocity."""
        if self._nav_proc and self._nav_proc.poll() is None:
            self._nav_proc.terminate()
        self._ssh(
            'pkill -SIGINT -f test_nav_v3.py 2>/dev/null; sleep 0.3; '
            'python3 -c \''
            'import rospy,time;'
            'from geometry_msgs.msg import Twist;'
            'rospy.init_node("stop",anonymous=True);'
            'p=rospy.Publisher("/joy_teleop/cmd_vel",Twist,queue_size=1);'
            'time.sleep(0.3);'
            '[p.publish(Twist()) or time.sleep(0.1) for _ in range(5)]\'',
            block=False)


# ─────────────────────────────────────────────────────────────
#  SPRAY EVENT
# ─────────────────────────────────────────────────────────────
class SprayEvent:
    def __init__(self, nozzle: int, x: float, y: float):
        self.nozzle  = nozzle
        self.x       = x
        self.y       = y
        self.firing  = False
        self.done    = False
        self.record  = None   # SprayEventRecord, set by whoever creates this


# ─────────────────────────────────────────────────────────────
#  DEMO
# ─────────────────────────────────────────────────────────────
class Demo:
    def __init__(self, dist: float, dry_run: bool,
                 model_path: str = None, dummy_detect: bool = False,
                 conf: float = None, imgsz: int = 512, device: str = None):
        self.target   = dist
        self.dry_run  = dry_run
        self.arduino  = Arduino(dry_run)
        self.odom     = OdomReceiver()
        self.husky    = HuskyDriver(dry_run)
        self._pending = []
        self._events  = []
        self._lock    = threading.Lock()

        # Real detection mode needs the detection stack importable AND
        # a model to run. Falls back to dummy (random-trigger) mode if
        # either is missing, rather than crashing -- this script needs
        # to stay usable for pure spray-timing tests without a trained
        # model on hand.
        self.dummy_detect = dummy_detect or not DETECTION_AVAILABLE
        self.engine   = None
        self.zone_mgr = None
        self.rgb_cfg  = None
        self.camera   = None
        self._imgsz   = imgsz

        if DETECTION_AVAILABLE and not self.dummy_detect:
            try:
                self.rgb_cfg = get_weed_config()
                if model_path:
                    self.rgb_cfg.model.weed_rgb_pt = Path(model_path)
                if conf is not None:
                    self.rgb_cfg.model.confidence_threshold = conf
                if device is not None:
                    self.rgb_cfg.model.device = device
                self.rgb_cfg.model.imgsz = imgsz
                self.engine   = RGBDetectionEngine(self.rgb_cfg)
                self.zone_mgr = ZoneManagerRGB(self.rgb_cfg)
            except Exception as e:
                print(f"[DEMO]  Detection init failed ({e}) — "
                      f"falling back to dummy mode")
                self.dummy_detect = True

        if not dry_run and not self.dummy_detect and CAMERA_AVAILABLE:
            self.camera = RGBCameraGrabber()
            if not self.camera.connected:
                print("[DEMO]  Camera unavailable — dummy detect")
                self.dummy_detect = True
        elif not self.dummy_detect and not CAMERA_AVAILABLE:
            print("[DEMO]  RGBCameraGrabber unavailable — dummy detect")
            self.dummy_detect = True

        print(f"\n{'='*55}")
        print(f"  ABEN SPRAY DEMO  (runs on Jetson)")
        print(f"  Distance:    {dist:.1f} m")
        print(f"  Speed:       {DRIVE_SPEED} m/s")
        print(f"  Look-ahead:  {LOOK_AHEAD_M*100:.0f} cm (16 in)")
        print(f"  Spray dur:   {SPRAY_WINDOW_M*100:.0f} cm (6 in)")
        print(f"  Dry run:     {dry_run}")
        print(f"  Detection:   {'DUMMY (random)' if self.dummy_detect else f'LIVE  model={model_path}'}")
        print(f"  Husky:       {HUSKY_USER}@{HUSKY_IP}")
        print(f"  Arduino:     {ARDUINO_PORT}")
        print(f"{'='*55}\n")

        # Telemetry — built last, after dummy_detect/camera/model fallback
        # logic above has fully resolved, so the snapshot reflects what
        # actually ran, not what was requested.
        self.report = None
        if REPORT_AVAILABLE:
            cfg_snapshot = SystemConfigSnapshot(
                camera_model=("none (dummy mode)" if self.dummy_detect
                              else "eMeet C960 4K (Dual RGB)"),
                model_path=str(model_path or "none"),
                imgsz=imgsz,
                confidence_threshold=(conf if conf is not None else
                                       (self.rgb_cfg.model.confidence_threshold
                                        if self.rgb_cfg else 0.45)),
                device=(device or (self.rgb_cfg.model.device
                                    if self.rgb_cfg else "n/a")),
                detection_mode="DUMMY" if self.dummy_detect else "LIVE",
                zone_threshold=(self.zone_mgr.zones[0].threshold
                                 if self.zone_mgr else 4),
                drive_speed_mps=DRIVE_SPEED,
                look_ahead_m=LOOK_AHEAD_M,
                max_spray_dist_m=LOOK_AHEAD_M + SPRAY_WINDOW_M,
                target_distance_m=dist,
                session_start_iso=time.strftime("%Y-%m-%dT%H:%M:%S"),
                field_id=(self.rgb_cfg.session.field_id if self.rgb_cfg else ""),
                researcher=(self.rgb_cfg.session.researcher if self.rgb_cfg else ""),
            )
            self.report = SessionReport(cfg_snapshot)
        self._frame_index = 0

    def _detection_loop(self):
        if self.dummy_detect:
            self._dummy_detection_loop()
        else:
            self._live_detection_loop()

    def _dummy_detection_loop(self):
        """Original random-trigger demo mode — no camera/model needed.
        Useful in isolation for testing spray timing/look-ahead logic."""
        zone_names = ["Zone A (N1)", "Zone B (N2)", "Zone C (N3)"]
        while True:
            time.sleep(random.uniform(*DETECT_INTERVAL))
            traveled = self.odom.distance_traveled()
            if traveled >= self.target:
                break
            x, y, _ = self.odom.get_pose()
            nozzle   = random.randint(1, 3)
            ev       = SprayEvent(nozzle, x, y)
            with self._lock:
                self._pending.append(ev)
            if self.report:
                ev.record = self.report.record_spray_trigger(
                    nozzle, zone_names[nozzle-1], x, y,
                    confirming_classes=["dummy"], confirming_confidences=[1.0])
            print(f"\n[DETECT] ⚡ Weed → {zone_names[nozzle-1]}"
                  f"  pos=({x:.2f},{y:.2f})"
                  f"  travel={traveled:.2f}m"
                  f"\n         N{nozzle} fires in "
                  f"{LOOK_AHEAD_M*100:.0f}cm")

    def _live_detection_loop(self):
        """
        Real detection: grab a synchronized dual-camera frame pair, run
        YOLO inference via RGBDetectionEngine, feed the result through
        ZoneManagerRGB's multi-frame debounce filter. A SprayEvent is
        only created when ZoneManagerRGB confirms a NEW trigger -- i.e.
        the same zone has shown a detection for `threshold` consecutive
        frames, not on a single noisy frame. The existing look-ahead
        timing in _update_spray() (fire at 16in travel, close at
        16in+6in) is unchanged -- this loop only decides WHICH nozzle
        and WHEN to start that timer, same as the dummy loop did.

        Matches spray_mission_rgb.py's real detection loop (the actual
        production mission runner) -- same RGBDetectionEngine.run(pair)
        / ZoneManagerRGB.update(dual_result) API, and the same
        never-spray-sugarbeet safety filter on confirmed triggers.
        """
        while True:
            traveled = self.odom.distance_traveled()
            if traveled >= self.target:
                break

            pair = self.camera.grab_pair()
            if self.report:
                self.report.record_camera_grab(
                    success=pair is not None,
                    sync_error_ms=(pair.sync_error_ms if pair else 0.0))
            if pair is None:
                time.sleep(0.02)
                continue

            try:
                dual_result = self.engine.run(pair)
            except Exception as e:
                print(f"[DEMO]  Inference error: {e}")
                time.sleep(0.02)
                continue

            decision = self.zone_mgr.update(dual_result)

            if self.report:
                x, y, speed = self.odom.get_pose()
                self.report.record_frame(
                    frame_index=self._frame_index,
                    preprocess_ms=dual_result.left.preprocess_ms,
                    inference_ms=dual_result.total_ms,
                    total_ms=dual_result.total_ms,
                    detections=[{"class_name": d.class_name,
                                 "confidence": d.confidence}
                                for d in dual_result.all_detections()],
                    robot_x=x, robot_y=y, robot_speed=speed,
                    distance_traveled=traveled,
                )
                self._frame_index += 1
                # Track near-misses: zones building up detections that
                # never cross the debounce threshold to actually confirm.
                for zone in self.zone_mgr.zones:
                    self.report.record_zone_counter(
                        zone.name, zone.counter, zone.threshold)

            if decision.new_triggers:
                x, y, _ = self.odom.get_pose()
                for zone_id in decision.new_triggers:
                    zone = self.zone_mgr.zones[zone_id]
                    nozzle = zone.nozzle_id + 1  # ZoneState is 0-indexed, Arduino is 1-indexed

                    # Weed-only -- never spray sugarbeet (matches
                    # spray_mission_rgb.py and the GUI's
                    # ZoneManagerRGB.non_spray_classes filter)
                    weed_dets = [d for d in zone.current_detections
                                 if d.class_name != "sugarbeet"]
                    if not weed_dets:
                        print(f"[DETECT] {zone.name} triggered but no "
                              f"weeds -- skipping (sugarbeet only)")
                        continue

                    ev = SprayEvent(nozzle, x, y)
                    with self._lock:
                        self._pending.append(ev)
                    names = [d.class_name for d in weed_dets]
                    confs = [d.confidence for d in weed_dets]
                    if self.report:
                        ev.record = self.report.record_spray_trigger(
                            nozzle, zone.name, x, y,
                            confirming_classes=names,
                            confirming_confidences=confs)
                    print(f"\n[DETECT] ⚡ {zone.name} confirmed "
                          f"({len(weed_dets)} det, {dual_result.total_ms:.0f}ms, "
                          f"{names}) → N{nozzle}"
                          f"  pos=({x:.2f},{y:.2f})"
                          f"  travel={traveled:.2f}m"
                          f"\n         N{nozzle} fires in "
                          f"{LOOK_AHEAD_M*100:.0f}cm")

    def _update_spray(self):
        x, y, speed = self.odom.get_pose()
        with self._lock:
            for ev in self._pending:
                if ev.done:
                    continue
                dist = self.odom.dist_from(ev.x, ev.y)
                if not ev.firing and dist >= LOOK_AHEAD_M:
                    ev.firing = True
                    self.arduino.nozzle_on(ev.nozzle)
                    print(f"[SPRAY]  ✓ N{ev.nozzle} OPEN  "
                          f"dist={dist:.3f}m  "
                          f"travel={self.odom.distance_traveled():.2f}m")
                    if self.report and ev.record is not None:
                        self.report.record_spray_fire(ev.record, x, y, dist)
                    self._events.append({
                        'nozzle': ev.nozzle,
                        'x': round(ev.x, 3),
                        'y': round(ev.y, 3),
                        'fire_dist': round(dist, 3),
                        'time': time.strftime("%H:%M:%S"),
                    })
                elif ev.firing and dist >= LOOK_AHEAD_M + SPRAY_WINDOW_M:
                    ev.done = True
                    self.arduino.nozzle_off(ev.nozzle)
                    print(f"[SPRAY]  ✗ N{ev.nozzle} CLOSED")
                    if self.report and ev.record is not None:
                        self.report.record_spray_close(ev.record, x, y)
            self._pending = [e for e in self._pending if not e.done]

    def run(self):
        # Wait for odometry
        print("[DEMO]  Waiting for odometry from Husky...")
        deadline = time.time() + 10.0
        while self.odom._home_x is None:
            if time.time() > deadline:
                print("[DEMO]  ⚠ No odometry received")
                print("        Is husky_odom_pub.py running on Husky PC?")
                return
            time.sleep(0.1)
        x, y, _ = self.odom.get_pose()
        print(f"[DEMO]  Odom OK — start ({x:.3f}, {y:.3f})")

        # Pump on
        self.arduino.send("pump on")
        time.sleep(0.5)

        # Start detection thread
        det = threading.Thread(target=self._detection_loop, daemon=True)
        det.start()

        # Drive forward via SSH
        print(f"[DEMO]  Starting Husky: forward {self.target}m "
              f"@ {DRIVE_SPEED}m/s\n")
        self.husky.start_forward(self.target, DRIVE_SPEED)

        # Monitor + spray
        start      = time.time()
        timeout    = (self.target / DRIVE_SPEED) * 3.0
        stall_t    = time.time()
        stall_last = 0.0
        abort_reason = "unknown"

        while True:
            traveled = self.odom.distance_traveled()

            if traveled >= self.target:
                print(f"\n[DEMO]  ✓ Target reached: {traveled:.3f}m")
                abort_reason = "target_reached"
                break

            if time.time() - start > timeout:
                print(f"\n[DEMO]  ⚠ Timeout after {timeout:.0f}s")
                abort_reason = "timeout"
                break

            # Stall detection
            if traveled > stall_last + 0.005:
                stall_last = traveled
                stall_t    = time.time()
            elif time.time() - stall_t > STALL_TIMEOUT:
                print(f"\n[DEMO]  ⚠ STALL — no movement for "
                      f"{STALL_TIMEOUT:.0f}s")
                abort_reason = "stall"
                break

            self._update_spray()
            print(f"\r[DEMO]  {traveled:.2f}/{self.target:.1f}m  "
                  f"pending={len(self._pending)}  ",
                  end='', flush=True)
            time.sleep(0.1)

        # Stop everything
        print()
        self.husky.stop()
        time.sleep(0.5)
        self.arduino.all_off()
        print("[DEMO]  ■ All nozzles closed")

        # Summary
        traveled = self.odom.distance_traveled()
        duration = time.time() - start
        print(f"\n{'='*55}")
        print(f"  DEMO COMPLETE")
        print(f"  Distance:      {traveled:.3f} m")
        print(f"  Duration:      {duration:.1f} s")
        print(f"  Spray events:  {len(self._events)}")
        print(f"{'='*55}")
        for i, ev in enumerate(self._events, 1):
            print(f"  {i}. N{ev['nozzle']}  "
                  f"({ev['x']}, {ev['y']})  "
                  f"+{ev['fire_dist']:.3f}m  "
                  f"@ {ev['time']}")
        print()

        if self.report:
            self.report.finalize(traveled, self.target, abort_reason)
            self.report.print_console_summary()
            json_path = f"session_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
            self.report.save_json(json_path)

        self.odom.stop()
        self.arduino.close()
        if self.camera is not None:
            self.camera.close()


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ABEN spray demo — runs on Jetson")
    parser.add_argument(
        '--dist', type=float, default=1.0,
        help='Distance in metres (default: 1.0)')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='No hardware — log only')
    parser.add_argument(
        '--model', type=str, default=None,
        help='Path to trained YOLO weights (.pt or .engine). '
             'If omitted, uses ModelConfig default path for weed/'
             'multispectral. Required for real detection mode.')
    parser.add_argument(
        '--dummy-detect', action='store_true',
        help='Use random fake detections instead of the camera + model '
             '(original demo behavior — useful for testing spray timing '
             'in isolation, no camera/model needed)')
    parser.add_argument(
        '--conf', type=float, default=None,
        help='Detection confidence threshold override '
             '(default: ModelConfig.confidence_threshold, 0.45)')
    parser.add_argument(
        '--imgsz', type=int, default=512,
        help='Inference image size — should match training imgsz (default: 512)')
    parser.add_argument(
        '--device', type=str, default=None,
        help="Inference device override, e.g. 'cpu' for testing off-Jetson "
             "(default: ModelConfig.device, 'cuda:0')")
    args, _ = parser.parse_known_args()

    demo = Demo(dist=args.dist, dry_run=args.dry_run,
                model_path=args.model, dummy_detect=args.dummy_detect,
                conf=args.conf, imgsz=args.imgsz, device=args.device)
    try:
        demo.run()
    except KeyboardInterrupt:
        print("\n[DEMO]  Interrupted")
        demo.husky.stop()
        demo.arduino.all_off()
        demo.odom.stop()
        demo.arduino.close()
        if demo.report:
            traveled = demo.odom.distance_traveled()
            demo.report.finalize(traveled, demo.target, "interrupted")
            demo.report.print_console_summary()
            json_path = f"session_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
            demo.report.save_json(json_path)
        if demo.camera is not None:
            demo.camera.close()


if __name__ == '__main__':
    main()