#!/usr/bin/env python3
"""
demo_spray.py
ABEN Field Imaging System — Spray Demo Script

Runs on JETSON — controls both robot (via SSH) and nozzles (via Arduino).

Architecture:
  Jetson runs this script:
    - Drives Husky via SSH → test_nav_v3.py
    - Receives odometry via UDP from husky_odom_pub.py
    - Fires nozzles directly via Arduino serial (/dev/ttyACM0)
    - Applies 12-inch look-ahead logic
    - Grabs frames from msCAM, runs YOLO inference via DetectionEngine,
      and feeds results through ZoneManager's multi-frame debounce filter
      to decide which nozzle to fire (real detection, not random)

Usage:
    cd ~/phd_project/multispec_camera
    source ~/.venvs/hslab/bin/activate
    python demo_spray.py --model models/weed_multispectral.pt   # real detection
    python demo_spray.py --dist 2.0 --model models/weed_multispectral.pt
    python demo_spray.py --dummy-detect    # old random-trigger mode (no camera/model needed)
    python demo_spray.py --dry-run         # no hardware at all — log only
"""

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

# ── Arduino serial ────────────────────────────────────────────
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# ── Detection pipeline ────────────────────────────────────────
# Try both import layouts: detection_engine.py lives in a `core/`
# subpackage in some checkouts (per its own self-test's
# `from core.detection_config import ...`), flat alongside this script
# in others. Fall back to dummy-detect mode rather than crashing if
# neither resolves — this script should still be runnable for spray
# timing tests without the full detection stack installed.
DETECTION_AVAILABLE = False
try:
    from core.detection_engine import DetectionEngine
    from core.zone_manager import ZoneManager
    from core.detection_config import get_weed_config, GrowthStage, CameraMode
    DETECTION_AVAILABLE = True
except ImportError:
    try:
        from detection_engine import DetectionEngine
        from zone_manager import ZoneManager
        from detection_config import get_weed_config, GrowthStage, CameraMode
        DETECTION_AVAILABLE = True
    except ImportError as e:
        print(f"[DEMO]  ⚠ Detection stack not importable ({e}) — "
              f"falling back to --dummy-detect mode only")

# ── Camera (harvesters / GenICam) ─────────────────────────────
try:
    from harvesters.core import Harvester
    HARVESTER_AVAILABLE = True
except ImportError:
    HARVESTER_AVAILABLE = False

CTI_PATH = "/opt/sentech/lib/libstgentl.cti"

# Matches camera_panel.py's authoritative _BAYER extraction exactly —
# stride 4 on a native 2048x2048 sensor frame, giving 512x512 per band.
BAYER_OFFSETS = {
    "580nm": (0, 0),
    "660nm": (0, 2),
    "735nm": (2, 0),
    "820nm": (2, 2),
}

# ── Session telemetry / reporting ─────────────────────────────
try:
    from session_report import SystemConfigSnapshot, SessionReport
    REPORT_AVAILABLE = True
except ImportError as e:
    REPORT_AVAILABLE = False
    print(f"[DEMO]  ⚠ session_report.py not importable ({e}) — "
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
#  MSCAM GRABBER (headless — no Qt/GUI dependency)
# ─────────────────────────────────────────────────────────────
class MsCamGrabber:
    """
    Minimal standalone GenICam frame grabber for the msCAM, used only by
    real-detection mode. Mirrors camera_panel.py's connection and band
    extraction exactly (same CTI path, same stride-4 BAYER_OFFSETS) so
    inference here sees the same pixel data the model was trained on.

    No display, no Qt — just connect, grab, extract bands, repeat.
    """
    def __init__(self):
        self.available = False
        self._h  = None
        self._ia = None
        if not HARVESTER_AVAILABLE:
            print("[CAMERA] harvesters not installed — no live camera")
            return
        try:
            self._h = Harvester()
            self._h.add_file(CTI_PATH)
            self._h.update()
            if not self._h.device_info_list:
                print("[CAMERA] No camera detected on CTI bus")
                return
            self._ia = self._h.create()
            model = self._ia.remote_device.node_map.DeviceModelName.value
            self._ia.start()
            self.available = True
            print(f"[CAMERA] Connected: {model}")
        except Exception as e:
            print(f"[CAMERA] Connection failed: {e}")
            self.available = False

    def grab_bands(self):
        """
        Fetch one frame and extract the 4 spectral bands.
        Returns {'580nm': arr, '660nm': arr, '735nm': arr, '820nm': arr}
        or None on timeout/error (caller should just retry — this is
        expected occasionally, not a hard failure).
        """
        if not self.available or self._ia is None:
            return None
        try:
            with self._ia.fetch(timeout=0.2) as buf:
                comp = buf.payload.components[0]
                raw = comp.data.reshape(comp.height, comp.width).copy()
            return {
                name: raw[r::4, c::4].copy()
                for name, (r, c) in BAYER_OFFSETS.items()
            }
        except Exception:
            return None

    def close(self):
        try:
            if self._ia is not None:
                self._ia.stop()
                self._ia.destroy()
            if self._h is not None:
                self._h.reset()
        except Exception:
            pass


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
        # either a model path or --dummy-detect explicitly NOT requested.
        # Falls back to the old random-trigger demo if either is missing,
        # rather than crashing — this script needs to stay usable for
        # pure spray-timing tests without a trained model on hand.
        self.dummy_detect = dummy_detect or not DETECTION_AVAILABLE
        self.engine   = None
        self.zone_mgr = None
        self.camera   = None
        self._imgsz   = imgsz

        if not self.dummy_detect:
            cfg = get_weed_config(
                field_id="spray_demo",
                growth_stage=GrowthStage.FOUR_LEAF,
                camera_mode=CameraMode.MULTISPECTRAL,
            )
            if conf is not None:
                cfg.model.confidence_threshold = conf
            if device is not None:
                cfg.model.device = device

            self.engine = DetectionEngine(cfg)
            loaded = self.engine.load(model_path)
            if not loaded:
                print(f"[DEMO]  ⚠ Model failed to load from "
                      f"{model_path or cfg.model.get_model_path(cfg.session.detection_mode, cfg.session.camera_mode)} "
                      f"— falling back to --dummy-detect mode")
                self.dummy_detect = True
            else:
                self.engine._imgsz = imgsz
                self.zone_mgr = ZoneManager(cfg)
                self.camera = MsCamGrabber()
                if not self.camera.available:
                    print("[DEMO]  ⚠ No camera available — "
                          "falling back to --dummy-detect mode")
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
                camera_model=(self.camera.available and "msCAM (connected)"
                              or "none (dummy mode)") if self.camera else "none (dummy mode)",
                model_path=str(model_path or "none"),
                imgsz=imgsz,
                confidence_threshold=(conf if conf is not None else
                                       (self.engine.cfg.model.confidence_threshold
                                        if self.engine else 0.45)),
                device=(device or (self.engine.cfg.model.device if self.engine else "n/a")),
                detection_mode="DUMMY" if self.dummy_detect else "LIVE",
                zone_threshold=(self.zone_mgr.zones[0].threshold
                                 if self.zone_mgr else 4),
                drive_speed_mps=DRIVE_SPEED,
                look_ahead_m=LOOK_AHEAD_M,
                spray_window_m=SPRAY_WINDOW_M,
                target_distance_m=dist,
                session_start_iso=time.strftime("%Y-%m-%dT%H:%M:%S"),
                field_id=(self.engine.cfg.session.field_id if self.engine else ""),
                researcher=(self.engine.cfg.session.researcher if self.engine else ""),
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
        Real detection: grab a frame, run YOLO inference, feed the result
        through ZoneManager's multi-frame debounce filter. A SprayEvent is
        only created when ZoneManager confirms a NEW trigger — i.e. the
        same zone has shown a detection for `threshold` consecutive frames,
        not on a single noisy frame. The existing look-ahead timing in
        _update_spray() (fire at 16in travel, close at 16in+6in) is
        unchanged — this loop only decides WHICH nozzle and WHEN to start
        that timer, same as the dummy loop did.
        """
        while True:
            traveled = self.odom.distance_traveled()
            if traveled >= self.target:
                break

            bands = self.camera.grab_bands()
            if self.report:
                self.report.record_camera_grab(success=bands is not None)
            if bands is None:
                time.sleep(0.02)
                continue

            result = self.engine.infer(bands)
            decision = self.zone_mgr.update(result)

            if self.report:
                x, y, speed = self.odom.get_pose()
                self.report.record_frame(
                    frame_index=self._frame_index,
                    preprocess_ms=result.preprocess_ms,
                    inference_ms=result.inference_ms,
                    total_ms=result.total_ms,
                    detections=[{"class_name": d.class_name,
                                 "confidence": d.confidence}
                                for d in result.detections],
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
                    ev = SprayEvent(nozzle, x, y)
                    with self._lock:
                        self._pending.append(ev)
                    names = [d.class_name for d in zone.current_detections]
                    confs = [d.confidence for d in zone.current_detections]
                    if self.report:
                        ev.record = self.report.record_spray_trigger(
                            nozzle, zone.name, x, y,
                            confirming_classes=names,
                            confirming_confidences=confs)
                    print(f"\n[DETECT] ⚡ {zone.name} confirmed "
                          f"({result.count} det, {result.inference_ms:.0f}ms, "
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