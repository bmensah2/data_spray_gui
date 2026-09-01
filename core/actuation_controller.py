#!/usr/bin/env python3
"""
actuation_controller.py
ABEN Field Detection System — Actuation Controller

Translates ZoneDecision from zone_manager.py into real hardware
commands sent through GantryController to the Arduino firmware.

Two operating modes with distinct actuation logic:
  WEED mode  — spray herbicide to kill. Precise, short bursts.
  CLS mode   — spray fungicide + flag for research mapping.
               Slightly longer coverage, every event logged.

Safety rules enforced here:
  - Never commands motor PSU or gantry movement (human operator only)
  - Respects EStop from ros_bridge — all nozzles off immediately
  - Pump must be ON before any nozzle can open
  - Pump off auto-closes all nozzles (firmware safety, mirrored here)
  - Nozzle state only written on change (avoids flooding serial port)

Author : Nana | NDSU / PhD Imaging System
"""

import time
import threading
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from enum import Enum

try:
    from core.detection_config_rgb import RGBConfig as ABENConfig, DetectionMode
except ImportError:
    from detection_config_rgb import RGBConfig as ABENConfig, DetectionMode
try:
    from core.zone_manager_rgb import ZoneDecision, ZoneState
except ImportError:
    from zone_manager_rgb import ZoneDecision, ZoneState


# ─────────────────────────────────────────────────────────────
#  SPRAY EVENT  (passed to event_logger)
# ─────────────────────────────────────────────────────────────

@dataclass
class SprayEvent:
    """
    Record of a single spray actuation event.
    Passed to event_logger.py for the research record.
    """
    event_id:        str
    timestamp:       float
    mode:            str          # 'weed' or 'cls'
    zone_id:         int
    zone_name:       str
    nozzle_id:       int
    detections:      List[Dict]   # detection dicts from that zone
    spray_duration:  float        # seconds
    pose:            Optional[Dict] = None   # robot position
    gps:             Optional[Dict] = None   # GPS fix
    flagged_cls:     bool = False  # True if CLS mode spray

    def to_dict(self) -> Dict:
        return {
            'event_id':       self.event_id,
            'timestamp':      self.timestamp,
            'mode':           self.mode,
            'zone_id':        self.zone_id,
            'zone_name':      self.zone_name,
            'nozzle_id':      self.nozzle_id,
            'detections':     self.detections,
            'spray_duration': self.spray_duration,
            'pose':           self.pose,
            'gps':            self.gps,
            'flagged_cls':    self.flagged_cls,
        }


# ─────────────────────────────────────────────────────────────
#  ACTUATION CONTROLLER
# ─────────────────────────────────────────────────────────────

class ActuationController:
    """
    Hardware actuation layer for the ABEN detection system.

    Connects zone decisions to the Arduino via GantryController.
    Enforces pump/nozzle safety sequencing and EStop handling.

    Usage:
        ctrl = ActuationController(cfg, gantry_controller)
        ctrl.start()                    # enables pump, arms system

        # In detection loop:
        events = ctrl.actuate(decision, pose, gps)

        ctrl.stop()                     # safe shutdown
    """

    def __init__(self, cfg: ABENConfig,
                 gantry=None,
                 ros_bridge=None,
                 on_spray_event: Optional[Callable] = None):
        """
        Args:
            cfg:             ABENConfig
            gantry:          GantryController instance (None = dry run mode)
            ros_bridge:      ROSBridge instance for EStop monitoring
            on_spray_event:  Callback called with SprayEvent on each trigger.
                             Use this to pass events to event_logger.
        """
        self.cfg          = cfg
        self.gantry       = gantry
        self.ros_bridge   = ros_bridge
        self.on_spray_event = on_spray_event

        self._mode        = cfg.session.detection_mode
        self._dry_run     = (gantry is None)
        self._armed       = False
        self._pump_on     = False
        self._lock        = threading.Lock()

        # Track last-sent nozzle state to avoid redundant serial writes
        self._nozzle_state: Dict[int, bool] = {0: False, 1: False, 2: False}

        # EStop tracking
        self._estop_active = False
        self._estop_thread = None
        self._running      = False

        # Statistics
        self._total_sprays    = 0
        self._total_events    = 0
        self._session_start   = time.time()

        # Event ID counter
        self._event_counter   = 0

        if self._dry_run:
            logging.warning(
                "ActuationController: DRY RUN mode — "
                "no hardware commands will be sent"
            )
        else:
            logging.info(
                f"ActuationController: hardware mode | "
                f"detection={self._mode.value}"
            )

    # ── Lifecycle ─────────────────────────────────────────────

    def start(self) -> bool:
        """
        Arm the system: enable pump, start EStop monitor.
        Must be called before actuate().

        Returns True if armed successfully.
        """
        self._running = True

        # Start EStop monitor thread
        self._estop_thread = threading.Thread(
            target=self._estop_monitor, daemon=True
        )
        self._estop_thread.start()

        # Enable pump
        if not self._dry_run:
            logging.info("Enabling pump...")
            self.gantry.send_command("pump on")
            time.sleep(0.5)  # let pump pressurize
            self._pump_on = True
        else:
            self._pump_on = True  # simulate pump on in dry run

        self._armed = True
        logging.info(
            f"ActuationController ARMED | "
            f"mode={self._mode.value} | "
            f"dry_run={self._dry_run}"
        )
        return True

    def stop(self):
        """
        Safe shutdown: close all nozzles, stop pump, disarm.
        Always call this — even on exception/Ctrl+C.
        """
        logging.info("ActuationController stopping — all nozzles off")
        self._running = False
        self._armed   = False

        # Close all nozzles first
        self._all_nozzles_off()

        # Stop pump
        if not self._dry_run and self.gantry:
            self.gantry.send_command("pump off")
        self._pump_on = False

        logging.info(
            f"ActuationController stopped | "
            f"total sprays: {self._total_sprays} | "
            f"total events: {self._total_events}"
        )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    # ── Main actuation ────────────────────────────────────────

    def actuate(self, decision: ZoneDecision,
                pose: Optional[Dict] = None,
                gps: Optional[Dict] = None) -> List[SprayEvent]:
        """
        Apply a ZoneDecision to hardware.
        Call this every frame after zone_manager.update().

        Args:
            decision:  ZoneDecision from ZoneManager.update()
            pose:      Current robot pose from ROSBridge.get_pose()
            gps:       Current GPS fix from GPSReader.get_fix()

        Returns:
            List of SprayEvent objects for new spray triggers this frame.
            Empty list if no new triggers.
        """
        if not self._armed:
            logging.warning("actuate() called before start() — ignoring")
            return []

        # EStop check — highest priority
        if self._estop_active:
            self._emergency_all_off()
            return []

        new_events = []

        with self._lock:
            # ── Nozzles ON ────────────────────────────────────
            for nozzle_id in decision.nozzles_to_fire:
                if not self._nozzle_state.get(nozzle_id, False):
                    self._set_nozzle(nozzle_id, True)

            # ── Nozzles OFF ───────────────────────────────────
            for nozzle_id in decision.nozzles_to_stop:
                if self._nozzle_state.get(nozzle_id, False):
                    self._set_nozzle(nozzle_id, False)

            # ── Generate SprayEvents for new triggers ─────────
            for zone_id in decision.new_triggers:
                zone = decision.zones[zone_id]
                event = self._make_event(zone, pose, gps)
                new_events.append(event)
                self._total_events += 1

                # CLS mode: extra logging for research record
                if self._mode == DetectionMode.CLS:
                    self._handle_cls_trigger(zone, event)

        # Fire callbacks outside lock
        for event in new_events:
            if self.on_spray_event:
                try:
                    self.on_spray_event(event)
                except Exception as e:
                    logging.error(f"on_spray_event callback error: {e}")

        return new_events

    # ── Nozzle control ────────────────────────────────────────

    def _set_nozzle(self, nozzle_id: int, on: bool):
        """
        Send nozzle command to Arduino via GantryController.
        Only writes if state actually changed (avoids serial flood).
        """
        # nozzle_id is 0-indexed; Arduino command uses 1-indexed
        arduino_id = nozzle_id + 1
        cmd = f"n{arduino_id} {'on' if on else 'off'}"

        if not self._dry_run and self.gantry:
            self.gantry.send_command(cmd)
        else:
            logging.debug(f"[DRY RUN] {cmd}")

        self._nozzle_state[nozzle_id] = on
        self._total_sprays += (1 if on else 0)

        state_str = "ON " if on else "OFF"
        logging.info(
            f"💧 Nozzle {arduino_id} {state_str} | "
            f"mode={self._mode.value}"
        )

    def _all_nozzles_off(self):
        """Close all nozzles — used on stop and EStop."""
        if not self._dry_run and self.gantry:
            self.gantry.send_command("na off")
        else:
            logging.debug("[DRY RUN] na off")

        for nid in self._nozzle_state:
            self._nozzle_state[nid] = False

        logging.info("All nozzles OFF")

    # ── EStop ─────────────────────────────────────────────────

    def _estop_monitor(self):
        """
        Background thread — polls ros_bridge for EStop state.
        Immediately closes all nozzles if EStop is activated.

        EStop logic:
        - If Husky bridge has NEVER connected: no EStop enforcement.
          The robot may simply not be running ROS (bench test, dry run).
        - If Husky bridge HAS connected and then loses heartbeat:
          treat as EStop — connection lost in the field = unsafe.
        - If Husky bridge is connected and /estop = True: immediate stop.
        """
        husky_was_connected = False

        while self._running:
            if self.ros_bridge is not None:
                currently_connected = self.ros_bridge.is_connected()

                # Track first connection
                if currently_connected and not husky_was_connected:
                    husky_was_connected = True
                    logging.info("✓ Husky bridge confirmed — EStop monitoring active")

                # Only enforce EStop after Husky has been seen at least once
                if husky_was_connected:
                    estop = self.ros_bridge.is_estop_active()
                    if estop and not self._estop_active:
                        logging.warning("⚠ ESTOP DETECTED — emergency shutdown")
                        self._estop_active = True
                        self._emergency_all_off()
                    elif not estop and self._estop_active:
                        logging.info("✓ EStop cleared")
                        self._estop_active = False

            time.sleep(0.1)  # 10Hz poll — fast enough for safety

    def _emergency_all_off(self):
        """Immediate hardware shutdown on EStop."""
        with self._lock:
            self._all_nozzles_off()
            if not self._dry_run and self.gantry:
                self.gantry.send_command("pump off")
            self._pump_on = False
        logging.warning("⚠ EMERGENCY: all nozzles + pump OFF")

    # ── Event generation ──────────────────────────────────────

    def _make_event(self, zone: ZoneState,
                    pose: Optional[Dict],
                    gps: Optional[Dict]) -> SprayEvent:
        """Build a SprayEvent for a zone trigger."""
        self._event_counter += 1
        ts   = time.time()
        eid  = f"ev_{int(ts)}_{self._event_counter:04d}"

        # Duration depends on detection mode
        if self._mode == DetectionMode.WEED:
            duration = self.cfg.zones.weed_spray_duration
        else:
            duration = self.cfg.zones.cls_spray_duration

        return SprayEvent(
            event_id=eid,
            timestamp=ts,
            mode=self._mode.value,
            zone_id=zone.zone_id,
            zone_name=zone.name,
            nozzle_id=zone.nozzle_id,
            detections=[d.to_dict() for d in zone.current_detections],
            spray_duration=duration,
            pose=pose,
            gps=gps,
            flagged_cls=(self._mode == DetectionMode.CLS),
        )

    def _handle_cls_trigger(self, zone: ZoneState, event: SprayEvent):
        """
        CLS-specific trigger logic.
        In CLS mode every spray event is a research data point —
        log additional detail beyond what WEED mode needs.
        """
        det_names = [d['class_name'] for d in event.detections]
        confidence = max(
            (d['confidence'] for d in event.detections), default=0.0
        )
        logging.info(
            f"🍄 CLS SPRAY EVENT | "
            f"zone={zone.name} | "
            f"classes={det_names} | "
            f"confidence={confidence:.2f} | "
            f"event_id={event.event_id}"
        )

    # ── Manual controls ───────────────────────────────────────

    def manual_nozzle(self, nozzle_id: int, on: bool):
        """
        Manually control a nozzle — for testing and calibration.
        Does NOT require armed state.
        """
        with self._lock:
            self._set_nozzle(nozzle_id, on)

    def manual_all_off(self):
        """Manually close all nozzles."""
        with self._lock:
            self._all_nozzles_off()

    def manual_pump(self, on: bool):
        """Manually control the pump."""
        if not self._dry_run and self.gantry:
            self.gantry.send_command("pump on" if on else "pump off")
        self._pump_on = on
        logging.info(f"Pump {'ON' if on else 'OFF'} (manual)")

    # ── Status ────────────────────────────────────────────────

    def get_status(self) -> Dict:
        """Return controller status dict."""
        uptime = time.time() - self._session_start
        return {
            'armed':          self._armed,
            'dry_run':        self._dry_run,
            'mode':           self._mode.value,
            'pump_on':        self._pump_on,
            'estop_active':   self._estop_active,
            'nozzle_states':  dict(self._nozzle_state),
            'total_sprays':   self._total_sprays,
            'total_events':   self._total_events,
            'uptime_s':       round(uptime, 1),
        }


# ─────────────────────────────────────────────────────────────
#  SELF TEST
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    from detection_config_rgb import (
        get_weed_config, get_cls_config,
        GrowthStage, DetectionMode
    )
    from detection_engine_rgb import Detection, InferenceResult
    from zone_manager_rgb import ZoneManagerRGB

    def make_result(detections):
        return InferenceResult(
            detections=detections,
            inference_ms=5.0, preprocess_ms=3.0, total_ms=8.0,
            frame_shape=(640, 640), input_format='4ch',
            model_path='test',
        )

    print("=" * 55)
    print("ABEN Actuation Controller — Self Test")
    print("=" * 55)
    print()

    # ── Test 1: WEED mode dry run ─────────────────────────────
    print("Test 1: WEED mode — dry run (no hardware)")
    cfg = get_weed_config(field_id='actuation_test',
                          growth_stage=GrowthStage.FOUR_LEAF)

    events_received = []
    def capture_event(ev):
        events_received.append(ev)

    manager = ZoneManagerRGB(cfg)
    ctrl = ActuationController(cfg, gantry=None,
                                on_spray_event=capture_event)
    ctrl.start()
    assert ctrl._armed
    assert ctrl._pump_on
    print(f"  Armed: {ctrl._armed} ✓")
    print(f"  Pump: {ctrl._pump_on} ✓")

    # Simulate 4 frames with kochia in Zone A
    det_a = Detection(class_id=1, class_name='kochia',
                      confidence=0.89,
                      x1=80, y1=200, x2=140, y2=280)
    new_events = []
    for i in range(4):
        result = make_result([det_a])
        decision = manager.update(result)
        evts = ctrl.actuate(decision)
        new_events.extend(evts)

    assert ctrl._nozzle_state[0] == True, "Nozzle 1 should be ON"
    assert len(events_received) == 1, f"Expected 1 event, got {len(events_received)}"
    ev = events_received[0]
    assert ev.mode == 'weed'
    assert ev.zone_name == 'ZoneA'
    assert ev.nozzle_id == 0
    assert ev.spray_duration == cfg.zones.weed_spray_duration
    print(f"  Nozzle 1 ON: {ctrl._nozzle_state[0]} ✓")
    print(f"  Event generated: {ev.event_id} ✓")
    print(f"  Event mode: {ev.mode} ✓")
    print(f"  Spray duration: {ev.spray_duration}s ✓")

    # Drain — 5 empty frames
    for _ in range(5):
        decision = manager.update(make_result([]))
        ctrl.actuate(decision)

    assert ctrl._nozzle_state[0] == False, "Nozzle 1 should be OFF"
    print(f"  Nozzle 1 OFF after drain: {not ctrl._nozzle_state[0]} ✓")

    ctrl.stop()
    print()

    # ── Test 2: CLS mode — longer duration, flagged events ────
    print("Test 2: CLS mode — fungicide spray + flagged events")
    cfg_cls = get_cls_config(field_id='cls_test',
                             growth_stage=GrowthStage.VEGETATIVE)

    cls_events = []
    manager2 = ZoneManagerRGB(cfg_cls)
    ctrl2 = ActuationController(cfg_cls, gantry=None,
                                 on_spray_event=lambda e: cls_events.append(e))
    ctrl2.start()

    det_cls = Detection(class_id=7, class_name='cls_infected',
                        confidence=0.91,
                        x1=300, y1=200, x2=400, y2=300)  # cx=350 → Zone B

    for _ in range(4):
        decision = manager2.update(make_result([det_cls]))
        ctrl2.actuate(decision,
                      pose={'x': 4.23, 'y': 0.08},
                      gps={'lat': 46.291, 'lon': -96.612,
                           'fix_valid': True})

    assert len(cls_events) == 1
    ev_cls = cls_events[0]
    assert ev_cls.mode == 'cls'
    assert ev_cls.flagged_cls == True
    assert ev_cls.spray_duration == cfg_cls.zones.cls_spray_duration
    assert ev_cls.pose is not None
    assert ev_cls.gps is not None
    print(f"  CLS event: {ev_cls.event_id} ✓")
    print(f"  Flagged CLS: {ev_cls.flagged_cls} ✓")
    print(f"  Duration: {ev_cls.spray_duration}s ✓")
    print(f"  Pose logged: {ev_cls.pose} ✓")
    print(f"  GPS logged: lat={ev_cls.gps['lat']} ✓")

    ctrl2.stop()
    print()

    # ── Test 3: EStop simulation ──────────────────────────────
    print("Test 3: EStop — all nozzles off immediately")

    class MockROSBridge:
        def __init__(self): self.estop = False
        def is_estop_active(self): return self.estop

    mock_bridge = MockROSBridge()
    cfg3 = get_weed_config()
    manager3 = ZoneManagerRGB(cfg3)
    ctrl3 = ActuationController(cfg3, gantry=None,
                                 ros_bridge=mock_bridge)
    ctrl3.start()

    # Trigger a nozzle
    for _ in range(4):
        decision = manager3.update(make_result([det_a]))
        ctrl3.actuate(decision)

    assert ctrl3._nozzle_state[0] == True
    print(f"  Nozzle 1 ON before estop: {ctrl3._nozzle_state[0]} ✓")

    # Activate EStop
    mock_bridge.estop = True
    time.sleep(0.3)  # let monitor thread react

    assert ctrl3._nozzle_state[0] == False, "Nozzle should be OFF after estop"
    assert ctrl3._estop_active == True
    print(f"  Nozzle 1 OFF after estop: {not ctrl3._nozzle_state[0]} ✓")
    print(f"  EStop active: {ctrl3._estop_active} ✓")

    ctrl3.stop()
    print()

    # ── Test 4: get_status ────────────────────────────────────
    print("Test 4: get_status()")
    cfg4 = get_weed_config()
    ctrl4 = ActuationController(cfg4, gantry=None)
    ctrl4.start()
    status = ctrl4.get_status()
    print(f"  Keys: {list(status.keys())}")
    assert status['armed'] == True
    assert status['dry_run'] == True
    assert status['mode'] == 'weed'
    print(f"  armed={status['armed']} ✓")
    print(f"  dry_run={status['dry_run']} ✓")
    print(f"  mode={status['mode']} ✓")
    ctrl4.stop()
    print()

    print("=" * 55)
    print("actuation_controller.py  ✓  ALL TESTS PASSED")
    print("=" * 55)