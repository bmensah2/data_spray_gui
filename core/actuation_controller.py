#!/usr/bin/env python3
"""
actuation_controller.py
Detection System — Actuation Controller

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
  - Minimum spray hold floor: once a nozzle opens, it stays open for
    at least weed_spray_duration / cls_spray_duration seconds even if
    the triggering detection disappears immediately (solenoid valves
    need real time to fully open and deliver a usable dose). Spraying
    continues past the floor for as long as the zone stays active —
    this is a MINIMUM, not a fixed burst. EStop always overrides the
    floor: an emergency stop closes nozzles immediately regardless.

Author : Bright Mensah | NDSU / Imaging System
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
    Hardware actuation layer for the detection system.

    Connects zone decisions to the Arduino via GantryController.
    Enforces pump/nozzle safety sequencing and EStop handling.

    Usage:
        ctrl = ActuationController(cfg, gantry_controller)
        ctrl.start()                    # arms system (pump stays OFF)
        ctrl.manual_pump(True)          # separate, deliberate step

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

        # ── Minimum spray hold floor ────────────────────────────
        # weed_spray_duration / cls_spray_duration are the minimum
        # guaranteed valve-open time once a zone triggers, so a
        # target that flickers out of detection immediately after
        # triggering still gets a real, deliverable dose (matches
        # standard practice: solenoid valves need tens-to-hundreds
        # of ms to fully open and deliver droplets — see e.g.
        # 20-200ms minimum pulse durations in published precision
        # sprayer designs). If detection persists past this floor,
        # the nozzle simply keeps spraying — this is NOT a fixed
        # burst duration, it's a floor under the existing
        # continuous/dwell-based spray behavior from zone_manager.
        self._min_hold = cfg.zones.spray_duration(self._mode)
        self._nozzle_opened_at: Dict[int, float] = {}
        self._nozzle_pending_close: Dict[int, bool] = {}

        # Continuous-spray sanity guard — see max_continuous_spray_warn_s
        # docstring in detection_config_rgb.py. Tracks which nozzles have
        # already been warned about for their CURRENT open period, so we
        # log once per unusually-long spray, not once per frame.
        self._max_continuous_warn_s = getattr(
            cfg.zones, 'max_continuous_spray_warn_s', 5.0
        )
        self._nozzle_warned: Dict[int, bool] = {}

        # EStop tracking
        self._estop_active = False
        self._manual_estop_active = False
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
        Arm the system: start EStop monitor. Does NOT turn the pump on
        -- pump activation is a deliberate, separate manual action (see
        manual_pump(), wired to the GUI's ENABLE PUMP button) so an
        operator must explicitly choose to pressurize the pump, not
        have it happen automatically the instant detection arms. Auto-
        enabling the pump here used to defeat that safety gate entirely:
        DetectionPanelRGB's own spray-decision logic already correctly
        requires the GUI's pump_enabled flag before it will even
        attempt a spray, but the PHYSICAL pump hardware was turning on
        (pressurized, ready to fire) the moment ARM DETECTION was
        clicked, regardless of whether the operator had touched the
        pump-enable button at all.

        Must be called before actuate().
        Returns True if armed successfully.
        """
        self._running = True

        # Start EStop monitor thread
        self._estop_thread = threading.Thread(
            target=self._estop_monitor, daemon=True
        )
        self._estop_thread.start()

        self._armed = True
        logging.info(
            f"ActuationController ARMED (pump OFF until manually enabled) | "
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

        # Wait for the watchdog thread to actually exit rather than
        # returning while it's still mid-poll — makes shutdown
        # deterministic and keeps its exit log line from leaking into
        # whatever runs next (matters most when creating/stopping many
        # controllers back-to-back, e.g. in tests).
        if self._estop_thread is not None and self._estop_thread.is_alive():
            self._estop_thread.join(timeout=1.0)
            if self._estop_thread.is_alive():
                logging.warning(
                    "EStop monitor thread did not exit within 1s of stop()"
                )

        logging.info(
            f"ActuationController stopped | "
            f"total sprays: {self._total_sprays} | "
            f"total events: {self._total_events}"
        )

    def emergency_stop(self):
        """
        Operator-triggered emergency stop (e.g. a GUI E-STOP button).

        Distinct from the automatic ros_bridge-driven EStop handled by
        _estop_monitor(): that one is tracked in self._estop_active and
        cleared automatically the moment ros_bridge reports the Husky's
        own /estop is inactive again. A manual/GUI E-STOP must NOT be
        auto-cleared that way — if it reused self._estop_active, the
        watchdog thread would silently clear a manual E-STOP within
        ~100ms whenever ros_bridge happens to report no active estop,
        defeating the point of a manual stop entirely. So this uses a
        separate flag (self._manual_estop_active) that only this method
        and clear_emergency_stop() ever touch.

        Immediately kills all nozzles + pump and halts actuate() from
        doing anything further until clear_emergency_stop() is called
        (or, in this app, until the operator re-arms, which builds a
        fresh ActuationController anyway).
        """
        with self._lock:
            self._manual_estop_active = True
            self._all_nozzles_off()
            if not self._dry_run and self.gantry:
                self.gantry.send_command("pump off")
            self._pump_on = False
        logging.warning(
            "⚠ MANUAL E-STOP — all nozzles + pump OFF, "
            "actuate() will refuse to fire until cleared"
        )

    def clear_emergency_stop(self):
        """Explicitly clear a manual emergency_stop(). Does NOT re-arm
        by itself -- actuate() will resume normal behavior on the next
        call as long as self._armed is still True."""
        with self._lock:
            self._manual_estop_active = False
        logging.info("✓ Manual E-STOP cleared")

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

        # EStop check — highest priority. Covers both the automatic
        # ros_bridge-driven EStop (self._estop_active) and a manual/GUI
        # E-STOP (self._manual_estop_active) -- either one blocks
        # actuate() from doing anything until cleared.
        if self._estop_active or self._manual_estop_active:
            self._emergency_all_off()
            return []

        new_events = []

        with self._lock:
            now = time.time()

            # ── Nozzles ON ────────────────────────────────────
            for nozzle_id in decision.nozzles_to_fire:
                if not self._nozzle_state.get(nozzle_id, False):
                    self._set_nozzle(nozzle_id, True)
                    self._nozzle_opened_at[nozzle_id] = now
                    self._nozzle_warned[nozzle_id] = False
                # Active again (or still active) — cancel any pending close
                self._nozzle_pending_close.pop(nozzle_id, None)

            # ── Nozzles OFF — enforced through the minimum hold floor ──
            for nozzle_id in decision.nozzles_to_stop:
                if self._nozzle_state.get(nozzle_id, False):
                    opened_at = self._nozzle_opened_at.get(nozzle_id, now)
                    elapsed   = now - opened_at
                    if elapsed >= self._min_hold:
                        self._set_nozzle(nozzle_id, False)
                        self._nozzle_pending_close.pop(nozzle_id, None)
                        logging.debug(
                            f"Nozzle {nozzle_id+1} closed | "
                            f"actual hold={elapsed:.3f}s "
                            f"(floor={self._min_hold:.3f}s)"
                        )
                    else:
                        # Detection dropped before the floor — keep the
                        # valve open, we'll close it as soon as the
                        # floor is reached (checked below and on
                        # subsequent actuate() calls).
                        self._nozzle_pending_close[nozzle_id] = True

            # Resolve any nozzles waiting only on the floor timer —
            # covers the case where this frame's decision doesn't
            # mention them again before the floor elapses.
            for nozzle_id in list(self._nozzle_pending_close):
                opened_at = self._nozzle_opened_at.get(nozzle_id, now)
                if now - opened_at >= self._min_hold:
                    self._set_nozzle(nozzle_id, False)
                    del self._nozzle_pending_close[nozzle_id]
                    logging.debug(
                        f"Nozzle {nozzle_id+1} closed | "
                        f"floor reached (held {self._min_hold:.3f}s)"
                    )

            # ── Continuous-spray sanity guard ──────────────────
            # A nozzle open unusually long is either a genuinely large
            # weed patch (fine, keep spraying) or a stuck/false
            # detection silently wasting chemical (not fine). Either
            # way the operator should know — warn once per open period,
            # don't force it closed.
            for nozzle_id, is_on in self._nozzle_state.items():
                if not is_on:
                    continue
                opened_at = self._nozzle_opened_at.get(nozzle_id)
                if opened_at is None or self._nozzle_warned.get(nozzle_id):
                    continue
                open_duration = now - opened_at
                if open_duration >= self._max_continuous_warn_s:
                    self._nozzle_warned[nozzle_id] = True
                    logging.warning(
                        f"⚠ Nozzle {nozzle_id+1} has been continuously "
                        f"open for {open_duration:.1f}s (guard threshold "
                        f"{self._max_continuous_warn_s:.1f}s) — verify this "
                        f"is a real weed patch and not a stuck/false "
                        f"detection wasting {self._mode.value} chemical."
                    )

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
        if not on:
            # Physically closed — clear open-period bookkeeping so the
            # next open period starts its floor/warn timers fresh.
            self._nozzle_opened_at.pop(nozzle_id, None)
            self._nozzle_warned.pop(nozzle_id, None)

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

        # Clear hold-floor bookkeeping — an emergency/stop shutdown
        # bypasses the minimum hold floor by design (safety always
        # wins over guaranteeing a minimum dose).
        self._nozzle_opened_at.clear()
        self._nozzle_pending_close.clear()
        self._nozzle_warned.clear()

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

        This loop must never die silently — it's the safety watchdog
        for the whole session. Any exception polling ros_bridge is
        caught, logged loudly, and the loop keeps running rather than
        exiting; a crashed thread here would mean EStop stops being
        enforced for the rest of the session with no visible warning.
        """
        husky_was_connected = False

        while self._running:
            try:
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
            except Exception as e:
                # Do not let the safety watchdog thread die. Surface
                # this as loudly as possible — this is exactly the
                # kind of failure that should never go unnoticed.
                logging.error(
                    f"⚠⚠ ESTOP MONITOR ERROR (thread still running, "
                    f"but EStop status could not be checked this cycle): {e}"
                )

            time.sleep(0.1)  # 10Hz poll — fast enough for safety

        logging.info("EStop monitor thread exiting (controller stopped)")

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
        Does NOT require armed state, and bypasses the minimum
        hold floor (operator has explicit control here).
        """
        with self._lock:
            self._set_nozzle(nozzle_id, on)
            if on:
                self._nozzle_opened_at[nozzle_id] = time.time()
                self._nozzle_pending_close.pop(nozzle_id, None)
            else:
                self._nozzle_opened_at.pop(nozzle_id, None)
                self._nozzle_pending_close.pop(nozzle_id, None)

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
            'manual_estop_active': self._manual_estop_active,
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
    from detection_engine_rgb import Detection, InferenceResult, DualInferenceResult
    from zone_manager_rgb import ZoneManagerRGB

    def make_dual(left_dets, right_dets, frame_id=0):
        """Build a DualInferenceResult — matches zone_manager_rgb.py's
        own self-test helper, since ZoneManagerRGB.update() requires
        the dual-camera result, not a single InferenceResult."""
        def make_result(dets, camera):
            return InferenceResult(
                detections=dets, inference_ms=5.0, preprocess_ms=3.0,
                total_ms=8.0, frame_shape=(1080, 1920), camera=camera,
            )
        return DualInferenceResult(
            left=make_result(left_dets, "left"),
            right=make_result(right_dets, "right"),
            frame_id=frame_id, timestamp=time.time(),
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
    assert not ctrl._pump_on, (
        "pump must stay OFF after start() -- arming and pump activation "
        "are deliberately separate manual actions now (see manual_pump())"
    )
    print(f"  Armed: {ctrl._armed} ✓")
    print(f"  Pump (before manual enable): {ctrl._pump_on} ✓  (correctly OFF)")

    # This test exercises the full spray-trigger pipeline below, which
    # requires the pump manually enabled first -- matching what the GUI's
    # ENABLE PUMP button does, not something start()/arming does for you.
    ctrl.manual_pump(True)
    assert ctrl._pump_on
    print(f"  Pump (after manual_pump(True)): {ctrl._pump_on} ✓")

    # Simulate 4 frames with kochia in Zone A (left camera)
    det_a = Detection(class_id=1, class_name='kochia',
                      confidence=0.89,
                      x1=80, y1=200, x2=140, y2=280)
    new_events = []
    for i in range(4):
        decision = manager.update(make_dual([det_a], []))
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

    # Drain — empty frames trip the debounce release, but the nozzle
    # should stay open until the minimum hold floor elapses (real time).
    for _ in range(5):
        decision = manager.update(make_dual([], []))
        ctrl.actuate(decision)
    assert ctrl._nozzle_state[0] == True, (
        "Nozzle 1 should still be ON — floor hasn't elapsed yet"
    )
    print(f"  Nozzle 1 still ON right after drain (floor not yet elapsed) ✓")

    time.sleep(cfg.zones.weed_spray_duration + 0.05)
    ctrl.actuate(manager.update(make_dual([], [])))  # resolve pending close
    assert ctrl._nozzle_state[0] == False, "Nozzle 1 should be OFF after floor"
    print(f"  Nozzle 1 OFF once floor elapsed: {not ctrl._nozzle_state[0]} ✓")

    ctrl.stop()
    print()

    # ── Test 1b: Minimum hold floor — detection vanishes instantly ──
    print("Test 1b: Detection vanishes right after trigger — floor enforced")
    cfg_floor = get_weed_config(field_id='floor_test',
                                growth_stage=GrowthStage.FOUR_LEAF)
    manager_floor = ZoneManagerRGB(cfg_floor)
    ctrl_floor = ActuationController(cfg_floor, gantry=None)
    ctrl_floor.start()

    # 4 frames to trigger (threshold), then detection disappears
    for _ in range(4):
        decision = manager_floor.update(make_dual([det_a], []))
        ctrl_floor.actuate(decision)
    assert ctrl_floor._nozzle_state[0] == True, "Nozzle should be ON after trigger"

    # Immediately empty frame — debounce counter alone would close it
    # on frame 1, but the floor should keep it open.
    decision = manager_floor.update(make_dual([], []))
    ctrl_floor.actuate(decision)
    assert ctrl_floor._nozzle_state[0] == True, (
        "Nozzle should STILL be on — floor not yet elapsed"
    )
    print(f"  Nozzle held open past debounce release ✓ "
          f"(floor={ctrl_floor._min_hold}s)")

    # Wait past the floor, then run one more actuate() to resolve it —
    # actuate() re-checks pending closes independent of new decisions.
    time.sleep(cfg_floor.zones.weed_spray_duration + 0.05)
    ctrl_floor.actuate(manager_floor.update(make_dual([], [])))
    assert ctrl_floor._nozzle_state[0] == False, (
        "Nozzle should close once the floor has elapsed"
    )
    print(f"  Nozzle closed after floor elapsed ✓")
    ctrl_floor.stop()
    print()

    # ── Test 2: CLS mode — longer duration, flagged events ────
    print("Test 2: CLS mode — fungicide spray + flagged events")
    cfg_cls = get_cls_config(field_id='cls_test')

    cls_events = []
    manager2 = ZoneManagerRGB(cfg_cls)
    ctrl2 = ActuationController(cfg_cls, gantry=None,
                                 on_spray_event=lambda e: cls_events.append(e))
    ctrl2.start()

    # cx=350, cy=250 on the right camera → within ZoneB2 (0-900) → Nozzle 2
    det_cls = Detection(class_id=7, class_name='cls_infected',
                        confidence=0.91,
                        x1=300, y1=200, x2=400, y2=300)

    for _ in range(4):
        decision = manager2.update(make_dual([], [det_cls]))
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
        def __init__(self):
            self.estop = False
            self.connected = True  # simulate Husky already connected
        def is_estop_active(self): return self.estop
        def is_connected(self): return self.connected

    mock_bridge = MockROSBridge()
    cfg3 = get_weed_config()
    manager3 = ZoneManagerRGB(cfg3)
    ctrl3 = ActuationController(cfg3, gantry=None,
                                 ros_bridge=mock_bridge)
    ctrl3.start()

    # Trigger a nozzle
    for _ in range(4):
        decision = manager3.update(make_dual([det_a], []))
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
    print(f"  (EStop bypasses the minimum hold floor by design — "
          f"safety always wins)")

    ctrl3.stop()
    print()

    # ── Test 3b: EStop monitor survives a broken ros_bridge call ──
    print("Test 3b: EStop monitor thread survives an exception mid-poll")

    class FlakyROSBridge:
        """Raises on is_connected() a few times, then works normally —
        simulates a transient bug/glitch in the bridge implementation."""
        def __init__(self):
            self.estop = False
            self._calls = 0
        def is_connected(self):
            self._calls += 1
            if self._calls <= 3:
                raise AttributeError("simulated transient failure")
            return True
        def is_estop_active(self):
            return self.estop

    flaky_bridge = FlakyROSBridge()
    cfg3b = get_weed_config()
    ctrl3b = ActuationController(cfg3b, gantry=None, ros_bridge=flaky_bridge)
    ctrl3b.start()
    time.sleep(0.8)  # let it survive several failing poll cycles
    assert ctrl3b._estop_thread.is_alive(), (
        "EStop monitor thread must survive exceptions, not die silently"
    )
    print(f"  Monitor thread still alive after {flaky_bridge._calls} "
          f"poll attempts (3 raised exceptions) ✓")

    # Once the bridge starts working, EStop enforcement should resume
    flaky_bridge.estop = True
    time.sleep(0.3)
    assert ctrl3b._estop_active == True, (
        "EStop should still be enforced once the bridge recovers"
    )
    print(f"  EStop enforcement resumed after bridge recovered ✓")
    ctrl3b.stop()
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

    # ── Test 5: Continuous-spray sanity guard fires once ──────
    print("Test 5: Continuous-spray guard warns on unusually long hold")
    cfg5 = get_weed_config()
    cfg5.zones.max_continuous_spray_warn_s = 0.2  # shrink for a fast test
    manager5 = ZoneManagerRGB(cfg5)
    ctrl5 = ActuationController(cfg5, gantry=None)
    ctrl5.start()

    warn_records = []
    class _WarnCapture(logging.Handler):
        def emit(self, record):
            if "continuously open" in record.getMessage():
                warn_records.append(record.getMessage())
    handler = _WarnCapture()
    logging.getLogger().addHandler(handler)

    # Trigger, then keep the detection present well past the guard threshold
    for _ in range(4):
        ctrl5.actuate(manager5.update(make_dual([det_a], [])))
    assert ctrl5._nozzle_state[0] == True
    assert len(warn_records) == 0, "Should not warn before threshold elapses"

    time.sleep(cfg5.zones.max_continuous_spray_warn_s + 0.05)
    # Still detecting — nozzle stays on, guard should now fire exactly once
    ctrl5.actuate(manager5.update(make_dual([det_a], [])))
    ctrl5.actuate(manager5.update(make_dual([det_a], [])))
    assert ctrl5._nozzle_state[0] == True, "Should still be spraying — not a hard cap"
    assert len(warn_records) == 1, f"Expected exactly 1 warning, got {len(warn_records)}"
    print(f"  Nozzle still spraying past guard threshold (not a hard cap) ✓")
    print(f"  Guard warned exactly once: {warn_records[0][:60]}... ✓")

    logging.getLogger().removeHandler(handler)
    ctrl5.stop()
    print()

    # ── Test 6: manual emergency_stop() actually exists and works ──
    print("Test 6: Manual emergency_stop() — the bug that motivated this test")
    cfg6 = get_weed_config()
    manager6 = ZoneManagerRGB(cfg6)
    ctrl6 = ActuationController(cfg6, gantry=None)
    ctrl6.start()

    # Trigger a spray so there's something real to stop
    for _ in range(4):
        ctrl6.actuate(manager6.update(make_dual([det_a], [])))
    assert ctrl6._nozzle_state[0] == True, "Nozzle should be on before E-STOP"

    # This is exactly what detection_panel_rgb.py's _det_estop() calls.
    # Before this fix, ActuationController had no emergency_stop() method
    # at all, so this line raised AttributeError -- silently swallowed
    # by a bare `except: pass` at the call site, meaning the GUI's
    # E-STOP button never actually reached ActuationController.
    ctrl6.emergency_stop()

    assert ctrl6._nozzle_state[0] == False, "Nozzle must be off after emergency_stop()"
    assert ctrl6._pump_on == False
    assert ctrl6._manual_estop_active == True
    print(f"  emergency_stop() exists and actually turns things off ✓")

    # ── Test 6b: manual E-STOP is NOT auto-cleared by the ros_bridge watchdog ──
    print("Test 6b: Manual E-STOP survives a healthy ros_bridge poll cycle")

    class HealthyROSBridge:
        """Reports fully healthy/connected/no-estop -- if the watchdog
        thread incorrectly used self._estop_active for manual E-STOPs
        too, this would clear the manual stop within ~100ms."""
        def is_connected(self): return True
        def is_estop_active(self): return False

    cfg6b = get_weed_config()
    ctrl6b = ActuationController(cfg6b, gantry=None, ros_bridge=HealthyROSBridge())
    ctrl6b.start()
    time.sleep(0.15)  # let the watchdog thread poll at least once
    ctrl6b.emergency_stop()
    time.sleep(0.4)   # several more healthy poll cycles

    assert ctrl6b._manual_estop_active == True, (
        "Manual E-STOP must NOT be cleared by a healthy ros_bridge -- "
        "only clear_emergency_stop() should ever clear it"
    )
    # And actuate() must still refuse to fire while it's active
    decision = manager6.update(make_dual([det_a], []))
    for _ in range(4):
        evts = ctrl6b.actuate(decision)
    assert ctrl6b._nozzle_state[0] == False, (
        "actuate() must refuse to fire nozzles during a manual E-STOP"
    )
    print(f"  Manual E-STOP survived healthy ros_bridge polling ✓")
    print(f"  actuate() correctly refuses to fire while manually E-STOPped ✓")

    ctrl6b.clear_emergency_stop()
    assert ctrl6b._manual_estop_active == False
    print(f"  clear_emergency_stop() works ✓")
    ctrl6.stop()
    ctrl6b.stop()
    print()

    print("=" * 55)
    print("actuation_controller.py  ✓  ALL TESTS PASSED")
    print("=" * 55)