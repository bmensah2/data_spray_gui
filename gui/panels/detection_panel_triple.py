"""
gui/panels/detection_panel_triple.py
ABEN Triple RGB Detection Panel

Integration step 2 of 3 for the triple-camera standalone app: wires
together Phase 1 (core/zone_manager_triple.py), Phase 3
(core/detection_engine_triple.py), and integration step 1
(gui/panels/triple_camera_panel.py) into a working Arm/Stop/E-Stop
detection tab that actually fires nozzles.

Built as a new, self-contained module -- zero import dependency on
gui/panels/detection_panel_rgb.py (DetectionPanelRGB) -- so the
existing 2-camera GUI is completely unaffected.

Reused AS-IS from the existing, proven 2-camera system rather than
reimplemented, since none of it is 2-camera-specific:
  - core/actuation_controller.py's ActuationController — nozzle
    firing/hold-floor/E-STOP logic already operates on plain nozzle
    IDs (0/1/2), never assumed a specific camera count.
  - core/ros_bridge_rgb.py's ROSBridge — Husky odometry, unrelated to
    camera count.
  - DetectionPanelRGB's DistanceBufferedZone class (imported directly,
    not copied) — geometry-based spray timing that operates per
    nozzle from a detection's pixel position, already camera-agnostic.
  - core/detection_config_rgb.py's RGBConfig/get_weed_config() —
    ActuationController requires cfg.zones.spray_duration() and
    cfg.session.detection_mode, so an RGBConfig-shaped object is still
    constructed here purely for that compatibility, SEPARATELY from
    Phase 1's TripleZoneConfig, which supplies the actual zone pixel
    geometry. Two config objects for two different jobs — not a
    workaround, just what each downstream piece already expects.

Deliberately scoped for this first pass -- ported the essential
camera → detection → zone → geometry-timed nozzle-fire pipeline
and E-STOP safety path, and left out for a later pass (none of which
block validating that pipeline on real hardware):
  - Session metadata dialog / provenance capture / session report
    writing (research-logging polish)
  - Manual pump prime/purge controls
  - Elaborate stats/event history tables (simple status labels used
    instead)
"""

import time
import logging

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton,
    QCheckBox,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal

from gui.style import _muted, _sec
from gui.theme_manager import theme_manager
from gui.shared_log import UnifiedLog

try:
    from core.detection_config_rgb import get_weed_config, get_cls_config
    from core.actuation_controller import ActuationController
    from core.zone_manager_triple import ZoneManagerTriple, TripleZoneConfig
    from core.detection_engine_triple import TripleDetectionEngine
except ImportError:
    from core.detection_config_rgb import get_weed_config, get_cls_config
    from core.actuation_controller import ActuationController
    from core.zone_manager_triple import ZoneManagerTriple, TripleZoneConfig
    from core.detection_engine_triple import TripleDetectionEngine

try:
    from gui.overlay_rendering_triple import draw_triple_detection_overlay
except ImportError:
    from overlay_rendering_triple import draw_triple_detection_overlay

# Reused directly from the existing 2-camera Detection tab -- see
# module docstring. Not copied: any future improvement to the
# geometry-timed debounce logic there applies here too.
try:
    from gui.panels.detection_panel_rgb import DistanceBufferedZone
except ImportError:
    from detection_panel_rgb import DistanceBufferedZone

try:
    from core.ros_bridge import ROSBridge
    from core.detection_config_rgb import NetworkConfig as _NetCfg
    ROS_BRIDGE_AVAILABLE = True
except ImportError:
    ROS_BRIDGE_AVAILABLE = False


# A minimal ZoneDecision-shaped object -- ActuationController.actuate()
# duck-types this (not a specific class), but needs MORE than just
# nozzles_to_fire/nozzles_to_stop for the physical spray itself: it
# also reads .new_triggers (nozzle IDs that newly triggered THIS
# frame) and .zones (searched by z.nozzle_id to build a SprayEvent
# record via _make_event(), which additionally needs zone.zone_id,
# zone.name, and zone.current_detections -- all satisfied by Phase
# 1's ZoneState/its .name property alias for .label). Discovered by
# testing an actual end-to-end fire, not by reading actuate() far
# enough the first time -- see this file's own test for the exact
# AttributeErrors this shape was built to satisfy.
class _ActuationDecision:
    __slots__ = ("nozzles_to_fire", "nozzles_to_stop",
                "new_triggers", "zones")

    def __init__(self, nozzles_to_fire, nozzles_to_stop,
                new_triggers, zones):
        self.nozzles_to_fire = nozzles_to_fire
        self.nozzles_to_stop = nozzles_to_stop
        self.new_triggers    = new_triggers
        self.zones           = zones


class DetectionPanelTriple(QWidget):
    """
    Arm/Stop/E-Stop detection panel for the triple-camera system.
    Wraps a TripleCameraPanel (integration step 1) and fires nozzles
    via ActuationController based on TripleDetectionEngine (Phase 3)
    + ZoneManagerTriple (Phase 1) decisions, geometry-timed through
    DistanceBufferedZone exactly like the 2-camera system.
    """

    # Emitted with the new armed state whenever _det_start()/
    # _det_stop() runs -- lets other UI (the camera panel's
    # fullscreen popout, see TripleCameraPanel.open_fullscreen_view())
    # stay in sync with the REAL armed state without polling, same
    # role DetectionTab.armed_changed plays in the 2-camera system.
    armed_changed = pyqtSignal(bool)

    def __init__(self, shared_log: UnifiedLog, camera,
                 gantry_ctrl_ref, parent=None):
        super().__init__(parent)
        self.shared_log      = shared_log
        self.camera          = camera            # TripleCameraPanel
        self.gantry_ctrl_ref = gantry_ctrl_ref    # callable -> GantryController or None

        self._armed     = False
        self._cfg        = None   # RGBConfig -- ActuationController compat
        self._zone_cfg   = TripleZoneConfig()
        self._engine      = None   # TripleDetectionEngine
        self._zones        = None   # ZoneManagerTriple
        self._actuation   = None   # ActuationController
        self._odom          = None   # ROSBridge
        self._dist_zones = [DistanceBufferedZone() for _ in range(3)]
        self._prev_spray  = [False, False, False]

        self._fps      = 0.0
        self._last_t    = 0.0
        self._events    = 0
        # Forces a simulated nonzero speed so DistanceBufferedZone can
        # trigger without the robot actually driving -- essential for
        # bench-testing camera/nozzle alignment with the robot
        # stationary (the same kind of validation the operator already
        # did by hand: a pot placed directly under each camera, nozzle
        # fired, confirmed it landed on the pot). Without this,
        # DistanceBufferedZone.update() returns immediately with
        # spray_active unchanged whenever speed < 0.05 m/s -- which is
        # exactly the situation with no Husky odometry connected.
        self._static_test = False

        # Camera watchdog -- see _check_camera_watchdog()'s docstring
        # for the real-hardware bug this exists to close: a camera
        # that stops producing frames (confirmed on real hardware --
        # a USB disconnect, errno=19 "No such device") means
        # read_triple() returns None, which means _run_inference()
        # simply never runs again until frames resume -- silently
        # leaving whatever nozzle was firing at that instant stuck in
        # its last-known state indefinitely, with no new frame ever
        # arriving to tell it to stop. Independent of and NOT the same
        # mechanism as ActuationController's own "continuously open"
        # guard, which is deliberately warn-only (a real, dense weed
        # patch legitimately SHOULD keep spraying) -- loss of camera
        # sensing is a different condition and gets a real stop, not
        # just a log line.
        self._last_frame_time  = 0.0
        self._camera_watchdog  = QTimer()
        self._camera_watchdog.timeout.connect(self._check_camera_watchdog)
        self.CAMERA_WATCHDOG_TIMEOUT_S = 3.0

        self._build_ui()

    # ── UI ─────────────────────────────────────────────────────

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        lay.addWidget(_sec("Detection — Triple Camera"))

        btn_row = QHBoxLayout()
        self.btn_arm = QPushButton("▶ ARM DETECTION")
        theme_manager.register_button(self.btn_arm, "green")
        self.btn_arm.clicked.connect(self._det_start)
        btn_row.addWidget(self.btn_arm)

        self.btn_stop = QPushButton("⏹ STOP")
        theme_manager.register_button(self.btn_stop, "dim_red")
        self.btn_stop.clicked.connect(self._det_stop)
        self.btn_stop.setEnabled(False)
        btn_row.addWidget(self.btn_stop)

        self.btn_estop = QPushButton("⚡ E-STOP")
        theme_manager.register_button(self.btn_estop, "estop")
        self.btn_estop.clicked.connect(self._det_estop)
        btn_row.addWidget(self.btn_estop)
        lay.addLayout(btn_row)

        status_grp = QGroupBox("Status")
        sg = QVBoxLayout(status_grp)
        self.lbl_armed = QLabel("Not armed")
        self.lbl_fps   = QLabel("FPS: --")
        self.lbl_inf   = QLabel("Inference: --")
        self.lbl_events = QLabel("Spray events: 0")
        for w in (self.lbl_armed, self.lbl_fps, self.lbl_inf, self.lbl_events):
            theme_manager.register_widget(
                w, lambda p: (
                    f"color:{p['text']};font-size:10px;"
                    f"font-family:'Noto Sans',Arial,sans-serif;"))
            sg.addWidget(w)
        lay.addWidget(status_grp)

        noz_grp = QGroupBox("Nozzles")
        ng = QHBoxLayout(noz_grp)
        self.lbl_nozzles = []
        for i in range(3):
            lb = QLabel(f"N{i+1}: --")
            theme_manager.register_widget(
                lb, lambda p: (
                    f"color:{p['muted']};font-size:10px;"
                    f"font-family:'Noto Sans',Arial,sans-serif;"))
            ng.addWidget(lb)
            self.lbl_nozzles.append(lb)
        lay.addWidget(noz_grp)

        self.chk_static_test = QCheckBox(
            "Static test (simulate 0.5 m/s — no odometry needed)")
        self.chk_static_test.setToolTip(
            "DistanceBufferedZone needs a nonzero robot speed to ever "
            "trigger a spray. With no Husky odometry connected (or "
            "while stationary on the bench), nothing would fire "
            "without this -- check it to validate camera/nozzle "
            "alignment without actually driving the robot.")
        self.chk_static_test.stateChanged.connect(
            lambda state: setattr(self, "_static_test", state == Qt.Checked))
        lay.addWidget(self.chk_static_test)

        lay.addStretch()

    # ── Arm / Stop / E-Stop ───────────────────────────────────

    def _det_start(self):
        if self._armed:
            return
        self._armed = True

        # RGBConfig -- required purely for ActuationController
        # compatibility (cfg.zones.spray_duration(), cfg.session.
        # detection_mode) and TripleDetectionEngine's model loading;
        # NOT the source of zone pixel geometry -- that's
        # self._zone_cfg (TripleZoneConfig, Phase 1).
        self._cfg = get_weed_config(field_id="triple_gui_run")
        self._zone_cfg = TripleZoneConfig()

        self._engine = TripleDetectionEngine(self._cfg)
        self._zones  = ZoneManagerTriple(self._zone_cfg)

        if ROS_BRIDGE_AVAILABLE:
            try:
                net_cfg = _NetCfg()
                net_cfg.husky_ip = "192.168.131.1"
                self._odom = ROSBridge(net_cfg)
                self._odom.start()
                self.shared_log.log(
                    "DETECT", "ROSBridge started (Husky odom)", "info")
            except Exception as e:
                self.shared_log.log(
                    "DETECT", "ROSBridge disabled — pose unavailable", "info")
                self._odom = None
        else:
            self._odom = None

        try:
            gantry = self.gantry_ctrl_ref() if self.gantry_ctrl_ref else None
            gantry_ctrl = (gantry if (gantry and gantry.state.connected)
                          else None)
            hardware_connected = gantry_ctrl is not None
            self._actuation = ActuationController(
                self._cfg, gantry=gantry_ctrl,
                on_spray_event=self._on_spray_event)
            self._actuation._dry_run = not hardware_connected
            self._actuation.start()
            mode_str = "HARDWARE" if hardware_connected else "DRY RUN"
            self.shared_log.log(
                "DETECT", f"ActuationController: {mode_str}",
                "ok" if hardware_connected else "warn")
        except Exception as e:
            self.shared_log.log("DETECT", f"ActuationController: {e}", "warn")
            self._actuation = None

        self.camera.on_detection_overlay = self._run_inference

        self._last_frame_time = time.time()
        self._camera_watchdog.start(500)   # check twice a second

        self.lbl_armed.setText(
            f"ARMED — {'stub mode' if self._engine.stub_mode else 'model loaded'}")
        self.btn_arm.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.shared_log.log(
            "DETECT",
            f"Armed (triple-camera) — threshold="
            f"{self._zone_cfg.detection_threshold} | "
            f"stub={self._engine.stub_mode}", "ok")
        self.armed_changed.emit(True)

    def _det_stop(self):
        if not self._armed:
            return
        self.camera.on_detection_overlay = None
        self._armed = False
        self._camera_watchdog.stop()

        if self._actuation:
            try:
                self._actuation.stop()
            except Exception:
                pass
        if self._odom:
            try:
                self._odom.stop()
            except Exception:
                pass

        if self._zones:
            self._zones.reset()
        for z in self._dist_zones:
            z.reset()
        self._prev_spray = [False, False, False]

        self._engine = None
        self._zones  = None
        self.lbl_armed.setText("Not armed")
        self.btn_arm.setEnabled(True)
        self.btn_stop.setEnabled(False)
        # Nozzle status labels are only ever updated from
        # _run_inference(), which stops running the moment
        # self._armed goes False above -- without resetting them
        # here explicitly, a label showing "FIRING" at the exact
        # instant Stop was pressed would freeze there forever,
        # looking like the system is still spraying when it isn't.
        for i, lb in enumerate(self.lbl_nozzles):
            lb.setText(f"N{i+1}: --")
        self.shared_log.log("DETECT", "Detection stopped", "info")
        self.armed_changed.emit(False)

    def _det_estop(self, reason: str = "Operator pressed E-STOP"):
        """
        Emergency stop — cuts actuation immediately. Never silently
        swallow a failure here: an E-STOP button that failed without
        telling the operator is exactly the kind of bug that matters
        most to catch, so any exception is logged loudly rather than
        passed over quietly (same hard-learned lesson as the
        2-camera system's _det_estop()).

        reason: logged verbatim so it's clear WHY this fired -- the
        operator's own button press, or (see
        _check_camera_watchdog()) an automatic safety stop triggered
        by loss of camera data, which needs a very different
        response (check the camera/USB connection, not the nozzles).
        """
        if self._actuation:
            try:
                self._actuation.emergency_stop()
                self.shared_log.log(
                    "DETECT",
                    f"ActuationController E-STOP acknowledged ({reason})",
                    "error")
            except Exception as e:
                self.shared_log.log(
                    "DETECT",
                    f"⚠⚠ ActuationController.emergency_stop() FAILED: {e} "
                    f"-- relying on direct gantry E-STOP only", "error")
                logging.error(
                    f"ActuationController.emergency_stop() failed during "
                    f"E-STOP: {e}", exc_info=True)
        else:
            self.shared_log.log(
                "DETECT",
                f"⚠ E-STOP ({reason}) but no ActuationController is "
                f"active (not armed) -- nothing to stop on this path",
                "warn")
        self.shared_log.log("DETECT", f"E-STOP — {reason}", "error")

        # Reset the nozzle labels IMMEDIATELY, not on the next
        # _run_inference() pass. Detection stays armed after E-STOP
        # (by design -- the operator may want to keep monitoring
        # while investigating), so _run_inference() keeps running and
        # would otherwise keep computing/displaying "FIRING" from the
        # raw zone/geometry decision alone, which has no idea E-STOP
        # exists -- only actuate() itself knows to refuse. Without
        # this, the display would contradict the log line right above
        # it and mislead the operator into thinking the system is
        # still spraying.
        self._prev_spray = [False, False, False]
        for i, lb in enumerate(self.lbl_nozzles):
            lb.setText(f"N{i+1}: E-STOP")

    def _check_camera_watchdog(self):
        """
        Runs independently every 500ms (self._camera_watchdog),
        NOT triggered by frame arrival -- that's the whole point.
        _run_inference() only runs when a fresh camera triple was
        actually read; if a camera disconnects (confirmed on real
        hardware: a USB drop, errno=19 "No such device"),
        read_triple() correctly returns None and _run_inference()
        simply stops being called at all. Without an independent
        check like this one, whatever nozzle was firing at that exact
        instant would stay in that last-known state indefinitely --
        no new frame ever arrives to tell it otherwise, and
        ActuationController's own "continuously open" guard is
        deliberately warn-only (correct for a real, dense weed patch
        that legitimately should keep spraying -- see its own test).
        Loss of camera SENSING is a different condition from "a real
        weed is still there", and gets a real stop, not just a log
        line.

        Reuses the already-tested E-STOP path rather than a separate
        stop mechanism -- same UI feedback (labels immediately show
        "E-STOP"), same requirement that the operator explicitly
        clears it before resuming, deliberately: a camera glitch that
        silently self-heals and silently auto-resumes spraying is a
        worse outcome than requiring the operator to notice and
        acknowledge it first.
        """
        if not self._armed:
            return
        if self._actuation and self._actuation._manual_estop_active:
            return   # already stopped (this call or the operator's own)
        stale_for = time.time() - self._last_frame_time
        if stale_for > self.CAMERA_WATCHDOG_TIMEOUT_S:
            self._det_estop(
                reason=f"no fresh camera data for {stale_for:.1f}s "
                       f"(camera watchdog — check USB connection)")

    def _on_spray_event(self, event):
        self._events += 1
        self.lbl_events.setText(f"Spray events: {self._events}")

    # ── Core inference loop ───────────────────────────────────

    def _run_inference(self, display_img):
        """
        Called by TripleCameraPanel.on_detection_overlay each frame.
        Receives the combined 3-way display BGR image, returns the
        overlay-drawn version.
        """
        if not self._armed:
            return display_img
        # This method only ever runs when TripleCameraPanel already
        # successfully read a fresh triple this cycle (see
        # _refresh_display() -- on_detection_overlay is only called
        # after a real triple.frames is obtained) -- so reaching this
        # line at all IS the signal that camera data is genuinely
        # current. See _check_camera_watchdog() for what happens when
        # this stops being called.
        self._last_frame_time = time.time()
        try:
            frames = self.camera.get_frame_snapshot()
            if any(f is None for f in frames):
                return display_img

            class _Triple:
                pass
            t = _Triple()
            t.frames     = frames
            t.frame_id   = 0
            t.timestamps = [time.time()] * 3

            result   = self._engine.run_triple(t)
            decision = self._zones.update(result.all_detections_by_camera())

            pose  = self._odom.get_pose() if self._odom else None
            speed = pose['speed'] if pose else 0.0
            if pose is None:
                pose = {'x': 0.0, 'y': 0.0, 'heading': 0.0, 'speed': 0.0}
            effective_speed = 0.5 if self._static_test else speed

            # Which nozzles have an active RAW zone decision this
            # frame -- 1:1 camera-to-nozzle now, so no more OR-ing
            # two sub-zones together like the 2-camera system needed.
            zone_hits = [False, False, False]
            for zone in self._zones.zones:
                if zone.spray_active:
                    zone_hits[zone.nozzle_id] = True

            geo = self._cfg.geometry
            nozzle_trigger = [None, None, None]
            nozzle_spray_t = [0.5, 0.5, 0.5]
            for zone in self._zones.zones:
                if not zone.current_detections:
                    continue
                nid = zone.nozzle_id
                weed_dets = [d for d in zone.current_detections
                            if d.class_name != "sugarbeet"]
                if not weed_dets:
                    continue
                det = max(weed_dets, key=lambda d: d.confidence)
                trig = geo.trigger_distance_m(det.cy)
                sprt = geo.spray_time_s(det.width, effective_speed)
                if nozzle_trigger[nid] is None or trig < nozzle_trigger[nid]:
                    nozzle_trigger[nid] = trig
                    nozzle_spray_t[nid] = sprt

            spray_states = []
            for i in range(3):
                trig = nozzle_trigger[i] if nozzle_trigger[i] is not None else 0.0
                state = self._dist_zones[i].update(
                    has_detection  = zone_hits[i],
                    pose           = pose,
                    speed          = effective_speed,
                    manual_purge   = False,
                    trigger_dist_m = trig,
                    spray_time_s   = nozzle_spray_t[i],
                )
                spray_states.append(state)

            # If E-STOP is active, the actuator already refuses to
            # fire internally (actuate() checks
            # self._manual_estop_active itself) -- but without this,
            # the DISPLAY would still show "FIRING" straight from the
            # raw zone/geometry decision, which has no notion of
            # E-STOP at all. Overriding here, before new_triggers/
            # new_releases and before actuate(), keeps everything
            # downstream (the actual fire call, the nozzle labels,
            # and _prev_spray's own bookkeeping for the next frame)
            # consistent with reality: nothing is firing.
            if self._actuation and self._actuation._manual_estop_active:
                spray_states = [False, False, False]

            prev = self._prev_spray
            new_triggers = [i for i, (c, p) in enumerate(zip(spray_states, prev)) if c and not p]
            new_releases = [i for i, (c, p) in enumerate(zip(spray_states, prev)) if not c and p]

            if self._actuation and (new_triggers or new_releases):
                act_decision = _ActuationDecision(
                    nozzles_to_fire = [i for i, s in enumerate(spray_states) if s],
                    nozzles_to_stop = [i for i, s in enumerate(spray_states) if not s],
                    new_triggers    = new_triggers,
                    zones           = self._zones.zones,
                )
                self._actuation.actuate(act_decision, pose=pose)

            self._prev_spray = list(spray_states)

            now = time.time()
            dt = max(now - self._last_t, 1e-6)
            self._last_t = now
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)

            self.lbl_fps.setText(f"FPS: {self._fps:.1f}")
            self.lbl_inf.setText(f"Inference: {result.total_ms:.1f}ms")
            estopped = bool(self._actuation and self._actuation._manual_estop_active)
            for i, lb in enumerate(self.lbl_nozzles):
                if estopped:
                    lb.setText(f"N{i+1}: E-STOP")
                else:
                    lb.setText(f"N{i+1}: {'FIRING' if spray_states[i] else 'idle'}")

            if display_img is not None:
                display_img = draw_triple_detection_overlay(
                    display_img, result, spray_states, self._zone_cfg,
                    self.camera.last_panel_width)

        except Exception as e:
            self.shared_log.log("DETECT", f"Inference error: {e}", "error")
        return display_img

    def is_armed(self) -> bool:
        return self._armed

    def emergency_stop(self):
        self._det_estop()

    def cleanup(self):
        if self._armed:
            self._det_stop()
