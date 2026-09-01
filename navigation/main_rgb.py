#!/usr/bin/env python3
"""
main_rgb.py
eMeet Dual RGB Detection System — Main Loop

Headless pipeline:
    DualEMEETCamera → RGBDetectionEngine → ZoneManagerRGB
        → actuation_controller (GantryController → Arduino)
        → event_logger

Run this first (without GUI) to validate that the full pipeline
produces correct spray decisions before adding the display.

Usage:
    cd /media/pagsun/Transcend/phd_project/emeet_dual_cam
    python3 main_rgb.py [--mode weed|cls] [--stub]

Flags:
    --mode weed     : Weed resistance experiment (default)
    --mode cls      : CLS disease experiment
    --stub          : Force stub mode (no Arduino, no spray)
    --no-display    : Suppress any OpenCV preview windows

Author : Nana | NDSU / PhD Imaging System
Path   : /media/pagsun/Transcend/phd_project/emeet_dual_cam/
"""

import cv2
import sys
import time
import logging
import argparse
import signal
import numpy as np
from pathlib import Path

from detection_config_rgb import (
    RGBConfig, DetectionMode, GrowthStage,
    get_weed_config, get_cls_config,
)
from dual_emeet_camera    import DualEMEETCamera
from detection_engine_rgb import RGBDetectionEngine
from zone_manager_rgb     import ZoneManagerRGB, ZONE_A, ZONE_B1, ZONE_B2, ZONE_C


# ─────────────────────────────────────────────────────────────
#  LOGGING SETUP
# ─────────────────────────────────────────────────────────────

def setup_logging(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"session_{time.strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(str(log_path)),
            logging.StreamHandler(sys.stdout),
        ]
    )
    logging.info(f"Log: {log_path}")


# ─────────────────────────────────────────────────────────────
#  OVERLAY DRAWING  (optional preview window)
# ─────────────────────────────────────────────────────────────

COLORS = {
    'zone_a':      (255, 100, 100),   # blue-ish
    'zone_b':      (100, 255, 100),   # green
    'zone_c':      (100, 100, 255),   # red-ish
    'detection':   (0,   255, 255),   # yellow
    'spray_on':    (0,   255,   0),   # bright green
    'spray_off':   (80,  80,  80),    # grey
    'text':        (255, 255, 255),
    'warning':     (0,   165, 255),
}


def draw_overlay(
    frame: np.ndarray,
    camera: str,
    zones,
    detections,
    frame_id: int,
    sync_error_ms: float,
    display_w: int = 960,
    display_h: int = 540,
) -> np.ndarray:
    """
    Draw zone boundaries, detections, and spray status on one camera frame.
    Scales down to display_w×display_h for the preview window.
    """
    orig_h, orig_w = frame.shape[:2]
    sx = display_w / orig_w
    sy = display_h / orig_h

    disp = cv2.resize(frame, (display_w, display_h))

    zone_colors = [COLORS['zone_a'], COLORS['zone_b'],
                   COLORS['zone_b'], COLORS['zone_c']]

    # Draw zone boundaries
    for zone in zones:
        if zone.camera != camera:
            continue
        x1 = int(zone.x1 * sx); y1 = int(zone.y1 * sy)
        x2 = int(zone.x2 * sx); y2 = int(zone.y2 * sy)
        col = zone_colors[zone.zone_id]
        alpha = 0.4 if zone.spray_active else 0.1
        overlay = disp.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), col, -1)
        cv2.addWeighted(overlay, alpha, disp, 1 - alpha, 0, disp)
        cv2.rectangle(disp, (x1, y1), (x2, y2), col, 2)

        # Zone label + counter
        state_str = f"{zone.name}  [{zone.counter}/{zone.threshold}]"
        if zone.spray_active:
            state_str += "  SPRAY"
        cv2.putText(disp, state_str,
                    (x1 + 8, y1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    # Draw detections
    cam_dets = [d for d in detections if d.camera == camera]
    for det in cam_dets:
        dx1 = int(det.x1 * sx); dy1 = int(det.y1 * sy)
        dx2 = int(det.x2 * sx); dy2 = int(det.y2 * sy)
        cv2.rectangle(disp, (dx1, dy1), (dx2, dy2), COLORS['detection'], 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(disp, label,
                    (dx1, dy1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['detection'], 1)

    # Header bar
    cam_label = "LEFT" if camera == "left" else "RIGHT"
    sync_col  = COLORS['spray_on'] if sync_error_ms < 50 else COLORS['warning']
    cv2.putText(disp, f"{cam_label}  |  Frame {frame_id:06d}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['text'], 2)
    cv2.putText(disp, f"Sync: {sync_error_ms:.1f}ms",
                (display_w - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, sync_col, 2)

    return disp


# ─────────────────────────────────────────────────────────────
#  STUB ACTUATOR  (used when --stub or Arduino not connected)
# ─────────────────────────────────────────────────────────────

class StubActuator:
    """Prints spray commands to console instead of firing hardware."""
    def __init__(self):
        self._nozzle_state = {0: False, 1: False, 2: False}

    def apply_decision(self, decision):
        for nid in decision.nozzles_to_fire:
            if not self._nozzle_state[nid]:
                logging.info(f"[STUB] 🔫 N{nid+1} ON")
                self._nozzle_state[nid] = True
        for nid in decision.nozzles_to_stop:
            if self._nozzle_state[nid]:
                logging.info(f"[STUB] ⬛ N{nid+1} OFF")
                self._nozzle_state[nid] = False

    def all_off(self):
        for nid in list(self._nozzle_state):
            if self._nozzle_state[nid]:
                logging.info(f"[STUB] ⬛ N{nid+1} OFF (shutdown)")
            self._nozzle_state[nid] = False


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ABEN eMeet Dual RGB Detection System"
    )
    parser.add_argument(
        "--mode", choices=["weed", "cls"], default="weed",
        help="Detection mode (default: weed)"
    )
    parser.add_argument(
        "--stub", action="store_true",
        help="Stub mode — no Arduino, no spray hardware"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Suppress OpenCV preview windows"
    )
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────
    if args.mode == "weed":
        cfg = get_weed_config()
    else:
        cfg = get_cls_config()

    project_root = Path("/media/pagsun/Transcend/phd_project/emeet_dual_cam")
    setup_logging(project_root / "logs")

    logging.info("=" * 60)
    logging.info("ABEN eMeet Dual RGB Detection System")
    logging.info(f"  Mode     : {cfg.session.detection_mode.value}")
    logging.info(f"  Stub     : {args.stub}")
    logging.info(f"  Display  : {not args.no_display}")
    logging.info(f"  B1 split : {cfg.zones.B1_SPLIT_X}px (PLACEHOLDER)")
    logging.info(f"  B2 split : {cfg.zones.B2_SPLIT_X}px (PLACEHOLDER)")
    logging.info("=" * 60)

    # ── Modules ───────────────────────────────────────────────
    camera  = DualEMEETCamera()
    engine  = RGBDetectionEngine(cfg)
    zones   = ZoneManagerRGB(cfg)

    if args.stub or not engine.ready:
        actuator = StubActuator()
        logging.info("Actuator: STUB mode")
    else:
        # Real actuator wired to GantryController
        # Import here so stub mode works without Arduino connected
        try:
            from actuation_controller import ActuationController
            from gantry_controller    import GantryController
            gantry   = GantryController()
            actuator = ActuationController(cfg, gantry)
            logging.info("Actuator: HARDWARE mode")
        except Exception as e:
            logging.warning(f"Actuator: hardware init failed ({e}) → STUB")
            actuator = StubActuator()

    # ── Graceful shutdown ─────────────────────────────────────
    running = [True]

    def _shutdown(sig, frame):
        logging.info("Shutdown requested …")
        running[0] = False

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Start camera ──────────────────────────────────────────
    camera.start()
    logging.info("Cameras started — beginning detection loop")
    logging.info("Press q in preview window or Ctrl-C to stop")

    # ── Stats ──────────────────────────────────────────────────
    frame_times = []
    spray_count  = 0

    try:
        while running[0]:
            t_loop = time.perf_counter()

            # 1. Read frame pair
            pair = camera.read_pair()
            if pair is None:
                time.sleep(0.005)
                continue

            if not pair.sync_ok:
                logging.warning(
                    f"Sync error: {pair.sync_error_ms:.1f}ms "
                    f"(frame {pair.frame_id})"
                )

            # 2. Run inference on both cameras
            dual_result = engine.run(pair)

            # 3. Update zone states
            decision = zones.update(dual_result)

            # 4. Actuate nozzles
            actuator.apply_decision(decision)

            if decision.new_triggers:
                spray_count += len(decision.new_triggers)

            # 5. Preview window (optional)
            if not args.no_display:
                all_dets = dual_result.all_detections()

                left_disp = draw_overlay(
                    pair.left, "left",
                    zones.zones, all_dets,
                    pair.frame_id, pair.sync_error_ms,
                )
                right_disp = draw_overlay(
                    pair.right, "right",
                    zones.zones, all_dets,
                    pair.frame_id, pair.sync_error_ms,
                )

                combined = np.hstack((left_disp, right_disp))

                # FPS overlay
                if len(frame_times) > 10:
                    fps = 1.0 / (sum(frame_times[-10:]) / 10)
                    cv2.putText(combined, f"FPS: {fps:.1f}",
                                (870, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                COLORS['text'], 2)

                cv2.imshow("ABEN Dual RGB", combined)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    running[0] = False

            # Loop timing
            t_end = time.perf_counter()
            frame_times.append(t_end - t_loop)
            if len(frame_times) > 300:
                frame_times.pop(0)

    finally:
        logging.info("Shutting down …")
        actuator.all_off()
        camera.stop()
        cv2.destroyAllWindows()

        # Session summary
        if frame_times:
            avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
            logging.info(f"Session summary:")
            logging.info(f"  Frames processed : {len(frame_times)}")
            logging.info(f"  Average FPS      : {avg_fps:.1f}")
            logging.info(f"  Spray triggers   : {spray_count}")
        logging.info("Done.")


if __name__ == "__main__":
    main()
