#!/usr/bin/env python3
"""
capture_tool.py
Field Detection System — Field Data Capture Tool

Option C capture strategy:
  - Automatic interval capture (every N seconds while running)
  - Manual trigger (press 'm') for targeted captures
  - Flag key (press 'f') marks frame as priority for annotation in Roboflow/CVAT

Output per captured frame:
  - <session>/<frame_id>_raw.tif      : Full 4-band raw image (for YOLO training)
  - <session>/<frame_id>_preview.jpg  : False-color RGB composite (for Roboflow)
  - <session>/<frame_id>_meta.json    : Full metadata (timestamp, pose, GPS, etc.)

Usage:
    python capture_tool.py --mode weed --field Field_01 --stage 4_leaf
    python capture_tool.py --mode cls  --field Plot_B   --stage vegetative

Author : Nana | NDSU / PhD Imaging System
"""

import cv2
import json
import time
import threading
import argparse
import logging
import numpy as np
import tifffile
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple

from core.detection_config_rgb import (
    ABENConfig, DetectionMode, CameraMode, GrowthStage,
    get_weed_config, get_cls_config
)

# Optional imports — graceful degradation if hardware not present
try:
    import pynmea2
    import serial as pyserial
    GPS_AVAILABLE = True
except ImportError:
    GPS_AVAILABLE = False
    print("⚠  pynmea2/serial not available — GPS disabled")

try:
    import socket
    SOCKET_AVAILABLE = True
except ImportError:
    SOCKET_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
#  GPS READER  (runs in background thread)
# ─────────────────────────────────────────────────────────────

class GPSReader:
    """
    Reads NMEA sentences from USB GPS dongle in a background thread.
    Provides latest fix to the capture tool without blocking the main loop.
    """

    def __init__(self, port: str, baudrate: int = 4800, timeout: float = 1.0):
        self.port       = port
        self.baudrate   = baudrate
        self.timeout    = timeout
        self.lat        = None
        self.lon        = None
        self.altitude   = None
        self.satellites = 0
        self.fix_valid  = False
        self.last_fix   = None
        self._running   = False
        self._thread    = None
        self._lock      = threading.Lock()
        self._serial    = None

    def start(self) -> bool:
        if not GPS_AVAILABLE:
            logging.warning("GPS libraries not available")
            return False
        try:
            self._serial = pyserial.Serial(
                self.port, self.baudrate, timeout=self.timeout
            )
            self._running = True
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
            logging.info(f"GPS reader started on {self.port}")
            return True
        except Exception as e:
            logging.warning(f"GPS failed to start: {e}")
            return False

    def stop(self):
        self._running = False
        if self._serial:
            self._serial.close()

    def _read_loop(self):
        while self._running:
            try:
                line = self._serial.readline().decode('ascii', errors='replace').strip()
                if line.startswith('$GNGGA') or line.startswith('$GPGGA'):
                    msg = pynmea2.parse(line)
                    with self._lock:
                        if msg.gps_qual > 0:  # 0 = no fix
                            self.lat        = float(msg.latitude)
                            self.lon        = float(msg.longitude)
                            self.altitude   = float(msg.altitude) if msg.altitude else None
                            self.satellites = int(msg.num_sats)
                            self.fix_valid  = True
                            self.last_fix   = time.time()
                        else:
                            self.fix_valid = False
            except Exception:
                pass  # Skip malformed sentences

    def get_fix(self) -> Dict:
        """Return latest GPS fix as a dict. fix_valid=False if no fix."""
        with self._lock:
            stale = (self.last_fix is None or
                     (time.time() - self.last_fix) > 5.0)
            return {
                'lat':        self.lat,
                'lon':        self.lon,
                'altitude':   self.altitude,
                'satellites': self.satellites,
                'fix_valid':  self.fix_valid and not stale,
            }


# ─────────────────────────────────────────────────────────────
#  ODOMETRY BRIDGE  (receives UDP from Husky ROS node)
# ─────────────────────────────────────────────────────────────

class OdomBridge:
    """
    Receives odometry from Husky onboard PC via UDP.
    The Husky PC runs a small ROS node (husky_odom_publisher.py)
    that serializes /odom to JSON and sends over UDP.
    """

    def __init__(self, port: int = 5005, timeout: float = 2.0):
        self.port       = port
        self.timeout    = timeout
        self.latest     = None
        self._running   = False
        self._thread    = None
        self._lock      = threading.Lock()
        self._sock      = None

    def start(self) -> bool:
        if not SOCKET_AVAILABLE:
            return False
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind(('', self.port))
            self._sock.settimeout(0.5)
            self._running = True
            self._thread = threading.Thread(target=self._recv_loop, daemon=True)
            self._thread.start()
            logging.info(f"Odometry bridge listening on UDP:{self.port}")
            return True
        except Exception as e:
            logging.warning(f"Odometry bridge failed: {e}. Pose logging disabled.")
            return False

    def stop(self):
        self._running = False
        if self._sock:
            self._sock.close()

    def _recv_loop(self):
        while self._running:
            try:
                data, _ = self._sock.recvfrom(1024)
                pose = json.loads(data.decode())
                pose['received_at'] = time.time()
                with self._lock:
                    self.latest = pose
            except socket.timeout:
                pass
            except Exception as e:
                logging.debug(f"Odom recv error: {e}")

    def get_pose(self) -> Optional[Dict]:
        """Return latest pose or None if stale/unavailable."""
        with self._lock:
            if self.latest is None:
                return None
            age = time.time() - self.latest.get('received_at', 0)
            if age > self.timeout:
                return None
            return {
                'x':        self.latest.get('x', 0.0),
                'y':        self.latest.get('y', 0.0),
                'heading':  self.latest.get('heading', 0.0),
                'speed':    self.latest.get('speed', 0.0),
                'age_ms':   round(age * 1000, 1),
            }


# ─────────────────────────────────────────────────────────────
#  BAND EXTRACTOR
# ─────────────────────────────────────────────────────────────

class BandExtractor:
    """
    Extracts individual spectral bands from a simulated raw Bayer-pattern frame.
    Matches the band extraction logic in existing multi_spec_v2.py.
    """

    def __init__(self, cfg: ABENConfig):
        self.offsets = cfg.ms_camera.band_offsets
        self.channel_order = cfg.ms_camera.channel_order

    def extract(self, raw: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract all bands from raw frame. Returns dict of band arrays."""
        bands = {}
        for name, (row, col) in self.offsets.items():
            bands[name] = raw[row::2, col::2]
        return bands

    def to_4ch_tensor(self, bands: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Stack bands into (H, W, 4) array in canonical channel order.
        This is the input format for 4-channel YOLO training/inference.
        Crops all bands to the minimum common shape — off-by-one differences
        arise from Bayer sub-sampling on non-even frame dimensions.
        """
        channels = [bands[name] for name in self.channel_order]
        min_h = min(c.shape[0] for c in channels)
        min_w = min(c.shape[1] for c in channels)
        channels = [c[:min_h, :min_w] for c in channels]
        return np.stack(channels, axis=-1)

    def to_preview_rgb(self, bands: Dict[str, np.ndarray],
                       r_band: str = '735nm',
                       g_band: str = '660nm',
                       b_band: str = '580nm') -> np.ndarray:
        """
        Build false-colour Red-Edge composite for annotation tools.
        Default: R=735nm (red-edge)  G=660nm (red)  B=580nm (green)
        Emphasises chlorophyll at emergence — primary annotation composite.
        Returns uint8 BGR array suitable for cv2.imwrite().
        """
        def norm(arr):
            a = arr.astype(np.float32)
            mn, mx = a.min(), a.max()
            if mx - mn < 1e-6:
                return np.zeros_like(arr, dtype=np.uint8)
            return ((a - mn) / (mx - mn) * 255).astype(np.uint8)

        r = norm(bands[r_band])
        g = norm(bands[g_band])
        b = norm(bands[b_band])

        # Crop to common shape (Bayer sub-sampling may produce off-by-one dims)
        min_h = min(r.shape[0], g.shape[0], b.shape[0])
        min_w = min(r.shape[1], g.shape[1], b.shape[1])
        r, g, b = r[:min_h, :min_w], g[:min_h, :min_w], b[:min_h, :min_w]

        # OpenCV uses BGR order
        return cv2.merge([b, g, r])


# ─────────────────────────────────────────────────────────────
#  CAPTURE TOOL  (main class)
# ─────────────────────────────────────────────────────────────

class CaptureTool:
    """
    Field data capture tool for ABEN detection system.

    Captures multispectral frames with full metadata for:
    - YOLO training dataset (4-band .tif files)
    - Annotation in Roboflow/CVAT (preview .jpg files)
    - Research record (JSON metadata per frame)

    Controls:
        'm'  : Manual capture (instant, outside auto-interval)
        'f'  : Flag current/next frame as annotation priority
        'p'  : Pause / resume auto-capture
        'q'  : Quit and save session summary
        's'  : Print current session stats
    """

    def __init__(self, cfg: ABENConfig):
        self.cfg            = cfg
        self.session_id     = self._make_session_id()
        self.frame_count    = 0
        self.flagged_count  = 0
        self.failed_count   = 0
        self.paused         = False
        self._flag_next     = False
        self._running       = False

        # Setup storage
        cfg.storage.create_session_dirs(self.session_id)
        self.session_dir    = cfg.storage.session_path(self.session_id)
        self.meta_dir       = cfg.storage.metadata_path(self.session_id)

        # Hardware
        self.extractor      = BandExtractor(cfg)
        self.gps            = GPSReader(cfg.gps.port, cfg.gps.baudrate)
        self.odom           = OdomBridge(cfg.network.odom_port)

        # Camera state
        self.sim_mode       = True
        self._last_auto_time = 0.0

        # Setup logging
        log_path = cfg.storage.base_path / cfg.storage.logs_dir / \
                   f"capture_{self.session_id}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )

    # ── Session ID ────────────────────────────────────────────

    def _make_session_id(self) -> str:
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode  = self.cfg.session.detection_mode.value
        cam   = self.cfg.session.camera_mode.value[:2]   # 'mu' or 'rg'
        stage = self.cfg.session.growth_stage.value
        field = self.cfg.session.field_id.replace(' ', '_') or "unknown"
        return f"{ts}_{mode}_{cam}_{stage}_{field}"

    # ── Camera Init ───────────────────────────────────────────

    def _init_camera(self) -> bool:
        logging.warning("Running in simulation mode — hardware capture removed")
        return True

    def _get_frame(self) -> Optional[np.ndarray]:
        """Generate one simulated raw frame."""
        if self.sim_mode:
            # Simulate a realistic 4-pattern raw frame for testing
            h = self.cfg.ms_camera.height
            w = self.cfg.ms_camera.width
            frame = np.random.randint(20, 200, (h, w), dtype=np.uint16)
            # Add some spatial structure to make it look real
            y_idx, x_idx = np.mgrid[0:h, 0:w]
            frame = frame + (30 * np.sin(y_idx / 20)).astype(np.int16)
            return np.clip(frame, 0, 4095).astype(np.uint16)

        return None

    # ── Quality Check ─────────────────────────────────────────

    def _check_quality(self, bands: Dict[str, np.ndarray]) -> Tuple[bool, str]:
        """Basic quality check on extracted bands."""
        if not self.cfg.capture.enable_quality_filter:
            return True, "ok"

        # Use 660nm (red) band as reference — most informative for vegetation
        ref = bands.get('660nm', list(bands.values())[0]).astype(np.float32)

        brightness = ref.mean()
        if brightness < self.cfg.capture.min_brightness:
            return False, f"too_dark ({brightness:.1f})"
        if brightness > self.cfg.capture.max_brightness:
            return False, f"too_bright ({brightness:.1f})"

        # Laplacian sharpness estimate
        ref_8bit = (ref / ref.max() * 255).astype(np.uint8)
        sharpness = cv2.Laplacian(ref_8bit, cv2.CV_64F).var()
        if sharpness < self.cfg.capture.min_sharpness:
            return False, f"blurry ({sharpness:.1f})"

        return True, "ok"

    # ── Frame Save ────────────────────────────────────────────

    def _save_frame(self, raw: np.ndarray, flagged: bool = False,
                    trigger: str = 'auto') -> Optional[str]:
        """
        Save a captured frame with all associated files and metadata.
        Returns frame_id on success, None on failure.
        """
        try:
            ts         = datetime.now()
            frame_id   = f"{ts.strftime('%Y%m%d_%H%M%S_%f')[:-3]}_{self.frame_count:05d}"
            flag_str   = "_FLAGGED" if flagged else ""
            base_name  = f"{frame_id}{flag_str}"

            # Extract bands
            bands = self.extractor.extract(raw)

            # Quality check
            ok, reason = self._check_quality(bands)
            if not ok:
                logging.warning(f"Frame rejected: {reason}")
                self.failed_count += 1
                return None

            # Build 4-channel tensor
            tensor_4ch = self.extractor.to_4ch_tensor(bands)

            # Build preview RGB
            preview = self.extractor.to_preview_rgb(
                bands,
                r_band=self.cfg.capture.preview_r_band,
                g_band=self.cfg.capture.preview_g_band,
                b_band=self.cfg.capture.preview_b_band,
            )

            # Save raw 4-band TIF (uint16, all 4 channels)
            if self.cfg.capture.save_raw_tif:
                tif_path = self.session_dir / f"{base_name}_raw.tif"
                # Save as (4, H, W) — standard multi-band raster format
                tifffile.imwrite(
                    str(tif_path),
                    tensor_4ch.transpose(2, 0, 1).astype(np.uint16),
                    photometric='minisblack'
                )

            # Save preview JPEG (for Roboflow/CVAT annotation)
            if self.cfg.capture.save_preview_jpg:
                jpg_path = self.session_dir / f"{base_name}_preview.jpg"
                cv2.imwrite(str(jpg_path), preview,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Save individual band JPEGs (optional)
            if self.cfg.capture.save_band_jpgs:
                for band_name, band_arr in bands.items():
                    norm = (band_arr.astype(np.float32) /
                            band_arr.max() * 255).astype(np.uint8)
                    colored = cv2.applyColorMap(norm, cv2.COLORMAP_HOT)
                    bpath = self.session_dir / f"{base_name}_{band_name}.jpg"
                    cv2.imwrite(str(bpath), colored)

            # Collect metadata
            gps_fix  = self.gps.get_fix()
            pose     = self.odom.get_pose()

            meta = {
                'frame_id':     frame_id,
                'session_id':   self.session_id,
                'timestamp':    ts.isoformat(),
                'unix_time':    ts.timestamp(),
                'trigger':      trigger,          # 'auto' | 'manual' | 'flagged'
                'flagged':      flagged,
                'frame_index':  self.frame_count,

                # Research labels (set at session start, not per-frame)
                'researcher':       self.cfg.session.researcher,
                'location':         self.cfg.session.location,
                'field_id':         self.cfg.session.field_id,
                'crop':             self.cfg.session.crop,
                'growth_stage':     self.cfg.session.growth_stage.value,
                'detection_mode':   self.cfg.session.detection_mode.value,
                'camera_mode':      self.cfg.session.camera_mode.value,
                'notes':            self.cfg.session.notes,

                # Camera info
                'camera': {
                    'model':        'simulated',
                    'width':        raw.shape[1],
                    'height':       raw.shape[0],
                    'bands':        self.cfg.ms_camera.channel_order,
                    'sim_mode':     self.sim_mode,
                },

                # Image statistics per band
                'band_stats': {
                    name: {
                        'mean':  float(arr.mean()),
                        'std':   float(arr.std()),
                        'min':   int(arr.min()),
                        'max':   int(arr.max()),
                    }
                    for name, arr in bands.items()
                },

                # Spatial data
                'gps':  gps_fix,
                'pose': pose,

                # File paths (relative to session dir)
                'files': {
                    'raw_tif':      f"{base_name}_raw.tif"
                                    if self.cfg.capture.save_raw_tif else None,
                    'preview_jpg':  f"{base_name}_preview.jpg"
                                    if self.cfg.capture.save_preview_jpg else None,
                },
            }

            # Save metadata JSON
            meta_path = self.meta_dir / f"{base_name}_meta.json"
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)

            self.frame_count += 1
            if flagged:
                self.flagged_count += 1

            status = "📌 FLAGGED" if flagged else "📷"
            gps_str = (f"GPS: {gps_fix['lat']:.5f},{gps_fix['lon']:.5f}"
                       if gps_fix['fix_valid'] else "GPS: no fix")
            pose_str = (f"pos: ({pose['x']:.2f},{pose['y']:.2f})"
                        if pose else "odom: --")
            logging.info(
                f"{status} [{trigger}] frame {self.frame_count:05d} | "
                f"{gps_str} | {pose_str}"
            )
            return frame_id

        except Exception as e:
            logging.error(f"Save failed: {e}")
            self.failed_count += 1
            return None

    # ── Session Summary ───────────────────────────────────────

    def _save_session_summary(self):
        """Write session summary JSON at end of session."""
        summary = {
            'session_id':       self.session_id,
            'start_time':       self.session_id[:15],  # embedded in ID
            'end_time':         datetime.now().isoformat(),
            'total_frames':     self.frame_count,
            'flagged_frames':   self.flagged_count,
            'failed_frames':    self.failed_count,
            'detection_mode':   self.cfg.session.detection_mode.value,
            'camera_mode':      self.cfg.session.camera_mode.value,
            'growth_stage':     self.cfg.session.growth_stage.value,
            'field_id':         self.cfg.session.field_id,
            'location':         self.cfg.session.location,
            'notes':            self.cfg.session.notes,
            'storage_path':     str(self.session_dir),
            'sim_mode':         self.sim_mode,
        }
        out = (self.cfg.storage.base_path / self.cfg.storage.sessions_dir /
               f"{self.session_id}_summary.json")
        with open(out, 'w') as f:
            json.dump(summary, f, indent=2)
        logging.info(f"Session summary saved → {out}")

    def _print_stats(self):
        print(f"\n  ┌─── Session Stats ───────────────┐")
        print(f"  │ Session  : {self.session_id[:30]}")
        print(f"  │ Frames   : {self.frame_count}")
        print(f"  │ Flagged  : {self.flagged_count}")
        print(f"  │ Failed   : {self.failed_count}")
        print(f"  │ Mode     : {self.cfg.session.detection_mode.value}")
        print(f"  │ Camera   : {self.cfg.session.camera_mode.value}")
        print(f"  │ Paused   : {self.paused}")
        print(f"  └─────────────────────────────────┘\n")

    # ── Main Run Loop ─────────────────────────────────────────

    def run(self):
        """Main capture loop. Blocking — press 'q' to exit."""
        print(self.cfg.summary())
        print()
        print("  Controls:")
        print(f"    [{self.cfg.capture.manual_trigger_key}] Manual capture")
        print(f"    [{self.cfg.capture.flag_key}] Flag next frame as priority")
        print(f"    [p] Pause / resume auto-capture")
        print(f"    [s] Session stats")
        print(f"    [q] Quit and save session")
        print()

        # Start hardware
        if not self._init_camera():
            logging.error("Camera initialization failed")
            return

        self.gps.start()
        self.odom.start()

        # Wait briefly for GPS to settle
        time.sleep(1.0)

        self._running = True
        self._last_auto_time = time.time()

        logging.info(f"Capture session started: {self.session_id}")
        logging.info(
            f"Auto-capture every {self.cfg.capture.auto_interval}s | "
            f"Manual: '{self.cfg.capture.manual_trigger_key}' | "
            f"Flag: '{self.cfg.capture.flag_key}'"
        )

        try:
            while self._running:
                now = time.time()

                # ── Read keyboard (non-blocking via OpenCV waitKey) ──
                key = cv2.waitKey(50) & 0xFF  # 50ms poll

                if key == ord('q'):
                    logging.info("Quit key pressed — ending session")
                    break

                elif key == ord(self.cfg.capture.manual_trigger_key):
                    raw = self._get_frame()
                    if raw is not None:
                        self._save_frame(raw, flagged=self._flag_next,
                                         trigger='manual')
                        self._flag_next = False

                elif key == ord(self.cfg.capture.flag_key):
                    self._flag_next = True
                    logging.info("📌 Next capture will be FLAGGED as priority")

                elif key == ord('p'):
                    self.paused = not self.paused
                    state = "PAUSED" if self.paused else "RESUMED"
                    logging.info(f"Auto-capture {state}")

                elif key == ord('s'):
                    self._print_stats()

                # ── Auto-capture ──
                if (not self.paused and
                        now - self._last_auto_time >= self.cfg.capture.auto_interval):
                    self._last_auto_time = now
                    raw = self._get_frame()
                    if raw is not None:
                        self._save_frame(raw, flagged=self._flag_next,
                                         trigger='auto')
                        self._flag_next = False

                    # Check max frames limit
                    if (self.cfg.capture.max_frames is not None and
                            self.frame_count >= self.cfg.capture.max_frames):
                        logging.info(
                            f"Max frames ({self.cfg.capture.max_frames}) reached"
                        )
                        break

        except KeyboardInterrupt:
            logging.info("Interrupted by user")

        finally:
            self._running = False
            self._shutdown()

    def _shutdown(self):
        """Clean shutdown of all hardware and save session summary."""
        logging.info("Shutting down...")

        self.gps.stop()
        self.odom.stop()
        cv2.destroyAllWindows()

        self._save_session_summary()
        self._print_stats()

        logging.info(
            f"Session complete. {self.frame_count} frames saved to:\n"
            f"  {self.session_dir}"
        )


# ─────────────────────────────────────────────────────────────
#  CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='ABEN Field Capture Tool — multispectral data collection'
    )
    parser.add_argument(
        '--mode', choices=['weed', 'cls'], default='weed',
        help='Detection mode (default: weed)'
    )
    parser.add_argument(
        '--field', type=str, default='',
        help='Field or plot ID (e.g. Field_01, Plot_A)'
    )
    parser.add_argument(
        '--stage',
        choices=[s.value for s in GrowthStage],
        default=GrowthStage.FOUR_LEAF.value,
        help='Crop growth stage'
    )
    parser.add_argument(
        '--interval', type=float, default=2.0,
        help='Auto-capture interval in seconds (default: 2.0)'
    )
    parser.add_argument(
        '--max-frames', type=int, default=None,
        help='Maximum frames to capture (default: unlimited)'
    )
    parser.add_argument(
        '--no-gps', action='store_true',
        help='Disable GPS even if dongle is connected'
    )
    parser.add_argument(
        '--notes', type=str, default='',
        help='Session notes (quoted string)'
    )
    parser.add_argument(
        '--rgb', action='store_true',
        help='Use RGB camera configuration'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Build config
    stage = GrowthStage(args.stage)
    cam   = CameraMode.RGB if args.rgb else CameraMode.MULTISPECTRAL

    if args.mode == 'weed':
        cfg = get_weed_config(field_id=args.field,
                              growth_stage=stage,
                              camera_mode=cam)
    else:
        cfg = get_cls_config(field_id=args.field,
                             growth_stage=stage,
                             camera_mode=cam)

    # Apply CLI overrides
    cfg.capture.auto_interval   = args.interval
    cfg.capture.max_frames      = args.max_frames
    cfg.session.notes           = args.notes

    if args.no_gps:
        cfg.gps.enabled = False

    # Run
    tool = CaptureTool(cfg)
    tool.run()