#!/usr/bin/env python3
"""
event_logger.py
ABEN Field Detection System — Research Event Logger

Logs every spray event with full spatial, spectral, and detection
metadata. This is the primary research data output — every logged
event is a publishable data point.

Per-event record includes:
  - Timestamp + session ID
  - Detection mode (WEED / CLS)
  - Detected class(es) + confidence scores
  - Robot position (odometry x, y, heading)
  - GPS coordinates (lat, lon) when available
  - Spray zone + nozzle fired
  - Spray duration
  - Growth stage + field ID
  - Band statistics at time of spray
  - Link to the captured frame files

Output formats:
  - JSON lines (.jsonl) — one event per line, streamable
  - CSV summary — for direct import into R / Excel / MATLAB
  - Session summary JSON — human-readable field report

Author : Nana | NDSU / PhD Imaging System
"""

import csv
import json
import time
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

try:
    from core.detection_config_rgb import RGBConfig as ABENConfig
except ImportError:
    from detection_config_rgb import RGBConfig as ABENConfig
try:
    from core.actuation_controller import SprayEvent
except ImportError:
    from actuation_controller import SprayEvent


# ─────────────────────────────────────────────────────────────
#  LOG ENTRY
# ─────────────────────────────────────────────────────────────

@dataclass
class EventLogEntry:
    """
    Complete research record for one spray event.
    Extends SprayEvent with additional context fields.
    """
    # Core identifiers
    event_id:        str
    session_id:      str
    timestamp:       float
    datetime_str:    str

    # Research context
    researcher:      str
    institution:     str
    location:        str
    field_id:        str
    crop:            str
    growth_stage:    str
    detection_mode:  str

    # Detection data
    zone_id:         int
    zone_name:       str
    nozzle_id:       int
    spray_duration:  float
    flagged_cls:     bool
    detections:      List[Dict]    # full detection records

    # Spatial data
    pose_x:          Optional[float] = None
    pose_y:          Optional[float] = None
    pose_heading:    Optional[float] = None
    pose_speed:      Optional[float] = None
    gps_lat:         Optional[float] = None
    gps_lon:         Optional[float] = None
    gps_altitude:    Optional[float] = None
    gps_satellites:  Optional[int]   = None
    gps_valid:       bool = False

    # Frame reference (links to capture_tool output)
    frame_id:        Optional[str] = None
    frame_path:      Optional[str] = None

    # Computed summary fields
    detection_count: int = 0
    top_class:       str = ''
    top_confidence:  float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_csv_row(self) -> Dict:
        """Flat dict for CSV export — one row per event."""
        return {
            'event_id':        self.event_id,
            'session_id':      self.session_id,
            'datetime':        self.datetime_str,
            'unix_timestamp':  self.timestamp,
            'researcher':      self.researcher,
            'location':        self.location,
            'field_id':        self.field_id,
            'crop':            self.crop,
            'growth_stage':    self.growth_stage,
            'detection_mode':  self.detection_mode,
            'zone_name':       self.zone_name,
            'nozzle_id':       self.nozzle_id + 1,  # 1-indexed for readability
            'spray_duration_s': self.spray_duration,
            'flagged_cls':     self.flagged_cls,
            'detection_count': self.detection_count,
            'top_class':       self.top_class,
            'top_confidence':  round(self.top_confidence, 4),
            'pose_x_m':        self.pose_x,
            'pose_y_m':        self.pose_y,
            'pose_heading_deg': self.pose_heading,
            'pose_speed_ms':   self.pose_speed,
            'gps_lat':         self.gps_lat,
            'gps_lon':         self.gps_lon,
            'gps_altitude_m':  self.gps_altitude,
            'gps_satellites':  self.gps_satellites,
            'gps_valid':       self.gps_valid,
            'frame_id':        self.frame_id,
        }


# CSV column order — defines the exported spreadsheet layout
CSV_COLUMNS = [
    'event_id', 'session_id', 'datetime', 'unix_timestamp',
    'researcher', 'location', 'field_id', 'crop', 'growth_stage',
    'detection_mode', 'zone_name', 'nozzle_id', 'spray_duration_s',
    'flagged_cls', 'detection_count', 'top_class', 'top_confidence',
    'pose_x_m', 'pose_y_m', 'pose_heading_deg', 'pose_speed_ms',
    'gps_lat', 'gps_lon', 'gps_altitude_m', 'gps_satellites', 'gps_valid',
    'frame_id',
]


# ─────────────────────────────────────────────────────────────
#  EVENT LOGGER
# ─────────────────────────────────────────────────────────────

class EventLogger:
    """
    Logs spray events to disk in real time.

    Designed for field use:
      - Thread-safe write queue (never blocks the detection loop)
      - JSONL format (append-only, survives power loss mid-session)
      - CSV exported at session end for direct analysis
      - Session summary JSON for field notes

    Usage:
        logger = EventLogger(cfg, session_id)
        logger.start()

        # Pass as callback to ActuationController:
        ctrl = ActuationController(cfg, gantry,
                                   on_spray_event=logger.log_event)

        # At end of session:
        summary = logger.stop()
    """

    def __init__(self, cfg: ABENConfig, session_id: str):
        self.cfg        = cfg
        self.session_id = session_id
        self._lock      = threading.Lock()
        # Independent pending queues per output destination -- JSONL and
        # CSV writes can fail independently (e.g. one file's directory
        # is fine, the other's underlying device just dropped), so they
        # must be retried independently. A shared queue that only clears
        # on "at least one destination succeeded" would either lose data
        # (clear-before-write, the original bug) or write duplicate
        # JSONL lines on retry after a CSV-only failure.
        self._pending_jsonl: List[EventLogEntry] = []
        self._pending_csv:   List[EventLogEntry] = []
        self._entries:  List[EventLogEntry] = []
        self._running   = False
        self._writer_thread = None

        # Setup paths — based on cfg.logging.base_dir, the same
        # already-working config path detection_panel_rgb.py's session
        # metadata uses. (Previously referenced cfg.storage, which does
        # not exist anywhere on RGBConfig -- constructing an EventLogger
        # at all raised AttributeError immediately. See git history for
        # the full story: this class was never actually wired into the
        # live GUI, which is why nobody hit this until now.)
        events_dir = cfg.logging.base_dir / "events" / session_id
        events_dir.mkdir(parents=True, exist_ok=True)

        self._jsonl_path = events_dir / f"{session_id}_events.jsonl"
        self._csv_path   = events_dir / f"{session_id}_events.csv"
        self._summary_path = (
            cfg.logging.base_dir / f"{session_id}_event_summary.json"
        )

        # CSV writer setup
        self._csv_file   = None
        self._csv_writer = None

        logging.info(
            f"EventLogger initialized | "
            f"session: {session_id}"
        )
        logging.info(f"  JSONL: {self._jsonl_path}")
        logging.info(f"  CSV:   {self._csv_path}")

    # ── Lifecycle ─────────────────────────────────────────────

    def start(self):
        """Open log files and start background writer thread."""
        self._running = True

        # Open CSV with header
        self._csv_file = open(self._csv_path, 'w', newline='')
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=CSV_COLUMNS
        )
        self._csv_writer.writeheader()
        self._csv_file.flush()

        # Background flush thread (writes queue to disk every 0.5s)
        self._writer_thread = threading.Thread(
            target=self._flush_loop, daemon=True
        )
        self._writer_thread.start()

        logging.info("EventLogger started — logging to disk")

    def stop(self) -> Dict:
        """
        Stop logger, flush remaining events, write summary.
        Returns session summary dict.
        """
        self._running = False

        # Final flush
        self._flush_to_disk()

        # Warn loudly if anything still didn't make it to disk after
        # the final attempt -- this is the one place the researcher
        # will actually see it, since the summary JSON below is built
        # from self._entries (always complete in memory) and would
        # otherwise look identical whether the on-disk files are
        # complete or silently missing events.
        with self._lock:
            lost_jsonl = len(self._pending_jsonl)
            lost_csv   = len(self._pending_csv)
        if lost_jsonl or lost_csv:
            logging.error(
                f"⚠⚠ EventLogger stop(): {lost_jsonl} event(s) never "
                f"written to JSONL, {lost_csv} never written to CSV, "
                f"after final flush attempt. These events ARE still in "
                f"the session summary below (in-memory), but the "
                f"{self._jsonl_path.name} / {self._csv_path.name} files "
                f"on disk are INCOMPLETE. Check disk space and that "
                f"the storage path is still mounted."
            )

        # Close CSV
        if self._csv_file:
            self._csv_file.close()

        # Write summary
        self._summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = self._build_summary()
        with open(self._summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logging.info(
            f"EventLogger stopped | "
            f"{len(self._entries)} events logged"
        )
        logging.info(f"  Summary: {self._summary_path}")

        return summary

    # ── Logging API ───────────────────────────────────────────

    def log_event(self, event: SprayEvent,
                  frame_id: Optional[str] = None,
                  frame_path: Optional[str] = None):
        """
        Log a spray event. Thread-safe.
        Pass this as on_spray_event callback to ActuationController.

        Args:
            event:      SprayEvent from ActuationController
            frame_id:   Optional link to the concurrent capture frame
            frame_path: Optional path to the frame files
        """
        entry = self._build_entry(event, frame_id, frame_path)

        with self._lock:
            self._entries.append(entry)
            self._pending_jsonl.append(entry)
            self._pending_csv.append(entry)

        logging.info(
            f"📝 Event logged: {entry.event_id} | "
            f"{entry.zone_name} | "
            f"{entry.top_class} ({entry.top_confidence:.2f}) | "
            f"GPS valid: {entry.gps_valid}"
        )

    def log_event_from_dict(self, event_dict: Dict):
        """Log from a pre-built dict (for replay/testing)."""
        # Minimal reconstruction
        class _FakeEvent:
            pass
        ev = _FakeEvent()
        for k, v in event_dict.items():
            setattr(ev, k, v)
        self.log_event(ev)

    # ── Entry construction ────────────────────────────────────

    def _build_entry(self, event: SprayEvent,
                     frame_id: Optional[str],
                     frame_path: Optional[str]) -> EventLogEntry:
        """Convert SprayEvent + context into a full EventLogEntry."""
        cfg = self.cfg
        ts  = event.timestamp
        dt  = datetime.fromtimestamp(ts).isoformat()

        # Extract top detection
        top_class = ''
        top_conf  = 0.0
        if event.detections:
            best = max(event.detections,
                       key=lambda d: d.get('confidence', 0))
            top_class = best.get('class_name', '')
            top_conf  = best.get('confidence', 0.0)

        # Extract spatial data
        pose    = event.pose or {}
        gps_fix = event.gps  or {}

        return EventLogEntry(
            event_id=event.event_id,
            session_id=self.session_id,
            timestamp=ts,
            datetime_str=dt,

            researcher=cfg.session.researcher,
            institution=cfg.session.institution,
            location=cfg.session.location,
            field_id=cfg.session.field_id,
            crop=cfg.session.crop,
            growth_stage=cfg.session.growth_stage.value,
            detection_mode=event.mode,

            zone_id=event.zone_id,
            zone_name=event.zone_name,
            nozzle_id=event.nozzle_id,
            spray_duration=event.spray_duration,
            flagged_cls=event.flagged_cls,
            detections=event.detections,

            pose_x=pose.get('x'),
            pose_y=pose.get('y'),
            pose_heading=pose.get('heading'),
            pose_speed=pose.get('speed'),

            gps_lat=gps_fix.get('lat'),
            gps_lon=gps_fix.get('lon'),
            gps_altitude=gps_fix.get('altitude'),
            gps_satellites=gps_fix.get('satellites'),
            gps_valid=gps_fix.get('fix_valid', False),

            frame_id=frame_id,
            frame_path=frame_path,

            detection_count=len(event.detections),
            top_class=top_class,
            top_confidence=top_conf,
        )

    # ── Disk I/O ──────────────────────────────────────────────

    def _flush_loop(self):
        """Background thread — flushes queue to disk every 0.5s."""
        while self._running:
            time.sleep(0.5)
            self._flush_to_disk()

    def _flush_to_disk(self):
        """
        Write all pending entries to JSONL and CSV.

        Each destination is written and retried independently: a batch
        is only removed from its pending queue after that destination's
        write actually succeeds. If a write fails (disk full, storage
        unmounted, permissions, etc.) the batch stays queued and is
        retried on the next flush cycle rather than being silently
        dropped -- this is what actually makes the JSONL file able to
        "survive power loss mid-session" as documented, instead of
        just losing whatever was in flight at the moment of failure.

        Uses del pending[:n] rather than clear() when removing a
        written batch, because new entries may have been appended to
        the pending list (from log_event(), on another thread) between
        when this batch was snapshotted and when the write completed --
        clear() would incorrectly drop those too.
        """
        with self._lock:
            jsonl_batch = list(self._pending_jsonl)
            csv_batch   = list(self._pending_csv)

        if jsonl_batch:
            try:
                with open(self._jsonl_path, 'a') as f:
                    for entry in jsonl_batch:
                        f.write(json.dumps(entry.to_dict()) + '\n')
                with self._lock:
                    del self._pending_jsonl[:len(jsonl_batch)]
            except Exception as e:
                logging.error(
                    f"JSONL write error (will retry next flush, "
                    f"{len(jsonl_batch)} event(s) still pending): {e}"
                )

        if csv_batch:
            try:
                if self._csv_writer:
                    for entry in csv_batch:
                        self._csv_writer.writerow(entry.to_csv_row())
                    self._csv_file.flush()
                    with self._lock:
                        del self._pending_csv[:len(csv_batch)]
            except Exception as e:
                logging.error(
                    f"CSV write error (will retry next flush, "
                    f"{len(csv_batch)} event(s) still pending): {e}"
                )

    # ── Summary ───────────────────────────────────────────────

    def _build_summary(self) -> Dict:
        """Build session summary from all logged events."""
        entries = self._entries

        if not entries:
            return {
                'session_id':   self.session_id,
                'total_events': 0,
                'message':      'No spray events recorded this session',
            }

        # Per-class breakdown
        class_counts: Dict[str, int] = {}
        zone_counts:  Dict[str, int] = {}
        for e in entries:
            class_counts[e.top_class] = (
                class_counts.get(e.top_class, 0) + 1
            )
            zone_counts[e.zone_name] = (
                zone_counts.get(e.zone_name, 0) + 1
            )

        # GPS coverage
        gps_events = [e for e in entries if e.gps_valid]
        gps_coverage = {
            'events_with_fix': len(gps_events),
            'coverage_pct':    round(
                len(gps_events) / len(entries) * 100, 1
            ),
        }
        if gps_events:
            lats = [e.gps_lat for e in gps_events if e.gps_lat]
            lons = [e.gps_lon for e in gps_events if e.gps_lon]
            if lats and lons:
                gps_coverage['lat_range'] = [min(lats), max(lats)]
                gps_coverage['lon_range'] = [min(lons), max(lons)]

        # Confidence distribution
        confidences = [e.top_confidence for e in entries
                       if e.top_confidence > 0]
        conf_stats = {}
        if confidences:
            conf_stats = {
                'mean':  round(sum(confidences) / len(confidences), 3),
                'min':   round(min(confidences), 3),
                'max':   round(max(confidences), 3),
            }

        # Session duration
        duration = entries[-1].timestamp - entries[0].timestamp

        return {
            'session_id':       self.session_id,
            'researcher':       self.cfg.session.researcher,
            'location':         self.cfg.session.location,
            'field_id':         self.cfg.session.field_id,
            'crop':             self.cfg.session.crop,
            'growth_stage':     self.cfg.session.growth_stage.value,
            'detection_mode':   self.cfg.session.detection_mode.value,
            'total_events':     len(entries),
            'session_duration_s': round(duration, 1),
            'events_per_minute': round(
                len(entries) / max(duration / 60, 0.01), 2
            ),
            'detections_by_class': class_counts,
            'events_by_zone':      zone_counts,
            'confidence_stats':    conf_stats,
            'gps_coverage':        gps_coverage,
            'files': {
                'jsonl':   str(self._jsonl_path),
                'csv':     str(self._csv_path),
                'summary': str(self._summary_path),
            },
        }

    # ── Query API ─────────────────────────────────────────────

    def get_events(self) -> List[EventLogEntry]:
        """Return all logged entries for this session."""
        with self._lock:
            return list(self._entries)

    def get_event_count(self) -> int:
        with self._lock:
            return len(self._entries)

    def export_geojson(self, output_path: Optional[Path] = None) -> Path:
        """
        Export spray events as GeoJSON for field mapping in QGIS/ArcGIS.
        Only includes events with valid GPS fix.

        Returns path to the exported .geojson file.
        """
        if output_path is None:
            events_dir = self.cfg.logging.base_dir / "events" / self.session_id
            output_path = events_dir / f"{self.session_id}_map.geojson"

        with self._lock:
            gps_entries = [e for e in self._entries if e.gps_valid]

        features = []
        for e in gps_entries:
            features.append({
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [e.gps_lon, e.gps_lat],
                },
                'properties': {
                    'event_id':      e.event_id,
                    'datetime':      e.datetime_str,
                    'mode':          e.detection_mode,
                    'zone':          e.zone_name,
                    'class':         e.top_class,
                    'confidence':    e.top_confidence,
                    'flagged_cls':   e.flagged_cls,
                    'pose_x':        e.pose_x,
                    'pose_y':        e.pose_y,
                    'heading':       e.pose_heading,
                },
            })

        geojson = {
            'type': 'FeatureCollection',
            'features': features,
            'metadata': {
                'session_id':  self.session_id,
                'field_id':    self.cfg.session.field_id,
                'location':    self.cfg.session.location,
                'total_points': len(features),
            },
        }

        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)

        logging.info(
            f"GeoJSON exported: {len(features)} GPS points → {output_path}"
        )
        return output_path


# ─────────────────────────────────────────────────────────────
#  SELF TEST
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import shutil
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    from core.detection_config_rgb import (
        get_weed_config, get_cls_config, GrowthStage
    )
    from core.actuation_controller import SprayEvent

    print("=" * 55)
    print("ABEN Event Logger — Self Test")
    print("=" * 55)
    print()

    # ── Setup ─────────────────────────────────────────────────
    cfg = get_weed_config(
        field_id='logger_test',
        growth_stage=GrowthStage.FOUR_LEAF
    )
    # Use temp directory for test
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    cfg.logging.base_dir = tmp

    session_id = "20260421_test_weed_mu_4_leaf_logger_test"
    logger = EventLogger(cfg, session_id)
    logger.start()

    # ── Test 1: log weed events ───────────────────────────────
    print("Test 1: Log 3 weed spray events")

    def make_event(zone_name, zone_id, nozzle_id,
                   class_name, conf, lat=None, lon=None):
        return SprayEvent(
            event_id=f"ev_{int(time.time())}_{zone_id:04d}",
            timestamp=time.time(),
            mode='weed',
            zone_id=zone_id,
            zone_name=zone_name,
            nozzle_id=nozzle_id,
            detections=[{
                'class_id': 1,
                'class_name': class_name,
                'confidence': conf,
                'bbox': [80, 200, 140, 280],
                'center': [110, 240],
                'area': 3600,
            }],
            spray_duration=0.5,
            pose={'x': 4.23, 'y': 0.08, 'heading': 5.0, 'speed': 0.15},
            gps={
                'lat': lat, 'lon': lon,
                'altitude': 299.0,
                'satellites': 8,
                'fix_valid': lat is not None,
            } if lat else {'fix_valid': False},
            flagged_cls=False,
        )

    events = [
        make_event('ZoneA', 0, 0, 'kochia',    0.89,
                   46.2910, -96.6120),
        make_event('ZoneB', 1, 1, 'waterhemp', 0.76,
                   46.2911, -96.6118),
        make_event('ZoneC', 2, 2, 'kochia',    0.93,
                   46.2912, -96.6115),
    ]

    for ev in events:
        logger.log_event(ev, frame_id=f"frame_{ev.zone_id:05d}")

    time.sleep(0.8)  # let background flush run

    assert logger.get_event_count() == 3
    print(f"  Events logged: {logger.get_event_count()} ✓")

    # Verify JSONL written
    assert logger._jsonl_path.exists()
    lines = logger._jsonl_path.read_text().strip().split('\n')
    assert len(lines) == 3
    first = json.loads(lines[0])
    assert first['event_id'] == events[0].event_id
    assert first['top_class'] == 'kochia'
    assert first['gps_valid'] == True
    assert first['frame_id'] == 'frame_00000'
    print(f"  JSONL written: {len(lines)} lines ✓")
    print(f"  First entry top_class: {first['top_class']} ✓")
    print(f"  GPS valid: {first['gps_valid']} ✓")
    print()

    # ── Test 2: CLS mode events ───────────────────────────────
    print("Test 2: CLS mode events — flagged")
    cfg_cls = get_cls_config(field_id='cls_logger_test')
    cfg_cls.logging.base_dir = tmp
    session_cls = "20260421_test_cls_mu_vegetative_cls_logger_test"
    logger_cls = EventLogger(cfg_cls, session_cls)
    logger_cls.start()

    cls_ev = SprayEvent(
        event_id="ev_cls_0001",
        timestamp=time.time(),
        mode='cls',
        zone_id=1,
        zone_name='ZoneB',
        nozzle_id=1,
        detections=[{
            'class_id': 7,
            'class_name': 'cls_infected',
            'confidence': 0.91,
            'bbox': [280, 150, 380, 250],
            'center': [330, 200],
            'area': 10000,
        }],
        spray_duration=1.0,
        pose={'x': 8.45, 'y': 0.12, 'heading': 4.5, 'speed': 0.0},
        gps={
            'lat': 46.2915, 'lon': -96.6110,
            'altitude': 299.0, 'satellites': 9,
            'fix_valid': True,
        },
        flagged_cls=True,
    )

    logger_cls.log_event(cls_ev)
    time.sleep(0.8)

    assert logger_cls.get_event_count() == 1
    entry = logger_cls.get_events()[0]
    assert entry.flagged_cls == True
    assert entry.top_class == 'cls_infected'
    assert entry.detection_mode == 'cls'
    print(f"  CLS event logged: {entry.event_id} ✓")
    print(f"  Flagged CLS: {entry.flagged_cls} ✓")
    print()

    # ── Test 3: session summary ───────────────────────────────
    print("Test 3: Session summary")
    summary = logger.stop()
    print(f"  Total events:    {summary['total_events']} ✓")
    print(f"  Classes found:   {summary['detections_by_class']} ✓")
    print(f"  Events by zone:  {summary['events_by_zone']} ✓")
    print(f"  GPS coverage:    {summary['gps_coverage']['coverage_pct']}% ✓")
    print(f"  Confidence mean: {summary['confidence_stats']['mean']} ✓")
    assert summary['total_events'] == 3
    assert 'kochia' in summary['detections_by_class']
    assert summary['gps_coverage']['coverage_pct'] == 100.0
    print()

    # ── Test 4: CSV export ────────────────────────────────────
    print("Test 4: CSV export")
    assert logger._csv_path.exists()
    with open(logger._csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 3
    assert rows[0]['top_class'] == 'kochia'
    assert rows[0]['zone_name'] == 'ZoneA'
    assert rows[0]['gps_lat'] == '46.291'
    print(f"  CSV rows: {len(rows)} ✓")
    print(f"  Columns: {len(rows[0])} ✓")
    print(f"  top_class: {rows[0]['top_class']} ✓")
    print()

    # ── Test 5: GeoJSON export ────────────────────────────────
    print("Test 5: GeoJSON field map export")
    geojson_path = logger.export_geojson()
    assert geojson_path.exists()
    with open(geojson_path) as f:
        gj = json.load(f)
    assert gj['type'] == 'FeatureCollection'
    assert len(gj['features']) == 3  # all 3 had GPS
    feat = gj['features'][0]
    assert feat['geometry']['type'] == 'Point'
    assert feat['properties']['class'] == 'kochia'
    print(f"  GeoJSON features: {len(gj['features'])} ✓")
    print(f"  First point: lon={feat['geometry']['coordinates'][0]}, "
          f"lat={feat['geometry']['coordinates'][1]} ✓")
    print(f"  Properties: {list(feat['properties'].keys())} ✓")
    print()

    # ── Test 6: write failure doesn't silently drop events ────
    print("Test 6: JSONL write failure is retried, not silently dropped")
    print("  (this is the actual bug: _flush_to_disk() used to clear")
    print("   the queue BEFORE confirming the write succeeded)")

    cfg6 = get_weed_config(field_id='resilience_test')
    tmp6 = Path(tempfile.mkdtemp())
    cfg6.logging.base_dir = tmp6
    session6 = "resilience_test_session"
    logger6 = EventLogger(cfg6, session6)
    logger6.start()

    ev6 = make_event('ZoneA', 0, 0, 'kochia', 0.85, 46.29, -96.61)
    logger6.log_event(ev6)

    # Simulate a transient write failure by making the JSONL path
    # temporarily unwritable (directory doesn't exist / permission-like
    # failure) right as a flush would occur.
    real_path = logger6._jsonl_path
    logger6._jsonl_path = Path("/nonexistent_dir_on_purpose/events.jsonl")

    logger6._flush_to_disk()  # this attempt should fail and NOT drop the event
    with logger6._lock:
        still_pending = len(logger6._pending_jsonl)
    assert still_pending == 1, (
        f"Event should still be pending after a failed write, "
        f"got {still_pending} pending (bug: event was dropped)"
    )
    print(f"  Event survived a failed write attempt (still pending) ✓")

    # "Fix" the path (simulating the disk/mount coming back) and retry
    logger6._jsonl_path = real_path
    logger6._flush_to_disk()
    with logger6._lock:
        still_pending = len(logger6._pending_jsonl)
    assert still_pending == 0, "Event should be written once the path is valid again"
    assert real_path.exists()
    lines = real_path.read_text().strip().split('\n')
    assert len(lines) == 1
    assert json.loads(lines[0])['event_id'] == ev6.event_id
    print(f"  Event successfully written on retry once path recovered ✓")

    logger6.stop()
    shutil.rmtree(tmp6)
    print()

    # Cleanup
    logger_cls.stop()
    shutil.rmtree(tmp)

    print("=" * 55)
    print("event_logger.py  ✓  ALL TESTS PASSED")
    print("=" * 55)
    print()
    print("Output files per session:")
    print("  <session>_events.jsonl  — append-only event stream")
    print("  <session>_events.csv    — for R / Excel / MATLAB")
    print("  <session>_event_summary.json — field report")
    print("  <session>_map.geojson   — for QGIS / ArcGIS mapping")