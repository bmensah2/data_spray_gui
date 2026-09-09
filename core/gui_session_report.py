"""
core/gui_session_report.py
ABEN Dual RGB Imaging System — GUI session report builder

Assembles a complete session record from an ARM DETECTION session:
the same spray events shown in the Session Analysis tab, plus the
system provenance needed to actually defend the numbers in a
publication (camera settings, model + class list, geometry, software
versions, git commit).

Why this exists: "Generate Session Report" previously ran
generate_report_rgb.js against the JSON written by
spray_mission_rgb.py -- the START MISSION subprocess. That is a
completely separate data path from the GUI's own detection sessions,
so a session run via ARM DETECTION produced rich data
(EventLogger + Session Analysis) that the report simply could not
see. This module reads the GUI's own session data and emits a JSON in
the shape generate_report_rgb.js already understands, so the existing
Word-document generator works for GUI sessions too.
"""

import json
import logging
import statistics
from datetime import datetime
from pathlib import Path


def _event_rows(events) -> list:
    """SprayEvent objects -> plain dicts, matching the Session
    Analysis feed's columns so the report and the on-screen table can
    never disagree about what happened."""
    rows = []
    for e in events:
        names = [d.get("class_name", "?") for d in (e.detections or [])]
        confs = [d.get("confidence", 0.0) for d in (e.detections or [])]
        pose  = e.pose or {}
        gps   = e.gps or {}
        rows.append({
            "timestamp":     e.timestamp,
            "time":          datetime.fromtimestamp(
                                e.timestamp).strftime("%H:%M:%S"),
            "zone_name":     e.zone_name,
            "nozzle":        f"N{e.nozzle_id + 1}",
            "nozzle_id":     e.nozzle_id,
            "mode":          e.mode,
            "classes":       names,
            "top_class":     names[0] if names else "",
            "confidence":    round(max(confs), 3) if confs else 0.0,
            "spray_duration_s": getattr(e, "spray_duration", None),
            "pose_x":        pose.get("x"),
            "pose_y":        pose.get("y"),
            "gps_lat":       gps.get("lat"),
            "gps_lon":       gps.get("lon"),
            "gps_valid":     bool(gps.get("fix_valid")),
            "flagged_cls":   bool(getattr(e, "flagged_cls", False)),
        })
    return rows


def _statistics(rows) -> dict:
    """Aggregate stats, computed from the same rows the report lists --
    so a number in the summary can always be traced to specific
    events rather than to a separately-maintained counter."""
    if not rows:
        return {"total_events": 0,
                "note": "No spray events recorded this session"}

    by_zone, by_class, by_nozzle = {}, {}, {}
    for r in rows:
        by_zone[r["zone_name"]] = by_zone.get(r["zone_name"], 0) + 1
        by_nozzle[r["nozzle"]]  = by_nozzle.get(r["nozzle"], 0) + 1
        for c in r["classes"]:
            by_class[c] = by_class.get(c, 0) + 1

    confs = [r["confidence"] for r in rows if r["confidence"] > 0]
    conf_stats = {}
    if confs:
        conf_stats = {
            "mean":   round(statistics.mean(confs), 3),
            "median": round(statistics.median(confs), 3),
            "min":    round(min(confs), 3),
            "max":    round(max(confs), 3),
        }
        if len(confs) > 1:
            conf_stats["stdev"] = round(statistics.stdev(confs), 3)

    t0, t1 = rows[0]["timestamp"], rows[-1]["timestamp"]
    duration = max(t1 - t0, 0.0)
    gps_ok = sum(1 for r in rows if r["gps_valid"])

    return {
        "total_events":        len(rows),
        "duration_s":          round(duration, 1),
        "events_per_minute":   round(len(rows) / max(duration / 60, 0.01), 2),
        "events_by_zone":      by_zone,
        "events_by_nozzle":    by_nozzle,
        "detections_by_class": by_class,
        "confidence_stats":    conf_stats,
        "cls_flagged":         sum(1 for r in rows if r["flagged_cls"]),
        "gps_coverage": {
            "events_with_fix": gps_ok,
            "coverage_pct":    round(gps_ok / len(rows) * 100, 1),
        },
        "first_event_iso": datetime.fromtimestamp(t0).isoformat(
                              timespec="seconds"),
        "last_event_iso":  datetime.fromtimestamp(t1).isoformat(
                              timespec="seconds"),
    }


def build_session_report(session_id: str,
                         events,
                         provenance: dict,
                         session_meta: dict,
                         started_at: float = None,
                         ended_at: float = None) -> dict:
    """
    Assemble the full report dict. Pure data assembly -- no file or
    hardware access -- so it's straightforward to test and can't fail
    partway through and leave a half-written record.
    """
    rows  = _event_rows(events)
    stats = _statistics(rows)

    report = {
        "report_type":   "aben_rgb_gui_session",
        "schema_version": 1,
        "generated_at":  datetime.now().isoformat(timespec="seconds"),
        "session_id":    session_id,
        "source":        "GUI ARM DETECTION (detection_panel_rgb)",
        "metadata":      dict(session_meta or {}),
        "provenance":    dict(provenance or {}),
        "statistics":    stats,
        "events":        rows,
    }

    if started_at:
        report["session_start_iso"] = datetime.fromtimestamp(
            started_at).isoformat(timespec="seconds")
    if ended_at:
        report["session_end_iso"] = datetime.fromtimestamp(
            ended_at).isoformat(timespec="seconds")
    if started_at and ended_at:
        report["session_wall_duration_s"] = round(ended_at - started_at, 1)

    # Flag gaps explicitly rather than letting a reader assume the
    # record is complete. A report that quietly omits which model was
    # used is worse than one that says so.
    warnings = []
    if not (session_meta or {}).get("operator"):
        warnings.append("No operator name recorded for this session.")
    model = (provenance or {}).get("model", {})
    if not model.get("model_exists", False):
        warnings.append(
            "Model file not found at the recorded path — detection may "
            "have run in stub mode.")
    if (provenance or {}).get("model_classes", {}).get("stub_mode"):
        warnings.append(
            "Detection engine was in STUB MODE — spray events are not "
            "from real inference.")
    cams = (provenance or {}).get("cameras", {})
    for side, c in cams.items():
        if not c.get("available"):
            warnings.append(
                f"{side.upper()} camera settings unavailable "
                f"({c.get('reason', 'unknown')}).")
    if (provenance or {}).get("software", {}).get("git_dirty") is True:
        warnings.append(
            "Working tree had uncommitted changes — the running code does "
            "not exactly match the recorded git commit.")
    if stats.get("total_events", 0) == 0:
        warnings.append("No spray events were recorded this session.")
    report["warnings"] = warnings

    return report


def write_session_report(path, report: dict) -> bool:
    """Write the report JSON. Returns True on success."""
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        return True
    except Exception as e:
        logging.error(f"could not write session report to {path}: {e}")
        return False


def format_console_summary(report: dict) -> str:
    """Human-readable summary, mirroring the Session Analysis tab."""
    L = []
    st = report.get("statistics", {})
    md = report.get("metadata", {})
    pv = report.get("provenance", {})
    sw = pv.get("software", {})
    mdl = pv.get("model", {})

    L.append("=" * 60)
    L.append(f"  SESSION REPORT — {report.get('session_id', '?')}")
    L.append("=" * 60)

    L.append("\nSESSION")
    L.append(f"  Operator:        {md.get('operator') or '(not recorded)'}")
    L.append(f"  Researcher:      {md.get('researcher') or '—'}")
    L.append(f"  Institution:     {md.get('institution') or '—'}")
    L.append(f"  Field / Location:{md.get('field_id') or '—'} / "
             f"{md.get('location') or '—'}")
    L.append(f"  Crop / Stage:    {md.get('crop') or '—'} / "
             f"{md.get('growth_stage') or '—'}")
    if md.get("notes"):
        L.append(f"  Notes:           {md['notes']}")

    L.append("\nSPRAY EVENTS")
    L.append(f"  Total events:    {st.get('total_events', 0)}")
    if st.get("total_events"):
        L.append(f"  Duration:        {st.get('duration_s', 0)} s "
                 f"({st.get('events_per_minute', 0)}/min)")
        L.append(f"  By zone:         {st.get('events_by_zone', {})}")
        L.append(f"  By nozzle:       {st.get('events_by_nozzle', {})}")
        L.append(f"  By class:        {st.get('detections_by_class', {})}")
        cs = st.get("confidence_stats", {})
        if cs:
            L.append(f"  Confidence:      mean={cs.get('mean')} "
                     f"median={cs.get('median')} "
                     f"range={cs.get('min')}–{cs.get('max')}")
        gc = st.get("gps_coverage", {})
        L.append(f"  GPS coverage:    {gc.get('events_with_fix', 0)}"
                 f"/{st.get('total_events')} ({gc.get('coverage_pct', 0)}%)")

    L.append("\nMODEL")
    L.append(f"  Path:            {mdl.get('model_path', '—')}")
    L.append(f"  Mode:            {mdl.get('detection_mode', '—')}")
    L.append(f"  Confidence thr:  {mdl.get('confidence_threshold', '—')}")
    L.append(f"  IoU threshold:   {mdl.get('iou_threshold', '—')}")
    mc = pv.get("model_classes", {})
    if mc.get("available"):
        L.append(f"  Classes ({mc.get('class_count')}): "
                 f"{list(mc.get('classes', {}).values())}")

    cams = pv.get("cameras", {})
    if cams:
        L.append("\nCAMERA SETTINGS")
        for side, c in cams.items():
            if c.get("available"):
                L.append(f"  {side.upper()}: exp={c.get('exposure')} "
                         f"gamma={c.get('gamma')} wb={c.get('wb_temp')} "
                         f"auto_wb={c.get('auto_wb')} "
                         f"auto_exp={c.get('auto_exposure')} "
                         f"autofocus={c.get('autofocus')}")
            else:
                L.append(f"  {side.upper()}: unavailable "
                         f"({c.get('reason', '?')})")

    L.append("\nSOFTWARE / REPRODUCIBILITY")
    L.append(f"  git commit:      {sw.get('git_commit', '—')}"
             f"{'  (DIRTY)' if sw.get('git_dirty') is True else ''}")
    L.append(f"  ultralytics:     {sw.get('ultralytics', '—')}")
    L.append(f"  torch:           {sw.get('torch', '—')}  "
             f"(CUDA: {sw.get('cuda_available', '—')})")
    L.append(f"  opencv:          {sw.get('opencv', '—')}")
    L.append(f"  python:          {sw.get('python', '—')}")

    warns = report.get("warnings", [])
    if warns:
        L.append("\n⚠ WARNINGS")
        for w in warns:
            L.append(f"  - {w}")

    L.append("=" * 60)
    return "\n".join(L)


if __name__ == "__main__":
    import sys, tempfile, time
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    from core.actuation_controller import SprayEvent

    print("=" * 55)
    print("core/gui_session_report.py — Self Test")
    print("=" * 55)

    now = time.time()
    events = [
        SprayEvent(event_id="e1", timestamp=now, mode="weed",
                   zone_id=0, zone_name="ZoneA", nozzle_id=0,
                   detections=[{"class_name": "common_ragweed",
                                "confidence": 0.91}],
                   spray_duration=0.5, pose={"x": 0.13, "y": -0.01},
                   gps={"fix_valid": True, "lat": 46.8, "lon": -96.8},
                   flagged_cls=False),
        SprayEvent(event_id="e2", timestamp=now + 12, mode="weed",
                   zone_id=3, zone_name="ZoneC", nozzle_id=2,
                   detections=[{"class_name": "kochia", "confidence": 0.76}],
                   spray_duration=0.5, pose={"x": 0.37, "y": -0.02},
                   gps=None, flagged_cls=False),
    ]
    meta = {"operator": "Nana Mensah", "researcher": "Nana Mensah",
            "institution": "NDSU", "field_id": "wilkin_plot_A",
            "location": "Wilkin County, MN", "crop": "sugarbeet",
            "growth_stage": "6_leaf", "notes": "clear, light wind"}
    prov = {
        "software": {"git_commit": "abc1234", "git_dirty": False,
                     "ultralytics": "8.4.40", "torch": "2.8.0",
                     "cuda_available": True, "opencv": "4.13.0",
                     "python": "3.10.12"},
        "model": {"model_path": "models/weed_rgb.pt", "model_exists": True,
                  "detection_mode": "weed", "confidence_threshold": 0.45,
                  "iou_threshold": 0.45},
        "model_classes": {"available": True, "class_count": 2,
                          "classes": {0: "sugarbeet", 1: "kochia"},
                          "stub_mode": False},
        "cameras": {"left": {"available": True, "exposure": 300,
                             "gamma": 214, "wb_temp": 5000, "auto_wb": 1,
                             "auto_exposure": 3, "autofocus": 1}},
    }

    rep = build_session_report("gui_test_session", events, prov, meta,
                               started_at=now - 5, ended_at=now + 20)

    assert rep["statistics"]["total_events"] == 2
    assert rep["statistics"]["events_by_zone"] == {"ZoneA": 1, "ZoneC": 1}
    assert rep["statistics"]["events_by_nozzle"] == {"N1": 1, "N3": 1}
    print("✓ Event rows and per-zone/per-nozzle aggregation correct "
          "(ZoneC correctly maps to N3)")

    assert rep["statistics"]["gps_coverage"]["coverage_pct"] == 50.0
    print("✓ GPS coverage computed correctly (1 of 2 events had a fix)")

    cs = rep["statistics"]["confidence_stats"]
    assert cs["max"] == 0.91 and cs["min"] == 0.76
    print(f"✓ Confidence stats: mean={cs['mean']} range={cs['min']}–{cs['max']}")

    assert rep["metadata"]["operator"] == "Nana Mensah"
    print("✓ Operator metadata carried into the report (not the old "
          "hardcoded default)")

    assert rep["warnings"] == [], f"unexpected warnings: {rep['warnings']}"
    print("✓ No spurious warnings on a complete, healthy session")

    # Warnings actually fire when data is missing/degraded
    bad = build_session_report("bad", [], {
        "model": {"model_exists": False},
        "model_classes": {"stub_mode": True},
        "cameras": {"right": {"available": False, "reason": "no device"}},
        "software": {"git_dirty": True},
    }, {})
    joined = " ".join(bad["warnings"])
    for expect in ("operator", "stub", "camera", "uncommitted", "No spray"):
        assert expect.lower() in joined.lower(), f"missing warning: {expect}"
    print(f"✓ All {len(bad['warnings'])} degradation warnings fire correctly "
          f"(missing operator, stub mode, camera, dirty tree, no events)")

    out = Path(tempfile.mkdtemp()) / "report.json"
    assert write_session_report(out, rep)
    reloaded = json.loads(out.read_text())
    assert reloaded["statistics"]["total_events"] == 2
    print("✓ Report writes to JSON and reloads correctly")

    summary = format_console_summary(rep)
    for expect in ("Nana Mensah", "ZoneA", "weed_rgb.pt", "abc1234",
                   "wilkin_plot_A"):
        assert expect in summary, f"summary missing {expect!r}"
    print("✓ Console summary includes operator, zones, model, git commit "
          "and field metadata")

    print()
    print(summary)
    print()
    print("=" * 55)
    print("core/gui_session_report.py ✓ ALL TESTS PASSED")
    print("=" * 55)
