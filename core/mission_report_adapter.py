"""
core/mission_report_adapter.py
ABEN Dual RGB Imaging System — Mission report → canonical schema

Converts the JSON written by spray_mission_rgb.py's SessionReportRGB
(system_config + summary.robot/camera/detection/spray + spray_events)
into the SAME canonical schema core/gui_session_report.py produces
(metadata/provenance/statistics/events/warnings), so
generate_gui_session_report.js can render EITHER source into an
identical document structure -- the operator asked for one unified
report regardless of whether a session came from ARM DETECTION or
START MISSION, and this is what makes that true without either source
needing to know about the other's internal shape.

Mission-specific data that has no GUI-session equivalent (robot
kinematics, camera grab success/sync-error stats, near-miss counts)
is folded into statistics["mission_extras"] -- present only when the
source is a mission report, so the unified generator can render an
extra section for it without GUI reports needing a placeholder.
"""

from datetime import datetime


def _mission_events_to_rows(spray_events: list) -> list:
    """
    Mission SprayEventRecord dicts -> the same row shape
    core/gui_session_report.py's _event_rows() produces, so the
    unified generator's Spray Event Log table works identically
    regardless of source.
    """
    rows = []
    for e in spray_events:
        classes = e.get("confirming_classes") or []
        confs   = e.get("confirming_confidences") or []
        ts = e.get("fire_time") or e.get("trigger_time") or 0
        rows.append({
            "timestamp":  ts,
            "time": (datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                     if ts else "—"),
            "zone_name":  e.get("zone_name", "—"),
            "nozzle":     f"N{e.get('nozzle', -1) + 1}",
            "nozzle_id":  e.get("nozzle"),
            "mode":       "mission",
            "classes":    classes,
            "top_class":  classes[0] if classes else "",
            "confidence": round(max(confs), 3) if confs else 0.0,
            "spray_duration_s": e.get("spray_time_s"),
            "pose_x":     e.get("trigger_x"),
            "pose_y":     e.get("trigger_y"),
            "gps_lat":    None,
            "gps_lon":    None,
            "gps_valid":  False,
            "flagged_cls": False,
            # Mission-specific extras not present on GUI events, kept
            # so the "missed" concept (confirmed but robot never
            # reached firing distance) isn't silently dropped.
            "fired":  e.get("fire_time") is not None,
        })
    return rows


def convert_mission_report(mission_json: dict) -> dict:
    """
    mission_json: the parsed JSON from SessionReportRGB.save_json()
    (top-level keys: system_config, summary, spray_events,
    subthreshold_events, detection_transitions, trajectory).

    Returns the canonical schema build_session_report() produces, so
    both can be handed to the same JS generator.
    """
    cfg = mission_json.get("system_config", {}) or {}
    sm  = mission_json.get("summary", {}) or {}
    events = _mission_events_to_rows(mission_json.get("spray_events", []))

    fired_events = [e for e in events if e["fired"]]

    by_zone, by_class, by_nozzle = {}, {}, {}
    for e in fired_events:
        by_zone[e["zone_name"]] = by_zone.get(e["zone_name"], 0) + 1
        by_nozzle[e["nozzle"]]  = by_nozzle.get(e["nozzle"], 0) + 1
        for c in e["classes"]:
            by_class[c] = by_class.get(c, 0) + 1

    confs = [e["confidence"] for e in fired_events if e["confidence"] > 0]
    conf_stats = {}
    if confs:
        conf_stats = {
            "mean": round(sum(confs) / len(confs), 3),
            "min":  round(min(confs), 3),
            "max":  round(max(confs), 3),
        }

    robot = sm.get("robot", {}) or {}
    statistics = {
        "total_events":        len(fired_events),
        "duration_s":          robot.get("duration_s", 0.0),
        "events_per_minute":   round(
            len(fired_events) / max((robot.get("duration_s") or 0.01) / 60, 0.01), 2),
        "events_by_zone":      by_zone,
        "events_by_nozzle":    by_nozzle,
        "detections_by_class": by_class,
        "confidence_stats":    conf_stats,
        "cls_flagged":         0,
        "gps_coverage": {"events_with_fix": 0,
                         "coverage_pct": 0.0},
        # Mission-only data with no GUI-session equivalent -- rendered
        # as an extra section by the unified generator when present.
        "mission_extras": {
            "robot":     robot,
            "camera":    sm.get("camera", {}),
            "detection": sm.get("detection", {}),
            "spray":     sm.get("spray", {}),
        },
    }

    cams = {}
    if cfg.get("left_camera_settings"):
        cams["left"] = cfg["left_camera_settings"]
    if cfg.get("right_camera_settings"):
        cams["right"] = cfg["right_camera_settings"]

    mcls = cfg.get("model_classes") or {}
    model_classes = {
        "available":   bool(mcls),
        "class_count": len(mcls),
        "classes":     mcls,
        "stub_mode":   cfg.get("detection_mode") == "DUMMY",
    }

    report = {
        "report_type":    "aben_rgb_mission_session",
        "schema_version": 1,
        "generated_at":   datetime.now().isoformat(timespec="seconds"),
        "session_id":     cfg.get("session_start_iso", "mission_session"),
        "source":         "START MISSION (spray_mission_rgb.py)",
        "metadata": {
            "operator":     cfg.get("operator", ""),
            "researcher":   cfg.get("researcher", ""),
            "institution":  cfg.get("institution", ""),
            "field_id":     cfg.get("field_id", ""),
            "location":     "",
            "crop":         cfg.get("crop", ""),
            "growth_stage": cfg.get("growth_stage", ""),
            "notes":        cfg.get("notes", ""),
        },
        "provenance": {
            "software": cfg.get("software_versions", {}) or {},
            "model": {
                "model_path":           cfg.get("model_path"),
                "detection_mode":       cfg.get("detection_mode"),
                "confidence_threshold": cfg.get("confidence_threshold"),
                "iou_threshold":        cfg.get("iou_threshold"),
                "imgsz":                cfg.get("imgsz"),
                "device":               cfg.get("device"),
                "model_exists":         None,   # not tracked by mission reports
            },
            "model_classes": model_classes,
            "geometry": {
                "b1_split_x":          cfg.get("b1_split_x"),
                "b2_split_x":          cfg.get("b2_split_x"),
                "n1_center_cam1":      cfg.get("n1_center_px"),
                "n2_center_cam1":      cfg.get("n2_center_cam1_px"),
                "n2_center_cam2":      cfg.get("n2_center_cam2_px"),
                "n3_center_cam2":      cfg.get("n3_center_px"),
                "detection_threshold": cfg.get("zone_threshold"),
                "non_spray_classes":   [],   # not tracked by mission reports
                "camera_height_m":     cfg.get("camera_height_m"),
                "gsd_m_per_px":        (cfg.get("gsd_mm_per_px") or 0) / 1000
                                        if cfg.get("gsd_mm_per_px") else None,
                "nozzle_y_px":         cfg.get("nozzle_y_px"),
                "min_spray_dist_m":    cfg.get("min_spray_dist_m"),
                "max_spray_dist_m":    cfg.get("max_spray_dist_m"),
            },
            "cameras": cams,
        },
        "statistics": statistics,
        "events": events,
        "session_start_iso": cfg.get("session_start_iso"),
    }

    warnings = []
    if not cfg.get("operator"):
        warnings.append("No operator name recorded for this session.")
    if model_classes["stub_mode"]:
        warnings.append(
            "Detection engine was in DUMMY MODE — spray events are not "
            "from real inference.")
    if not cams:
        warnings.append("Camera settings unavailable for this session.")
    if cfg.get("git_dirty") is True:
        warnings.append(
            "Working tree had uncommitted changes — the running code does "
            "not exactly match the recorded git commit.")
    if statistics["total_events"] == 0:
        warnings.append("No spray events were recorded this session.")
    missed = sm.get("spray", {}).get("missed", 0)
    if missed:
        warnings.append(
            f"{missed} confirmed detection(s) did not result in a completed "
            f"spray — the run ended before the robot reached firing distance.")
    report["warnings"] = warnings

    return report


if __name__ == "__main__":
    import sys, json, tempfile
    from pathlib import Path
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    print("=" * 55)
    print("core/mission_report_adapter.py — Self Test")
    print("=" * 55)

    from session_report_rgb import SessionReportRGB, SystemConfigSnapshot

    cfg = SystemConfigSnapshot(
        camera_model="eMeet C960 4K (Dual RGB)", model_path="models/weed_rgb.pt",
        confidence_threshold=0.45, iou_threshold=0.45, device="cuda:0",
        detection_mode="LIVE", zone_threshold=4, drive_speed_mps=0.3,
        target_distance_m=2.0, field_id="test_field", researcher="Nana Mensah",
        operator="Nana Mensah", institution="NDSU", crop="sugarbeet",
        growth_stage="6_leaf", notes="adapter test",
        left_camera_settings={"available": True, "exposure": 300, "auto_wb": 1},
        model_classes={0: "sugarbeet", 1: "kochia"},
        git_commit="abc1234", git_dirty=False,
        software_versions={"python": "3.10.12", "git_commit": "abc1234",
                           "git_dirty": False},
    )
    rep = SessionReportRGB(cfg)
    rep.record_camera_grab(success=True, sync_error_ms=10.0)
    rep.record_frame(frame_index=0, preprocess_ms=2, inference_ms=90,
                     total_ms=95, detections=[], robot_x=0, robot_y=0,
                     robot_speed=0.3, distance_traveled=1.0)
    ev = rep.record_spray_trigger(
        nozzle=2, zone_name="ZoneC", x=0.5, y=0.0,
        confirming_classes=["kochia"], confirming_confidences=[0.85])
    rep.record_spray_fire(ev, x=0.6, y=0.0, distance_m=0.1)
    rep.finalize(distance_traveled=2.0, target_distance=2.0,
                abort_reason="target_reached")

    mission_json = rep.to_dict()
    tmp = Path(tempfile.mkdtemp()) / "mission.json"
    tmp.write_text(json.dumps(mission_json, default=str))

    canonical = convert_mission_report(json.loads(tmp.read_text()))

    assert canonical["metadata"]["operator"] == "Nana Mensah"
    print(f"✓ Operator carried through: {canonical['metadata']['operator']}")

    assert canonical["provenance"]["model_classes"]["classes"] == \
        {"0": "sugarbeet", "1": "kochia"} or \
        canonical["provenance"]["model_classes"]["classes"] == \
        {0: "sugarbeet", 1: "kochia"}
    print(f"✓ Model classes carried through: "
          f"{canonical['provenance']['model_classes']['classes']}")

    assert "mission_extras" in canonical["statistics"]
    assert "robot" in canonical["statistics"]["mission_extras"]
    print(f"✓ mission_extras present with robot/camera/detection/spray data")

    assert canonical["statistics"]["total_events"] == 1
    assert canonical["statistics"]["events_by_zone"] == {"ZoneC": 1}
    assert canonical["statistics"]["events_by_nozzle"] == {"N3": 1}
    assert canonical["events"][0]["fired"] is True
    print(f"✓ Fired spray event correctly counted, with ZoneC correctly "
          f"mapping to N3 (nozzle=2 -> N3)")

    assert canonical["provenance"]["geometry"]["detection_threshold"] == 4
    print(f"✓ Geometry fields correctly remapped "
          f"(detection_threshold={canonical['provenance']['geometry']['detection_threshold']})")

    json.dumps(canonical, default=str)   # must be serializable
    print("✓ Canonical output is JSON-serializable")

    # Missing/degraded case
    bare = convert_mission_report({"system_config": {}, "summary": {},
                                   "spray_events": []})
    assert len(bare["warnings"]) >= 3
    print(f"✓ Degraded/empty mission JSON produces {len(bare['warnings'])} "
          f"warnings without crashing: {bare['warnings']}")

    print()
    print("=" * 55)
    print("core/mission_report_adapter.py ✓ ALL TESTS PASSED")
    print("=" * 55)
