"""
core/session_provenance.py
ABEN Dual RGB Imaging System — System provenance capture

Collects everything needed to reproduce and defend a session's
results in a publication: exact camera settings actually in use, the
model file and its class list, zone/nozzle geometry, and the software
versions the run happened on.

Motivation: the session report previously described spray events but
almost nothing about the SYSTEM that produced them. "We detected 14
weeds" is not reproducible; "we detected 14 weeds at 1920x1080,
exposure 300, auto-WB on, with weed_rgb.pt (6 classes, conf 0.45,
IoU 0.45) on ultralytics X / torch Y at commit abc1234" is.

Everything here is best-effort: any individual probe that fails
records an explicit "unavailable" reason rather than raising, because
a missing git binary or an unreadable model file must never prevent a
session from being recorded.
"""

import logging
import platform
import subprocess
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# v4l2 control -> the key name used in our settings dicts
_V4L2_CONTROLS = {
    "exposure_time_absolute":     "exposure",
    "brightness":                 "brightness",
    "contrast":                   "contrast",
    "saturation":                 "saturation",
    "gamma":                      "gamma",
    "gain":                       "gain",
    "sharpness":                  "sharpness",
    "white_balance_temperature":  "wb_temp",
    "focus_absolute":             "focus",
    "backlight_compensation":     "backlight",
    "power_line_frequency":       "freq",
    "white_balance_automatic":    "auto_wb",
    "auto_exposure":              "auto_exposure",
    "focus_automatic_continuous": "autofocus",
}


def _run(cmd, timeout=5):
    """Run a command, returning stdout or None. Never raises."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout)
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return None


def capture_camera_settings(device: str) -> dict:
    """
    Read the camera's ACTUAL current v4l2 values -- not what a preset
    says they should be. These can differ (a preset may have failed to
    apply, or the operator may have tuned individual controls since),
    and for a publication record what matters is what the camera was
    genuinely running with.
    """
    out = _run(["v4l2-ctl", "-d", device, "--list-ctrls"])
    if out is None:
        return {"available": False,
                "reason": "v4l2-ctl not available or device unreadable",
                "device": device}

    settings = {"available": True, "device": device}
    for line in out.splitlines():
        line = line.strip()
        for ctrl, key in _V4L2_CONTROLS.items():
            if line.startswith(ctrl):
                for part in line.split():
                    if part.startswith("value="):
                        try:
                            settings[key] = int(part.split("=")[1])
                        except ValueError:
                            pass
                        break
    return settings


def capture_model_info(cfg) -> dict:
    """Model file, size, mtime, class list and inference thresholds."""
    info = {}
    try:
        m = cfg.model
        mode = getattr(cfg.session, "detection_mode", None)
        mode_val = getattr(mode, "value", str(mode))
        path = (Path(m.cls_rgb_pt) if mode_val == "cls"
                else Path(m.weed_rgb_pt))

        info["detection_mode"]        = mode_val
        info["model_path"]            = str(path)
        info["confidence_threshold"]  = m.confidence_threshold
        info["iou_threshold"]         = m.iou_threshold
        info["imgsz"]                 = getattr(m, "imgsz", None)
        info["device"]                = getattr(m, "device", None)

        if path.exists():
            st = path.stat()
            info["model_exists"]     = True
            info["model_size_mb"]     = round(st.st_size / (1024 * 1024), 2)
            info["model_modified"]    = datetime.fromtimestamp(
                st.st_mtime).isoformat(timespec="seconds")
        else:
            info["model_exists"] = False
    except Exception as e:
        info["error"] = f"could not read model config: {e}"
    return info


def capture_model_classes(engine) -> dict:
    """
    Class names as reported by the LOADED model, which is the
    authoritative list -- config defaults can drift from what a
    checkpoint was actually trained on.
    """
    try:
        if engine is None:
            return {"available": False, "reason": "no engine loaded"}
        names = getattr(engine, "class_names", None)
        if not names:
            return {"available": False, "reason": "engine exposes no class_names"}
        return {
            "available": True,
            "class_count": len(names),
            "classes": {int(k): str(v) for k, v in names.items()},
            "stub_mode": bool(getattr(engine, "stub_mode", False)),
        }
    except Exception as e:
        return {"available": False, "reason": str(e)}


def capture_geometry(cfg) -> dict:
    """Zone boundaries, nozzle centres and spray geometry."""
    try:
        z, g = cfg.zones, cfg.geometry
        return {
            "b1_split_x":          z.B1_SPLIT_X,
            "b2_split_x":          z.B2_SPLIT_X,
            "n1_center_cam1":      z.n1_center_cam1,
            "n2_center_cam1":      z.n2_center_cam1,
            "n2_center_cam2":      z.n2_center_cam2,
            "n3_center_cam2":      z.n3_center_cam2,
            "detection_threshold": z.detection_threshold,
            "non_spray_classes":   list(z.non_spray_classes),
            "camera_height_m":     getattr(g, "camera_height_m", None),
            "gsd_m_per_px":        getattr(g, "gsd_m_per_px", None),
            "nozzle_y_px":         getattr(g, "nozzle_y_px", None),
            "min_spray_dist_m":    getattr(g, "min_spray_dist_m", None),
            "max_spray_dist_m":    getattr(g, "max_spray_dist_m", None),
        }
    except Exception as e:
        return {"error": f"could not read geometry: {e}"}


def capture_software_versions() -> dict:
    """
    Library versions and the exact git commit -- the difference
    between "we ran the spray system" and a result someone else can
    actually reproduce.
    """
    versions = {
        "python":   platform.python_version(),
        "platform": platform.platform(),
    }

    for mod in ("ultralytics", "torch", "cv2", "numpy", "PyQt5.QtCore"):
        try:
            if mod == "cv2":
                import cv2
                versions["opencv"] = cv2.__version__
            elif mod == "PyQt5.QtCore":
                from PyQt5.QtCore import QT_VERSION_STR
                versions["qt"] = QT_VERSION_STR
            else:
                m = __import__(mod)
                versions[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            versions[mod.split(".")[0]] = "not installed"

    try:
        import torch
        versions["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            versions["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        versions["cuda_available"] = "unknown"

    commit = _run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"])
    versions["git_commit"] = commit or "unavailable"
    status = _run(["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"])
    # A dirty tree means the running code doesn't exactly match the
    # commit -- worth recording rather than quietly implying it does.
    versions["git_dirty"] = bool(status) if status is not None else "unknown"

    return versions


def capture_full_provenance(cfg, engine=None,
                            left_device: str = None,
                            right_device: str = None) -> dict:
    """
    One call collecting everything. Safe to call at ARM time; each
    section degrades independently rather than failing as a whole.
    """
    prov = {
        "captured_at":       datetime.now().isoformat(timespec="seconds"),
        "software":          capture_software_versions(),
        "model":             capture_model_info(cfg),
        "model_classes":     capture_model_classes(engine),
        "geometry":          capture_geometry(cfg),
    }

    cams = {}
    if left_device:
        cams["left"] = capture_camera_settings(left_device)
    if right_device:
        cams["right"] = capture_camera_settings(right_device)
    prov["cameras"] = cams

    try:
        prov["session"] = {
            "operator":     cfg.session.operator,
            "researcher":   cfg.session.researcher,
            "institution":  cfg.session.institution,
            "field_id":     cfg.session.field_id,
            "location":     cfg.session.location,
            "crop":         cfg.session.crop,
            "growth_stage": cfg.session.growth_stage.value,
            "notes":        cfg.session.notes,
        }
    except Exception as e:
        prov["session"] = {"error": str(e)}

    return prov


if __name__ == "__main__":
    import sys, json
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    print("=" * 55)
    print("core/session_provenance.py — Self Test")
    print("=" * 55)

    from core.detection_config_rgb import get_weed_config
    cfg = get_weed_config()

    sw = capture_software_versions()
    assert "python" in sw and "git_commit" in sw
    print(f"\n✓ Software versions captured:")
    for k in ("python", "opencv", "numpy", "ultralytics", "torch",
              "cuda_available", "git_commit", "git_dirty"):
        if k in sw:
            print(f"    {k:<16} = {sw[k]}")

    mi = capture_model_info(cfg)
    assert "model_path" in mi and "confidence_threshold" in mi
    print(f"\n✓ Model info captured: mode={mi['detection_mode']}, "
          f"conf={mi['confidence_threshold']}, exists={mi.get('model_exists')}")

    geo = capture_geometry(cfg)
    assert geo.get("b2_split_x") == cfg.zones.B2_SPLIT_X
    assert "sugarbeet" in geo.get("non_spray_classes", [])
    print(f"✓ Geometry captured: B2_SPLIT_X={geo['b2_split_x']}, "
          f"non_spray={geo['non_spray_classes']}")

    cam = capture_camera_settings("/dev/nonexistent-camera")
    assert cam["available"] is False and "reason" in cam
    print(f"✓ Missing camera degrades gracefully: {cam['reason']}")

    mc = capture_model_classes(None)
    assert mc["available"] is False
    print(f"✓ Missing engine degrades gracefully: {mc['reason']}")

    full = capture_full_provenance(cfg, engine=None)
    for key in ("captured_at", "software", "model", "geometry",
                "cameras", "session"):
        assert key in full, f"missing {key}"
    json.dumps(full)   # must be serializable for the report/JSON export
    print(f"✓ Full provenance assembled and JSON-serializable "
          f"({len(json.dumps(full))} bytes)")

    print()
    print("=" * 55)
    print("core/session_provenance.py ✓ ALL TESTS PASSED")
    print("=" * 55)
