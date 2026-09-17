#!/usr/bin/env python3
"""
gui/overlay_rendering_triple.py
ABEN Triple RGB Imaging System — Shared detection overlay rendering (3 cameras)

Phase 4 of the triple-camera software redesign: the display/overlay
layer's core, testable logic (side-by-side stitching + zone-boundary
drawing). Built as a new module alongside the existing, still-live
gui/overlay_rendering.py (draw_detection_overlay(), for the 2-camera
A/B1/B2/C zone system) rather than modifying it -- zero import
dependency between the two, so the live 2-camera Detection tab is
completely unaffected.

Deliberately scoped to the stitching/drawing FUNCTIONS only, not the
Qt widget layer (a DualCameraPanel-equivalent for 3 cameras) -- that
part needs visual/real-hardware confirmation the operator would have
to do anyway, so it's a natural next step once this core logic is
reviewed, not bundled into the same change.

Key difference from the old A/B1/B2/C system: each camera now maps to
exactly ONE nozzle (see core/zone_manager_triple.py's module
docstring for the physical rationale), so there's no more shared zone
to draw across two camera panels -- each camera's own panel gets ONE
owned-range highlight instead of being split into two named sub-zones.

Nozzle centerline: drawn at each nozzle's MEASURED physical pixel
position (TripleZoneConfig.nozzle_centers_x(), from a direct tape-
measure reading of N1/N2/N3's physical position, not derived from the
owned-range boundaries -- a nozzle's physical position has no reason
to sit at the midpoint of its camera's owned zone). Same dash style
as the old 2-camera system's centerline for visual consistency.
"""

import cv2
import numpy as np

try:
    from gui.frame_text import put_text, text_size
except ImportError:
    from frame_text import put_text, text_size

ZONE_COLORS = [
    (0,   200, 100),   # Cam 1 / N1 — green
    (255, 180,   0),   # Cam 2 / N2 — blue-ish
    (0,   140, 255),   # Cam 3 / N3 — orange
]

CLASS_COLORS = {
    "sugarbeet": (100, 255, 100),   # green
    "weed":      (0,   200, 255),   # yellow-cyan
}
DEFAULT_DET_COLOR = (0, 220, 255)


def build_triple_side_by_side(frames, target_width=None):
    """
    Stitch 3 BGR frames side by side with a 4px divider between each,
    preserving each frame's own aspect ratio (no stretching) -- same
    intent as DualCameraPanel._build_display()'s Side-by-Side mode,
    generalized from 2 panels to 3.

    frames: list of 3 np.ndarray BGR frames (any matching resolution;
            typically all three cameras' native 1920x1080).
    target_width: total output width to aim for (e.g. a display
            label's current width). Each panel gets
            (target_width - 8) // 3 (subtracting 2x 4px dividers).
            Defaults to the frames' own combined native width if not
            given (no resizing at all -- full resolution).

    Returns the stitched BGR image and the per-panel width actually
    used (needed by draw_triple_detection_overlay() to convert zone
    pixel boundaries measured in each camera's OWN 1920px frame into
    positions within the possibly-resized combined image).
    """
    if len(frames) != 3:
        raise ValueError(f"build_triple_side_by_side requires exactly "
                         f"3 frames, got {len(frames)}")

    src_h, src_w = frames[0].shape[:2]
    aspect = src_w / src_h if src_h else 16 / 9

    if target_width is None:
        panel_w = src_w
    else:
        panel_w = max(1, (target_width - 8) // 3)
    panel_h = max(1, int(panel_w / aspect))

    resized = [cv2.resize(f, (panel_w, panel_h)) for f in frames]
    divider = np.zeros((panel_h, 4, 3), dtype=np.uint8)

    combined = cv2.hconcat([
        resized[0], divider, resized[1], divider, resized[2]
    ])
    return combined, panel_w


def draw_triple_detection_overlay(img, triple_result, spray_states,
                                  zone_cfg, panel_width,
                                  font_size=13, box_thick=1,
                                  show_zones=True):
    """
    Draw zone boundaries and detection boxes on a 3-way side-by-side
    display image (build_triple_side_by_side()'s output).

    triple_result: a TripleInferenceResult (core/detection_engine_triple.py)
        -- reads .results[i].detections for each camera i.
    spray_states: [bool, bool, bool] for N1/N2/N3.
    zone_cfg: a TripleZoneConfig (core/zone_manager_triple.py) -- reads
        .owned_ranges() for the per-camera boundary pixel positions.
    panel_width: the per-panel width build_triple_side_by_side()
        actually used (its second return value) -- needed to convert
        zone boundaries (measured in each camera's own 1920px frame)
        into positions within this possibly-resized combined image.
    font_size/box_thick/show_zones: same meaning as the 2-camera
        draw_detection_overlay() -- see that function's docstring.
    """
    if zone_cfg is None:
        return img

    FONT_SIZE  = font_size
    BOX_THICK  = box_thick
    ZONE_THICK = 1

    out  = img.copy()
    h, _ = out.shape[:2]
    scale = panel_width / zone_cfg.FRAME_WIDTH
    ranges = zone_cfg.owned_ranges()
    labels = zone_cfg.camera_labels

    panel_offsets = [0, panel_width + 4, 2 * (panel_width + 4)]

    for cam_idx in range(3):
        offset = panel_offsets[cam_idx]
        min_x, max_x = ranges[cam_idx]
        nozzle_id = zone_cfg.zone_nozzle_map[cam_idx]
        color = ZONE_COLORS[cam_idx % len(ZONE_COLORS)]
        active = spray_states[nozzle_id] if nozzle_id < len(spray_states) else False

        # Owned-range pixel positions within THIS panel, converted
        # from the camera's own 1920px frame via the same scale
        # _build_display()/build_triple_side_by_side() used to build
        # the panel in the first place.
        px_min = 0          if min_x is None else int(min_x * scale)
        px_max = panel_width if max_x is None else int(max_x * scale)

        if show_zones:
            thickness = ZONE_THICK + 1 if active else ZONE_THICK
            cv2.rectangle(
                out, (offset + px_min, 0), (offset + px_max, h),
                color, thickness)

            label = f"N{nozzle_id + 1}"
            tw, th = text_size(label, FONT_SIZE)
            lx = offset + px_min + max(4, (px_max - px_min - tw) // 2)
            put_text(out, label, (lx, h - 6 - th),
                    font_size=FONT_SIZE, color_bgr=color)

            # Nozzle centerline — short fine dashes at the MEASURED
            # physical nozzle position (TripleZoneConfig.nozzle_centers_x(),
            # not derived from the owned-range boundaries -- a nozzle's
            # physical position has no reason to sit at its zone's
            # midpoint). Same dash style as the old dual-camera
            # gui/overlay_rendering.py for visual consistency.
            noz_cx = offset + int(zone_cfg.nozzle_centers_x()[cam_idx] * scale)
            DASH, GAP = 6, 10
            dash_color = (230, 230, 230) if active else color
            for y_s in range(0, h, DASH + GAP):
                cv2.line(out, (noz_cx, y_s),
                        (noz_cx, min(y_s + DASH, h)),
                        dash_color, 1)

            if active:
                ovl = out.copy()
                cv2.rectangle(
                    ovl, (offset + px_min, 0), (offset + px_max, h),
                    color, -1)
                cv2.addWeighted(ovl, 0.10, out, 0.90, 0, out)

        # ── Detection boxes for this camera ──
        dets = (triple_result.results[cam_idx].detections
               if triple_result is not None else [])
        for det in dets:
            bx1 = offset + int(det.x1 * scale)
            by1 = int(det.y1 * h / zone_cfg.FRAME_HEIGHT)
            bx2 = offset + int(det.x2 * scale)
            by2 = int(det.y2 * h / zone_cfg.FRAME_HEIGHT)
            col = CLASS_COLORS.get(det.class_name, DEFAULT_DET_COLOR)

            cv2.rectangle(out, (bx1, by1), (bx2, by2), col, BOX_THICK)

            det_label = f"{det.class_name} {det.confidence:.2f}"
            tw, th = text_size(det_label, FONT_SIZE)
            ty = max(by1 - th - 4, 2)
            put_text(out, det_label, (bx1 + 2, ty),
                    font_size=FONT_SIZE, color_bgr=col,
                    bg_color_bgr=(15, 15, 15), pad=2)

    return out


# ─────────────────────────────────────────────────────────────
#  SELF TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys as _sys
    from pathlib import Path as _Path
    _ROOT = _Path(__file__).resolve().parent.parent
    if str(_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_ROOT))

    import time

    from core.zone_manager_triple import TripleZoneConfig
    from core.detection_engine_rgb import Detection, InferenceResult
    from core.detection_engine_triple import TripleInferenceResult

    print("=" * 60)
    print("gui/overlay_rendering_triple.py — self test")
    print("=" * 60)

    # ── Test 1: stitching preserves distinguishable camera order ──
    f1 = np.full((1080, 1920, 3), (255, 0, 0), dtype=np.uint8)
    f2 = np.full((1080, 1920, 3), (0, 255, 0), dtype=np.uint8)
    f3 = np.full((1080, 1920, 3), (0, 0, 255), dtype=np.uint8)
    combined, panel_w = build_triple_side_by_side([f1, f2, f3], target_width=1440)
    h, w = combined.shape[:2]
    assert w == panel_w * 3 + 8
    assert panel_w == (1440 - 8) // 3
    mid_row = h // 2
    assert tuple(combined[mid_row, panel_w // 2]) == (255, 0, 0)
    assert tuple(combined[mid_row, panel_w + 4 + panel_w // 2]) == (0, 255, 0)
    assert tuple(combined[mid_row, 2 * (panel_w + 4) + panel_w // 2]) == (0, 0, 255)
    print(f"✓ build_triple_side_by_side() correctly stitches 3 panels "
          f"in order (Cam1, Cam2, Cam3), sized {w}x{h} from a "
          f"target_width of 1440")

    # ── Test 2: no stretching -- aspect ratio preserved ──
    assert abs((panel_w / h) - (1920 / 1080)) < 0.01
    print(f"✓ Each panel preserves the 16:9 source aspect ratio "
          f"(no stretching) -- {panel_w}x{h}")

    # ── Test 3: overlay drawing renders zone boundaries + detections,
    # and the default (no override) still renders SOMETHING sensible ──
    zone_cfg = TripleZoneConfig()

    def det(cls_name, cx):
        return Detection(class_id=0, class_name=cls_name, confidence=0.9,
                         x1=cx - 20, y1=500, x2=cx + 20, y2=540, camera="test")

    results = [
        InferenceResult(detections=[det("kochia", 500)], inference_ms=5,
                        preprocess_ms=1, total_ms=6, frame_shape=(1080,1920),
                        camera="cam1"),
        InferenceResult(detections=[], inference_ms=5, preprocess_ms=1,
                        total_ms=6, frame_shape=(1080,1920), camera="cam2"),
        InferenceResult(detections=[], inference_ms=5, preprocess_ms=1,
                        total_ms=6, frame_shape=(1080,1920), camera="cam3"),
    ]
    triple_result = TripleInferenceResult(results=results, frame_id=0,
                                          timestamp=time.time())

    dark_bg = np.full((1080, 1920, 3), (40, 60, 30), dtype=np.uint8)
    combined2, panel_w2 = build_triple_side_by_side(
        [dark_bg, dark_bg, dark_bg], target_width=1440)

    overlay = draw_triple_detection_overlay(
        combined2, triple_result, [False, False, False],
        zone_cfg, panel_w2)

    def count_drawn(im, bg=(40, 60, 30)):
        return int(np.sum(np.any(im != np.array(bg), axis=-1)))

    ink_with_zones = count_drawn(overlay)
    assert ink_with_zones > 0
    print(f"✓ draw_triple_detection_overlay() draws zone boundaries + "
          f"a detection box ({ink_with_zones} pixels of ink)")

    overlay_no_zones = draw_triple_detection_overlay(
        combined2, triple_result, [False, False, False],
        zone_cfg, panel_w2, show_zones=False)
    ink_no_zones = count_drawn(overlay_no_zones)
    assert 0 < ink_no_zones < ink_with_zones
    print(f"✓ show_zones=False draws LESS ink ({ink_with_zones} -> "
          f"{ink_no_zones}) but still shows the detection box "
          f"(not a blank frame)")

    # ── Test 4: None zone_cfg returns the image unchanged, matching
    # the 2-camera function's behavior before a config is loaded ──
    unchanged = draw_triple_detection_overlay(
        combined2, triple_result, [False, False, False], None, panel_w2)
    assert np.array_equal(unchanged, combined2)
    print("✓ zone_cfg=None returns the image completely unchanged "
          "(matches draw_detection_overlay()'s behavior before a "
          "config is loaded)")

    # ── Test 5: active zone highlight visibly differs from inactive ──
    overlay_active = draw_triple_detection_overlay(
        combined2, triple_result, [True, False, False],
        zone_cfg, panel_w2)
    assert not np.array_equal(overlay, overlay_active)
    print("✓ An active spray zone (N1 firing) renders visibly "
          "differently from the idle state (highlight fill + "
          "thicker boundary)")

    # ── Test 6: exactly 3 frames required ──
    try:
        build_triple_side_by_side([f1, f2])
        assert False, "should have raised"
    except ValueError as e:
        print(f"✓ Correctly rejects a non-3-frame list: {e}")

    # ── Test 7: the nozzle centerline is drawn at the exact measured
    # pixel position (not just "some extra ink somewhere") ──
    scale_check = panel_w2 / zone_cfg.FRAME_WIDTH
    expected_cam2_noz_x = int(zone_cfg.nozzle_centers_x()[1] * scale_check)
    cam2_offset = panel_w2 + 4
    col_x = cam2_offset + expected_cam2_noz_x
    # Sample down the expected centerline column -- at least one dash
    # segment should be non-background along it (dashes have gaps, so
    # check a run of rows rather than one exact pixel).
    bg = np.array((40, 60, 30))
    column_has_dash = any(
        not np.array_equal(overlay[y, col_x], bg)
        for y in range(0, overlay.shape[0], 2)
    )
    assert column_has_dash, (
        f"expected a dashed centerline at column {col_x} "
        f"(Cam2's measured nozzle position), found none")
    print(f"✓ Nozzle centerline is drawn at the exact measured pixel "
          f"position for Cam 2 (column {col_x}, from its measured "
          f"39in physical nozzle center) -- not just present "
          f"somewhere, but at the specific correct location")

    print()
    print("=" * 60)
    print("gui/overlay_rendering_triple.py ✓  ALL TESTS PASSED")
    print("=" * 60)
