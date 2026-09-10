"""
gui/overlay_rendering.py
ABEN Dual RGB Imaging System — Shared detection overlay rendering

Extracted from DetectionPanelRGB._draw_overlay() so the live Detection
tab and the offline video/image review tab draw zone boundaries and
detection boxes identically from ONE implementation, rather than a
live version and a second copy built for offline review that could
drift apart in appearance over time.

The original method only ever read self._cfg and nothing else from
self, so this is a straightforward extraction to a pure function of
(img, dual_result, spray_states, cfg) -- DetectionPanelRGB._draw_overlay()
is now a thin wrapper calling this.
"""

import cv2

from gui.frame_text import put_text, text_size


def draw_detection_overlay(img, dual_result, spray_states, cfg,
                           font_size=13, box_thick=1):
    """
    Draw zone boundaries and detection boxes on a side-by-side display
    image (DualCameraPanel._build_display()'s Side-by-Side output).

    cfg: an RGBConfig (or None -- returns img unchanged if so, matching
    the original method's behavior before a model/config is loaded).
    spray_states: [bool, bool, bool] for N1/N2/N3 -- pass [False, False,
    False] for offline review, where no nozzle is actually spraying;
    zone boundaries then always render in their normal (non-highlighted)
    style, which is the correct behavior for reviewing recorded footage.

    font_size/box_thick: override the defaults tuned for the live
    Detection tab's display context. Offline review's combined
    side-by-side frame is typically shown at a different scale than
    live, so it passes larger values here rather than this function
    silently looking too small there -- defaults are unchanged so
    every existing call (live Detection tab) renders identically to
    before.

    Style constants — all in one place for easy tuning:
      FONT_SIZE   : 13   — Noto Sans (via gui/frame_text.py), matches
                    the rest of the app's UI font instead of
                    OpenCV's blocky Hershey vector font
      BOX_THICK   : 1     — thin detection boxes
      ZONE_THICK  : 1     — thin zone boundary lines (2 when active)
      DASH        : 6/12  — short dashes on nozzle centerline
    """
    if cfg is None:
        return img

    # ── Style constants ───────────────────────────────────
    FONT_SIZE   = font_size
    BOX_THICK   = box_thick
    ZONE_THICK  = 1                          # thin zone lines

    # Per-class colours
    CLASS_COLORS = {
        "sugarbeet": (100, 255, 100),   # green
        "weed":      (0,   200, 255),   # yellow-cyan
    }
    DEFAULT_DET_COLOR = (0, 220, 255)

    out    = img.copy()
    h, w   = out.shape[:2]
    # NOT w // 2. The incoming image is DualCameraPanel._build_display()'s
    # Side-by-Side output: hstack(left[half_w], divider[4px], right[half_w]),
    # so w == 2*half_w + 4 -- recovering half_w via (w-4)//2 exactly
    # matches how _build_display() computed it in the first place.
    half_w = (w - 4) // 2

    zones_cfg = cfg.zones
    scale     = half_w / 1920

    # ── Zone boundaries ───────────────────────────────────
    b1_x    = int(zones_cfg.B1_SPLIT_X * scale)
    n1_cx   = int(zones_cfg.n1_center_cam1 * scale)
    n2_cx_l = int(zones_cfg.n2_center_cam1 * scale)

    off = half_w + 4
    b2_x    = int(zones_cfg.B2_SPLIT_X * scale)
    n2_cx_r = int(zones_cfg.n2_center_cam2 * scale)
    n3_cx   = int(zones_cfg.n3_center_cam2 * scale)

    left_zones = [
        (0,    b1_x,  "A",  n1_cx,   0, (0,   200, 100)),
        (b1_x, half_w,"B1", n2_cx_l, 1, (255, 180,   0)),
    ]
    right_zones = [
        (off,        off + b2_x, "B2", off + n2_cx_r, 1, (255, 180, 0)),
        (off + b2_x, w,          "C",  off + n3_cx,   2, (0,  140, 255)),
    ]

    for dx1, dx2, zlbl, noz_cx, noz_idx, color in \
            left_zones + right_zones:
        active = spray_states[noz_idx] if noz_idx < 3 else False

        # Zone boundary — thin line, thicker + brighter when active
        thickness = ZONE_THICK + 1 if active else ZONE_THICK
        cv2.rectangle(out, (dx1, 0), (dx2, h), color, thickness)

        # Zone label — small, bottom of frame
        tw, th = text_size(zlbl, FONT_SIZE)
        lx = dx1 + max(4, (dx2 - dx1 - tw) // 2)
        put_text(out, zlbl, (lx, h - 6 - th),
                 font_size=FONT_SIZE, color_bgr=color)

        # Nozzle centerline — short fine dashes
        DASH, GAP = 6, 10
        dash_color = (230, 230, 230) if active else color
        for y_s in range(0, h, DASH + GAP):
            cv2.line(out, (noz_cx, y_s),
                     (noz_cx, min(y_s + DASH, h)),
                     dash_color, 1)

        # Active zone — subtle fill
        if active:
            ovl = out.copy()
            cv2.rectangle(ovl, (dx1, 0), (dx2, h), color, -1)
            cv2.addWeighted(ovl, 0.10, out, 0.90, 0, out)

    # ── Detection boxes ───────────────────────────────────
    def _draw_detection(det, x_offset=0):
        bx1 = x_offset + int(det.x1 * scale)
        by1 = int(det.y1 * h / 1080)
        bx2 = x_offset + int(det.x2 * scale)
        by2 = int(det.y2 * h / 1080)
        col = CLASS_COLORS.get(det.class_name, DEFAULT_DET_COLOR)

        # Thin bounding box
        cv2.rectangle(out, (bx1, by1), (bx2, by2), col, BOX_THICK)

        # Label: "classname conf" — small text, dark background pill
        label = f"{det.class_name} {det.confidence:.2f}"
        tw, th = text_size(label, FONT_SIZE)
        ty = max(by1 - th - 4, 2)
        put_text(out, label, (bx1 + 2, ty),
                 font_size=FONT_SIZE, color_bgr=col,
                 bg_color_bgr=(15, 15, 15), pad=2)

    for det in dual_result.left.detections:
        _draw_detection(det, x_offset=0)

    for det in dual_result.right.detections:
        _draw_detection(det, x_offset=off)

    return out
