"""
gui/frame_text.py
Field Imaging System — Theme-consistent text rendering for OpenCV overlays

cv2.putText only supports OpenCV's built-in Hershey vector fonts, which
render blocky/aliased at the small scales used for the HUD/zone/
detection labels burned onto the live camera feed — visually
inconsistent with the rest of the app's Noto Sans UI (see
gui/theme_manager.py and the font-family choice documented there).
This module renders that overlay text with PIL using an actual
TrueType font instead, so it matches.

Performance: only the small patch of frame pixels the text actually
occupies is converted to/from PIL, not the whole frame, and the font
itself is loaded from disk once and cached — keeps this cheap enough
for a real-time video loop (benchmarked at ~0.1-0.3ms per short label
on a typical HUD-sized patch; see the self-test at the bottom of this
file).

Falls back to cv2.putText automatically if PIL or a usable font file
can't be found, so a missing font on some future machine never breaks
the camera feed — it just looks like it used to.
"""

import functools
import glob
import os
import subprocess

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


def _find_font_file() -> str:
    """
    Resolve an actual installed font file for 'Noto Sans' (or the
    system's best substitute), matching what the rest of the app's Qt
    stylesheets request (font-family:'Noto Sans',Arial,sans-serif).

    Uses fontconfig's own matching logic (fc-match) rather than a
    hardcoded path list as the primary strategy, since exact install
    locations vary by distro/image — fc-match uses the same
    substitution rules the OS itself uses, so it finds a sensible
    real font even if literal "Noto Sans" isn't installed. Falls back
    to a short list of common paths, then to matplotlib's always-
    bundled DejaVu Sans as a last resort (matplotlib is already a
    dependency here via the YOLO comparison tooling), before giving
    up and letting the caller fall back to cv2's Hershey font.
    """
    try:
        result = subprocess.run(
            ["fc-match", "Noto Sans", "-f", "%{file}"],
            capture_output=True, text=True, timeout=2,
        )
        path = result.stdout.strip()
        if path and os.path.exists(path):
            return path
    except Exception:
        pass

    for pattern in [
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if glob.glob(pattern):
            return pattern

    try:
        import matplotlib
        mpl_font = os.path.join(
            os.path.dirname(matplotlib.__file__),
            "mpl-data", "fonts", "ttf", "DejaVuSans.ttf")
        if os.path.exists(mpl_font):
            return mpl_font
    except Exception:
        pass

    return ""


@functools.lru_cache(maxsize=32)
def _load_font(size: int):
    """Cached font load — the TTF is only read from disk once per
    distinct size, never per frame/per call."""
    if not _PIL_AVAILABLE:
        return None
    path = _find_font_file()
    if not path:
        return None
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return None


def text_size(text: str, font_size: int = 14):
    """Returns (width, height) in pixels for `text` at `font_size`,
    measured with the same font put_text() will actually draw with
    (or the Hershey fallback estimate if no TrueType font is available,
    so callers can size backing rectangles consistently either way)."""
    font = _load_font(font_size)
    if font is None:
        (w, h), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_DUPLEX, font_size / 30.0, 1)
        return (w, h)
    if hasattr(font, "getbbox"):
        l, t, r, b = font.getbbox(text)
        return (r - l, b - t)
    return font.getsize(text)  # older Pillow versions


def put_text(img_bgr: np.ndarray, text: str, org, font_size: int = 14,
             color_bgr=(255, 255, 255), bg_color_bgr=None, pad: int = 2):
    """
    Draw `text` onto img_bgr (a BGR numpy array — modified in place
    and also returned for convenient chaining) with `org=(x, y)` as
    the TOP-LEFT corner of the text.

    NOTE the origin convention difference from cv2.putText: that
    function's `org` is the BOTTOM-LEFT of the text baseline. Callers
    replacing a cv2.putText call need to adjust their y coordinate
    accordingly (typically: drop the "+ text_height" baseline offset
    they were adding before).

    Renders with an actual Noto Sans TrueType font via PIL for visual
    consistency with the rest of the app. Falls back to cv2.putText's
    Hershey font automatically if PIL or a usable font file isn't
    available, so this never raises just because a font is missing —
    worst case, overlay text looks like it did before this module
    existed.
    """
    x, y = int(org[0]), int(org[1])
    font = _load_font(font_size)

    if font is None:
        cv2.putText(img_bgr, text, (x, y + font_size),
                    cv2.FONT_HERSHEY_DUPLEX, font_size / 30.0,
                    color_bgr, 1, cv2.LINE_AA)
        return img_bgr

    tw, th = text_size(text, font_size)
    h_img, w_img = img_bgr.shape[:2]

    px0 = max(0, x - pad)
    py0 = max(0, y - pad)
    px1 = min(w_img, x + tw + pad * 2)
    py1 = min(h_img, y + th + pad * 2)
    if px1 <= px0 or py1 <= py0:
        return img_bgr

    patch = img_bgr[py0:py1, px0:px1]
    patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(patch_rgb)
    draw = ImageDraw.Draw(pil_img)

    rel_x = x - px0
    rel_y = y - py0

    if bg_color_bgr is not None:
        bg_rgb = (bg_color_bgr[2], bg_color_bgr[1], bg_color_bgr[0])
        draw.rectangle([0, 0, patch.shape[1], patch.shape[0]], fill=bg_rgb)

    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text((rel_x, rel_y), text, font=font, fill=color_rgb)

    patch_out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img_bgr[py0:py1, px0:px1] = patch_out
    return img_bgr


if __name__ == "__main__":
    import time

    print("=" * 55)
    print("gui/frame_text.py — Self Test")
    print("=" * 55)

    font_path = _find_font_file()
    print(f"\nResolved font: {font_path or '(none found -- Hershey fallback active)'}")
    print(f"PIL available: {_PIL_AVAILABLE}")

    # ── Correctness: draw, confirm pixels actually changed ──
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    before = frame.copy()
    put_text(frame, "N1", (50, 50), font_size=16, color_bgr=(0, 255, 255))
    assert not np.array_equal(frame, before), "put_text should modify the image"
    print("✓ put_text() actually draws pixels")

    # Text near the frame edge shouldn't crash (clipping test)
    put_text(frame, "EDGE", (395, 195), font_size=16, color_bgr=(255, 0, 0))
    put_text(frame, "NEG", (-5, -5), font_size=16, color_bgr=(255, 0, 0))
    print("✓ edge/out-of-bounds positions handled without error")

    # Background pill
    frame2 = np.zeros((100, 200, 3), dtype=np.uint8)
    put_text(frame2, "kochia 0.85", (10, 10), font_size=14,
              color_bgr=(255, 255, 255), bg_color_bgr=(20, 20, 20))
    print("✓ background-pill rendering works")

    tw, th = text_size("kochia 0.85", font_size=14)
    assert tw > 0 and th > 0
    print(f"✓ text_size() returns sane dimensions: {tw}x{th}px")

    # ── Performance: this runs in a real-time video loop ──
    labels = ["N1", "N2", "N3", "A", "B1", "B2", "C", "LEFT", "RIGHT",
              "kochia 0.85", "waterhemp 0.72", "WEED FPS:12 Det:2 Ev:5"]
    frame3 = np.zeros((1080, 1920, 3), dtype=np.uint8)

    n_iters = 100
    t0 = time.perf_counter()
    for _ in range(n_iters):
        for i, lbl in enumerate(labels):
            put_text(frame3, lbl, (50 + i * 10, 50 + i * 20),
                      font_size=13, color_bgr=(255, 255, 255))
    elapsed = time.perf_counter() - t0
    per_frame_ms = (elapsed / n_iters) * 1000
    per_label_ms = (elapsed / (n_iters * len(labels))) * 1000

    print(f"\nPerformance ({len(labels)} labels/frame, {n_iters} frames):")
    print(f"  {per_frame_ms:.2f} ms/frame total for all {len(labels)} labels")
    print(f"  {per_label_ms:.3f} ms/label")
    print(f"  (for reference: a 30fps camera has a ~33ms/frame budget; "
          f"a 10fps detection loop has ~100ms)")

    if per_frame_ms > 5.0:
        print(f"  ⚠ this is higher than expected -- worth profiling on "
              f"the Jetson's actual CPU before relying on it")
    else:
        print(f"  ✓ comfortably cheap relative to any realistic frame budget")

    print()
    print("=" * 55)
    print("gui/frame_text.py ✓ ALL TESTS PASSED")
    print("=" * 55)
