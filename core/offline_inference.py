"""
core/offline_inference.py
ABEN Dual RGB Imaging System — Offline video/image inference

Runs the SAME RGBDetectionEngine used live against recorded field
data (paired Left/Right videos, or paired Left/Right image folders),
for qualitative review -- watch the model's real detections on real
field footage frame by frame -- and for accumulating detection
statistics (counts, confidence distribution, per-class breakdown).

Deliberately NOT an accuracy-evaluation tool: this has no ground
truth to compare against, so nothing here computes mAP/precision/
recall. That needs labeled data and belongs to the existing
evaluate_model_rgb.py / compare_yolo_models_rgb.py scripts (see
core/offline_evaluation_launcher.py for the GUI wrapper around those).
What this module produces is honestly a detection SUMMARY, not an
evaluation -- the two are kept conceptually separate throughout so
the GUI never implies a metric it can't actually support.
"""

import logging
import statistics
from pathlib import Path

import cv2
import numpy as np

# core.X / gui.X imports below are deliberately NOT at module level --
# running this file directly (`python3 core/offline_inference.py`,
# for the self-test) puts only core/ on sys.path, not the project
# root, so `from core.dual_emeet_camera import ...` would fail before
# the self-test's own sys.path fix ever runs (that fix lives in
# __main__, which executes AFTER top-level imports). Same reasoning
# as core/gui_session_report.py and core/mission_report_adapter.py,
# neither of which import other core/gui submodules at module level
# either. Imported lazily instead, right where each is actually used.


class OfflineDualVideoSource:
    """
    Opens a pair of recorded Left/Right video files and yields
    synchronized frame pairs. Frame index N in the left file
    corresponds to frame index N in the right file by construction --
    both streams were written together, one frame per loop iteration,
    by the live dual-stream recorder (see
    gui/panels/dual_camera_panel.py's video recording and
    realsense_camera.py's _SimpleDualStreamRecorder for the same
    pattern) -- so no cross-file timestamp matching is needed or
    possible here, unlike the live camera's sync-error tracking.
    """

    def __init__(self, left_path, right_path):
        self.left_path  = Path(left_path)
        self.right_path = Path(right_path)

        # Validate both paths exist BEFORE ever calling cv2.VideoCapture
        # -- an empty or missing path fed straight to VideoCapture can
        # raise a cryptic OpenCV assertion from deep inside its
        # image-sequence fallback backend (cap_images.cpp) rather than
        # a clear error, which is confusing to debug from the outside.
        # Catching it here up front means the operator sees "file does
        # not exist" instead of a raw C++ assertion trace.
        if not self.left_path.exists():
            raise IOError(f"Left video file does not exist: {self.left_path}")
        if not self.right_path.exists():
            raise IOError(f"Right video file does not exist: {self.right_path}")

        try:
            self._left_cap  = cv2.VideoCapture(str(self.left_path))
            self._right_cap = cv2.VideoCapture(str(self.right_path))
        except cv2.error as e:
            raise IOError(
                f"OpenCV could not open the video pair "
                f"({self.left_path.name} / {self.right_path.name}): {e}")

        if not self._left_cap.isOpened():
            raise IOError(f"Could not open left video: {self.left_path}")
        if not self._right_cap.isOpened():
            raise IOError(f"Could not open right video: {self.right_path}")

        left_n  = int(self._left_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        right_n = int(self._right_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_frames = min(left_n, right_n)
        if left_n != right_n:
            logging.warning(
                f"OfflineDualVideoSource: left has {left_n} frames, "
                f"right has {right_n} -- using the shorter count "
                f"({self.total_frames}); the two recordings may not "
                f"be from the same session.")

        self.fps = self._left_cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._idx = 0

    def read_pair(self):
        """Read the next synchronized frame pair, or (None, None) at EOF."""
        if self._idx >= self.total_frames:
            return None, None
        ok_l, left  = self._left_cap.read()
        ok_r, right = self._right_cap.read()
        if not (ok_l and ok_r):
            return None, None
        self._idx += 1
        return left, right

    def seek(self, frame_idx: int):
        """Jump to a specific frame index (for scrubbing)."""
        frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        self._left_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self._right_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self._idx = frame_idx

    def close(self):
        self._left_cap.release()
        self._right_cap.release()


class OfflineImagePairSource:
    """
    Alternative source: a folder of paired Left_*/Right_* still images
    (matching the naming AcquisitionPanelRGB's manual/auto capture
    already writes), rather than video. Same read_pair()/seek()/close()
    interface as OfflineDualVideoSource so OfflineInferenceRunner can
    use either without caring which.
    """

    def __init__(self, folder):
        folder = Path(folder)
        left_files  = sorted(folder.glob("*LEFT*")) or sorted(folder.glob("*_L.*"))
        right_files = sorted(folder.glob("*RIGHT*")) or sorted(folder.glob("*_R.*"))
        if not left_files or not right_files:
            raise IOError(
                f"No paired LEFT/RIGHT images found in {folder} "
                f"(expected filenames containing 'LEFT'/'RIGHT' or "
                f"ending in '_L'/'_R')")
        self.total_frames = min(len(left_files), len(right_files))
        self._lefts  = left_files[:self.total_frames]
        self._rights = right_files[:self.total_frames]
        self.fps = 1.0   # not meaningful for stills; kept for API parity
        self._idx = 0

    def read_pair(self):
        if self._idx >= self.total_frames:
            return None, None
        left  = cv2.imread(str(self._lefts[self._idx]))
        right = cv2.imread(str(self._rights[self._idx]))
        self._idx += 1
        return left, right

    def seek(self, frame_idx: int):
        self._idx = max(0, min(frame_idx, self.total_frames - 1))

    def close(self):
        pass


def _build_side_by_side(left, right):
    """
    Minimal Side-by-Side stitch matching
    DualCameraPanel._build_display()'s Side-by-Side output exactly:
    each half gets HALF THE WIDTH, with height derived from the
    camera's own aspect ratio (not the input frame's raw height) so
    neither half is stretched -- half_w-by-full-height would squash
    the image vertically-relative-to-horizontally, which is exactly
    the bug this was caught doing before this fix. The shape produced
    here is what draw_detection_overlay() expects (its own scale
    factor, half_w/1920, assumes a proportionally-scaled frame).
    """
    h, w = left.shape[:2]
    cam_aspect = w / h if h else 16 / 9   # source frame's own aspect ratio
    half_w = w // 2 if w == right.shape[1] else min(w, right.shape[1]) // 2
    disp_h = max(1, int(half_w / cam_aspect))
    l_disp = cv2.resize(left,  (half_w, disp_h))
    r_disp = cv2.resize(right, (half_w, disp_h))
    divider = np.zeros((disp_h, 4, 3), dtype=np.uint8)
    return cv2.hconcat([l_disp, divider, r_disp])


class DetectionSummary:
    """
    Accumulates detection statistics across a processed video/image
    set. NOT an accuracy evaluation (no ground truth involved) -- see
    the module docstring. Mirrors the aggregation style of
    core/gui_session_report.py's _statistics() for a consistent look
    across the app's various summary displays.
    """

    def __init__(self):
        self.frames_processed = 0
        self.frames_with_detection = 0
        self._all_detections = []   # list of (class_name, confidence)

    def add(self, dual_result):
        self.frames_processed += 1
        dets = dual_result.all_detections()
        if dets:
            self.frames_with_detection += 1
        for d in dets:
            self._all_detections.append((d.class_name, d.confidence))

    def finalize(self) -> dict:
        by_class, confs = {}, []
        for cls, conf in self._all_detections:
            by_class[cls] = by_class.get(cls, 0) + 1
            confs.append(conf)

        conf_stats = {}
        if confs:
            conf_stats = {
                "mean": round(statistics.mean(confs), 3),
                "min":  round(min(confs), 3),
                "max":  round(max(confs), 3),
            }
            if len(confs) > 1:
                conf_stats["stdev"] = round(statistics.stdev(confs), 3)

        return {
            "frames_processed":       self.frames_processed,
            "frames_with_detection":  self.frames_with_detection,
            "frames_with_detection_pct": (
                round(self.frames_with_detection / self.frames_processed * 100, 1)
                if self.frames_processed else 0.0),
            "total_detections":       len(self._all_detections),
            "detections_by_class":    by_class,
            "confidence_stats":       conf_stats,
        }


class OfflineInferenceRunner:
    """
    Runs RGBDetectionEngine on a video/image source, frame by frame.
    process() is a generator -- yields (frame_idx, overlay_img,
    dual_result) one at a time rather than holding a whole processed
    video in memory, since field recordings can run to many thousands
    of frames.
    """

    def __init__(self, model_path: str, mode: str = "weed",
                confidence_threshold: float = None,
                iou_threshold: float = None):
        from core.detection_engine_rgb import RGBDetectionEngine
        from core.detection_config_rgb import get_weed_config, get_cls_config

        self.cfg = get_cls_config() if mode == "cls" else get_weed_config()
        if model_path:
            path = Path(model_path)
            if mode == "cls":
                self.cfg.model.cls_rgb_pt = path
            else:
                self.cfg.model.weed_rgb_pt = path
        if confidence_threshold is not None:
            self.cfg.model.confidence_threshold = confidence_threshold
        if iou_threshold is not None:
            self.cfg.model.iou_threshold = iou_threshold
        self.engine = RGBDetectionEngine(self.cfg)

    def process(self, source, spray_states=(False, False, False),
                should_stop=None):
        """
        source: an OfflineDualVideoSource or OfflineImagePairSource
        (anything with read_pair()/total_frames/fps).
        should_stop: optional callable returning True to abort early
        (wired to a GUI Stop button).

        Yields (frame_idx, overlay_img, dual_result) per frame. Caller
        is responsible for calling source.close() when done -- kept
        external rather than closed here so a caller can seek/reuse
        the same source across multiple process() calls if needed.
        """
        from core.dual_emeet_camera import FramePair
        from gui.overlay_rendering import draw_detection_overlay

        frame_idx = 0
        while True:
            if should_stop is not None and should_stop():
                break
            left, right = source.read_pair()
            if left is None:
                break

            pair = FramePair(
                left=left, right=right, frame_id=frame_idx,
                left_ts=frame_idx / source.fps,
                right_ts=frame_idx / source.fps,
                sync_error_ms=0.0,
            )
            try:
                dual_result = self.engine.run(pair)
            except Exception as e:
                logging.error(f"OfflineInferenceRunner: inference failed "
                              f"on frame {frame_idx}: {e}")
                frame_idx += 1
                continue

            side_by_side = _build_side_by_side(left, right)
            overlay_img = draw_detection_overlay(
                side_by_side, dual_result, list(spray_states), self.cfg,
                # Larger than the live Detection tab's defaults -- this
                # combined side-by-side frame is typically viewed at a
                # different scale in the offline review UI, and the
                # live-tuned size (13px/1px) read as too small there.
                font_size=20, box_thick=2)

            yield frame_idx, overlay_img, dual_result
            frame_idx += 1


if __name__ == "__main__":
    import sys, tempfile, numpy as np

    # Running this file directly puts core/ on sys.path, not the
    # project root -- so `from core.dual_emeet_camera import ...`
    # would fail. Same fix as tools/demo_spray.py and
    # core/camera_presets.py use.
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    print("=" * 55)
    print("core/offline_inference.py — Self Test")
    print("=" * 55)

    # Build two tiny synthetic dual-camera videos to test against,
    # since no real field footage is available in this sandbox.
    tmp = Path(tempfile.mkdtemp())
    left_path  = tmp / "left.mp4"
    right_path = tmp / "right.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    n_frames = 5
    for path in (left_path, right_path):
        w = cv2.VideoWriter(str(path), fourcc, 10.0, (320, 240))
        for i in range(n_frames):
            frame = np.full((240, 320, 3), (40 + i * 10, 60, 30), dtype=np.uint8)
            w.write(frame)
        w.release()

    src = OfflineDualVideoSource(left_path, right_path)
    assert src.total_frames == n_frames
    print(f"✓ OfflineDualVideoSource opened both files, "
          f"total_frames={src.total_frames}")

    left, right = src.read_pair()
    assert left is not None and right is not None
    assert left.shape == (240, 320, 3)
    print(f"✓ read_pair() returns real frame data: {left.shape}")

    src.seek(2)
    left2, right2 = src.read_pair()
    assert left2 is not None
    print("✓ seek() correctly repositions both streams")
    src.close()

    # Mismatched-length source (left has fewer frames) should degrade
    # gracefully to the shorter count, not raise
    short_left = tmp / "short_left.mp4"
    w = cv2.VideoWriter(str(short_left), fourcc, 10.0, (320, 240))
    for i in range(3):
        w.write(np.zeros((240, 320, 3), dtype=np.uint8))
    w.release()
    src2 = OfflineDualVideoSource(short_left, right_path)
    assert src2.total_frames == 3
    print(f"✓ Mismatched frame counts (3 vs {n_frames}) correctly use "
          f"the shorter count ({src2.total_frames}), not raise")
    src2.close()

    # _build_side_by_side produces the shape draw_detection_overlay expects
    l = np.zeros((240, 320, 3), dtype=np.uint8)
    r = np.zeros((240, 320, 3), dtype=np.uint8)
    sbs = _build_side_by_side(l, r)
    assert sbs.shape[1] == (320 // 2) * 2 + 4
    print(f"✓ _build_side_by_side() produces the (half_w*2+4) width shape "
          f"draw_detection_overlay() expects: {sbs.shape}")

    # DetectionSummary aggregation, using real Detection objects
    from core.detection_engine_rgb import Detection, InferenceResult, DualInferenceResult
    import time as _t
    summary = DetectionSummary()
    for i in range(3):
        det = Detection(class_id=1, class_name="kochia", confidence=0.8 + i * 0.05,
                        x1=10, y1=10, x2=50, y2=50, camera="left")
        dual = DualInferenceResult(
            left=InferenceResult(detections=[det], inference_ms=5, preprocess_ms=2,
                                 total_ms=7, frame_shape=(1080,1920), camera="left"),
            right=InferenceResult(detections=[], inference_ms=5, preprocess_ms=2,
                                  total_ms=7, frame_shape=(1080,1920), camera="right"),
            frame_id=i, timestamp=_t.time())
        summary.add(dual)
    summary.add(DualInferenceResult(  # a frame with no detections
        left=InferenceResult(detections=[], inference_ms=5, preprocess_ms=2,
                             total_ms=7, frame_shape=(1080,1920), camera="left"),
        right=InferenceResult(detections=[], inference_ms=5, preprocess_ms=2,
                              total_ms=7, frame_shape=(1080,1920), camera="right"),
        frame_id=3, timestamp=_t.time()))
    result = summary.finalize()
    assert result["frames_processed"] == 4
    assert result["frames_with_detection"] == 3
    assert result["total_detections"] == 3
    assert result["detections_by_class"] == {"kochia": 3}
    assert result["confidence_stats"]["min"] == 0.8
    print(f"✓ DetectionSummary correctly aggregates: {result}")

    print()
    print("=" * 55)
    print("core/offline_inference.py ✓ ALL TESTS PASSED")
    print("=" * 55)
