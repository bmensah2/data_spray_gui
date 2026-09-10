"""
gui/tabs/tab_offline_review.py
ABEN Dual RGB Imaging System — Offline Video/Image Review Tab

Load recorded field data (paired Left/Right videos or image folders),
run it through the same detection model used live, and review the
results frame by frame -- qualitative review of real field footage,
not an accuracy evaluation (no ground truth is involved here; see the
evaluate_model_rgb.py launcher section for that, once labeled data
exists).

Threading: inference on a long video can take minutes, so processing
runs in a background thread. Frame updates are marshaled back to the
GUI thread via pyqtSignal (a queued connection is automatic for
cross-thread signal emission in PyQt5), not QMetaObject.invokeMethod --
cleaner here since each frame update carries a numpy array and an
arbitrary result object, not just simple types.
"""

import time
import threading
from pathlib import Path

import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QPushButton, QLineEdit, QComboBox, QDoubleSpinBox,
    QFileDialog, QProgressBar, QSlider, QMessageBox, QTabWidget,
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap

from gui.style import _divider, _muted, _sec
from gui.theme_manager import theme_manager
from gui.shared_log import UnifiedLog, LogPanel
from gui.spray_event_table import build_stats_table, update_stats_row


class OfflineReviewTab(QWidget):
    """
    Tab 4 (was "Future"): load recorded dual-camera video/images, run
    the live model against them, watch results, export an annotated
    video. Self-contained -- doesn't share state with the live
    Detection tab (a fresh RGBDetectionEngine is built here from
    whichever model the operator picks), so reviewing footage never
    interferes with a live/armed session.
    """

    # Emitted from the worker thread per processed frame -- queued
    # automatically across threads by Qt, updates the GUI safely.
    _frame_ready = pyqtSignal(int, int, object, object)  # idx, total, overlay_img, dual_result
    _processing_done = pyqtSignal(bool, str)              # success, message

    def __init__(self, shared_log: UnifiedLog, parent=None):
        super().__init__(parent)
        self.log = shared_log

        self._runner       = None
        self._source        = None
        self._worker_thread = None
        self._stop_flag     = threading.Event()
        self._summary       = None
        self._video_writer  = None
        self._export_path   = None
        self._last_overlay  = None

        self._frame_ready.connect(self._on_frame_ready)
        self._processing_done.connect(self._on_processing_done)

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        sub_tabs = QTabWidget()
        sub_tabs.addTab(self._review_subtab(), "🎞  Video / Image Review")
        sub_tabs.addTab(self._evaluation_subtab(), "📊  Model Evaluation")
        root.addWidget(sub_tabs)

        root.addWidget(
            LogPanel(self.log, sources=["REVIEW", "EVAL", "SYS"],
                     height=90))

    def _review_subtab(self) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(8)

        # ── Left: source + controls ────────────────────────
        left = QWidget()
        left.setMinimumWidth(340)
        left.setMaximumWidth(420)
        llay = QVBoxLayout(left)
        llay.setSpacing(8)

        src_grp = QGroupBox("Source")
        sg = QGridLayout(src_grp)
        r = 0
        sg.addWidget(_muted("Mode:"), r, 0)
        self.cmb_source_mode = QComboBox()
        self.cmb_source_mode.addItems(
            ["Dual Video (Left + Right)", "Image Folder (paired L/R)"])
        self.cmb_source_mode.currentIndexChanged.connect(
            self._on_source_mode_changed)
        sg.addWidget(self.cmb_source_mode, r, 1); r += 1

        sg.addWidget(_muted("Left:"), r, 0)
        self.ed_left = QLineEdit()
        self.ed_left.setPlaceholderText("Left.mp4")
        sg.addWidget(self.ed_left, r, 1)
        btn_left = QPushButton("Browse")
        btn_left.clicked.connect(lambda: self._browse_file(self.ed_left))
        sg.addWidget(btn_left, r, 2); r += 1

        sg.addWidget(_muted("Right:"), r, 0)
        self.ed_right = QLineEdit()
        self.ed_right.setPlaceholderText("Right.mp4")
        sg.addWidget(self.ed_right, r, 1)
        self.btn_right = QPushButton("Browse")
        self.btn_right.clicked.connect(lambda: self._browse_file(self.ed_right))
        sg.addWidget(self.btn_right, r, 2); r += 1

        sg.addWidget(_muted("Model:"), r, 0)
        self.ed_model = QLineEdit("models/weed_rgb.pt")
        sg.addWidget(self.ed_model, r, 1)
        btn_model = QPushButton("Browse")
        btn_model.clicked.connect(lambda: self._browse_file(
            self.ed_model, "Model files (*.pt *.engine)"))
        sg.addWidget(btn_model, r, 2); r += 1

        sg.addWidget(_muted("Mode:"), r, 0)
        self.cmb_model_mode = QComboBox()
        self.cmb_model_mode.addItems(["WEED — herbicide", "CLS — fungicide"])
        sg.addWidget(self.cmb_model_mode, r, 1); r += 1

        sg.addWidget(_muted("Confidence:"), r, 0)
        self.spn_conf = QDoubleSpinBox()
        self.spn_conf.setRange(0.05, 0.95)
        self.spn_conf.setSingleStep(0.05)
        self.spn_conf.setValue(0.45)
        sg.addWidget(self.spn_conf, r, 1); r += 1

        llay.addWidget(src_grp)

        ctrl_grp = QGroupBox("Playback")
        cg = QVBoxLayout(ctrl_grp)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("▶  START REVIEW")
        theme_manager.register_button(self.btn_start, "green")
        self.btn_start.clicked.connect(self._start_processing)
        btn_row.addWidget(self.btn_start)

        self.btn_stop = QPushButton("⏹  STOP")
        theme_manager.register_button(self.btn_stop, "dim_red")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_processing)
        btn_row.addWidget(self.btn_stop)
        cg.addLayout(btn_row)

        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        cg.addWidget(self.progress)

        self.lbl_frame_pos = _muted("Frame: — / —")
        cg.addWidget(self.lbl_frame_pos)

        llay.addWidget(ctrl_grp)

        export_grp = QGroupBox("Export")
        eg = QVBoxLayout(export_grp)
        self.chk_export_label = _muted(
            "Export writes an annotated video (detections burned in) "
            "alongside processing — enable before START REVIEW.")
        self.chk_export_label.setWordWrap(True)
        eg.addWidget(self.chk_export_label)

        exp_row = QHBoxLayout()
        self.ed_export_path = QLineEdit()
        self.ed_export_path.setPlaceholderText(
            "(leave blank to skip export)")
        exp_row.addWidget(self.ed_export_path)
        btn_export = QPushButton("Browse")
        btn_export.clicked.connect(self._browse_export_path)
        exp_row.addWidget(btn_export)
        eg.addLayout(exp_row)
        llay.addWidget(export_grp)

        stats_grp = QGroupBox("Detection Summary")
        sg2 = QVBoxLayout(stats_grp)
        sg2.addWidget(_muted(
            "Counts from this footage only — not an accuracy metric "
            "(no ground truth). See Model Evaluation for that."))
        self.tbl_summary = build_stats_table(
            ["Frames", "With Det.", "Total Det.", "Top Class", "Mean Conf"])
        sg2.addWidget(self.tbl_summary)
        llay.addWidget(stats_grp)

        llay.addStretch()
        lay.addWidget(left)

        # ── Right: frame display ───────────────────────────
        right = QWidget()
        rlay = QVBoxLayout(right)
        self.lbl_display = QLabel("Load a video/image pair and press "
                                  "START REVIEW")
        self.lbl_display.setAlignment(Qt.AlignCenter)
        self.lbl_display.setMinimumSize(640, 360)
        theme_manager.register_widget(
            self.lbl_display, lambda p: (
                f"background-color:#000;color:{p['muted']};"
                f"font-family:'Noto Sans',Arial,sans-serif;"))
        rlay.addWidget(self.lbl_display, stretch=1)
        lay.addWidget(right, stretch=1)

        return w

    def _evaluation_subtab(self) -> QWidget:
        """
        Launcher for the existing evaluate_model_rgb.py /
        compare_yolo_models_rgb.py scripts, for whenever labeled
        ground truth exists for some of this footage. Deliberately
        just a thin subprocess launcher (matching the pattern used
        for every other script-launching button in this app) rather
        than reimplementing evaluation logic in the GUI -- those
        scripts are the source of truth for accuracy metrics.
        """
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(20, 20, 20, 20)
        lay.setSpacing(10)

        lay.addWidget(_sec("Model Evaluation (requires labeled data)"))
        info = _muted(
            "Runs evaluate_model_rgb.py against a labeled YOLO-format "
            "validation set (images/ + labels/) and reports real mAP, "
            "precision and recall — this needs ground-truth labels, "
            "unlike the Review tab, which only counts raw detections.")
        info.setWordWrap(True)
        lay.addWidget(info)
        lay.addWidget(_divider())

        grid = QGridLayout()
        grid.addWidget(_muted("Model:"), 0, 0)
        self.ed_eval_model = QLineEdit("models/weed_rgb.pt")
        grid.addWidget(self.ed_eval_model, 0, 1)
        btn1 = QPushButton("Browse")
        btn1.clicked.connect(lambda: self._browse_file(
            self.ed_eval_model, "Model files (*.pt *.engine)"))
        grid.addWidget(btn1, 0, 2)

        grid.addWidget(_muted("Dataset (data.yaml):"), 1, 0)
        self.ed_eval_data = QLineEdit()
        self.ed_eval_data.setPlaceholderText("path/to/data_rgb.yaml")
        grid.addWidget(self.ed_eval_data, 1, 1)
        btn2 = QPushButton("Browse")
        btn2.clicked.connect(lambda: self._browse_file(
            self.ed_eval_data, "YAML files (*.yaml *.yml)"))
        grid.addWidget(btn2, 1, 2)
        lay.addLayout(grid)

        self.btn_run_eval = QPushButton("▶  RUN EVALUATION")
        theme_manager.register_button(self.btn_run_eval, "green")
        self.btn_run_eval.clicked.connect(self._run_evaluation)
        lay.addWidget(self.btn_run_eval)

        self.lbl_eval_status = _muted(
            "No evaluation run yet this session.")
        self.lbl_eval_status.setWordWrap(True)
        lay.addWidget(self.lbl_eval_status)

        lay.addStretch()
        return w

    # ── Source mode ──────────────────────────────────────────

    def _on_source_mode_changed(self, idx):
        is_images = (idx == 1)
        self.ed_left.setPlaceholderText(
            "image folder" if is_images else "Left.mp4")
        self.ed_right.setVisible(not is_images)
        self.btn_right.setVisible(not is_images)

    def _browse_file(self, target_edit, filter_str="Video files (*.mp4 *.avi)"):
        if self.cmb_source_mode.currentIndex() == 1 and target_edit is self.ed_left:
            path = QFileDialog.getExistingDirectory(self, "Select image folder")
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select file", "", filter_str)
        if path:
            target_edit.setText(path)

    def _browse_export_path(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save annotated video as", "", "MP4 video (*.mp4)")
        if path:
            self.ed_export_path.setText(path)

    # ── Processing ────────────────────────────────────────────

    def _start_processing(self):
        left_str  = self.ed_left.text().strip()
        right_str = self.ed_right.text().strip()
        model_str = self.ed_model.text().strip()

        if not left_str or (self.cmb_source_mode.currentIndex() == 0
                            and not right_str):
            self.log.log("REVIEW", "Select source file(s) first", "warn")
            return
        if not model_str:
            self.log.log("REVIEW", "Select a model first", "warn")
            return

        from core.offline_inference import (
            OfflineDualVideoSource, OfflineImagePairSource,
            OfflineInferenceRunner, DetectionSummary)

        try:
            if self.cmb_source_mode.currentIndex() == 1:
                self._source = OfflineImagePairSource(left_str)
            else:
                self._source = OfflineDualVideoSource(left_str, right_str)
        except Exception as e:
            self.log.log("REVIEW", f"Could not open source: {e}", "error")
            return

        mode = "cls" if self.cmb_model_mode.currentIndex() == 1 else "weed"
        try:
            self._runner = OfflineInferenceRunner(
                model_path=model_str, mode=mode,
                confidence_threshold=self.spn_conf.value())
        except Exception as e:
            self.log.log("REVIEW", f"Could not load model: {e}", "error")
            self._source.close()
            self._source = None
            return

        self._summary = DetectionSummary()
        self.progress.setMaximum(self._source.total_frames)
        self.progress.setValue(0)

        export_path = self.ed_export_path.text().strip()
        self._video_writer = None
        self._export_path  = export_path or None
        # VideoWriter needs a fixed frame size known up front, which
        # isn't available until the first overlay frame is actually
        # drawn -- constructed lazily in _write_export_frame() instead
        # of guessing a size here.

        self._stop_flag.clear()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.log.log("REVIEW", f"Processing {self._source.total_frames} "
                     f"frame(s)…", "info")

        self._worker_thread = threading.Thread(
            target=self._worker_run, daemon=True)
        self._worker_thread.start()

    def _worker_run(self):
        total = self._source.total_frames
        success, message = True, ""
        try:
            for frame_idx, overlay_img, dual_result in self._runner.process(
                    self._source, should_stop=self._stop_flag.is_set):
                self._summary.add(dual_result)
                if self._export_path is not None:
                    self._write_export_frame(overlay_img)
                self._frame_ready.emit(
                    frame_idx, total, overlay_img, dual_result)
        except Exception as e:
            success, message = False, str(e)
        finally:
            self._processing_done.emit(success, message)

    def _write_export_frame(self, overlay_img):
        # VideoWriter needs a fixed frame size decided up front; only
        # known once the first overlay frame is actually drawn, so the
        # writer is constructed here, on the first call, rather than
        # guessing a size in _start_processing().
        if self._video_writer is None:
            h, w = overlay_img.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = self._source.fps if self._source.fps > 0 else 10.0
            self._video_writer = cv2.VideoWriter(
                self._export_path, fourcc, fps, (w, h))
        self._video_writer.write(overlay_img)

    def _stop_processing(self):
        self._stop_flag.set()
        self.log.log("REVIEW", "Stopping…", "info")

    def _on_frame_ready(self, frame_idx, total, overlay_img, dual_result):
        self._last_overlay = overlay_img
        self._show_frame(overlay_img)
        self.progress.setValue(frame_idx + 1)
        self.lbl_frame_pos.setText(f"Frame: {frame_idx + 1} / {total}")

        s = self._summary.finalize()
        by_class = s.get("detections_by_class", {})
        top_class = max(by_class, key=by_class.get) if by_class else "—"
        conf = s.get("confidence_stats", {}).get("mean", 0.0)
        update_stats_row(self.tbl_summary, [
            s["frames_processed"], s["frames_with_detection"],
            s["total_detections"], top_class, f"{conf:.2f}" if conf else "—",
        ])

    def _on_processing_done(self, success, message):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
            self.log.log("REVIEW", f"Annotated video saved: "
                         f"{self._export_path}", "ok")
        if self._source is not None:
            self._source.close()
            self._source = None
        if success:
            self.log.log("REVIEW", "Processing complete", "ok")
        else:
            self.log.log("REVIEW", f"Processing error: {message}", "error")

    def _show_frame(self, img):
        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            rgb = np.ascontiguousarray(rgb)
            qimg = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format_RGB888)
            self.lbl_display.setPixmap(
                QPixmap.fromImage(qimg).scaled(
                    self.lbl_display.size(), Qt.KeepAspectRatio,
                    Qt.SmoothTransformation))
        except Exception:
            pass

    # ── Evaluation launcher ──────────────────────────────────

    def _run_evaluation(self):
        import shutil, subprocess

        model_path = self.ed_eval_model.text().strip()
        data_path  = self.ed_eval_data.text().strip()
        if not model_path or not data_path:
            self.log.log("EVAL", "Select a model and dataset first", "warn")
            return
        if not shutil.which("python3"):
            self.log.log("EVAL", "python3 not found", "error")
            return

        script = (Path(__file__).resolve().parent.parent.parent /
                 "evaluate_model_rgb.py")
        if not script.exists():
            self.log.log("EVAL", f"evaluate_model_rgb.py not found at "
                         f"{script}", "error")
            return

        cmd = ["python3", str(script),
               "--model", model_path, "--data", data_path]
        self.log.log("EVAL", f"Running evaluation…", "info")
        self.btn_run_eval.setEnabled(False)
        self.lbl_eval_status.setText("Running…")

        def _run():
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True,
                    cwd=str(script.parent), timeout=600)
                ok = result.returncode == 0
                out = (result.stdout[-2000:] if ok
                      else result.stderr[-2000:])
            except subprocess.TimeoutExpired:
                ok, out = False, "Timed out after 600s"
            except Exception as e:
                ok, out = False, str(e)
            QTimer.singleShot(0, lambda: self._eval_done(ok, out))

        threading.Thread(target=_run, daemon=True).start()

    def _eval_done(self, ok, output):
        self.btn_run_eval.setEnabled(True)
        if ok:
            self.log.log("EVAL", "Evaluation complete — see log/output "
                         "files next to the dataset", "ok")
            self.lbl_eval_status.setText(
                f"✓ Evaluation complete.\n\n{output}")
        else:
            self.log.log("EVAL", "Evaluation failed", "error")
            self.lbl_eval_status.setText(f"✗ Evaluation failed:\n\n{output}")

    def cleanup(self):
        self._stop_flag.set()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        if self._video_writer is not None:
            self._video_writer.release()
        if self._source is not None:
            self._source.close()
