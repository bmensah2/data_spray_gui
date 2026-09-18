"""
gui/panels/triple_capture_panel.py
ABEN Triple RGB Data Collection Panel

Group D of the triple-camera standalone app's capability pass: the
Data Collection tab, which didn't exist at all for the triple-camera
system before this. Manual image capture with labels, and interval-
based auto-capture -- ported from
gui/panels/acquisition_panel_rgb.py's AcquisitionPanelRGB
(_tab_capture()/_tab_image()/_tab_auto()/_save_pair()/_capture_image()/
_auto_start()/_auto_pause()/_auto_stop()), reusing its generic label
constants (CROPS/STAGES/TARGETS/BASE_PATH -- none of which are
camera-count-specific) directly rather than duplicating them.

Deliberately different from the 2-camera version in one key way: all
3 cameras save into ONE SHARED session folder per capture (matching
core/triple_emeet_camera.py's TripleEMEETCamera.save_triple(), Phase
2's own design, per the operator's explicit request that training
data from all cameras live together this time), not split into
per-camera left/right-style subfolders. Because of this, the 2-camera
system's "Capture Source" (Both/Left/Right) radio group has no
equivalent here and is dropped entirely -- every capture always saves
all 3 cameras together.

Deliberately scoped for this first pass -- ported manual capture +
auto-capture and left out for a later pass (neither blocks collecting
labeled training data, which is this tab's actual purpose):
  - Video recording (_tab_video()/_vid_start()/_write_video_frame()/
    _vid_stop())
"""

import json
import re
from datetime import datetime
from pathlib import Path

import cv2
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QPushButton, QComboBox, QCheckBox, QLineEdit,
    QTabWidget, QSpinBox, QDoubleSpinBox, QProgressBar, QFileDialog,
)
from PyQt5.QtCore import QTimer

from gui.style import _muted, _sec
from gui.theme_manager import theme_manager
from gui.shared_log import UnifiedLog

try:
    from gui.panels.acquisition_panel_rgb import CROPS, STAGES, TARGETS, BASE_PATH
except ImportError:
    from acquisition_panel_rgb import CROPS, STAGES, TARGETS, BASE_PATH


class TripleCapturePanel(QWidget):
    """Data Collection tab content for the triple-camera system."""

    def __init__(self, shared_log: UnifiedLog, camera, parent=None):
        super().__init__(parent)
        self.shared_log = shared_log
        self.camera     = camera   # TripleCameraPanel

        self._session_path   = None
        self._session_labels = None
        self._capture_count  = 0
        self._session_id     = None
        self._img_count      = 0

        self._auto_timer = None
        self._auto_count = 0
        self._auto_max   = 0

        self._build_ui()

        self._info_timer = QTimer()
        self._info_timer.timeout.connect(self._refresh_device_status)
        self._info_timer.start(5000)

    # ── Build UI ──────────────────────────────────────────────

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        lay.addWidget(_sec("DATA CAPTURE — Triple Camera"))

        self.lbl_device_status = QLabel("—")
        theme_manager.register_widget(
            self.lbl_device_status, lambda p: (
                f"color:{p['muted']};font-size:9px;"
                f"font-family:'Noto Sans',Arial,sans-serif;"))
        lay.addWidget(self.lbl_device_status)

        meta_grp = QGroupBox("Session Labels")
        mg = QGridLayout(meta_grp)
        mg.setSpacing(6)

        mg.addWidget(_muted("Crop:"), 0, 0)
        self.cmb_crop = QComboBox()
        self.cmb_crop.addItems(CROPS)
        self.cmb_crop.currentTextChanged.connect(self.reset_session)
        mg.addWidget(self.cmb_crop, 0, 1)

        mg.addWidget(_muted("Target:"), 1, 0)
        self.cmb_target = QComboBox()
        self.cmb_target.addItems(TARGETS)
        self.cmb_target.currentTextChanged.connect(self.reset_session)
        mg.addWidget(self.cmb_target, 1, 1)

        mg.addWidget(_muted("Stage:"), 2, 0)
        self.cmb_stage = QComboBox()
        self.cmb_stage.addItems(STAGES)
        self.cmb_stage.currentTextChanged.connect(self.reset_session)
        mg.addWidget(self.cmb_stage, 2, 1)

        mg.addWidget(_muted("Save to:"), 3, 0)
        folder_row = QHBoxLayout()
        self.lbl_folder = QLabel(str(BASE_PATH))
        theme_manager.register_widget(
            self.lbl_folder, lambda p: (
                f"color:{p['dim']};font-size:9px;"
                f"font-family:'Noto Sans',Arial,sans-serif;"))
        self.lbl_folder.setWordWrap(True)
        folder_row.addWidget(self.lbl_folder, stretch=1)
        btn_browse = QPushButton("Browse")
        theme_manager.register_button(btn_browse, "blue")
        btn_browse.setFixedWidth(60)
        btn_browse.clicked.connect(self._browse_folder)
        folder_row.addWidget(btn_browse)
        mg.addLayout(folder_row, 3, 1)
        lay.addWidget(meta_grp)

        extra_grp = QGroupBox("Additional Labels")
        xg = QGridLayout(extra_grp)
        xg.setSpacing(6)

        xg.addWidget(_muted("Disease present:"), 0, 0)
        self.chk_disease = QCheckBox()
        self.chk_disease.stateChanged.connect(self._on_disease_toggle)
        xg.addWidget(self.chk_disease, 0, 1)

        xg.addWidget(_muted("Disease type:"), 1, 0)
        self.cmb_disease_type = QComboBox()
        self.cmb_disease_type.addItems([
            "none", "cercospora", "powdery_mildew",
            "rust", "blight", "root_rot", "other"
        ])
        self.cmb_disease_type.setEnabled(False)
        xg.addWidget(self.cmb_disease_type, 1, 1)

        xg.addWidget(_muted("Weed type:"), 2, 0)
        self.cmb_weed_type = QComboBox()
        self.cmb_weed_type.addItems([
            "none", "kochia", "waterhemp", "ragweed",
            "wild_oat", "pigweed", "lambsquarters",
            "foxtail", "bindweed", "mixed", "unknown"
        ])
        xg.addWidget(self.cmb_weed_type, 2, 1)

        xg.addWidget(_muted("Notes:"), 3, 0)
        self.entry_notes = QLineEdit()
        self.entry_notes.setPlaceholderText(
            "e.g. heavy canopy, wet, 3 DAE …")
        xg.addWidget(self.entry_notes, 3, 1)
        lay.addWidget(extra_grp)

        cap_tabs = QTabWidget()
        cap_tabs.addTab(self._tab_image(), "Image")
        cap_tabs.addTab(self._tab_auto(),  "Auto")
        lay.addWidget(cap_tabs, stretch=1)

    def _tab_image(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        self.lbl_img_count = _muted("Captures this session: 0")
        lay.addWidget(self.lbl_img_count)

        self.lbl_last_saved = QLabel("—")
        theme_manager.register_widget(
            self.lbl_last_saved, lambda p: (
                f"color:{p['dim']};font-size:9px;"
                f"font-family:'Noto Sans',Arial,sans-serif;"))
        self.lbl_last_saved.setWordWrap(True)
        lay.addWidget(self.lbl_last_saved)

        btn_cap = QPushButton("📷  CAPTURE IMAGE (all 3 cameras)")
        theme_manager.register_button(btn_cap, "green")
        btn_cap.setMinimumHeight(36)
        btn_cap.clicked.connect(self._capture_image)
        lay.addWidget(btn_cap)

        lay.addStretch()
        return w

    def _tab_auto(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        cfg_grp = QGroupBox("Auto Capture Config")
        cg = QGridLayout(cfg_grp)

        cg.addWidget(_muted("Interval (s):"), 0, 0)
        self.spn_interval = QDoubleSpinBox()
        self.spn_interval.setRange(0.5, 60.0)
        self.spn_interval.setValue(2.0)
        self.spn_interval.setSingleStep(0.5)
        cg.addWidget(self.spn_interval, 0, 1)

        cg.addWidget(_muted("Max captures:"), 1, 0)
        self.spn_max_cap = QSpinBox()
        self.spn_max_cap.setRange(1, 9999)
        self.spn_max_cap.setValue(50)
        cg.addWidget(self.spn_max_cap, 1, 1)
        lay.addWidget(cfg_grp)

        self.auto_progress = QProgressBar()
        self.auto_progress.setRange(0, 100)
        self.auto_progress.setValue(0)
        self.auto_progress.setFormat("Ready")
        lay.addWidget(self.auto_progress)

        btn_row = QHBoxLayout()
        self.btn_auto_start = QPushButton("▶  START")
        theme_manager.register_button(self.btn_auto_start, "green")
        self.btn_auto_start.clicked.connect(self._auto_start)
        btn_row.addWidget(self.btn_auto_start)

        self.btn_auto_pause = QPushButton("⏸  PAUSE")
        theme_manager.register_button(self.btn_auto_pause, "amber")
        self.btn_auto_pause.setEnabled(False)
        self.btn_auto_pause.clicked.connect(self._auto_pause)
        btn_row.addWidget(self.btn_auto_pause)

        self.btn_auto_stop = QPushButton("⏹  STOP")
        theme_manager.register_button(self.btn_auto_stop, "dim_red")
        self.btn_auto_stop.setEnabled(False)
        self.btn_auto_stop.clicked.connect(self._auto_stop)
        btn_row.addWidget(self.btn_auto_stop)
        lay.addLayout(btn_row)

        lay.addStretch()
        return w

    # ── Labels / session folder ───────────────────────────────

    def _get_labels(self) -> dict:
        return {
            "crop_type":       self.cmb_crop.currentText(),
            "target":          self.cmb_target.currentText(),
            "growth_stage":    self.cmb_stage.currentText(),
            "disease_present": self.chk_disease.isChecked(),
            "disease_type":    self.cmb_disease_type.currentText(),
            "weed_type":       self.cmb_weed_type.currentText(),
            "notes":           self.entry_notes.text().strip(),
        }

    def _get_session_dir(self, labels: dict) -> Path:
        """
        Return (and create) the session directory. ONE shared folder
        for all 3 cameras' images -- see module docstring -- not
        per-camera subfolders.
        """
        crop   = re.sub(r"[^a-z0-9_]", "_", labels["crop_type"].lower())
        stage  = re.sub(r"[^a-z0-9_]", "_", labels["growth_stage"].lower())
        target = re.sub(r"[^a-z0-9_]", "_", labels["target"].lower())
        date   = datetime.now().strftime("%Y%m%d")
        key    = f"triple_{crop}_{target}_{stage}_{date}"

        if (self._session_labels == key
                and self._session_path is not None
                and self._session_path.exists()
                and (self._session_path / "metadata").exists()):
            return self._session_path

        base = Path(self.lbl_folder.text()) / key
        base.mkdir(parents=True, exist_ok=True)
        (base / "metadata").mkdir(parents=True, exist_ok=True)

        is_new_key = self._session_labels != key
        if is_new_key:
            self._capture_count = 0
            self._session_id    = key

        self._session_path   = base
        self._session_labels = key

        log_path = base / "session.log"
        with open(log_path, "a") as f:
            reason = "started" if is_new_key else "recreated (folder was deleted)"
            f.write(
                f"\n=== Session {reason}: "
                f"{datetime.now().isoformat()} ===\n"
                f"Labels: {json.dumps(labels, indent=2)}\n"
            )

        action = "New session" if is_new_key else "Session folder recreated"
        self.shared_log.log("CAMERA", f"{action}: {key}", "info")
        return base

    def _save_triple(self, frames: list, labels: dict,
                     session_dir: Path) -> dict:
        """
        Save all 3 frames into session_dir directly (not a per-camera
        subfolder), filenames "<capture_id>_CAM1/2/3.jpg" -- matching
        core/triple_emeet_camera.py's save_triple() naming so images
        from the live pipeline and manually-captured training data
        sort the same way. Write metadata JSON. Return metadata dict.
        """
        cid = f"{self._session_id}_{self._capture_count:04d}"
        ts  = datetime.now().isoformat()
        saved = []
        meta = {
            "capture_id":   cid,
            "timestamp":    ts,
            "labels":       labels,
            "camera_model": "eMeet C960 4K (Triple)",
            "resolution":   "1920x1080",
        }

        for i, frame in enumerate(frames):
            if frame is None:
                continue
            p = session_dir / f"{cid}_CAM{i+1}.jpg"
            cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            meta[f"cam{i+1}_image"] = p.name
            saved.append(f"CAM{i+1}")

        mp = session_dir / "metadata" / f"{cid}_meta.json"
        with open(mp, "w") as f:
            json.dump(meta, f, indent=2)

        lp = session_dir / "session.log"
        with open(lp, "a") as f:
            f.write(
                f"{ts}  [{cid}]  saved={saved}  "
                f"target={labels['target']}  "
                f"stage={labels['growth_stage']}\n")

        return meta

    # ── Single image capture ──────────────────────────────────

    def _capture_image(self):
        frames = self.camera.get_frame_snapshot()
        if any(f is None for f in frames):
            self.shared_log.log(
                "CAMERA", "No frame available from all 3 cameras", "warn")
            return

        labels      = self._get_labels()
        session_dir = self._get_session_dir(labels)
        meta = self._save_triple(frames, labels, session_dir)
        self._capture_count += 1
        self._img_count     += 1

        self.lbl_img_count.setText(f"Captures this session: {self._img_count}")
        self.lbl_last_saved.setText(f"Saved: {meta['capture_id']}")
        self.shared_log.log(
            "CAMERA", f"Captured {meta['capture_id']} (3 cameras)", "ok")

    # ── Auto capture ──────────────────────────────────────────

    def _auto_start(self):
        labels      = self._get_labels()
        session_dir = self._get_session_dir(labels)
        self._auto_count = 0
        self._auto_max   = self.spn_max_cap.value()
        interval_ms      = int(self.spn_interval.value() * 1000)

        def _do():
            if self._auto_count >= self._auto_max:
                self._auto_stop()
                self.shared_log.log(
                    "CAMERA",
                    f"Auto complete: {self._auto_count}/{self._auto_max}",
                    "ok")
                return

            frames = self.camera.get_frame_snapshot()
            if any(f is None for f in frames):
                return

            lbl = self._get_labels()
            lbl["auto_n"] = self._auto_count
            self._save_triple(frames, lbl, session_dir)

            self._capture_count += 1
            self._auto_count    += 1
            pct = int(self._auto_count / self._auto_max * 100)
            self.auto_progress.setValue(pct)
            self.auto_progress.setFormat(f"{self._auto_count}/{self._auto_max}")

        self._auto_timer = QTimer()
        self._auto_timer.timeout.connect(_do)
        self._auto_timer.start(interval_ms)

        self.btn_auto_start.setEnabled(False)
        self.btn_auto_pause.setEnabled(True)
        self.btn_auto_stop.setEnabled(True)
        self.shared_log.log(
            "CAMERA",
            f"Auto started: {self._auto_max} captures "
            f"@ {self.spn_interval.value()}s (all 3 cameras)", "info")

    def _auto_pause(self):
        if self._auto_timer:
            if self._auto_timer.isActive():
                self._auto_timer.stop()
                self.shared_log.log("CAMERA", "Auto paused", "info")
            else:
                self._auto_timer.start()
                self.shared_log.log("CAMERA", "Auto resumed", "info")

    def _auto_stop(self):
        if self._auto_timer:
            self._auto_timer.stop()
            self._auto_timer = None
        self.btn_auto_start.setEnabled(True)
        self.btn_auto_pause.setEnabled(False)
        self.btn_auto_stop.setEnabled(False)
        self.auto_progress.setValue(0)
        self.auto_progress.setFormat("Ready")

    # ── Misc ──────────────────────────────────────────────────

    def _refresh_device_status(self):
        connected = self.camera.is_acquiring
        self.lbl_device_status.setText(
            "Cameras: LIVE (3)" if connected else "Cameras: not started")

    def _on_disease_toggle(self, state):
        self.cmb_disease_type.setEnabled(bool(state))

    def _browse_folder(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select save folder", self.lbl_folder.text())
        if d:
            self.lbl_folder.setText(d)
            self.reset_session()

    def reset_session(self):
        self._session_path   = None
        self._session_labels = None

    def cleanup(self):
        self._auto_stop()
        if self._info_timer:
            self._info_timer.stop()
