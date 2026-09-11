"""
gui/session_metadata_dialog.py
ABEN Dual RGB Imaging System — Session Metadata Prompt

Shown when the operator ARMs detection, so every recorded session
carries real, operator-entered provenance instead of the hardcoded
config defaults (operator/researcher both defaulted to "nana", which
is useless in a publication record and actively misleading if someone
else ran the session).

Values are remembered between sessions (session_metadata.json) and
pre-filled, so a repeat run in the same field is one Enter press --
the prompt should not become friction that tempts anyone to skip it.
"""

import json
import logging
from pathlib import Path

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QLineEdit, QComboBox, QPushButton, QTextEdit,
)
from PyQt5.QtCore import Qt

from gui.theme_manager import theme_manager
from core.detection_config_rgb import GrowthStage

META_FILE = Path(__file__).resolve().parent.parent / "session_metadata.json"

# Free-text with suggestions rather than a locked dropdown -- a field
# season turns up crops and locations nobody predicted at build time.
CROP_SUGGESTIONS = [
    "sugarbeet", "wheat", "corn", "soybean", "sunflower", "canola", "other",
]


def load_last_metadata() -> dict:
    """Last-used values, or sensible blanks. Never raises."""
    defaults = {
        "operator":     "",
        "researcher":   "",
        "institution":  "NDSU",
        "field_id":     "",
        "location":     "",
        "crop":         "sugarbeet",
        "growth_stage": GrowthStage.SIX_LEAF.value,
        "notes":        "",
    }
    if not META_FILE.exists():
        return defaults
    try:
        with open(META_FILE) as f:
            saved = json.load(f)
        if isinstance(saved, dict):
            for k in defaults:
                if k in saved and isinstance(saved[k], str):
                    defaults[k] = saved[k]
    except Exception as e:
        logging.warning(f"session metadata: could not read {META_FILE} ({e})")
    return defaults


def save_last_metadata(meta: dict) -> bool:
    """Persist for pre-filling next session. Never raises."""
    try:
        with open(META_FILE, "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        return True
    except Exception as e:
        logging.warning(f"session metadata: could not write {META_FILE} ({e})")
        return False


class SessionMetadataDialog(QDialog):
    """
    Returns metadata via .metadata() after exec_() == QDialog.Accepted.

    Deliberately does NOT hard-block on empty fields: an operator
    mid-field with the robot running should never be unable to arm
    because a text box is empty. Operator name is the one field
    flagged inline when blank, since it's the whole point of the
    prompt -- but it's a nudge, not a lock.
    """

    def __init__(self, parent=None, detection_mode: str = "weed"):
        super().__init__(parent)
        self._mode = detection_mode
        self._meta = load_last_metadata()

        self.setWindowTitle("Session Details")
        self.setMinimumWidth(460)
        theme_manager.register_widget(
            self, lambda p: (
                f"QDialog{{background:{p['bg']};color:{p['text']};}}"
                f"QLabel{{color:{p['text']};"
                f"font-family:'Noto Sans',Arial,sans-serif;font-size:11px;}}"
                f"QLineEdit,QComboBox,QTextEdit{{background:{p['input_bg']};"
                f"color:{p['text']};border:1px solid {p['border2']};"
                f"border-radius:3px;padding:4px;"
                f"font-family:'Noto Sans',Arial,sans-serif;font-size:11px;}}"
                f"QComboBox QAbstractItemView{{background:{p['input_bg']};"
                f"color:{p['text']};"
                f"selection-background-color:{p['btn_bg']};}}"))
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        blurb = QLabel(
            "Recorded with every spray event and included in the session "
            "report. Pre-filled from your last session.")
        blurb.setWordWrap(True)
        theme_manager.register_widget(
            blurb, lambda p: (
                f"color:{p['muted']};font-size:10px;"
                f"font-family:'Noto Sans',Arial,sans-serif;"))
        lay.addWidget(blurb)

        grid = QGridLayout()
        grid.setSpacing(6)
        r = 0

        self.ed_session_name = QLineEdit("")
        self.ed_session_name.setPlaceholderText(
            "optional \u2014 e.g. row3_morning (left blank: session is "
            "identified by timestamp only)")
        self.ed_session_name.setToolTip(
            "A short, human-readable label folded into this session's "
            "folder/file names, so logs are easier to find later than "
            "a bare timestamp alone. Left blank each time deliberately "
            "(not remembered like the other fields) -- reusing the "
            "same name across sessions would make them harder to tell "
            "apart, not easier.")
        grid.addWidget(QLabel("Session Name"), r, 0)
        grid.addWidget(self.ed_session_name, r, 1); r += 1

        self.ed_operator = QLineEdit(self._meta["operator"])
        self.ed_operator.setPlaceholderText("who is running the robot today")
        grid.addWidget(QLabel("Operator *"), r, 0)
        grid.addWidget(self.ed_operator, r, 1); r += 1

        self.ed_researcher = QLineEdit(self._meta["researcher"])
        self.ed_researcher.setPlaceholderText(
            "whose research program this data belongs to")
        grid.addWidget(QLabel("Researcher"), r, 0)
        grid.addWidget(self.ed_researcher, r, 1); r += 1

        self.ed_institution = QLineEdit(self._meta["institution"])
        grid.addWidget(QLabel("Institution"), r, 0)
        grid.addWidget(self.ed_institution, r, 1); r += 1

        self.ed_field = QLineEdit(self._meta["field_id"])
        self.ed_field.setPlaceholderText("e.g. wilkin_plot_A")
        grid.addWidget(QLabel("Field ID"), r, 0)
        grid.addWidget(self.ed_field, r, 1); r += 1

        self.ed_location = QLineEdit(self._meta["location"])
        self.ed_location.setPlaceholderText("e.g. Wilkin County, MN")
        grid.addWidget(QLabel("Location"), r, 0)
        grid.addWidget(self.ed_location, r, 1); r += 1

        self.cmb_crop = QComboBox()
        self.cmb_crop.setEditable(True)
        self.cmb_crop.addItems(CROP_SUGGESTIONS)
        self.cmb_crop.setCurrentText(self._meta["crop"])
        grid.addWidget(QLabel("Crop"), r, 0)
        grid.addWidget(self.cmb_crop, r, 1); r += 1

        self.cmb_stage = QComboBox()
        for gs in GrowthStage:
            self.cmb_stage.addItem(gs.value)
        idx = self.cmb_stage.findText(self._meta["growth_stage"])
        if idx >= 0:
            self.cmb_stage.setCurrentIndex(idx)
        grid.addWidget(QLabel("Growth stage"), r, 0)
        grid.addWidget(self.cmb_stage, r, 1); r += 1

        lay.addLayout(grid)

        lay.addWidget(QLabel("Notes"))
        self.ed_notes = QTextEdit(self._meta["notes"])
        self.ed_notes.setPlaceholderText(
            "conditions, wind, anything worth recording for this run")
        self.ed_notes.setMaximumHeight(70)
        lay.addWidget(self.ed_notes)

        self.lbl_warn = QLabel("")
        theme_manager.register_widget(
            self.lbl_warn, lambda p: (
                f"color:{p['amber']};font-size:10px;"
                f"font-family:'Noto Sans',Arial,sans-serif;"))
        lay.addWidget(self.lbl_warn)

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        btn_cancel = QPushButton("Cancel")
        theme_manager.register_widget(
            btn_cancel, lambda p: (
                f"QPushButton{{background:{p['bg2']};color:{p['muted']};"
                f"border:1px solid {p['border2']};border-radius:4px;"
                f"padding:6px 14px;"
                f"font-family:'Noto Sans',Arial,sans-serif;font-size:11px;}}"
                f"QPushButton:hover{{background:{p['btn_hover']};"
                f"color:{p['text']};}}"))
        btn_cancel.setToolTip("Do not arm detection")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)

        self.btn_ok = QPushButton("✓  Start Session")
        theme_manager.register_button(self.btn_ok, "green")
        self.btn_ok.clicked.connect(self._accept)
        btn_row.addWidget(self.btn_ok)

        lay.addLayout(btn_row)
        self.ed_operator.setFocus()

    def _accept(self):
        if not self.ed_operator.text().strip():
            # Nudge once, then allow through on a second press -- never
            # block arming over a text field.
            if not self.lbl_warn.text():
                self.lbl_warn.setText(
                    "Operator name is blank — press again to continue anyway.")
                self.ed_operator.setFocus()
                return
        # session_name deliberately excluded from what gets persisted
        # for next time's pre-fill -- reusing the same name across
        # sessions would make folders harder to tell apart, not
        # easier (load_last_metadata()'s own defaults dict already
        # doesn't include this key, so it would never be read back
        # even if saved, but leaving it out of the file entirely is
        # cleaner than relying on that).
        to_persist = self.metadata()
        to_persist.pop("session_name", None)
        save_last_metadata(to_persist)
        self.accept()

    def metadata(self) -> dict:
        return {
            "session_name": self.ed_session_name.text().strip(),
            "operator":     self.ed_operator.text().strip(),
            "researcher":   self.ed_researcher.text().strip(),
            "institution":  self.ed_institution.text().strip(),
            "field_id":     self.ed_field.text().strip(),
            "location":     self.ed_location.text().strip(),
            "crop":         self.cmb_crop.currentText().strip(),
            "growth_stage": self.cmb_stage.currentText(),
            "notes":        self.ed_notes.toPlainText().strip(),
        }
