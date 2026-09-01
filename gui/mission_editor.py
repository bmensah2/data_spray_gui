"""
gui/mission_editor.py
ABEN Field Imaging System — Mission YAML Editor Dialog

Built-in editor for creating and editing mission YAML files.
Features:
  - Dark-theme monospace YAML editor
  - Live YAML validation on every keystroke
  - Full validation with step-by-step checks
  - Quick-edit fields (description, speed, interval)
  - Save to local missions/ folder
  - SCP sync directly to Husky

Usage:
    from gui.mission_editor import MissionEditorDialog
    dlg = MissionEditorDialog(parent, filename, content, ...)
    dlg.exec_()
"""

import subprocess
from pathlib import Path

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QLineEdit,
    QScrollArea, QSplitter, QMessageBox, QWidget
)
from PyQt5.QtCore import Qt

# ── Constants (mirror navigation_panel.py) ────────────────────
HUSKY_IP       = "192.168.131.1"
HUSKY_USER     = "administrator"
MISSIONS_HUSKY = "~/missions"

# ── Styles ────────────────────────────────────────────────────
EDITOR_STYLE = (
    "background:#0a0d14;"
    "color:#c8d0e0;"
    "font-family:'Courier New',monospace;"
    "font-size:11px;"
    "border:1px solid #2a2f3d;"
    "border-radius:4px;"
    "selection-background-color:#1e4060;"
)

DIALOG_STYLE = (
    "QDialog{background:#0f1117;color:#e8eaf0;}"
    "QLabel{color:#e8eaf0;font-family:'Courier New';font-size:10px;}"
    "QPushButton{font-family:'Courier New';font-size:11px;"
    "border-radius:4px;padding:6px 14px;}"
    "QLineEdit{background:#1a1e2e;color:#e8eaf0;"
    "border:1px solid #2a2f3d;border-radius:4px;padding:4px;"
    "font-family:'Courier New';font-size:11px;}"
)

HELP_HTML = """
<b style='color:#4a9eff'>YAML Mission Reference</b><br><br>
<b style='color:#8090a8'>Top-level fields:</b><br>
<span style='color:#c8d0e0'>description:</span>
<span style='color:#8090a8'> Short mission name</span><br>
<span style='color:#c8d0e0'>speed:</span>
<span style='color:#8090a8'> Robot speed m/s (0.05–0.5)</span><br>
<span style='color:#c8d0e0'>capture_interval:</span>
<span style='color:#8090a8'> Distance between captures (m)</span><br>
<span style='color:#c8d0e0'>return_home:</span>
<span style='color:#8090a8'> true or false</span><br><br>
<b style='color:#8090a8'>Step actions:</b><br>
<span style='color:#00c896'>forward</span> — drive forward N meters<br>
<span style='color:#00c896'>backward</span> — drive backward N meters<br>
<span style='color:#00c896'>left / right</span> — rotate N degrees<br>
<span style='color:#00c896'>capture_stop</span> — stop auto-capture<br>
<span style='color:#00c896'>return_home</span> — return to start<br><br>
<b style='color:#8090a8'>Step modes:</b><br>
<span style='color:#f5a623'>capture</span> — camera only, no spray<br>
<span style='color:#f5a623'>detect</span>  — camera + spray armed<br>
<span style='color:#f5a623'>navigate</span> — move only, no capture<br><br>
<b style='color:#8090a8'>Example:</b><br>
<code style='color:#b0c8b0;background:#0a0d14;display:block;padding:4px'>
description: 'Weed plot'<br>
speed: 0.1<br>
capture_interval: 0.5<br>
return_home: true<br>
steps:<br>
&nbsp;&nbsp;- action: forward<br>
&nbsp;&nbsp;&nbsp;&nbsp;distance: 5.0<br>
&nbsp;&nbsp;&nbsp;&nbsp;mode: capture<br>
&nbsp;&nbsp;- action: capture_stop<br>
&nbsp;&nbsp;- action: forward<br>
&nbsp;&nbsp;&nbsp;&nbsp;distance: 5.0<br>
&nbsp;&nbsp;&nbsp;&nbsp;mode: detect<br>
&nbsp;&nbsp;- action: return_home
</code>
"""

TEMPLATE_YAML = """\
# ABEN Mission File
description: 'New mission'
speed: 0.1            # m/s — field speed
capture_interval: 0.5 # meters between captures (20in)
return_home: true
steps:
  - action: forward
    distance: 5.0     # 197in — data collection zone
    mode: capture     # camera only, no spray
  - action: capture_stop
  - action: forward
    distance: 5.0     # 197in — spray testing zone
    mode: detect      # camera + spray armed
  - action: return_home
"""


class MissionEditorDialog(QDialog):
    """Built-in YAML mission editor with live validation and Husky sync."""

    def __init__(self, parent, filename, content,
                 save_path=None, missions_dir=None,
                 log_fn=None, refresh_fn=None, sync_fn=None):
        super().__init__(parent)
        self._save_path    = save_path
        self._missions_dir = missions_dir or Path("missions")
        self._log          = log_fn     or (lambda m, t="ok": None)
        self._refresh      = refresh_fn or (lambda: None)
        self._sync_panel   = sync_fn    or (lambda: None)

        self.setWindowTitle(f"Mission Editor — {filename}")
        self.setMinimumSize(860, 560)
        self.resize(960, 640)
        self.setStyleSheet(DIALOG_STYLE)
        self._build_ui(filename, content)

    def _build_ui(self, filename, content):
        lay = QVBoxLayout(self)
        lay.setSpacing(8)
        lay.setContentsMargins(10, 10, 10, 10)

        # ── Filename ──────────────────────────────────────────
        fn_row = QHBoxLayout()
        fn_row.addWidget(QLabel("Filename:"))
        self.txt_filename = QLineEdit(filename)
        self.txt_filename.setToolTip(
            "Change to save as a new mission file")
        fn_row.addWidget(self.txt_filename, stretch=1)
        lay.addLayout(fn_row)

        # ── Splitter: editor (left) | help+quickfields (right) 
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)
        splitter.setStyleSheet(
            "QSplitter::handle{background:#2a2f3d;}")

        # LEFT — YAML editor
        left_w = QWidget()
        elay = QVBoxLayout(left_w)
        elay.setContentsMargins(0, 0, 0, 0)
        elay.setSpacing(4)
        elay.addWidget(QLabel("YAML Content:"))

        self.editor = QTextEdit()
        self.editor.setStyleSheet(EDITOR_STYLE)
        self.editor.setPlainText(content)
        self.editor.setTabStopDistance(16)
        self.editor.setAcceptRichText(False)
        self.editor.textChanged.connect(self._on_text_changed)
        elay.addWidget(self.editor, stretch=1)

        self.lbl_valid = QLabel("● Ready")
        self.lbl_valid.setStyleSheet(
            "color:#00c896;font-family:'Courier New';font-size:10px;")
        elay.addWidget(self.lbl_valid)
        splitter.addWidget(left_w)

        # RIGHT — quick fields + help
        right_w = QWidget()
        right_w.setMinimumWidth(240)
        right_w.setMaximumWidth(340)
        hlay = QVBoxLayout(right_w)
        hlay.setContentsMargins(6, 0, 0, 0)
        hlay.setSpacing(6)

        hlay.addWidget(QLabel("Quick Edit:"))

        def field_row(label, val=""):
            row = QHBoxLayout()
            lbl = QLabel(f"{label}:")
            lbl.setFixedWidth(130)
            inp = QLineEdit(str(val))
            inp.setFixedHeight(24)
            row.addWidget(lbl)
            row.addWidget(inp, stretch=1)
            return row, inp

        # Parse existing YAML for quick fields
        try:
            import yaml
            parsed = yaml.safe_load(self.editor.toPlainText()) or {}
        except Exception:
            parsed = {}

        r, self.qf_desc  = field_row(
            "description", parsed.get("description", ""))
        hlay.addLayout(r)
        r, self.qf_speed = field_row(
            "speed (m/s)", parsed.get("speed", "0.1"))
        hlay.addLayout(r)
        r, self.qf_intv  = field_row(
            "capture_interval (m)", parsed.get("capture_interval", "0.5"))
        hlay.addLayout(r)

        btn_apply = QPushButton("Apply quick fields to YAML")
        btn_apply.setStyleSheet(
            "QPushButton{background:#1a2a1a;color:#60c060;"
            "border:1px solid #2a5a2a;padding:4px;}"
            "QPushButton:hover{background:#253525;}")
        btn_apply.clicked.connect(self._apply_quick_fields)
        hlay.addWidget(btn_apply)

        hlay.addWidget(QLabel("─" * 28))

        # Help
        help_lbl = QLabel(HELP_HTML)
        help_lbl.setWordWrap(True)
        help_lbl.setAlignment(Qt.AlignTop)
        help_lbl.setStyleSheet(
            "color:#8090a8;font-family:'Courier New';font-size:10px;"
            "background:#0a0d14;border:1px solid #1a1e2e;"
            "border-radius:4px;padding:6px;")
        scr = QScrollArea()
        scr.setWidget(help_lbl)
        scr.setWidgetResizable(True)
        scr.setStyleSheet("QScrollArea{background:#0a0d14;border:none;}")
        hlay.addWidget(scr, stretch=1)
        splitter.addWidget(right_w)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        lay.addWidget(splitter, stretch=1)

        # ── Buttons ───────────────────────────────────────────
        btn_row = QHBoxLayout()

        def _btn(text, style, fn):
            b = QPushButton(text)
            b.setStyleSheet(style)
            b.clicked.connect(fn)
            return b

        S_VALIDATE = ("QPushButton{background:#1a2030;color:#8090c0;"
                      "border:1px solid #2a3050;}"
                      "QPushButton:hover{background:#2a3040;}")
        S_SAVE     = ("QPushButton{background:#1a2a1a;color:#60c060;"
                      "border:1px solid #2a5a2a;}"
                      "QPushButton:hover{background:#253525;}")
        S_SYNC     = ("QPushButton{background:#1a2a3a;color:#60a0d0;"
                      "border:1px solid #2a4a6a;}"
                      "QPushButton:hover{background:#253545;}")
        S_CLOSE    = ("QPushButton{background:#1a1a1a;color:#8090a8;"
                      "border:1px solid #2a2f3d;}"
                      "QPushButton:hover{background:#252525;}")

        btn_row.addWidget(_btn("✓ Validate",            S_VALIDATE, self._validate))
        btn_row.addWidget(_btn("💾 Save",               S_SAVE,     self._save))
        btn_row.addWidget(_btn("💾⇅ Save & Sync Husky", S_SYNC,     self._save_and_sync))
        btn_row.addStretch()
        btn_row.addWidget(_btn("✕ Close",               S_CLOSE,    self.reject))
        lay.addLayout(btn_row)

    # ── Helpers ───────────────────────────────────────────────

    def _on_text_changed(self):
        try:
            import yaml
            yaml.safe_load(self.editor.toPlainText())
            self.lbl_valid.setText("● YAML valid")
            self.lbl_valid.setStyleSheet(
                "color:#00c896;font-family:'Courier New';font-size:10px;")
        except Exception as e:
            short = str(e)[:90]
            self.lbl_valid.setText(f"✗ {short}")
            self.lbl_valid.setStyleSheet(
                "color:#e84545;font-family:'Courier New';font-size:10px;")

    def _validate(self):
        try:
            import yaml
            data = yaml.safe_load(self.editor.toPlainText())
            errors = []
            if not isinstance(data, dict):
                errors.append("Root must be a YAML mapping")
            else:
                for key in ("description", "steps"):
                    if key not in data:
                        errors.append(f"Missing required field: {key}")
                steps = data.get("steps", [])
                if isinstance(steps, list):
                    for i, step in enumerate(steps):
                        if not isinstance(step, dict):
                            errors.append(f"Step {i+1}: must be a mapping")
                        elif "action" not in step:
                            errors.append(f"Step {i+1}: missing 'action'")
                spd = data.get("speed", 0.1)
                if not isinstance(spd, (int, float)) or spd <= 0:
                    errors.append("speed must be a positive number")
            if errors:
                QMessageBox.warning(self, "Validation Issues",
                    "Issues found:\n\n• " + "\n• ".join(errors))
            else:
                n = len(data.get("steps", []))
                QMessageBox.information(self, "Valid ✓",
                    f"YAML is valid!\n\n"
                    f"Description: {data.get('description', '—')}\n"
                    f"Speed:       {data.get('speed', '—')} m/s\n"
                    f"Interval:    {data.get('capture_interval', '—')} m\n"
                    f"Steps:       {n}\n"
                    f"Return home: {data.get('return_home', True)}")
        except Exception as e:
            QMessageBox.critical(self, "YAML Error", str(e))

    def _apply_quick_fields(self):
        try:
            import yaml
            data = yaml.safe_load(self.editor.toPlainText()) or {}
            desc = self.qf_desc.text().strip()
            spd  = self.qf_speed.text().strip()
            intv = self.qf_intv.text().strip()
            if desc:
                data["description"] = desc
            if spd:
                try:
                    data["speed"] = float(spd)
                except ValueError:
                    pass
            if intv:
                try:
                    data["capture_interval"] = float(intv)
                except ValueError:
                    pass
            new_yaml = yaml.dump(data, default_flow_style=False,
                                 sort_keys=False, allow_unicode=True)
            self.editor.setPlainText(new_yaml)
        except Exception as e:
            QMessageBox.critical(self, "Apply Error", str(e))

    def _get_path(self):
        name = self.txt_filename.text().strip()
        if not name.endswith((".yaml", ".yml")):
            name += ".yaml"
        self._missions_dir.mkdir(parents=True, exist_ok=True)
        return self._missions_dir / name, name

    def _save(self) -> bool:
        try:
            import yaml
            yaml.safe_load(self.editor.toPlainText())
        except Exception as e:
            QMessageBox.critical(self, "YAML Error",
                f"Cannot save — fix YAML errors first:\n{e}")
            return False
        try:
            path, name = self._get_path()
            with open(path, "w") as f:
                f.write(self.editor.toPlainText())
            self._save_path = path
            self._log(f"Mission saved: {name}", "ok")
            self._refresh()
            self.lbl_valid.setText(f"● Saved: {name}")
            self.lbl_valid.setStyleSheet(
                "color:#00c896;font-family:'Courier New';font-size:10px;")
            return True
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))
            return False

    def _save_and_sync(self):
        if not self._save():
            return
        path, name = self._get_path()
        try:
            result = subprocess.run([
                "scp", "-o", "StrictHostKeyChecking=no",
                str(path),
                f"{HUSKY_USER}@{HUSKY_IP}:{MISSIONS_HUSKY}/{name}"
            ], capture_output=True, text=True, timeout=15)
            if result.returncode == 0:
                self._log(f"Synced to Husky: {name}", "ok")
                QMessageBox.information(self, "Sync Complete",
                    f"'{name}' saved and synced to Husky.\n"
                    f"Path on Husky: {MISSIONS_HUSKY}/{name}")
            else:
                err = (result.stderr or "SSH/SCP error").strip()
                self._log(f"Sync failed: {err}", "error")
                QMessageBox.critical(self, "Sync Failed",
                    f"SCP failed:\n{err}\n\n"
                    "Mission was saved locally.\n"
                    "Use the Sync button in the nav panel to retry.")
        except subprocess.TimeoutExpired:
            self._log("Sync timed out — check Husky connection", "warn")
            QMessageBox.warning(self, "Sync Timeout",
                "SCP timed out.\nCheck that Husky is reachable at 192.168.131.1.\n"
                "Mission was saved locally.")
        except Exception as e:
            self._log(f"Sync error: {e}", "error")
            QMessageBox.critical(self, "Sync Error", str(e))
