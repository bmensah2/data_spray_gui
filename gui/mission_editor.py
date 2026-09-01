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

from gui.theme_manager import theme_manager

# -- Constants (mirror navigation_panel_rgb.py) -----------------
HUSKY_IP       = "192.168.131.1"
HUSKY_USER     = "administrator"
MISSIONS_HUSKY = "~/missions"

# -- Styles -------------------------------------------------------
# These used to be module-level string constants computed once at
# import time -- meaning they were permanently frozen to whichever
# theme happened to be active the first time this module was
# imported, and never reflected the theme active when a mission
# editor dialog was actually opened. Converted to functions that
# build fresh from theme_manager.palette() each time they're called
# (at dialog construction, and again via register_widget() for any
# widget that should also update live if the theme changes while
# the dialog is open).

def _editor_style(p):
    return (
        f"background:{p['bg0']};"
        f"color:{p['text_dim']};"
        f"font-family:'Noto Sans',Arial,sans-serif;"
        f"font-size:11px;"
        f"border:1px solid {p['border2']};"
        f"border-radius:4px;"
        f"selection-background-color:{p['btn_bg']};"
    )


def _dialog_style(p):
    return (
        f"QDialog{{background:{p['bg']};color:{p['text']};}}"
        f"QLabel{{color:{p['text']};font-family:'Noto Sans',Arial,sans-serif;"
        f"font-size:10px;}}"
        f"QPushButton{{font-family:'Noto Sans',Arial,sans-serif;font-size:11px;"
        f"border-radius:4px;padding:6px 14px;}}"
        f"QLineEdit{{background:{p['input_bg']};color:{p['text']};"
        f"border:1px solid {p['border2']};border-radius:4px;padding:4px;"
        f"font-family:'Noto Sans',Arial,sans-serif;font-size:11px;}}"
    )


def _help_html(p):
    return f"""
<b style='color:{p['blue']}'>YAML Mission Reference</b><br><br>
<b style='color:{p['muted']}'>Top-level fields:</b><br>
<span style='color:{p['text_dim']}'>description:</span>
<span style='color:{p['muted']}'> Short mission name</span><br>
<span style='color:{p['text_dim']}'>speed:</span>
<span style='color:{p['muted']}'> Robot speed m/s (0.05–0.5)</span><br>
<span style='color:{p['text_dim']}'>capture_interval:</span>
<span style='color:{p['muted']}'> Distance between captures (m)</span><br>
<span style='color:{p['text_dim']}'>return_home:</span>
<span style='color:{p['muted']}'> true or false</span><br><br>
<b style='color:{p['muted']}'>Step actions:</b><br>
<span style='color:{p['green']}'>forward</span> — drive forward N meters<br>
<span style='color:{p['green']}'>backward</span> — drive backward N meters<br>
<span style='color:{p['green']}'>left / right</span> — rotate N degrees<br>
<span style='color:{p['green']}'>capture_stop</span> — stop auto-capture<br>
<span style='color:{p['green']}'>return_home</span> — return to start<br><br>
<b style='color:{p['muted']}'>Step modes:</b><br>
<span style='color:{p['amber']}'>capture</span> — camera only, no spray<br>
<span style='color:{p['amber']}'>detect</span>  — camera + spray armed<br>
<span style='color:{p['amber']}'>navigate</span> — move only, no capture<br><br>
<b style='color:{p['muted']}'>Example:</b><br>
<code style='color:{p['text_dim']};background:{p['bg0']};display:block;padding:4px'>
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
        theme_manager.register_widget(self, _dialog_style)
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
        theme_manager.register_widget(
            splitter, lambda p: f"QSplitter::handle{{background:{p['border2']};}}")

        # LEFT — YAML editor
        left_w = QWidget()
        elay = QVBoxLayout(left_w)
        elay.setContentsMargins(0, 0, 0, 0)
        elay.setSpacing(4)
        elay.addWidget(QLabel("YAML Content:"))

        self.editor = QTextEdit()
        theme_manager.register_widget(self.editor, _editor_style)
        self.editor.setPlainText(content)
        self.editor.setTabStopDistance(16)
        self.editor.setAcceptRichText(False)
        self.editor.textChanged.connect(self._on_text_changed)
        elay.addWidget(self.editor, stretch=1)

        self.lbl_valid = QLabel("● Ready")
        theme_manager.register_widget(
            self.lbl_valid, lambda p: (
                f"color:{p['green']};font-family:'Noto Sans',Arial,sans-serif;"
                f"font-size:10px;"))
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
        theme_manager.register_button(btn_apply, "green")
        btn_apply.clicked.connect(self._apply_quick_fields)
        hlay.addWidget(btn_apply)

        hlay.addWidget(QLabel("─" * 28))

        # Help — HTML content itself depends on the palette (not just
        # CSS), so it's built once at dialog-open time rather than
        # wired to theme_manager.on_change(): that hook has no cleanup
        # mechanism (unlike register_widget/register_button, which use
        # weakrefs), so a callback registered per-dialog-open would
        # leak a reference to this widget forever after the dialog
        # closes. This dialog is short-lived; correct-at-open-time is
        # the right tradeoff here.
        help_lbl = QLabel(_help_html(theme_manager.palette()))
        help_lbl.setWordWrap(True)
        help_lbl.setAlignment(Qt.AlignTop)
        theme_manager.register_widget(
            help_lbl, lambda p: (
                f"color:{p['muted']};font-family:'Noto Sans',Arial,sans-serif;"
                f"font-size:10px;"
                f"background:{p['bg0']};border:1px solid {p['input_bg']};"
                f"border-radius:4px;padding:6px;"))

        scr = QScrollArea()
        scr.setWidget(help_lbl)
        scr.setWidgetResizable(True)
        theme_manager.register_widget(
            scr, lambda p: f"QScrollArea{{background:{p['bg0']};border:none;}}")
        hlay.addWidget(scr, stretch=1)
        splitter.addWidget(right_w)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        lay.addWidget(splitter, stretch=1)

        # ── Buttons ───────────────────────────────────────────
        btn_row = QHBoxLayout()

        def _btn(text, role, fn):
            b = QPushButton(text)
            theme_manager.register_button(b, role)
            b.clicked.connect(fn)
            return b

        btn_row.addWidget(_btn("✓ Validate",            "dim_blue", self._validate))
        btn_row.addWidget(_btn("💾 Save",               "green",    self._save))
        btn_row.addWidget(_btn("💾⇅ Save & Sync Husky", "blue",     self._save_and_sync))
        btn_row.addStretch()
        btn_close = QPushButton("✕ Close")
        theme_manager.register_widget(
            btn_close, lambda p: (
                f"QPushButton{{background:{p['bg2']};color:{p['muted']};"
                f"border:1px solid {p['border2']};border-radius:4px;"
                f"padding:6px 14px;font-family:'Noto Sans',Arial,sans-serif;"
                f"font-size:11px;}}"
                f"QPushButton:hover{{background:{p['btn_hover']};color:{p['text']};}}"))
        btn_close.clicked.connect(self.reject)
        btn_row.addWidget(btn_close)
        lay.addLayout(btn_row)

    # ── Helpers ───────────────────────────────────────────────

    def _on_text_changed(self):
        try:
            import yaml
            yaml.safe_load(self.editor.toPlainText())
            self.lbl_valid.setText("● YAML valid")
            theme_manager.register_widget(
                self.lbl_valid, lambda p: (
                    f"color:{p['green']};font-family:'Noto Sans',Arial,sans-serif;"
                    f"font-size:10px;"))
        except Exception as e:
            short = str(e)[:90]
            self.lbl_valid.setText(f"✗ {short}")
            theme_manager.register_widget(
                self.lbl_valid, lambda p: (
                    f"color:{p['red']};font-family:'Noto Sans',Arial,sans-serif;"
                    f"font-size:10px;"))

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
            theme_manager.register_widget(
                self.lbl_valid, lambda p: (
                    f"color:{p['green']};font-family:'Noto Sans',Arial,sans-serif;"
                    f"font-size:10px;"))
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
