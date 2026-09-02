"""
gui/keyboard_nav.py
ABEN Field Imaging System — Keyboard Navigation Controller

Arrow key → Husky movement (hold = move, release = stop).
Installed on MainWindow — active from any tab.

Keys:
  ↑  Forward      ↓  Backward
  ←  Turn left    →  Turn right
  Space           Emergency stop

Architecture:
  keyPressEvent  → start SSH nav command (non-blocking)
  keyReleaseEvent → kill process + send zero velocity
  autoRepeat filtered → key held = one command, not hundreds

Usage:
    from gui.keyboard_nav import KeyboardNav
    self.kb_nav = KeyboardNav(shared_log)
    self.kb_nav.install(main_window)   # installs key handlers
    header_layout.addWidget(self.kb_nav.widget())
"""

import subprocess
import threading
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QPushButton
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QKeyEvent

from gui.style import LED, _muted
from gui.theme_manager import theme_manager, _lighten, _darken

# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────
HUSKY_IP    = "192.168.131.1"
HUSKY_USER  = "administrator"
ROS_SOURCE  = "source /opt/ros/noetic/setup.bash && "
CMD_VEL     = "/joy_teleop/cmd_vel"

LINEAR_SPEED  = 0.1   # m/s  forward / backward
ANGULAR_SPEED = 0.1   # rad/s  left / right

# Arrow key Qt codes
KEY_UP    = Qt.Key_Up
KEY_DOWN  = Qt.Key_Down
KEY_LEFT  = Qt.Key_Left
KEY_RIGHT = Qt.Key_Right
KEY_SPACE = Qt.Key_Space


# ─────────────────────────────────────────────────────────────
#  D-PAD WIDGET
# ─────────────────────────────────────────────────────────────
class _DPadWidget(QWidget):
    """
    D-pad visual widget.
    Cross shape with:
      ↑ top arm    = forward
      ↓ bottom arm = backward
      ↺ left arm   = turn left (rotation symbol)
      ↻ right arm  = turn right (rotation symbol)
    Center circle = idle indicator.
    Active direction lights up orange (matching the icon style).
    """

    def __init__(self, nav_controller, parent=None):
        super().__init__(parent)
        self.nav   = nav_controller
        self._active = None   # "up","down","left","right" or None
        self.setMouseTracking(True)

    def set_active(self, direction):
        self._active = direction
        self.update()

    def paintEvent(self, event):
        from PyQt5.QtGui import (
            QPainter, QColor, QPen, QBrush,
            QPainterPath, QFont
        )
        from PyQt5.QtCore import QRectF, QPointF

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        W = self.width()
        H = self.height()
        cx = W / 2
        cy = H / 2

        # Arm dimensions
        ARM  = W * 0.28   # arm half-width
        REACH = W * 0.48  # arm reach from center

        # Colors
        pal = theme_manager.palette()
        C_IDLE   = QColor(pal["btn_bg"])
        C_ACTIVE = QColor(pal["amber"])
        C_RING   = QColor(pal["border"])
        C_ARROW  = QColor(pal["text"])
        C_BG     = QColor(pal["bg0"])

        # Background circle
        painter.setPen(QPen(C_RING, 1.2))
        painter.setBrush(QBrush(C_BG))
        painter.drawEllipse(QRectF(0.5, 0.5, W-1, H-1))

        def arm_color(name):
            return C_ACTIVE if self._active == name else C_IDLE

        # ── Draw 4 arms ───────────────────────────────────────
        # Each arm is a rounded rectangle
        r = ARM * 0.35   # corner radius

        def draw_arm(x, y, w, h, name):
            painter.setBrush(QBrush(arm_color(name)))
            painter.setPen(Qt.NoPen)
            path = QPainterPath()
            path.addRoundedRect(QRectF(x, y, w, h), r, r)
            painter.fillPath(path, arm_color(name))

        # UP arm
        draw_arm(cx - ARM, cy - REACH, ARM*2, REACH - ARM*0.6, "up")
        # DOWN arm
        draw_arm(cx - ARM, cy + ARM*0.6, ARM*2, REACH - ARM*0.6, "down")
        # LEFT arm
        draw_arm(cx - REACH, cy - ARM, REACH - ARM*0.6, ARM*2, "left")
        # RIGHT arm
        draw_arm(cx + ARM*0.6, cy - ARM, REACH - ARM*0.6, ARM*2, "right")

        # ── Center circle ─────────────────────────────────────
        cr = ARM * 0.95
        painter.setBrush(QBrush(C_ACTIVE if self._active else C_IDLE))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QRectF(cx-cr, cy-cr, cr*2, cr*2))

        # Center dot
        painter.setBrush(QBrush(QColor(pal["text"])))
        dr = cr * 0.42
        painter.drawEllipse(QRectF(cx-dr, cy-dr, dr*2, dr*2))

        # ── Arrow symbols ─────────────────────────────────────
        painter.setPen(QPen(C_ARROW, 1.5))
        font = QFont("Arial", max(int(W * 0.18), 6), QFont.Bold)
        painter.setFont(font)

        sym_off = REACH * 0.52

        # UP — chevron ∧
        self._draw_chevron(painter, cx, cy - sym_off, W*0.12, "up")
        # DOWN — chevron ∨
        self._draw_chevron(painter, cx, cy + sym_off, W*0.12, "down")
        # LEFT — rotation ↺
        self._draw_rotation(painter, cx - sym_off, cy, W*0.13, "left")
        # RIGHT — rotation ↻
        self._draw_rotation(painter, cx + sym_off, cy, W*0.13, "right")

        painter.end()

    def _draw_chevron(self, p, cx, cy, size, direction):
        """Draw ∧ or ∨ chevron."""
        from PyQt5.QtGui import QPen, QColor, QPolygonF
        from PyQt5.QtCore import QPointF
        pal = theme_manager.palette()
        p.setPen(QPen(QColor(pal["text"]), max(1.5, size*0.18),
                      Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        s = size
        if direction == "up":
            pts = [QPointF(cx-s, cy+s*0.5),
                   QPointF(cx,   cy-s*0.5),
                   QPointF(cx+s, cy+s*0.5)]
        else:
            pts = [QPointF(cx-s, cy-s*0.5),
                   QPointF(cx,   cy+s*0.5),
                   QPointF(cx+s, cy-s*0.5)]
        p.drawPolyline(QPolygonF(pts))

    def _draw_rotation(self, p, cx, cy, size, direction):
        """Draw ↺ (left) or ↻ (right) rotation arc with arrow."""
        from PyQt5.QtGui import QPen, QColor
        from PyQt5.QtCore import QRectF
        import math
        pw = max(1.5, size * 0.18)
        pal = theme_manager.palette()
        p.setPen(QPen(QColor(pal["text"]), pw,
                      Qt.SolidLine, Qt.RoundCap))
        r = size * 0.72
        rect = QRectF(cx-r, cy-r, r*2, r*2)

        if direction == "left":
            # Arc from ~30° to ~300° (counterclockwise appearance)
            p.drawArc(rect, 30*16, 240*16)
            # Arrow tip at start of arc (pointing counterclockwise)
            angle = math.radians(30)
            ax = cx + r * math.cos(angle)
            ay = cy - r * math.sin(angle)
            self._draw_arrowhead(p, ax, ay, -60, size*0.25)
        else:
            # Arc from ~150° to ~-90° (clockwise appearance)
            p.drawArc(rect, 150*16, -240*16)
            # Arrow tip pointing clockwise
            angle = math.radians(150)
            ax = cx + r * math.cos(angle)
            ay = cy - r * math.sin(angle)
            self._draw_arrowhead(p, ax, ay, 120, size*0.25)

    def _draw_arrowhead(self, p, x, y, angle_deg, size):
        """Draw small arrowhead."""
        from PyQt5.QtGui import QPolygonF, QPen, QBrush, QColor
        from PyQt5.QtCore import QPointF
        import math
        a = math.radians(angle_deg)
        cos_a, sin_a = math.cos(a), math.sin(a)
        pts = [
            QPointF(x, y),
            QPointF(x - size*(cos_a - sin_a*0.5),
                    y - size*(sin_a + cos_a*0.5)),
            QPointF(x - size*(cos_a + sin_a*0.5),
                    y - size*(sin_a - cos_a*0.5)),
        ]
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(theme_manager.palette()["text"])))
        p.drawPolygon(QPolygonF(pts))

    def mousePressEvent(self, event):
        """Click D-pad arms to move."""
        direction = self._hit_test(event.x(), event.y())
        if direction and self.nav._enabled:
            key_map = {"up": KEY_UP, "down": KEY_DOWN,
                       "left": KEY_LEFT, "right": KEY_RIGHT}
            key = key_map.get(direction)
            if key:
                self.nav._active_key = key
                self.nav._start_movement(key)

    def mouseReleaseEvent(self, event):
        if self.nav._active_key is not None:
            self.nav._stop_movement()

    def _hit_test(self, x, y):
        """Determine which arm was clicked."""
        cx, cy = self.width()/2, self.height()/2
        dx, dy = x - cx, y - cy
        # Dead zone at center
        if abs(dx) < self.width()*0.2 and abs(dy) < self.height()*0.2:
            return None
        if abs(dy) > abs(dx):
            return "up" if dy < 0 else "down"
        else:
            return "left" if dx < 0 else "right"


# ─────────────────────────────────────────────────────────────
#  KEYBOARD NAV
# ─────────────────────────────────────────────────────────────
class KeyboardNav(QObject):
    """
    Keyboard navigation controller.
    Install on MainWindow to intercept arrow keys globally.
    """

    status_changed = pyqtSignal(str, str)  # direction, state

    def __init__(self, shared_log=None):
        super().__init__()
        self.shared_log   = shared_log
        self._active_key  = None      # currently held key
        self._nav_proc    = None      # SSH subprocess
        self._enabled     = True      # can disable during detection
        self._widget      = None
        self._dir_labels  = {}
        self._build_widget()

    # ── Widget ────────────────────────────────────────────────

    def _build_widget(self) -> QWidget:
        """
        Build D-pad style widget:
          ↑ = forward, ↓ = backward
          ↺ = turn left, ↻ = turn right
        Clickable with mouse AND keyboard arrows.
        """
        from PyQt5.QtWidgets import QGridLayout, QSizePolicy
        from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QPolygonF
        from PyQt5.QtCore import QPointF

        outer = QWidget()
        outer.setFixedHeight(72)
        outer_lay = QHBoxLayout(outer)
        outer_lay.setContentsMargins(4, 4, 4, 4)
        outer_lay.setSpacing(6)

        # ── D-pad SVG widget ──────────────────────────────────
        dpad = _DPadWidget(self)
        dpad.setFixedSize(64, 64)
        self._dpad = dpad
        outer_lay.addWidget(dpad)

        # ── Right side: speed + enable ─────────────────────── 
        from PyQt5.QtWidgets import QVBoxLayout as _VBox
        right = QWidget()
        rv = _VBox(right)
        rv.setContentsMargins(2, 0, 2, 0)
        rv.setSpacing(4)

        # Speed label
        spd = QLabel(
            f"fwd {LINEAR_SPEED:.1f}m/s\n"
            f"rot {ANGULAR_SPEED:.1f}r/s")
        theme_manager.register_widget(
            spd, lambda p: (
                f"color:{p['dim']};font-size:9px;"
                f"font-family:'Noto Sans',Arial,sans-serif;"))
        self._speed_lbl = spd
        rv.addWidget(spd)

        # Enable/disable toggle
        self._btn_enable = QPushButton("KB ON")
        self._btn_enable.setFixedHeight(24)
        self._btn_enable.setCheckable(True)
        self._btn_enable.setChecked(True)
        self._apply_enable_style(True)
        self._btn_enable.clicked.connect(self._toggle_enable)
        rv.addWidget(self._btn_enable)

        # Space = stop hint
        hint = QLabel("SPC=stop")
        theme_manager.register_widget(
            hint, lambda p: (
                f"color:{p['dim']};font-size:8px;"
                f"font-family:'Noto Sans',Arial,sans-serif;"))
        rv.addWidget(hint)
        outer_lay.addWidget(right)

        # Repaint D-pad on theme switch
        theme_manager.on_change(self._dpad.update)

        # Keep dir_labels dict for API compatibility (unused visually now)
        for name in ["up","down","left","right"]:
            self._dir_labels[name] = QLabel()

        self._widget = outer
        return outer

    def widget(self) -> QWidget:
        return self._widget

    # ── Install on main window ────────────────────────────────

    def install(self, main_window):
        """
        Hook key events on the main window.
        Overrides keyPressEvent and keyReleaseEvent.
        """
        self._main_window = main_window
        orig_press   = main_window.keyPressEvent
        orig_release = main_window.keyReleaseEvent

        def _press(event: QKeyEvent):
            if not self._handle_press(event):
                orig_press(event)

        def _release(event: QKeyEvent):
            if not self._handle_release(event):
                orig_release(event)

        main_window.keyPressEvent   = _press
        main_window.keyReleaseEvent = _release
        # Ensure main window receives key events
        main_window.setFocusPolicy(Qt.StrongFocus)

    # ── Key handlers ──────────────────────────────────────────

    def _handle_press(self, event: QKeyEvent) -> bool:
        """Returns True if we consumed the event."""
        if not self._enabled:
            return False
        key = event.key()
        if key not in (KEY_UP, KEY_DOWN, KEY_LEFT,
                       KEY_RIGHT, KEY_SPACE):
            return False

        # Filter autoRepeat — holding key fires one command only
        if event.isAutoRepeat():
            return True   # consume but don't restart

        # Space = emergency stop
        if key == KEY_SPACE:
            self._emergency_stop()
            return True

        # Already moving — same key held, ignore
        if self._active_key == key:
            return True

        # New key — stop previous, start new
        if self._active_key is not None:
            self._stop_movement()

        self._active_key = key
        self._start_movement(key)
        return True

    def _handle_release(self, event: QKeyEvent) -> bool:
        """Returns True if we consumed the event."""
        if event.isAutoRepeat():
            return True
        key = event.key()
        if key not in (KEY_UP, KEY_DOWN, KEY_LEFT,
                       KEY_RIGHT, KEY_SPACE):
            return False
        if key == self._active_key:
            self._stop_movement()
        return True

    # ── Movement ──────────────────────────────────────────────

    def _start_movement(self, key):
        """Start continuous movement in direction."""
        direction, lin, ang = self._key_to_command(key)
        self._update_indicator(direction, active=True)

        # Publish continuous velocity via SSH python inline
        # Tagged "aben_kb_nav" so pkill can find it reliably
        cmd = (
            f"python3 -c 'import sys; sys.argv[0]=\"aben_kb_nav\";"
            f"import rospy,time;"
            f"from geometry_msgs.msg import Twist;"
            f"rospy.init_node(\"kb_nav\",anonymous=True);"
            f"p=rospy.Publisher(\"{CMD_VEL}\",Twist,queue_size=1);"
            f"t=Twist();"
            f"t.linear.x={lin};"
            f"t.angular.z={ang};"
            f"rate=rospy.Rate(10);"
            f"[p.publish(t) or rate.sleep() for _ in range(500)]'"
        )

        def _run():
            try:
                self._nav_proc = subprocess.Popen(
                    ['ssh', '-o', 'StrictHostKeyChecking=no',
                     f'{HUSKY_USER}@{HUSKY_IP}',
                     f'{ROS_SOURCE}{cmd}'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except Exception as e:
                if self.shared_log:
                    self.shared_log.log(
                        "NAV", f"KB nav error: {e}", "error")

        threading.Thread(target=_run, daemon=True).start()

        if self.shared_log:
            self.shared_log.log(
                "NAV", f"KB: {direction}", "info")

    def _stop_movement(self):
        """Kill SSH process + send zero velocity."""
        key = self._active_key
        if key:
            direction, _, _ = self._key_to_command(key)
            self._update_indicator(direction, active=False)
        self._active_key = None

        # Kill the running nav process
        if self._nav_proc and self._nav_proc.poll() is None:
            self._nav_proc.terminate()
            self._nav_proc = None

        # Kill remote process + send zero velocity
        def _send_stop():
            try:
                subprocess.run(
                    ['ssh', '-o', 'StrictHostKeyChecking=no',
                     '-o', 'ConnectTimeout=2',
                     f'{HUSKY_USER}@{HUSKY_IP}',
                     f'{ROS_SOURCE}'
                     # Kill the remote aben_kb_nav process first
                     f'pkill -f aben_kb_nav 2>/dev/null; '
                     f'sleep 0.1; '
                     # Then send zero velocity to guarantee stop
                     f'python3 -c \''
                     f'import rospy,time;'
                     f'from geometry_msgs.msg import Twist;'
                     f'rospy.init_node("kb_stop",anonymous=True);'
                     f'p=rospy.Publisher("{CMD_VEL}",Twist,queue_size=1);'
                     f'time.sleep(0.15);'
                     f'[p.publish(Twist()) or time.sleep(0.05)'
                     f' for _ in range(5)]\''
                    ],
                    timeout=4,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except Exception:
                pass
        threading.Thread(target=_send_stop, daemon=True).start()

    def _emergency_stop(self):
        """Space bar — immediate stop."""
        self._active_key = None
        if self._nav_proc and self._nav_proc.poll() is None:
            self._nav_proc.terminate()
            self._nav_proc = None
        for name in self._dir_labels:
            self._update_indicator(name, active=False)
        threading.Thread(target=self._send_estop, daemon=True).start()
        if self.shared_log:
            self.shared_log.log("NAV", "KB: SPACE — E-STOP", "warn")

    def _send_estop(self):
        try:
            subprocess.run(
                ['ssh', '-o', 'StrictHostKeyChecking=no',
                 f'{HUSKY_USER}@{HUSKY_IP}',
                 f'{ROS_SOURCE}'
                 f'pkill -SIGINT -f test_nav_v3.py 2>/dev/null;'
                 f'pkill -SIGINT -f field_nav.py  2>/dev/null;'
                 f'pkill -f aben_kb_nav 2>/dev/null;'
                 f'sleep 0.2;'
                 f'python3 -c \''
                 f'import rospy,time;'
                 f'from geometry_msgs.msg import Twist;'
                 f'rospy.init_node("estop",anonymous=True);'
                 f'p=rospy.Publisher("{CMD_VEL}",Twist,queue_size=1);'
                 f'time.sleep(0.2);'
                 f'[p.publish(Twist()) or time.sleep(0.05)'
                 f' for _ in range(5)]\''
                ],
                timeout=5,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception:
            pass

    # ── Helpers ───────────────────────────────────────────────

    def _key_to_command(self, key):
        """Map key to (direction_name, linear, angular)."""
        return {
            KEY_UP:    ("up",    LINEAR_SPEED,  0.0),
            KEY_DOWN:  ("down", -LINEAR_SPEED,  0.0),
            KEY_LEFT:  ("left",  0.0,           ANGULAR_SPEED),
            KEY_RIGHT: ("right", 0.0,          -ANGULAR_SPEED),
        }.get(key, ("none", 0.0, 0.0))

    def _update_indicator(self, direction: str, active: bool):
        """Update the D-pad visual state."""
        if hasattr(self, '_dpad'):
            if active:
                self._dpad.set_active(direction)
            else:
                self._dpad.set_active(None)

    def _toggle_enable(self, checked: bool):
        """Enable/disable keyboard nav."""
        self._enabled = checked
        if not checked and self._active_key:
            self._stop_movement()
        self._apply_enable_style(checked)
        if self.shared_log:
            self.shared_log.log(
                "NAV",
                f"Keyboard nav {'enabled' if checked else 'disabled'}",
                "info")

    def _apply_enable_style(self, checked: bool):
        def _style(p):
            if checked:
                bg, fg, bdr = _darken(p["green"], 55), p["green"], _darken(p["green"], 30)
                hover = _darken(p["green"], 40)
            else:
                bg, fg, bdr = p["disabled_bg"], p["dim"], p["border2"]
                hover = p["disabled_bg"]
            return (
                f"QPushButton{{background:{bg};color:{fg};"
                f"border:1px solid {bdr};border-radius:3px;"
                f"font-size:9px;font-weight:bold;}}"
                f"QPushButton:hover{{background:{hover};}}")
        theme_manager.register_widget(self._btn_enable, _style)

    # ── Public API ────────────────────────────────────────────

    def set_enabled(self, enabled: bool):
        """Enable or disable from external code."""
        self._enabled = enabled
        self._btn_enable.setChecked(enabled)
        self._toggle_enable(enabled)

    @property
    def is_active(self) -> bool:
        return self._active_key is not None

    def cleanup(self):
        if self._active_key:
            self._stop_movement()