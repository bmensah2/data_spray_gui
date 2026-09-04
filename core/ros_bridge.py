#!/usr/bin/env python3
"""
ros_bridge.py
Field Detection System — ROS Bridge (Jetson Side)

Runs ON THE JETSON AGX ORIN — no ROS installation required.

Receives UDP packets from husky_odom_pub.py running on the Husky PC
and makes the latest pose, estop state, and connection status
available to the rest of the detection pipeline.

Used by:
  - capture_tool.py    → tags every frame with robot position
  - event_logger.py    → logs pose at each spray event
  - actuation_controller.py → checks estop before firing nozzles

Author : Bright Mensah | NDSU / Imaging System
Runs on: Jetson AGX Orin 64GB
"""

import json
import socket
import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict


from .detection_config_rgb import NetworkConfig


# ─────────────────────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class RobotPose:
    """Latest known robot pose from /odom."""
    x:         float = 0.0     # meters from session start point
    y:         float = 0.0     # meters lateral offset
    z:         float = 0.0
    heading:   float = 0.0     # degrees, 0=east
    speed:     float = 0.0     # m/s
    ros_time:  float = 0.0     # ROS timestamp from Husky
    received:  float = 0.0     # local time.time() when received
    seq:       int   = 0       # message sequence number

    @property
    def age_ms(self) -> float:
        """How old this pose reading is in milliseconds."""
        return (time.time() - self.received) * 1000.0

    def to_dict(self) -> Dict:
        return {
            'x':        round(self.x, 4),
            'y':        round(self.y, 4),
            'heading':  round(self.heading, 2),
            'speed':    round(self.speed, 4),
            'age_ms':   round(self.age_ms, 1),
            'seq':      self.seq,
        }


@dataclass
class BridgeStatus:
    """Connection status of the ROS bridge."""
    connected:         bool  = False
    last_heartbeat:    float = 0.0
    husky_uptime:      float = 0.0
    odom_count:        int   = 0
    estop_active:      bool  = False
    packets_received:  int   = 0

    @property
    def heartbeat_age(self) -> float:
        """Seconds since last heartbeat from Husky PC."""
        if self.last_heartbeat == 0.0:
            return float('inf')
        return time.time() - self.last_heartbeat


# ─────────────────────────────────────────────────────────────
#  ROS BRIDGE
# ─────────────────────────────────────────────────────────────

class ROSBridge:
    """
    Receives UDP telemetry from the Husky onboard PC and
    exposes it as thread-safe Python objects.

    No ROS installation needed on the Jetson.

    Usage:
        bridge = ROSBridge(cfg.network)
        bridge.start()

        # In your capture/detection loop:
        pose   = bridge.get_pose()        # None if stale
        estop  = bridge.is_estop_active() # True = stop everything
        status = bridge.get_status()      # connection diagnostics

        bridge.stop()
    """

    # How old an odom reading can be before we consider it stale
    STALE_THRESHOLD_SEC = 2.0

    # How long without a heartbeat before we consider connection lost
    CONNECTION_TIMEOUT_SEC = 5.0

    def __init__(self, cfg: NetworkConfig):
        self.cfg     = cfg
        self._lock   = threading.Lock()
        self._running = False

        # Latest data
        self._pose   = RobotPose()
        self._status = BridgeStatus()

        # Threads — one per UDP channel
        self._threads = []

    # ── Public API ────────────────────────────────────────────

    def start(self) -> bool:
        """
        Start all UDP receiver threads.
        Returns True if at least the odom socket opened successfully.
        """
        self._running = True
        success = False

        for port, handler, name in [
            (self.cfg.odom_port,      self._handle_odom,      "odom"),
            (self.cfg.estop_port,     self._handle_estop,     "estop"),
            (self.cfg.heartbeat_port, self._handle_heartbeat, "heartbeat"),
        ]:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(('', port))
                sock.settimeout(0.5)

                t = threading.Thread(
                    target=self._recv_loop,
                    args=(sock, handler, name),
                    daemon=True
                )
                t.start()
                self._threads.append((t, sock))

                logging.info(f"ROS bridge listening: {name} on UDP:{port}")
                if name == "odom":
                    success = True

            except Exception as e:
                logging.warning(f"ROS bridge failed to open {name} port {port}: {e}")

        if success:
            logging.info(
                f"ROS bridge started — expecting Husky at {self.cfg.husky_ip}"
            )
        else:
            logging.error("ROS bridge failed to start — pose logging disabled")

        return success

    def stop(self):
        """Stop all receiver threads and close sockets."""
        self._running = False
        for t, sock in self._threads:
            try:
                sock.close()
            except Exception:
                pass
        self._threads.clear()
        logging.info("ROS bridge stopped")

    def get_pose(self) -> Optional[Dict]:
        """
        Return latest pose as dict, or None if data is stale/unavailable.
        Safe to call from any thread.

        Returns None when:
        - No data received yet
        - Last pose is older than STALE_THRESHOLD_SEC
        - Husky PC connection appears lost
        """
        with self._lock:
            if self._pose.received == 0.0:
                return None
            if self._pose.age_ms > self.STALE_THRESHOLD_SEC * 1000:
                return None
            return self._pose.to_dict()

    def get_pose_raw(self) -> RobotPose:
        """Return the raw RobotPose dataclass (may be stale)."""
        with self._lock:
            return self._pose

    def is_estop_active(self) -> bool:
        """
        Return True if Husky emergency stop is active.
        Also returns True if connection to Husky is lost —
        fail-safe behavior: treat unknown state as stopped.
        """
        with self._lock:
            # Connection lost → treat as estop for safety
            hb_age = self._status.heartbeat_age
            if hb_age > self.CONNECTION_TIMEOUT_SEC:
                return True
            return self._status.estop_active

    def is_connected(self) -> bool:
        """Return True if Husky PC is sending heartbeats."""
        # Note: do NOT call this while holding self._lock
        with self._lock:
            return self._status.heartbeat_age < self.CONNECTION_TIMEOUT_SEC

    def get_status(self) -> Dict:
        """Return full connection status dict for diagnostics/logging."""
        with self._lock:
            hb_age = self._status.heartbeat_age
            connected = hb_age < self.CONNECTION_TIMEOUT_SEC
            return {
                'connected':        connected,
                'heartbeat_age_s':  round(hb_age, 1),
                'husky_uptime_s':   round(self._status.husky_uptime, 1),
                'odom_count':       self._status.odom_count,
                'estop_active':     self._status.estop_active,
                'packets_received': self._status.packets_received,
                'latest_pose':      self._pose.to_dict()
                                    if self._pose.received > 0 else None,
            }

    # ── Internal: receive loop ────────────────────────────────

    def _recv_loop(self, sock: socket.socket,
                   handler, channel_name: str):
        """Generic UDP receive loop. Dispatches to channel handler."""
        while self._running:
            try:
                data, addr = sock.recvfrom(2048)
                msg = json.loads(data.decode('utf-8'))
                handler(msg)
                with self._lock:
                    self._status.packets_received += 1
            except socket.timeout:
                pass
            except json.JSONDecodeError as e:
                logging.debug(f"Bad JSON on {channel_name}: {e}")
            except Exception as e:
                if self._running:
                    logging.debug(f"Recv error on {channel_name}: {e}")

    # ── Internal: message handlers ────────────────────────────

    def _handle_odom(self, msg: Dict):
        """Process incoming odometry message."""
        if msg.get('type') != 'odom':
            return
        with self._lock:
            self._pose = RobotPose(
                x=msg.get('x', 0.0),
                y=msg.get('y', 0.0),
                z=msg.get('z', 0.0),
                heading=msg.get('heading', 0.0),
                speed=msg.get('speed', 0.0),
                ros_time=msg.get('ros_time', 0.0),
                received=time.time(),
                seq=msg.get('seq', 0),
            )
            self._status.odom_count += 1

    def _handle_estop(self, msg: Dict):
        """Process incoming estop message."""
        if msg.get('type') != 'estop':
            return
        with self._lock:
            prev = self._status.estop_active
            self._status.estop_active = msg.get('active', False)

        # Log state changes at WARNING level — important for field safety
        if self._status.estop_active != prev:
            if self._status.estop_active:
                logging.warning("⚠ ESTOP ACTIVATED — robot stopped")
            else:
                logging.info("✓ Estop cleared — robot operational")

    def _handle_heartbeat(self, msg: Dict):
        """Process incoming heartbeat message."""
        if msg.get('type') != 'heartbeat':
            return
        with self._lock:
            # Check connected BEFORE updating last_heartbeat
            was_connected = (
                self._status.heartbeat_age < self.CONNECTION_TIMEOUT_SEC
            )
            self._status.last_heartbeat = time.time()
            self._status.husky_uptime   = msg.get('uptime', 0.0)
            self._status.estop_active   = msg.get('estop', False)
            uptime = self._status.husky_uptime

        # Log outside lock
        if not was_connected:
            logging.info(
                f"✓ Husky PC connected (uptime: {uptime:.0f}s)"
            )


# ─────────────────────────────────────────────────────────────
#  STANDALONE TEST
# ─────────────────────────────────────────────────────────────

def _test_listener():
    """
    Run this on the Jetson to verify the bridge receives data
    from the Husky PC.

    Usage:
        python3 ros_bridge.py

    Expected output when husky_odom_pub.py is running on Husky PC:
        Waiting for data from Husky PC...
        ✓ Connected!
        Pose: x=0.00, y=0.00, heading=0.0°, speed=0.00m/s, age=12ms
        Pose: x=0.04, y=0.00, heading=0.1°, speed=0.45m/s, age=8ms
        ...
    """
    from core.detection_config_rgb import RGBCameraConfig
    cfg = RGBCameraConfig()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    bridge = ROSBridge(cfg.network)
    bridge.start()

    print(f"\nWaiting for data from Husky PC at {cfg.network.husky_ip}...")
    print("(Make sure husky_odom_pub.py is running on the Husky PC)\n")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            status = bridge.get_status()
            pose   = bridge.get_pose()
            estop  = bridge.is_estop_active()

            if bridge.is_connected():
                if pose:
                    print(
                        f"✓ Pose: "
                        f"x={pose['x']:6.2f}m  "
                        f"y={pose['y']:6.2f}m  "
                        f"heading={pose['heading']:6.1f}°  "
                        f"speed={pose['speed']:.2f}m/s  "
                        f"age={pose['age_ms']:.0f}ms  "
                        f"{'⚠ ESTOP' if estop else ''}"
                    )
                else:
                    print(
                        f"✓ Connected (heartbeat OK) — "
                        f"waiting for odom..."
                    )
            else:
                print(
                    f"✗ No connection  "
                    f"(packets received: {status['packets_received']})"
                )

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        bridge.stop()
        print(f"Final status: {bridge.get_status()}")


if __name__ == '__main__':
    _test_listener()