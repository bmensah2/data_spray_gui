#!/usr/bin/env python3
"""
husky_odom_pub.py  v1
Field Detection System — Husky Odometry UDP Publisher

Runs ON THE HUSKY ONBOARD PC (Ubuntu 20.04 / ROS Noetic)

v1: Uses 'rostopic echo' subprocess instead of rospy.init_node()
    Completely avoids ROS_IP / hanging initialization issues.
    No rospy needed — just the rostopic CLI tool which is always
    available when ROS is sourced.

Setup on Husky PC:
    python3 husky_odom_pub.py --jetson-ip 192.168.131.51

    Optional — use filtered odometry (better accuracy):
    python3 husky_odom_pub.py --jetson-ip 192.168.131.51 \
            --odom-topic /odometry/filtered

Author : Nana | NDSU / PhD Imaging System
Runs on: Husky onboard PC (cpr-a200-0943)
"""

import sys
import json
import socket
import math
import argparse
import subprocess
import threading
import time
import logging


# ─────────────────────────────────────────────────────────────
#  YAML PARSERS
# ─────────────────────────────────────────────────────────────

def _find_value(lines, key) -> float:
    """Find 'key: value' in a list of lines, return float."""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(key + ':'):
            try:
                return float(stripped.split(':', 1)[1].strip())
            except (ValueError, IndexError):
                pass
    return 0.0


def parse_odom_yaml(block: str) -> dict:
    """
    Parse a single /odom YAML block from rostopic echo output.
    Extracts position, orientation (yaw), and linear velocity.
    """
    lines = block.split('\n')

    # Find 'position:' section
    pos_idx = next((i for i, l in enumerate(lines)
                    if l.strip() == 'position:'), -1)
    px = py = pz = 0.0
    if pos_idx >= 0:
        sec = lines[pos_idx+1:pos_idx+8]
        px = _find_value(sec, 'x')
        py = _find_value(sec, 'y')
        pz = _find_value(sec, 'z')

    # Find 'orientation:' section
    ori_idx = next((i for i, l in enumerate(lines)
                    if l.strip() == 'orientation:'), -1)
    qx = qy = qz = 0.0; qw = 1.0
    if ori_idx >= 0:
        sec = lines[ori_idx+1:ori_idx+8]
        qx = _find_value(sec, 'x')
        qy = _find_value(sec, 'y')
        qz = _find_value(sec, 'z')
        qw = _find_value(sec, 'w') or 1.0

    # Find 'linear:' section (velocity)
    lin_idx = next((i for i, l in enumerate(lines)
                    if l.strip() == 'linear:'), -1)
    vx = vy = 0.0
    if lin_idx >= 0:
        sec = lines[lin_idx+1:lin_idx+6]
        vx = _find_value(sec, 'x')
        vy = _find_value(sec, 'y')

    # Quaternion → yaw
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw_deg = math.degrees(math.atan2(siny, cosy))
    speed = math.sqrt(vx**2 + vy**2)

    return {
        'type':    'odom',
        'x':       round(px, 4),
        'y':       round(py, 4),
        'z':       round(pz, 4),
        'heading': round(yaw_deg, 2),
        'speed':   round(speed, 4),
        'ros_time': time.time(),
    }


def parse_estop_yaml(block: str) -> dict:
    """Parse std_msgs/Bool from /estop topic."""
    for line in block.split('\n'):
        s = line.strip()
        if s.startswith('data:'):
            val = s.split(':', 1)[1].strip().lower()
            return {
                'type':    'estop',
                'active':  val in ('true', '1', 'yes'),
                'ros_time': time.time(),
            }
    return {'type': 'estop', 'active': False, 'ros_time': time.time()}


# ─────────────────────────────────────────────────────────────
#  UDP SENDER
# ─────────────────────────────────────────────────────────────

class UDPSender:
    def __init__(self, jetson_ip: str):
        self.jetson_ip = jetson_ip
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, data: dict, port: int):
        try:
            self._sock.sendto(
                json.dumps(data).encode('utf-8'),
                (self.jetson_ip, port)
            )
        except Exception as e:
            logging.warning(f"UDP send error port {port}: {e}")

    def close(self):
        self._sock.close()


# ─────────────────────────────────────────────────────────────
#  TOPIC STREAMER
# ─────────────────────────────────────────────────────────────

class TopicStreamer:
    """
    Pipes 'rostopic echo <topic>' and forwards parsed messages
    over UDP. No rospy needed — just the CLI tool.
    """

    def __init__(self, topic: str, sender: UDPSender,
                 port: int, parser, name: str,
                 throttle_hz: float = 10.0):
        self.topic     = topic
        self.sender    = sender
        self.port      = port
        self.parser    = parser
        self.name      = name
        self.throttle  = 1.0 / throttle_hz
        self._running  = False
        self._thread   = None
        self._count    = 0
        self._last_send = 0.0
        self._proc     = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True
        )
        self._thread.start()
        logging.info(
            f"Streaming {self.topic} → UDP:{self.port} "
            f"@ {1/self.throttle:.0f}Hz"
        )

    def stop(self):
        self._running = False
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass

    def _loop(self):
        cmd = ['rostopic', 'echo', self.topic]
        logging.info(f"Running: {' '.join(cmd)}")

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
        except FileNotFoundError:
            logging.error(
                "rostopic not found. "
                "Make sure ROS is sourced: source /opt/ros/noetic/setup.bash"
            )
            return

        block = []

        for line in self._proc.stdout:
            if not self._running:
                break

            line = line.rstrip()

            if line == '---':
                if block:
                    now = time.time()
                    if now - self._last_send >= self.throttle:
                        try:
                            data = self.parser('\n'.join(block))
                            data['seq'] = self._count
                            self.sender.send(data, self.port)
                            self._count += 1
                            self._last_send = now
                            # Log every 50 messages
                            if self._count % 50 == 1:
                                logging.info(
                                    f"{self.name} "
                                    f"#{self._count}: "
                                    f"x={data.get('x',0):.2f} "
                                    f"y={data.get('y',0):.2f} "
                                    f"spd={data.get('speed',0):.2f}"
                                )
                        except Exception as e:
                            logging.debug(f"Parse error: {e}")
                    block = []
            else:
                block.append(line)

        logging.warning(f"{self.name} streamer ended")


# ─────────────────────────────────────────────────────────────
#  HEARTBEAT
# ─────────────────────────────────────────────────────────────

def _heartbeat_loop(sender: UDPSender, port: int,
                    odom: TopicStreamer,
                    stop_event: threading.Event):
    start = time.time()
    while not stop_event.is_set():
        sender.send({
            'type':       'heartbeat',
            'uptime':     round(time.time() - start, 1),
            'odom_count': odom._count,
            'estop':      False,
            'ros_time':   time.time(),
        }, port)
        stop_event.wait(1.0)


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Field Detection System — Husky Odometry UDP Publisher v1'
    )
    ap.add_argument('--jetson-ip', required=True,
                    help='Jetson IP (192.168.131.51)')
    ap.add_argument('--odom-port',      type=int,   default=5005)
    ap.add_argument('--estop-port',     type=int,   default=5006)
    ap.add_argument('--heartbeat-port', type=int,   default=5007)
    ap.add_argument('--odom-hz',        type=float, default=10.0,
                    help='Rate to forward odom to Jetson (default 10Hz)')
    ap.add_argument('--odom-topic', default='/husky_velocity_controller/odom',
                    help='Odom topic (default /husky_velocity_controller/odom). '
                         'Alt: /odometry/filtered or '
                         '/odom')
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    logging.info("=" * 50)
    logging.info("Field Detection System — Husky Odometry UDP Publisher v1")
    logging.info(f"Jetson target : {args.jetson_ip}")
    logging.info(f"Odom topic    : {args.odom_topic}")
    logging.info(f"Odom rate     : {args.odom_hz} Hz")
    logging.info("=" * 50)

    sender = UDPSender(args.jetson_ip)

    odom = TopicStreamer(
        topic=args.odom_topic,
        sender=sender,
        port=args.odom_port,
        parser=parse_odom_yaml,
        name='odom',
        throttle_hz=args.odom_hz
    )
    estop = TopicStreamer(
        topic='/estop',
        sender=sender,
        port=args.estop_port,
        parser=parse_estop_yaml,
        name='estop',
        throttle_hz=5.0
    )

    odom.start()
    estop.start()

    stop_evt = threading.Event()
    hb = threading.Thread(
        target=_heartbeat_loop,
        args=(sender, args.heartbeat_port, odom, stop_evt),
        daemon=True
    )
    hb.start()

    logging.info("Bridge running — Ctrl+C to stop")

    try:
        while True:
            time.sleep(10.0)
            logging.info(
                f"Heartbeat | odom sent: {odom._count} | "
                f"estop sent: {estop._count}"
            )
    except KeyboardInterrupt:
        logging.info("Stopping bridge...")
    finally:
        stop_evt.set()
        odom.stop()
        estop.stop()
        sender.close()
        logging.info(
            f"Stopped. odom={odom._count} "
            f"estop={estop._count} messages sent."
        )


if __name__ == '__main__':
    main()