#!/usr/bin/env python3
"""
field_nav.py
ABEN Field Imaging System — Autonomous Field Navigation

Executes a YAML mission file on the Husky PC.
Supports boustrophedon (row scan) and custom waypoint patterns.
Logs every step and returns home via direct odometry path.

Usage:
    python3 field_nav.py missions/sugarbeet_rows.yaml
    python3 field_nav.py missions/sugarbeet_rows.yaml --dry-run
    python3 field_nav.py missions/sugarbeet_rows.yaml --speed 0.2

Runs on: Husky onboard PC (cpr-a200-0943)
Author : Nana | NDSU / PhD Imaging System
"""

import math
import argparse
import yaml
import json
import time
import os
import sys
from datetime import datetime
from pathlib import Path

import rospy
import tf
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


# ── Default speeds (safe, accurate) ──────────────────────────
DEFAULT_LINEAR_SPEED  = 0.1   # m/s
DEFAULT_ANGULAR_SPEED = 0.1   # rad/s  (~5.7°/s)


# ─────────────────────────────────────────────────────────────
#  MISSION LOG
# ─────────────────────────────────────────────────────────────
class MissionLog:
    """Records every step for post-mission review."""

    def __init__(self, mission_name: str, log_dir: str = "~/missions/logs"):
        self.mission_name = mission_name
        self.log_dir = Path(log_dir).expanduser()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.log_dir / f"{mission_name}_{ts}.jsonl"
        self.start_time = time.time()
        self.entries = []
        self._log_event("mission_start", {
            "mission": mission_name,
            "timestamp": datetime.now().isoformat()
        })
        print(f"[LOG] Mission log: {self.log_path}")

    def _log_event(self, event: str, data: dict):
        entry = {
            "event":     event,
            "elapsed_s": round(time.time() - self.start_time, 2),
            **data
        }
        self.entries.append(entry)
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def step_start(self, step_idx: int, total: int,
                   step_def: dict, pose: dict):
        label = step_def.get('label', '')
        print(f"\n[STEP {step_idx+1}/{total}] {step_def}  {label}")
        self._log_event("step_start", {
            "step":     step_idx + 1,
            "total":    total,
            "command":  step_def,
            "label":    label,
            "pose_start": pose,
        })

    def step_done(self, step_idx: int, success: bool,
                  pose: dict, actual: float, commanded: float):
        error = round(actual - commanded, 4)
        print(f"  → {'✓' if success else '✗'}  "
              f"actual={actual:.3f}  commanded={commanded:.3f}  "
              f"err={error:+.3f}")
        self._log_event("step_done", {
            "step":      step_idx + 1,
            "success":   success,
            "commanded": commanded,
            "actual":    actual,
            "error":     error,
            "pose_end":  pose,
        })

    def mission_done(self, success: bool, total_steps: int,
                     final_pose: dict):
        duration = round(time.time() - self.start_time, 1)
        self._log_event("mission_done", {
            "success":      success,
            "total_steps":  total_steps,
            "duration_s":   duration,
            "final_pose":   final_pose,
        })
        print(f"\n[MISSION {'COMPLETE' if success else 'FAILED'}] "
              f"{total_steps} steps  |  {duration}s  |  "
              f"log: {self.log_path}")


# ─────────────────────────────────────────────────────────────
#  HUSKY MOVER  (P-controller — same as test_nav_v3.py)
# ─────────────────────────────────────────────────────────────
class HuskyMover:

    YAW_CALIBRATION = 0.92   # confirmed cpr-a200-0943 April 2026
    KP_LINEAR       = 1.2
    KP_ANGULAR      = 1.5
    MIN_LIN_SPEED   = 0.05   # m/s
    MIN_ANG_SPEED   = 0.08   # rad/s
    LIN_TOLERANCE   = 0.005  # 5 mm
    ANG_TOLERANCE   = 0.002  # ~0.1°
    TIMEOUT_FACTOR  = 3.0

    def __init__(self):
        rospy.init_node('aben_field_nav', anonymous=True)
        self.pub = rospy.Publisher('/joy_teleop/cmd_vel', Twist, queue_size=10)
        self.sub = rospy.Subscriber(
            '/odometry/filtered', Odometry, self._odom_cb
        )
        self.current_x   = 0.0
        self.current_y   = 0.0
        self.current_yaw = 0.0   # radians

        self._start_x   = None
        self._start_y   = None
        self._start_yaw = None

        # Home position — set at mission start
        self.home_x   = 0.0
        self.home_y   = 0.0
        self.home_yaw = 0.0

        rospy.loginfo("HuskyMover ready — field navigation mode.")

    # ── Odometry ──────────────────────────────────────────────

    def _odom_cb(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        euler = tf.transformations.euler_from_quaternion(
            [q.x, q.y, q.z, q.w]
        )
        self.current_yaw = euler[2]

        if self._start_x is None:
            self._start_x   = self.current_x
            self._start_y   = self.current_y
            self._start_yaw = self.current_yaw

    def _reset(self):
        self._start_x = self._start_y = self._start_yaw = None
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and self._start_x is None:
            rate.sleep()

    def get_pose(self) -> dict:
        return {
            'x':       round(self.current_x, 4),
            'y':       round(self.current_y, 4),
            'yaw_deg': round(math.degrees(self.current_yaw), 2),
        }

    def set_home(self):
        """Record current position as home."""
        self.home_x   = self.current_x
        self.home_y   = self.current_y
        self.home_yaw = self.current_yaw
        rospy.loginfo(
            f"Home set: x={self.home_x:.3f}  "
            f"y={self.home_y:.3f}  "
            f"yaw={math.degrees(self.home_yaw):.1f}°"
        )

    def _stop(self):
        self.pub.publish(Twist())

    # ── Linear move ───────────────────────────────────────────

    def move_linear(self, distance: float,
                    max_speed: float = DEFAULT_LINEAR_SPEED) -> tuple:
        """
        Returns (success, actual_distance_traveled).
        distance > 0 = forward, < 0 = backward.
        """
        self._reset()
        abs_target = abs(distance)
        direction  = 1.0 if distance > 0 else -1.0
        label      = "forward" if distance > 0 else "backward"

        timeout_sec = (abs_target / self.MIN_LIN_SPEED) * self.TIMEOUT_FACTOR
        start_time  = rospy.Time.now()

        rospy.loginfo(f"  Linear {label} {abs_target:.2f}m "
                      f"max={max_speed:.2f}m/s")

        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            traveled = math.sqrt(
                (self.current_x - self._start_x) ** 2 +
                (self.current_y - self._start_y) ** 2
            )
            error = abs_target - traveled

            if error <= self.LIN_TOLERANCE:
                self._stop()
                return True, traveled

            elapsed = (rospy.Time.now() - start_time).to_sec()
            if elapsed > timeout_sec:
                self._stop()
                rospy.logwarn(f"  Timeout after {elapsed:.1f}s")
                return False, traveled

            speed = max(self.MIN_LIN_SPEED,
                        min(max_speed, error * self.KP_LINEAR))
            rospy.loginfo_throttle(
                1.0, f"  {traveled:.2f}/{abs_target:.2f}m  vel={speed:.2f}")

            cmd = Twist()
            cmd.linear.x = direction * speed
            self.pub.publish(cmd)
            rate.sleep()

        self._stop()
        return False, 0.0

    # ── Rotation ──────────────────────────────────────────────

    def rotate(self, degrees: float,
               max_speed: float = DEFAULT_ANGULAR_SPEED) -> tuple:
        """
        Returns (success, actual_degrees_turned).
        degrees > 0 = left, < 0 = right.
        """
        self._reset()
        target_rad = abs(math.radians(degrees)) * self.YAW_CALIBRATION
        direction  = 1.0 if degrees > 0 else -1.0
        label      = "left" if degrees > 0 else "right"

        timeout_sec = (target_rad / self.MIN_ANG_SPEED) * self.TIMEOUT_FACTOR
        start_time  = rospy.Time.now()

        rospy.loginfo(f"  Rotate {label} {abs(degrees):.1f}° "
                      f"max={math.degrees(max_speed):.0f}°/s")

        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            diff       = self.current_yaw - self._start_yaw
            turned_rad = abs(math.atan2(math.sin(diff), math.cos(diff)))
            error      = target_rad - turned_rad

            if error <= self.ANG_TOLERANCE:
                self._stop()
                return True, math.degrees(turned_rad)

            elapsed = (rospy.Time.now() - start_time).to_sec()
            if elapsed > timeout_sec:
                self._stop()
                rospy.logwarn(f"  Turn timeout after {elapsed:.1f}s")
                return False, math.degrees(turned_rad)

            speed = max(self.MIN_ANG_SPEED,
                        min(max_speed, error * self.KP_ANGULAR))
            rospy.loginfo_throttle(
                1.0,
                f"  {math.degrees(turned_rad):.1f}/{abs(degrees):.1f}°  "
                f"vel={math.degrees(speed):.0f}°/s"
            )

            cmd = Twist()
            cmd.angular.z = direction * speed
            self.pub.publish(cmd)
            rate.sleep()

        self._stop()
        return False, 0.0

    # ── Return home ───────────────────────────────────────────

    def return_home(self, lin_speed: float = DEFAULT_LINEAR_SPEED,
                    ang_speed: float = DEFAULT_ANGULAR_SPEED) -> bool:
        """
        Calculate direct vector from current position back to home.
        1. Rotate to face home direction
        2. Drive straight distance to home
        3. Rotate to restore home heading
        """
        dx = self.home_x - self.current_x
        dy = self.home_y - self.current_y
        dist = math.sqrt(dx**2 + dy**2)

        rospy.loginfo(
            f"\n[RETURN HOME] "
            f"dist={dist:.2f}m  "
            f"from ({self.current_x:.2f},{self.current_y:.2f}) "
            f"to ({self.home_x:.2f},{self.home_y:.2f})"
        )

        if dist < 0.05:
            rospy.loginfo("  Already at home — skipping return move")
        else:
            # Angle to home from current position
            target_heading = math.atan2(dy, dx)
            # How much to turn from current yaw
            turn_needed = math.degrees(
                math.atan2(
                    math.sin(target_heading - self.current_yaw),
                    math.cos(target_heading - self.current_yaw)
                )
            )

            rospy.loginfo(f"  Turn to home heading: {turn_needed:+.1f}°")
            if abs(turn_needed) > 2.0:
                ok, _ = self.rotate(turn_needed, ang_speed)
                if not ok:
                    rospy.logwarn("  Return heading turn failed")

            rospy.loginfo(f"  Drive home: {dist:.2f}m")
            ok, actual = self.move_linear(dist, lin_speed)
            if not ok:
                rospy.logwarn("  Return drive failed")

        # Restore home heading — only if error > 10°
        # (small drift after long straight return is acceptable)
        heading_diff = math.degrees(
            math.atan2(
                math.sin(self.home_yaw - self.current_yaw),
                math.cos(self.home_yaw - self.current_yaw)
            )
        )
        if abs(heading_diff) > 10.0:
            rospy.loginfo(f"  Restore home heading: {heading_diff:+.1f}°")
            self.rotate(heading_diff, ang_speed)
        else:
            rospy.loginfo(
                f"  Heading error {heading_diff:+.1f}° — within tolerance, skipping")

        rospy.loginfo("  ✓ Home reached")
        return True


# ─────────────────────────────────────────────────────────────
#  MISSION EXECUTOR
# ─────────────────────────────────────────────────────────────
class MissionExecutor:

    def __init__(self, mission_path: str,
                 speed_override: float = None,
                 dry_run: bool = False):
        self.mission_path   = Path(mission_path)
        self.speed_override = speed_override
        self.dry_run        = dry_run
        self.mission        = self._load(mission_path)
        self.mover          = None if dry_run else HuskyMover()

    def _load(self, path: str) -> dict:
        with open(path) as f:
            m = yaml.safe_load(f)
        # Validate required fields
        if 'steps' not in m:
            raise ValueError("Mission YAML must have 'steps' list")
        print(f"[MISSION] {m.get('name','unnamed')} — "
              f"{len(m['steps'])} steps")
        if m.get('description'):
            print(f"          {m['description']}")
        return m

    def run(self) -> bool:
        m       = self.mission
        name    = m.get('name', self.mission_path.stem)
        steps   = m['steps']
        ret_home = m.get('return_home', True)

        # Default speeds from YAML or global defaults
        lin_spd = (self.speed_override or
                   m.get('default_linear_speed', DEFAULT_LINEAR_SPEED))
        ang_spd = (self.speed_override or
                   m.get('default_angular_speed', DEFAULT_ANGULAR_SPEED))

        if self.dry_run:
            print(f"\n[DRY RUN] {name}")
            for i, step in enumerate(steps):
                print(f"  Step {i+1}: {step}")
            if ret_home:
                print(f"  Step {len(steps)+1}: return_home")
            print("[DRY RUN COMPLETE]")
            return True

        # Wait for odometry
        rospy.sleep(0.5)
        log = MissionLog(name)

        # Set home at start
        self.mover.set_home()
        all_ok = True

        print(f"\n[MISSION START] {name}  |  {len(steps)} steps")
        print(f"  Linear speed:  {lin_spd:.2f} m/s")
        print(f"  Angular speed: {math.degrees(ang_spd):.1f}°/s")
        print(f"  Return home:   {ret_home}")

        for i, step in enumerate(steps):
            if rospy.is_shutdown():
                break

            pose = self.mover.get_pose()
            log.step_start(i, len(steps), step, pose)

            # Parse step
            step_lin  = step.get('linear_speed',  lin_spd)
            step_ang  = step.get('angular_speed',  ang_spd)

            if 'forward' in step:
                dist = float(step['forward'])
                ok, actual = self.mover.move_linear(dist, step_lin)
                log.step_done(i, ok, self.mover.get_pose(), actual, dist)

            elif 'backward' in step:
                dist = float(step['backward'])
                ok, actual = self.mover.move_linear(-dist, step_lin)
                log.step_done(i, ok, self.mover.get_pose(), actual, dist)

            elif 'turn' in step:
                deg = float(step['turn'])
                ok, actual = self.mover.rotate(deg, step_ang)
                log.step_done(i, ok, self.mover.get_pose(), actual, abs(deg))

            else:
                rospy.logwarn(f"  Unknown step: {step}")
                ok = False
                log.step_done(i, False, self.mover.get_pose(), 0, 0)

            if not ok:
                all_ok = False
                rospy.logwarn(f"  Step {i+1} failed — continuing")

            # Brief pause between steps
            rospy.sleep(0.3)

        # Return home
        if ret_home and not rospy.is_shutdown():
            print(f"\n[RETURN HOME]")
            log.step_start(len(steps), len(steps)+1,
                           {'label': 'return_home'},
                           self.mover.get_pose())
            home_ok = self.mover.return_home(lin_spd, ang_spd)
            log.step_done(len(steps), home_ok,
                          self.mover.get_pose(), 0, 0)

        log.mission_done(all_ok, len(steps), self.mover.get_pose())
        return all_ok


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ABEN Husky field mission executor"
    )
    parser.add_argument(
        'mission', type=str,
        help='Path to mission YAML file (e.g. missions/sugarbeet_rows.yaml)'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print mission steps without executing'
    )
    parser.add_argument(
        '--speed', type=float, default=None,
        help='Override all speeds (m/s and rad/s)'
    )
    args = parser.parse_args()

    if not Path(args.mission).exists():
        print(f"ERROR: Mission file not found: {args.mission}")
        sys.exit(1)

    executor = MissionExecutor(
        mission_path=args.mission,
        speed_override=args.speed,
        dry_run=args.dry_run
    )

    try:
        success = executor.run()
        sys.exit(0 if success else 1)
    except rospy.ROSInterruptException:
        print("\n[INTERRUPTED] Mission stopped by user")
        sys.exit(1)


if __name__ == '__main__':
    main()