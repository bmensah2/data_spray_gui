#!/usr/bin/env python3
"""
test_nav_v3.py
ABEN Field Imaging System — Husky Precision Navigation
Uses a P-controller (proportional deceleration) for accurate
stop-on-target movement regardless of max speed setting.

Calibration (cpr-a200-0943, confirmed physical measurement):
  YAW_CALIBRATION = 0.92
  Husky skid-steer yaw odometry underreads by ~7°
  Physical turns confirmed accurate with tape measure.

Usage:
    python3 test_nav_v3.py forward 1.0
    python3 test_nav_v3.py forward 1.0 --speed 0.3
    python3 test_nav_v3.py backward 0.5
    python3 test_nav_v3.py left 90
    python3 test_nav_v3.py right 90 --speed 0.8

Runs on: Husky onboard PC (cpr-a200-0943)
Author : Nana | NDSU / PhD Imaging System
"""

import math
import argparse
import rospy
import tf
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


class HuskyMover:

    # ── Calibration ───────────────────────────────────────────
    # YAW_CALIBRATION: compensates for Husky skid-steer yaw odometry
    # underread. Confirmed on cpr-a200-0943 with physical tape measure.
    # Odometry reports ~83° when robot physically turns 90°.
    # Formula: commanded / physical  e.g. 90/95 = 0.947
    # Tuned value: 0.92 — confirmed accurate April 2026.
    YAW_CALIBRATION = 0.92

    # P-controller gains — speed = error × Kp
    # Higher Kp = snappier response but risks overshoot
    # Lower Kp  = smoother but slower final approach
    KP_LINEAR  = 1.2
    KP_ANGULAR = 1.5

    # Minimum speeds — prevent motor stalling during final approach
    MIN_LIN_SPEED = 0.05   # m/s
    MIN_ANG_SPEED = 0.08   # rad/s

    # Stop tolerances
    LIN_TOLERANCE = 0.005  # 5 mm
    ANG_TOLERANCE = 0.002  # ~0.1 degree

    # Safety timeout
    TIMEOUT_FACTOR = 3.0

    def __init__(self):
        rospy.init_node('husky_precision_mover', anonymous=True)
        self.pub = rospy.Publisher('/joy_teleop/cmd_vel', Twist, queue_size=10)
        self.sub = rospy.Subscriber(
            '/odometry/filtered', Odometry, self._odom_callback
        )

        self.current_x   = 0.0
        self.current_y   = 0.0
        self.current_yaw = 0.0

        self._start_x   = None
        self._start_y   = None
        self._start_yaw = None

        rospy.loginfo("HuskyMover ready — P-controller deceleration active.")

    # ── Odometry callback ─────────────────────────────────────

    def _odom_callback(self, msg):
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
            rospy.loginfo(
                f"Start: x={self._start_x:.3f}  "
                f"y={self._start_y:.3f}  "
                f"yaw={math.degrees(self._start_yaw):.1f}°"
            )

    # ── Helpers ───────────────────────────────────────────────

    def _reset_metrics(self):
        """Reset start points and wait for fresh odom callback."""
        self._start_x   = None
        self._start_y   = None
        self._start_yaw = None
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and self._start_x is None:
            rate.sleep()

    def _distance_traveled(self) -> float:
        return math.sqrt(
            (self.current_x - self._start_x) ** 2 +
            (self.current_y - self._start_y) ** 2
        )

    def _angle_turned(self) -> float:
        """Absolute angle turned in radians."""
        diff = self.current_yaw - self._start_yaw
        return abs(math.atan2(math.sin(diff), math.cos(diff)))

    def _stop(self):
        self.pub.publish(Twist())

    # ── Precision linear move ─────────────────────────────────

    def move_linear(self, distance: float, max_speed: float) -> bool:
        """
        Drive forward or backward with P-controller deceleration.
        Speed automatically scales down as robot approaches target.

        Args:
            distance  : metres — positive = forward, negative = backward
            max_speed : maximum speed in m/s
        """
        self._reset_metrics()

        abs_target = abs(distance)
        direction  = 1.0 if distance > 0 else -1.0
        label      = "forward" if distance > 0 else "backward"

        timeout_sec = (abs_target / self.MIN_LIN_SPEED) * self.TIMEOUT_FACTOR
        start_time  = rospy.Time.now()

        rospy.loginfo(
            f"Moving {label} {abs_target:.2f} m  "
            f"max {max_speed:.2f} m/s  "
            f"(timeout {timeout_sec:.0f}s)"
        )

        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            traveled = self._distance_traveled()
            error    = abs_target - traveled

            # ── Target reached ────────────────────────────────
            if error <= self.LIN_TOLERANCE:
                self._stop()
                rospy.loginfo(
                    f"✓ Target reached — "
                    f"{traveled:.3f} m / {abs_target:.2f} m  "
                    f"(error {(traveled-abs_target)*100:+.1f} cm)"
                )
                return True

            # ── Timeout safety ────────────────────────────────
            elapsed = (rospy.Time.now() - start_time).to_sec()
            if elapsed > timeout_sec:
                self._stop()
                rospy.logwarn(
                    f"⚠ Timeout after {elapsed:.1f}s — "
                    f"traveled {traveled:.3f} m / {abs_target:.2f} m"
                )
                return False

            # ── P-controller speed ────────────────────────────
            speed = max(self.MIN_LIN_SPEED,
                        min(max_speed, error * self.KP_LINEAR))

            rospy.loginfo_throttle(
                1.0,
                f"  {traveled:.2f} / {abs_target:.2f} m  "
                f"({traveled/abs_target*100:.0f}%)  "
                f"vel={speed:.2f} m/s"
            )

            cmd = Twist()
            cmd.linear.x = direction * speed
            self.pub.publish(cmd)
            rate.sleep()

        self._stop()
        return False

    # ── Precision rotation ────────────────────────────────────

    def rotate(self, degrees: float, max_speed: float) -> bool:
        """
        Turn in place with P-controller deceleration.
        Speed automatically scales down as robot approaches target angle.
        YAW_CALIBRATION corrects for skid-steer odometry underread.

        Args:
            degrees   : angle — positive = left, negative = right
            max_speed : maximum angular speed in rad/s
        """
        self._reset_metrics()

        # Apply calibration factor for skid-steer odometry underread
        target_rad = abs(math.radians(degrees)) * self.YAW_CALIBRATION
        direction  = 1.0 if degrees > 0 else -1.0
        label      = "left" if degrees > 0 else "right"

        timeout_sec = (target_rad / self.MIN_ANG_SPEED) * self.TIMEOUT_FACTOR
        start_time  = rospy.Time.now()

        rospy.loginfo(
            f"Rotating {label} {abs(degrees):.1f}°  "
            f"max {math.degrees(max_speed):.0f}°/s  "
            f"(timeout {timeout_sec:.0f}s)"
        )

        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            turned_rad = self._angle_turned()
            turned_deg = math.degrees(turned_rad)
            error      = target_rad - turned_rad

            # ── Target reached ────────────────────────────────
            if error <= self.ANG_TOLERANCE:
                self._stop()
                # Note: odom underreads yaw — physical angle is accurate
                rospy.loginfo(
                    f"✓ Turn complete — "
                    f"odom={turned_deg:.1f}°  physical≈{abs(degrees):.1f}°"
                )
                return True

            # ── Timeout safety ────────────────────────────────
            elapsed = (rospy.Time.now() - start_time).to_sec()
            if elapsed > timeout_sec:
                self._stop()
                rospy.logwarn(
                    f"⚠ Timeout after {elapsed:.1f}s — "
                    f"turned {turned_deg:.1f}° / {abs(degrees):.1f}°"
                )
                return False

            # ── P-controller speed ────────────────────────────
            speed = max(self.MIN_ANG_SPEED,
                        min(max_speed, error * self.KP_ANGULAR))

            rospy.loginfo_throttle(
                1.0,
                f"  {turned_deg:.1f} / {abs(degrees):.1f}°  "
                f"({turned_deg/abs(degrees)*100:.0f}%)  "
                f"vel={math.degrees(speed):.0f}°/s"
            )

            cmd = Twist()
            cmd.angular.z = direction * speed
            self.pub.publish(cmd)
            rate.sleep()

        self._stop()
        return False


# ── CLI ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ABEN Husky precision navigation"
    )
    parser.add_argument(
        'mode',
        choices=['forward', 'backward', 'left', 'right'],
        help='Direction of movement'
    )
    parser.add_argument(
        'val', type=float,
        help='Distance in metres or angle in degrees'
    )
    parser.add_argument(
        '--speed', type=float, default=None,
        help='Max speed override (m/s linear, rad/s angular)'
    )
    args = parser.parse_args()

    mover = HuskyMover()

    if args.mode == 'forward':
        mover.move_linear(args.val, args.speed or 0.1)
    elif args.mode == 'backward':
        mover.move_linear(-args.val, args.speed or 0.1)
    elif args.mode == 'left':
        mover.rotate(abs(args.val), args.speed or 0.5)
    elif args.mode == 'right':
        mover.rotate(-abs(args.val), args.speed or 0.5)


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass