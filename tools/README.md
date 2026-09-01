# Tools

Standalone scripts that run on the Jetson.

  demo_spray.py   — spray pipeline demo (drives Husky + fires nozzles randomly)
                    Usage: python tools/demo_spray.py --dist 1.0
  capture_tool.py — standalone image capture tool

## Demo Spray
Uses SSH to drive Husky and local Arduino for nozzle control.
Requires: husky_odom_pub.py running on Husky (auto via systemd).
