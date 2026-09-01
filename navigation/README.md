# Navigation Scripts

These scripts run standalone — either on the Husky PC or Jetson.

## Husky PC (192.168.131.1)
Scripts that run on the Husky's onboard PC:
  test_nav_v3.py      — manual move commands (forward/backward/left/right)
  field_nav.py        — autonomous YAML mission executor
  husky_odom_pub.py   — UDP odometry bridge → Jetson (auto-started by systemd)

## Jetson (runs here, SSHes to Husky)
  field_run.py        — field deployment wrapper
  nav_mission_panel.py — legacy mission panel widget

## Deployment
Husky PC copies are maintained separately at ~/
Sync with: scp navigation/test_nav_v3.py administrator@192.168.131.1:~/
