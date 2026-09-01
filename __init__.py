"""
ABEN Field Spray & Imaging System
Jetson-side GUI, detection, and actuation package.

This file intentionally does NOT eagerly import submodules.
Entry points (main_gui_rgb.py, tools/*, navigation/*) import
directly from `core`, `gui`, and `navigation` as needed; core's
own __init__.py already does defensive per-module importing
(see core/__init__.py _safe_import) so a missing optional
dependency there doesn't break the whole package.

NOTE: earlier versions of this file imported
`core.acquisition_manager` / `core.capture_scheduler`, which no
longer exist in this codebase (superseded by the RGB detection
pipeline: core.detection_engine_rgb, core.zone_manager_rgb,
core.actuation_controller). Any code that still does
`import data_spray_gui` expecting those names needs updating.
"""

__version__ = '3.0.0'
__author__ = 'Agricultural Team'
