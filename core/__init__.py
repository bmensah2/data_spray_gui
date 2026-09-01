"""
Core Package
ABEN Field Imaging System — Core Modules
"""

from importlib import import_module

__version__ = '3.0.0'
__author__ = 'Agricultural Team'

_IMPORT_ERRORS = {}


def _safe_import(module_name: str, names: tuple[str, ...]):
    try:
        module = import_module(module_name, package=__name__)
    except Exception as exc:  # pragma: no cover - defensive import guard
        _IMPORT_ERRORS[module_name] = exc
        return {}

    imported = {}
    for name in names:
        if hasattr(module, name):
            imported[name] = getattr(module, name)
    return imported


for module_name, names in [
    ('.detection_config_rgb', (
        'ABENConfig', 'DetectionMode', 'CameraMode', 'GrowthStage',
        'ModelInputFormat', 'NetworkConfig', 'get_weed_config', 'get_cls_config'
    )),
    ('.detection_engine_rgb', ('RGBDetectionEngine', 'Detection', 'InferenceResult')),
    ('.actuation_controller', ('ActuationController', 'SprayEvent')),
    ('.event_logger', ('EventLogger', 'EventLogEntry')),
    ('.ros_bridge', ('ROSBridge', 'BridgeStatus')),
    ('.zone_manager_rgb', ('ZoneManagerRGB', 'ZoneDecision', 'ZoneState')),
]:
    globals().update(_safe_import(module_name, names))

try:
    from .gantry_controller import GantryController, GantryState
except Exception as exc:  # pragma: no cover - optional hardware dependency
    _IMPORT_ERRORS['.gantry_controller'] = exc
else:
    globals().update({'GantryController': GantryController, 'GantryState': GantryState})


__all__ = [
    'ABENConfig', 'DetectionMode', 'CameraMode', 'GrowthStage',
    'ModelInputFormat', 'NetworkConfig', 'get_weed_config', 'get_cls_config',
    'RGBDetectionEngine', 'Detection', 'InferenceResult',
    'ActuationController', 'SprayEvent',
    'EventLogger', 'EventLogEntry',
    'ROSBridge', 'BridgeStatus',
    'ZoneManagerRGB', 'ZoneDecision', 'ZoneState',
]

if 'GantryController' in globals():
    __all__.extend(['GantryController', 'GantryState'])

__all__ = [name for name in __all__ if name in globals()]
