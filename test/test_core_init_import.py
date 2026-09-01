import importlib


def test_core_package_imports():
    core = importlib.import_module('core')
    from core.detection_config_rgb import NetworkConfig, RGBConfig

    assert hasattr(core, 'ActuationController')
    assert hasattr(core, 'RGBDetectionEngine')
    assert hasattr(core, 'ZoneManagerRGB')
    assert hasattr(core, 'EventLogger')

    cfg = NetworkConfig()
    assert cfg.husky_ip == '192.168.131.1'
    assert isinstance(RGBConfig(), RGBConfig)
