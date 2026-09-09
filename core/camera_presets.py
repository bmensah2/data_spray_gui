"""
core/camera_presets.py
ABEN Dual RGB Imaging System — Camera preset persistence

The three lighting presets (Outdoor / Cloudy / Indoor) used to live
only as hardcoded dicts in core/dual_emeet_camera.py, so tuning them
for actual field conditions meant editing source. This module keeps
the hardcoded values as immutable FACTORY defaults and layers
user-edited overrides on top, persisted to a JSON file next to the
project.

Design notes:
  - Factory defaults are never modified, so "restore defaults" always
    works no matter how badly a preset gets tuned.
  - A user preset stores the FULL settings dict (not a sparse diff),
    so a preset's behavior can't silently change underneath the
    operator if the factory defaults are ever edited in a future
    release -- what you saved is what you get.
  - Unknown keys in a saved file are preserved on load but ignored,
    and missing keys fall back to the factory value for that preset,
    so a preset file written by an older/newer version won't break
    or silently drop settings.
  - Load failures (missing file, corrupt JSON, bad permissions) fall
    back to factory defaults rather than raising -- a bad preset file
    should never stop the camera from opening.
"""

import copy
import json
import logging
from pathlib import Path

PRESET_FILE = Path(__file__).resolve().parent.parent / "camera_presets.json"

# Canonical preset names, in the order the GUI shows them.
PRESET_NAMES = ["outdoor", "cloudy", "indoor"]

DISPLAY_NAMES = {
    "outdoor": "Outdoor / Field",
    "cloudy":  "Cloudy / Shade",
    "indoor":  "Indoor / Lab",
}

# Every editable setting, with a human label and valid range, so the
# editor dialog can build itself from this rather than hardcoding a
# parallel list that could drift out of sync.
# Ranges match the actual v4l2 limits used by the Image Controls in
# gui/panels/acquisition_panel_rgb.py -- keep the two in sync, or the
# editor could store values the camera will reject.
# (key, label, min, max)
SETTING_SPECS = [
    ("exposure",      "Exposure",              1, 5000),
    ("brightness",    "Brightness",         -64,   64),
    ("contrast",      "Contrast",             0,   64),
    ("saturation",    "Saturation",           0,  128),
    ("gamma",         "Gamma",               72,  500),
    ("gain",          "Gain",                 0,  100),
    ("sharpness",     "Sharpness",            0,    6),
    ("wb_temp",       "White balance (K)", 2800, 6500),
    ("focus",         "Focus",                0, 1023),
    ("backlight",     "Backlight comp.",      0,    2),
    ("freq",          "Power line freq.",     0,    2),
    ("autofocus",     "Autofocus (0/1)",      0,    1),
    ("auto_wb",       "Auto white bal. (0/1)", 0,   1),
    ("auto_exposure", "Auto exposure (1=man, 3=auto)", 1, 3),
]

SETTING_KEYS = [s[0] for s in SETTING_SPECS]


def _factory_presets() -> dict:
    """
    The hardcoded presets from DualEMEETCamera, deep-copied so callers
    can never mutate the originals through a returned reference.
    Imported lazily to avoid a circular import (dual_emeet_camera
    imports nothing from here, but this keeps it one-directional even
    if that changes).
    """
    from core.dual_emeet_camera import DualEMEETCamera
    return {
        "outdoor": copy.deepcopy(DualEMEETCamera.PRESET_OUTDOOR),
        "cloudy":  copy.deepcopy(DualEMEETCamera.PRESET_CLOUDY),
        "indoor":  copy.deepcopy(DualEMEETCamera.PRESET_INDOOR),
    }


def factory_preset(name: str) -> dict:
    """Return the unmodified factory defaults for one preset."""
    return _factory_presets().get(name, {})


def load_presets() -> dict:
    """
    Return {name: settings_dict} for all three presets: user-saved
    values where they exist, factory defaults otherwise. Never raises
    -- a missing/corrupt preset file falls back to factory.
    """
    presets = _factory_presets()

    if not PRESET_FILE.exists():
        return presets

    try:
        with open(PRESET_FILE) as f:
            saved = json.load(f)
    except Exception as e:
        logging.warning(
            f"camera_presets: could not read {PRESET_FILE} ({e}) -- "
            f"using factory defaults")
        return presets

    if not isinstance(saved, dict):
        logging.warning(
            f"camera_presets: {PRESET_FILE} is not a JSON object -- "
            f"using factory defaults")
        return presets

    for name in PRESET_NAMES:
        entry = saved.get(name)
        if not isinstance(entry, dict):
            continue
        # Merge onto the factory dict rather than replacing it, so a
        # preset file missing a key (older version, hand-edited) still
        # yields a complete, usable settings dict.
        for key, value in entry.items():
            if key in SETTING_KEYS and isinstance(value, (int, float)):
                presets[name][key] = int(value)

    return presets


def save_preset(name: str, settings: dict) -> bool:
    """
    Persist one preset's settings. Returns True on success.
    Reads-modifies-writes the whole file so saving one preset never
    discards the other two.
    """
    if name not in PRESET_NAMES:
        logging.error(f"camera_presets: unknown preset name {name!r}")
        return False

    existing = {}
    if PRESET_FILE.exists():
        try:
            with open(PRESET_FILE) as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                existing = loaded
        except Exception:
            # Corrupt file -- start fresh rather than refusing to save
            existing = {}

    existing[name] = {k: int(v) for k, v in settings.items()
                      if k in SETTING_KEYS}

    try:
        with open(PRESET_FILE, "w") as f:
            json.dump(existing, f, indent=2, sort_keys=True)
        return True
    except Exception as e:
        logging.error(f"camera_presets: could not write {PRESET_FILE} ({e})")
        return False


def reset_preset(name: str) -> bool:
    """
    Drop a user-saved preset so it reverts to factory defaults.
    Returns True on success (including when there was nothing saved).
    """
    if name not in PRESET_NAMES:
        return False
    if not PRESET_FILE.exists():
        return True

    try:
        with open(PRESET_FILE) as f:
            existing = json.load(f)
        if not isinstance(existing, dict):
            return True
        existing.pop(name, None)
        with open(PRESET_FILE, "w") as f:
            json.dump(existing, f, indent=2, sort_keys=True)
        return True
    except Exception as e:
        logging.error(f"camera_presets: could not reset {name} ({e})")
        return False


def is_customized(name: str) -> bool:
    """True if this preset has user-saved values differing from factory."""
    if not PRESET_FILE.exists():
        return False
    try:
        with open(PRESET_FILE) as f:
            saved = json.load(f)
        entry = saved.get(name)
        if not isinstance(entry, dict):
            return False
        factory = factory_preset(name)
        return any(factory.get(k) != v for k, v in entry.items()
                   if k in SETTING_KEYS)
    except Exception:
        return False


if __name__ == "__main__":
    import tempfile, os, sys

    # Running this file directly puts core/ on sys.path, not the
    # project root -- so `from core.dual_emeet_camera import ...`
    # inside _factory_presets() would fail. Same fix as
    # tools/demo_spray.py uses.
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    print("=" * 55)
    print("core/camera_presets.py — Self Test")
    print("=" * 55)

    # Redirect to a temp file so the real preset file is untouched
    _real = PRESET_FILE
    tmpdir = tempfile.mkdtemp()
    globals()["PRESET_FILE"] = Path(tmpdir) / "camera_presets.json"

    p = load_presets()
    assert set(p.keys()) == {"outdoor", "cloudy", "indoor"}
    for name in PRESET_NAMES:
        assert p[name] == factory_preset(name)
    print("✓ With no saved file, load_presets() returns factory defaults")
    assert not is_customized("outdoor")
    print("✓ is_customized() is False before any edit")

    custom = dict(p["outdoor"])
    custom["exposure"] = 42
    custom["auto_wb"]  = 1
    assert save_preset("outdoor", custom)
    print("✓ save_preset() wrote a customized outdoor preset")

    reloaded = load_presets()
    assert reloaded["outdoor"]["exposure"] == 42
    assert reloaded["outdoor"]["auto_wb"] == 1
    print("✓ Saved values survive a reload")

    assert reloaded["cloudy"] == factory_preset("cloudy")
    assert reloaded["indoor"] == factory_preset("indoor")
    print("✓ Saving one preset did not affect the other two")

    assert factory_preset("outdoor")["exposure"] != 42
    print("✓ Factory defaults remain unmodified (restore is always possible)")
    assert is_customized("outdoor")
    print("✓ is_customized() correctly reports the edited preset")

    assert reset_preset("outdoor")
    after = load_presets()
    assert after["outdoor"] == factory_preset("outdoor")
    print("✓ reset_preset() restores factory defaults")

    # Corrupt-file resilience
    with open(PRESET_FILE, "w") as f:
        f.write("{ this is not valid json ][")
    fallback = load_presets()
    assert fallback["indoor"] == factory_preset("indoor")
    print("✓ A corrupt preset file falls back to factory defaults "
          "instead of raising")

    # Partial/legacy file: only some keys present
    with open(PRESET_FILE, "w") as f:
        json.dump({"cloudy": {"exposure": 99}}, f)
    partial = load_presets()
    assert partial["cloudy"]["exposure"] == 99
    assert partial["cloudy"]["gamma"] == factory_preset("cloudy")["gamma"]
    print("✓ A partial preset file merges onto factory defaults "
          "(missing keys keep their factory values)")

    globals()["PRESET_FILE"] = _real
    print()
    print("=" * 55)
    print("core/camera_presets.py ✓ ALL TESTS PASSED")
    print("=" * 55)
