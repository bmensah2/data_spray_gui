#!/usr/bin/env python3
"""
check_yolo_env.py
Dual RGB Detection System — Training Environment Diagnostic

Run this FIRST, before any real training, especially on the Jetson.
Answers exactly the questions that are easy to get wrong on a Jetson:

  1. Is `ultralytics` installed, and is it recent enough for YOLO26
     (released Jan 2026)?
  2. Is `torch` actually seeing the GPU? A plain `pip install torch` /
     `pip install ultralytics` on a Jetson pulls a generic x86-built
     wheel that has NO Jetson CUDA support -- it will "work" (import
     fine) but silently run everything on CPU, which is 10-50x slower
     for training and can turn a 2-hour training run into a 2-days training run.
     The correct wheel comes from NVIDIA's Jetson-specific PyTorch
     builds (matched to your JetPack/L4T version), not plain PyPI.
  3. Do yolov8n-seg / yolo11n-seg / yolo26n-seg pretrained weights
     actually resolve (network reachable, or already cached locally)?
  4. Basic sanity: can we construct a YOLO model object at all with
     the installed ultralytics version?

Usage
-----
    python check_yolo_env.py

Exits 0 if everything needed for training is in order, 1 otherwise,
so it's also safe to use as a pre-flight check in a script:
    python check_yolo_env.py || exit 1

Author : Bright Mensah | NDSU / Dual RGB Detection System
Path   : /media/pagsun/Transcend/phd_project/emeet_dual_cam/
"""

import shutil
import subprocess
import sys


def _ok(msg):
    print(f"  \u2713 {msg}")


def _warn(msg):
    print(f"  \u26a0 {msg}")


def _fail(msg):
    print(f"  \u2717 {msg}")


def check_ultralytics():
    """Returns (ok: bool, version: str|None)."""
    print("\n[1/4] ultralytics package")
    try:
        import ultralytics
    except ImportError:
        _fail("ultralytics is NOT installed.")
        print("       Install with: pip install -U ultralytics")
        return False, None

    version = getattr(ultralytics, "__version__", "unknown")
    _ok(f"ultralytics {version} installed")

    # YOLO26 needs a build that recognizes the yolo26*.yaml/pt configs.
    # ultralytics ships YOLO26 support from roughly the 8.4.x series
    # onward (it was released alongside/after that package line) --
    # rather than hardcode a brittle version-number cutoff that will
    # go stale, do the real test: try to resolve a yolo26n-seg config.
    try:
        from ultralytics.nn.tasks import yaml_model_load  # noqa: F401
        has_yolo26_cfg = True
    except Exception:
        has_yolo26_cfg = None  # inconclusive, not fatal

    try:
        from ultralytics import YOLO
        # Constructing from a .yaml (not .pt) doesn't need network
        # access or a download -- just checks the architecture is
        # known to this ultralytics version.
        YOLO("yolo26n-seg.yaml")
        _ok("this ultralytics version recognizes YOLO26 architecture")
    except Exception as e:
        _warn(
            f"could not construct a YOLO26 model from this ultralytics "
            f"version ({e}). You likely need a newer release: "
            f"pip install -U ultralytics"
        )
        return True, version  # ultralytics itself is fine, just old

    return True, version


def check_torch_cuda():
    """Returns (has_torch: bool, has_cuda: bool)."""
    print("\n[2/4] PyTorch + CUDA (GPU) availability")
    try:
        import torch
    except ImportError:
        _fail("torch is NOT installed (should come with ultralytics).")
        return False, False

    _ok(f"torch {torch.__version__} installed")

    cuda_ok = False
    try:
        cuda_ok = torch.cuda.is_available()
    except Exception as e:
        _warn(f"torch.cuda.is_available() raised: {e}")

    if cuda_ok:
        try:
            name = torch.cuda.get_device_name(0)
            _ok(f"CUDA available -- GPU: {name}")
        except Exception:
            _ok("CUDA available")
    else:
        _fail(
            "CUDA is NOT available to torch -- training/inference will "
            "run on CPU only."
        )
        print(
            "       On a Jetson this almost always means the pip-installed\n"
            "       torch wheel is a generic build without Jetson GPU\n"
            "       support. Fix: install the NVIDIA-provided PyTorch\n"
            "       wheel matched to your JetPack/L4T version instead of\n"
            "       plain `pip install torch`. Check your JetPack version\n"
            "       with `dpkg -l | grep nvidia-jetpack`, then get the\n"
            "       matching wheel from NVIDIA's Jetson Zoo / forums page\n"
            "       for that JetPack release, uninstall the generic torch\n"
            "       first (`pip uninstall torch torchvision`), then install\n"
            "       the Jetson-specific wheel."
        )

    return True, cuda_ok


def check_pretrained_weights():
    """Check whether the three seg-pretrained checkpoints can be
    constructed (network reachable to download, or already cached)."""
    print("\n[3/4] Pretrained seg weights reachability (yolov8n / yolo11n / yolo26n)")
    try:
        from ultralytics import YOLO
    except ImportError:
        _fail("skipped -- ultralytics not installed")
        return {}

    families = {
        "yolov8n-seg.pt": "YOLOv8",
        "yolo11n-seg.pt": "YOLO11",
        "yolo26n-seg.pt": "YOLO26",
    }
    results = {}
    for weights, label in families.items():
        try:
            YOLO(weights)
            _ok(f"{label}: {weights} resolved (downloaded or cached)")
            results[label] = True
        except Exception as e:
            _fail(f"{label}: {weights} failed to resolve -- {e}")
            results[label] = False
    return results


def check_disk_and_misc():
    print("\n[4/4] Disk space + misc")
    try:
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024 ** 3)
        if free_gb < 5:
            _warn(f"only {free_gb:.1f} GB free on this drive -- training "
                  f"runs (checkpoints, logs, cached weights) can use "
                  f"several GB per run, especially comparing 3 model "
                  f"families.")
        else:
            _ok(f"{free_gb:.1f} GB free disk space")
    except Exception as e:
        _warn(f"could not check disk space: {e}")

    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            _ok("nvidia-smi available (system-level GPU driver present)")
        else:
            _warn("nvidia-smi ran but returned an error")
    except FileNotFoundError:
        _warn(
            "nvidia-smi not found -- on a Jetson this is often normal "
            "(Jetson uses tegrastats instead), not necessarily a problem"
        )
    except Exception:
        pass


def main():
    print("=" * 60)
    print("  ABEN YOLO Training Environment Check")
    print("=" * 60)

    ultra_ok, ultra_version = check_ultralytics()
    torch_ok, cuda_ok = check_torch_cuda()
    weight_results = check_pretrained_weights()
    check_disk_and_misc()

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    problems = []
    if not ultra_ok:
        problems.append("ultralytics not installed")
    if not torch_ok:
        problems.append("torch not installed")
    if torch_ok and not cuda_ok:
        problems.append(
            "CUDA not available to torch -- training will be VERY slow "
            "on CPU, especially for 3 separate model families"
        )
    for label, resolved in weight_results.items():
        if not resolved:
            problems.append(f"{label} pretrained weights did not resolve")

    if not problems:
        print("\n  \u2713 Environment looks ready for training all three "
              "model families.")
        print(f"    ultralytics: {ultra_version}")
        print(f"    CUDA: {'available' if cuda_ok else 'not available'}")
        sys.exit(0)
    else:
        print("\n  Issues found:")
        for p in problems:
            print(f"    - {p}")
        print("\n  Fix these before running real training -- especially")
        print("  the CUDA one, since it silently makes training slow")
        print("  rather than failing outright.")
        sys.exit(1)


if __name__ == "__main__":
    main()
