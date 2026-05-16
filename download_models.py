#!/usr/bin/env python3
"""Download all AI models required by the hybrid pipeline.

All models are OPTIONAL — ai_server.py has CPU fallbacks for every feature.
Face detection defaults to OpenCV Haar cascade, skin masking to HSV,
denoising to OpenCV, bg removal to GrabCut. No download is required.

Downloads available:
  - RealESRGAN x4plus ~67MB  (working GitHub release link)
  - RealESRGAN x2plus ~67MB  (working GitHub release link)

Models from Google Drive (manual download only):
  - RetinaFace MobileNet0.25  ~1.7MB
  - BiSeNet face parsing      ~51MB
  - NAFNet-64 denoising       ~260MB

Auto-download via pip:
  - rembg (U2-Net)            ~50MB   (pip install rembg)

Usage:
  python download_models.py                # Download all with working URLs
  python download_models.py --list         # List all models
  python download_models.py --check        # Check what's downloaded
  python download_models.py --all          # Try all models including gdrive
"""

import sys
import os
from pathlib import Path
import hashlib
import json
import subprocess

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Model Registry ──────────────────────────────────────────
# status: "url_ok" = direct download works
#         "gdrive"  = Google Drive only, needs manual download
#         "pip"     = auto-installed via pip package
#         "unknown" = source unknown

MODELS = {
    "retinaface": {
        "filename": "retinaface_mobilenet0.25_Final.pth",
        "size_mb": 1.7,
        "description": "Face detection (MobileNet0.25 backbone)",
        "status": "gdrive",
        "urls": [
            "https://drive.google.com/uc?export=download&id=1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1",
        ],
        "note": "Google Drive download. Also at: https://github.com/biubug6/Pytorch_Retinaface (weights dir)",
        "fallback": "OpenCV Haar cascade — always available, no download needed",
    },
    "bisenet": {
        "filename": "bisenet_fp32.onnx",
        "size_mb": 51,
        "description": "Face parsing — 19-class face segmentation",
        "status": "gdrive",
        "urls": [
            "https://drive.google.com/uc?export=download&id=154JgAn7dWrxKQpy3qQ7Yv0esNqzw4H_d",
        ],
        "note": "Google Drive download from https://github.com/CoinCheung/BiSeNet",
        "fallback": "HSV color-range skin detection — always available",
    },
    "realesrgan_x4": {
        "filename": "RealESRGAN_x4plus.pth",
        "size_mb": 67,
        "description": "RealESRGAN 4x super resolution",
        "status": "url_ok",
        "urls": [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        ],
        "note": "Requires basicsr: pip install basicsr",
        "fallback": "None — super resolution requires this model",
    },
    "realesrgan_x2": {
        "filename": "RealESRGAN_x2plus.pth",
        "size_mb": 67,
        "description": "RealESRGAN 2x super resolution (lightweight)",
        "status": "url_ok",
        "urls": [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        ],
        "note": "Requires basicsr: pip install basicsr",
        "fallback": "None — super resolution requires this model",
    },
    "nafnet": {
        "filename": "nafnet_64.pth",
        "size_mb": 260,
        "description": "NAFNet image denoising (width=64, fits 8GB VRAM)",
        "status": "gdrive",
        "urls": [
            "https://drive.google.com/uc?export=download&id=1T2zK3xIXS3WKFNkBLgLqCxPnTfFJNr7J",
        ],
        "note": "Google Drive download from https://github.com/megvii-research/NAFNet (no GitHub releases)",
        "fallback": "OpenCV fastNlMeansDenoising — always available",
    },
    "rembg": {
        "filename": "u2net.onnx",
        "size_mb": 50,
        "description": "U2-Net background removal",
        "status": "pip",
        "urls": [],
        "auto_package": "rembg",
        "note": "Run: pip install rembg  (auto-downloads model on first use)",
        "fallback": "OpenCV GrabCut — always available",
    },
}


def download_direct(url: str, dest: Path) -> bool:
    """Download a file with progress indication."""
    try:
        import urllib.request

        print(f"  Downloading: {url}")

        def report(count, block_size, total_size):
            if total_size <= 0:
                return
            percent = min(100, int(count * block_size * 100 / total_size))
            if count % 5 == 0:
                downloaded_mb = min(count * block_size, total_size) // 1024 // 1024
                total_mb = total_size // 1024 // 1024
                print(f"\r  {percent}% ({downloaded_mb}/{total_mb} MB)", end="", flush=True)

        urllib.request.urlretrieve(url, str(dest), reporthook=report)
        print()
        return True
    except Exception as e:
        print(f"\n  ERROR: {e}")
        return False


def install_pip_package(package: str) -> bool:
    """Install a pip package."""
    print(f"  Running: pip install {package}")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", package],
            check=True, capture_output=True, text=True,
        )
        print(f"  OK: '{package}' installed (model auto-downloads on first use)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: {e.stderr[-300:] if e.stderr else e}")
        return False


def verify_file(path: Path, expected_mb: float) -> bool:
    if not path.exists():
        return False
    actual = path.stat().st_size / (1024 * 1024)
    if actual < expected_mb * 0.1:
        print(f"  WARNING: {path.name} is {actual:.1f}MB, expected ~{expected_mb:.1f}MB (incomplete?)")
        return False
    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download AI models for hybrid_pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_models.py              # Download models with working URLs
  python download_models.py --all        # Try ALL models (include gdrive)
  python download_models.py --models realesrgan_x4,rembg
  python download_models.py --list       # List models and status
  python download_models.py --check      # Show what's already downloaded
        """,
    )
    parser.add_argument("--models", type=str, default="",
                        help="Comma-separated model names to download")
    parser.add_argument("--all", action="store_true", help="Try ALL models including Google Drive")
    parser.add_argument("--list", action="store_true", help="List all models and exit")
    parser.add_argument("--check", action="store_true", help="Check which models are downloaded")
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    args = parser.parse_args()

    if args.list:
        print("{:<20s} {:>7s} {:8s} {:s}".format("Model", "Size", "Status", "Description"))
        print("-" * 80)
        for name, info in MODELS.items():
            status_str = {
                "url_ok": "✓ URL",
                "gdrive": "GDrive",
                "pip": "pip",
                "unknown": "?",
            }.get(info.get("status", "?"), info.get("status", "?"))
            print(f"  {name:18s} {info['size_mb']:4.0f}MB  {status_str:6s}  {info['description']}")
            if info.get("fallback") and info["fallback"] != "None — super resolution requires this model":
                print(f"  {'':18s}         ↳ fallback: {info['fallback'][:78]}")
        print(f"\nStatus key: ✓ URL = direct download works | GDrive = Google Drive (manual)")
        print(f"All models are OPTIONAL. Every feature has a CPU fallback in ai_server.py.")
        return

    if args.check:
        print("Model check:\n")
        for name, info in MODELS.items():
            path = MODELS_DIR / info["filename"]
            if path.exists():
                size = path.stat().st_size / (1024 * 1024)
                print(f"  ✓ {name:20s} {size:6.1f}MB  {info['filename']}")
            else:
                status = info.get("status", "")
                if status == "pip" and info.get("auto_package"):
                    print(f"  ○ {name:20s} (auto via pip install {info['auto_package']})")
                elif status == "url_ok":
                    print(f"  ✗ {name:20s} — run: python download_models.py --models {name}")
                else:
                    print(f"  ✗ {name:20s} — {info.get('note', 'manual download')[:60]}")
        print(f"\nAll features work without models via CPU fallbacks.")
        print(f"Only ai_super_resolution() requires RealESRGAN model.")
        return

    # Determine which models to download
    if args.models:
        selected = [m.strip() for m in args.models.split(",")]
    elif args.all:
        selected = [name for name, info in MODELS.items() if info.get("urls")]
    else:
        # Default: only models with direct download URLs (status=url_ok)
        selected = [name for name, info in MODELS.items() if info.get("status") == "url_ok"]

    print(f"Models directory: {MODELS_DIR}")
    print(f"To download: {', '.join(selected) if selected else 'none (try --all or --list)'}")
    print()

    success, failed, skipped = [], [], []

    for name in selected:
        if name not in MODELS:
            print(f"  Unknown: {name}")
            continue
        info = MODELS[name]
        dest = MODELS_DIR / info["filename"]

        if verify_file(dest, info["size_mb"]) and not args.force:
            size = dest.stat().st_size // 1024 // 1024
            print(f"  ✓ [{name}] already downloaded ({size}MB)")
            skipped.append(name)
            continue

        print(f"  [{name}] {info['description']} ({info['size_mb']:.0f}MB)")

        # pip auto-install
        if info.get("auto_package"):
            if install_pip_package(info["auto_package"]):
                success.append(name)
            else:
                failed.append(name)
            continue

        # Direct download
        downloaded = False
        for url in info.get("urls", []):
            if download_direct(url, dest):
                downloaded = True
                break

        if not downloaded and info.get("status") == "gdrive":
            print(f"  ℹ  Google Drive download failed (common — needs browser auth)")
            print(f"  ℹ  Manual download: open this URL in browser:")
            for url in info["urls"]:
                print(f"       {url}")
            print(f"  ℹ  Save as: {dest}")
            if info.get("fallback"):
                print(f"  ℹ  Fallback: {info['fallback']}")
            skipped.append(name)
            continue

        if downloaded and verify_file(dest, info["size_mb"]):
            print(f"  ✓ [{name}] downloaded")
            success.append(name)
        elif downloaded:
            print(f"  ⚠ [{name}] file size mismatch")
            failed.append(name)
        else:
            print(f"  ✗ [{name}] all download attempts failed")
            if info.get("fallback"):
                print(f"    Fallback available: {info['fallback']}")
            failed.append(name)

    print(f"\n{'='*60}")
    print(f"Done: {len(success)} ok, {len(skipped)} skipped, {len(failed)} failed")

    if not success and not skipped:
        print(f"\nNo models downloaded. This is OK — ai_server.py has CPU fallbacks.")
        print(f"The only feature that needs a model is ai_super_resolution().")
        print(f"To download RealESRGAN (working URL):")
        print(f"  python download_models.py --models realesrgan_x4")
        print(f"\nTo install background removal:")
        print(f"  pip install rembg")

    sys.exit(0)


if __name__ == "__main__":
    main()
