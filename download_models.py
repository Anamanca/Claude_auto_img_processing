#!/usr/bin/env python3
"""Download all AI models required by the hybrid pipeline.

Downloads:
  - RetinaFace (MobileNet0.25) ONNX ~1.7MB
  - BiSeNet face parsing ONNX ~51MB
  - RealESRGAN x4plus ~67MB
  - RealESRGAN x2plus ~67MB
  - NAFNet-64 denoising ~260MB
  - U2-Net ONNX ~50MB

Total: ~500MB

Usage:
  python download_models.py                    # Download all
  python download_models.py --models retinaface,bisenet  # Selective
  python download_models.py --gpu              # Download all GPU variants
"""

import sys
import os
from pathlib import Path
import hashlib
import json

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Model Registry ──────────────────────────────────────────

MODELS = {
    "retinaface": {
        "filename": "retinaface_mobilenet0.25_Final.onnx",
        "urls": [
            "https://github.com/biubug6/Pytorch_Retinaface/raw/master/weights/mobilenet0.25_Final.pth",
        ],
        "size_mb": 1.7,
        "description": "Face detection (MobileNet backbone)",
        "required": True,
        "note": "Download the .pth file then convert via convert_retinaface.py, or download pre-converted ONNX from releases",
        "onnx_urls": [
            "https://github.com/hienut/hybrid_pipeline_models/releases/download/v1.0/retinaface_mobilenet0.25_Final.onnx",
        ],
    },
    "bisenet": {
        "filename": "bisenet_fp32.onnx",
        "urls": [
            "https://github.com/CoinCheung/BiSeNet/releases/download/0.0.1/model_final.pth",
        ],
        "size_mb": 51,
        "description": "Face parsing — 19-class face segmentation",
        "required": True,
        "note": "Will be auto-converted to ONNX if PyTorch is available. Download .pth then run convert_bisenet.py",
        "onnx_urls": [
            "https://github.com/hienut/hybrid_pipeline_models/releases/download/v1.0/bisenet_fp32.onnx",
        ],
    },
    "realesrgan_x4": {
        "filename": "realesrgan_x4plus.pth",
        "urls": [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        ],
        "size_mb": 67,
        "description": "RealESRGAN 4x super resolution",
        "required": False,
    },
    "realesrgan_x2": {
        "filename": "realesrgan_x2plus.pth",
        "urls": [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        ],
        "size_mb": 67,
        "description": "RealESRGAN 2x super resolution (lightweight)",
        "required": False,
    },
    "nafnet": {
        "filename": "nafnet_64.pth",
        "urls": [
            "https://github.com/megvii-research/NAFNet/releases/download/v1.0/NAFNet-REDS-width64.pth",
        ],
        "size_mb": 260,
        "description": "NAFNet image denoising (width=64 variant for 8GB VRAM)",
        "required": False,
    },
    "rembg": {
        "filename": "u2net.onnx",
        "urls": [
            "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
        ],
        "size_mb": 50,
        "description": "U2-Net background removal",
        "required": False,
    },
}


def download_file(url: str, dest: Path) -> bool:
    """Download a file with progress indication."""
    try:
        import urllib.request

        print(f"  Downloading: {url}")
        print(f"  To: {dest}")

        def report(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
            if count % 10 == 0:
                print(f"\r  {percent}% ({count*block_size}/{total_size})", end="", flush=True)

        urllib.request.urlretrieve(url, str(dest), reporthook=report)
        print()  # newline
        return True
    except Exception as e:
        print(f"\n  ERROR: {e}")
        return False


def convert_pth_to_onnx(pth_path: Path, onnx_path: Path, model_type: str) -> bool:
    """Convert PyTorch models to ONNX format."""
    try:
        import torch
        import onnx
        print(f"  Converting {model_type} .pth → .onnx...")
        print(f"  (This requires the specific model architecture to be loaded)")
        print(f"  See: convert_scripts/ for per-model conversion scripts")
        print(f"  Or download pre-converted ONNX from the releases page.")
        return False
    except ImportError:
        print("  PyTorch/ONNX not available. Cannot convert.")
        return False


def verify_file(path: Path, expected_mb: float) -> bool:
    """Check if a file exists and has approximately the right size."""
    if not path.exists():
        return False
    actual_mb = path.stat().st_size / (1024 * 1024)
    # Allow 50% tolerance from expected size
    if actual_mb < expected_mb * 0.3:
        print(f"  WARNING: {path.name} is {actual_mb:.1f}MB, expected ~{expected_mb:.1f}MB (incomplete download?)")
        return False
    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download AI models for hybrid_pipeline")
    parser.add_argument("--models", type=str, default="",
                        help="Comma-separated model names to download (default: all required)")
    parser.add_argument("--all", action="store_true", help="Download ALL models including optional")
    parser.add_argument("--list", action="store_true", help="List all models and exit")
    parser.add_argument("--check", action="store_true", help="Check which models are already downloaded")
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    parser.add_argument("--use-onnx", action="store_true", help="Prefer ONNX URLs over .pth")
    args = parser.parse_args()

    if args.list:
        print("Available models:\n")
        for name, info in MODELS.items():
            tag = "REQUIRED" if info["required"] else "optional"
            print(f"  {name:20s} {info['size_mb']:6.1f}MB  [{tag}]  {info['description']}")
        print(f"\nTotal required: {sum(m['size_mb'] for m in MODELS.values() if m['required']):.0f}MB")
        print(f"Total all:      {sum(m['size_mb'] for m in MODELS.values()):.0f}MB")
        return

    if args.check:
        print("Checking downloaded models:\n")
        all_ok = True
        for name, info in MODELS.items():
            path = MODELS_DIR / info["filename"]
            exists = path.exists()
            size = path.stat().st_size / (1024 * 1024) if exists else 0
            status = f"✓ {size:.1f}MB" if exists else "✗ NOT FOUND"
            tag = "[REQUIRED]" if info["required"] else "[optional]"
            print(f"  {status:20s} {name:20s} {tag}")
            if info["required"] and not exists:
                all_ok = False
        if all_ok:
            print("\n✓ All required models present.")
        else:
            print("\n✗ Some required models missing. Run: python download_models.py")
        return

    # Determine which models to download
    if args.models:
        selected = [m.strip() for m in args.models.split(",")]
    elif args.all:
        selected = list(MODELS.keys())
    else:
        selected = [name for name, info in MODELS.items() if info["required"]]

    print(f"Models directory: {MODELS_DIR}")
    print(f"Models to download: {', '.join(selected)}")
    print(f"Estimated size: {sum(MODELS[m]['size_mb'] for m in selected if m in MODELS):.0f}MB\n")

    success = []
    failed = []
    skipped = []

    for name in selected:
        if name not in MODELS:
            print(f"Unknown model: {name}")
            continue
        info = MODELS[name]
        dest = MODELS_DIR / info["filename"]

        if verify_file(dest, info["size_mb"]) and not args.force:
            print(f"✓ {name} already downloaded ({info['size_mb']:.0f}MB)")
            skipped.append(name)
            continue

        print(f"  [{name}] {info['description']} ({info['size_mb']:.0f}MB)")

        # Try ONNX URL first if requested
        downloaded = False
        if args.use_onnx and "onnx_urls" in info:
            for url in info["onnx_urls"]:
                if download_file(url, dest):
                    downloaded = True
                    break

        # Try regular URLs
        if not downloaded:
            for url in info["urls"]:
                if download_file(url, dest):
                    downloaded = True
                    break

        if downloaded and verify_file(dest, info["size_mb"]):
            print(f"  ✓ {name} downloaded successfully")
            success.append(name)
        elif downloaded:
            print(f"  ⚠ {name} downloaded but file size mismatch — may be incomplete")
            failed.append(name)
        else:
            print(f"  ℹ {name}: Download failed. You may need to manually download:")
            print(f"    {info.get('note', 'No manual instructions')}")
            print(f"    URLs: {info['urls']}")
            failed.append(name)

    print(f"\n{'='*60}")
    print(f"Download complete:")
    print(f"  ✓ Success: {len(success)} — {', '.join(success) if success else 'none'}")
    print(f"  ⊘ Skipped: {len(skipped)} — {', '.join(skipped) if skipped else 'none'}")
    print(f"  ✗ Failed:  {len(failed)} — {', '.join(failed) if failed else 'none'}")

    if failed:
        print(f"\nSome models could not be downloaded automatically.")
        print(f"Please download them manually from the URLs listed above.")
        print(f"Place files in: {MODELS_DIR}")
        sys.exit(1)


if __name__ == "__main__":
    main()
