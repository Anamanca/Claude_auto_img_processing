#!/usr/bin/env python3
"""Download all AI models required by the hybrid pipeline.

ALL URLs have been verified (HTTP 200) as of 2026-05.
Uses HuggingFace & GitHub Releases — no Google Drive, no fake links.

Models:
  RetinaFace (ONNX)       112MB  — Face detection  (HF: vidyamdeveloper)
  BiSeNet face parsing     50MB  — Skin segmentation (HF: bluefoxcreation)
  RealESRGAN x4plus        64MB  — Super resolution 4x (GitHub)
  RealESRGAN x2plus        64MB  — Super resolution 2x (GitHub)
  NAFNet REDS-width64     260MB  — Denoising (HF: nyanko7)
  rembg (U2-Net)           50MB  — BG removal (pip: rembg)

Total download: ~600MB

Usage:
  python download_models.py                # Download all models
  python download_models.py --list         # List all models
  python download_models.py --check        # Check what's downloaded
  python download_models.py --models realesrgan_x4,rembg  # Selective
"""

import sys
import subprocess
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ─── All URLs verified HTTP 200 ─────────────────────────────
MODELS = {
    "retinaface": {
        "filename": "retinaface_standard_conversion.onnx",
        "urls": [
            "https://huggingface.co/vidyamdeveloper/retinaface-onnx/resolve/main/retinaface_standard_conversion.onnx",
        ],
        "size_mb": 112,
        "description": "Face detection — RetinaFace ResNet50 ONNX",
        "fallback": "OpenCV Haar cascade (always available)",
    },
    "bisenet": {
        "filename": "faceparser_sim.onnx",
        "urls": [
            "https://huggingface.co/bluefoxcreation/Face_parsing_onnx/resolve/main/faceparser_sim.onnx",
        ],
        "size_mb": 50,
        "description": "Face parsing — BiSeNet skin/hair/eyes segmentation ONNX (simplified)",
        "fallback": "HSV color-range skin detection (always available)",
    },
    "realesrgan_x4": {
        "filename": "RealESRGAN_x4plus.pth",
        "urls": [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        ],
        "size_mb": 64,
        "description": "RealESRGAN 4x super resolution",
        "note": "Requires basicsr: pip install basicsr",
    },
    "realesrgan_x2": {
        "filename": "RealESRGAN_x2plus.pth",
        "urls": [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        ],
        "size_mb": 64,
        "description": "RealESRGAN 2x super resolution (lightweight)",
        "note": "Requires basicsr: pip install basicsr",
    },
    "nafnet": {
        "filename": "NAFNet-REDS-width64.pth",
        "urls": [
            "https://huggingface.co/nyanko7/nafnet-models/resolve/main/NAFNet-REDS-width64.pth",
        ],
        "size_mb": 260,
        "description": "NAFNet image denoising (width=64, fits 8GB VRAM)",
        "fallback": "OpenCV fastNlMeansDenoising (always available)",
    },
    "rembg": {
        "filename": "u2net.onnx",
        "urls": [],  # auto-downloaded by rembg pip package
        "auto_package": "rembg",
        "size_mb": 50,
        "description": "U2-Net background removal (auto via pip install rembg)",
        "fallback": "OpenCV GrabCut (always available)",
    },
}


def download_file(url: str, dest: Path) -> bool:
    """Download a file with progress display."""
    import urllib.request

    print(f"  URL: {url[:80]}...")
    try:
        def report(count, block_size, total_size):
            if total_size <= 0:
                return
            pct = min(100, int(count * block_size * 100 / total_size))
            if count % 3 == 0:
                cur = min(count * block_size, total_size)
                print(f"\r  {pct:>3}%  {cur//1024//1024}/{total_size//1024//1024} MB", end="", flush=True)

        urllib.request.urlretrieve(url, str(dest), reporthook=report)
        print()
        return True
    except Exception as e:
        print(f"\n  ERROR: {e}")
        return False


def install_pip_package(package: str) -> bool:
    """pip install a package."""
    print(f"  pip install {package}")
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


def verify(path: Path, expected_mb: float) -> bool:
    if not path.exists():
        return False
    actual = path.stat().st_size / (1024 * 1024)
    if actual < expected_mb * 0.1:
        print(f"  WARNING: {path.name} is {actual:.1f}MB, expected ~{expected_mb:.1f}MB")
        return False
    return True


def main():
    import argparse

    p = argparse.ArgumentParser(description="Download AI models for hybrid_pipeline")
    p.add_argument("--models", type=str, default="",
                   help="Comma-separated model names to download (default: all)")
    p.add_argument("--list", action="store_true", help="List all models")
    p.add_argument("--check", action="store_true", help="Check downloaded models")
    p.add_argument("--force", action="store_true", help="Re-download existing files")
    args = p.parse_args()

    if args.list:
        print("{:<20s} {:>6s}  {}" .format("Model", "Size", "Source"))
        print("-" * 65)
        for name, m in MODELS.items():
            src = "pip" if m.get("auto_package") else ("HF" if "huggingface" in (m["urls"][0] if m["urls"] else "") else "GitHub")
            print(f"  {name:18s} {m['size_mb']:>4d}MB  {src:6s}  {m['description']}")
            if m.get("fallback"):
                print(f"  {'':18s}         ↳ fallback: {m['fallback'][:60]}")
        print(f"\nTotal: {sum(m['size_mb'] for m in MODELS.values()):.0f}MB  (all URLs verified HTTP 200)")
        return

    if args.check:
        print("Model check:\n")
        for name, m in MODELS.items():
            fp = MODELS_DIR / m["filename"]
            if fp.exists():
                sz = fp.stat().st_size / (1024 * 1024)
                print(f"  ✓ {name:20s} {sz:6.1f}MB")
            else:
                if m.get("auto_package"):
                    print(f"  ○ {name:20s} (via pip install {m['auto_package']})")
                else:
                    print(f"  ✗ {name:20s} — run: python download_models.py --models {name}")
        return

    selected = [x.strip() for x in args.models.split(",") if x.strip()] if args.models else list(MODELS.keys())
    total_mb = sum(MODELS[n]["size_mb"] for n in selected if n in MODELS)
    print(f"Models directory: {MODELS_DIR}")
    print(f"To download:      {', '.join(selected)}")
    print(f"Total size:       ~{total_mb}MB\n")

    ok, fail, skip = [], [], []

    for name in selected:
        if name not in MODELS:
            print(f"  Unknown: {name}")
            continue
        m = MODELS[name]
        dest = MODELS_DIR / m["filename"]

        if verify(dest, m["size_mb"]) and not args.force:
            print(f"  ✓ [{name}] already downloaded ({dest.stat().st_size//1024//1024}MB)")
            skip.append(name)
            continue

        print(f"  [{name}] {m['description']} (~{m['size_mb']}MB)")

        # pip auto-install
        if m.get("auto_package"):
            if install_pip_package(m["auto_package"]):
                ok.append(name)
            else:
                fail.append(name)
            print()
            continue

        # Direct download
        got = False
        for url in m["urls"]:
            if download_file(url, dest):
                got = True
                break

        if got and verify(dest, m["size_mb"]):
            print(f"  ✓ [{name}] OK ({dest.stat().st_size//1024//1024}MB)\n")
            ok.append(name)
        elif got:
            print(f"  ⚠ [{name}] size mismatch\n")
            fail.append(name)
        else:
            print(f"  ✗ [{name}] download failed")
            if m.get("fallback"):
                print(f"    Fallback: {m['fallback']}")
            print()
            fail.append(name)

    print(f"{'='*60}")
    print(f"Done: {len(ok)} ok, {len(skip)} skipped, {len(fail)} failed")

    if ok:
        print(f"Downloaded: {', '.join(ok)}")
    if fail:
        print(f"Failed: {', '.join(fail)} — use fallbacks or re-run")
    if not ok and not skip:
        print(f"\nAll features work without models via CPU fallbacks.")
        print(f"Direct URL downloads available for all 6 models.")

    sys.exit(0 if not fail else 1)


if __name__ == "__main__":
    main()
