#!/usr/bin/env python3
"""Utility MCP Server — RAW development, EXIF, file management, image comparison.

Wraps rawpy (LibRaw), exiftool, Pillow, scikit-image, pathlib.
"""

from mcp.server.fastmcp import FastMCP
from pathlib import Path
import json
import os
from typing import Optional

from common import (
    load_config, resolve_path, run_cmd, setup_logging,
    TempManager, check_binary, file_hash, ensure_dir, is_image_ext,
)

cfg = load_config()
log = setup_logging(cfg, "util_server")
TEMP = TempManager(cfg)

EXIFTOOL = "exiftool"

mcp = FastMCP(
    "util-mcp",
    instructions="Utility MCP server — RAW development, EXIF read/write, file management, image comparison.",
)


# ─── RAW DEVELOPMENT ─────────────────────────────────────────

@mcp.tool()
def util_raw_develop(input_path: str, output_path: str,
                     demosaic_algorithm: str = "AMAZE",
                     use_camera_wb: bool = True,
                     brightness: float = 1.0,
                     highlight_mode: int = 1,
                     exp_shift: float = 0.0,
                     output_color_space: str = "sRGB",
                     output_bps: int = 16,
                     auto_bright: bool = True,
                     median_filter_passes: int = 1) -> str:
    """Develop a RAW file (CR2/NEF/ARW/DNG/RW2) to 16-bit TIFF using rawpy (LibRaw).

    demosaic_algorithm: AMAZE, AHD, AAHD, LMMSE, PPG
    highlight_mode: 0=clip, 1=unclip, 2=blend
    output_color_space: sRGB, Adobe, XYZ, ProPhoto, ACES
    auto_bright: auto brightness (thx)
    """
    try:
        import rawpy
        from PIL import Image
        import numpy as np
    except ImportError:
        raise RuntimeError("Missing dependencies. Install: pip install rawpy Pillow numpy")

    if not Path(input_path).exists():
        raise FileNotFoundError(f"RAW file not found: {input_path}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with rawpy.imread(input_path) as raw:
        rgb = raw.postprocess(
            demosaic_algorithm=getattr(rawpy.DemosaicAlgorithm, demosaic_algorithm, rawpy.DemosaicAlgorithm.AMAZE),
            use_camera_wb=use_camera_wb,
            brightness=int(brightness * 1000),
            highlight_mode=highlight_mode,
            exp_shift=exp_shift,
            output_color=getattr(rawpy.ColorSpace, output_color_space, rawpy.ColorSpace.sRGB),
            output_bps=output_bps,
            auto_bright_thr=0.01 if auto_bright else None,
            median_filter_passes=median_filter_passes,
        )

    if output_bps == 16:
        img = Image.fromarray(rgb)
        img.save(str(output), format="TIFF", compression="lzw")
    else:
        img = Image.fromarray(rgb)
        img.save(str(output), format="TIFF", quality=95)

    size_mb = output.stat().st_size / (1024 * 1024)
    return json.dumps({
        "output": str(output),
        "size_mb": round(size_mb, 2),
        "colorspace": output_color_space,
        "bps": output_bps,
        "status": "ok",
    })


@mcp.tool()
def util_raw_to_jpeg(input_path: str, output_path: str,
                     demosaic_algorithm: str = "AMAZE",
                     use_camera_wb: bool = True,
                     quality: int = 92,
                     auto_bright: bool = True) -> str:
    """Develop RAW directly to JPEG (shorthand for raw_develop + convert)."""
    try:
        import rawpy
        from PIL import Image
    except ImportError:
        raise RuntimeError("Missing dependencies. Install: pip install rawpy Pillow")

    with rawpy.imread(input_path) as raw:
        rgb = raw.postprocess(
            demosaic_algorithm=getattr(rawpy.DemosaicAlgorithm, demosaic_algorithm, rawpy.DemosaicAlgorithm.AMAZE),
            use_camera_wb=use_camera_wb,
            output_color=rawpy.ColorSpace.sRGB,
            output_bps=8,
            auto_bright_thr=0.01 if auto_bright else None,
        )

    img = Image.fromarray(rgb)
    img.save(output_path, format="JPEG", quality=quality, progressive=True)

    return json.dumps({
        "output": output_path,
        "quality": quality,
        "status": "ok",
    })


@mcp.tool()
def util_raw_metadata(input_path: str) -> str:
    """Read camera EXIF from RAW file (ISO, aperture, shutter, lens, WB, GPS)."""
    try:
        import rawpy
    except ImportError:
        raise RuntimeError("Missing dependency: pip install rawpy")

    with rawpy.imread(input_path) as raw:
        info = {
            "file": input_path,
            "camera_model": raw.camera_whitebalance.camera_model or "",
            "iso": raw.iso_speed,
            "shutter": float(raw.shutter) if raw.shutter > 0 else 0,
            "aperture": float(raw.aperture) if raw.aperture > 0 else 0,
            "focal_length": float(raw.focal_length) if raw.focal_length > 0 else 0,
            "lens": raw.lens or "",
            "color_desc": raw.color_desc.decode() if isinstance(raw.color_desc, bytes) else str(raw.color_desc),
            "sizes": {
                "raw_width": raw.sizes.raw_width,
                "raw_height": raw.sizes.raw_height,
                "width": raw.sizes.width,
                "height": raw.sizes.height,
            },
            "num_colors": raw.num_colors,
            "white_level": raw.white_level,
        }
    return json.dumps(info, indent=2)


# ─── EXIF ────────────────────────────────────────────────────

@mcp.tool()
def util_exif_read(input_path: str) -> str:
    """Read EXIF/XMP/IPTC metadata using exiftool. Returns JSON."""
    r = run_cmd([EXIFTOOL, "-json", "-g", input_path], timeout=30)
    if r["returncode"] != 0:
        raise RuntimeError(f"util_exif_read failed: {r['stderr']}")
    return r["stdout"]


@mcp.tool()
def util_exif_write(input_path: str, metadata_json: str) -> str:
    """Write metadata to image using exiftool. metadata_json: '{"Copyright": "...", "Artist": "..."}'."""
    meta = json.loads(metadata_json)
    args = [EXIFTOOL, "-overwrite_original"]
    for key, val in meta.items():
        args.append(f"-{key}={val}")
    args.append(input_path)
    r = run_cmd(args, timeout=30)
    if r["returncode"] != 0:
        raise RuntimeError(f"util_exif_write failed: {r['stderr']}")
    return json.dumps({"file": input_path, "updated": list(meta.keys()), "status": "ok"})


# ─── IMAGE COMPARISON ────────────────────────────────────────

@mcp.tool()
def util_compare_images(input_a: str, input_b: str) -> str:
    """Compare two images: PSNR, SSIM, MSE. Returns quality metrics."""
    try:
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import peak_signal_noise_ratio as psnr
        from skimage import io as skio
        import numpy as np
    except ImportError:
        raise RuntimeError("Missing dependencies. Install: pip install scikit-image numpy")

    a = skio.imread(input_a)
    b = skio.imread(input_b)

    if a.shape != b.shape:
        from skimage.transform import resize
        b = resize(b, a.shape[:2], anti_aliasing=True)
        b = (b * 255).astype(np.uint8) if b.max() <= 1.0 else b.astype(np.uint8)

    data_range = 255 if a.max() > 1 else 1.0

    return json.dumps({
        "psnr": round(psnr(a, b, data_range=data_range), 2),
        "ssim": round(ssim(a, b, channel_axis=-1, data_range=data_range), 4),
        "mse": round(float(np.mean((a.astype(float) - b.astype(float)) ** 2)), 2),
        "file_a": input_a,
        "file_b": input_b,
    }, indent=2)


# ─── FILE MANAGEMENT ─────────────────────────────────────────

@mcp.tool()
def util_file_list(
    input_dir: str, extensions: str = ".jpg,.jpeg,.png,.tiff,.cr2,.nef,.arw,.mp4,.mov,.mkv",
    recursive: bool = True, min_size_kb: int = 0, max_size_kb: int = 0,
    sort_by: str = "name",
) -> str:
    """List media files in directory. extensions: comma-separated. sort_by: name, size, date."""
    exts = set(e.strip().lower() for e in extensions.split(",") if e.strip())
    if not exts:
        exts = {".jpg", ".jpeg", ".png"}

    files = []
    root = Path(input_dir)
    if not root.exists():
        return json.dumps({"error": f"Directory not found: {input_dir}"})

    iterator = root.rglob("*") if recursive else root.glob("*")
    for f in iterator:
        if f.is_file() and f.suffix.lower() in exts:
            size_kb = f.stat().st_size // 1024
            if min_size_kb > 0 and size_kb < min_size_kb:
                continue
            if max_size_kb > 0 and size_kb > max_size_kb:
                continue
            files.append({
                "path": str(f),
                "name": f.name,
                "size_kb": size_kb,
                "mtime": f.stat().st_mtime,
            })

    if sort_by == "size":
        files.sort(key=lambda x: x["size_kb"])
    elif sort_by == "date":
        files.sort(key=lambda x: x["mtime"], reverse=True)
    else:
        files.sort(key=lambda x: x["name"])

    return json.dumps({
        "directory": input_dir,
        "total": len(files),
        "files": files[:500],
        "truncated": len(files) > 500,
    }, indent=2)


@mcp.tool()
def util_hash(input_path: str, algo: str = "sha256") -> str:
    """Compute file hash for deduplication."""
    h = file_hash(input_path, algo)
    return json.dumps({"file": input_path, "algorithm": algo, "hash": h})


@mcp.tool()
def util_organize(
    input_dir: str, output_dir: str,
    pattern: str = "{date:%Y}/{date:%m}/{category}/{filename}",
    category_map_json: str = '{"portrait": ["jpg","jpeg"], "video": ["mp4","mov"]}',
) -> str:
    """Organize files into a structured directory by date/category.

    pattern variables: {date}, {year}, {month}, {day}, {category}, {camera}, {filename}, {ext}
    """
    from datetime import datetime

    ensure_dir(Path(output_dir))
    exts = {"jpg", "jpeg", "png", "tiff", "cr2", "nef", "arw", "mp4", "mov", "mkv", "webm"}
    root = Path(input_dir)
    results = {"moved": 0, "errors": 0, "details": []}

    for f in root.rglob("*"):
        if not f.is_file() or f.suffix.lower().lstrip(".") not in exts:
            continue

        try:
            r = run_cmd([EXIFTOOL, "-DateTimeOriginal", "-json", str(f)], timeout=15)
            date = None
            camera = ""
            if r["returncode"] == 0 and r["stdout"].strip():
                meta = json.loads(r["stdout"])
                if meta:
                    dt_str = meta[0].get("DateTimeOriginal", "")
                    if dt_str:
                        date = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")

            if date is None:
                date = datetime.fromtimestamp(f.stat().st_mtime)

            ext_lower = f.suffix.lower().lstrip(".")
            if ext_lower in ("jpg", "jpeg", "png", "tiff", "webp"):
                category = "images"
            elif ext_lower in ("mp4", "mov", "mkv", "webm"):
                category = "video"
            elif ext_lower in ("cr2", "nef", "arw", "dng"):
                category = "raw"
            else:
                category = "other"

            dest_path = Path(output_dir) / pattern.format(
                date=date,
                year=date.strftime("%Y"),
                month=date.strftime("%m"),
                day=date.strftime("%d"),
                category=category,
                camera=camera or "unknown",
                filename=f.stem,
                ext=f.suffix.lstrip("."),
            )
            dest_path = dest_path.with_suffix(f.suffix)
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            if dest_path.exists():
                dest_path = dest_path.with_stem(f"{f.stem}_{file_hash(str(f))[:6]}")

            os.rename(str(f), str(dest_path))
            results["moved"] += 1
            results["details"].append({"from": str(f), "to": str(dest_path), "date": date.isoformat()})
        except Exception as e:
            results["errors"] += 1
            results["details"].append({"from": str(f), "error": str(e)})

    return json.dumps(results, indent=2)


if __name__ == "__main__":
    log.info("Starting Util MCP server")
    exif_path = check_binary(EXIFTOOL)
    log.info("exiftool found at: %s", exif_path)
    if not exif_path:
        log.warning("exiftool not found — EXIF tools will fail. Install: sudo apt install exiftool")
    mcp.run()
