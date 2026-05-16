#!/usr/bin/env python3
"""ImageMagick MCP Server — 32 tools for batch image processing.

Wraps ImageMagick CLI (convert / mogrify / identify) as MCP tools.
All tools accept and return file paths, never base64.
"""

from mcp.server.fastmcp import FastMCP
from pathlib import Path
import json
import re
from typing import Optional

from common import (
    load_config, resolve_path, run_cmd, setup_logging,
    TempManager, check_binary, file_hash, ensure_dir, is_image_ext,
)

cfg = load_config()
log = setup_logging(cfg, "im_server")
TEMP = TempManager(cfg)

IMG = cfg["imagemagick"]["binary"]
IDENTIFY = cfg["imagemagick"]["identify_binary"]
MOGRIFY = cfg["imagemagick"]["mogrify_binary"]

mcp = FastMCP(
    "imagemagick-mcp",
    instructions="ImageMagick MCP server — batch image processing: resize, crop, color, filter, composite. All operations use file paths.",
)


def _im(*args, timeout: int = 300) -> dict:
    cmd = [IMG] + list(args)
    return run_cmd(cmd, timeout=timeout, log=log)


def _identify(*args, timeout: int = 60) -> dict:
    cmd = [IDENTIFY] + list(args)
    return run_cmd(cmd, timeout=timeout, log=log)


# ─── TRANSFORM ───────────────────────────────────────────────

@mcp.tool()
def im_resize(
    input_path: str,
    output_path: str,
    width: int,
    height: int = 0,
    keep_aspect: bool = True,
    filter_type: str = "Lanczos",
) -> str:
    """Resize an image. If height=0, auto from aspect ratio. filter: Lanczos, Mitchell, CatmullRom, Cubic..."""
    geom = f"{width}x{height}" if height else f"{width}"
    if keep_aspect and height == 0:
        geom = f"{width}x"
    args = [input_path, "-filter", filter_type, "-resize", geom]
    if not keep_aspect:
        args += ["!"]
    args.append(output_path)
    r = _im(*args)
    if r["returncode"] != 0:
        raise RuntimeError(f"im_resize failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def im_crop(input_path: str, output_path: str, width: int, height: int, x: int = 0, y: int = 0) -> str:
    """Crop a rectangular region (WxH+X+Y) from the image."""
    r = _im(input_path, "-crop", f"{width}x{height}+{x}+{y}", "+repage", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"im_crop failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def im_rotate(input_path: str, output_path: str, degrees: float, background: str = "white") -> str:
    """Rotate image by degrees. Background color fills exposed areas."""
    r = _im(input_path, "-background", background, "-rotate", str(degrees), output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"im_rotate failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def im_flip(input_path: str, output_path: str, direction: str = "vertical") -> str:
    """Flip image. direction: 'vertical' (-flip) or 'horizontal' (-flop)."""
    op = "-flip" if direction == "vertical" else "-flop"
    r = _im(input_path, op, output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"im_flip failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def im_trim(input_path: str, output_path: str, fuzz_percent: int = 5) -> str:
    """Auto-trim borders of similar color. fuzz_percent: color tolerance."""
    r = _im(input_path, "-fuzz", f"{fuzz_percent}%", "-trim", "+repage", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"im_trim failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def im_distort(
    input_path: str, output_path: str,
    method: str = "Perspective",
    coords: str = "0,0 0,0 100,0 100,0 0,100 0,100 100,100 100,100",
) -> str:
    """Distort image. method: Perspective, Affine, Arc, Barrel, Polar, Shepards... coords: space-separated pairs."""
    r = _im(input_path, "-distort", method, coords, output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"im_distort failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


# ─── COLOR ───────────────────────────────────────────────────

@mcp.tool()
def im_brightness_contrast(
    input_path: str, output_path: str,
    brightness: int = 0, contrast: int = 0,
) -> str:
    """Adjust brightness (-100 to 100) and contrast (-100 to 100)."""
    r = _im(input_path, "-brightness-contrast", f"{brightness}x{contrast}", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"im_brightness_contrast failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def im_modulate(
    input_path: str, output_path: str,
    brightness: float = 100.0, saturation: float = 100.0, hue: float = 100.0,
) -> str:
    """Vary brightness, saturation, hue. 100 = no change. brightness/saturation in %, hue in degrees."""
    r = _im(input_path, "-modulate", f"{brightness},{saturation},{hue}", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"im_modulate failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def im_level(
    input_path: str, output_path: str,
    black_point: float = 0.0, gamma: float = 1.0, white_point: float = 255.0,
) -> str:
    """Adjust levels: black_point, gamma, white_point (0-255 for 8-bit)."""
    r = _im(input_path, "-level", f"{black_point}x{white_point},{gamma}", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"im_level failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def im_auto_level(input_path: str, output_path: str) -> str:
    """Automatically adjust color levels."""
    r = _im(input_path, "-auto-level", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"im_auto_level failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def im_equalize(input_path: str, output_path: str) -> str:
    """Histogram equalization - enhance contrast."""
    r = _im(input_path, "-equalize", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"im_equalize failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def im_grayscale(input_path: str, output_path: str, method: str = "Rec709Luma") -> str:
    """Convert to grayscale. method: Rec709Luma, Rec601Luma, Average, Lightness."""
    r = _im(input_path, "-grayscale", method, output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"im_grayscale failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def im_sepia(input_path: str, output_path: str, threshold: int = 80) -> str:
    """Apply sepia tone effect (threshold as percentage)."""
    r = _im(input_path, "-sepia-tone", f"{threshold}%", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"im_sepia failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def im_negate(input_path: str, output_path: str) -> str:
    """Negate image (invert colors)."""
    r = _im(input_path, "-negate", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"im_negate failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def im_colorize(input_path: str, output_path: str, color: str, percent: int = 50) -> str:
    """Colorize the image with a fill color. color examples: 'red', '#ff0000', 'rgb(255,0,0)'."""
    r = _im(input_path, "-fill", color, "-colorize", f"{percent}%", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"im_colorize failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def im_lut3d(input_path: str, output_path: str, lut_path: str) -> str:
    """Apply a Hald CLUT (3D color lookup table) for color grading."""
    r = _im(input_path, output_path, "-hald-clut", lut_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"im_lut3d failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


# ─── FILTER ──────────────────────────────────────────────────

@mcp.tool()
def im_blur(input_path: str, output_path: str, radius: float = 0, sigma: float = 3.0) -> str:
    """Gaussian blur. radius=0 auto-computes from sigma."""
    r = _im(input_path, "-blur", f"{radius}x{sigma}", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"im_blur failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def im_sharpen(input_path: str, output_path: str, radius: float = 0, sigma: float = 1.0) -> str:
    """Sharpen the image."""
    r = _im(input_path, "-sharpen", f"{radius}x{sigma}", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"im_sharpen failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def im_unsharp_mask(
    input_path: str, output_path: str,
    radius: float = 0, sigma: float = 1.0, gain: float = 1.0, threshold: float = 0.05,
) -> str:
    """Unsharp mask — professional sharpening. radius, sigma, gain (amount), threshold."""
    r = _im(input_path, "-unsharp", f"{radius}x{sigma}+{gain}+{threshold}", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"im_unsharp_mask failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def im_adaptive_blur(input_path: str, output_path: str, radius: float = 0, sigma: float = 3.0) -> str:
    """Adaptive blur — blurs less near edges (good for skin smoothing on full image)."""
    r = _im(input_path, "-adaptive-blur", f"{radius}x{sigma}", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"im_adaptive_blur failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def im_median(input_path: str, output_path: str, radius: int = 1) -> str:
    """Median filter — good for salt-and-pepper noise removal."""
    r = _im(input_path, "-median", str(radius), output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"im_median failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def im_denoise(input_path: str, output_path: str) -> str:
    """Apply enhance + despeckle to reduce noise."""
    r = _im(input_path, "-enhance", "-despeckle", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"im_denoise failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def im_vignette(
    input_path: str, output_path: str,
    radius: float = 0, sigma: float = 20.0, x: float = 1.0, y: float = 1.0,
) -> str:
    """Soften image edges in vignette style."""
    r = _im(input_path, "-vignette", f"{radius}x{sigma}+{x}+{y}", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"im_vignette failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


# ─── COMPOSITE ───────────────────────────────────────────────

@mcp.tool()
def im_composite(
    input_path: str, overlay_path: str, output_path: str,
    gravity: str = "center", blend_mode: str = "over", opacity: int = 100,
) -> str:
    """Composite an overlay onto an image. gravity: NorthWest, North, Center etc. blend_mode: over, multiply, screen..."""
    dissolve_arg = f"{overlay_path}" if opacity >= 100 else f"{opacity}%x{opacity}%"
    r = _im(input_path, overlay_path, "-gravity", gravity, "-compose", blend_mode,
            "-define", f"compose:args={dissolve_arg}", "-composite", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"im_composite failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def im_watermark(
    input_path: str, watermark_path: str, output_path: str,
    gravity: str = "SouthEast", opacity: int = 30,
) -> str:
    """Apply watermark with specified opacity and position."""
    r = _im(input_path, watermark_path, "-gravity", gravity,
            "-define", f"compose:args={opacity}%", "-compose", "dissolve",
            "-composite", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"im_watermark failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def im_annotate(
    input_path: str, output_path: str, text: str,
    gravity: str = "South", font_size: int = 24,
    color: str = "white", font: str = "",
) -> str:
    """Draw text annotation on image."""
    args = [input_path, "-gravity", gravity, "-pointsize", str(font_size)]
    if font:
        args += ["-font", font]
    args += ["-fill", color, "-annotate", "+0+10", text, output_path]
    r = _im(*args)
    if r["returncode"] != 0:
        raise RuntimeError(f"im_annotate failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def im_append(
    output_path: str, image_paths: str, direction: str = "vertical", gap: int = 0,
) -> str:
    """Append multiple images together. image_paths: JSON array of file paths."""
    paths = json.loads(image_paths)
    op = "-append" if direction == "vertical" else "+append"
    if gap > 0 and direction == "vertical":
        spacer = TEMP.create(suffix=".png", prefix="spacer")
        _im("-size", "1x{}".format(gap), "xc:transparent", str(spacer))
        expanded = []
        for p in paths:
            expanded.append(p)
            expanded.append(str(spacer))
        paths = expanded[:-1]
    r = _im(*(paths + [op, output_path]))
    if r["returncode"] != 0:
        raise RuntimeError(f"im_append failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


# ─── MASKED EDIT ─────────────────────────────────────────────

@mcp.tool()
def im_masked_blur(
    input_path: str, mask_path: str, output_path: str,
    radius: float = 0, sigma: float = 3.0,
) -> str:
    """Blur only the areas specified by a mask (grayscale PNG: 255=blur, 0=skip)."""
    r = _im(input_path, "-mask", mask_path, "-blur", f"{radius}x{sigma}", "+mask", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"im_masked_blur failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def im_masked_sharpen(
    input_path: str, mask_path: str, output_path: str,
    radius: float = 0, sigma: float = 1.0,
) -> str:
    """Sharpen only the areas specified by a mask."""
    r = _im(input_path, "-mask", mask_path, "-sharpen", f"{radius}x{sigma}", "+mask", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"im_masked_sharpen failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def im_masked_modulate(
    input_path: str, mask_path: str, output_path: str,
    brightness: float = 100.0, saturation: float = 100.0, hue: float = 100.0,
) -> str:
    """Modulate brightness/saturation/hue only in the masked region."""
    r = _im(input_path, "-mask", mask_path, "-modulate", f"{brightness},{saturation},{hue}", "+mask", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"im_masked_modulate failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


# ─── FORMAT & META ───────────────────────────────────────────

@mcp.tool()
def im_convert_format(
    input_path: str, output_path: str, output_format: str = "JPEG",
    quality: int = 92, progressive: bool = True,
) -> str:
    """Convert image format. output_format: JPEG, PNG, WebP, AVIF, TIFF, BMP, GIF..."""
    args = [input_path]
    if output_format.upper() in ("JPEG", "JPG"):
        args += ["-quality", str(quality)]
        if progressive:
            args += ["-interlace", "Plane"]
    elif output_format.upper() == "PNG":
        args += ["-quality", "95"]
    elif output_format.upper() in ("WEBP",):
        args += ["-quality", str(quality)]
    elif output_format.upper() == "AVIF":
        args += ["-quality", str(min(quality, 63))]
    args.append(output_path)
    r = _im(*args)
    if r["returncode"] != 0:
        raise RuntimeError(f"im_convert_format failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"], "format": output_format})


@mcp.tool()
def im_identify(input_path: str) -> str:
    """Read all metadata from an image: dimensions, colorspace, bit depth, EXIF, histogram statistics."""
    r = _identify(
        "-verbose",
        "-format",
        "%w|%h|%m|%z|%[colorspace]|%[mean]|%[standard-deviation]|%[kurtosis]|%[skewness]|%r",
        input_path,
        timeout=30,
    )
    if r["returncode"] != 0:
        raise RuntimeError(f"im_identify failed: {r['stderr']}")

    parts = r["stdout"].strip().split("|")
    info = {
        "file": input_path,
        "width": int(parts[0]) if parts[0].isdigit() else parts[0],
        "height": int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else (parts[1] if len(parts) > 1 else ""),
        "format": parts[2] if len(parts) > 2 else "",
        "depth": parts[3] if len(parts) > 3 else "",
        "colorspace": parts[4] if len(parts) > 4 else "",
        "mean": parts[5] if len(parts) > 5 else "",
        "stddev": parts[6] if len(parts) > 6 else "",
        "kurtosis": parts[7] if len(parts) > 7 else "",
        "skewness": parts[8] if len(parts) > 8 else "",
        "resolution": parts[9] if len(parts) > 9 else "",
    }

    # Get file size
    try:
        info["file_size_bytes"] = Path(input_path).stat().st_size
    except OSError:
        info["file_size_bytes"] = 0

    return json.dumps(info, indent=2)


@mcp.tool()
def im_strip(input_path: str, output_path: str) -> str:
    """Strip all profiles, EXIF, comments. Reduces file size, removes privacy-sensitive metadata."""
    r = _im(input_path, "-strip", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"im_strip failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def im_set_icc(input_path: str, icc_profile_path: str, output_path: str) -> str:
    """Apply ICC color profile (e.g., sRGB, AdobeRGB, ProPhoto)."""
    r = _im(input_path, "-profile", icc_profile_path, output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"im_set_icc failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


# ─── BATCH ───────────────────────────────────────────────────

@mcp.tool()
def im_batch(
    input_dir: str, output_dir: str, pattern: str = "*.jpg",
    operation: str = "resize", params: str = "{}",
) -> str:
    """Run the same operation on all matching files in a directory.
    operation: one of resize, crop, sharpen, blur, convert_format, strip, auto_level, grayscale.
    params: JSON string with operation-specific parameters.
    Example: im_batch('/photos', '/output', '*.jpg', 'resize', '{"width": 800}')
    """
    import glob as _glob
    p = json.loads(params)
    pattern_full = str(Path(input_dir) / pattern)
    files = sorted(_glob.glob(pattern_full))
    if not files:
        return json.dumps({"error": f"No files matching {pattern_full}"})

    ensure_dir(Path(output_dir))
    results = []
    errors = []
    for i, f in enumerate(files):
        base = Path(f).stem
        ext = p.get("output_format", Path(f).suffix.lower().lstrip("."))
        out = str(Path(output_dir) / f"{base}.{ext}")

        try:
            if operation == "resize":
                im_resize(f, out, p.get("width", 800), p.get("height", 0),
                          p.get("keep_aspect", True), p.get("filter_type", "Lanczos"))
            elif operation == "crop":
                im_crop(f, out, p.get("width", 100), p.get("height", 100),
                        p.get("x", 0), p.get("y", 0))
            elif operation == "sharpen":
                im_sharpen(f, out, p.get("radius", 0), p.get("sigma", 1.0))
            elif operation == "blur":
                im_blur(f, out, p.get("radius", 0), p.get("sigma", 3.0))
            elif operation == "convert_format":
                im_convert_format(f, out, p.get("output_format", "JPEG"),
                                  p.get("quality", 92), p.get("progressive", True))
            elif operation == "strip":
                im_strip(f, out)
            elif operation == "auto_level":
                im_auto_level(f, out)
            elif operation == "grayscale":
                im_grayscale(f, out, p.get("method", "Rec709Luma"))
            else:
                errors.append({"file": f, "error": f"Unknown operation: {operation}"})
                continue
            results.append({"file": f, "output": out, "status": "ok"})
        except RuntimeError as e:
            errors.append({"file": f, "error": str(e)})

    return json.dumps({
        "total": len(files),
        "success": len(results),
        "errors": len(errors),
        "error_details": errors[:20],
        "first_outputs": [r["output"] for r in results[:5]],
    }, indent=2)


if __name__ == "__main__":
    log.info("Starting ImageMagick MCP server (binary: %s)", IMG)
    if not check_binary(IMG):
        log.error("ImageMagick convert not found. Install: sudo apt install imagemagick")
        exit(1)
    mcp.run()
