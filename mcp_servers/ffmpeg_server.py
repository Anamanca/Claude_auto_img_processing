#!/usr/bin/env python3
"""FFmpeg MCP Server — 28 tools for video processing with NVENC support.

Wraps FFmpeg CLI as MCP tools for: transform, speed, color, filter,
composite, codec, metadata extraction.
"""

from mcp.server.fastmcp import FastMCP
from pathlib import Path
import json
from typing import Optional

from common import (
    load_config, resolve_path, run_cmd, setup_logging,
    TempManager, check_binary, is_video_ext,
)

cfg = load_config()
log = setup_logging(cfg, "ffmpeg_server")
TEMP = TempManager(cfg)

FFMPEG = cfg["ffmpeg"]["binary"]
FFPROBE = cfg["ffmpeg"]["ffprobe_binary"]

mcp = FastMCP(
    "ffmpeg-mcp",
    instructions="FFmpeg MCP server — video processing, transcoding (NVENC), filters, metadata. All operations use file paths.",
)


def _ffmpeg(*args, timeout: int = 3600) -> dict:
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error"] + list(args)
    return run_cmd(cmd, timeout=timeout, log=log)


def _ffmpeg_stats(*args, timeout: int = 3600) -> dict:
    """Run FFmpeg with progress pipe for real-time stats."""
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
           "-progress", "pipe:1", "-stats_period", "5"] + list(args)
    return run_cmd(cmd, timeout=timeout, log=log)


def _has_nvenc() -> bool:
    r = run_cmd([FFMPEG, "-hide_banner", "-encoders"], timeout=10)
    return "h264_nvenc" in r.get("stdout", "")


def _pick_vcodec(want_nvenc: bool = True) -> str:
    if want_nvenc and cfg["ffmpeg"].get("nvenc_enabled", True) and _has_nvenc():
        return "h264_nvenc"
    return "libx264"


# ─── TRANSFORM ───────────────────────────────────────────────

@mcp.tool()
def ffmpeg_scale(
    input_path: str, output_path: str,
    width: int, height: int = -1,
    algo: str = "lanczos",
) -> str:
    """Scale video. If height=-1, auto from aspect ratio. algo: lanczos, bilinear, bicubic, neighbor."""
    vf = f"scale={width}:{height}" if height > 0 else f"scale={width}:-1"
    vf += f":flags={algo}"
    r = _ffmpeg("-i", input_path, "-vf", vf, "-c:a", "copy", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"ffmpeg_scale failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def ffmpeg_crop(input_path: str, output_path: str, w: int, h: int, x: int = 0, y: int = 0) -> str:
    """Crop video region (W:H:X:Y)."""
    r = _ffmpeg("-i", input_path, "-vf", f"crop={w}:{h}:{x}:{y}", "-c:a", "copy", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"ffmpeg_crop failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def ffmpeg_rotate(input_path: str, output_path: str, angle: str = "90") -> str:
    """Rotate video. angle: '90' (clockwise), '270' (counter-clockwise), '180', or transpose_dir: 'clock', 'cclock', 'clock_flip'."""
    transpose_map = {"90": "1", "270": "2", "180": "2,transpose=2"}
    if angle in transpose_map:
        vf_expr = f"transpose={transpose_map[angle]}"
    else:
        vf_expr = f"rotate={angle}*PI/180"
    r = _ffmpeg("-i", input_path, "-vf", vf_expr, "-c:a", "copy", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"ffmpeg_rotate failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


HAS_NO_FILTER = object()

@mcp.tool()
def ffmpeg_trim(input_path: str, output_path: str, start: str = "0", end: str = "",
                duration: str = "", re_encode: bool = False) -> str:
    """Trim video segment. start/end in seconds or HH:MM:SS. If re_encode=False, uses stream copy (fast)."""
    args = ["-ss", start]
    if end:
        args += ["-to", end]
    if duration:
        args += ["-t", duration]
    args += ["-i", input_path]
    if re_encode:
        args += ["-c:v", _pick_vcodec(True), "-preset", cfg["ffmpeg"]["default_preset"]]
    else:
        args += ["-c", "copy"]
    r = _ffmpeg(*args, output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"ffmpeg_trim failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def ffmpeg_pad(input_path: str, output_path: str, width: int, height: int,
               x: int = -1, y: int = -1, color: str = "black") -> str:
    """Pad video to dimensions. x/y = -1 centers the video."""
    x_expr = f"(ow-iw)/2" if x < 0 else str(x)
    y_expr = f"(oh-ih)/2" if y < 0 else str(y)
    r = _ffmpeg("-i", input_path, "-vf", f"pad={width}:{height}:{x_expr}:{y_expr}:{color}",
                "-c:a", "copy", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"ffmpeg_pad failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def ffmpeg_deshake(input_path: str, output_path: str) -> str:
    """Basic video stabilization using deshake filter."""
    r = _ffmpeg("-i", input_path, "-vf", "deshake", "-c:a", "copy", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"ffmpeg_deshake failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def ffmpeg_vstab(input_path: str, output_path: str, shakiness: int = 5, smoothing: int = 15) -> str:
    """Professional 2-pass video stabilization (vidstabdetect + vidstabtransform)."""
    transforms_file = str(TEMP.create(suffix=".trf", prefix="vstab"))
    r1 = _ffmpeg("-i", input_path, "-vf", f"vidstabdetect=shakiness={shakiness}:accuracy=15:result={transforms_file}",
                 "-f", "null", "-")
    if r1["returncode"] != 0:
        raise RuntimeError(f"ffmpeg_vstab pass1 failed: {r1['stderr']}")
    r2 = _ffmpeg("-i", input_path, "-vf", f"vidstabtransform=smoothing={smoothing}:input={transforms_file}",
                 "-c:a", "copy", output_path)
    if r2["returncode"] != 0:
        raise RuntimeError(f"ffmpeg_vstab pass2 failed: {r2['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r1["elapsed"] + r2["elapsed"]})


# ─── SPEED & TIME ────────────────────────────────────────────

@mcp.tool()
def ffmpeg_speed(input_path: str, output_path: str, factor: float = 2.0) -> str:
    """Change video playback speed. factor: 2.0 = 2x faster, 0.5 = half speed."""
    if factor == 1.0:
        return ffmpeg_trim(input_path, output_path, start="0")
    p = 1.0 / factor if factor > 0 else 1.0
    r = _ffmpeg("-i", input_path,
                "-filter_complex", f"[0:v]setpts={p}*PTS[v];[0:a]atempo={factor}[a]",
                "-map", "[v]", "-map", "[a]", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"ffmpeg_speed failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def ffmpeg_slowmo(input_path: str, output_path: str, target_fps: int = 60) -> str:
    """Create smooth slow-motion using motion interpolation (minterpolate)."""
    r = _ffmpeg("-i", input_path, "-vf", f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc",
                "-c:a", "copy", output_path, timeout=7200)
    if r["returncode"] != 0:
        raise RuntimeError(f"ffmpeg_slowmo failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def ffmpeg_reverse(input_path: str, output_path: str) -> str:
    """Reverse video (audio included)."""
    r = _ffmpeg("-i", input_path,
                "-filter_complex", "[0:v]reverse[v];[0:a]areverse[a]",
                "-map", "[v]", "-map", "[a]", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"ffmpeg_reverse failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def ffmpeg_fade(input_path: str, output_path: str, fade_type: str = "in",
                duration: float = 1.0, start_frame: int = 0) -> str:
    """Fade video in/out. fade_type: 'in' or 'out'."""
    r = _ffmpeg("-i", input_path, "-vf", f"fade={fade_type}:st={start_frame}:d={duration}",
                "-c:a", "copy", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"ffmpeg_fade failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


# ─── COLOR ───────────────────────────────────────────────────

@mcp.tool()
def ffmpeg_colorbalance(
    input_path: str, output_path: str,
    rs: float = 0, gs: float = 0, bs: float = 0,
    rm: float = 0, gm: float = 0, bm: float = 0,
    rh: float = 0, gh: float = 0, bh: float = 0,
) -> str:
    """Adjust color balance (shadows, midtones, highlights) for RGB channels."""
    vf = (f"colorbalance=rs={rs}:gs={gs}:bs={bs}"
          f":rm={rm}:gm={gm}:bm={bm}"
          f":rh={rh}:gh={gh}:bh={bh}")
    r = _ffmpeg("-i", input_path, "-vf", vf, "-c:a", "copy", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"ffmpeg_colorbalance failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def ffmpeg_eq(
    input_path: str, output_path: str,
    brightness: float = 0, contrast: float = 1.0,
    gamma: float = 1.0, saturation: float = 1.0,
) -> str:
    """Adjust brightness (-1..1), contrast (-1000..1000), gamma (0.1..10), saturation (0..3)."""
    vf = f"eq=brightness={brightness}:contrast={contrast}:gamma={gamma}:saturation={saturation}"
    r = _ffmpeg("-i", input_path, "-vf", vf, "-c:a", "copy", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"ffmpeg_eq failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def ffmpeg_curves(input_path: str, output_path: str, preset: str = "none",
                  master: str = "0/0 1/1", red: str = "0/0 1/1",
                  green: str = "0/0 1/1", blue: str = "0/0 1/1") -> str:
    """Adjust color curves. preset: none, color_negative, cross_process, darker, lighter, increase_contrast, linear_contrast, medium_contrast, strong_contrast, negative, vintage."""
    if preset != "none":
        vf = f"curves=preset={preset}"
    else:
        vf = f"curves=master='{master}':red='{red}':green='{green}':blue='{blue}'"
    r = _ffmpeg("-i", input_path, "-vf", vf, "-c:a", "copy", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"ffmpeg_curves failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def ffmpeg_lut3d(input_path: str, output_path: str, lut_path: str) -> str:
    """Apply 3D LUT (.cube format) for color grading."""
    r = _ffmpeg("-i", input_path, "-vf", f"lut3d={lut_path}", "-c:a", "copy", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"ffmpeg_lut3d failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def ffmpeg_normalize(input_path: str, output_path: str) -> str:
    """Normalize video (histogram stretching)."""
    r = _ffmpeg("-i", input_path, "-vf", "normalize", "-c:a", "copy", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"ffmpeg_normalize failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def ffmpeg_hue(input_path: str, output_path: str, h_degrees: float = 0, s_mult: float = 1.0) -> str:
    """Adjust hue (degrees) and saturation (multiplier)."""
    r = _ffmpeg("-i", input_path, "-vf", f"hue=h={h_degrees}:s={s_mult}", "-c:a", "copy", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"ffmpeg_hue failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


# ─── FILTER ──────────────────────────────────────────────────

@mcp.tool()
def ffmpeg_blur(input_path: str, output_path: str, sigma: float = 3.0, filter_type: str = "gblur") -> str:
    """Blur video. filter_type: gblur (Gaussian) or boxblur."""
    if filter_type == "boxblur":
        vf = f"boxblur={sigma}"
    else:
        vf = f"gblur=sigma={sigma}"
    r = _ffmpeg("-i", input_path, "-vf", vf, "-c:a", "copy", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"ffmpeg_blur failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def ffmpeg_sharpen(input_path: str, output_path: str,
                   luma_msize_x: int = 5, luma_msize_y: int = 5,
                   luma_amount: float = 1.0, filter_type: str = "unsharp") -> str:
    """Sharpen video. filter_type: unsharp (standard) or cas (Contrast Adaptive Sharpen)."""
    if filter_type == "cas":
        vf = f"cas=strength={luma_amount}"
    else:
        vf = f"unsharp={luma_msize_x}:{luma_msize_y}:{luma_amount}"
    r = _ffmpeg("-i", input_path, "-vf", vf, "-c:a", "copy", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"ffmpeg_sharpen failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def ffmpeg_denoise(input_path: str, output_path: str, strength: float = 1.0,
                   filter_type: str = "nlmeans") -> str:
    """Denoise video. filter_type: nlmeans (high quality) or hqdn3d (fast)."""
    if filter_type == "hqdn3d":
        vf = f"hqdn3d={strength}"
    else:
        vf = f"nlmeans={strength}"
    r = _ffmpeg("-i", input_path, "-vf", vf, "-c:a", "copy", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"ffmpeg_denoise failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def ffmpeg_deband(input_path: str, output_path: str, threshold: int = 16,
                  direction: float = 2, blur: bool = True) -> str:
    """Deband video (remove color banding)."""
    vf = f"deband=1thr={threshold}:2thr={threshold*2}:3thr={threshold*4}:4thr={threshold*8}:dir={direction}:blur={1 if blur else 0}"
    r = _ffmpeg("-i", input_path, "-vf", vf, "-c:a", "copy", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"ffmpeg_deband failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


# ─── OVERLAY & COMPOSITE ─────────────────────────────────────

@mcp.tool()
def ffmpeg_overlay(input_path: str, overlay_path: str, output_path: str,
                   x: str = "0", y: str = "0", alpha: float = 1.0) -> str:
    """Overlay a video/image on top of another. x/y can be expressions like 'W-w-10'."""
    if alpha < 1.0:
        vf = f"[1:v]format=rgba,colorchannelmixer=aa={alpha}[ov];[0:v][ov]overlay={x}:{y}"
        r = _ffmpeg("-i", input_path, "-i", overlay_path,
                    "-filter_complex", vf, "-c:a", "copy", output_path)
    else:
        r = _ffmpeg("-i", input_path, "-i", overlay_path,
                    "-filter_complex", f"overlay={x}:{y}", "-c:a", "copy", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"ffmpeg_overlay failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def ffmpeg_drawtext(input_path: str, output_path: str, text: str,
                    fontfile: str = "", fontsize: int = 24, fontcolor: str = "white",
                    x: str = "0", y: str = "0", shadow: bool = False) -> str:
    """Draw text on video frames."""
    vf = f"drawtext=text='{text}':fontsize={fontsize}:fontcolor={fontcolor}:x={x}:y={y}"
    if fontfile:
        vf += f":fontfile={fontfile}"
    if shadow:
        vf += ":shadowx=2:shadowy=2"
    r = _ffmpeg("-i", input_path, "-vf", vf, "-c:a", "copy", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"ffmpeg_drawtext failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def ffmpeg_xfade(input_path_a: str, input_path_b: str, output_path: str,
                 transition: str = "fade", duration: float = 1.0, offset: float = 0) -> str:
    """Cross-fade between two videos. transition: fade, wipeleft, wiperight, wipeup, wipedown, slideleft, slideright, slideup, slidedown, diagtl, diagtr..."""
    if offset <= 0:
        # Compute offset from first video duration
        probe_a = ffmpeg_probe(input_path_a)
        dur_a = float(json.loads(probe_a).get("duration", 10))
        offset = max(0, dur_a - duration)
    r = _ffmpeg("-i", input_path_a, "-i", input_path_b,
                "-filter_complex", f"xfade=transition={transition}:duration={duration}:offset={offset}",
                "-c:a", "copy", output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"ffmpeg_xfade failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def ffmpeg_concat(file_list_json: str, output_path: str, re_encode: bool = False) -> str:
    """Concatenate multiple video files. file_list_json: JSON array of file paths.
    If all same codec and re_encode=False, uses demuxer (fast stream copy).
    """
    files = json.loads(file_list_json)
    concat_file = TEMP.create(suffix=".txt", prefix="concat")
    with open(concat_file, "w") as f:
        for fp in files:
            f.write(f"file '{fp}'\n")
    args = ["-f", "concat", "-safe", "0", "-i", str(concat_file)]
    if re_encode:
        args += ["-c:v", _pick_vcodec(True), "-preset", cfg["ffmpeg"]["default_preset"]]
    else:
        args += ["-c", "copy"]
    r = _ffmpeg(*args, output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"ffmpeg_concat failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


# ─── CODEC & FORMAT ──────────────────────────────────────────

@mcp.tool()
def ffmpeg_transcode(
    input_path: str, output_path: str,
    video_codec: str = "h264", crf: int = 23, preset: str = "medium",
    pixel_format: str = "yuv420p",
    audio_codec: str = "aac", audio_bitrate: str = "192k",
) -> str:
    """Full video transcode. video_codec: h264, h265, av1, prores, vp9. crf: lower=better quality (18-28 sane range)."""
    args = ["-i", input_path]
    if video_codec == "h264":
        args += ["-c:v", "libx264", "-crf", str(crf), "-preset", preset]
    elif video_codec == "h265":
        args += ["-c:v", "libx265", "-crf", str(crf), "-preset", preset]
    elif video_codec == "av1":
        args += ["-c:v", "libaom-av1", "-crf", str(crf), "-cpu-used", "4"]
    elif video_codec == "prores":
        args += ["-c:v", "prores_ks", "-profile:v", "2"]
    elif video_codec == "vp9":
        args += ["-c:v", "libvpx-vp9", "-crf", str(crf), "-b:v", "0"]
    args += ["-pix_fmt", pixel_format, "-c:a", audio_codec, "-b:a", audio_bitrate]
    r = _ffmpeg(*args + [output_path])
    if r["returncode"] != 0:
        raise RuntimeError(f"ffmpeg_transcode failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def ffmpeg_nvenc(
    input_path: str, output_path: str,
    codec: str = "hevc", cq: int = 23, preset: str = "p4",
    pixel_format: str = "yuv420p", audio_codec: str = "aac",
    audio_bitrate: str = "192k",
) -> str:
    """Hardware-accelerated NVENC transcode. codec: h264 or hevc. cq: 1-51 (lower=better). preset: p1(fast)-p7(slow)."""
    if codec == "h264":
        vcodec = "h264_nvenc"
    else:
        vcodec = "hevc_nvenc"
    r = _ffmpeg("-i", input_path,
                "-c:v", vcodec, "-cq", str(cq), "-preset", preset,
                "-pix_fmt", pixel_format,
                "-c:a", audio_codec, "-b:a", audio_bitrate,
                output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"ffmpeg_nvenc failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def ffmpeg_extract_audio(input_path: str, output_path: str,
                         audio_codec: str = "mp3", bitrate: str = "192k") -> str:
    """Extract audio track from video."""
    r = _ffmpeg("-i", input_path, "-vn", "-c:a", audio_codec, "-b:a", bitrate, output_path)
    if r["returncode"] != 0:
        raise RuntimeError(f"ffmpeg_extract_audio failed: {r['stderr']}")
    return json.dumps({"output": output_path, "elapsed": r["elapsed"]})


@mcp.tool()
def ffmpeg_extract_frames(
    input_path: str, output_dir: str,
    fps: float = 1, start: str = "0", end: str = "",
    image_format: str = "jpg",
) -> str:
    """Extract frames as images. fps: frames per second to extract (1 = one per second)."""
    from common import ensure_dir
    ensure_dir(Path(output_dir))
    args = ["-ss", start]
    if end:
        args += ["-to", end]
    args += ["-i", input_path, "-vf", f"fps={fps}"]
    if image_format == "jpg":
        args += ["-q:v", "2"]
    out_pattern = str(Path(output_dir) / f"frame_%05d.{image_format}")
    r = _ffmpeg(*args, out_pattern)
    if r["returncode"] != 0:
        raise RuntimeError(f"ffmpeg_extract_frames failed: {r['stderr']}")
    import glob as _glob
    extracted = sorted(_glob.glob(str(Path(output_dir) / f"frame_*.{image_format}")))
    return json.dumps({
        "output_dir": output_dir,
        "frames_extracted": len(extracted),
        "elapsed": r["elapsed"],
    })


@mcp.tool()
def ffmpeg_probe(input_path: str) -> str:
    """Extract comprehensive metadata from a video file using ffprobe."""
    r = run_cmd([
        FFPROBE, "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", input_path,
    ], timeout=30)
    if r["returncode"] != 0:
        raise RuntimeError(f"ffmpeg_probe failed: {r['stderr']}")

    data = json.loads(r["stdout"])
    fmt = data.get("format", {})
    streams = data.get("streams", [])

    result = {
        "file": input_path,
        "duration": float(fmt.get("duration", 0)),
        "size_bytes": int(fmt.get("size", 0)),
        "bit_rate": int(fmt.get("bit_rate", 0)),
        "format_name": fmt.get("format_name", ""),
        "format_long_name": fmt.get("format_long_name", ""),
        "streams": [],
    }

    for s in streams:
        si = {
            "index": s.get("index"),
            "codec_type": s.get("codec_type"),
            "codec_name": s.get("codec_name"),
            "codec_long_name": s.get("codec_long_name", ""),
        }
        if s["codec_type"] == "video":
            si.update({
                "width": s.get("width"),
                "height": s.get("height"),
                "pix_fmt": s.get("pix_fmt"),
                "r_frame_rate": s.get("r_frame_rate"),
                "duration": float(s.get("duration", 0)),
                "bit_rate": int(s.get("bit_rate", 0)),
                "color_space": s.get("color_space", ""),
                "color_transfer": s.get("color_transfer", ""),
                "color_primaries": s.get("color_primaries", ""),
            })
        elif s["codec_type"] == "audio":
            si.update({
                "sample_rate": s.get("sample_rate"),
                "channels": s.get("channels"),
                "channel_layout": s.get("channel_layout", ""),
                "duration": float(s.get("duration", 0)),
                "bit_rate": int(s.get("bit_rate", 0)),
            })
        result["streams"].append(si)

    return json.dumps(result, indent=2)


# ─── BATCH ───────────────────────────────────────────────────

@mcp.tool()
def ffmpeg_batch_transcode(
    input_dir: str, output_dir: str, pattern: str = "*.mp4",
    codec: str = "hevc", cq: int = 23, preset: str = "p4",
) -> str:
    """Batch transcode all matching videos in a directory using NVENC."""
    import glob as _glob
    from common import ensure_dir

    pattern_full = str(Path(input_dir) / pattern)
    files = sorted(_glob.glob(pattern_full))
    if not files:
        return json.dumps({"error": f"No files matching {pattern_full}"})

    ensure_dir(Path(output_dir))
    results = []
    for f in files:
        out = str(Path(output_dir) / f"{Path(f).stem}.mp4")
        try:
            info = ffmpeg_nvenc(f, out, codec=codec, cq=cq, preset=preset)
            results.append({"file": f, "output": out, "status": "ok", "info": info})
        except RuntimeError as e:
            results.append({"file": f, "error": str(e)})

    ok = sum(1 for r in results if r.get("status") == "ok")
    return json.dumps({"total": len(files), "success": ok, "errors": len(results) - ok})


if __name__ == "__main__":
    log.info("Starting FFmpeg MCP server (binary: %s)", FFMPEG)
    if not check_binary(FFMPEG):
        log.error("FFmpeg not found. Install: sudo apt install ffmpeg")
        exit(1)
    log.info("NVENC available: %s", _has_nvenc())
    mcp.run()
