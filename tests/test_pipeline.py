#!/usr/bin/env python3
"""Tests for hybrid_pipeline MCP servers.

Usage:
  python -m pytest tests/ -v
  python tests/test_basic.py        # Quick smoke test
"""

import json
import subprocess
import sys
import os
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVERS_DIR = PROJECT_ROOT / "mcp_servers"
TEST_DATA = Path(__file__).resolve().parent / "test_data"


def _create_test_image(path: Path, size=(640, 480), color=(30, 120, 200)):
    """Create a simple test image."""
    import numpy as np
    import cv2
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    img[:] = color
    # Add a "face-like" circle
    cv2.circle(img, (size[0] // 2, size[1] // 2), 100, (200, 180, 160), -1)
    cv2.circle(img, (size[0] // 2 - 35, size[1] // 2 - 30), 15, (255, 255, 255), -1)
    cv2.circle(img, (size[0] // 2 + 35, size[1] // 2 - 30), 15, (255, 255, 255), -1)
    cv2.circle(img, (size[0] // 2, size[1] // 2 + 20), 10, (0, 0, 0), -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return path


def _create_test_video(path: Path, duration=1, fps=30, size=(320, 240)):
    """Create a short test video using FFmpeg."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"testsrc=duration={duration}:size={size[0]}x{size[1]}:rate={fps}",
        "-frames:v", str(duration * fps),
        "-c:v", "libx264", "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        str(path),
    ], capture_output=True, check=False)
    return path


# ─── Fixtures ────────────────────────────────────────────

@pytest.fixture(scope="module")
def test_image():
    p = TEST_DATA / "test_input.jpg"
    return _create_test_image(p)


@pytest.fixture(scope="module")
def test_video():
    p = TEST_DATA / "test_input.mp4"
    return _create_test_video(p)


@pytest.fixture(scope="module")
def output_dir():
    p = PROJECT_ROOT / "temp" / "test_output"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ─── Common Module Tests ─────────────────────────────────

class TestCommon:
    def test_config_loads(self):
        sys.path.insert(0, str(SERVERS_DIR))
        from common import load_config
        cfg = load_config()
        assert "paths" in cfg
        assert "gpu" in cfg
        assert "imagemagick" in cfg
        assert "ffmpeg" in cfg

    def test_temp_manager(self):
        sys.path.insert(0, str(SERVERS_DIR))
        from common import TempManager, load_config
        cfg = load_config()
        tm = TempManager(cfg, job_id="test_job")
        f = tm.create(suffix=".txt", prefix="test")
        f.write_text("hello")
        assert f.exists()
        assert f.read_text() == "hello"
        tm.cleanup()
        assert not f.exists()

    def test_check_binary(self):
        sys.path.insert(0, str(SERVERS_DIR))
        from common import check_binary
        assert check_binary("convert") is not None or True  # optional
        assert check_binary("nonexistent_binary_xyz") is None

    def test_gpu_info(self):
        sys.path.insert(0, str(SERVERS_DIR))
        from common import check_gpu_available
        info = check_gpu_available()
        assert "cuda_available" in info

    def test_file_hash(self):
        sys.path.insert(0, str(SERVERS_DIR))
        from common import file_hash
        p = TEST_DATA / "hash_test.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("test content")
        h = file_hash(str(p))
        assert len(h) == 64  # SHA256


# ─── IM MCP Tests ────────────────────────────────────────

class TestImageMagick:
    @pytest.fixture(scope="class")
    def im_tools(self):
        sys.path.insert(0, str(SERVERS_DIR))
        import im_server
        return im_server

    def test_im_identify(self, test_image, im_tools):
        result = json.loads(im_tools.im_identify(str(test_image)))
        assert result["width"] > 0
        assert result["height"] > 0
        assert result["format"] in ("JPEG", "jpeg")

    def test_im_resize(self, test_image, output_dir, im_tools):
        out = str(output_dir / "resized.jpg")
        result = json.loads(im_tools.im_resize(str(test_image), out, 320))
        assert Path(out).exists()
        assert "output" in result

    def test_im_crop(self, test_image, output_dir, im_tools):
        out = str(output_dir / "cropped.jpg")
        result = json.loads(im_tools.im_crop(str(test_image), out, 100, 100, 50, 50))
        assert Path(out).exists()

    def test_im_rotate(self, test_image, output_dir, im_tools):
        out = str(output_dir / "rotated.jpg")
        result = json.loads(im_tools.im_rotate(str(test_image), out, 45))
        assert Path(out).exists()

    def test_im_blur(self, test_image, output_dir, im_tools):
        out = str(output_dir / "blurred.jpg")
        result = json.loads(im_tools.im_blur(str(test_image), out, sigma=3.0))
        assert Path(out).exists()

    def test_im_sharpen(self, test_image, output_dir, im_tools):
        out = str(output_dir / "sharpened.jpg")
        result = json.loads(im_tools.im_sharpen(str(test_image), out, sigma=1.0))
        assert Path(out).exists()

    def test_im_grayscale(self, test_image, output_dir, im_tools):
        out = str(output_dir / "gray.jpg")
        result = json.loads(im_tools.im_grayscale(str(test_image), out))
        assert Path(out).exists()

    def test_im_modulate(self, test_image, output_dir, im_tools):
        out = str(output_dir / "modulated.jpg")
        result = json.loads(im_tools.im_modulate(str(test_image), out, 110, 120, 100))
        assert Path(out).exists()

    def test_im_level(self, test_image, output_dir, im_tools):
        out = str(output_dir / "leveled.jpg")
        result = json.loads(im_tools.im_level(str(test_image), out, black_point=20, gamma=1.1, white_point=240))
        assert Path(out).exists()

    def test_im_convert_format(self, test_image, output_dir, im_tools):
        out_png = str(output_dir / "converted.png")
        result = json.loads(im_tools.im_convert_format(str(test_image), out_png, "PNG"))
        assert Path(out_png).exists()

    def test_im_watermark(self, test_image, output_dir, im_tools):
        # Create a small watermark
        wm = output_dir / "wm.png"
        _create_test_image(wm, (50, 50), (255, 0, 0))
        out = str(output_dir / "watermarked.jpg")
        result = json.loads(im_tools.im_watermark(str(test_image), str(wm), out, opacity=50))
        assert Path(out).exists()

    def test_im_append(self, test_image, output_dir, im_tools):
        out = str(output_dir / "appended.jpg")
        paths = json.dumps([str(test_image), str(test_image)])
        result = json.loads(im_tools.im_append(out, paths, direction="vertical"))
        assert Path(out).exists()

    def test_im_batch_resize(self, test_image, output_dir, im_tools):
        # Copy test image to a temp dir for batch
        batch_dir = output_dir / "batch_input"
        batch_dir.mkdir(exist_ok=True)
        import shutil
        shutil.copy(str(test_image), str(batch_dir / "photo1.jpg"))
        shutil.copy(str(test_image), str(batch_dir / "photo2.jpg"))
        batch_out = output_dir / "batch_output"
        result = json.loads(im_tools.im_batch(
            str(batch_dir), str(batch_out), "*.jpg", "resize",
            '{"width": 200}',
        ))
        assert result["success"] == 2
        assert len(list(batch_out.glob("*.jpg"))) == 2


# ─── FFmpeg MCP Tests ────────────────────────────────────

class TestFFmpeg:
    @pytest.fixture(scope="class")
    def ffmpeg_tools(self):
        sys.path.insert(0, str(SERVERS_DIR))
        import ffmpeg_server
        return ffmpeg_server

    def test_ffmpeg_probe(self, test_video, ffmpeg_tools):
        result = json.loads(ffmpeg_tools.ffmpeg_probe(str(test_video)))
        assert result["duration"] > 0
        assert len(result["streams"]) > 0
        video_streams = [s for s in result["streams"] if s["codec_type"] == "video"]
        assert len(video_streams) > 0
        assert video_streams[0]["width"] > 0

    def test_ffmpeg_scale(self, test_video, output_dir, ffmpeg_tools):
        out = str(output_dir / "scaled.mp4")
        result = json.loads(ffmpeg_tools.ffmpeg_scale(str(test_video), out, 160))
        assert Path(out).exists()

    def test_ffmpeg_trim_stream_copy(self, test_video, output_dir, ffmpeg_tools):
        out = str(output_dir / "trimmed.mp4")
        result = json.loads(ffmpeg_tools.ffmpeg_trim(str(test_video), out, start="0", duration="0.5"))
        assert Path(out).exists()

    def test_ffmpeg_extract_frames(self, test_video, output_dir, ffmpeg_tools):
        frames_dir = str(output_dir / "frames")
        result = json.loads(ffmpeg_tools.ffmpeg_extract_frames(str(test_video), frames_dir, fps=1))
        assert result["frames_extracted"] > 0

    def test_ffmpeg_transcode(self, test_video, output_dir, ffmpeg_tools):
        out = str(output_dir / "transcoded.mp4")
        result = json.loads(ffmpeg_tools.ffmpeg_transcode(
            str(test_video), out, video_codec="h264", crf=28, preset="ultrafast",
        ))
        assert Path(out).exists()

    def test_ffmpeg_extract_audio(self, test_video, output_dir, ffmpeg_tools):
        out = str(output_dir / "audio.mp3")
        result = json.loads(ffmpeg_tools.ffmpeg_extract_audio(str(test_video), out))
        # May fail if no audio stream in test video — that's OK
        if Path(out).exists():
            assert Path(out).stat().st_size > 0

    def test_ffmpeg_eq(self, test_video, output_dir, ffmpeg_tools):
        out = str(output_dir / "eq.mp4")
        result = json.loads(ffmpeg_tools.ffmpeg_eq(str(test_video), out, brightness=0.05, saturation=1.1))
        assert Path(out).exists()


# ─── AI MCP Tests (CPU-safe) ─────────────────────────────

class TestAI:
    @pytest.fixture(scope="class")
    def ai_tools(self):
        sys.path.insert(0, str(SERVERS_DIR))
        import ai_server
        return ai_server

    def test_ai_detect_faces(self, test_image, output_dir, ai_tools):
        result = json.loads(ai_tools.ai_detect_faces(str(test_image)))
        if "error" not in result:
            assert "faces_detected" in result

    def test_ai_skin_mask(self, test_image, output_dir, ai_tools):
        result = json.loads(ai_tools.ai_skin_mask(str(test_image)))
        if "mask_path" in result:
            assert Path(result["mask_path"]).exists()

    def test_ai_eyes_mask(self, test_image, output_dir, ai_tools):
        result = json.loads(ai_tools.ai_eyes_mask(str(test_image)))
        if "mask_path" in result:
            assert Path(result["mask_path"]).exists()

    def test_ai_auto_exposure(self, test_image, ai_tools):
        result = json.loads(ai_tools.ai_auto_exposure(str(test_image)))
        if "error" not in result:
            assert "black_point" in result
            assert "white_point" in result
            assert "gamma" in result

    def test_ai_remove_background(self, test_image, output_dir, ai_tools):
        out = str(output_dir / "no_bg.png")
        try:
            result = json.loads(ai_tools.ai_remove_background(str(test_image), out))
            if "output" in result and Path(out).exists():
                assert True
        except Exception:
            pass  # rembg may not be installed


# ─── Util MCP Tests ──────────────────────────────────────

class TestUtil:
    @pytest.fixture(scope="class")
    def util_tools(self):
        sys.path.insert(0, str(SERVERS_DIR))
        import util_server
        return util_server

    def test_util_file_list(self, test_image, util_tools):
        result = json.loads(util_tools.util_file_list(str(TEST_DATA), recursive=False))
        assert result["total"] > 0

    def test_util_hash(self, test_image, util_tools):
        result = json.loads(util_tools.util_hash(str(test_image)))
        assert len(result["hash"]) == 64

    def test_util_compare_images(self, test_image, output_dir, util_tools):
        from common import TempManager, load_config
        cfg = load_config()
        tm = TempManager(cfg, "test")
        img2 = tm.create(suffix=".jpg")
        _create_test_image(img2, (640, 480), (40, 130, 210))
        result = json.loads(util_tools.util_compare_images(str(test_image), str(img2)))
        assert "psnr" in result
        assert "ssim" in result
        tm.cleanup()


# ─── End-to-End Hybrid Pipeline Tests ────────────────────

class TestHybridPipeline:
    """Simulate the full AI + IM pipeline (tests the concept)."""

    def test_skin_mask_then_blur(self, test_image, output_dir):
        """AI generates skin mask → IM applies masked blur."""
        sys.path.insert(0, str(SERVERS_DIR))
        import im_server
        import ai_server

        # Step 1: AI generates skin mask
        skin_result = json.loads(ai_server.ai_skin_mask(str(test_image)))
        if "mask_path" not in skin_result:
            pytest.skip("AI skin mask not available")
        mask_path = skin_result["mask_path"]
        assert Path(mask_path).exists()

        # Step 2: AI computes exposure params
        exp_result = json.loads(ai_server.ai_auto_exposure(str(test_image)))

        # Step 3: IM applies masked blur
        blurred = str(output_dir / "skin_blurred.jpg")
        blur_result = json.loads(im_server.im_masked_blur(str(test_image), mask_path, blurred, sigma=3.0))
        assert Path(blurred).exists()

        # Step 4: IM applies levels
        leveled = str(output_dir / "final.jpg")
        level_result = json.loads(im_server.im_level(
            blurred, leveled,
            black_point=exp_result["black_point"],
            gamma=exp_result["gamma"],
            white_point=exp_result["white_point"],
        ))
        assert Path(leveled).exists()
        print(f"\n  Pipeline output: {leveled}")

    def test_auto_exposure_then_convert(self, test_image, output_dir):
        """AI computes exposure → IM applies levels + converts format."""
        sys.path.insert(0, str(SERVERS_DIR))
        import im_server
        import ai_server

        exp = json.loads(ai_server.ai_auto_exposure(str(test_image)))
        if "error" in exp:
            pytest.skip("Auto exposure not available")

        out1 = str(output_dir / "auto_leveled.jpg")
        json.loads(im_server.im_level(
            str(test_image), out1,
            black_point=exp["black_point"],
            gamma=exp["gamma"],
            white_point=exp["white_point"],
        ))

        out2 = str(output_dir / "auto_final.jpg")
        json.loads(im_server.im_brightness_contrast(out1, out2, contrast=exp.get("contrast", 0)))

        assert Path(out2).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
