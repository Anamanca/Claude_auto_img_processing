# Hybrid AI + CLI Image/Video Processing Pipeline

**Four MCP servers** — AI decides WHAT & WHERE, CLI tools execute HOW at scale.

**Target hardware**: RTX 2060 Super (8GB VRAM) + Ryzen 7 5800X (8C/16T) + 32GB RAM

## Quick Start

```bash
# 1. Install
bash install.sh
source venv/bin/activate

# 2. Download AI models
python download_models.py

# 3. Start MCP servers (in separate terminals, or configure in CLAUDE.md)
python mcp_servers/im_server.py      # ImageMagick
python mcp_servers/ffmpeg_server.py  # FFmpeg + NVENC
python mcp_servers/ai_server.py      # AI models (GPU)
python mcp_servers/util_server.py    # RAW, EXIF, file management
```

## Architecture

```
User Prompt (Claude Code)
    │
    ├─ AI MCP  → Hiểu nội dung ảnh (face, skin, exposure params)
    ├─ IM MCP  → Thực thi pixel operations (resize, crop, color, masked blur...)
    ├─ FFmpeg MCP → Video processing (NVENC, filters, transcode...)
    └─ Util MCP → RAW development, EXIF, file management
```

**Principle**: AI tools return analysis/masks → CLI tools execute batch operations using those results.

## MCP Server Summary

### IM MCP (32 tools)
`python mcp_servers/im_server.py`

| Category | Tools |
|---|---|
| **Transform** | `im_resize`, `im_crop`, `im_rotate`, `im_flip`, `im_trim`, `im_distort` |
| **Color** | `im_brightness_contrast`, `im_modulate`, `im_level`, `im_auto_level`, `im_equalize`, `im_grayscale`, `im_sepia`, `im_negate`, `im_colorize`, `im_lut3d` |
| **Filter** | `im_blur`, `im_sharpen`, `im_unsharp_mask`, `im_adaptive_blur`, `im_median`, `im_denoise`, `im_vignette` |
| **Composite** | `im_composite`, `im_watermark`, `im_annotate`, `im_append` |
| **Masked** | `im_masked_blur`, `im_masked_sharpen`, `im_masked_modulate` |
| **Format** | `im_convert_format`, `im_identify`, `im_strip`, `im_set_icc` |
| **Batch** | `im_batch` (run one operation on all files matching a glob) |

### FFmpeg MCP (28 tools)
`python mcp_servers/ffmpeg_server.py`

| Category | Tools |
|---|---|
| **Transform** | `ffmpeg_scale`, `ffmpeg_crop`, `ffmpeg_rotate`, `ffmpeg_trim`, `ffmpeg_pad`, `ffmpeg_deshake`, `ffmpeg_vstab` |
| **Speed** | `ffmpeg_speed`, `ffmpeg_slowmo`, `ffmpeg_reverse`, `ffmpeg_fade` |
| **Color** | `ffmpeg_colorbalance`, `ffmpeg_eq`, `ffmpeg_curves`, `ffmpeg_lut3d`, `ffmpeg_normalize`, `ffmpeg_hue` |
| **Filter** | `ffmpeg_blur`, `ffmpeg_sharpen`, `ffmpeg_denoise`, `ffmpeg_deband` |
| **Composite** | `ffmpeg_overlay`, `ffmpeg_drawtext`, `ffmpeg_xfade`, `ffmpeg_concat` |
| **Codec** | `ffmpeg_transcode`, `ffmpeg_nvenc`, `ffmpeg_extract_audio`, `ffmpeg_extract_frames`, `ffmpeg_probe` |
| **Batch** | `ffmpeg_batch_transcode` |

### AI MCP (12 tools)
`python mcp_servers/ai_server.py`

| Tool | Purpose | GPU |
|---|---|---|
| `ai_detect_faces` | Face detection → bounding boxes + landmarks | ✓ |
| `ai_face_mesh` | 468-point face mesh (MediaPipe) | — |
| `ai_skin_mask` | Skin segmentation mask | ✓ |
| `ai_eyes_mask` | Eye region mask | — |
| `ai_auto_exposure` | Analyze image → optimal levels params | — |
| `ai_super_resolution` | RealESRGAN upscaling (2x/4x) | ✓ |
| `ai_denoise` | NAFNet denoising | ✓ |
| `ai_remove_background` | U2-Net background removal | ✓ |
| `ai_portrait_matting` | Portrait-specific matting | ✓ |
| `ai_face_track` | Track faces across video frames | — |
| `ai_scene_detect` | Scene cut detection | — |
| `ai_unload_models` | Free VRAM by unloading models | — |

### Util MCP (9 tools)
`python mcp_servers/util_server.py`

| Tool | Purpose |
|---|---|
| `util_raw_develop` | Develop RAW → 16-bit TIFF (rawpy/LibRaw) |
| `util_raw_to_jpeg` | RAW → JPEG direct |
| `util_raw_metadata` | Read camera EXIF from RAW |
| `util_exif_read` | Read all metadata via exiftool |
| `util_exif_write` | Write metadata |
| `util_compare_images` | PSNR, SSIM, MSE comparison |
| `util_file_list` | List media files in directory |
| `util_hash` | SHA256 file hash (dedup) |
| `util_organize` | Organize files by date/category |

## Key Workflows

### Portrait Enhancement (Batch)
```
1. ai_detect_faces → bounding boxes for every face
2. ai_skin_mask → grayscale PNG (255=skin, 0=non-skin)
3. ai_eyes_mask → eye regions to exclude
4. Subtract eyes from skin mask → final mask
5. ai_auto_exposure → {black_point, gamma, white_point, contrast}
6. im_masked_blur(mask=final_mask, sigma=3) → smooths only skin
7. im_masked_sharpen(mask=eyes_mask, sigma=1) → sharpens eyes
8. im_level(black, gamma, white) → fixes exposure
9. im_convert_format(JPEG, quality=92) → export
```

### RAW to JPEG (Batch)
```
1. util_file_list → all .CR2/.NEF files
2. For each: util_raw_develop → 16-bit TIFF
3. ai_auto_exposure → optimal params
4. im_level + im_modulate → apply params
5. im_convert_format(JPEG) → export
```

### Video Transcode (NVENC)
```
1. ffmpeg_nvenc → H.265 NVENC encode at cq=23
   or ffmpeg_transcode → H.264 software encode
```

### Smart Library Organization
```
1. util_file_list(recursive=true)
2. For each: util_exif_read → date, camera
3. util_organize(pattern="{date:%Y}/{date:%m}/{category}/{filename}")
```

## GPU Memory Budget

| State | Models Loaded | VRAM Used |
|---|---|---|
| Idle | none | ~0.5 GB |
| Face detection | RetinaFace | ~0.9 GB |
| Skin mask | RetinaFace + BiSeNet | ~1.2 GB |
| Upscaling | RealESRGAN x4 | ~2.0 GB |
| Denoising | NAFNet | ~1.1 GB |
| BG removal | U2-Net | ~1.0 GB |

RTX 2060S (8GB) can handle face detection + skin mask + one heavy model simultaneously.
Call `ai_unload_models()` to free VRAM after heavy operations.

## Performance Estimates

| Task | Throughput | Bottleneck |
|---|---|---|
| Batch resize (24MP) | ~12,000/hr | CPU (16 threads) |
| Portrait enhancement | ~5,000-7,000/hr | CPU (IM) |
| AI upscale (1080p→4K) | ~60-120/hr | GPU (8GB VRAM) |
| Video transcode (NVENC) | 5-8x realtime | NVENC ASIC |
| Video transcode (x265 CPU) | 1-2x realtime | CPU |
| RAW → JPEG (24MP) | ~500-1,200/hr | CPU (LibRaw) |

## Configuration

Edit `config.yaml` to change:
- Binary paths (if ImageMagick/FFmpeg not on PATH)
- GPU device and VRAM limits
- Default JPEG quality, sharpen/blur params
- Model file locations

## File Structure

```
hybrid_pipeline/
├── config.yaml              ← Configuration
├── install.sh               ← One-command installer
├── requirements.txt         ← Python dependencies
├── download_models.py       ← AI model downloader
├── CLAUDE.md                ← This file
├── luts/                    ← Color LUTs (.cube files)
├── profiles/                ← ICC profiles
├── models/                  ← Downloaded AI models
├── output/                  ← Default output directory
├── logs/                    ← Server logs
├── temp/                    ← Temporary masks & intermediates
├── mcp_servers/
│   ├── common.py            ← Shared utilities
│   ├── im_server.py         ← ImageMagick MCP
│   ├── ffmpeg_server.py     ← FFmpeg MCP
│   ├── ai_server.py         ← AI models MCP
│   └── util_server.py       ← Utility MCP
└── tests/
    └── test_pipeline.py     ← Test suite
```

## Troubleshooting

| Problem | Solution |
|---|---|
| `convert: command not found` | `sudo apt install imagemagick` |
| `ffmpeg: command not found` | `sudo apt install ffmpeg` |
| CUDA not detected | Reinstall PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
| AI tools error about model not found | `python download_models.py` |
| NVENC not available | Check NVIDIA driver: `nvidia-smi` |
| Out of VRAM | Call `ai_unload_models()` before another heavy model |
| FastMCP import error | `pip install fastmcp` |
