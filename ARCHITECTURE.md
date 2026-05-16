# Hybrid AI + CLI Image/Video Processing Pipeline — Architecture Plan

**Target hardware**: RTX 2060 Super (8GB VRAM) + Ryzen 7 5800X (8C/16T) + 32GB RAM  
**Scope**: Architecture only — no implementation in this document  
**Principle**: AI decides WHAT & WHERE, CLI tools execute HOW at scale

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Claude Code (Orchestrator)             │
│  - Nhận yêu cầu người dùng bằng tiếng Việt / tiếng Anh   │
│  - Lập kế hoạch pipeline                                 │
│  - Điều phối các MCP server                              │
│  - Truyền tham số giữa các bước                          │
└──────────┬──────────┬──────────┬──────────┬─────────────┘
           │          │          │          │
     ┌─────▼────┐┌───▼───┐┌────▼────┐┌───▼────┐
     │ IM MCP   ││FFmpeg ││ AI MCP  ││Util MCP │
     │ Server   ││ MCP   ││ Server  ││ Server  │
     └────┬─────┘└───┬───┘└────┬────┘└───┬─────┘
          │          │         │         │
    ┌─────▼────┐┌───▼───┐┌───▼────┐┌───▼──────┐
    │ImageMagick││ FFmpeg││PyTorch ││ rawpy    │
    │  (CPU)   ││(NVENC)││(CUDA)  ││ exiftool │
    └──────────┘└───────┘└────────┘└──────────┘
```

### Four MCP Servers (one process per domain)

| MCP Server | Language | Binds to | Runs on |
|---|---|---|---|
| **IM MCP** | Python (subprocess) | ImageMagick `convert`/`mogrify`/`identify` CLI | CPU |
| **FFmpeg MCP** | Python (subprocess) | FFmpeg CLI + NVENC | GPU (encode) + CPU (filters) |
| **AI MCP** | Python (native) | PyTorch, ONNX Runtime, OpenCV | GPU (CUDA) |
| **Util MCP** | Python (subprocess) | rawpy, exiftool, Pillow | CPU |

### Why 4 separate MCP servers (not 1 monolith)?

- **Isolation**: AI server loads heavy PyTorch models (~2-4GB VRAM). Nếu crash thì IM/FFmpeg vẫn chạy.
- **Queue management**: IM MCP có thể xử lý hàng ngàn ảnh song song CPU-bound, trong khi AI MCP xử lý tuần tự GPU-bound.
- **Resource scheduling**: Tránh GPU OOM khi vừa load model AI vừa chạy NVENC encode.
- **Independent scaling**: Có thể tắt AI MCP để tiết kiệm VRAM khi chỉ cần resize/crop batch.

---

## 2. IM MCP Server (ImageMagick)

### 2.1 Tool List (32 tools)

#### Category: TRANSFORM
| Tool | ImageMagick command | Params |
|---|---|---|
| `im_resize` | `-resize` | WxH, filter (Lanczos/Mitchell/CatmullRom), keep_aspect |
| `im_crop` | `-crop` | WxH+X+Y |
| `im_rotate` | `-rotate` | degrees, background_color |
| `im_flip` | `-flip` / `-flop` | direction (vertical/horizontal) |
| `im_trim` | `-trim` | fuzz_percent |
| `im_distort` | `-distort` | method (Perspective/Affine/Arc/Barrel), coords |

#### Category: COLOR
| Tool | ImageMagick command | Params |
|---|---|---|
| `im_brightness_contrast` | `-brightness-contrast` | brightness, contrast |
| `im_modulate` | `-modulate` | brightness, saturation, hue |
| `im_level` | `-level` | black_point, gamma, white_point |
| `im_auto_level` | `-auto-level` | — |
| `im_auto_gamma` | `-auto-gamma` | — |
| `im_equalize` | `-equalize` | — |
| `im_grayscale` | `-grayscale` | method (Rec709Luma/Rec601Luma/Average/Lightness) |
| `im_sepia` | `-sepia-tone` | threshold |
| `im_negate` | `-negate` | — |
| `im_colorize` | `-colorize` | color, percent |
| `im_lut3d` | `-hald-clut` | lut_file_path |

#### Category: FILTER
| Tool | ImageMagick command | Params |
|---|---|---|
| `im_blur` | `-blur` / `-gaussian-blur` | radius, sigma |
| `im_sharpen` | `-sharpen` | radius, sigma |
| `im_unsharp_mask` | `-unsharp` | radius, sigma, gain, threshold |
| `im_adaptive_blur` | `-adaptive-blur` | radius, sigma |
| `im_median` | `-median` | radius |
| `im_denoise` | `-enhance` / `-despeckle` | — |
| `im_vignette` | `-vignette` | radius, sigma, x, y |
| `im_border` | `-border` / `-frame` | WxH, color |

#### Category: COMPOSITE
| Tool | ImageMagick command | Params |
|---|---|---|
| `im_composite` | `composite` sub-command | overlay_image, position (gravity), blend_mode, opacity |
| `im_watermark` | `composite -dissolve` | watermark_file, opacity_percent, gravity |
| `im_annotate` | `-annotate` / `-draw text` | text, gravity, font_size, color, font_file |
| `im_append` | `-append` / `+append` | direction (vertical/horizontal), gap |

#### Category: MASKED EDIT (region-aware — used with AI mask)
| Tool | ImageMagick command | Params |
|---|---|---|
| `im_masked_blur` | `-mask` + `-blur` | mask_file_path, radius, sigma |
| `im_masked_sharpen` | `-mask` + `-sharpen` | mask_file_path, radius, sigma |
| `im_masked_modulate` | `-mask` + `-modulate` | mask_file_path, brightness, saturation, hue |

#### Category: FORMAT & META
| Tool | ImageMagick command | Params |
|---|---|---|
| `im_convert_format` | format conversion | input, output_format (JPEG/PNG/WebP/AVIF/HEIC/TIFF), quality, progressive |
| `im_identify` | `identify` | file_path → returns: size, colorspace, bit_depth, EXIF, mean, stddev, histogram |
| `im_strip` | `-strip` | — (removes EXIF/profiles) |
| `im_set_icc` | `-profile` | icc_profile_path |

### 2.2 Key Design Decisions

- **Single-file vs batch**: Mỗi tool nhận 1 file path. Riêng `im_batch` là meta-tool nhận thư mục + glob pattern, chạy cùng 1 operation trên tất cả file match. Dùng `mogrify` thay `convert` cho in-place, hoặc `convert` cho output dir riêng.
- **Output convention**: Mọi tool trả về path file output hoặc JSON metadata (cho `im_identify`). Không trả base64.
- **Error handling**: Capture stderr, parse ImageMagick error codes, trả về structured error JSON.
- **Progress**: Với batch > 100 files, dùng `-monitor` flag và parse progress stream.

---

## 3. FFmpeg MCP Server

### 3.1 Tool List (28 tools)

#### Category: TRANSFORM
| Tool | FFmpeg command | Params |
|---|---|---|
| `ffmpeg_scale` | `scale` filter | WxH, algo (lanczos/bilinear/bicubic/neighbor) |
| `ffmpeg_crop` | `crop` filter | W:H:X:Y |
| `ffmpeg_rotate` | `rotate` / `transpose` filter | angle or transpose_dir |
| `ffmpeg_trim` | `-ss -to` or `trim` filter | start_time, end_time/duration |
| `ffmpeg_pad` | `pad` filter | W:H:X:Y:color |
| `ffmpeg_deshake` | `deshake` filter | — |
| `ffmpeg_vstab` | `vidstabdetect` + `vidstabtransform` | shakiness, smoothing |

#### Category: SPEED & TIME
| Tool | FFmpeg command | Params |
|---|---|---|
| `ffmpeg_speed` | `setpts` + `atempo` | speed_multiplier |
| `ffmpeg_slowmo` | `minterpolate` filter | target_fps (for smooth slow-mo) |
| `ffmpeg_reverse` | `reverse` filter | — |
| `ffmpeg_fade` | `fade` filter | type (in/out), duration, start_frame |

#### Category: COLOR
| Tool | FFmpeg command | Params |
|---|---|---|
| `ffmpeg_colorbalance` | `colorbalance` filter | rs, gs, bs, rm, gm, bm, rh, gh, bh |
| `ffmpeg_eq` | `eq` filter | brightness, contrast, gamma, saturation |
| `ffmpeg_curves` | `curves` filter | preset or master/red/green/blue points |
| `ffmpeg_lut3d` | `lut3d` filter | cube_file_path |
| `ffmpeg_normalize` | `normalize` filter | — |
| `ffmpeg_hue` | `hue` filter | h (degrees), s (multiplier) |

#### Category: FILTER
| Tool | FFmpeg command | Params |
|---|---|---|
| `ffmpeg_blur` | `gblur` / `boxblur` filter | sigma / radius |
| `ffmpeg_sharpen` | `unsharp` / `cas` filter | luma_msize_x, luma_amount... |
| `ffmpeg_denoise` | `nlmeans` / `hqdn3d` filter | strength |
| `ffmpeg_deband` | `deband` filter | range, direction, blur |

#### Category: COMPOSITE
| Tool | FFmpeg command | Params |
|---|---|---|
| `ffmpeg_overlay` | `overlay` filter | overlay_video, x, y, alpha |
| `ffmpeg_drawtext` | `drawtext` filter | text, fontfile, fontsize, x, y, fontcolor, shadow |
| `ffmpeg_xfade` | `xfade` filter | transition_type, duration, offset |
| `ffmpeg_concat` | `concat` demuxer or filter | file_list (for stream-copy) or transition (for filter) |

#### Category: CODEC & FORMAT
| Tool | FFmpeg command | Params |
|---|---|---|
| `ffmpeg_transcode` | full transcode | video_codec (h264/h265/av1/prores), audio_codec, crf/bitrate, preset, pixel_format |
| `ffmpeg_nvenc` | nvenc transcode | h264_nvenc / hevc_nvenc, cq/bitrate, preset (p1-p7), b_frames |
| `ffmpeg_extract_audio` | `-vn` | audio_codec (mp3/aac/flac/opus), bitrate |
| `ffmpeg_extract_frames` | `-vf fps=N` | fps, start_time, end_time, image_format |
| `ffmpeg_probe` | `ffprobe` | file_path → JSON: codec, bitrate, duration, resolution, streams, pixel_format, color_space |

### 3.2 Key Design Decisions

- **NVENC path**: Khi `ffmpeg_nvenc` được gọi và GPU available, tự động dùng `h264_nvenc` / `hevc_nvenc`. Fallback về `libx264` / `libx265` nếu không có GPU.
- **Stream-copy where possible**: `ffmpeg_trim` và `ffmpeg_concat` default dùng stream copy (`-c copy`) để tránh re-encode. Chỉ re-encode khi có filter.
- **Progress reporting**: Parse `progress=continue` line từ stderr để báo % hoàn thành.

---

## 4. AI MCP Server (PyTorch / ONNX)

### 4.1 Model Inventory

| Model | Framework | VRAM | Purpose | Input → Output |
|---|---|---|---|---|
| **RetinaFace** (ResNet50) | ONNX Runtime | ~400MB | Face detection | Image → [(x,y,w,h, landmarks)] |
| **MediaPipe Face Mesh** | MediaPipe | ~200MB | 468-point face landmark | Image → [[x,y,z] × 468] |
| **BiSeNet** (face parsing) | ONNX Runtime | ~300MB | Skin/hair/eyes/mouth segmentation | Face crop → mask [skin:1, eyes:2, brows:3, lips:4, hair:5...] |
| **RealESRGAN** (x4) | PyTorch CUDA | ~1.5GB | Super resolution / sharpening | Image → 4x upscaled image |
| **RealESRGAN** (x2) | PyTorch CUDA | ~800MB | Lightweight upscale | Image → 2x upscaled image |
| **NAFNet** (denoise) | PyTorch CUDA | ~600MB | Denoising | Noisy image → clean image |
| **rembg** (U2-Net) | ONNX Runtime | ~500MB | Background removal | Image → RGBA (alpha matte) |
| **AutoLevel** (custom) | NumPy/OpenCV | N/A (CPU) | Smart exposure analysis | Image → {black_point, white_point, gamma, contrast} |

### 4.2 Tool List (12 tools)

#### Face & Portrait
| Tool | Uses | Description |
|---|---|---|
| `ai_detect_faces` | RetinaFace | Trả về bounding boxes + 5-point landmarks cho mọi khuôn mặt trong ảnh |
| `ai_face_mesh` | MediaPipe | 468-điểm face mesh cho khuôn mặt lớn nhất |
| `ai_skin_mask` | BiSeNet (parsing) | Trả về path tới file mask (grayscale PNG: 255 = skin, 0 = background/features). **Đây là input cho `im_masked_*` tools** |
| `ai_eyes_mask` | BiSeNet (parsing) | Trả về mask vùng mắt (để tránh làm mờ mắt khi làm mịn da) |
| `ai_beauty_score` | Custom CNN | Cho điểm chất lượng chân dung (ánh sáng, độ nét, góc mặt) — dùng để filter ảnh đẹp/xấu |

#### Enhancement
| Tool | Uses | Description |
|---|---|---|
| `ai_super_resolution` | RealESRGAN x2/x4 | Upscale ảnh với AI (giữ chi tiết tốt hơn Lanczos nhiều lần) |
| `ai_denoise` | NAFNet | Khử noise ảnh chụp ISO cao |
| `ai_auto_exposure` | AutoLevel | Phân tích histogram, trả về tham số `-level` tối ưu cho IM |

#### Segmentation & Compositing
| Tool | Uses | Description |
|---|---|---|
| `ai_remove_background` | rembg / U2-Net | Tách nền, trả về ảnh PNG trong suốt |
| `ai_portrait_matting` | MODNet (lightweight) | Tách người khỏi nền (nhanh hơn U2-Net, tốt cho portrait) |

#### Video-specific
| Tool | Uses | Description |
|---|---|---|
| `ai_face_track` | RetinaFace per-frame + IoU tracking | Theo dõi khuôn mặt qua các frame video → trả về list [frame_idx, x, y, w, h] |
| `ai_scene_detect` | PySceneDetect + custom | Phát hiện ranh giới cảnh → list timestamps |

### 4.3 GPU Memory Budget

```
Total VRAM:          8 GB (RTX 2060S)
Reserved (system):  ~500 MB
─────────────────────────────
Available:          ~7.5 GB

Active models at any time:
- RetinaFace (ONNX):     400 MB
- BiSeNet (ONNX):        300 MB  
- RealESRGAN (PyTorch): 1500 MB  (only when upscaling)
- NAFNet (PyTorch):      600 MB  (only when denoising)
- rembg (ONNX):          500 MB  (only when matting)
─────────────────────────────
Max simultaneous:    ~2.8 GB (face + skin + 1 heavy model)
```

Strategy: Load/unload heavy models (RealESRGAN, NAFNet, rembg) on demand. Keep lightweight models (RetinaFace, BiSeNet) always loaded as they're needed for nearly every portrait task.

---

## 5. Util MCP Server

### 5.1 Tool List

| Tool | Purpose | Backend |
|---|---|---|
| `util_raw_develop` | Phát triển file RAW (CR2/NEF/ARW/DNG) → 16-bit TIFF | `rawpy` (LibRaw) |
| `util_raw_develop_jpeg` | RAW → JPEG trực tiếp với tham số | `rawpy` → PIL |
| `util_raw_metadata` | Đọc EXIF camera từ RAW (ISO, aperture, shutter, lens, WB) | `rawpy` / `exiftool` |
| `util_exif_read` | Đọc EXIF/XMP/IPTC từ JPEG/TIFF/PNG | `exiftool` |
| `util_exif_write` | Ghi copyright, author, keywords | `exiftool` |
| `util_hash` | SHA256 hash file để dedup | `hashlib` |
| `util_compare_images` | So sánh 2 ảnh (PSNR, SSIM, MSE) | `skimage.metrics` |
| `util_file_list` | Liệt kê files trong thư mục (filter extension, size, date) | `pathlib` |
| `util_organize` | Tổ chức files theo pattern: date/camera/model/rating | `pathlib` + `exiftool` |

### 5.2 RAW Processing Details

`util_raw_develop` params:
```json
{
  "input": "photo.cr2",
  "output": "photo.tiff",
  "demosaic_algorithm": "AMAZE",       // AMAZE, AHD, AAHD, LMMSE, PPG
  "use_camera_wb": true,
  "brightness": 1.0,
  "highlight_mode": 1,                 // 0=clip, 1=unclip, 2=blend
  "exp_shift": 0.0,
  "output_color_space": "sRGB",        // sRGB, Adobe, XYZ, ProPhoto, ACES
  "output_bps": 16,                    // 8 or 16
  "auto_bright": true,                 // auto brightness thx
  "median_filter_passes": 1,           // denoise RAW gốc
  "enable_lens_correction": false      // cần lensfun database
}
```

---

## 6. Hybrid Pipelines — Complete Workflows

### Pipeline A: Batch Portrait Enhancement

**User nói**: "Làm đẹp 1000 ảnh chân dung: làm mịn da, làm nét mắt, cân bằng sáng, xuất JPEG chất lượng 90%"

```
Step 1: AI Agent (Claude) phân tích yêu cầu → lập plan
        Tools cần: ai_detect_faces, ai_skin_mask, ai_auto_exposure,
                    im_masked_blur, im_masked_sharpen, im_level,
                    im_convert_format

Step 2: util_file_list → lấy danh sách 1000 files

Step 3: FOR EACH image (có thể chạy song song 8 ảnh trên CPU):
  3a:  ai_detect_faces(image)
       → Nếu không có mặt → skip, đánh dấu "non-portrait"
       → Nếu có mặt → lấy bbox khuôn mặt

  3b:  ai_skin_mask(face_crop)
       → File mask_skin.png (255=vùng da, 0=phần còn lại)
       → TRỪ vùng mắt: ai_eyes_mask → mask_eyes.png
       → mask_final = mask_skin - mask_eyes

  3c:  ai_auto_exposure(image)
       → {black: 12, gamma: 1.05, white: 240}

  3d:  IM MCP: im_masked_blur(image, mask=mask_final, radius=0, sigma=3)
       → Làm mịn da, giữ nét mắt/tóc

  3e:  IM MCP: im_masked_sharpen(image, mask=mask_eyes, radius=0, sigma=1.0)
       → Làm nét mắt

  3f:  IM MCP: im_level(image, black=12, gamma=1.05, white=240)
       → Cân bằng sáng theo tham số AI tính

  3g:  IM MCP: im_convert_format(image, format="JPEG", quality=90, progressive=true)
       → Export JPEG

Step 4: AI Agent báo cáo:
        ✓ 847 portraits enhanced
        ⊘ 153 skipped (no face detected)
        ✗ 0 errors
        ⏱  Average: 2.3s/image
        📦 Output: /output/enhanced/
```

**Performance estimate trên RTX 2060S + R7 5800X**:
- RetinaFace: ~30ms/ảnh (ONNX GPU)
- BiSeNet parsing: ~40ms/ảnh (ONNX GPU)
- ImageMagick operations: ~200-400ms/ảnh (CPU, ảnh 24MP)
- **Total: ~300-500ms/ảnh** → ~2000-3000 ảnh/giờ single-threaded
- Với 8 ảnh song song (CPU-bound bước IM): **~5000-7000 ảnh/giờ**

### Pipeline B: RAW Workflow

**User nói**: "Chuyển 500 file RAW sang JPEG, tự động cân bằng sáng, giảm noise, áp preset màu Fuji"

```
Step 1: util_file_list → 500 files *.CR2

Step 2: FOR EACH raw file (GPU 1 file/lần, CPU xử lý pipeline):
  2a: util_raw_develop(raw, output="temp_16bit.tiff",
       demosaic="AMAZE", use_camera_wb=true, auto_bright=true,
       median_filter_passes=1, output_bps=16)
       → 16-bit TIFF (linear)

  2b: [Optional: nếu ISO > 3200] ai_denoise(tiff) 
       → Denoised TIFF (NAFNet)

  2c: ai_auto_exposure(tiff)
       → Tham số level

  2d: IM MCP: im_level(tiff, ...) + im_modulate(tiff, saturation=1.05)
       → Base color correction

  2e: IM MCP: im_lut3d(tiff, lut="Fuji_ClassicChrome.cube")
       → Áp preset màu

  2f: IM MCP: im_convert_format(tiff, "JPEG", quality=92, progressive=true)
       + im_strip (keep minimal EXIF: copyright, camera, lens, ISO)
       → JPEG cuối
```

### Pipeline C: Video Enhancement

**User nói**: "Video 1 giờ từ iPhone, chống rung, làm nét, nén H.265 NVENC"

```
Step 1: ffmpeg_probe(video) → metadata

Step 2: ffmpeg_vstab(video, shaking=medium, smoothing=15)
        → Pass 1: detect transforms to file
        → Pass 2: apply transforms
        → Stabilized intermediate (ProRes HQ để tránh generational loss)

Step 3: ffmpeg_transcode(intermediate,
        video_codec="hevc_nvenc",      # NVENC GPU encode
        preset="p4",                    # Medium quality
        cq=23,                          # Constant quality
        filter_chain="unsharp=5:5:1.5", # Sharpen
        pixel_format="yuv420p",
        audio_codec="aac",
        audio_bitrate="192k")
        → Final MP4

Performance: ~5-8x realtime (NVENC encode), 30-50 fps filter chain
```

### Pipeline D: AI-Assisted Batch Crop & Composition

**User nói**: "Cắt ảnh sản phẩm tự động, xóa nền, resize về 1000x1000, nền trắng"

```
Step 1: ai_remove_background(image)
        → PNG with alpha

Step 2: IM MCP: im_trim(png, fuzz=5%) 
        → Tự động cắt sát sản phẩm

Step 3: IM MCP: im_resize(trimmed, "900x900", filter="Lanczos")
        → Resize giữ aspect ratio

Step 4: IM MCP: im_extent(resized, "1000x1000", gravity="center",
        background="white")
        → Đặt lên canvas trắng 1000x1000

Step 5: IM MCP: im_convert_format(..., "JPEG", quality=85)
        → Export
```

### Pipeline E: Smart Library Organization

**User nói**: "Tổ chức ảnh vào thư mục theo ngày chụp + phân loại portrait/landscape/other"

```
Step 1: util_file_list(recursive=true)

Step 2: FOR EACH image:
  2a: im_identify(image) → kích thước, aspect ratio
  2b: util_exif_read(image) → ngày chụp, camera
  2c: ai_detect_faces(image) → có/k
hông có mặt, số lượng
  2d: Phân loại:
      - Có ≥1 mặt + aspect portrait → "portrait"
      - Không có mặt + aspect wide → "landscape"
      - Còn lại → "other"

Step 3: util_organize(pattern="/{date:%Y/%m}/{category}/{filename}")
```

---

## 7. Coordination & State Management

### 7.1 How the AI Agent Orchestrates

```
User prompt
    │
    ▼
Claude Code (via MCP tools)
    │
    ├─ Gọi AI MCP tools để HIỂU ảnh (face location, skin mask, exposure params)
    │
    ├─ Nhận kết quả phân tích (JSON: bounding boxes, mask paths, level params)
    │
    ├─ Dùng kết quả đó để BUILD COMMAND cho IM/FFmpeg
    │
    └─ Gọi IM/FFmpeg MCP tools với tham số đã tính
```

Claude **không** xử lý pixel. Claude là orchestrator:
- Hiểu user intent bằng ngôn ngữ tự nhiên
- Chọn pipeline phù hợp
- Truyền tham số giữa các bước (output của AI MCP → input của IM MCP)
- Xử lý edge cases (ảnh không có mặt, file hỏng, VRAM đầy...)
- Báo cáo kết quả

### 7.2 Data Flow Convention

- **Images**: Luôn truyền bằng **file path**, không bao giờ base64. AI MCP trả về file path của mask, IM MCP nhận file path của mask.
- **Metadata**: JSON objects.
- **Masks**: Grayscale PNG, 255 = vùng cần xử lý, 0 = bỏ qua. Lưu trong thư mục temp, xóa sau pipeline.
- **Temp files**: Tạo trong `/tmp/hybrid_pipeline/{job_id}/`, dọn sau 1 giờ.

### 7.3 Error Handling Strategy

| Lỗi | Cách xử lý |
|---|---|
| AI model OOM | Unload model không dùng, retry với batch_size=1 |
| Ảnh hỏng / không đọc được | Skip, ghi vào error.log |
| FFmpeg encode fail | Thử codec fallback (NVENC → x265 → x264) |
| Mask rỗng (không detect được da) | Fallback: làm mịn toàn ảnh với radius nhỏ hơn |
| Hết disk | Dừng pipeline, báo user |

---

## 8. Hardware Utilization Strategy

```
┌─────────────────────────────────────────────┐
│                  TASK TYPE                   │
├──────────┬──────────┬──────────┬────────────┤
│ AI Infer │ IM/FF   │ RAW Dev  │ NVENC      │
│ (GPU)    │ (CPU)   │ (CPU)   │ (GPU)      │
├──────────┼──────────┼──────────┼────────────┤
│ CUDA     │ 8-16     │ libraw   │ NVENC ASIC │
│ cores    │ threads  │ (AVX2)   │ (separate  │
│          │          │          │  from CUDA)│
└──────────┴──────────┴──────────┴────────────┘

Có thể chạy SONG SONG:
- NVENC encode (dùng NVENC ASIC riêng) + AI inference (dùng CUDA cores)
- ImageMagick batch (dùng 16 threads CPU) + AI inference (GPU)
- KHÔNG nên: 2 model AI cùng chạy GPU (VRAM fragmentation)
```

Thứ tự ưu tiên GPU:
1. AI inference (CUDA cores + VRAM)
2. NVENC encode (NVIDIA Encoder ASIC — không chiếm CUDA cores nhiều)
3. (NVENC có thể chạy song song với AI inference)

---

## 9. Directory Structure (on target PC)

```
~/hybrid_pipeline/
├── config.yaml                     # Paths, GPU settings, defaults
├── mcp_servers/
│   ├── im_server.py                # ImageMagick MCP
│   ├── ffmpeg_server.py            # FFmpeg MCP
│   ├── ai_server.py                # AI models MCP
│   └── util_server.py              # Utility MCP
├── models/                         # Downloaded AI models
│   ├── retinaface.onnx
│   ├── bisenet.onnx
│   ├── realesrgan_x4.pth
│   ├── nafnet.pth
│   ├── u2net.onnx
│   └── face_mesh/
├── luts/                           # Color presets
│   ├── Fuji_ClassicChrome.cube
│   ├── Kodak_Portra400.cube
│   └── Cinematic_TealOrange.cube
├── profiles/                       # ICC profiles
│   ├── sRGB.icc
│   ├── AdobeRGB1998.icc
│   └── ProPhoto.icc
├── logs/
├── temp/                           # Mask files, intermediates
└── CLAUDE.md                       # MCP config for Claude Code
```

### CLAUDE.md snippet (MCP config)

```json
{
  "mcpServers": {
    "im": {
      "command": "python",
      "args": ["~/hybrid_pipeline/mcp_servers/im_server.py"],
      "description": "ImageMagick batch image processing"
    },
    "ffmpeg": {
      "command": "python",
      "args": ["~/hybrid_pipeline/mcp_servers/ffmpeg_server.py"],
      "description": "FFmpeg video processing with NVENC"
    },
    "ai": {
      "command": "python",
      "args": ["~/hybrid_pipeline/mcp_servers/ai_server.py"],
      "description": "AI models: face detection, skin segmentation, upscaling, denoising"
    },
    "util": {
      "command": "python",
      "args": ["~/hybrid_pipeline/mcp_servers/util_server.py"],
      "description": "RAW processing, EXIF, file management"
    }
  }
}
```

---

## 10. Dependencies (target PC)

### System
```bash
sudo apt install imagemagick ffmpeg exiftool
```

### Python
```
# MCP framework
mcp>=1.0.0
fastmcp

# AI inference
onnxruntime-gpu>=1.17
torch>=2.1  (CUDA 12.1)
mediapipe

# Image/video
opencv-python-headless
rawpy
pillow
scikit-image
pyscenedetect

# Utilities
pyyaml
tqdm
exiftool  (via subprocess)
```

### Model downloads
```
RetinaFace ONNX     → ~50 MB
BiSeNet ONNX        → ~30 MB
RealESRGAN x4       → ~67 MB
RealESRGAN x2       → ~67 MB  
NAFNet              → ~260 MB
U2-Net ONNX         → ~50 MB
MediaPipe Face Mesh → tự động download
─────────────────────────
Total download:     ~524 MB
Total on disk:      ~600 MB (some compressed)
```

---

## 11. Implementation Priority

| Phase | What | Effort | Value |
|---|---|---|---|
| **Phase 1** | IM MCP + FFmpeg MCP (no AI yet) | 2-3 days | Đã dùng được batch resize/crop/convert/watermark/transcode ngay |
| **Phase 2** | AI MCP: face detection + skin mask + auto exposure | 2-3 days | Mở khóa portrait enhancement pipeline |
| **Phase 3** | Util MCP: RAW develop + EXIF | 1 day | RAW workflow |
| **Phase 4** | AI MCP: RealESRGAN + NAFNet | 1-2 days | Upscale & denoise |
| **Phase 5** | AI MCP: background removal + matting | 1 day | Product photo pipeline |
| **Phase 6** | AI MCP: face tracking + scene detect | 1-2 days | Video enhancement |

**Total: ~10-14 days solo developer effort**

---

## 12. Key Limitations (honest assessment)

1. **Không thay thế được Lightroom/Photoshop cho tác vụ thủ công phức tạp.** Pipeline này mạnh về batch, yếu về single-image precision editing (liquify, dodge/burn, frequency separation).
2. **AI skin smoothing có thể bị uncanny valley nếu quá mạnh.** Cần tham số conservative làm default.
3. **RAW color science không bằng Adobe/Lightroom.** LibRaw xử lý RAW OK nhưng color profiles của hãng máy ảnh (Camera Matching profiles) là độc quyền.
4. **VRAM 8GB là giới hạn cho model lớn.** Không chạy được SDXL/Flux trong cùng session với các model khác.
5. **Video AI upscaling (Video2X) sẽ rất chậm trên RTX 2060S.** ~0.5-1 fps cho 1080p → 4K. Chỉ khả thi cho clip ngắn, không cho phim dài.
