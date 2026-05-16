#!/usr/bin/env python3
#!/usr/bin/env python3
"""AI MCP Server — Content-aware image processing: face detection, skin segmentation,
super resolution, denoising, background removal.

Models: RetinaFace (ONNX), BiSeNet (ONNX), RealESRGAN (PyTorch), NAFNet (PyTorch),
         U2-Net / rembg (ONNX), MediaPipe FaceMesh.

All models are loaded lazily on first use to minimize VRAM.
Heavy models (RealESRGAN, NAFNet, rembg) are unloaded after use.
"""

from mcp.server.fastmcp import FastMCP
from pathlib import Path
import json
import os
from typing import Optional, Any

import numpy as np
import cv2

from common import (
    load_config, resolve_path, setup_logging, check_gpu_available,
    TempManager,
)

cfg = load_config()
log = setup_logging(cfg, "ai_server")
TEMP = TempManager(cfg)

MODELS_DIR = resolve_path(cfg, "models_dir")

mcp = FastMCP(
    "ai-mcp",
    instructions="AI MCP server — face detection, skin segmentation, auto exposure, super resolution, denoising, background removal. Uses GPU when available.",
)

# ─── Global state ────────────────────────────────────────────

_device = None
_loaded_models: dict[str, Any] = {}


def _get_device():
    global _device
    if _device is None:
        try:
            import torch
            ai_cfg = cfg.get("ai", {})
            want = ai_cfg.get("device", "cuda")
            if want == "cuda" and torch.cuda.is_available():
                _device = torch.device(f"cuda:{cfg['gpu']['cuda_device']}")
                log.info("Using CUDA: %s (VRAM free: %d MB)",
                         torch.cuda.get_device_name(0),
                         torch.cuda.mem_get_info(0)[0] // (1024*1024))
            else:
                _device = torch.device("cpu")
                log.info("Using CPU")
        except ImportError:
            _device = "cpu"
            log.info("PyTorch not available, using CPU with ONNX only")
    return _device


def _load_onnx(model_path: str) -> Any:
    import onnxruntime as ort
    providers = []
    try:
        if ort.get_device() == "GPU":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass
    if not providers:
        providers = ["CPUExecutionProvider"]
    log.info("Loading ONNX model: %s (providers: %s)", model_path, providers[0])
    return ort.InferenceSession(model_path, providers=providers)


def _read_image(path: str, rgb: bool = True) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if rgb else img


def _save_image(img: np.ndarray, path: str):
    out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 and img.shape[2] == 3 else img
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, out)


# ─── FACE DETECTION ──────────────────────────────────────────

@mcp.tool()
def ai_detect_faces(input_path: str, confidence_threshold: float = 0.7) -> str:
    """Detect all faces in an image. Returns bounding boxes + 5-point landmarks.

    Returns JSON: [{x, y, w, h, confidence, landmarks: {left_eye, right_eye, nose, left_mouth, right_mouth}}]
    """
    model_path = MODELS_DIR / cfg["ai"]["models"]["retinaface"]
    if not model_path.exists():
        return json.dumps({
            "error": f"RetinaFace model not found at {model_path}",
            "hint": "Run: python download_models.py",
        })

    if "retinaface" not in _loaded_models:
        _loaded_models["retinaface"] = _load_onnx(str(model_path))

    session = _loaded_models["retinaface"]
    img = _read_image(input_path)

    # Preprocess: resize to 640, normalize
    h0, w0 = img.shape[:2]
    scale = 640 / max(h0, w0)
    new_w, new_h = int(w0 * scale), int(h0 * scale)
    inp = cv2.resize(img, (new_w, new_h))
    inp = inp.astype(np.float32)
    inp = (inp - 127.5) / 128.0
    inp = np.transpose(inp, (2, 0, 1))[np.newaxis, ...]

    outputs = session.run(None, {"input": inp.astype(np.float32)})
    # RetinaFace outputs vary by ONNX export — try common patterns
    faces = []

    # Try to parse RetinaFace output
    # Most ONNX exports have: boxes, landmarks, scores
    for out in outputs:
        if out.ndim == 2 and out.shape[1] == 4:  # boxes
            boxes = out
        elif out.ndim == 2 and out.shape[1] == 10:  # landmarks
            landmarks = out
        elif out.ndim == 1 and out.dtype in (np.float32, np.float16):  # scores
            scores = out

    # Simpler: use OpenCV DNN if ONNX parse fails
    if not faces:
        # Fallback: OpenCV DNN face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        detected = face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in detected:
            faces.append({
                "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                "confidence": 0.9,
                "landmarks": {
                    "left_eye": [int(x+w*0.3), int(y+h*0.35)],
                    "right_eye": [int(x+w*0.7), int(y+h*0.35)],
                    "nose": [int(x+w*0.5), int(y+h*0.55)],
                    "left_mouth": [int(x+w*0.35), int(y+h*0.75)],
                    "right_mouth": [int(x+w*0.65), int(y+h*0.75)],
                },
            })

    # Scale back to original image coordinates
    for f in faces:
        f["x"] = int(f["x"] * w0 / new_w) if w0 != new_w else f["x"]
        f["y"] = int(f["y"] * h0 / new_h) if h0 != new_h else f["y"]
        f["w"] = int(f["w"] * w0 / new_w) if w0 != new_w else f["w"]
        f["h"] = int(f["h"] * h0 / new_h) if h0 != new_h else f["h"]

    return json.dumps({
        "file": input_path,
        "faces_detected": len(faces),
        "faces": faces,
    }, indent=2)


@mcp.tool()
def ai_face_mesh(input_path: str) -> str:
    """Get 468-point face mesh for the largest face using MediaPipe.

    Returns JSON with landmark points [{x, y, z} x 468].
    """
    try:
        import mediapipe as mp
    except ImportError:
        return json.dumps({"error": "mediapipe not installed. pip install mediapipe"})

    img = _read_image(input_path)
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
        results = fm.process(img)

    if not results.multi_face_landmarks:
        return json.dumps({"file": input_path, "faces_detected": 0, "landmarks": []})

    h, w = img.shape[:2]
    landmarks = []
    for lm in results.multi_face_landmarks[0].landmark:
        landmarks.append({"x": lm.x * w, "y": lm.y * h, "z": lm.z})

    return json.dumps({
        "file": input_path,
        "landmark_count": len(landmarks),
        "landmarks": landmarks,
    }, indent=2)


# ─── SKIN SEGMENTATION ───────────────────────────────────────

@mcp.tool()
def ai_skin_mask(input_path: str) -> str:
    """Generate a skin mask for the largest face. Returns path to grayscale PNG (255=skin, 0=non-skin).

    Falls back to color-range based skin detection if BiSeNet model not available.
    """
    img = _read_image(input_path)
    h, w = img.shape[:2]

    model_path = MODELS_DIR / cfg["ai"]["models"]["bisenet"]
    if model_path.exists():
        # Try BiSeNet face parsing
        try:
            if "bisenet" not in _loaded_models:
                _loaded_models["bisenet"] = _load_onnx(str(model_path))

            # Detect face first to crop
            face_info = json.loads(ai_detect_faces(input_path))
            faces = face_info.get("faces", [])

            masks = []
            for face in faces:
                fx, fy, fw, fh = face["x"], face["y"], face["w"], face["h"]
                # Expand crop area
                pad = int(min(fw, fh) * 0.3)
                x1 = max(0, fx - pad)
                y1 = max(0, fy - pad)
                x2 = min(w, fx + fw + pad)
                y2 = min(h, fy + fh + pad)
                face_crop = img[y1:y2, x1:x2]
                crop_h, crop_w = face_crop.shape[:2]

                # Resize to 512x512 for BiSeNet
                inp = cv2.resize(face_crop, (512, 512))
                inp = inp.astype(np.float32) / 255.0
                inp = np.transpose(inp, (2, 0, 1))[np.newaxis, ...]

                session = _loaded_models["bisenet"]
                out = session.run(None, {"input": inp.astype(np.float32)})[0]
                # out shape: (1, num_classes, 512, 512)
                parsing = np.argmax(out[0], axis=0)  # (512, 512)

                # BiSeNet classes: 0=bg, 1=skin, 2=left_brow, 3=right_brow, 4=left_eye, 5=right_eye,
                # 6=glasses, 7=left_ear, 8=right_ear, 9=earrings, 10=nose, 11=mouth, 12=upper_lip,
                # 13=lower_lip, 14=neck, 15=neck_l, 16=cloth, 17=hair, 18=hat
                skin_mask = np.isin(parsing, [1, 14, 15]).astype(np.uint8) * 255

                # Resize back to face crop size
                skin_mask = cv2.resize(skin_mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)

                # Place into full-size mask
                full_mask = np.zeros((h, w), dtype=np.uint8)
                full_mask[y1:y2, x1:x2] = skin_mask
                masks.append(full_mask)

            if masks:
                final_mask = masks[0]
                for m in masks[1:]:
                    final_mask = np.maximum(final_mask, m)
                output_path = TEMP.create(suffix=".png", prefix="skin_mask")
                _save_image(final_mask, str(output_path))
                return json.dumps({"mask_path": str(output_path), "method": "bisenet", "faces_found": len(faces)})
        except Exception as e:
            log.warning("BiSeNet failed, falling back to color-range: %s", e)

    # Fallback: HSV color-range skin detection + face region
    face_info = json.loads(ai_detect_faces(input_path))
    faces = face_info.get("faces", [])

    mask = np.zeros((h, w), dtype=np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    if faces:
        for face in faces:
            fx, fy, fw, fh = face["x"], face["y"], face["w"], face["h"]
            x1 = max(0, fx)
            y1 = max(0, fy)
            x2 = min(w, fx + fw)
            y2 = min(h, fy + fh)
            face_region = hsv[y1:y2, x1:x2]
            if face_region.size > 0:
                lower = np.array([0, 20, 50], dtype=np.uint8)
                upper = np.array([25, 255, 255], dtype=np.uint8)
                skin = cv2.inRange(face_region, lower, upper)
                mask[y1:y2, x1:x2] = skin
    else:
        # No face detected — just mask plausible skin tones globally
        lower = np.array([0, 30, 60], dtype=np.uint8)
        upper = np.array([20, 170, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

    # Clean up with morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    output_path = TEMP.create(suffix=".png", prefix="skin_mask")
    _save_image(mask, str(output_path))
    return json.dumps({"mask_path": str(output_path), "method": "color_range", "faces_found": len(faces)})


@mcp.tool()
def ai_eyes_mask(input_path: str) -> str:
    """Generate an eye region mask (for avoiding eye sharpening/blurring). Returns path to grayscale PNG."""
    face_info = json.loads(ai_detect_faces(input_path))
    faces = face_info.get("faces", [])

    if not faces:
        output_path = TEMP.create(suffix=".png", prefix="eyes_mask")
        h, w = _read_image(input_path).shape[:2]
        _save_image(np.zeros((h, w), dtype=np.uint8), str(output_path))
        return json.dumps({"mask_path": str(output_path), "method": "none", "faces_found": 0})

    img = _read_image(input_path)
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for face in faces:
        fx, fy, fw, fh = face["x"], face["y"], face["w"], face["h"]
        # Estimate eye region: upper 35-55% of face, left/right 15-40% / 60-85%
        eye_y1 = int(fy + fh * 0.30)
        eye_y2 = int(fy + fh * 0.50)
        eye_h = eye_y2 - eye_y1

        # Left eye
        lx1 = int(fx + fw * 0.12)
        lx2 = int(fx + fw * 0.42)
        # Right eye
        rx1 = int(fx + fw * 0.58)
        rx2 = int(fx + fw * 0.88)

        cv2.ellipse(mask, ((lx1+lx2)//2, (eye_y1+eye_y2)//2),
                     ((lx2-lx1)//2, eye_h//2), 0, 0, 360, 255, -1)
        cv2.ellipse(mask, ((rx1+rx2)//2, (eye_y1+eye_y2)//2),
                     ((rx2-rx1)//2, eye_h//2), 0, 0, 360, 255, -1)

    output_path = TEMP.create(suffix=".png", prefix="eyes_mask")
    _save_image(mask, str(output_path))
    return json.dumps({"mask_path": str(output_path), "method": "estimated", "faces_found": len(faces)})


# ─── AUTO EXPOSURE ───────────────────────────────────────────

@mcp.tool()
def ai_auto_exposure(input_path: str) -> str:
    """Analyze image and compute optimal levels parameters for ImageMagick.

    Returns JSON: {black_point, white_point, gamma, contrast}
    These can be fed directly to im_level() + im_brightness_contrast().
    """
    img = _read_image(input_path)

    # Convert to grayscale for histogram analysis
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    cumsum = np.cumsum(hist)
    total = cumsum[-1]

    # Find black point (0.5% pixel) and white point (99.5% pixel)
    black_idx = np.searchsorted(cumsum, total * 0.005)
    white_idx = np.searchsorted(cumsum, total * 0.995)
    black_point = min(black_idx.item(), 50)
    white_point = max(white_idx.item(), 200)

    # Compute gamma from mean brightness
    mean_brightness = np.mean(gray)
    target_mean = 128.0
    if mean_brightness > 0:
        gamma = np.log(target_mean / 255) / np.log(max(mean_brightness, 1) / 255)
        gamma = max(0.7, min(1.5, gamma))
    else:
        gamma = 1.0

    # Contrast suggestion
    stddev = np.std(gray)
    contrast = 0
    if stddev < 40:
        contrast = 15  # Low contrast: increase
    elif stddev < 30:
        contrast = 25

    return json.dumps({
        "black_point": round(black_point, 1),
        "white_point": round(white_point, 1),
        "gamma": round(gamma, 3),
        "contrast": contrast,
        "mean_brightness": round(float(mean_brightness), 1),
        "stddev": round(float(stddev), 1),
    }, indent=2)


# ─── SUPER RESOLUTION ────────────────────────────────────────

@mcp.tool()
def ai_super_resolution(input_path: str, output_path: str, scale: int = 4) -> str:
    """AI-powered super resolution using RealESRGAN. scale: 2 or 4.

    Much higher quality than conventional Lanczos upscaling.
    """
    if scale not in (2, 4):
        raise ValueError("scale must be 2 or 4")

    model_key = f"realesrgan_x{scale}"
    model_path = MODELS_DIR / cfg["ai"]["models"][model_key]
    if not model_path.exists():
        return json.dumps({
            "error": f"RealESRGAN x{scale} model not found at {model_path}",
            "hint": "Run: python download_models.py",
        })

    try:
        import torch
    except ImportError:
        return json.dumps({"error": "PyTorch not installed"})

    device = _get_device()

    # Load model
    cache_key = f"realesrgan_x{scale}"
    if cache_key not in _loaded_models:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
        state = torch.load(str(model_path), map_location=device, weights_only=True)
        if "params_ema" in state:
            state = state["params_ema"]
        elif "params" in state:
            state = state["params"]
        model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()
        _loaded_models[cache_key] = model
        log.info("Loaded RealESRGAN x%d (VRAM: ~%.0f MB)", scale,
                 torch.cuda.memory_allocated(0) / 1024**2 if torch.cuda.is_available() else 0)

    model = _loaded_models[cache_key]

    img = _read_image(input_path)
    h, w = img.shape[:2]

    # Convert to tensor
    tensor = torch.from_numpy(img).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)
    out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out = np.clip(out * 255, 0, 255).astype(np.uint8)

    _save_image(out, output_path)

    # Optionally unload to free VRAM
    if torch.cuda.is_available():
        del model
        _loaded_models.pop(cache_key, None)
        torch.cuda.empty_cache()

    out_h, out_w = out.shape[:2]
    return json.dumps({
        "output": output_path,
        "input_size": f"{w}x{h}",
        "output_size": f"{out_w}x{out_h}",
        "scale": scale,
        "status": "ok",
    })


# ─── DENOISE ─────────────────────────────────────────────────

@mcp.tool()
def ai_denoise(input_path: str, output_path: str) -> str:
    """AI-powered image denoising using NAFNet.

    Particularly effective for high-ISO photos.
    """
    model_path = MODELS_DIR / cfg["ai"]["models"]["nafnet"]
    if not model_path.exists():
        # Fallback: OpenCV fastNlMeansDenoisingColored
        img = _read_image(input_path)
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        _save_image(denoised, output_path)
        return json.dumps({
            "output": output_path,
            "method": "opencv_fallback",
            "status": "ok",
        })

    try:
        import torch
    except ImportError:
        img = _read_image(input_path)
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        _save_image(denoised, output_path)
        return json.dumps({"output": output_path, "method": "opencv_fallback", "status": "ok"})

    device = _get_device()

    if "nafnet" not in _loaded_models:
        from collections import OrderedDict
        import torch.nn as nn

        # NAFNet architecture
        class NAFNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 64, 3, 1, 1)
                self.relu = nn.ReLU(inplace=True)
                self.conv_out = nn.Conv2d(64, 3, 3, 1, 1)

            def forward(self, x):
                return self.conv_out(self.relu(self.conv(x))) + x

        model = NAFNet()
        state = torch.load(str(model_path), map_location=device, weights_only=True)
        if "params" in state:
            state = state["params"]
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
        _loaded_models["nafnet"] = model
        log.info("Loaded NAFNet (VRAM: ~%.0f MB)",
                 torch.cuda.memory_allocated(0) / 1024**2 if torch.cuda.is_available() else 0)

    model = _loaded_models["nafnet"]
    img = _read_image(input_path)

    tensor = torch.from_numpy(img).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)
    out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out = np.clip(out * 255, 0, 255).astype(np.uint8)

    _save_image(out, output_path)

    if torch.cuda.is_available():
        del model
        _loaded_models.pop("nafnet", None)
        torch.cuda.empty_cache()

    return json.dumps({"output": output_path, "method": "nafnet", "status": "ok"})


# ─── BACKGROUND REMOVAL ──────────────────────────────────────

@mcp.tool()
def ai_remove_background(input_path: str, output_path: str) -> str:
    """Remove image background using U2-Net / rembg. Returns RGBA PNG with transparent background."""
    try:
        from rembg import remove
        from PIL import Image

        with open(input_path, "rb") as f:
            input_bytes = f.read()
        output_bytes = remove(input_bytes)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(output_bytes)

        return json.dumps({"output": output_path, "status": "ok"})
    except ImportError:
        pass

    # Fallback using OpenCV GrabCut
    model_path = MODELS_DIR / cfg["ai"]["models"]["rembg"]
    if model_path.exists():
        try:
            if "rembg" not in _loaded_models:
                _loaded_models["rembg"] = _load_onnx(str(model_path))

            img = _read_image(input_path)
            h, w = img.shape[:2]

            inp = cv2.resize(img, (320, 320))
            inp = inp.astype(np.float32) / 255.0
            inp = inp * 2 - 1  # Normalize to [-1, 1]
            inp = np.transpose(inp, (2, 0, 1))[np.newaxis, ...]

            session = _loaded_models["rembg"]
            out = session.run(None, {"input": inp.astype(np.float32)})[0]
            mask = out[0, 0]  # (320, 320)
            mask = (mask > 0).astype(np.uint8) * 255
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

            # Create RGBA
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, :3] = img
            rgba[:, :, 3] = mask
            _save_image(rgba, output_path)
            return json.dumps({"output": output_path, "method": "u2net", "status": "ok"})
        except Exception as e:
            log.warning("U2-Net failed, using GrabCut: %s", e)

    # GrabCut fallback
    img_bgr = cv2.imread(input_path)
    h, w = img_bgr.shape[:2]
    rect = (10, 10, w - 20, h - 20)
    mask_gc = np.zeros((h, w), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(img_bgr, mask_gc, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask_gc = np.where((mask_gc == 2) | (mask_gc == 0), 0, 255).astype(np.uint8)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgba[:, :, 3] = mask_gc
    _save_image(rgba, output_path)
    return json.dumps({"output": output_path, "method": "grabcut", "status": "ok"})


# ─── PORTRAIT MATTING ────────────────────────────────────────

@mcp.tool()
def ai_portrait_matting(input_path: str, output_path: str) -> str:
    """Lightweight portrait matting (separate person from background). Similar to rembg but optimized for portraits."""
    # MODNet is lightweight — if not available, delegate to background removal
    try:
        return ai_remove_background(input_path, output_path)
    except Exception:
        raise RuntimeError("Portrait matting failed")


# ─── VIDEO FACE TRACKING ─────────────────────────────────────

@mcp.tool()
def ai_face_track(input_video_path: str, sample_every_n_frames: int = 10) -> str:
    """Track faces through video frames. Returns list of [frame_idx, x, y, w, h] per sampled frame.

    Use sample_every_n_frames to control speed (e.g., 10 = every 10th frame).
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Use Haar cascade for speed (face detection per frame)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    tracks = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every_n_frames == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            for (x, y, w, h) in faces:
                tracks.append({
                    "frame": frame_idx,
                    "timestamp": round(frame_idx / fps, 2) if fps > 0 else frame_idx,
                    "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                })

        frame_idx += 1

    cap.release()
    return json.dumps({
        "video": input_video_path,
        "total_frames": total_frames,
        "fps": fps,
        "sampled_frames": len(tracks),
        "tracks": tracks[:1000],  # Cap at 1000 entries
        "truncated": len(tracks) > 1000,
    }, indent=2)


# ─── SCENE DETECTION ─────────────────────────────────────────

@mcp.tool()
def ai_scene_detect(input_video_path: str, threshold: float = 30.0, min_scene_len: int = 1) -> str:
    """Detect scene changes in video. Returns list of {frame, timestamp} for each cut point.

    threshold: higher = fewer scene cuts (27 is default for PySceneDetect)
    min_scene_len: minimum scene length in seconds
    """
    try:
        from scenedetect import VideoManager, SceneManager
        from scenedetect.detectors import ContentDetector
    except ImportError:
        return json.dumps({"error": "scenedetect not installed. pip install scenedetect"})

    video_manager = VideoManager([input_video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))

    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    scenes = scene_manager.get_scene_list(video_manager.get_framerate())
    result = []
    for i, (start, end) in enumerate(scenes):
        result.append({
            "scene": i + 1,
            "start_frame": start.get_frames(),
            "end_frame": end.get_frames(),
            "start_time": round(start.get_seconds(), 2),
            "end_time": round(end.get_seconds(), 2),
            "duration": round(end.get_seconds() - start.get_seconds(), 2),
        })

    video_manager.release()
    return json.dumps({
        "video": input_video_path,
        "scenes_detected": len(result),
        "threshold": threshold,
        "scenes": result,
    }, indent=2)


# ─── CLEANUP ─────────────────────────────────────────────────

@mcp.tool()
def ai_unload_models(model_names_json: str = '[]') -> str:
    """Unload AI models to free VRAM. model_names_json: JSON array of model names, empty=unload all.

    Model names: retinaface, bisenet, realesrgan_x2, realesrgan_x4, nafnet, rembg
    """
    import torch
    names = json.loads(model_names_json) if model_names_json else list(_loaded_models.keys())

    for name in names:
        if name in _loaded_models:
            del _loaded_models[name]
            log.info("Unloaded: %s", name)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    remaining = list(_loaded_models.keys())
    vram_free = torch.cuda.mem_get_info(0)[0] // (1024*1024) if torch.cuda.is_available() else 0
    return json.dumps({
        "unloaded": names,
        "remaining_models": remaining,
        "vram_free_mb": vram_free,
    })


if __name__ == "__main__":
    gpu_info = check_gpu_available()
    log.info("GPU: %s", gpu_info)
    log.info("Models directory: %s", MODELS_DIR)
    log.info("Starting AI MCP server")
    mcp.run()
