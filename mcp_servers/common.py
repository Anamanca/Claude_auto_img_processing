"""Shared utilities for all MCP servers."""
import subprocess
import json
import os
import sys
import time
import hashlib
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Optional, Any
from datetime import datetime, timedelta

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def resolve_path(cfg: dict, key: str) -> Path:
    """Resolve a path from config relative to the project root."""
    p = Path(cfg["paths"].get(key, key))
    if not p.is_absolute():
        p = Path(__file__).resolve().parent.parent / p
    return p


def setup_logging(cfg: dict, name: str) -> logging.Logger:
    log_cfg = cfg.get("logging", {})
    log_dir = resolve_path(cfg, "log_dir")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format=log_cfg.get("format", "%(asctime)s [%(levelname)s] %(message)s"),
        handlers=[
            logging.FileHandler(log_dir / f"{name}.log"),
            logging.StreamHandler(sys.stderr),
        ],
    )
    return logging.getLogger(name)


def run_cmd(cmd: list[str], timeout: int = 300, log: Optional[logging.Logger] = None) -> dict:
    """Run a subprocess and return {returncode, stdout, stderr, elapsed}."""
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        elapsed = time.perf_counter() - t0
        if log and proc.returncode != 0:
            log.error("Command failed: %s\nstderr: %s", " ".join(cmd), proc.stderr[-500:])
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "elapsed": round(elapsed, 3),
        }
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        if log:
            log.error("Command timed out after %ds: %s", timeout, " ".join(cmd))
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"Timeout after {timeout}s",
            "elapsed": round(elapsed, 3),
        }
    except FileNotFoundError:
        if log:
            log.error("Binary not found: %s", cmd[0])
        return {
            "returncode": -2,
            "stdout": "",
            "stderr": f"Binary not found: {cmd[0]}",
            "elapsed": 0,
        }


class TempManager:
    """Manage temporary files for a pipeline job."""

    def __init__(self, cfg: dict, job_id: Optional[str] = None):
        self.job_id = job_id or datetime.now().strftime("%Y%m%d_%H%M%S_") + hashlib.md5(
            os.urandom(8)
        ).hexdigest()[:8]
        temp_root = resolve_path(cfg, "temp_dir")
        self.job_dir = temp_root / self.job_id
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self._files: list[Path] = []

    def create(self, suffix: str = ".png", prefix: str = "tmp") -> Path:
        p = self.job_dir / f"{prefix}_{len(self._files):04d}{suffix}"
        self._files.append(p)
        return p

    def cleanup(self):
        if self.job_dir.exists():
            shutil.rmtree(self.job_dir, ignore_errors=True)

    def cleanup_old(self, max_age_hours: int = 1):
        """Remove temp dirs older than max_age_hours."""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        temp_root = self.job_dir.parent
        if not temp_root.exists():
            return
        for d in temp_root.iterdir():
            if d.is_dir():
                mtime = datetime.fromtimestamp(d.stat().st_mtime)
                if mtime < cutoff:
                    shutil.rmtree(d, ignore_errors=True)


def file_hash(path: str, algo: str = "sha256") -> str:
    """SHA256 hash of a file."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_gpu_available() -> dict:
    """Check GPU availability and VRAM."""
    info = {"cuda_available": False, "device_name": None, "vram_mb": None}
    try:
        import torch
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["device_name"] = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info(0)
            info["vram_total_mb"] = total // (1024 * 1024)
            info["vram_free_mb"] = free // (1024 * 1024)
    except ImportError:
        pass
    return info


def check_binary(binary_name: str) -> Optional[str]:
    """Return binary path if found, else None."""
    return shutil.which(binary_name)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def is_image_ext(path: str) -> bool:
    return Path(path).suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp",
        ".webp", ".avif", ".heic", ".heif", ".gif", ".dng",
        ".cr2", ".nef", ".arw", ".orf", ".rw2", ".pef", ".raf",
    }


def is_video_ext(path: str) -> bool:
    return Path(path).suffix.lower() in {
        ".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv",
        ".ts", ".mts", ".m2ts", ".wmv", ".m4v", ".3gp",
    }
