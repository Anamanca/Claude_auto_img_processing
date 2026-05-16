#!/bin/bash
# Hybrid Pipeline — Install Script
# Tested on: Ubuntu 22.04 / 24.04 with RTX 2060 Super + Ryzen 7
# Run: bash install.sh

set -e

BOLD="\e[1m"
GREEN="\e[32m"
YELLOW="\e[33m"
RED="\e[31m"
RESET="\e[0m"

echo -e "${BOLD}╔══════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║   Hybrid AI + CLI Pipeline — Installer       ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════╝${RESET}"
echo ""
echo "Target: ImageMagick + FFmpeg + PyTorch + ONNX Runtime"
echo ""

cd "$(dirname "$0")"

# ─── Step 1: System Dependencies ─────────────────────────
echo -e "${BOLD}[1/6] Installing system packages...${RESET}"

if command -v apt-get &> /dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        imagemagick \
        ffmpeg \
        exiftool \
        python3-pip \
        python3-venv \
        cuda-toolkit-12-1 2>/dev/null || true
elif command -v dnf &> /dev/null; then
    sudo dnf install -y \
        ImageMagick \
        ffmpeg \
        perl-Image-ExifTool \
        python3-pip
elif command -v pacman &> /dev/null; then
    sudo pacman -S --noconfirm \
        imagemagick \
        ffmpeg \
        perl-image-exiftool \
        python-pip
else
    echo -e "${YELLOW}⚠ Unknown package manager. Install manually:${RESET}"
    echo "  - ImageMagick (convert + identify)"
    echo "  - FFmpeg (ffmpeg + ffprobe)"
    echo "  - exiftool"
fi

# Verify
for bin in convert ffmpeg ffprobe exiftool; do
    if command -v "$bin" &> /dev/null; then
        echo -e "  ${GREEN}✓${RESET} $bin found at $(which $bin)"
    else
        echo -e "  ${RED}✗${RESET} $bin NOT FOUND — please install manually"
    fi
done

# ─── Step 2: Python Environment ──────────────────────────
echo ""
echo -e "${BOLD}[2/6] Setting up Python virtual environment...${RESET}"

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "  ${GREEN}✓${RESET} Created virtual environment"
else
    echo -e "  ${YELLOW}⊘${RESET} Virtual environment already exists"
fi

source venv/bin/activate
pip install --upgrade pip -q

# ─── Step 3: Python Dependencies ─────────────────────────
echo ""
echo -e "${BOLD}[3/6] Installing Python packages...${RESET}"

# Install PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q

# Install ONNX Runtime GPU
pip install onnxruntime-gpu -q 2>/dev/null || pip install onnxruntime -q

# Install other requirements
pip install -r requirements.txt -q

echo -e "  ${GREEN}✓${RESET} Python packages installed"

# ─── Step 4: GPU Verification ────────────────────────────
echo ""
echo -e "${BOLD}[4/6] Checking GPU...${RESET}"

python3 -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_mem // (1024**3)
    print(f'  ✓ CUDA available: {name} ({total}GB VRAM)')
    print(f'  ✓ NVENC should work (same GPU)')
else:
    print('  ⚠ CUDA NOT available — AI tools will run on CPU (slow)')
    print('  ⚠ NVENC may not work')
"

# ─── Step 5: Download AI Models ──────────────────────────
echo ""
echo -e "${BOLD}[5/6] Downloading AI models...${RESET}"

python3 download_models.py --check || {
    echo ""
    echo -e "${YELLOW}Some models are missing. Download them now?${RESET}"
    echo -e "Run: ${BOLD}python download_models.py${RESET}"
    echo -e "Or for just required models: ${BOLD}python download_models.py${RESET}"
    echo ""
    read -p "Download now? [Y/n] " -r
    REPLY=${REPLY:-Y}
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 download_models.py
    fi
}

# ─── Step 6: Create output directories ───────────────────
echo ""
echo -e "${BOLD}[6/6] Creating directories...${RESET}"

mkdir -p output logs temp luts profiles
echo -e "  ${GREEN}✓${RESET} Directories ready"

# ─── Summary ─────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║   Installation Complete!                      ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════╝${RESET}"
echo ""
echo -e "${BOLD}MCP Server commands:${RESET}"
echo ""
echo "  # ImageMagick"
echo "  python mcp_servers/im_server.py"
echo ""
echo "  # FFmpeg"
echo "  python mcp_servers/ffmpeg_server.py"
echo ""
echo "  # AI Models"
echo "  python mcp_servers/ai_server.py"
echo ""
echo "  # Utilities (RAW, EXIF)"
echo "  python mcp_servers/util_server.py"
echo ""
echo -e "${BOLD}Claude Code config (add to CLAUDE.md):${RESET}"
echo ""
echo '  {'
echo '    "mcpServers": {'
echo '      "im": {"command": "python", "args": ["mcp_servers/im_server.py"]},'
echo '      "ffmpeg": {"command": "python", "args": ["mcp_servers/ffmpeg_server.py"]},'
echo '      "ai": {"command": "python", "args": ["mcp_servers/ai_server.py"]},'
echo '      "util": {"command": "python", "args": ["mcp_servers/util_server.py"]}'
echo '    }'
echo '  }'
echo ""
echo -e "${YELLOW}Important: Make sure the venv is activated before running servers:${RESET}"
echo -e "  ${BOLD}source venv/bin/activate${RESET}"
echo ""
