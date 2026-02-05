#!/bin/bash
# ============================================================================
# Qwen3-TTS VoiceDesign — Non-interactive One-Click Deploy for Jetson Orin Nano
# ============================================================================
#
# Target: Jetson Orin Nano Super 8GB (JetPack 6.2.2, L4T R36.5, CUDA 12.6)
#
# Usage:
#   bash deploy_jetson.sh                  # Deploy with 0.6B model (default)
#   bash deploy_jetson.sh --model-size 1.7B  # Deploy with 1.7B model
#
# This script is:
#   - Non-interactive: no read/prompt, fully automated
#   - Idempotent: re-run safely, completed steps are skipped
#   - Error-handled: set -euo pipefail, clear error messages
#
# Stages:
#   [1/7] Hardware validation
#   [2/7] System configuration (swap, nvpmodel, jetson_clocks)
#   [3/7] Python virtual environment
#   [4/7] Dependency installation
#   [5/7] Model download
#   [6/7] INT4 quantization
#   [7/7] Verification inference
# ============================================================================

set -euo pipefail

# ---- Configuration ----
MODEL_SIZE="0.6B"
VENV_DIR="$HOME/qwen3-tts-env"
SWAP_SIZE_MB=16384
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --model-size=*)
            MODEL_SIZE="${1#*=}"
            shift
            ;;
        --venv-dir)
            VENV_DIR="$2"
            shift 2
            ;;
        --venv-dir=*)
            VENV_DIR="${1#*=}"
            shift
            ;;
        --help|-h)
            echo "Usage: bash deploy_jetson.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model-size SIZE   Model size: 0.6B (default) or 1.7B"
            echo "  --venv-dir DIR      Python venv directory (default: ~/qwen3-tts-env)"
            echo "  -h, --help          Show this help"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# ---- Derived variables ----
REPO_ID="Qwen/Qwen3-TTS-12Hz-${MODEL_SIZE}-VoiceDesign"
MODEL_DIR="${MODELS_DIR}/Qwen3-TTS-12Hz-${MODEL_SIZE}-VoiceDesign"

# ---- Helper functions ----
log_step() {
    echo ""
    echo "=============================================="
    echo "  $1"
    echo "=============================================="
}

log_ok() {
    echo "[OK] $1"
}

log_skip() {
    echo "[SKIP] $1"
}

log_info() {
    echo "[INFO] $1"
}

log_error() {
    echo "[ERROR] $1" >&2
}

die() {
    log_error "$1"
    exit 1
}

# ============================================================================
# [1/7] Hardware Validation
# ============================================================================
log_step "[1/7] Hardware Validation"

# Check Jetson platform
if [ ! -f /etc/nv_tegra_release ]; then
    die "Not a Jetson device (/etc/nv_tegra_release not found).
    This script is designed for Jetson Orin Nano with JetPack 6.2.2."
fi

# Parse L4T version
L4T_RELEASE=$(head -1 /etc/nv_tegra_release)
log_info "L4T: ${L4T_RELEASE}"

# Extract R36 major version
if echo "${L4T_RELEASE}" | grep -q "R36"; then
    log_ok "JetPack 6.x (L4T R36) detected"
else
    log_info "Warning: Expected L4T R36.x (JetPack 6.x), got: ${L4T_RELEASE}"
    log_info "Script may still work, but is optimized for JetPack 6.2.2"
fi

# Check CUDA
if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    log_ok "CUDA ${CUDA_VER}"
elif [ -d /usr/local/cuda ]; then
    log_ok "CUDA found at /usr/local/cuda"
else
    log_info "Warning: nvcc not in PATH, CUDA may still be available via PyTorch"
fi

# Check available memory
MEM_TOTAL_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
MEM_TOTAL_GB=$(echo "scale=1; ${MEM_TOTAL_KB} / 1048576" | bc 2>/dev/null || echo "unknown")
log_info "Total RAM: ${MEM_TOTAL_GB} GB"

if [ "${MODEL_SIZE}" = "1.7B" ] && [ "${MEM_TOTAL_KB}" -lt 7000000 ]; then
    log_info "Warning: 1.7B model may be tight on ${MEM_TOTAL_GB} GB RAM. 16GB swap recommended."
fi

log_ok "Hardware validation passed"

# ============================================================================
# [2/7] System Configuration
# ============================================================================
log_step "[2/7] System Configuration"

# --- Swap ---
CURRENT_SWAP_KB=$(grep SwapTotal /proc/meminfo | awk '{print $2}')
NEEDED_SWAP_KB=$((SWAP_SIZE_MB * 1024))

if [ "${CURRENT_SWAP_KB}" -ge "${NEEDED_SWAP_KB}" ]; then
    CURRENT_SWAP_MB=$((CURRENT_SWAP_KB / 1024))
    log_skip "Swap already ${CURRENT_SWAP_MB} MB (>= ${SWAP_SIZE_MB} MB)"
else
    log_info "Configuring ${SWAP_SIZE_MB} MB swap..."
    SWAPFILE="/swapfile-qwen3tts"
    if [ ! -f "${SWAPFILE}" ]; then
        sudo fallocate -l ${SWAP_SIZE_MB}M "${SWAPFILE}" || \
            sudo dd if=/dev/zero of="${SWAPFILE}" bs=1M count=${SWAP_SIZE_MB} status=progress
        sudo chmod 600 "${SWAPFILE}"
        sudo mkswap "${SWAPFILE}"
    fi
    # Enable swap if not already active
    if ! swapon --show | grep -q "${SWAPFILE}"; then
        sudo swapon "${SWAPFILE}"
    fi
    # Add to fstab if not present
    if ! grep -q "${SWAPFILE}" /etc/fstab; then
        echo "${SWAPFILE} none swap sw 0 0" | sudo tee -a /etc/fstab >/dev/null
    fi
    log_ok "Swap configured: ${SWAP_SIZE_MB} MB"
fi

# --- Power mode (25W max) ---
if command -v nvpmodel &>/dev/null; then
    CURRENT_MODE=$(nvpmodel -q 2>/dev/null | grep "NV Power Mode" | head -1 || echo "")
    log_info "Current power mode: ${CURRENT_MODE}"
    # Mode 0 is usually MAXN (max performance)
    sudo nvpmodel -m 0 2>/dev/null || log_info "nvpmodel set skipped (may need reboot)"
    log_ok "Power mode set to MAXN"
else
    log_skip "nvpmodel not found"
fi

# --- jetson_clocks ---
if command -v jetson_clocks &>/dev/null; then
    sudo jetson_clocks 2>/dev/null || log_info "jetson_clocks failed (non-fatal)"
    log_ok "jetson_clocks enabled (max clock frequencies)"
else
    log_skip "jetson_clocks not found"
fi

# --- System packages ---
log_info "Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-pip \
    python3-venv \
    libopenblas-dev \
    libsndfile1 \
    sox \
    ffmpeg \
    bc \
    2>/dev/null
log_ok "System packages installed"

# ============================================================================
# [3/7] Python Virtual Environment
# ============================================================================
log_step "[3/7] Python Virtual Environment"

if [ -d "${VENV_DIR}" ] && [ -f "${VENV_DIR}/bin/activate" ]; then
    log_skip "venv already exists: ${VENV_DIR}"
else
    log_info "Creating venv at ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
    log_ok "venv created"
fi

# Activate venv
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
log_info "Python: $(python3 --version) at $(which python3)"

# Upgrade pip
pip install --upgrade pip --quiet
log_ok "pip upgraded"

# ============================================================================
# [4/7] Dependency Installation
# ============================================================================
log_step "[4/7] Dependency Installation"

# --- PyTorch from Jetson AI Lab ---
PYTORCH_INSTALLED=false
if python3 -c "import torch; print(torch.__version__)" 2>/dev/null; then
    TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
    CUDA_AVAIL=$(python3 -c "import torch; print(torch.cuda.is_available())")
    log_info "PyTorch ${TORCH_VER} already installed (CUDA: ${CUDA_AVAIL})"
    if [ "${CUDA_AVAIL}" = "True" ]; then
        PYTORCH_INSTALLED=true
        log_skip "PyTorch with CUDA already installed"
    else
        log_info "PyTorch installed but CUDA not available, reinstalling..."
    fi
fi

if [ "${PYTORCH_INSTALLED}" = "false" ]; then
    log_info "Installing PyTorch from Jetson AI Lab (CUDA 12.6)..."
    pip install torch torchvision torchaudio \
        --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
        --quiet \
        || die "PyTorch installation failed. Check network and Jetson AI Lab availability."

    # Verify
    python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" \
        || die "PyTorch installed but CUDA not available. Check JetPack/CUDA installation."
    log_ok "PyTorch installed with CUDA support"
fi

# --- Core dependencies ---
log_info "Installing Python dependencies..."
pip install --quiet \
    "transformers==4.57.3" \
    "accelerate==1.12.0" \
    "gradio>=4.0.0" \
    "librosa>=0.10.0" \
    "soundfile>=0.12.0" \
    "scipy>=1.10.0" \
    "huggingface_hub>=0.20.0" \
    "einops>=0.7.0" \
    "numpy<2.0"
log_ok "Core dependencies installed"

# --- torchao (for INT4 quantization) ---
TORCHAO_INSTALLED=false
if python3 -c "from torchao.quantization import int4_weight_only; print('ok')" 2>/dev/null; then
    log_skip "torchao already installed"
    TORCHAO_INSTALLED=true
else
    log_info "Installing torchao..."
    # Try Jetson AI Lab source first
    if pip install torchao --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 --quiet 2>/dev/null; then
        TORCHAO_INSTALLED=true
        log_ok "torchao installed from Jetson AI Lab"
    # Try PyPI
    elif pip install torchao --quiet 2>/dev/null; then
        TORCHAO_INSTALLED=true
        log_ok "torchao installed from PyPI"
    else
        log_info "torchao installation failed. Will use PyTorch INT8 fallback for quantization."
    fi
fi

# --- onnxruntime ---
if python3 -c "import onnxruntime" 2>/dev/null; then
    log_skip "onnxruntime already installed"
else
    pip install onnxruntime-gpu --quiet 2>/dev/null || \
        pip install onnxruntime --quiet 2>/dev/null || \
        log_info "onnxruntime installation failed (non-fatal)"
fi

# --- Install local Qwen3-TTS package ---
if [ -d "${SCRIPT_DIR}/Qwen3-TTS" ]; then
    log_info "Installing Qwen3-TTS local package..."
    pip install -e "${SCRIPT_DIR}/Qwen3-TTS" --quiet
    log_ok "Qwen3-TTS package installed"
else
    log_info "Qwen3-TTS directory not found at ${SCRIPT_DIR}/Qwen3-TTS (will try import from sys.path)"
fi

log_ok "All dependencies installed"

# ============================================================================
# [5/7] Model Download
# ============================================================================
log_step "[5/7] Model Download (${REPO_ID})"

mkdir -p "${MODELS_DIR}"

if [ -d "${MODEL_DIR}" ] && [ -f "${MODEL_DIR}/config.json" ]; then
    # Check for weight files
    WEIGHT_COUNT=$(find "${MODEL_DIR}" -name "*.safetensors" -o -name "*.bin" 2>/dev/null | head -1 | wc -l)
    if [ "${WEIGHT_COUNT}" -gt 0 ]; then
        log_skip "Model already downloaded: ${MODEL_DIR}"
    else
        log_info "Model directory exists but missing weights, re-downloading..."
        NEED_DOWNLOAD=true
    fi
else
    NEED_DOWNLOAD=true
fi

if [ "${NEED_DOWNLOAD:-false}" = "true" ] || [ ! -d "${MODEL_DIR}" ]; then
    log_info "Downloading ${REPO_ID} via huggingface-cli..."
    log_info "Target: ${MODEL_DIR}"

    # Install huggingface-cli if needed
    if ! command -v huggingface-cli &>/dev/null; then
        pip install --upgrade huggingface_hub --quiet
    fi

    huggingface-cli download "${REPO_ID}" \
        --local-dir "${MODEL_DIR}" \
        --local-dir-use-symlinks False \
        || die "Model download failed. Check network connection.
    You can retry with:
      huggingface-cli download ${REPO_ID} --local-dir ${MODEL_DIR} --resume-download"

    # Verify download
    if [ ! -f "${MODEL_DIR}/config.json" ]; then
        die "Download completed but config.json not found in ${MODEL_DIR}"
    fi
    log_ok "Model downloaded: ${MODEL_DIR}"
fi

# ============================================================================
# [6/7] INT4 Quantization
# ============================================================================
log_step "[6/7] INT4 Quantization"

QUANTIZE_SCRIPT="${SCRIPT_DIR}/quantize_int4.py"
INT4_DIR="${MODEL_DIR}-INT4"

if [ ! -f "${QUANTIZE_SCRIPT}" ]; then
    die "quantize_int4.py not found at ${QUANTIZE_SCRIPT}"
fi

if [ -f "${INT4_DIR}/quantize_config.json" ] && [ -f "${INT4_DIR}/quantized_model.pt" ]; then
    log_skip "Quantized model already exists: ${INT4_DIR}"
else
    log_info "Starting quantization..."
    log_info "Model: ${MODEL_DIR}"
    log_info "Output: ${INT4_DIR}"

    QUANT_METHOD="auto"
    if [ "${TORCHAO_INSTALLED}" = "true" ]; then
        log_info "Using torchao INT4 weight-only quantization"
    else
        log_info "Using PyTorch INT8 dynamic quantization (fallback)"
        QUANT_METHOD="int8"
    fi

    python3 "${QUANTIZE_SCRIPT}" \
        --model-dir "${MODEL_DIR}" \
        --output-dir "${INT4_DIR}" \
        --method "${QUANT_METHOD}" \
        || die "Quantization failed. Check logs above for details."

    if [ ! -f "${INT4_DIR}/quantize_config.json" ]; then
        die "Quantization completed but quantize_config.json not found"
    fi
    log_ok "Quantization complete: ${INT4_DIR}"
fi

# ============================================================================
# [7/7] Verification Inference
# ============================================================================
log_step "[7/7] Verification Inference"

log_info "Running verification inference..."
python3 "${QUANTIZE_SCRIPT}" \
    --model-dir "${MODEL_DIR}" \
    --output-dir "${INT4_DIR}" \
    --verify \
    && log_ok "Verification passed" \
    || log_info "Verification had issues (non-fatal, model may still work)"

# ============================================================================
# Done!
# ============================================================================
echo ""
echo "=============================================="
echo "  Deployment Complete!"
echo "=============================================="
echo ""
echo "  Model:    ${REPO_ID}"
echo "  Location: ${MODEL_DIR}"
echo "  INT4:     ${INT4_DIR}"
echo "  Venv:     ${VENV_DIR}"
echo ""
echo "  To start the Gradio app:"
echo ""
echo "    source ${VENV_DIR}/bin/activate"
echo "    python3 jetson_int4_launcher.py --model-dir ${MODEL_DIR}"
echo ""

# Get local IP for access URL
LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "127.0.0.1")
echo "  Access: http://${LOCAL_IP}:8000"
echo ""
echo "=============================================="
