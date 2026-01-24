#!/bin/bash
# Qwen3-TTS Installation Script for NVIDIA Jetson Orin
# Tested on: JetPack 6.x (Python 3.10)
#
# Usage: bash install_jetson.sh

set -e

echo "=============================================="
echo "Qwen3-TTS Jetson Installation Script"
echo "=============================================="

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "Warning: This script is designed for NVIDIA Jetson devices."
    echo "Detected system may not be a Jetson. Continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Detect JetPack version
echo ""
echo "[1/6] Detecting JetPack version..."
if [ -f /etc/nv_tegra_release ]; then
    cat /etc/nv_tegra_release
fi

# Install system dependencies
echo ""
echo "[2/6] Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    libopenblas-dev \
    libsndfile1 \
    sox \
    ffmpeg

# Upgrade pip
echo ""
echo "[3/6] Upgrading pip..."
python3 -m pip install --upgrade pip

# Install numpy first (pinned version)
echo ""
echo "[4/6] Installing numpy..."
python3 -m pip install numpy==1.26.1

# PyTorch and torchaudio installation
echo ""
echo "[5/6] Installing PyTorch and torchaudio..."
echo ""
echo "Please select your JetPack version:"
echo "  1) JetPack 6.x (L4T R36.x) - Python 3.10 [Recommended: Jetson AI Lab PyPI]"
echo "  2) JetPack 6.0 (L4T R36.x) - Python 3.10 [NVIDIA wheel]"
echo "  3) JetPack 5.1.x (L4T R35.x) - Python 3.8 [NVIDIA wheel]"
echo "  4) Skip (already installed)"
echo ""
read -p "Enter choice [1-4]: " jp_choice

case $jp_choice in
    1)
        echo "Installing PyTorch + torchaudio from Jetson AI Lab PyPI..."
        # JetPack 6.x with CUDA 12.6 - recommended method
        python3 -m pip install torch torchvision torchaudio \
            --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
        ;;
    2)
        echo "Installing PyTorch for JetPack 6.0 from NVIDIA..."
        # JetPack 6.0 / L4T R36.x / Python 3.10
        TORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+07cecf4168.nv24.05.14710581-cp310-cp310-linux_aarch64.whl"
        python3 -m pip install --no-cache "$TORCH_URL"

        echo ""
        echo "Attempting to install torchaudio from Jetson AI Lab..."
        python3 -m pip install torchaudio --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 || {
            echo ""
            echo "WARNING: torchaudio pre-built wheel not available."
            echo "You may need to build from source. See README.md for instructions."
        }
        ;;
    3)
        echo "Installing PyTorch for JetPack 5.1.x..."
        # JetPack 5.1.x / L4T R35.x / Python 3.8
        TORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl"
        python3 -m pip install --no-cache "$TORCH_URL"

        echo ""
        echo "WARNING: torchaudio for JetPack 5.x may require building from source."
        echo "See README.md for build instructions."
        ;;
    4)
        echo "Skipping PyTorch/torchaudio installation..."
        ;;
    *)
        echo "Invalid choice. Please install PyTorch manually."
        echo "Visit: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html"
        echo "Or use: https://pypi.jetson-ai-lab.io/"
        ;;
esac

# Verify PyTorch installation
echo ""
echo "Verifying PyTorch installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" || {
    echo "ERROR: PyTorch not properly installed!"
    echo "Please install PyTorch manually before continuing."
    exit 1
}

# Verify torchaudio installation
echo ""
echo "Verifying torchaudio installation..."
python3 -c "import torchaudio; print(f'torchaudio version: {torchaudio.__version__}')" || {
    echo ""
    echo "WARNING: torchaudio not installed or not working."
    echo "The model may still work, but some audio features might be limited."
    echo "To build from source, run:"
    echo "  sudo apt install -y libavformat-dev libavcodec-dev libavutil-dev libavdevice-dev libavfilter-dev"
    echo "  git clone https://github.com/pytorch/audio && cd audio"
    echo "  USE_CUDA=1 pip install -v -e . --no-use-pep517"
}

# Install remaining dependencies
echo ""
echo "[6/6] Installing remaining dependencies..."
python3 -m pip install \
    "transformers==4.57.3" \
    "accelerate==1.12.0" \
    "gradio>=4.0.0" \
    "librosa>=0.10.0" \
    "soundfile>=0.12.0" \
    "scipy>=1.10.0" \
    "huggingface_hub>=0.20.0" \
    "einops>=0.7.0"

# Install onnxruntime for Jetson (GPU version)
echo ""
echo "Installing onnxruntime-gpu for Jetson..."
python3 -m pip install onnxruntime-gpu || {
    echo "Warning: onnxruntime-gpu installation failed."
    echo "You may need to install from Jetson Zoo: https://elinux.org/Jetson_Zoo"
}

# Install Qwen3-TTS local package
echo ""
echo "Installing Qwen3-TTS local package..."
if [ -d "Qwen3-TTS" ]; then
    python3 -m pip install -e ./Qwen3-TTS
else
    echo "Warning: Qwen3-TTS directory not found. Skipping local installation."
fi

echo ""
echo "=============================================="
echo "Installation complete!"
echo "=============================================="
echo ""
echo "To run the Gradio app:"
echo "  python3 jetson_gradio_app.py <model_checkpoint> --no-flash-attn"
echo ""
echo "Example:"
echo "  python3 jetson_gradio_app.py Qwen/Qwen3-TTS-Base --no-flash-attn --dtype float16"
echo ""
