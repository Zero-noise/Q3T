# Qwen3-TTS Jetson Orin 部署

在 NVIDIA Jetson Orin 上部署 Qwen3-TTS 语音合成模型。

## 要求

- **硬件**: Jetson Orin Nano 8GB+ / Orin NX / AGX Orin
- **系统**: JetPack 6.0+ (Python 3.10, CUDA 12.x)

## 快速开始

### 1. 系统准备

```bash
# 高性能模式
sudo nvpmodel -m 0 && sudo jetson_clocks

# 配置 swap (防止 OOM)
sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
```

### 2. 创建虚拟环境

```bash
python3 -m venv ~/qwen3-tts-env
source ~/qwen3-tts-env/bin/activate
```

### 3. 一键安装

```bash
bash install_jetson.sh
```

脚本会自动安装 PyTorch、torchaudio 和所有依赖。运行时选择 **选项 1** (Jetson AI Lab PyPI)。

### 4. 下载模型

```bash
# 使用 ModelScope (国内)
pip install modelscope
modelscope download --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --local_dir ~/models/Qwen3-TTS-0.6B

# 或使用 HuggingFace
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir ~/models/Qwen3-TTS-0.6B
```

### 5. 运行

```bash
python3 jetson_gradio_app.py ~/models/Qwen3-TTS-0.6B --no-flash-attn --dtype float16
```

访问 `http://<jetson-ip>:8000`

## 手动安装 (可选)

如果一键脚本不适用，可手动安装：

```bash 
# 创建虚拟环境
python3 -m venv ~/qwen3-tts-env && source ~/qwen3-tts-env/bin/activate

# 安装 PyTorch (必须使用 Jetson 专用源)
pip install torch torchvision torchaudio --index-url https://pypi.jetson-ai-lab.io/jp6/cu126

# 安装依赖
pip install transformers==4.57.3 accelerate==1.12.0 gradio librosa soundfile
pip install -e ./Qwen3-TTS
```

## 模型选择

| 模型 | 显存 | 兼容性 |
|-----|------|-------|
| 0.6B-Base / CustomVoice | ~1.5 GB | ✅ 推荐 |
| 1.7B 系列 | ~4 GB | ⚠️ 需 16GB+ |

## 故障排除

| 问题 | 解决方案 |
|-----|---------|
| CUDA out of memory | 增加 swap 或使用 0.6B 模型 |
| torchaudio 安装失败 | 使用 `--index-url https://pypi.jetson-ai-lab.io/jp6/cu126` |
| 推理速度慢 | 运行 `sudo nvpmodel -m 0 && sudo jetson_clocks` |

## 参考

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [Jetson AI Lab PyPI](https://pypi.jetson-ai-lab.io/)
- [NVIDIA PyTorch for Jetson](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)
