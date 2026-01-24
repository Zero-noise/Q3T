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

### 4. 运行

```bash
# 直接启动 - 自动检测本地模型或显示下载界面
python3 jetson_gradio_app.py --no-flash-attn

# 或指定模型路径
python3 jetson_gradio_app.py ./Qwen__Qwen3-TTS-12Hz-0.6B-Base --no-flash-attn --dtype float16
```

访问 `http://<jetson-ip>:8000`

如果本地没有模型，程序会自动显示下载界面，可以：
- 查看已检测到的本地模型
- 选择模型并下载到当前目录(默认)或自定义目录
- 手动指定已有模型的路径

### 5. 手动下载模型 (可选)

如果希望提前下载模型(与应用默认目录一致)：

```bash
# 使用 ModelScope (国内)
pip install modelscope
modelscope download --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --local_dir ./Qwen__Qwen3-TTS-12Hz-0.6B-Base

# 或使用 HuggingFace
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir ./Qwen__Qwen3-TTS-12Hz-0.6B-Base
```

如需自定义下载根目录，可设置环境变量：

```bash
export QWEN3_TTS_DOWNLOAD_DIR=/path/to/models
```

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

## 自动模型检测

程序会自动扫描以下位置查找已下载的模型：

- 当前工作目录下的默认下载子目录(或 `QWEN3_TTS_DOWNLOAD_DIR`)
- 环境变量指定的 HuggingFace 目录: `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`
- HuggingFace 默认缓存目录: `~/.cache/huggingface/hub`
- 命令行指定的路径

支持的模型：
- `Qwen/Qwen3-TTS-12Hz-0.6B-Base` - 语音克隆
- `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` - 预定义说话人
- `Qwen/Qwen3-TTS-12Hz-0.6B-VoiceDesign` - 文字描述控制风格
- `Qwen/Qwen3-TTS-25Hz-0.6B-*` - 25Hz 版本 (质量更高)

## 模型选择

| 模型 | 显存 | 兼容性 |
|-----|------|-------|
| 0.6B-Base / CustomVoice | ~1.5 GB | ✅ 推荐 |
| 1.7B 系列 | ~4 GB | ⚠️ 需 16GB+ |

**12Hz vs 25Hz**: 12Hz 模型推理更快，25Hz 模型音质更高

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
