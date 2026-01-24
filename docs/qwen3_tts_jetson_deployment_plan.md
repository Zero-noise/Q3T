# Qwen3-TTS 在 Jetson Orin Nano 上的部署方案

## 1. 项目概述

### 1.1 Qwen3-TTS 简介

Qwen3-TTS 是阿里巴巴 Qwen 团队开发的端到端语音合成模型，具备以下核心能力：
- **语音克隆**: 3 秒快速克隆任意音色
- **语音设计**: 通过自然语言描述生成定制音色
- **多语言支持**: 中、英、日、韩等 10 种语言
- **流式生成**: 端到端延迟低至 97ms

### 1.2 模型规格

| 模型版本 | 参数量 | 估算显存 (FP16) | Jetson 可行性 |
|---------|--------|----------------|--------------|
| **0.6B-Base** | 0.6B | ~1.5 GB | ✅ 推荐 |
| **0.6B-CustomVoice** | 0.6B | ~1.5 GB | ✅ 推荐 |
| **1.7B-Base** | 1.7B | ~4 GB | ⚠️ 可行 (需优化) |
| **1.7B-CustomVoice** | 1.7B | ~4 GB | ⚠️ 可行 (需优化) |
| **1.7B-VoiceDesign** | 1.7B | ~4 GB | ⚠️ 可行 (需优化) |

> **注**: 还需要加载 Tokenizer 模型约 0.5-1GB 显存

### 1.3 Jetson Orin Nano 8GB 资源评估

| 资源 | 总量 | 系统占用 | 可用于推理 |
|-----|------|---------|-----------|
| **内存/显存** | 8 GB | ~1.5 GB | ~6.5 GB |
| **GPU 算力** | 67 TOPS | - | 全部可用 |
| **存储** | NVMe | ~5 GB (系统) | 按需扩展 |

## 2. 部署方案

### 2.1 推荐配置

| 配置项 | 推荐值 | 说明 |
|-------|-------|------|
| **模型版本** | 0.6B-Base/CustomVoice | 资源占用适中，效果良好 |
| **数据类型** | FP16 (torch.float16) | Jetson 优化最佳 |
| **注意力实现** | eager 或 sdpa | 不使用 FlashAttention |
| **功耗模式** | 25W (Super Mode) | 最佳推理性能 |
| **存储** | NVMe SSD | 模型加载更快 |

### 2.2 部署架构

```
┌─────────────────────────────────────────────────────────┐
│                   Jetson Orin Nano 8GB                  │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────┐    │
│  │                 应用层 (Application)              │    │
│  │  ┌───────────┐  ┌───────────┐  ┌─────────────┐  │    │
│  │  │ Gradio UI │  │ REST API  │  │ gRPC Server │  │    │
│  │  └─────┬─────┘  └─────┬─────┘  └──────┬──────┘  │    │
│  └────────┼──────────────┼───────────────┼─────────┘    │
│           └──────────────┼───────────────┘              │
│  ┌───────────────────────▼─────────────────────────┐    │
│  │              推理引擎 (Inference Engine)          │    │
│  │  ┌───────────────────────────────────────────┐  │    │
│  │  │         Qwen3TTSModel (0.6B/1.7B)         │  │    │
│  │  │  • generate_custom_voice()                │  │    │
│  │  │  • generate_voice_clone()                 │  │    │
│  │  │  • generate_voice_design()                │  │    │
│  │  └───────────────────────────────────────────┘  │    │
│  │  ┌───────────────────────────────────────────┐  │    │
│  │  │       Qwen3TTSTokenizer (12Hz)            │  │    │
│  │  │  • encode() / decode()                    │  │    │
│  │  └───────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────┐    │
│  │              运行时 (Runtime Stack)              │    │
│  │  PyTorch 2.8 | CUDA 12.6 | cuDNN 9.3            │    │
│  │  JetPack 6.2 | Ubuntu 22.04                     │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## 3. 环境配置步骤

### 3.1 系统准备

```bash
# 1. 刷入 JetPack 6.2
# 使用 NVIDIA SDK Manager 或从官网下载镜像

# 2. 设置高性能模式
sudo nvpmodel -m 0  # Super Mode (25W)
sudo jetson_clocks

# 3. 配置 swap 空间 (防止 OOM)
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# 4. 更新系统
sudo apt update && sudo apt upgrade -y
```

### 3.2 Python 环境

```bash
# 1. 创建虚拟环境
python3 -m venv ~/qwen3-tts-env
source ~/qwen3-tts-env/bin/activate

# 2. 安装 Jetson 优化的 PyTorch
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126

# 3. 安装基础依赖
pip install transformers==4.57.3 accelerate==1.12.0
pip install soundfile librosa numpy scipy

# 4. 安装 qwen-tts 包
pip install qwen-tts

# 或从源码安装 (推荐，便于调试)
git clone https://github.com/QwenLM/Qwen3-TTS.git
cd Qwen3-TTS
pip install -e .
```

### 3.3 模型下载

```bash
# 推荐: 使用 ModelScope (国内访问更快)
pip install modelscope
modelscope download --model Qwen/Qwen3-TTS-Tokenizer-12Hz \
    --local_dir ~/models/Qwen3-TTS-Tokenizer-12Hz

modelscope download --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    --local_dir ~/models/Qwen3-TTS-12Hz-0.6B-Base

# 或使用 HuggingFace
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    --local-dir ~/models/Qwen3-TTS-12Hz-0.6B-Base
```

## 4. 代码适配

### 4.1 Jetson 专用推理脚本

```python
#!/usr/bin/env python3
"""
Qwen3-TTS Jetson Orin Nano Inference Script
针对 Jetson 优化的推理脚本
"""

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

def load_model_for_jetson(model_path: str):
    """
    为 Jetson Orin Nano 优化的模型加载

    关键适配:
    1. 使用 FP16 而非 BF16
    2. 禁用 FlashAttention
    3. 显式指定设备
    """
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map="cuda:0",
        dtype=torch.float16,              # Jetson 上 FP16 性能更优
        attn_implementation="eager",       # 不使用 FlashAttention
        low_cpu_mem_usage=True,           # 减少 CPU 内存使用
    )
    return model


def generate_speech(
    model,
    text: str,
    language: str = "Chinese",
    speaker: str = "Vivian",
    output_path: str = "output.wav"
):
    """
    生成语音 (CustomVoice 模型)
    """
    with torch.inference_mode():
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            max_new_tokens=1024,  # 限制生成长度以控制延迟
        )

    sf.write(output_path, wavs[0], sr)
    print(f"Audio saved to: {output_path}")
    return output_path


def generate_voice_clone(
    model,
    text: str,
    ref_audio: str,
    ref_text: str,
    language: str = "Chinese",
    output_path: str = "output_clone.wav"
):
    """
    语音克隆 (Base 模型)
    """
    with torch.inference_mode():
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            max_new_tokens=1024,
        )

    sf.write(output_path, wavs[0], sr)
    print(f"Cloned audio saved to: {output_path}")
    return output_path


def main():
    # 模型路径 (使用本地下载的模型)
    MODEL_PATH = "~/models/Qwen3-TTS-12Hz-0.6B-CustomVoice"

    print("Loading model...")
    model = load_model_for_jetson(MODEL_PATH)
    print("Model loaded successfully!")

    # 测试生成
    test_text = "你好，欢迎使用通义语音合成系统。"
    generate_speech(model, test_text, speaker="Vivian")


if __name__ == "__main__":
    main()
```

### 4.2 内存优化技巧

```python
import torch
import gc

def optimize_memory():
    """清理 GPU 内存"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def inference_with_memory_management(model, texts: list):
    """
    带内存管理的批量推理
    """
    results = []

    for text in texts:
        # 推理
        with torch.inference_mode():
            wav, sr = model.generate_custom_voice(
                text=text,
                language="Chinese",
                speaker="Vivian",
            )
        results.append((wav[0], sr))

        # 每次推理后清理内存
        optimize_memory()

    return results
```

### 4.3 流式输出服务

```python
"""
基于 FastAPI 的流式 TTS 服务
适用于低延迟场景
"""

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import io
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

app = FastAPI()

# 全局模型实例
model = None

@app.on_event("startup")
async def load_model():
    global model
    model = Qwen3TTSModel.from_pretrained(
        "~/models/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        device_map="cuda:0",
        dtype=torch.float16,
        attn_implementation="eager",
    )


@app.post("/tts")
async def text_to_speech(request: Request):
    data = await request.json()
    text = data.get("text", "")
    speaker = data.get("speaker", "Vivian")
    language = data.get("language", "Chinese")

    with torch.inference_mode():
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
        )

    # 返回 WAV 音频流
    buffer = io.BytesIO()
    sf.write(buffer, wavs[0], sr, format='WAV')
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=output.wav"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 5. 性能测试与基准

### 5.1 预期性能

| 模型 | 文本长度 | 预计首包延迟 | 预计 RTF |
|-----|---------|-------------|---------|
| 0.6B-CustomVoice | 20 字 | 300-500ms | 0.3-0.5 |
| 0.6B-Base (Clone) | 20 字 | 400-600ms | 0.4-0.6 |
| 1.7B-CustomVoice | 20 字 | 800-1200ms | 0.6-0.9 |

> **RTF (Real-Time Factor)**: < 1 表示实时生成

### 5.2 性能测试脚本

```python
import time
import torch
from qwen_tts import Qwen3TTSModel

def benchmark(model, text, iterations=10):
    """性能基准测试"""
    times = []

    # 预热
    with torch.inference_mode():
        model.generate_custom_voice(text=text, language="Chinese", speaker="Vivian")

    # 正式测试
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.time()

        with torch.inference_mode():
            wavs, sr = model.generate_custom_voice(
                text=text,
                language="Chinese",
                speaker="Vivian",
            )

        torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)

    audio_duration = len(wavs[0]) / sr
    avg_time = sum(times) / len(times)
    rtf = avg_time / audio_duration

    print(f"Text: {text}")
    print(f"Audio duration: {audio_duration:.2f}s")
    print(f"Average inference time: {avg_time:.3f}s")
    print(f"RTF: {rtf:.3f}")
    print(f"Real-time capable: {'Yes' if rtf < 1 else 'No'}")


if __name__ == "__main__":
    model = Qwen3TTSModel.from_pretrained(
        "~/models/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        device_map="cuda:0",
        dtype=torch.float16,
        attn_implementation="eager",
    )

    test_texts = [
        "你好世界",
        "欢迎使用通义语音合成系统，这是一段测试文本。",
        "人工智能正在改变我们的生活方式，语音合成技术让机器能够像人一样说话。",
    ]

    for text in test_texts:
        benchmark(model, text)
        print("-" * 50)
```

## 6. 部署检查清单

### 6.1 硬件检查

- [ ] Jetson Orin Nano 8GB 开发套件
- [ ] 主动散热风扇 (推荐)
- [ ] NVMe SSD 存储 (推荐 256GB+)
- [ ] 5V/5A 电源适配器
- [ ] 网络连接 (首次下载模型)

### 6.2 软件检查

- [ ] JetPack 6.2 已安装
- [ ] Python 3.10 环境已配置
- [ ] PyTorch 2.8 Jetson 版本已安装
- [ ] qwen-tts 包已安装
- [ ] 模型权重已下载到本地

### 6.3 配置检查

- [ ] nvpmodel 设置为 Super Mode (25W)
- [ ] Swap 空间已配置 (16GB+)
- [ ] GPU 频率未被降频
- [ ] 模型使用 FP16 精度
- [ ] 禁用 FlashAttention

## 7. 故障排除

### 7.1 常见错误

| 错误信息 | 原因 | 解决方案 |
|---------|------|---------|
| `CUDA out of memory` | 显存不足 | 使用 0.6B 模型；增加 swap；减少 batch size |
| `FlashAttention not supported` | 不兼容的注意力实现 | 设置 `attn_implementation="eager"` |
| `RuntimeError: cuDNN error` | cuDNN 版本问题 | 确保使用 Jetson PyPI 的 PyTorch |
| 推理速度过慢 | 降频或功耗模式限制 | 检查 `jtop`；运行 `jetson_clocks` |

### 7.2 监控命令

```bash
# 实时监控系统状态
sudo jtop

# 查看 GPU 使用情况
nvidia-smi  # 注意: Jetson 上可能需要使用 tegrastats

# 查看内存使用
free -h

# 查看功耗模式
sudo nvpmodel -q

# 查看温度
cat /sys/devices/virtual/thermal/thermal_zone*/temp
```

## 8. 扩展方案

### 8.1 量化加速 (进阶)

如果 1.7B 模型性能不足，考虑使用量化：

```python
# INT8 动态量化示例 (需要适配 Qwen3-TTS)
from torch.quantization import quantize_dynamic

model_int8 = quantize_dynamic(
    model.model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
```

### 8.2 TensorRT 加速 (进阶)

```python
# 导出 ONNX 并转换为 TensorRT
# 注意: 需要适配 Qwen3-TTS 的模型结构
import torch.onnx

# 导出核心模块为 ONNX
torch.onnx.export(
    model.model,
    dummy_input,
    "qwen3_tts.onnx",
    opset_version=17,
)

# 使用 trtexec 转换
# trtexec --onnx=qwen3_tts.onnx --saveEngine=qwen3_tts.trt --fp16
```

## 9. 参考资源

- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
- [Qwen3-TTS HuggingFace](https://huggingface.co/collections/Qwen/qwen3-tts)
- [Jetson AI Lab](https://www.jetson-ai-lab.com/)
- [JetPack 6.2 Release Notes](https://developer.nvidia.com/embedded/jetpack)
- [PyTorch Jetson 安装指南](https://pypi.jetson-ai-lab.io/)
