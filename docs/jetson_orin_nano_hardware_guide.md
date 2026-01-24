# NVIDIA Jetson Orin Nano 硬件配置与注意事项

## 1. 硬件规格总览

### 1.1 处理器与计算能力

| 项目 | Jetson Orin Nano 8GB | Jetson Orin Nano Super 8GB |
|------|---------------------|---------------------------|
| **GPU** | NVIDIA Ampere (GA10B) | NVIDIA Ampere (GA10B) |
| **CUDA 核心数** | 1024 | 1024 |
| **Tensor 核心数** | 32 (第三代) | 32 (第三代) |
| **GPU 最大频率** | 625 MHz | 1020 MHz |
| **AI 算力 (INT8)** | 40 TOPS | 67 TOPS |
| **CPU** | 6-core Arm Cortex-A78AE | 6-core Arm Cortex-A78AE |
| **CPU 最大频率** | 1.5 GHz | 1.7 GHz |

### 1.2 内存与存储

| 项目 | 规格 |
|------|------|
| **内存类型** | LPDDR5 |
| **内存容量** | 8 GB (共享 CPU/GPU) |
| **内存带宽** | 64-102 GB/s |
| **内存位宽** | 128-bit |
| **存储接口** | NVMe M.2 Key-M (PCIe Gen3 x4) |
| **microSD 卡槽** | 支持 UHS-I |

### 1.3 功耗与散热

| 功耗模式 | 功率 | 适用场景 |
|---------|------|---------|
| **7W** | 低功耗模式 | 电池供电/移动设备 |
| **15W** | 标准模式 | 常规推理任务 |
| **25W** | Super 模式 | 高性能 LLM 推理 |

> **散热建议**: 在 15W 以上模式下建议使用主动散热风扇

### 1.4 接口与连接

- **USB**: 4x USB 3.2 Gen2 Type-A
- **显示**: DisplayPort 1.4a, HDMI 2.1
- **网络**: Gigabit Ethernet
- **扩展**: 40-pin GPIO, CSI 摄像头接口
- **无线**: 需外接 M.2 Key-E WiFi/BT 模块

## 2. 软件环境

### 2.1 JetPack 版本对比

| JetPack 版本 | Ubuntu | CUDA | TensorRT | cuDNN | Python |
|-------------|--------|------|----------|-------|--------|
| **6.2** (推荐) | 22.04 | 12.6 | 10.3 | 9.3 | 3.10 |
| **6.1** | 22.04 | 12.4 | 10.0 | 8.9 | 3.10 |
| **5.1.3** | 20.04 | 11.4 | 8.6 | 8.6 | 3.8 |

### 2.2 深度学习框架支持

| 框架 | 版本 | 安装方式 |
|-----|------|---------|
| **PyTorch** | 2.8.0 | Jetson PyPI |
| **TensorFlow** | 2.16 | Jetson PyPI |
| **ONNX Runtime** | 1.19 | pip |
| **TensorRT-LLM** | 0.12.0-jetson | 源码编译 |

## 3. 关键注意事项

### 3.1 内存限制

- **统一内存架构**: CPU 和 GPU 共享 8GB 内存
- **实际可用**: 系统占用约 1-1.5GB，实际可用约 6.5GB
- **模型参数限制**:
  - FP16 精度: 建议 ≤ 3B 参数
  - INT8 量化: 可支持 4-8B 参数
  - INT4 量化: 可支持更大模型

### 3.2 FlashAttention 限制

| 问题 | 说明 |
|-----|------|
| **不支持 FlashAttention 2** | Jetson Orin Nano 的 Ampere GPU 不完全兼容 FlashAttention 2 的某些优化 |
| **替代方案** | 使用 `sdpa_kernel` 或 `xformers` 替代 |
| **注意力实现** | 设置 `attn_implementation="eager"` 或 `attn_implementation="sdpa"` |

### 3.3 数据类型支持

| 数据类型 | 支持情况 | 备注 |
|---------|---------|------|
| **FP32** | 完全支持 | 内存消耗最大 |
| **FP16** | 完全支持 | 推荐使用 |
| **BF16** | 部分支持 | Ampere 架构原生支持，但 Jetson 上性能可能不如 FP16 |
| **INT8** | 完全支持 | 需要量化校准 |
| **INT4** | 支持 (通过 GPTQ/AWQ) | 需要专门的量化模型 |

### 3.4 存储与 I/O

- **NVMe 推荐**: 模型加载速度比 SD 卡快 5-10 倍
- **Swap 配置**: 建议配置 8-16GB swap 空间防止 OOM
- **模型存储**: 预留足够空间，单个模型可能占用 2-10GB

### 3.5 散热与降频

```bash
# 查看当前温度和频率
sudo jtop

# 锁定最大性能模式 (需要良好散热)
sudo jetson_clocks

# 设置功耗模式 (7W/15W/25W 等)
sudo nvpmodel -m 0  # Super mode (25W)
sudo nvpmodel -m 1  # Normal mode (15W)
```

> **警告**: 温度过高会导致自动降频，影响推理性能

### 3.6 电源要求

- **推荐电源**: 5V 5A (25W) 或更高
- **USB-C 供电**: 支持 PD 协议
- **不稳定电源**: 可能导致系统重启或数据损坏

## 4. 性能优化建议

### 4.1 系统级优化

```bash
# 禁用桌面环境释放内存
sudo systemctl set-default multi-user.target

# 增加 swap 空间
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 配置 zram 压缩内存
sudo systemctl enable nvzramconfig
```

### 4.2 推理优化

1. **使用 TensorRT 加速**: 将 PyTorch/ONNX 模型转换为 TensorRT 引擎
2. **启用量化**: INT8 量化可减少 50% 内存占用
3. **批处理**: 合理设置 batch size 提高吞吐量
4. **流式生成**: 使用流式输出降低首 token 延迟

### 4.3 监控工具

| 工具 | 用途 |
|-----|------|
| `jtop` | GPU/CPU 使用率、温度、内存监控 |
| `tegrastats` | 系统资源实时监控 |
| `nvpmodel` | 功耗模式管理 |
| `jetson_clocks` | 性能模式切换 |

## 5. 常见问题

### Q1: 内存不足 (OOM)

**解决方案**:
- 增加 swap 空间
- 使用量化模型 (INT8/INT4)
- 减少 batch size
- 关闭不必要的后台进程

### Q2: 推理速度慢

**解决方案**:
- 检查是否处于高性能模式 (`nvpmodel -q`)
- 确认 GPU 频率未被降频 (`jtop`)
- 使用 TensorRT 优化模型
- 使用 NVMe 存储替代 SD 卡

### Q3: 模型加载失败

**解决方案**:
- 检查模型文件完整性
- 确认依赖包版本兼容
- 使用 `dtype=torch.float16` 减少内存需求
- 禁用 FlashAttention (`attn_implementation="eager"`)

## 6. 参考资源

- [NVIDIA Jetson Orin Nano 官方页面](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/)
- [JetPack SDK 文档](https://developer.nvidia.com/embedded/jetpack)
- [Jetson AI Lab](https://www.jetson-ai-lab.com/)
- [NVIDIA Developer Forums - Jetson](https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/)
- [Hello AI World 教程](https://github.com/dusty-nv/jetson-inference)
