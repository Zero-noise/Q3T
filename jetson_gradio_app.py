#!/usr/bin/env python3
# coding=utf-8
"""
Minimal Gradio app for deploying Qwen3-TTS on Jetson Orin.

Supports Base / CustomVoice / VoiceDesign models based on the checkpoint.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import torch
from transformers.generation.logits_process import LogitsProcessorList

from log_config import get_logger, setup_logging

logger = get_logger(__name__)

_STARTUP_CWD = Path.cwd().resolve()

class _NanClampLogitsProcessor:
    def __call__(self, input_ids, scores):
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores = torch.nan_to_num(scores, nan=-1e4, posinf=-1e4, neginf=-1e4)
        return scores

def _add_local_qwen_repo() -> None:
    env_root = os.getenv("QWEN3_TTS_ROOT")
    candidates = []
    if env_root:
        candidates.append(env_root)
    candidates.append(str(Path(__file__).resolve().parent / "Qwen3-TTS"))
    for c in candidates:
        if c and Path(c).is_dir():
            if c not in sys.path:
                sys.path.insert(0, c)
            return


_add_local_qwen_repo()

from qwen_tts import Qwen3TTSModel


def _format_bytes(num_bytes: Optional[int]) -> str:
    if num_bytes is None:
        return "unknown"
    if num_bytes <= 0:
        return "0 B"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(num_bytes)
    for u in units:
        if size < 1024 or u == units[-1]:
            return f"{size:.2f} {u}"
        size /= 1024.0
    return f"{size:.2f} TiB"


def _get_default_download_root() -> Path:
    env_root = os.getenv("QWEN3_TTS_DOWNLOAD_DIR")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return _STARTUP_CWD


def _get_default_download_dir(repo_id: str) -> Path:
    safe_repo = repo_id.replace("/", "__")
    return _get_default_download_root() / safe_repo


def _coerce_int(value: Optional[float], default: int) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, float) and np.isnan(value):
            return default
        return int(value)
    except Exception:
        return default


def _coerce_float(value: Optional[float], default: float) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, float) and np.isnan(value):
            return default
        return float(value)
    except Exception:
        return default


def _read_meminfo() -> Dict[str, int]:
    meminfo: Dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) != 2:
                    continue
                key = parts[0].strip()
                value = parts[1].strip().split()
                if not value:
                    continue
                try:
                    meminfo[key] = int(value[0]) * 1024
                except ValueError:
                    continue
    except OSError:
        pass
    return meminfo


def _check_swap() -> Dict[str, Any]:
    result: Dict[str, Any] = {"enabled": False, "total_bytes": 0, "used_bytes": 0, "entries": []}
    try:
        with open("/proc/swaps", "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    except OSError:
        return result

    if len(lines) <= 1:
        return result

    total_kib = 0
    used_kib = 0
    entries = []
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) < 5:
            continue
        filename, _, size_kib, used_kib_entry, _ = parts[:5]
        try:
            size_kib = int(size_kib)
            used_kib_entry = int(used_kib_entry)
        except ValueError:
            continue
        total_kib += size_kib
        used_kib += used_kib_entry
        entries.append({"path": filename, "size_kib": size_kib, "used_kib": used_kib_entry})

    result["enabled"] = total_kib > 0
    result["total_bytes"] = total_kib * 1024
    result["used_bytes"] = used_kib * 1024
    result["entries"] = entries
    return result


def _check_cuda_mem() -> Dict[str, Optional[int]]:
    if not torch.cuda.is_available():
        return {"total": None, "free": None, "used": None}
    if hasattr(torch.cuda, "mem_get_info"):
        free, total = torch.cuda.mem_get_info()
        return {"total": int(total), "free": int(free), "used": int(total - free)}
    return {"total": None, "free": None, "used": None}


class JetsonMonitor:
    """Parse tegrastats output for real-time Jetson hardware monitoring."""

    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._process:
            try:
                self._process.terminate()
            except Exception:
                logger.debug("Failed to terminate tegrastats process", exc_info=True)

    def _poll_loop(self) -> None:
        """Try tegrastats first, fall back to sysfs readings."""
        try:
            self._process = subprocess.Popen(
                ["tegrastats", "--interval", "2000"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            for line in iter(self._process.stdout.readline, ""):
                if not self._running:
                    break
                self._parse_tegrastats(line.strip())
        except (FileNotFoundError, PermissionError):
            # tegrastats not available or no permission, use sysfs fallback
            while self._running:
                self._read_sysfs()
                time.sleep(2)

    def _parse_tegrastats(self, line: str) -> None:
        """Parse a single tegrastats output line."""
        data: Dict[str, Any] = {}

        # RAM: 3456/7620MB
        ram_match = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
        if ram_match:
            data["ram_used_mb"] = int(ram_match.group(1))
            data["ram_total_mb"] = int(ram_match.group(2))

        # SWAP: 0/8192MB
        swap_match = re.search(r"SWAP\s+(\d+)/(\d+)MB", line)
        if swap_match:
            data["swap_used_mb"] = int(swap_match.group(1))
            data["swap_total_mb"] = int(swap_match.group(2))

        # GR3D_FREQ 45%
        gr3d_match = re.search(r"GR3D_FREQ\s+(\d+)%", line)
        if gr3d_match:
            data["gpu_util_pct"] = int(gr3d_match.group(1))

        # CPU temperatures: CPU@45.5C or cpu@45.5C
        cpu_temp_match = re.search(r"[Cc][Pp][Uu]@([\d.]+)C", line)
        if cpu_temp_match:
            data["cpu_temp_c"] = float(cpu_temp_match.group(1))

        # GPU temperature: GPU@42C or gpu@42C
        gpu_temp_match = re.search(r"[Gg][Pp][Uu]@([\d.]+)C", line)
        if gpu_temp_match:
            data["gpu_temp_c"] = float(gpu_temp_match.group(1))

        # VDD_IN power: VDD_IN 4500mW or VDD_IN 4500/5000
        vdd_match = re.search(r"VDD_IN\s+(\d+)(?:mW|/\d+)", line)
        if vdd_match:
            data["power_mw"] = int(vdd_match.group(1))

        with self._lock:
            self._data.update(data)

    def _read_sysfs(self) -> None:
        """Fallback: read temperatures from sysfs thermal zones."""
        data: Dict[str, Any] = {}
        thermal_base = Path("/sys/devices/virtual/thermal")
        if thermal_base.exists():
            for tz in sorted(thermal_base.glob("thermal_zone*")):
                try:
                    tz_type = (tz / "type").read_text().strip().lower()
                    temp_raw = (tz / "temp").read_text().strip()
                    temp_c = int(temp_raw) / 1000.0
                    if "gpu" in tz_type:
                        data["gpu_temp_c"] = temp_c
                    elif "cpu" in tz_type or "soc" in tz_type:
                        data.setdefault("cpu_temp_c", temp_c)
                except (OSError, ValueError):
                    continue

        with self._lock:
            self._data.update(data)

    def get_data(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._data)

    def format_summary(self) -> str:
        d = self.get_data()
        if not d:
            return "Jetson 监控: 等待数据..."

        lines = []
        # Temperatures
        gpu_temp = d.get("gpu_temp_c")
        cpu_temp = d.get("cpu_temp_c")
        if gpu_temp is not None or cpu_temp is not None:
            temp_parts = []
            if gpu_temp is not None:
                temp_parts.append(f"GPU {gpu_temp:.0f}°C")
            if cpu_temp is not None:
                temp_parts.append(f"CPU {cpu_temp:.0f}°C")
            lines.append(f"温度: {' | '.join(temp_parts)}")

        # GPU utilization
        gpu_util = d.get("gpu_util_pct")
        if gpu_util is not None:
            lines.append(f"GPU 利用率: {gpu_util}%")

        # Power
        power = d.get("power_mw")
        if power is not None:
            lines.append(f"功耗: {power/1000:.1f} W")

        # RAM
        ram_used = d.get("ram_used_mb")
        ram_total = d.get("ram_total_mb")
        if ram_used is not None and ram_total is not None:
            lines.append(f"RAM: {ram_used}/{ram_total} MB ({ram_used*100//ram_total}%)")

        # Swap
        swap_used = d.get("swap_used_mb")
        swap_total = d.get("swap_total_mb")
        if swap_used is not None and swap_total is not None:
            lines.append(f"Swap: {swap_used}/{swap_total} MB")

        return "\n".join(lines) if lines else "Jetson 监控: 无数据"


class InferenceTracker:
    """Track inference latency and compute RTF (Real-Time Factor)."""

    def __init__(self, max_history: int = 100):
        self._history: List[Dict[str, float]] = []
        self._max_history = max_history
        self._lock = threading.Lock()

    def record(self, latency_s: float, text_len: int, audio_len_s: float) -> None:
        rtf = latency_s / audio_len_s if audio_len_s > 0 else float("inf")
        entry = {
            "latency_s": latency_s,
            "text_len": text_len,
            "audio_len_s": audio_len_s,
            "rtf": rtf,
            "timestamp": time.time(),
        }
        with self._lock:
            self._history.append(entry)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

    def format_summary(self) -> str:
        with self._lock:
            history = list(self._history)

        if not history:
            return "推理统计: 暂无数据"

        n = len(history)
        avg_latency = sum(e["latency_s"] for e in history) / n
        avg_rtf = sum(e["rtf"] for e in history) / n
        last = history[-1]

        lines = [
            f"推理统计 (共 {n} 次):",
            f"  平均延迟: {avg_latency:.2f}s | 平均 RTF: {avg_rtf:.2f}",
            f"  最近一次: {last['latency_s']:.2f}s | RTF {last['rtf']:.2f} | "
            f"文本 {last['text_len']} 字 | 音频 {last['audio_len_s']:.1f}s",
        ]
        return "\n".join(lines)


def _check_quantization_status(model: Optional[Any] = None) -> str:
    """Detect if model contains quantized layers and return status string."""
    if model is None:
        return "量化状态: 模型未加载"

    # Check for quantize_config.json in model directory
    total_linear = 0
    quantized_count = 0
    quant_types = set()

    inner_model = getattr(model, "model", model)
    talker = getattr(inner_model, "talker", inner_model)

    for m in talker.modules():
        cls_name = type(m).__name__
        if isinstance(m, torch.nn.Linear):
            total_linear += 1
        elif any(kw in cls_name.lower() for kw in ("int4", "int8", "quant", "affine", "dynamic")):
            quantized_count += 1
            quant_types.add(cls_name)

    if quantized_count > 0:
        types_str = ", ".join(sorted(quant_types))
        return f"量化状态: {quantized_count} 层已量化 (类型: {types_str})"
    elif total_linear > 0:
        return f"量化状态: FP16/FP32 (未量化, {total_linear} Linear 层)"
    else:
        return "量化状态: 无法检测"


# Global monitor and tracker instances
_jetson_monitor: Optional[JetsonMonitor] = None
_inference_tracker: Optional[InferenceTracker] = None


def _get_monitor() -> JetsonMonitor:
    global _jetson_monitor
    if _jetson_monitor is None:
        _jetson_monitor = JetsonMonitor()
        _jetson_monitor.start()
    return _jetson_monitor


def _get_tracker() -> InferenceTracker:
    global _inference_tracker
    if _inference_tracker is None:
        _inference_tracker = InferenceTracker()
    return _inference_tracker


def _cuda_brief_info() -> str:
    if not torch.cuda.is_available():
        return "CUDA available: False"
    try:
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
    except Exception:
        logger.debug("Failed to query CUDA device info", exc_info=True)
        idx = "unknown"
        name = "unknown"
    return f"CUDA available: True | current_device={idx} | name={name}"


def _infer_model_device(model: torch.nn.Module) -> str:
    try:
        p = next(model.parameters())
        return str(p.device)
    except Exception:
        logger.debug("Cannot infer device from parameters, falling back to attr", exc_info=True)
        dev = getattr(model, "device", None)
        return str(dev) if dev is not None else "unknown"


def _move_speech_tokenizer(st, device: str, dtype: Optional[torch.dtype]) -> None:
    if st is None:
        return
    target_device = torch.device(device)
    target_dtype = dtype
    if target_device.type == "cpu" and target_dtype in (torch.float16, torch.bfloat16):
        target_dtype = torch.float32
    if getattr(st, "model", None) is not None:
        if target_dtype is not None:
            st.model = st.model.to(device=target_device, dtype=target_dtype)
        else:
            st.model = st.model.to(device=target_device)
    st.device = target_device


# 支持的 Qwen3-TTS 模型列表
SUPPORTED_MODELS = [
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
]


def _validate_model_dir(model_dir: Path) -> Tuple[bool, bool]:
    has_config = (model_dir / "config.json").exists()
    has_weights = any(model_dir.glob("*.safetensors")) or any(model_dir.glob("*.bin"))
    return has_config, has_weights


def _get_hf_cache_dirs() -> List[Path]:
    """获取 HuggingFace 常见的缓存目录列表"""
    cache_dirs = []

    # 1. 环境变量指定的缓存目录
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        cache_dirs.append(Path(hf_home) / "hub")

    hf_cache = os.getenv("HUGGINGFACE_HUB_CACHE")
    if hf_cache:
        cache_dirs.append(Path(hf_cache))

    transformers_cache = os.getenv("TRANSFORMERS_CACHE")
    if transformers_cache:
        cache_dirs.append(Path(transformers_cache))

    # 2. 默认缓存目录 (基于当前用户主目录)
    home = Path.home()
    default_dirs = [
        home / ".cache" / "huggingface" / "hub",
        home / ".cache" / "huggingface" / "transformers",
        home / ".huggingface" / "hub",
    ]
    cache_dirs.extend(default_dirs)

    # 3. 去重并只返回存在且可访问的目录
    seen = set()
    result = []
    for d in cache_dirs:
        try:
            d = d.resolve()
            if d not in seen and d.exists():
                seen.add(d)
                result.append(d)
        except (PermissionError, OSError):
            # 跳过无权限访问的目录
            continue
    return result


def _find_model_in_hf_cache(repo_id: str) -> Optional[Path]:
    """在 HuggingFace 缓存目录中查找模型"""
    # HuggingFace 缓存使用 models--org--name 格式
    cache_folder_name = "models--" + repo_id.replace("/", "--")

    for cache_dir in _get_hf_cache_dirs():
        try:
            model_cache = cache_dir / cache_folder_name
            if model_cache.exists():
                # 检查 snapshots 目录
                snapshots = model_cache / "snapshots"
                if snapshots.exists():
                    # 返回最新的 snapshot
                    snapshot_dirs = sorted(snapshots.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
                    for snap in snapshot_dirs:
                        if snap.is_dir():
                            # 验证是否有必要的文件
                            has_config, has_weights = _validate_model_dir(snap)
                            if has_config and has_weights:
                                return snap
        except (PermissionError, OSError):
            # 跳过无权限访问的目录
            continue
    return None


def _check_model_downloaded(checkpoint: str) -> Dict[str, Any]:
    # 1. 检查是否是本地路径
    if os.path.exists(checkpoint):
        ckpt_path = Path(checkpoint)
        if ckpt_path.is_file():
            return {
                "status": "local_file",
                "path": str(ckpt_path),
                "error": "checkpoint 必须是目录或 HuggingFace repo id",
            }
        if ckpt_path.is_dir():
            has_config, has_weights = _validate_model_dir(ckpt_path)
            status = "local_dir" if (has_config and has_weights) else "local_dir_invalid"
            return {
                "status": status,
                "path": str(ckpt_path),
                "has_config": has_config,
                "has_weights": has_weights,
                "error": "模型目录缺少 config.json 或权重文件" if status == "local_dir_invalid" else None,
            }
        return {"status": "local_missing", "path": str(ckpt_path)}

    # 2. 尝试使用 huggingface_hub 的 local_files_only 模式
    try:
        from huggingface_hub import snapshot_download

        repo_dir = snapshot_download(
            repo_id=checkpoint,
            local_files_only=True,
            allow_patterns=["*.safetensors", "*.bin", "config.json", "generation_config.json", "*.json"],
        )
        repo_dir = Path(repo_dir)
        has_config, has_weights = _validate_model_dir(repo_dir)
        if has_config and has_weights:
            return {"status": "cached", "path": str(repo_dir)}
        return {
            "status": "cached_invalid",
            "path": str(repo_dir),
            "has_config": has_config,
            "has_weights": has_weights,
            "error": "缓存模型缺少 config.json 或权重文件",
        }
    except Exception:
        logger.debug("huggingface_hub local cache lookup failed for '%s'", checkpoint, exc_info=True)

    # 3. 检查当前目录下的默认下载位置
    try:
        local_dir = _get_default_download_dir(checkpoint)
        if local_dir.exists():
            has_config, has_weights = _validate_model_dir(local_dir)
            status = "local_dir" if (has_config and has_weights) else "local_dir_invalid"
            return {
                "status": status,
                "path": str(local_dir),
                "has_config": has_config,
                "has_weights": has_weights,
                "error": "模型目录缺少 config.json 或权重文件" if status == "local_dir_invalid" else None,
            }
    except Exception:
        logger.debug("Default download dir lookup failed for '%s'", checkpoint, exc_info=True)

    # 4. 手动搜索 HuggingFace 缓存目录
    found_path = _find_model_in_hf_cache(checkpoint)
    if found_path:
        return {"status": "cached", "path": str(found_path)}

    return {"status": "not_cached", "error": "模型未在本地找到"}


def _ensure_output_dir(output_dir: str) -> str:
    path = Path(output_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _save_wav(wav: np.ndarray, sr: int, output_dir: str, prefix: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = f"{prefix}_{ts}_{uuid.uuid4().hex[:6]}.wav"
    out_path = os.path.join(output_dir, name)
    try:
        import soundfile as sf

        sf.write(out_path, wav, sr)
    except Exception:
        logger.debug("soundfile unavailable, falling back to scipy.io.wavfile", exc_info=True)
        try:
            from scipy.io import wavfile

            wav_int16 = np.clip(wav, -1.0, 1.0)
            wav_int16 = (wav_int16 * 32767.0).astype(np.int16)
            wavfile.write(out_path, sr, wav_int16)
        except Exception as e:
            logger.error("Failed to save audio to %s: %s", out_path, e, exc_info=True)
            raise RuntimeError(f"Failed to save audio: {e}") from e
    return out_path


def _system_check_summary(checkpoint: str, output_dir: str, model: Any = None) -> str:
    meminfo = _read_meminfo()
    mem_total = meminfo.get("MemTotal")
    mem_avail = meminfo.get("MemAvailable")
    mem_used = mem_total - mem_avail if mem_total and mem_avail else None

    swap = _check_swap()
    cuda = _check_cuda_mem()
    model_status = _check_model_downloaded(checkpoint)

    lines = []
    lines.append(f"模型检查: {model_status.get('status')}")
    if model_status.get("status") in {"local_dir", "local_dir_invalid", "cached_invalid"}:
        lines.append(f"- 路径: {model_status.get('path')}")
        lines.append(f"- config.json: {model_status.get('has_config')}")
        lines.append(f"- 权重文件: {model_status.get('has_weights')}")
        if model_status.get("error"):
            lines.append(f"- 错误: {model_status.get('error')}")
    elif model_status.get("status") in {"local_file", "cached"}:
        lines.append(f"- 路径: {model_status.get('path')}")
        if model_status.get("error"):
            lines.append(f"- 错误: {model_status.get('error')}")
    elif model_status.get("status") == "not_cached":
        lines.append("- 未在本地缓存检测到模型，可提前离线下载")

    lines.append(f"内存: { _format_bytes(mem_used) } / { _format_bytes(mem_total) } (used/total)")
    if swap["enabled"]:
        lines.append(f"Swap: { _format_bytes(swap['used_bytes']) } / { _format_bytes(swap['total_bytes']) }")
    else:
        lines.append("Swap: 未启用")
    if cuda["total"]:
        lines.append(
            f"CUDA 显存: { _format_bytes(cuda['used']) } / { _format_bytes(cuda['total']) }"
        )
    lines.append(f"输出目录: {output_dir}")

    # Jetson monitor data
    monitor = _get_monitor()
    mon_summary = monitor.format_summary()
    if "等待数据" not in mon_summary and "无数据" not in mon_summary:
        lines.append("")
        lines.append(mon_summary)

    # Quantization status
    lines.append("")
    lines.append(_check_quantization_status(model))

    # Inference stats
    tracker = _get_tracker()
    tracker_summary = tracker.format_summary()
    if "暂无数据" not in tracker_summary:
        lines.append("")
        lines.append(tracker_summary)

    return "\n".join(lines)


def _dtype_from_str(s: str) -> torch.dtype:
    s = (s or "").strip().lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {s}. Use bfloat16/float16/float32.")


def _maybe_auto_language(lang: str) -> str:
    lang = (lang or "").strip()
    return lang if lang else "Auto"


def _collect_gen_kwargs(max_new_tokens, temperature, top_k, top_p, repetition_penalty) -> Dict[str, Any]:
    mapping = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
    }
    return {k: v for k, v in mapping.items() if v is not None}


def _build_logits_processor(enabled: bool) -> Optional[LogitsProcessorList]:
    if not enabled:
        return None
    return LogitsProcessorList([_NanClampLogitsProcessor()])


def _load_tts(args: argparse.Namespace) -> Qwen3TTSModel:
    dtype = _dtype_from_str(args.dtype)
    attn_impl = None if args.no_flash_attn else "flash_attention_2"
    device_map = args.device
    if isinstance(device_map, str) and device_map.startswith("cuda:"):
        parts = device_map.split(":", 1)
        if len(parts) == 2 and parts[1].isdigit():
            try:
                torch.cuda.set_device(int(parts[1]))
            except Exception:
                logger.warning("Failed to set CUDA device %s", parts[1], exc_info=True)
        device_map = "cuda"
    logger.info("device_map=%s (from '%s') | dtype=%s | flash_attn=%s",
                device_map, args.device, dtype, "on" if attn_impl else "off")
    logger.info("%s", _cuda_brief_info())

    # 在加载前清理 GPU 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    use_staged = bool(getattr(args, "staged_load", False))
    tokenizer_on_cpu = bool(getattr(args, "tokenizer_on_cpu", False))
    logger.info("staged_load=%s | tokenizer_on_cpu=%s",
                "on" if use_staged else "off", "on" if tokenizer_on_cpu else "off")

    if use_staged and device_map == "cuda":
        logger.info("staged_load path: CPU -> dtype -> GPU")
        tts = Qwen3TTSModel.from_pretrained(
            args.checkpoint,
            device_map="cpu",
            torch_dtype=dtype if dtype not in (torch.float16, torch.bfloat16) else torch.float32,
            attn_implementation=attn_impl,
            low_cpu_mem_usage=True,
        )
        if dtype in (torch.float16, torch.bfloat16):
            tts.model = tts.model.to(dtype=dtype)
        tts.model = tts.model.to("cuda")
        tts.device = torch.device("cuda")
        if hasattr(tts.model, "device"):
            tts.model.device = torch.device("cuda")
        if hasattr(tts.model, "speech_tokenizer"):
            st = getattr(tts.model, "speech_tokenizer", None)
            if tokenizer_on_cpu:
                _move_speech_tokenizer(st, "cpu", dtype)
            else:
                _move_speech_tokenizer(st, "cuda", dtype)
        return tts

    tts = Qwen3TTSModel.from_pretrained(
        args.checkpoint,
        device_map=device_map,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,  # 减少加载时的内存峰值
    )
    if hasattr(tts.model, "speech_tokenizer"):
        st = getattr(tts.model, "speech_tokenizer", None)
        if tokenizer_on_cpu:
            _move_speech_tokenizer(st, "cpu", dtype)
    return tts


def _build_base_ui(tts: Qwen3TTSModel, output_dir: str, save_audio: bool):
    with gr.Tab("语音克隆 (Base)"):
        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(label="合成文本", lines=4, placeholder="请输入要合成的文本...", info="支持中英文混合")
                with gr.Row():
                    language = gr.Dropdown(label="语言", choices=["Auto", "Chinese", "English"], value="Auto", allow_custom_value=True, scale=1)
                    xvec_only = gr.Checkbox(label="仅使用音色特征 (无需参考文本)", value=False, scale=2)
            with gr.Column(scale=2):
                ref_audio = gr.Audio(label="参考音频", type="filepath", sources=["upload", "microphone"])
                ref_text = gr.Textbox(label="参考音频文本", lines=2, placeholder="输入参考音频中说的内容...", info="关闭「仅音色」时必填")

        with gr.Accordion("生成参数", open=False):
            with gr.Row():
                max_new_tokens = gr.Slider(label="max_new_tokens", minimum=256, maximum=4096, value=1024, step=64)
                temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.05)
            with gr.Row():
                top_k = gr.Slider(label="top_k", minimum=1, maximum=100, value=50, step=1)
                top_p = gr.Slider(label="top_p", minimum=0.1, maximum=1.0, value=0.9, step=0.05)
                repetition_penalty = gr.Slider(label="repetition_penalty", minimum=1.0, maximum=2.0, value=1.05, step=0.01)

        gen_btn = gr.Button("生成语音", variant="primary")
        with gr.Row():
            audio_out = gr.Audio(label="生成结果", type="numpy", scale=3)
            status = gr.Textbox(label="状态", lines=2, scale=1)

        def _infer_base(
            text_in: str,
            lang_in: str,
            ref_audio_path: Optional[str],
            ref_text_in: str,
            xvec_only_in: bool,
            max_new_tokens_in: float,
            temperature_in: float,
            top_k_in: float,
            top_p_in: float,
            repetition_penalty_in: float,
        ) -> Tuple[Optional[Tuple[int, Any]], str]:
            if not ref_audio_path:
                return None, "请先上传参考音频"
            if not xvec_only_in and not (ref_text_in or "").strip():
                return None, "ICL 模式需要参考文本"

            gen_kwargs = _collect_gen_kwargs(
                _coerce_int(max_new_tokens_in, 1024),
                _coerce_float(temperature_in, 0.8),
                _coerce_int(top_k_in, 50),
                _coerce_float(top_p_in, 0.9),
                _coerce_float(repetition_penalty_in, 1.05),
            )
            wavs, sr = tts.generate_voice_clone(
                text=text_in,
                language=_maybe_auto_language(lang_in),
                ref_audio=ref_audio_path,
                ref_text=ref_text_in,
                x_vector_only_mode=bool(xvec_only_in),
                **gen_kwargs,
            )
            saved_path = ""
            if save_audio:
                saved_path = _save_wav(wavs[0], sr, output_dir, "base")
            status_msg = f"OK{f' | Saved: {saved_path}' if saved_path else ''}"
            return (sr, wavs[0]), status_msg

        gen_btn.click(
            _infer_base,
            inputs=[
                text,
                language,
                ref_audio,
                ref_text,
                xvec_only,
                max_new_tokens,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
            ],
            outputs=[audio_out, status],
        )


def _build_custom_ui(tts: Qwen3TTSModel, output_dir: str, save_audio: bool):
    speakers = tts.model.get_supported_speakers() or []
    with gr.Tab("预设角色 (CustomVoice)"):
        with gr.Row():
            with gr.Column(scale=2):
                text = gr.Textbox(label="合成文本", lines=4, placeholder="请输入要合成的文本...", info="支持中英文混合")
                language = gr.Dropdown(label="语言", choices=["Auto", "Chinese", "English"], value="Auto", allow_custom_value=True)
            with gr.Column(scale=1):
                speaker = gr.Dropdown(label="选择角色", choices=speakers, value=speakers[0] if speakers else None, info="模型内置的预设说话人")
                instruct = gr.Textbox(label="风格指令 (可选)", lines=2, placeholder="例如: 开心地、悄悄地、快速...")

        with gr.Accordion("生成参数", open=False):
            with gr.Row():
                max_new_tokens = gr.Slider(label="max_new_tokens", minimum=256, maximum=4096, value=1024, step=64)
                temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.05)
            with gr.Row():
                top_k = gr.Slider(label="top_k", minimum=1, maximum=100, value=50, step=1)
                top_p = gr.Slider(label="top_p", minimum=0.1, maximum=1.0, value=0.9, step=0.05)
                repetition_penalty = gr.Slider(label="repetition_penalty", minimum=1.0, maximum=2.0, value=1.05, step=0.01)

        gen_btn = gr.Button("生成语音", variant="primary")
        with gr.Row():
            audio_out = gr.Audio(label="生成结果", type="numpy", scale=3)
            status = gr.Textbox(label="状态", lines=2, scale=1)

        def _infer_custom(
            text_in: str,
            lang_in: str,
            speaker_in: str,
            instruct_in: str,
            max_new_tokens_in: float,
            temperature_in: float,
            top_k_in: float,
            top_p_in: float,
            repetition_penalty_in: float,
        ) -> Tuple[Optional[Tuple[int, Any]], str]:
            if not speaker_in:
                return None, "请选择 speaker"

            gen_kwargs = _collect_gen_kwargs(
                _coerce_int(max_new_tokens_in, 1024),
                _coerce_float(temperature_in, 0.8),
                _coerce_int(top_k_in, 50),
                _coerce_float(top_p_in, 0.9),
                _coerce_float(repetition_penalty_in, 1.05),
            )
            wavs, sr = tts.generate_custom_voice(
                text=text_in,
                language=_maybe_auto_language(lang_in),
                speaker=speaker_in,
                instruct=instruct_in,
                **gen_kwargs,
            )
            saved_path = ""
            if save_audio:
                saved_path = _save_wav(wavs[0], sr, output_dir, "custom")
            status_msg = f"OK{f' | Saved: {saved_path}' if saved_path else ''}"
            return (sr, wavs[0]), status_msg

        gen_btn.click(
            _infer_custom,
            inputs=[
                text,
                language,
                speaker,
                instruct,
                max_new_tokens,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
            ],
            outputs=[audio_out, status],
        )


def _build_voice_design_ui(tts: Qwen3TTSModel, output_dir: str, save_audio: bool):
    with gr.Tab("风格设计 (VoiceDesign)"):
        with gr.Row():
            with gr.Column(scale=2):
                text = gr.Textbox(label="合成文本", lines=4, placeholder="请输入要合成的文本...", info="支持中英文混合")
                language = gr.Dropdown(label="语言", choices=["Auto", "Chinese", "English"], value="Auto", allow_custom_value=True)
            with gr.Column(scale=1):
                instruct = gr.Textbox(label="语音风格描述", lines=4, placeholder="描述你想要的声音特点...\n例如:\n- 温柔的女声，语速缓慢\n- 低沉有磁性的男声", info="用自然语言描述声音特点")

        with gr.Accordion("生成参数", open=False):
            with gr.Row():
                max_new_tokens = gr.Slider(label="max_new_tokens", minimum=256, maximum=4096, value=1024, step=64)
                temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.05)
            with gr.Row():
                top_k = gr.Slider(label="top_k", minimum=1, maximum=100, value=50, step=1)
                top_p = gr.Slider(label="top_p", minimum=0.1, maximum=1.0, value=0.9, step=0.05)
                repetition_penalty = gr.Slider(label="repetition_penalty", minimum=1.0, maximum=2.0, value=1.05, step=0.01)

        gen_btn = gr.Button("生成语音", variant="primary")
        with gr.Row():
            audio_out = gr.Audio(label="生成结果", type="numpy", scale=3)
            status = gr.Textbox(label="状态", lines=2, scale=1)

        def _infer_design(
            text_in: str,
            lang_in: str,
            instruct_in: str,
            max_new_tokens_in: float,
            temperature_in: float,
            top_k_in: float,
            top_p_in: float,
            repetition_penalty_in: float,
        ) -> Tuple[Optional[Tuple[int, Any]], str]:
            gen_kwargs = _collect_gen_kwargs(
                _coerce_int(max_new_tokens_in, 1024),
                _coerce_float(temperature_in, 0.8),
                _coerce_int(top_k_in, 50),
                _coerce_float(top_p_in, 0.9),
                _coerce_float(repetition_penalty_in, 1.05),
            )
            t0 = time.time()
            wavs, sr = tts.generate_voice_design(
                text=text_in,
                language=_maybe_auto_language(lang_in),
                instruct=instruct_in,
                **gen_kwargs,
            )
            latency = time.time() - t0
            audio_len = len(wavs[0]) / sr if sr > 0 else 0
            _get_tracker().record(latency, len(text_in), audio_len)

            saved_path = ""
            if save_audio:
                saved_path = _save_wav(wavs[0], sr, output_dir, "design")
            rtf = latency / audio_len if audio_len > 0 else 0
            status_msg = f"OK | {latency:.2f}s | RTF {rtf:.2f}{f' | Saved: {saved_path}' if saved_path else ''}"
            return (sr, wavs[0]), status_msg

        gen_btn.click(
            _infer_design,
            inputs=[
                text,
                language,
                instruct,
                max_new_tokens,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
            ],
            outputs=[audio_out, status],
        )


def _scan_all_cached_models() -> List[Dict[str, Any]]:
    """扫描所有已缓存的 Qwen3-TTS 模型"""
    found = []
    for repo_id in SUPPORTED_MODELS:
        result = _check_model_downloaded(repo_id)
        if result["status"] in ("cached", "local_dir"):
            found.append({
                "repo_id": repo_id,
                "path": result.get("path", ""),
                "status": result["status"],
            })
    return found


def _download_model(repo_id: str, local_dir: Optional[str], progress=gr.Progress()) -> str:
    """下载模型到指定目录"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return "错误: 请先安装 huggingface_hub: pip install huggingface_hub"

    progress(0, desc=f"开始下载 {repo_id}...")

    try:
        kwargs = {
            "repo_id": repo_id,
            "allow_patterns": [
                "*.safetensors",
                "*.bin",
                "*.pt",
                "*.npz",
                "*.json",
                "*.jsonl",
                "*.txt",
                "*.model",
                "*.vocab",
                "*.tiktoken",
                "*.spm",
                "*.sentencepiece",
                "*.merges",
            ],
        }
        if local_dir and local_dir.strip():
            local_path = Path(local_dir).expanduser().resolve()
        else:
            local_path = _get_default_download_dir(repo_id)
        local_path.mkdir(parents=True, exist_ok=True)
        kwargs["local_dir"] = str(local_path)

        progress(0.1, desc="正在下载模型文件...")
        result_path = snapshot_download(**kwargs)
        progress(1.0, desc="下载完成!")
        return f"下载成功!\n模型路径: {result_path}\n\n请重启应用并使用以下命令:\npython jetson_gradio_app.py {result_path}"
    except Exception as e:
        logger.error("Model download failed for %s: %s", repo_id, e, exc_info=True)
        return f"下载失败: {str(e)}"


def build_download_ui() -> gr.Blocks:
    """构建模型下载界面"""
    with gr.Blocks(title="Qwen3-TTS 下载") as demo:
        gr.Markdown("# Qwen3-TTS 模型下载")
        gr.Markdown("检测到本地没有可用的模型，请先下载模型")

        # 显示已缓存的模型
        cached_models = _scan_all_cached_models()
        if cached_models:
            with gr.Accordion("已检测到的本地模型", open=True):
                cached_info = "\n".join([f"- **{m['repo_id']}**\n  路径: `{m['path']}`" for m in cached_models])
                gr.Markdown(cached_info)
                gr.Markdown("可以使用以下命令直接启动:")
                for m in cached_models:
                    gr.Code(f"python jetson_gradio_app.py {m['path']}", language="bash")

        # 下载新模型
        with gr.Accordion("下载新模型", open=not cached_models):
            with gr.Group():
                model_choice = gr.Dropdown(
                    label="选择模型",
                    choices=SUPPORTED_MODELS,
                    value=SUPPORTED_MODELS[0],
                    info="0.6B 轻量快速，1.7B 质量更高 | Base=语音克隆, CustomVoice=预设角色, VoiceDesign=风格描述(仅1.7B)"
                )

                default_root = _get_default_download_root()
                gr.Markdown(f"📁 **下载目录**: `{default_root}`")

                use_custom_dir = gr.Checkbox(label="使用自定义下载目录", value=False)
                custom_dir = gr.Textbox(
                    label="自定义目录",
                    placeholder=f"例如: ~/models/Qwen3-TTS-0.6B",
                    visible=False
                )

            def toggle_custom_dir(use_custom):
                return gr.update(visible=use_custom)

            use_custom_dir.change(toggle_custom_dir, inputs=[use_custom_dir], outputs=[custom_dir])

            download_btn = gr.Button("开始下载", variant="primary")
            download_status = gr.Textbox(label="下载状态", lines=5, interactive=False)

            def do_download(model, use_custom, custom_path, progress=gr.Progress()):
                local_dir = custom_path if use_custom and custom_path.strip() else None
                return _download_model(model, local_dir, progress)

            download_btn.click(
                do_download,
                inputs=[model_choice, use_custom_dir, custom_dir],
                outputs=[download_status]
            )

        # 手动指定模型路径
        with gr.Accordion("手动指定本地模型路径", open=False):
            gr.Markdown("如果模型已经下载到其他位置，可以直接指定路径:")
            with gr.Row():
                manual_path = gr.Textbox(
                    label="模型路径",
                    placeholder="例如: /path/to/Qwen3-TTS-12Hz-0.6B-Base",
                    scale=3
                )
                check_btn = gr.Button("检查", scale=1)
            check_result = gr.Textbox(label="检查结果", lines=3, interactive=False)

            def check_manual_path(path):
                if not path or not path.strip():
                    return "请输入路径"
                result = _check_model_downloaded(path.strip())
                if result["status"] in ("local_dir", "cached"):
                    return f"✅ 检测到有效模型!\n路径: {result.get('path', path)}\n\n启动命令:\npython jetson_gradio_app.py {result.get('path', path)}"
                return f"❌ 未检测到有效模型\n状态: {result['status']}\n错误: {result.get('error', '路径不存在或缺少必要文件')}"

            check_btn.click(check_manual_path, inputs=[manual_path], outputs=[check_result])

        gr.Markdown("---")
        with gr.Accordion("使用说明", open=False):
            gr.Markdown("""
**模型类型说明:**
- **Base**: 语音克隆模型，需要参考音频
- **CustomVoice**: 预定义说话人模型
- **VoiceDesign**: 通过文字描述控制语音风格

**参数规模说明:**
- **0.6B**: 轻量版本，推理更快，适合 Jetson 部署（仅 Base 和 CustomVoice）
- **1.7B**: 旗舰版本，质量更高（包含 VoiceDesign）

下载完成后，使用显示的命令重启应用。
            """)

    return demo


def build_demo(
    tts: Qwen3TTSModel,
    checkpoint: str,
    output_dir: str,
    save_audio: bool,
    force_model_type: Optional[str] = None,
) -> gr.Blocks:
    with gr.Blocks(title="Qwen3-TTS") as demo:
        gr.Markdown("# Qwen3-TTS Jetson Orin")
        gr.Markdown("文本转语音演示 | Text-to-Speech Demo")

        model_type = (force_model_type or getattr(tts.model, "tts_model_type", "")).strip()
        if model_type == "base":
            _build_base_ui(tts, output_dir, save_audio)
        elif model_type == "custom_voice":
            _build_custom_ui(tts, output_dir, save_audio)
        elif model_type == "voice_design":
            _build_voice_design_ui(tts, output_dir, save_audio)
        else:
            gr.Markdown(f"⚠️ 不支持的模型类型: {model_type}")

        with gr.Accordion("系统监控 (System Monitor)", open=False):
            with gr.Row():
                with gr.Column(scale=2):
                    sys_info = gr.Textbox(
                        label="系统状态",
                        lines=10,
                        value=_system_check_summary(checkpoint, output_dir, model=tts),
                    )
                with gr.Column(scale=1):
                    quant_info = gr.Textbox(
                        label="量化 & 推理",
                        lines=10,
                        value=_check_quantization_status(tts) + "\n\n" + _get_tracker().format_summary(),
                    )
            with gr.Row():
                auto_refresh_cb = gr.Checkbox(label="自动刷新", value=False)
                refresh_btn = gr.Button("手动刷新")

            def _refresh() -> Tuple[str, str]:
                sys_text = _system_check_summary(checkpoint, output_dir, model=tts)
                quant_text = _check_quantization_status(tts) + "\n\n" + _get_tracker().format_summary()
                return sys_text, quant_text

            refresh_btn.click(_refresh, outputs=[sys_info, quant_info])

            # Auto-refresh via Timer if Gradio supports it
            try:
                timer = gr.Timer(value=5, active=False)

                def _auto_tick() -> Tuple[str, str]:
                    return _refresh()

                timer.tick(_auto_tick, outputs=[sys_info, quant_info])

                def _toggle_timer(enabled: bool):
                    return gr.Timer(active=enabled)

                auto_refresh_cb.change(_toggle_timer, inputs=[auto_refresh_cb], outputs=[timer])
            except (AttributeError, TypeError):
                logger.debug("Gradio Timer not supported in this version, auto-refresh disabled")

        gr.Markdown("---")
        gr.Markdown(
            "<center style='color: #888; font-size: 0.85em;'>"
            "⚠️ 生成的音频仅供演示使用，请勿用于非法或有害用途。"
            "</center>"
        )
    return demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3-TTS Gradio demo for Jetson Orin.")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=None,
        help="Model checkpoint path or HuggingFace repo id. If not provided, will auto-detect or show download UI.",
    )
    parser.add_argument("--device", default="cpu", help="Device for device_map (default: cpu).")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="Torch dtype for loading the model (default: float32).",
    )
    parser.add_argument(
        "--no-flash-attn",
        dest="no_flash_attn",
        action="store_true",
        default=True,
        help="Disable FlashAttention-2 (default: disabled).",
    )
    parser.add_argument(
        "--flash-attn",
        dest="no_flash_attn",
        action="store_false",
        help="Enable FlashAttention-2.",
    )
    parser.add_argument(
        "--staged-load",
        action="store_true",
        default=True,
        help="Staged load for Jetson (CPU -> dtype -> GPU).",
    )
    parser.add_argument(
        "--no-staged-load",
        dest="staged_load",
        action="store_false",
        help="Disable staged load.",
    )
    parser.add_argument(
        "--tokenizer-on-cpu",
        action="store_true",
        default=True,
        help="Keep speech tokenizer on CPU (reduce GPU memory).",
    )
    parser.add_argument(
        "--tokenizer-on-gpu",
        dest="tokenizer_on_cpu",
        action="store_false",
        help="Move speech tokenizer to GPU.",
    )
    parser.add_argument("--ip", default="0.0.0.0", help="Gradio server bind IP.")
    parser.add_argument("--port", type=int, default=8000, help="Gradio server port.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link.")
    parser.add_argument("--concurrency", type=int, default=4, help="Gradio queue concurrency.")
    parser.add_argument("--ssl-certfile", default=None, help="Path to SSL cert file for HTTPS.")
    parser.add_argument("--ssl-keyfile", default=None, help="Path to SSL key file for HTTPS.")
    parser.add_argument("--output-dir", default="outputs", help="Directory to save generated audio.")
    parser.add_argument("--no-save", action="store_true", help="Disable saving generated audio.")
    parser.add_argument(
        "--auto-detect",
        action="store_true",
        help="Auto-detect and use the first available cached model.",
    )
    parser.add_argument(
        "--backend",
        choices=["torch", "trt"],
        default="torch",
        help="Inference backend. Use 'trt' for TensorRT-LLM INT4 engine.",
    )
    parser.add_argument(
        "--engine-path",
        default=None,
        help="TensorRT-LLM engine path (required when --backend trt).",
    )
    parser.add_argument(
        "--tokenizer-dir",
        default=None,
        help="Tokenizer directory (required when --backend trt).",
    )
    parser.add_argument(
        "--model-type",
        choices=["voice_design"],
        default=None,
        help="Force model type when backend doesn't expose it.",
    )
    return parser


def _scan_local_models() -> List[Dict[str, Any]]:
    """扫描当前目录下已下载的模型"""
    found = []
    for repo_id in SUPPORTED_MODELS:
        local_dir = _get_default_download_dir(repo_id)
        if local_dir.exists():
            has_config, has_weights = _validate_model_dir(local_dir)
            if has_config and has_weights:
                found.append({
                    "repo_id": repo_id,
                    "path": str(local_dir),
                    "status": "local_dir",
                })
    return found


def _scan_models_in_directory(directory: str) -> List[Dict[str, Any]]:
    """扫描指定目录下的所有有效模型"""
    found = []
    if not directory or not directory.strip():
        return found

    dir_path = Path(directory).expanduser().resolve()
    if not dir_path.exists() or not dir_path.is_dir():
        return found

    try:
        # 检查目录本身是否是模型目录
        has_config, has_weights = _validate_model_dir(dir_path)
        if has_config and has_weights:
            found.append({
                "name": dir_path.name,
                "path": str(dir_path),
                "type": _detect_model_type(dir_path),
            })
            return found

        # 扫描子目录
        for sub in sorted(dir_path.iterdir()):
            if sub.is_dir():
                has_config, has_weights = _validate_model_dir(sub)
                if has_config and has_weights:
                    found.append({
                        "name": sub.name,
                        "path": str(sub),
                        "type": _detect_model_type(sub),
                    })
    except (PermissionError, OSError):
        logger.debug("Cannot scan model directory %s", dir_path, exc_info=True)

    return found


def _detect_model_type(model_dir: Path) -> str:
    """检测模型类型 (base/custom_voice/voice_design)"""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return "unknown"

    try:
        import json
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get("tts_model_type", "unknown")
    except Exception:
        logger.debug("Failed to detect model type from %s", config_path, exc_info=True)
        return "unknown"


def _get_all_model_locations() -> List[str]:
    """获取所有可能的模型位置"""
    locations = []

    # 1. 当前工作目录
    cwd = str(Path.cwd().resolve())
    locations.append(cwd)

    # 2. 环境变量指定的目录
    env_root = os.getenv("QWEN3_TTS_DOWNLOAD_DIR")
    if env_root:
        locations.append(str(Path(env_root).expanduser().resolve()))

    # 3. HuggingFace 缓存目录
    for cache_dir in _get_hf_cache_dirs():
        locations.append(str(cache_dir))

    # 4. 常见的模型存放位置
    home = Path.home()
    common_dirs = [
        home / "models",
        home / "Models",
        home / ".cache" / "models",
        Path("/models"),
        Path("/data/models"),
    ]
    for d in common_dirs:
        try:
            if d.exists():
                locations.append(str(d.resolve()))
        except (PermissionError, OSError):
            logger.debug("Cannot access model directory %s", d)

    # 去重并保持顺序
    seen = set()
    result = []
    for loc in locations:
        if loc not in seen:
            seen.add(loc)
            result.append(loc)

    return result


def build_lazy_demo(args: argparse.Namespace) -> gr.Blocks:
    """构建延迟加载模型的界面 - 先启动UI，后加载模型"""
    if args.backend != "torch":
        raise ValueError("Lazy demo only supports --backend torch.")
    # 使用可变容器存储模型状态
    state = {"tts": None, "checkpoint": None, "model_type": None, "sanitize_logits": True}
    output_dir = _ensure_output_dir(args.output_dir)
    save_audio = not args.no_save

    # 启动时扫描可能的模型位置
    all_locations = _get_all_model_locations()
    default_location = all_locations[0] if all_locations else str(Path.cwd())

    # 扫描默认位置的模型
    initial_models = _scan_models_in_directory(default_location)
    initial_model_choices = [f"{m['name']} ({m['type']})" for m in initial_models]
    initial_model_paths = {f"{m['name']} ({m['type']})": m['path'] for m in initial_models}

    with gr.Blocks(title="Qwen3-TTS") as demo:
        gr.Markdown("# Qwen3-TTS Jetson Orin")
        gr.Markdown("文本转语音演示 | Text-to-Speech Demo")

        # 用于存储模型路径映射的状态
        model_paths_state = gr.State(initial_model_paths)

        # ===== 区域1: 模型下载 =====
        with gr.Accordion("📥 模型下载", open=False):
            gr.Markdown("从 HuggingFace 下载 Qwen3-TTS 模型到本地")

            with gr.Row():
                with gr.Column(scale=2):
                    download_model_dropdown = gr.Dropdown(
                        label="选择要下载的模型",
                        choices=SUPPORTED_MODELS,
                        value=SUPPORTED_MODELS[0],
                        info="0.6B 轻量快速，1.7B 质量更高 | Base=语音克隆, CustomVoice=预设角色, VoiceDesign=风格描述(仅1.7B)"
                    )
                with gr.Column(scale=2):
                    download_dir_input = gr.Textbox(
                        label="下载位置",
                        value=default_location,
                        placeholder="模型将下载到此目录",
                        info="模型会保存在该目录下的子文件夹中"
                    )
                with gr.Column(scale=1):
                    download_btn = gr.Button("开始下载", variant="primary", size="lg")

            download_status = gr.Textbox(
                label="下载状态",
                lines=2,
                interactive=False,
                value="选择模型和下载位置后，点击「开始下载」"
            )

            def do_download(repo_id: str, download_dir: str, progress=gr.Progress()):
                if not download_dir.strip():
                    return "请指定下载位置"
                # 构建目标路径
                safe_name = repo_id.replace("/", "__")
                target_path = str(Path(download_dir).expanduser().resolve() / safe_name)
                return _download_model(repo_id, target_path, progress)

            download_btn.click(
                do_download,
                inputs=[download_model_dropdown, download_dir_input],
                outputs=[download_status]
            )

        # ===== 区域2: 模型选择与加载 =====
        with gr.Accordion("🔧 模型选择与加载", open=True):
            with gr.Row():
                # 左侧: 模型位置和选择
                with gr.Column(scale=2):
                    with gr.Group():
                        gr.Markdown("### 选择模型")

                        model_location_dropdown = gr.Dropdown(
                            label="模型位置",
                            choices=all_locations,
                            value=default_location,
                            allow_custom_value=True,
                            info="选择或输入模型所在的目录"
                        )

                        with gr.Row():
                            scan_btn = gr.Button("🔍 扫描", scale=1)
                            auto_detect_btn = gr.Button("🔄 自动检测全部位置", scale=2)

                        model_select_dropdown = gr.Dropdown(
                            label="检测到的模型",
                            choices=initial_model_choices,
                            value=initial_model_choices[0] if initial_model_choices else None,
                            info="选择要加载的模型"
                        )

                        model_path_display = gr.Textbox(
                            label="模型完整路径",
                            value=initial_models[0]['path'] if initial_models else "",
                            interactive=True,
                            info="可直接编辑路径，或通过上方下拉框选择"
                        )

                # 右侧: 加载选项
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("### 加载选项")

                        device_input = gr.Dropdown(
                            label="Device",
                            choices=["cpu", "cuda", "cuda:0", "auto"],
                            value=args.device,
                            allow_custom_value=True,
                        )
                        dtype_dropdown = gr.Dropdown(
                            label="精度 (DType)",
                            choices=["float16", "bfloat16", "float32"],
                            value="float16" if args.dtype in ["fp16", "float16"] else args.dtype,
                        )

                        with gr.Row():
                            flash_attn_checkbox = gr.Checkbox(
                                label="FlashAttn",
                                value=not args.no_flash_attn,
                            )
                            sanitize_logits_checkbox = gr.Checkbox(
                                label="Sanitize",
                                value=True,
                            )

                        with gr.Row():
                            staged_load_checkbox = gr.Checkbox(
                                label="Staged",
                                value=bool(args.staged_load),
                            )
                            tokenizer_cpu_checkbox = gr.Checkbox(
                                label="Tok CPU",
                                value=bool(args.tokenizer_on_cpu),
                            )

            # 加载按钮
            load_btn = gr.Button("🚀 加载模型", variant="primary", size="lg")

            # 状态显示
            model_status_text = gr.Textbox(
                label="状态",
                lines=2,
                interactive=False,
                value=f"检测到 {len(initial_models)} 个模型。选择模型后点击「加载模型」" if initial_models else "未检测到模型。请先下载或指定模型位置。"
            )

            # === 事件处理 ===

            # 扫描指定位置的模型
            def scan_location(location: str, current_paths: dict):
                models = _scan_models_in_directory(location)
                if not models:
                    return (
                        gr.update(choices=[], value=None),
                        "",
                        current_paths,
                        f"在 {location} 未找到有效模型"
                    )

                choices = [f"{m['name']} ({m['type']})" for m in models]
                paths = {f"{m['name']} ({m['type']})": m['path'] for m in models}

                return (
                    gr.update(choices=choices, value=choices[0]),
                    models[0]['path'],
                    paths,
                    f"找到 {len(models)} 个模型"
                )

            scan_btn.click(
                scan_location,
                inputs=[model_location_dropdown, model_paths_state],
                outputs=[model_select_dropdown, model_path_display, model_paths_state, model_status_text]
            )

            # 位置变化时自动扫描
            model_location_dropdown.change(
                scan_location,
                inputs=[model_location_dropdown, model_paths_state],
                outputs=[model_select_dropdown, model_path_display, model_paths_state, model_status_text]
            )

            # 自动检测全部位置
            def auto_detect_all(current_paths: dict):
                all_models = []
                for loc in _get_all_model_locations():
                    models = _scan_models_in_directory(loc)
                    for m in models:
                        m['location'] = loc
                    all_models.extend(models)

                # 同时扫描 HuggingFace 缓存
                for repo_id in SUPPORTED_MODELS:
                    result = _check_model_downloaded(repo_id)
                    if result["status"] in ("cached", "local_dir"):
                        path = result.get("path", "")
                        if path and not any(m['path'] == path for m in all_models):
                            all_models.append({
                                "name": repo_id.split("/")[-1],
                                "path": path,
                                "type": _detect_model_type(Path(path)),
                                "location": "HuggingFace Cache"
                            })

                if not all_models:
                    return (
                        gr.update(choices=[], value=None),
                        "",
                        current_paths,
                        "未找到任何模型。请先下载模型。"
                    )

                choices = [f"{m['name']} ({m['type']}) @ {m.get('location', 'local')}" for m in all_models]
                paths = {c: m['path'] for c, m in zip(choices, all_models)}

                return (
                    gr.update(choices=choices, value=choices[0]),
                    all_models[0]['path'],
                    paths,
                    f"共找到 {len(all_models)} 个模型"
                )

            auto_detect_btn.click(
                auto_detect_all,
                inputs=[model_paths_state],
                outputs=[model_select_dropdown, model_path_display, model_paths_state, model_status_text]
            )

            # 选择模型时更新路径
            def on_model_select(selection: str, paths: dict):
                if selection and selection in paths:
                    return paths[selection]
                return ""

            model_select_dropdown.change(
                on_model_select,
                inputs=[model_select_dropdown, model_paths_state],
                outputs=[model_path_display]
            )

        # ===== 区域3: TTS 生成 (初始隐藏) =====
        with gr.Column(visible=False) as tts_area:
            gr.Markdown("---")
            gr.Markdown("## 🎙️ 语音合成")

            # Base 模式 UI
            with gr.Tab("语音克隆 (Base)", visible=False) as base_tab:
                with gr.Row():
                    with gr.Column(scale=3):
                        base_text = gr.Textbox(
                            label="合成文本",
                            lines=4,
                            placeholder="请输入要合成的文本...",
                            info="支持中英文混合"
                        )
                        with gr.Row():
                            base_language = gr.Dropdown(
                                label="语言",
                                choices=["Auto", "Chinese", "English"],
                                value="Auto",
                                allow_custom_value=True,
                                scale=1
                            )
                            base_xvec_only = gr.Checkbox(
                                label="仅使用音色特征 (无需参考文本)",
                                value=False,
                                scale=2
                            )
                    with gr.Column(scale=2):
                        base_ref_audio = gr.Audio(
                            label="参考音频",
                            type="filepath",
                            sources=["upload", "microphone"]
                        )
                        base_ref_text = gr.Textbox(
                            label="参考音频文本",
                            lines=2,
                            placeholder="输入参考音频中说的内容...",
                            info="关闭「仅音色」时必填"
                        )

                with gr.Accordion("生成参数", open=False):
                    with gr.Row():
                        base_max_tokens = gr.Slider(label="max_new_tokens", minimum=256, maximum=4096, value=1024, step=64)
                        base_temp = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.05)
                    with gr.Row():
                        base_top_k = gr.Slider(label="top_k", minimum=1, maximum=100, value=50, step=1)
                        base_top_p = gr.Slider(label="top_p", minimum=0.1, maximum=1.0, value=0.9, step=0.05)
                        base_rep_pen = gr.Slider(label="repetition_penalty", minimum=1.0, maximum=2.0, value=1.05, step=0.01)

                base_gen_btn = gr.Button("生成语音", variant="primary")

                with gr.Row():
                    base_audio_out = gr.Audio(label="生成结果", type="numpy", scale=3)
                    base_status = gr.Textbox(label="状态", lines=2, scale=1)

            # CustomVoice 模式 UI
            with gr.Tab("预设角色 (CustomVoice)", visible=False) as custom_tab:
                with gr.Row():
                    with gr.Column(scale=2):
                        custom_text = gr.Textbox(
                            label="合成文本",
                            lines=4,
                            placeholder="请输入要合成的文本...",
                            info="支持中英文混合"
                        )
                        custom_language = gr.Dropdown(
                            label="语言",
                            choices=["Auto", "Chinese", "English"],
                            value="Auto",
                            allow_custom_value=True
                        )
                    with gr.Column(scale=1):
                        custom_speaker = gr.Dropdown(
                            label="选择角色",
                            choices=[],
                            value=None,
                            info="模型内置的预设说话人"
                        )
                        custom_instruct = gr.Textbox(
                            label="风格指令 (可选)",
                            lines=2,
                            placeholder="例如: 开心地、悄悄地、快速..."
                        )

                with gr.Accordion("生成参数", open=False):
                    with gr.Row():
                        custom_max_tokens = gr.Slider(label="max_new_tokens", minimum=256, maximum=4096, value=1024, step=64)
                        custom_temp = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.05)
                    with gr.Row():
                        custom_top_k = gr.Slider(label="top_k", minimum=1, maximum=100, value=50, step=1)
                        custom_top_p = gr.Slider(label="top_p", minimum=0.1, maximum=1.0, value=0.9, step=0.05)
                        custom_rep_pen = gr.Slider(label="repetition_penalty", minimum=1.0, maximum=2.0, value=1.05, step=0.01)

                custom_gen_btn = gr.Button("生成语音", variant="primary")

                with gr.Row():
                    custom_audio_out = gr.Audio(label="生成结果", type="numpy", scale=3)
                    custom_status = gr.Textbox(label="状态", lines=2, scale=1)

            # VoiceDesign 模式 UI
            with gr.Tab("风格设计 (VoiceDesign)", visible=False) as design_tab:
                with gr.Row():
                    with gr.Column(scale=2):
                        design_text = gr.Textbox(
                            label="合成文本",
                            lines=4,
                            placeholder="请输入要合成的文本...",
                            info="支持中英文混合"
                        )
                        design_language = gr.Dropdown(
                            label="语言",
                            choices=["Auto", "Chinese", "English"],
                            value="Auto",
                            allow_custom_value=True
                        )
                    with gr.Column(scale=1):
                        design_instruct = gr.Textbox(
                            label="语音风格描述",
                            lines=4,
                            placeholder="描述你想要的声音特点...\n例如:\n- 温柔的女声，语速缓慢\n- 低沉有磁性的男声\n- 活泼的播音腔",
                            info="用自然语言描述声音特点"
                        )

                with gr.Accordion("生成参数", open=False):
                    with gr.Row():
                        design_max_tokens = gr.Slider(label="max_new_tokens", minimum=256, maximum=4096, value=1024, step=64)
                        design_temp = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.05)
                    with gr.Row():
                        design_top_k = gr.Slider(label="top_k", minimum=1, maximum=100, value=50, step=1)
                        design_top_p = gr.Slider(label="top_p", minimum=0.1, maximum=1.0, value=0.9, step=0.05)
                        design_rep_pen = gr.Slider(label="repetition_penalty", minimum=1.0, maximum=2.0, value=1.05, step=0.01)

                design_gen_btn = gr.Button("生成语音", variant="primary")

                with gr.Row():
                    design_audio_out = gr.Audio(label="生成结果", type="numpy", scale=3)
                    design_status = gr.Textbox(label="状态", lines=2, scale=1)

            # 系统监控面板
            with gr.Accordion("系统监控 (System Monitor)", open=False):
                with gr.Row():
                    with gr.Column(scale=2):
                        sys_info = gr.Textbox(label="系统状态", lines=10, value="")
                    with gr.Column(scale=1):
                        quant_info = gr.Textbox(label="量化 & 推理", lines=10, value="")
                with gr.Row():
                    auto_refresh_cb = gr.Checkbox(label="自动刷新", value=False)
                    refresh_btn = gr.Button("手动刷新")

        gr.Markdown("---")
        gr.Markdown(
            "<center style='color: #888; font-size: 0.85em;'>"
            "⚠️ 生成的音频仅供演示使用，请勿用于非法或有害用途。"
            "</center>"
        )

        # ===== 加载模型逻辑 =====
        def load_model_fn(model_path: str, device_in: str, dtype_in: str, flash_attn_in: bool, sanitize_logits_in: bool, staged_load_in: bool, tokenizer_cpu_in: bool):
            nonlocal state

            if not model_path or not model_path.strip():
                return (
                    "请先选择或输入模型路径",
                    gr.update(visible=False),
                    "",
                    "",
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(choices=[], value=None),
                )

            try:
                checkpoint = model_path.strip()

                # 验证模型路径
                model_dir = Path(checkpoint)
                if model_dir.exists():
                    has_config, has_weights = _validate_model_dir(model_dir)
                    if not has_config or not has_weights:
                        return (
                            f"无效的模型路径: {checkpoint}\n缺少 config.json 或权重文件",
                            gr.update(visible=False),
                            "",
                            "",
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(choices=[], value=None),
                        )

                # 应用加载参数
                args.device = (device_in or "").strip() or args.device
                args.dtype = (dtype_in or "").strip() or args.dtype
                args.no_flash_attn = not bool(flash_attn_in)
                args.staged_load = bool(staged_load_in)
                args.tokenizer_on_cpu = bool(tokenizer_cpu_in)

                if args.device.lower().startswith("cpu") and args.dtype.lower() in ("bfloat16", "bf16", "float16", "fp16"):
                    logger.info("CPU 模式下将 dtype 自动切换为 float32，以避免 NaN/Inf")
                    args.dtype = "float32"

                state["sanitize_logits"] = bool(sanitize_logits_in)

                # 加载模型
                args.checkpoint = checkpoint
                tts = _load_tts(args)
                model_type = getattr(tts.model, "tts_model_type", "")
                logger.info("model_type=%s | model_device=%s", model_type, _infer_model_device(tts.model))

                try:
                    st = getattr(tts.model, "speech_tokenizer", None)
                    st_model = getattr(st, "model", None) if st is not None else None
                    if st_model is not None:
                        logger.info("speech_tokenizer_device=%s", _infer_model_device(st_model))
                except Exception:
                    logger.debug("Failed to query speech_tokenizer device", exc_info=True)

                state["tts"] = tts
                state["checkpoint"] = checkpoint
                state["model_type"] = model_type

                # 获取 speakers (仅 custom_voice 模式)
                speakers = []
                if model_type == "custom_voice":
                    speakers = tts.model.get_supported_speakers() or []

                status_msg = (
                    f"✅ 模型加载成功!\n"
                    f"路径: {checkpoint}\n"
                    f"类型: {model_type} | device={args.device} | dtype={args.dtype}"
                )
                sys_check = _system_check_summary(checkpoint, output_dir, model=tts)

                quant_check = _check_quantization_status(tts) + "\n\n" + _get_tracker().format_summary()

                # 返回 UI 更新
                return (
                    status_msg,  # model_status_text
                    gr.update(visible=True),  # tts_area
                    sys_check,  # sys_info
                    quant_check,  # quant_info
                    gr.update(visible=(model_type == "base")),  # base_tab
                    gr.update(visible=(model_type == "custom_voice")),  # custom_tab
                    gr.update(visible=(model_type == "voice_design")),  # design_tab
                    gr.update(choices=speakers, value=speakers[0] if speakers else None),  # custom_speaker
                )
            except Exception as e:
                logger.error("模型加载失败: %s", e, exc_info=True)
                error_msg = f"❌ 模型加载失败: {str(e)}"
                return (
                    error_msg,
                    gr.update(visible=False),
                    "",
                    "",
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(choices=[], value=None),
                )

        load_btn.click(
            load_model_fn,
            inputs=[model_path_display, device_input, dtype_dropdown, flash_attn_checkbox, sanitize_logits_checkbox, staged_load_checkbox, tokenizer_cpu_checkbox],
            outputs=[model_status_text, tts_area, sys_info, quant_info, base_tab, custom_tab, design_tab, custom_speaker]
        )

        # 刷新系统检查
        def refresh_sys_check():
            tts_obj = state.get("tts")
            if state["checkpoint"]:
                sys_text = _system_check_summary(state["checkpoint"], output_dir, model=tts_obj)
            else:
                sys_text = "模型未加载"
            quant_text = _check_quantization_status(tts_obj) + "\n\n" + _get_tracker().format_summary()
            return sys_text, quant_text

        refresh_btn.click(refresh_sys_check, outputs=[sys_info, quant_info])

        # Auto-refresh via Timer if Gradio supports it
        try:
            lazy_timer = gr.Timer(value=5, active=False)
            lazy_timer.tick(refresh_sys_check, outputs=[sys_info, quant_info])

            def _toggle_lazy_timer(enabled: bool):
                return gr.Timer(active=enabled)

            auto_refresh_cb.change(_toggle_lazy_timer, inputs=[auto_refresh_cb], outputs=[lazy_timer])
        except (AttributeError, TypeError):
            logger.debug("Gradio Timer not supported in this version, auto-refresh disabled")

        # ===== TTS 生成函数 =====
        def infer_base(text_in, lang_in, ref_audio_path, ref_text_in, xvec_only_in,
                       max_new_tokens_in, temperature_in, top_k_in, top_p_in, repetition_penalty_in):
            if state["tts"] is None:
                return None, "请先加载模型"
            if not ref_audio_path:
                return None, "请先上传参考音频"
            if not xvec_only_in and not (ref_text_in or "").strip():
                return None, "ICL 模式需要参考文本"

            gen_kwargs = _collect_gen_kwargs(
                _coerce_int(max_new_tokens_in, 1024),
                _coerce_float(temperature_in, 0.8),
                _coerce_int(top_k_in, 50),
                _coerce_float(top_p_in, 0.9),
                _coerce_float(repetition_penalty_in, 1.05),
            )
            logits_processor = _build_logits_processor(state.get("sanitize_logits", False))
            if logits_processor is not None:
                gen_kwargs["logits_processor"] = logits_processor
                gen_kwargs["subtalker_dosample"] = False
            wavs, sr = state["tts"].generate_voice_clone(
                text=text_in,
                language=_maybe_auto_language(lang_in),
                ref_audio=ref_audio_path,
                ref_text=ref_text_in,
                x_vector_only_mode=bool(xvec_only_in),
                **gen_kwargs,
            )
            saved_path = ""
            if save_audio:
                saved_path = _save_wav(wavs[0], sr, output_dir, "base")
            status_msg = f"OK{f' | Saved: {saved_path}' if saved_path else ''}"
            return (sr, wavs[0]), status_msg

        base_gen_btn.click(
            infer_base,
            inputs=[base_text, base_language, base_ref_audio, base_ref_text, base_xvec_only,
                    base_max_tokens, base_temp, base_top_k, base_top_p, base_rep_pen],
            outputs=[base_audio_out, base_status]
        )

        def infer_custom(text_in, lang_in, speaker_in, instruct_in,
                         max_new_tokens_in, temperature_in, top_k_in, top_p_in, repetition_penalty_in):
            if state["tts"] is None:
                return None, "请先加载模型"
            if not speaker_in:
                return None, "请选择 speaker"

            gen_kwargs = _collect_gen_kwargs(
                _coerce_int(max_new_tokens_in, 1024),
                _coerce_float(temperature_in, 0.8),
                _coerce_int(top_k_in, 50),
                _coerce_float(top_p_in, 0.9),
                _coerce_float(repetition_penalty_in, 1.05),
            )
            logits_processor = _build_logits_processor(state.get("sanitize_logits", False))
            if logits_processor is not None:
                gen_kwargs["logits_processor"] = logits_processor
                gen_kwargs["subtalker_dosample"] = False
            wavs, sr = state["tts"].generate_custom_voice(
                text=text_in,
                language=_maybe_auto_language(lang_in),
                speaker=speaker_in,
                instruct=instruct_in,
                **gen_kwargs,
            )
            saved_path = ""
            if save_audio:
                saved_path = _save_wav(wavs[0], sr, output_dir, "custom")
            status_msg = f"OK{f' | Saved: {saved_path}' if saved_path else ''}"
            return (sr, wavs[0]), status_msg

        custom_gen_btn.click(
            infer_custom,
            inputs=[custom_text, custom_language, custom_speaker, custom_instruct,
                    custom_max_tokens, custom_temp, custom_top_k, custom_top_p, custom_rep_pen],
            outputs=[custom_audio_out, custom_status]
        )

        def infer_design(text_in, lang_in, instruct_in,
                         max_new_tokens_in, temperature_in, top_k_in, top_p_in, repetition_penalty_in):
            if state["tts"] is None:
                return None, "请先加载模型"

            gen_kwargs = _collect_gen_kwargs(
                _coerce_int(max_new_tokens_in, 1024),
                _coerce_float(temperature_in, 0.8),
                _coerce_int(top_k_in, 50),
                _coerce_float(top_p_in, 0.9),
                _coerce_float(repetition_penalty_in, 1.05),
            )
            logits_processor = _build_logits_processor(state.get("sanitize_logits", False))
            if logits_processor is not None:
                gen_kwargs["logits_processor"] = logits_processor
                gen_kwargs["subtalker_dosample"] = False
            t0 = time.time()
            wavs, sr = state["tts"].generate_voice_design(
                text=text_in,
                language=_maybe_auto_language(lang_in),
                instruct=instruct_in,
                **gen_kwargs,
            )
            latency = time.time() - t0
            audio_len = len(wavs[0]) / sr if sr > 0 else 0
            _get_tracker().record(latency, len(text_in), audio_len)

            saved_path = ""
            if save_audio:
                saved_path = _save_wav(wavs[0], sr, output_dir, "design")
            rtf = latency / audio_len if audio_len > 0 else 0
            status_msg = f"OK | {latency:.2f}s | RTF {rtf:.2f}{f' | Saved: {saved_path}' if saved_path else ''}"
            return (sr, wavs[0]), status_msg

        design_gen_btn.click(
            infer_design,
            inputs=[design_text, design_language, design_instruct,
                    design_max_tokens, design_temp, design_top_k, design_top_p, design_rep_pen],
            outputs=[design_audio_out, design_status]
        )

    return demo


def _get_local_ip() -> str:
    """获取本机局域网 IP 地址"""
    import socket
    try:
        # 创建一个 UDP socket 连接到外部地址（不实际发送数据）
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        logger.debug("Cannot detect LAN IP, falling back to 127.0.0.1", exc_info=True)
        return "127.0.0.1"

def _load_trt_tts(args: argparse.Namespace) -> Qwen3TTSModel:
    """Load TensorRT-LLM INT4 engine for Qwen3-TTS."""
    engine_path = (args.engine_path or "").strip()
    tokenizer_dir = (args.tokenizer_dir or "").strip()
    if not engine_path:
        raise ValueError("--engine-path is required when --backend trt")
    if not tokenizer_dir:
        raise ValueError("--tokenizer-dir is required when --backend trt")

    module_name = os.getenv("TRT_QWEN_TTS_MODULE", "trt_qwen_tts")
    logger.info("Loading TRT backend: module=%s, engine=%s, tokenizer=%s",
                module_name, engine_path, tokenizer_dir)
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        logger.error("Failed to import TRT module '%s': %s", module_name, e, exc_info=True)
        raise ImportError(
            f"Failed to import TRT backend module '{module_name}'. "
            "Set TRT_QWEN_TTS_MODULE to your module or ensure trt_qwen_tts.py is available."
        ) from e

    if not hasattr(mod, "TRTQwen3TTSModel"):
        logger.error("Module '%s' does not expose TRTQwen3TTSModel", module_name)
        raise ImportError(f"Module '{module_name}' does not expose TRTQwen3TTSModel")

    cls = getattr(mod, "TRTQwen3TTSModel")
    try:
        tts = cls.from_engine(engine_path=engine_path, tokenizer_dir=tokenizer_dir)
    except Exception as e:
        logger.error("TRT engine loading failed (engine=%s): %s", engine_path, e, exc_info=True)
        raise

    logger.info("TRT engine loaded successfully")

    # Ensure model_type is visible to UI
    if not hasattr(tts, "model"):
        from types import SimpleNamespace
        tts.model = SimpleNamespace(tts_model_type="voice_design")
    else:
        if not getattr(tts.model, "tts_model_type", None):
            setattr(tts.model, "tts_model_type", "voice_design")
    return tts


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Initialize logging early — before any logger.xxx calls
    setup_logging()
    logger.info("Qwen3-TTS Jetson Gradio app starting")

    # Initialize Jetson hardware monitor (background thread)
    _get_monitor()
    _get_tracker()

    launch_kwargs: Dict[str, Any] = dict(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
    )
    if args.ssl_certfile:
        launch_kwargs["ssl_certfile"] = args.ssl_certfile
    if args.ssl_keyfile:
        launch_kwargs["ssl_keyfile"] = args.ssl_keyfile

    # 注释掉自动扫描和加载模型的逻辑
    # 现在直接启动 UI，让用户在界面中选择下载和加载模型
    # checkpoint = args.checkpoint
    #
    # # 如果没有指定 checkpoint，尝试自动检测
    # if checkpoint is None or args.auto_detect:
    #     print("正在扫描本地模型缓存...")
    #     auto_path = _auto_detect_model()
    #     if auto_path:
    #         print(f"检测到已缓存的模型: {auto_path}")
    #         if checkpoint is None:
    #             checkpoint = auto_path
    #     else:
    #         print("未检测到本地模型，将显示下载界面。")
    #
    # # 如果仍然没有 checkpoint，显示下载界面
    # if checkpoint is None:
    #     print("启动模型下载界面...")
    #     demo = build_download_ui()
    #     demo.queue(default_concurrency_limit=int(args.concurrency)).launch(**launch_kwargs)
    #     return 0
    #
    # # 检查指定的 checkpoint 是否有效
    # model_status = _check_model_downloaded(checkpoint)
    # if model_status["status"] in {"local_file", "local_dir_invalid", "cached_invalid", "local_missing"}:
    #     print(f"错误: 指定的模型路径无效: {checkpoint}")
    #     if model_status.get("error"):
    #         print(f"原因: {model_status['error']}")
    #     print("请提供模型目录或 HuggingFace repo id。")
    #     return 1
    # if model_status["status"] == "not_cached":
    #     print(f"警告: 指定的模型 '{checkpoint}' 未在本地找到。")
    #     print("尝试从 HuggingFace 下载模型...")
    #
    # output_dir = _ensure_output_dir(args.output_dir)
    # args.checkpoint = checkpoint  # 更新 args 以便 _load_tts 使用
    # tts = _load_tts(args)
    # demo = build_demo(tts, checkpoint, output_dir, save_audio=not args.no_save)

    if args.backend == "trt":
        logger.info("启动 Qwen3-TTS Gradio 界面 (TRT backend)")
        output_dir = _ensure_output_dir(args.output_dir)
        tts = _load_trt_tts(args)
        demo = build_demo(
            tts,
            checkpoint=args.engine_path or "",
            output_dir=output_dir,
            save_audio=not args.no_save,
            force_model_type=args.model_type or "voice_design",
        )
    else:
        logger.info("启动 Qwen3-TTS Gradio 界面 (torch backend)")
        logger.info("模型将在界面中选择后加载")
        demo = build_lazy_demo(args)

    # 获取并显示访问地址
    local_ip = _get_local_ip()
    protocol = "https" if args.ssl_certfile else "http"
    logger.info("Gradio 服务启动中 — 本机: %s://127.0.0.1:%s  局域网: %s://%s:%s%s",
                protocol, args.port, protocol, local_ip, args.port,
                " (公网分享已启用)" if args.share else "")

    demo.queue(default_concurrency_limit=int(args.concurrency)).launch(**launch_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
