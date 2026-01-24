#!/usr/bin/env python3
# coding=utf-8
"""
Minimal Gradio app for deploying Qwen3-TTS on Jetson Orin.

Supports Base / CustomVoice / VoiceDesign models based on the checkpoint.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import torch

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
    return Path.cwd().resolve()


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


# 支持的 Qwen3-TTS 模型列表
SUPPORTED_MODELS = [
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-0.6B-VoiceDesign",
    "Qwen/Qwen3-TTS-25Hz-0.6B-Base",
    "Qwen/Qwen3-TTS-25Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-25Hz-0.6B-VoiceDesign",
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
        pass

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
        pass

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
        try:
            from scipy.io import wavfile

            wav_int16 = np.clip(wav, -1.0, 1.0)
            wav_int16 = (wav_int16 * 32767.0).astype(np.int16)
            wavfile.write(out_path, sr, wav_int16)
        except Exception as e:
            raise RuntimeError(f"Failed to save audio: {e}") from e
    return out_path


def _system_check_summary(checkpoint: str, output_dir: str) -> str:
    meminfo = _read_meminfo()
    mem_total = meminfo.get("MemTotal")
    mem_avail = meminfo.get("MemAvailable")
    mem_used = mem_total - mem_avail if mem_total and mem_avail else None

    swap = _check_swap()
    cuda = _check_cuda_mem()
    model = _check_model_downloaded(checkpoint)

    lines = []
    lines.append(f"模型检查: {model.get('status')}")
    if model.get("status") in {"local_dir", "local_dir_invalid", "cached_invalid"}:
        lines.append(f"- 路径: {model.get('path')}")
        lines.append(f"- config.json: {model.get('has_config')}")
        lines.append(f"- 权重文件: {model.get('has_weights')}")
        if model.get("error"):
            lines.append(f"- 错误: {model.get('error')}")
    elif model.get("status") in {"local_file", "cached"}:
        lines.append(f"- 路径: {model.get('path')}")
        if model.get("error"):
            lines.append(f"- 错误: {model.get('error')}")
    elif model.get("status") == "not_cached":
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


def _load_tts(args: argparse.Namespace) -> Qwen3TTSModel:
    dtype = _dtype_from_str(args.dtype)
    attn_impl = None if args.no_flash_attn else "flash_attention_2"
    return Qwen3TTSModel.from_pretrained(
        args.checkpoint,
        device_map=args.device,
        dtype=dtype,
        attn_implementation=attn_impl,
    )


def _build_base_ui(tts: Qwen3TTSModel, output_dir: str, save_audio: bool):
    with gr.Tab("Base (Voice Clone)"):
        text = gr.Textbox(label="Text", lines=4, placeholder="请输入要合成的文本")
        language = gr.Textbox(label="Language (Auto/English/Chinese/...)", value="Auto")
        ref_audio = gr.Audio(label="Reference Audio (wav)", type="filepath")
        ref_text = gr.Textbox(label="Reference Text (required if not x-vector only)", lines=3)
        xvec_only = gr.Checkbox(label="X-vector only (no ref text)", value=False)

        with gr.Row():
            max_new_tokens = gr.Number(label="max_new_tokens", value=1024, precision=0)
            temperature = gr.Number(label="temperature", value=0.8)
            top_k = gr.Number(label="top_k", value=50, precision=0)
            top_p = gr.Number(label="top_p", value=0.9)
            repetition_penalty = gr.Number(label="repetition_penalty", value=1.05)

        gen_btn = gr.Button("Generate")
        audio_out = gr.Audio(label="Output", type="numpy")
        status = gr.Textbox(label="Status")

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
    with gr.Tab("CustomVoice"):
        text = gr.Textbox(label="Text", lines=4, placeholder="请输入要合成的文本")
        language = gr.Textbox(label="Language (Auto/English/Chinese/...)", value="Auto")
        speaker = gr.Dropdown(label="Speaker", choices=speakers, value=speakers[0] if speakers else None)
        instruct = gr.Textbox(label="Instruction (optional)", lines=3)

        with gr.Row():
            max_new_tokens = gr.Number(label="max_new_tokens", value=1024, precision=0)
            temperature = gr.Number(label="temperature", value=0.8)
            top_k = gr.Number(label="top_k", value=50, precision=0)
            top_p = gr.Number(label="top_p", value=0.9)
            repetition_penalty = gr.Number(label="repetition_penalty", value=1.05)

        gen_btn = gr.Button("Generate")
        audio_out = gr.Audio(label="Output", type="numpy")
        status = gr.Textbox(label="Status")

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
    with gr.Tab("VoiceDesign"):
        text = gr.Textbox(label="Text", lines=4, placeholder="请输入要合成的文本")
        language = gr.Textbox(label="Language (Auto/English/Chinese/...)", value="Auto")
        instruct = gr.Textbox(label="Instruction", lines=3, placeholder="例如: 温柔、低沉、广播腔")

        with gr.Row():
            max_new_tokens = gr.Number(label="max_new_tokens", value=1024, precision=0)
            temperature = gr.Number(label="temperature", value=0.8)
            top_k = gr.Number(label="top_k", value=50, precision=0)
            top_p = gr.Number(label="top_p", value=0.9)
            repetition_penalty = gr.Number(label="repetition_penalty", value=1.05)

        gen_btn = gr.Button("Generate")
        audio_out = gr.Audio(label="Output", type="numpy")
        status = gr.Textbox(label="Status")

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
            wavs, sr = tts.generate_voice_design(
                text=text_in,
                language=_maybe_auto_language(lang_in),
                instruct=instruct_in,
                **gen_kwargs,
            )
            saved_path = ""
            if save_audio:
                saved_path = _save_wav(wavs[0], sr, output_dir, "design")
            status_msg = f"OK{f' | Saved: {saved_path}' if saved_path else ''}"
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
        return f"下载失败: {str(e)}"


def build_download_ui() -> gr.Blocks:
    """构建模型下载界面"""
    with gr.Blocks(css=".gradio-container {max-width: 100% !important;}") as demo:
        gr.Markdown("# Qwen3-TTS 模型下载器")
        gr.Markdown("检测到本地没有可用的模型，请先下载模型。")

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
            gr.Markdown("### 选择要下载的模型")

            model_choice = gr.Dropdown(
                label="模型",
                choices=SUPPORTED_MODELS,
                value=SUPPORTED_MODELS[0],
                info="选择要下载的 Qwen3-TTS 模型"
            )

            default_root = _get_default_download_root()
            gr.Markdown(f"**默认下载目录**: `{default_root}`")
            gr.Markdown("将自动在当前目录下为每个模型创建子目录。")

            use_custom_dir = gr.Checkbox(label="使用自定义下载目录", value=False)
            custom_dir = gr.Textbox(
                label="自定义目录 (留空使用默认缓存)",
                placeholder=f"例如: ~/models/Qwen3-TTS-0.6B",
                visible=False
            )

            def toggle_custom_dir(use_custom):
                return gr.update(visible=use_custom)

            use_custom_dir.change(toggle_custom_dir, inputs=[use_custom_dir], outputs=[custom_dir])

            download_btn = gr.Button("开始下载", variant="primary")
            download_status = gr.Textbox(label="下载状态", lines=6, interactive=False)

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
            gr.Markdown("如果模型已经下载到其他位置，可以直接指定路径启动:")
            manual_path = gr.Textbox(
                label="模型路径",
                placeholder="例如: /path/to/Qwen3-TTS-12Hz-0.6B-Base"
            )
            check_btn = gr.Button("检查路径")
            check_result = gr.Textbox(label="检查结果", lines=4, interactive=False)

            def check_manual_path(path):
                if not path or not path.strip():
                    return "请输入路径"
                result = _check_model_downloaded(path.strip())
                if result["status"] in ("local_dir", "cached"):
                    return f"检测到有效模型!\n路径: {result.get('path', path)}\n\n启动命令:\npython jetson_gradio_app.py {result.get('path', path)}"
                return f"未检测到有效模型\n状态: {result['status']}\n错误: {result.get('error', '路径不存在或缺少必要文件')}"

            check_btn.click(check_manual_path, inputs=[manual_path], outputs=[check_result])

        gr.Markdown("---")
        gr.Markdown("### 使用说明")
        gr.Markdown("""
1. 选择要下载的模型类型:
   - **Base**: 语音克隆模型，需要参考音频
   - **CustomVoice**: 预定义说话人模型
   - **VoiceDesign**: 通过文字描述控制语音风格

2. **12Hz vs 25Hz**: 12Hz 模型更快，25Hz 模型质量更高

3. 下载完成后，使用显示的命令重启应用
        """)

    return demo


def build_demo(tts: Qwen3TTSModel, checkpoint: str, output_dir: str, save_audio: bool) -> gr.Blocks:
    with gr.Blocks(css=".gradio-container {max-width: 100% !important;}") as demo:
        gr.Markdown("# Qwen3-TTS Jetson Orin Gradio Demo")
        gr.Markdown("模型类型会根据 checkpoint 自动选择对应的界面。")

        with gr.Accordion("System Checks", open=True):
            sys_info = gr.Textbox(label="状态", lines=8, value=_system_check_summary(checkpoint, output_dir))
            refresh_btn = gr.Button("刷新检查")

            def _refresh() -> str:
                return _system_check_summary(checkpoint, output_dir)

            refresh_btn.click(_refresh, outputs=[sys_info])

        model_type = getattr(tts.model, "tts_model_type", "")
        if model_type == "base":
            _build_base_ui(tts, output_dir, save_audio)
        elif model_type == "custom_voice":
            _build_custom_ui(tts, output_dir, save_audio)
        elif model_type == "voice_design":
            _build_voice_design_ui(tts, output_dir, save_audio)
        else:
            gr.Markdown(f"Unsupported model type: {model_type}")

        gr.Markdown(
            "Disclaimer: Generated audio is for demo use only. Do not use for illegal or harmful purposes."
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
    parser.add_argument("--device", default="cuda:0", help="Device for device_map (default: cuda:0).")
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="Torch dtype for loading the model (default: float16).",
    )
    parser.add_argument(
        "--no-flash-attn",
        action="store_true",
        help="Disable FlashAttention-2 (recommended on Jetson).",
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
    return parser


def _auto_detect_model() -> Optional[str]:
    """自动检测已缓存的模型，返回第一个可用的模型路径"""
    cached = _scan_all_cached_models()
    if cached:
        return cached[0]["path"]
    return None


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    launch_kwargs: Dict[str, Any] = dict(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
    )
    if args.ssl_certfile:
        launch_kwargs["ssl_certfile"] = args.ssl_certfile
    if args.ssl_keyfile:
        launch_kwargs["ssl_keyfile"] = args.ssl_keyfile

    checkpoint = args.checkpoint

    # 如果没有指定 checkpoint，尝试自动检测
    if checkpoint is None or args.auto_detect:
        print("正在扫描本地模型缓存...")
        auto_path = _auto_detect_model()
        if auto_path:
            print(f"检测到已缓存的模型: {auto_path}")
            if checkpoint is None:
                checkpoint = auto_path
        else:
            print("未检测到本地模型，将显示下载界面。")

    # 如果仍然没有 checkpoint，显示下载界面
    if checkpoint is None:
        print("启动模型下载界面...")
        demo = build_download_ui()
        demo.queue(default_concurrency_limit=int(args.concurrency)).launch(**launch_kwargs)
        return 0

    # 检查指定的 checkpoint 是否有效
    model_status = _check_model_downloaded(checkpoint)
    if model_status["status"] in {"local_file", "local_dir_invalid", "cached_invalid", "local_missing"}:
        print(f"错误: 指定的模型路径无效: {checkpoint}")
        if model_status.get("error"):
            print(f"原因: {model_status['error']}")
        print("请提供模型目录或 HuggingFace repo id。")
        return 1
    if model_status["status"] == "not_cached":
        print(f"警告: 指定的模型 '{checkpoint}' 未在本地找到。")
        print("尝试从 HuggingFace 下载模型...")

    output_dir = _ensure_output_dir(args.output_dir)
    args.checkpoint = checkpoint  # 更新 args 以便 _load_tts 使用
    tts = _load_tts(args)
    demo = build_demo(tts, checkpoint, output_dir, save_audio=not args.no_save)

    demo.queue(default_concurrency_limit=int(args.concurrency)).launch(**launch_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
