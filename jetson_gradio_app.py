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


def _check_model_downloaded(checkpoint: str) -> Dict[str, Any]:
    if os.path.exists(checkpoint):
        ckpt_path = Path(checkpoint)
        if ckpt_path.is_file():
            return {"status": "local_file", "path": str(ckpt_path)}
        if ckpt_path.is_dir():
            has_config = (ckpt_path / "config.json").exists()
            has_weights = any(ckpt_path.glob("*.safetensors")) or any(ckpt_path.glob("*.bin"))
            return {
                "status": "local_dir",
                "path": str(ckpt_path),
                "has_config": has_config,
                "has_weights": has_weights,
            }
        return {"status": "local_missing", "path": str(ckpt_path)}

    try:
        from huggingface_hub import snapshot_download

        repo_dir = snapshot_download(
            repo_id=checkpoint,
            local_files_only=True,
            allow_patterns=["*.safetensors", "*.bin", "config.json", "generation_config.json", "*.json"],
        )
        return {"status": "cached", "path": repo_dir}
    except Exception as e:
        return {"status": "not_cached", "error": str(e)}


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
    if model.get("status") == "local_dir":
        lines.append(f"- 路径: {model.get('path')}")
        lines.append(f"- config.json: {model.get('has_config')}")
        lines.append(f"- 权重文件: {model.get('has_weights')}")
    elif model.get("status") in {"local_file", "cached"}:
        lines.append(f"- 路径: {model.get('path')}")
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
                int(max_new_tokens_in),
                float(temperature_in),
                int(top_k_in),
                float(top_p_in),
                float(repetition_penalty_in),
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
                int(max_new_tokens_in),
                float(temperature_in),
                int(top_k_in),
                float(top_p_in),
                float(repetition_penalty_in),
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
                int(max_new_tokens_in),
                float(temperature_in),
                int(top_k_in),
                float(top_p_in),
                float(repetition_penalty_in),
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
    parser.add_argument("checkpoint", help="Model checkpoint path or HuggingFace repo id.")
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
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = _ensure_output_dir(args.output_dir)
    tts = _load_tts(args)
    demo = build_demo(tts, args.checkpoint, output_dir, save_audio=not args.no_save)

    launch_kwargs: Dict[str, Any] = dict(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
    )
    if args.ssl_certfile:
        launch_kwargs["ssl_certfile"] = args.ssl_certfile
    if args.ssl_keyfile:
        launch_kwargs["ssl_keyfile"] = args.ssl_keyfile

    demo.queue(default_concurrency_limit=int(args.concurrency)).launch(**launch_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
