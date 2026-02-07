#!/usr/bin/env python3
# coding=utf-8
"""
Qwen3-TTS INT4 weight-only quantization for Jetson Orin.

Primary: torchao int4_weight_only (quantize talker only)
Fallback: torch.ao.quantization.quantize_dynamic INT8

Usage:
    python3 quantize_int4.py --model-dir models/Qwen3-TTS-12Hz-1.7B-VoiceDesign
    python3 quantize_int4.py --model-dir models/Qwen3-TTS-12Hz-1.7B-VoiceDesign --verify
    python3 quantize_int4.py --model-dir models/Qwen3-TTS-12Hz-1.7B-VoiceDesign --method int8
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def _add_local_qwen_repo() -> None:
    """Add local Qwen3-TTS repo to sys.path if available."""
    env_root = os.getenv("QWEN3_TTS_ROOT")
    candidates = []
    if env_root:
        candidates.append(env_root)
    candidates.append(str(Path(__file__).resolve().parent / "Qwen3-TTS"))
    for c in candidates:
        if c and Path(c).is_dir() and c not in sys.path:
            sys.path.insert(0, c)
            return


_add_local_qwen_repo()


def _format_bytes(num_bytes: int) -> str:
    if num_bytes <= 0:
        return "0 B"
    units = ["B", "KiB", "MiB", "GiB"]
    size = float(num_bytes)
    for u in units:
        if size < 1024 or u == units[-1]:
            return f"{size:.1f} {u}"
        size /= 1024.0
    return f"{size:.1f} GiB"


def _gpu_mem_info() -> str:
    if not torch.cuda.is_available():
        return "CUDA not available"
    free, total = torch.cuda.mem_get_info()
    used = total - free
    return f"GPU mem: {_format_bytes(used)} / {_format_bytes(total)} (free: {_format_bytes(free)})"


def _check_torchao() -> bool:
    """Check if torchao with int4_weight_only is available."""
    try:
        from torchao.quantization import int4_weight_only, quantize_  # noqa: F401
        return True
    except ImportError:
        return False


def _load_model_cpu(model_dir: str):
    """Load Qwen3-TTS model to CPU in float16 for quantization."""
    from qwen_tts import Qwen3TTSModel

    print(f"[Load] Loading model from {model_dir} to CPU (float32 first)...")
    print(f"[Load] {_gpu_mem_info()}")

    tts = Qwen3TTSModel.from_pretrained(
        model_dir,
        device_map="cpu",
        torch_dtype=torch.float32,
        attn_implementation=None,
        low_cpu_mem_usage=True,
    )
    print(f"[Load] Model loaded to CPU. {_gpu_mem_info()}")
    return tts


def _count_linear_layers(module: torch.nn.Module) -> int:
    """Count total Linear layers in a module."""
    count = 0
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            count += 1
    return count


def quantize_torchao_int4(
    model_dir: str,
    output_dir: str,
    group_size: int = 128,
) -> str:
    """
    INT4 weight-only quantization via torchao.

    Only quantizes model.talker (the LLM body).
    Preserves speech_tokenizer and speaker_encoder in original precision.

    Returns output_dir path.
    """
    from torchao.quantization import int4_weight_only, quantize_

    tts = _load_model_cpu(model_dir)
    model = tts.model

    # Identify the talker sub-module (LLM body)
    talker = getattr(model, "talker", None)
    if talker is None:
        print("[Warn] model.talker not found, quantizing entire model")
        talker = model

    total_linear = _count_linear_layers(talker)
    print(f"[Quantize] Target: model.talker | Linear layers: {total_linear}")
    print(f"[Quantize] Method: int4_weight_only(group_size={group_size})")
    print(f"[Quantize] {_gpu_mem_info()}")

    # Move talker to GPU for quantization
    print("[Quantize] Moving talker to GPU for quantization...")
    talker = talker.to("cuda", dtype=torch.float16)
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[Quantize] {_gpu_mem_info()}")

    # Apply INT4 quantization
    t0 = time.time()
    quantize_(talker, int4_weight_only(group_size=group_size))
    elapsed = time.time() - t0
    print(f"[Quantize] INT4 quantization done in {elapsed:.1f}s")

    # Move back to CPU for saving
    talker = talker.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    # Count quantized layers
    quantized_count = 0
    for m in talker.modules():
        cls_name = type(m).__name__
        if "int4" in cls_name.lower() or "affine" in cls_name.lower() or "quant" in cls_name.lower():
            quantized_count += 1

    print(f"[Quantize] Quantized layers detected: {quantized_count}")

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save the full model state dict
    model_save_path = output_path / "quantized_model.pt"
    print(f"[Save] Saving quantized model to {model_save_path}...")
    torch.save(model.state_dict(), str(model_save_path))

    # Copy config files from original model dir
    src = Path(model_dir)
    for pattern in ["*.json", "*.txt", "*.model", "*.vocab", "*.tiktoken", "*.spm"]:
        for f in src.glob(pattern):
            if f.name != "quantize_config.json":
                dst = output_path / f.name
                if not dst.exists():
                    shutil.copy2(str(f), str(dst))

    # Write quantization metadata
    config = {
        "method": "torchao_int4_weight_only",
        "group_size": group_size,
        "quantized_target": "model.talker",
        "total_linear_layers": total_linear,
        "quantized_layers": quantized_count,
        "source_model_dir": str(Path(model_dir).resolve()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch_version": torch.__version__,
    }
    config_path = output_path / "quantize_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"[Save] Quantization config: {config_path}")
    print(f"[Done] Output: {output_dir}")
    return str(output_path)


def quantize_torch_dynamic_int8(
    model_dir: str,
    output_dir: str,
) -> str:
    """
    Fallback: dynamic INT8 quantization via PyTorch built-in.

    Only quantizes model.talker Linear layers.
    """
    tts = _load_model_cpu(model_dir)
    model = tts.model

    talker = getattr(model, "talker", None)
    if talker is None:
        print("[Warn] model.talker not found, quantizing entire model")
        talker = model

    total_linear = _count_linear_layers(talker)
    print(f"[Quantize] Target: model.talker | Linear layers: {total_linear}")
    print("[Quantize] Method: torch.ao.quantization.quantize_dynamic (INT8)")

    # ARM/Jetson 平台需要显式设置 qnnpack 后端
    supported = torch.backends.quantized.supported_engines
    current = torch.backends.quantized.engine
    print(f"[Quantize] Quantization backends: supported={supported}, current={current}")
    if "qnnpack" in supported and current != "qnnpack":
        torch.backends.quantized.engine = "qnnpack"
        print("[Quantize] Switched quantization backend to qnnpack")

    t0 = time.time()
    torch.ao.quantization.quantize_dynamic(
        talker,
        {torch.nn.Linear},
        dtype=torch.qint8,
        inplace=True,
    )
    elapsed = time.time() - t0
    print(f"[Quantize] INT8 dynamic quantization done in {elapsed:.1f}s")

    # Count quantized layers
    quantized_count = 0
    for m in talker.modules():
        cls_name = type(m).__name__
        if "dynamic" in cls_name.lower() or "quantized" in cls_name.lower():
            quantized_count += 1

    print(f"[Quantize] Quantized layers: {quantized_count}/{total_linear}")

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_save_path = output_path / "quantized_model.pt"
    print(f"[Save] Saving quantized model to {model_save_path}...")
    torch.save(model.state_dict(), str(model_save_path))

    # Copy config files
    src = Path(model_dir)
    for pattern in ["*.json", "*.txt", "*.model", "*.vocab", "*.tiktoken", "*.spm"]:
        for f in src.glob(pattern):
            if f.name != "quantize_config.json":
                dst = output_path / f.name
                if not dst.exists():
                    shutil.copy2(str(f), str(dst))

    config = {
        "method": "torch_dynamic_int8",
        "quantized_target": "model.talker",
        "total_linear_layers": total_linear,
        "quantized_layers": quantized_count,
        "source_model_dir": str(Path(model_dir).resolve()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch_version": torch.__version__,
    }
    config_path = output_path / "quantize_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"[Save] Quantization config: {config_path}")
    print(f"[Done] Output: {output_dir}")
    return str(output_path)


def load_quantized_model(model_dir: str, quantized_dir: str):
    """
    Load a quantized model for inference.

    Loads config and weights from quantized_dir, using original model_dir
    for model architecture.
    """
    from qwen_tts import Qwen3TTSModel

    quantized_path = Path(quantized_dir)
    config_path = quantized_path / "quantize_config.json"
    model_path = quantized_path / "quantized_model.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"quantize_config.json not found in {quantized_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"quantized_model.pt not found in {quantized_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        qconfig = json.load(f)

    method = qconfig.get("method", "unknown")
    print(f"[Load] Loading quantized model (method={method}) from {quantized_dir}")

    # For torchao INT4, we need to load the model structure first then apply weights
    if method == "torchao_int4_weight_only":
        if not _check_torchao():
            raise ImportError("torchao is required to load INT4 quantized models")

        from torchao.quantization import int4_weight_only, quantize_

        group_size = qconfig.get("group_size", 128)

        # Load model structure
        tts = Qwen3TTSModel.from_pretrained(
            model_dir,
            device_map="cpu",
            torch_dtype=torch.float16,
            attn_implementation=None,
            low_cpu_mem_usage=True,
        )

        # Apply quantization structure to talker
        talker = getattr(tts.model, "talker", tts.model)
        talker = talker.to("cuda", dtype=torch.float16)
        quantize_(talker, int4_weight_only(group_size=group_size))

        # Load saved weights
        state_dict = torch.load(str(model_path), map_location="cpu", weights_only=False)
        tts.model.load_state_dict(state_dict, strict=False)
        print(f"[Load] INT4 model loaded successfully")
        return tts

    elif method == "torch_dynamic_int8":
        # For dynamic INT8, load model and apply quantization, then load weights
        tts = Qwen3TTSModel.from_pretrained(
            model_dir,
            device_map="cpu",
            torch_dtype=torch.float32,
            attn_implementation=None,
            low_cpu_mem_usage=True,
        )
        talker = getattr(tts.model, "talker", tts.model)
        torch.ao.quantization.quantize_dynamic(
            talker,
            {torch.nn.Linear},
            dtype=torch.qint8,
            inplace=True,
        )
        state_dict = torch.load(str(model_path), map_location="cpu", weights_only=False)
        tts.model.load_state_dict(state_dict, strict=False)
        print(f"[Load] INT8 model loaded successfully")
        return tts

    else:
        raise ValueError(f"Unknown quantization method: {method}")


def verify_quantized_model(model_dir: str, quantized_dir: str) -> bool:
    """Run a short inference to verify the quantized model works."""
    print("\n[Verify] Running test inference with quantized model...")

    try:
        tts = load_quantized_model(model_dir, quantized_dir)

        # Move to GPU for inference
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            # Move non-quantized parts
            speech_tok = getattr(tts.model, "speech_tokenizer", None)
            speaker_enc = getattr(tts.model, "speaker_encoder", None)
            if speech_tok is not None:
                try:
                    speech_tok.model = speech_tok.model.to(device)
                    speech_tok.device = torch.device(device)
                except Exception:
                    pass
            if speaker_enc is not None:
                try:
                    speaker_enc = speaker_enc.to(device)
                except Exception:
                    pass
            tts.device = torch.device(device)

        # Detect model type for test
        model_type = getattr(tts.model, "tts_model_type", "voice_design")
        print(f"[Verify] Model type: {model_type}, device: {device}")

        t0 = time.time()
        if model_type == "voice_design":
            wavs, sr = tts.generate_voice_design(
                text="Hello, this is a test.",
                language="Auto",
                instruct="A calm female voice.",
                max_new_tokens=256,
            )
        elif model_type == "base":
            print("[Verify] Base model requires reference audio, skipping generation test.")
            print("[Verify] Model loaded successfully (structure OK)")
            return True
        else:
            print(f"[Verify] Model type '{model_type}' — skipping generation, structure OK")
            return True

        elapsed = time.time() - t0
        audio_len = len(wavs[0]) / sr if sr > 0 else 0
        rtf = elapsed / audio_len if audio_len > 0 else float("inf")

        print(f"[Verify] Test passed!")
        print(f"[Verify] Generated {audio_len:.2f}s audio in {elapsed:.2f}s (RTF={rtf:.2f})")
        print(f"[Verify] Sample rate: {sr}")

        # Optionally save test audio
        test_audio_path = Path(quantized_dir) / "test_output.wav"
        try:
            import numpy as np
            import soundfile as sf
            sf.write(str(test_audio_path), wavs[0], sr)
            print(f"[Verify] Test audio saved: {test_audio_path}")
        except Exception:
            pass

        return True

    except Exception as e:
        print(f"[Verify] FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_quantization_info(quantized_dir: str) -> Optional[Dict[str, Any]]:
    """Read quantization config from a quantized model directory."""
    config_path = Path(quantized_dir) / "quantize_config.json"
    if not config_path.exists():
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Qwen3-TTS INT4/INT8 quantization for Jetson Orin.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # INT4 quantization (default)
  python3 quantize_int4.py --model-dir models/Qwen3-TTS-12Hz-1.7B-VoiceDesign

  # INT4 with custom group size
  python3 quantize_int4.py --model-dir models/Qwen3-TTS-12Hz-1.7B-VoiceDesign --group-size 64

  # Fallback to INT8
  python3 quantize_int4.py --model-dir models/Qwen3-TTS-12Hz-1.7B-VoiceDesign --method int8

  # Quantize and verify
  python3 quantize_int4.py --model-dir models/Qwen3-TTS-12Hz-1.7B-VoiceDesign --verify
        """,
    )
    p.add_argument("--model-dir", required=True, help="FP16 model directory.")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for quantized model (default: <model-dir>-INT4 or -INT8).",
    )
    p.add_argument(
        "--method",
        choices=["int4", "int8", "auto"],
        default="auto",
        help="Quantization method. 'auto' tries INT4 first, falls back to INT8. (default: auto)",
    )
    p.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Group size for INT4 weight-only quantization. (default: 128)",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="Run test inference after quantization to verify.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Force re-quantization even if output already exists.",
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.exists():
        print(f"[Error] Model directory not found: {model_dir}")
        return 1

    # Validate model dir
    has_config = (model_dir / "config.json").exists()
    has_weights = any(model_dir.glob("*.safetensors")) or any(model_dir.glob("*.bin"))
    if not has_config:
        print(f"[Error] config.json not found in {model_dir}")
        return 1
    if not has_weights:
        print(f"[Error] No weight files found in {model_dir}")
        return 1

    # Determine method
    method = args.method
    if method == "auto":
        if _check_torchao():
            method = "int4"
            print("[Info] torchao available, using INT4 quantization")
        else:
            method = "int8"
            print("[Info] torchao not available, falling back to INT8 quantization")

    if method == "int4" and not _check_torchao():
        print("[Error] torchao is required for INT4 quantization but not installed.")
        print("[Info] Install: pip install torchao")
        print("[Info] Or use --method int8 for fallback.")
        return 1

    # Determine output dir
    suffix = "-INT4" if method == "int4" else "-INT8"
    output_dir = args.output_dir or str(model_dir) + suffix
    output_path = Path(output_dir).expanduser().resolve()

    # Check if already quantized
    if not args.force and (output_path / "quantize_config.json").exists():
        print(f"[Skip] Quantized model already exists: {output_path}")
        print("[Info] Use --force to re-quantize.")
        if args.verify:
            ok = verify_quantized_model(str(model_dir), str(output_path))
            return 0 if ok else 1
        return 0

    print("=" * 60)
    print(f"Qwen3-TTS {method.upper()} Quantization")
    print(f"  Source:  {model_dir}")
    print(f"  Output:  {output_path}")
    if method == "int4":
        print(f"  Group:   {args.group_size}")
    print("=" * 60)

    # Run quantization
    t_start = time.time()
    try:
        if method == "int4":
            quantize_torchao_int4(
                model_dir=str(model_dir),
                output_dir=str(output_path),
                group_size=args.group_size,
            )
        else:
            quantize_torch_dynamic_int8(
                model_dir=str(model_dir),
                output_dir=str(output_path),
            )
    except Exception as e:
        print(f"\n[Error] Quantization failed: {e}")
        import traceback
        traceback.print_exc()

        # If INT4 failed and method was auto-selected, suggest fallback
        if method == "int4":
            print("\n[Hint] Try --method int8 as fallback:")
            print(f"  python3 {__file__} --model-dir {model_dir} --method int8")
        return 1

    t_total = time.time() - t_start
    print(f"\n[Done] Quantization completed in {t_total:.1f}s")

    # Verify if requested
    if args.verify:
        ok = verify_quantized_model(str(model_dir), str(output_path))
        if not ok:
            print("[Warn] Verification failed, but quantized model was saved.")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
