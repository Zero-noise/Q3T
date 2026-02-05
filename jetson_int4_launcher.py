#!/usr/bin/env python3
# coding=utf-8
"""
Qwen3-TTS VoiceDesign INT4 launcher for Jetson Orin.

Simplified workflow:
1. Check if quantized model exists (<model-dir>-INT4/quantize_config.json)
2. If not, run quantize_int4.py to create it
3. Launch jetson_gradio_app.py with the quantized model

Usage:
    python3 jetson_int4_launcher.py --model-dir models/Qwen3-TTS-12Hz-0.6B-VoiceDesign
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def _path(p: Optional[str]) -> Optional[Path]:
    if not p:
        return None
    return Path(p).expanduser().resolve()


def _run(cmd: List[str], desc: str) -> None:
    """Run a command, printing it first."""
    print(f"[{desc}] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"{desc} failed with exit code {result.returncode}")


def _quantized_ready(int4_dir: Path) -> bool:
    """Check if a valid quantized model exists."""
    config = int4_dir / "quantize_config.json"
    weights = int4_dir / "quantized_model.pt"
    return config.exists() and weights.exists()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="INT4 quantize + launch Gradio (VoiceDesign).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: quantize 0.6B model and launch Gradio
  python3 jetson_int4_launcher.py --model-dir models/Qwen3-TTS-12Hz-0.6B-VoiceDesign

  # Skip quantization (use existing quantized model)
  python3 jetson_int4_launcher.py --model-dir models/Qwen3-TTS-12Hz-0.6B-VoiceDesign --skip-quantize

  # Force re-quantization
  python3 jetson_int4_launcher.py --model-dir models/Qwen3-TTS-12Hz-0.6B-VoiceDesign --force-quantize

  # Custom port
  python3 jetson_int4_launcher.py --model-dir models/Qwen3-TTS-12Hz-0.6B-VoiceDesign --port 7860
        """,
    )
    p.add_argument("--model-dir", required=True, help="FP16 model directory.")
    p.add_argument("--int4-dir", default=None, help="INT4 output dir (default: <model-dir>-INT4).")
    p.add_argument("--skip-quantize", action="store_true", help="Skip quantization, use FP16 model directly.")
    p.add_argument("--force-quantize", action="store_true", help="Force re-quantization.")
    p.add_argument("--port", type=int, default=8000, help="Gradio server port (default: 8000).")
    p.add_argument("--ip", default="0.0.0.0", help="Gradio bind IP (default: 0.0.0.0).")
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    model_dir = _path(args.model_dir)
    if not model_dir or not model_dir.exists():
        print(f"[Error] Model directory not found: {args.model_dir}")
        return 1

    int4_dir = _path(args.int4_dir) or Path(str(model_dir) + "-INT4")

    # Determine the script directory (for finding quantize_int4.py and gradio app)
    script_dir = Path(__file__).resolve().parent
    quantize_script = script_dir / "quantize_int4.py"
    gradio_script = script_dir / "jetson_gradio_app.py"

    # Step 1: Quantization
    if args.skip_quantize:
        print("[Skip] Quantization skipped. Using FP16 model directly.")
        checkpoint = str(model_dir)
    elif _quantized_ready(int4_dir) and not args.force_quantize:
        print(f"[OK] Quantized model found: {int4_dir}")
        checkpoint = str(model_dir)  # Gradio app loads FP16 structure; INT4 dir for reference
    else:
        if not quantize_script.exists():
            print(f"[Error] quantize_int4.py not found at {quantize_script}")
            return 1

        print("[Quantize] Starting INT4 quantization...")
        quant_cmd = [
            sys.executable,
            str(quantize_script),
            "--model-dir", str(model_dir),
            "--output-dir", str(int4_dir),
            "--verify",
        ]
        if args.force_quantize:
            quant_cmd.append("--force")

        if args.dry_run:
            print(f"[Dry-Run] {' '.join(quant_cmd)}")
        else:
            _run(quant_cmd, "Quantize")
            if not _quantized_ready(int4_dir):
                print(f"[Error] Quantization finished but output not found: {int4_dir}")
                return 1

        checkpoint = str(model_dir)

    # Step 2: Launch Gradio
    gradio_cmd = [
        sys.executable,
        str(gradio_script),
        checkpoint,
        "--device", "cuda",
        "--dtype", "float16",
        "--port", str(args.port),
        "--ip", args.ip,
        "--no-flash-attn",
        "--staged-load",
        "--tokenizer-on-cpu",
    ]

    if args.dry_run:
        print(f"[Dry-Run] {' '.join(gradio_cmd)}")
        return 0

    print("\n[Launch] Starting Gradio app...")
    print(f"[Launch] Model: {checkpoint}")
    print(f"[Launch] INT4:  {int4_dir}")
    print(f"[Launch] URL:   http://{args.ip}:{args.port}")

    _run(gradio_cmd, "Gradio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
