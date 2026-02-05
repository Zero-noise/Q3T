#!/usr/bin/env python3
# coding=utf-8
"""
Qwen3-TTS VoiceDesign INT4 launcher for Jetson Orin.

Startup modes:
1) Provide a custom quantization command (--quantize-cmd)
2) Auto-detect existing INT4 engine
3) If missing, run quantization script
4) After quantization, launch Gradio automatically
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def _path(p: Optional[str]) -> Optional[Path]:
    if not p:
        return None
    return Path(p).expanduser().resolve()


def _engine_ready(engine_path: Path) -> bool:
    return engine_path.exists() and engine_path.is_file()


def _run_cmd(cmd: List[str], shell: bool = False) -> None:
    print(f"[Run] {' '.join(cmd) if not shell else cmd}")
    if shell:
        result = subprocess.run(cmd, shell=True)
    else:
        result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def _build_quantize_cmd(args, model_dir: Path, int4_dir: Path) -> List[str]:
    if args.quantize_cmd:
        return [args.quantize_cmd]

    script = _path(args.quantize_script)
    if not script or not script.exists():
        raise FileNotFoundError(
            "Quantization script not found. Provide --quantize-script or use --quantize-cmd."
        )

    if not args.calib_dataset:
        raise ValueError("Missing --calib-dataset for quantization.")

    cmd = [
        sys.executable,
        str(script),
        "--model_dir",
        str(model_dir),
        "--output_dir",
        str(int4_dir),
        "--calib_dataset",
        str(_path(args.calib_dataset)),
        "--dtype",
        "int4",
        "--max_batch_size",
        str(args.max_batch_size),
        "--max_input_len",
        str(args.max_input_len),
        "--max_output_len",
        str(args.max_output_len),
    ]

    if args.extra_quantize_args:
        cmd.extend(shlex.split(args.extra_quantize_args))

    return cmd


def _build_gradio_cmd(args, engine_path: Path, tokenizer_dir: Path) -> List[str]:
    cmd = [
        sys.executable,
        str(_path(args.gradio_script) or "jetson_gradio_app.py"),
        "--backend",
        "trt",
        "--engine-path",
        str(engine_path),
        "--tokenizer-dir",
        str(tokenizer_dir),
        "--model-type",
        "voice_design",
    ]

    if args.gradio_args:
        cmd.extend(shlex.split(args.gradio_args))

    return cmd


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="INT4 quantize + launch Gradio (VoiceDesign only).")
    p.add_argument("--model-dir", required=True, help="FP16 model directory.")
    p.add_argument("--int4-dir", default=None, help="INT4 output directory (default: <model-dir>-int4).")
    p.add_argument("--engine-path", default=None, help="TensorRT engine plan path.")
    p.add_argument("--tokenizer-dir", default=None, help="Tokenizer directory (default: --model-dir).")

    # Quantization controls
    p.add_argument("--quantize-cmd", default=None, help="Custom quantization command (runs as shell).")
    p.add_argument("--quantize-script", default=None, help="Quantization script path.")
    p.add_argument("--calib-dataset", default=None, help="Calibration dataset jsonl.")
    p.add_argument("--max-batch-size", type=int, default=4)
    p.add_argument("--max-input-len", type=int, default=256)
    p.add_argument("--max-output-len", type=int, default=1024)
    p.add_argument("--extra-quantize-args", default=None, help="Extra args forwarded to quantize script.")
    p.add_argument("--skip-quantize", action="store_true", help="Skip quantization step.")
    p.add_argument("--force-quantize", action="store_true", help="Force re-quantization.")

    # Gradio launch
    p.add_argument("--gradio-script", default="jetson_gradio_app.py", help="Gradio app entrypoint.")
    p.add_argument("--gradio-args", default=None, help="Extra args for Gradio app.")
    p.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    model_dir = _path(args.model_dir)
    if not model_dir or not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {args.model_dir}")

    int4_dir = _path(args.int4_dir) or Path(str(model_dir) + "-int4")
    engine_path = _path(args.engine_path) or (int4_dir / "trt_engine.plan")
    tokenizer_dir = _path(args.tokenizer_dir) or model_dir

    ready = _engine_ready(engine_path)
    if args.force_quantize:
        ready = False

    if args.skip_quantize:
        print("[Skip] Quantization step skipped.")
    elif not ready:
        print("[Quantize] INT4 engine not found. Starting quantization...")
        int4_dir.mkdir(parents=True, exist_ok=True)
        cmd = _build_quantize_cmd(args, model_dir, int4_dir)
        if args.dry_run:
            print(f"[Dry-Run] Quantize command: {cmd}")
        else:
            if args.quantize_cmd:
                _run_cmd(cmd[0], shell=True)
            else:
                _run_cmd(cmd, shell=False)

        if not _engine_ready(engine_path):
            raise RuntimeError(f"Quantization finished but engine not found: {engine_path}")
    else:
        print(f"[OK] INT4 engine found: {engine_path}")

    gradio_cmd = _build_gradio_cmd(args, engine_path, tokenizer_dir)
    if args.dry_run:
        print(f"[Dry-Run] Gradio command: {gradio_cmd}")
        return 0

    print("[Launch] Starting Gradio...")
    _run_cmd(gradio_cmd, shell=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
