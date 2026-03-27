#!/usr/bin/env python3
"""Serve a fine-tuned model (base + LoRA adapter) via vLLM's OpenAI-compatible API.

Usage examples
--------------
# Interactive model picker:
python serve.py

# By short id from models.json:
python serve.py --model qwen2.5-coder-1.5b

# Serve a merged model instead of base+LoRA:
python serve.py --model qwen2.5-coder-7b --merged

# Customise port, GPU count, max-model-len:
python serve.py --model qwen2.5-coder-1.5b --port 8000 --tp 1 --max-model-len 2048

# Specify a specific checkpoint instead of final/:
python serve.py --model qwen2.5-coder-1.5b --adapter-path output/qwen2.5-coder-1.5b/checkpoint-1000
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

MODELS_FILE = Path(__file__).parent / "models.json"
OUTPUT_DIR = Path(__file__).parent / "output"


def load_models() -> list[dict]:
    if not MODELS_FILE.exists():
        print(f"[error] {MODELS_FILE} not found.")
        sys.exit(1)
    with open(MODELS_FILE) as f:
        return json.load(f)


def find_model(models: list[dict], query: str) -> dict | None:
    for m in models:
        if m["id"] == query or m["hf_id"] == query:
            return m
    return None


def pick_model(models: list[dict]) -> dict:
    print("\n" + "=" * 60)
    print("  Available fine-tuned models")
    print("=" * 60)

    available = []
    for i, m in enumerate(models, start=1):
        adapter_dir = OUTPUT_DIR / m["id"] / "final"
        has_adapter = adapter_dir.exists()
        status = "ready" if has_adapter else "no adapter"
        available.append((m, has_adapter))
        print(f"  {i:<4} {m['id']:<28} [{status}]")

    print()
    while True:
        raw = input("  Select a number or type a model id: ").strip()
        if not raw:
            continue
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(models):
                return models[idx]
            print(f"  Enter 1-{len(models)}.")
            continue
        match = find_model(models, raw)
        if match:
            return match
        print(f"  Unknown model: {raw}")


def resolve_adapter_path(model: dict, adapter_override: str | None) -> Path | None:
    if adapter_override:
        p = Path(adapter_override)
        if not p.exists():
            print(f"[error] Adapter path does not exist: {p}")
            sys.exit(1)
        return p

    default = OUTPUT_DIR / model["id"] / "final"
    if default.exists():
        return default
    return None


def get_max_lora_rank(adapter_path: Path) -> int:
    config_path = adapter_path / "adapter_config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        return cfg.get("r", 64)
    return 64


def build_command(args, model: dict, adapter_path: Path | None) -> list[str]:
    base_model = model["hf_id"]

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        base_model,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(args.tp),
        "--max-model-len",
        str(args.max_model_len),
        "--dtype",
        "auto",
        "--trust-remote-code",
    ]

    if args.merged:
        # Serve a merged model (must have been created with merge_adapter.py)
        merged_dir = OUTPUT_DIR / model["id"] / "merged"
        if not merged_dir.exists():
            print(f"[error] Merged model not found at {merged_dir}")
            print("  Run: python merge_adapter.py --model", model["id"])
            sys.exit(1)
        cmd[cmd.index(base_model)] = str(merged_dir)
    elif adapter_path is not None:
        max_rank = get_max_lora_rank(adapter_path)
        lora_name = f"{model['id']}-finetuned"
        cmd += [
            "--enable-lora",
            "--lora-modules",
            f"{lora_name}={adapter_path}",
            "--max-lora-rank",
            str(max_rank),
        ]
        print(f"\n  LoRA adapter will be served as model name: {lora_name}")
        print(f"  You can also query the base model as: {base_model}")
    else:
        print(f"[warning] No adapter found for {model['id']}; serving base model only.")

    if args.gpu_memory_utilization is not None:
        cmd += ["--gpu-memory-utilization", str(args.gpu_memory_utilization)]

    if args.host:
        cmd += ["--host", args.host]

    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Serve a fine-tuned model with vLLM (OpenAI-compatible API).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Short id from models.json (e.g. qwen2.5-coder-1.5b) or HF id.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Override adapter directory (default: output/<model>/final/).",
    )
    parser.add_argument(
        "--merged",
        action="store_true",
        help="Serve a pre-merged model from output/<model>/merged/ instead of base+LoRA.",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallel size (number of GPUs). Default: 1.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Maximum sequence length for the model context.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Fraction of GPU memory to use (0.0-1.0). vLLM default is 0.9.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the vLLM command without executing.",
    )

    args = parser.parse_args()
    models = load_models()

    if args.model:
        model = find_model(models, args.model)
        if not model:
            print(f"[error] Unknown model: {args.model}")
            print("  Available:", ", ".join(m["id"] for m in models))
            sys.exit(1)
    else:
        model = pick_model(models)

    adapter_path = resolve_adapter_path(model, args.adapter_path)
    cmd = build_command(args, model, adapter_path)

    print("\n" + "-" * 60)
    print("  Command:")
    print("  " + " \\\n    ".join(cmd))
    print("-" * 60 + "\n")

    if args.dry_run:
        print("[dry-run] Not launching.")
        return

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[info] Server stopped.")
    except FileNotFoundError:
        print("[error] vllm not found. Install with: pip install vllm")
        sys.exit(1)


if __name__ == "__main__":
    main()
