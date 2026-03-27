#!/usr/bin/env python3
"""Merge a LoRA adapter into the base model for standalone vLLM serving.

This produces a full merged model at output/<model>/merged/ that can be served
directly by vLLM without --enable-lora.

Usage:
    python merge_adapter.py --model qwen2.5-coder-1.5b
    python merge_adapter.py --model qwen2.5-coder-7b --adapter-path output/qwen2.5-coder-7b/checkpoint-1000
"""

import argparse
import json
import sys
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Short id from models.json or HF model id.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Override adapter directory (default: output/<model>/final/).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to save merged model (default: output/<model>/merged/).",
    )
    args = parser.parse_args()

    models = load_models()
    model_info = find_model(models, args.model)
    if not model_info:
        print(f"[error] Unknown model: {args.model}")
        print("  Available:", ", ".join(m["id"] for m in models))
        sys.exit(1)

    adapter_path = (
        Path(args.adapter_path)
        if args.adapter_path
        else OUTPUT_DIR / model_info["id"] / "final"
    )
    if not adapter_path.exists():
        print(f"[error] Adapter not found at {adapter_path}")
        sys.exit(1)

    save_dir = (
        Path(args.output_dir)
        if args.output_dir
        else OUTPUT_DIR / model_info["id"] / "merged"
    )

    base_id = model_info["hf_id"]
    print(f"  Base model  : {base_id}")
    print(f"  Adapter     : {adapter_path}")
    print(f"  Output      : {save_dir}")
    print()

    print("[1/3] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)

    print("[2/3] Loading and merging LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model = model.merge_and_unload()

    print(f"[3/3] Saving merged model to {save_dir}...")
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print(f"\n[done] Merged model saved to {save_dir}")
    print(f"  Serve with: python serve.py --model {model_info['id']} --merged")


if __name__ == "__main__":
    main()
