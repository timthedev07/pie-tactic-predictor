#!/usr/bin/env python3
"""Serve a fine-tuned model via a simple REST API backed by vLLM.

Exposes:
    POST /predict   { "prompt": "<context + goal string>" }
                 -> { "tactic": "intro n" }

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
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel as PydanticBaseModel
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

MODELS_FILE = Path(__file__).parent / "models.json"
OUTPUT_DIR = Path(__file__).parent / "output"

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class PredictRequest(PydanticBaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.0


class PredictResponse(PydanticBaseModel):
    tactic: str


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


def load_engine(
    args, model: dict, adapter_path: Path | None
) -> tuple[LLM, LoRARequest | None]:
    """Initialise the vLLM engine and optional LoRA adapter."""
    base_model = model["hf_id"]
    lora_request = None

    engine_kwargs = dict(
        model=base_model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        dtype="auto",
        trust_remote_code=True,
    )

    if args.gpu_memory_utilization is not None:
        engine_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization

    if args.merged:
        merged_dir = OUTPUT_DIR / model["id"] / "merged"
        if not merged_dir.exists():
            print(f"[error] Merged model not found at {merged_dir}")
            print("  Run: python merge_adapter.py --model", model["id"])
            sys.exit(1)
        engine_kwargs["model"] = str(merged_dir)
    elif adapter_path is not None:
        max_rank = get_max_lora_rank(adapter_path)
        engine_kwargs["enable_lora"] = True
        engine_kwargs["max_lora_rank"] = max_rank
        lora_request = LoRARequest(
            lora_name=f"{model['id']}-finetuned",
            lora_int_id=1,
            lora_path=str(adapter_path),
        )
        print(f"\n  LoRA adapter: {adapter_path}")
    else:
        print(f"[warning] No adapter found for {model['id']}; serving base model only.")

    print(f"  Loading model: {engine_kwargs['model']} ...")
    llm = LLM(**engine_kwargs)
    return llm, lora_request


def create_app(llm: LLM, lora_request: LoRARequest | None) -> FastAPI:
    app = FastAPI(title="Pie Tactic Predictor")

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        sampling = SamplingParams(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            stop=["\n"],
        )
        generate_kwargs = dict(
            prompts=[req.prompt],
            sampling_params=sampling,
        )
        if lora_request is not None:
            generate_kwargs["lora_request"] = lora_request

        outputs = llm.generate(**generate_kwargs)
        tactic = outputs[0].outputs[0].text.strip()
        return PredictResponse(tactic=tactic)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


def main():
    parser = argparse.ArgumentParser(
        description="Serve a fine-tuned model with a POST /predict endpoint.",
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
    llm, lora_request = load_engine(args, model, adapter_path)
    app = create_app(llm, lora_request)

    print(f"\n  Serving on http://{args.host}:{args.port}")
    print(f'  POST /predict  {{ "prompt": "..." }} -> {{ "tactic": "..." }}')
    print(f"  GET  /health\n")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
