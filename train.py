"""
Pie Tactic Prediction — Fine-tuning Script
==========================================
Usage:
    python finetune.py                              # interactive picker, loads hyperparams from models.json
    python finetune.py --model deepseek-coder-1.3b  # use short id from models.json
    python finetune.py --model deepseek-ai/deepseek-coder-1.3b-base  # or full HF id directly

    # Any hyperparameter flag overrides the model's default from models.json:
    python finetune.py --model codellama-7b --epochs 3 --lr 5e-5

Requirements:
    pip install transformers peft trl datasets accelerate bitsandbytes
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


# ---------------------------------------------------------------------------
# 1. Models catalogue
# ---------------------------------------------------------------------------

MODELS_FILE = "models.json"

# Fallback defaults used when a custom HF id is entered that isn't in models.json
DEFAULT_HYPERPARAMETERS = {
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "epochs": 5,
    "batch_size": 8,
    "grad_accum": 4,
    "lr": 1e-4,
    "max_seq_len": 1024,
    "warmup_ratio": 0.05,
    "load_in_4bit": False,
    "load_in_8bit": False,
}


def load_models_list() -> list[dict]:
    script_dir = Path(__file__).parent
    models_path = script_dir / MODELS_FILE
    if not models_path.exists():
        print(f"[warning] {MODELS_FILE} not found — model list unavailable.")
        return []
    with open(models_path) as f:
        return json.load(f)


def find_model(models: list[dict], query: str) -> dict | None:
    """Match by short id or full hf_id."""
    for m in models:
        if m["id"] == query or m["hf_id"] == query:
            return m
    return None


# ---------------------------------------------------------------------------
# 2. Interactive picker
# ---------------------------------------------------------------------------


def pick_model_interactively(models: list[dict]) -> dict:
    """
    Display a table of available models and return the selected model dict.
    If the user enters a custom HF id, return a synthetic model dict
    with default hyperparameters.
    """
    print("\n" + "=" * 70)
    print("  Pie Tactic Fine-tuning — Model Selection")
    print("=" * 70)

    if models:
        print(
            f"\n  {'#':<4} {'id':<24} {'lora_r':<8} {'batch':<7} "
            f"{'grad_accum':<12} {'lr':<10} {'epochs'}"
        )
        print("  " + "-" * 70)
        for i, m in enumerate(models, start=1):
            hp = m["hyperparameters"]
            print(
                f"  {str(i):<4} {m['id']:<24} {hp['lora_r']:<8} "
                f"{hp['batch_size']:<7} {hp['grad_accum']:<12} "
                f"{hp['lr']:<10} {hp['epochs']}"
            )
        print(f"\n  {len(models) + 1}   Enter a custom HuggingFace model ID")

    print("\n" + "-" * 70)

    while True:
        raw = input("  Select a number, or type a model id / HuggingFace id: ").strip()

        if not raw:
            print("  Please enter a number or id.")
            continue

        # Numeric selection from list
        if raw.isdigit():
            choice = int(raw)
            if 1 <= choice <= len(models):
                selected = models[choice - 1]
                print(f"\n  Selected : {selected['id']}  ({selected['hf_id']})")
                return selected
            elif choice == len(models) + 1:
                raw = ""
            else:
                print(f"  Enter 1–{len(models) + 1}.")
                continue

        # Typed id — could be short id, full hf_id, or completely custom
        if not raw:
            raw = input("  HuggingFace model ID (e.g. facebook/opt-1.3b): ").strip()
            if not raw:
                print("  Cannot be empty.")
                continue

        # Check if it matches a known model
        match = find_model(models, raw)
        if match:
            print(f"\n  Matched  : {match['id']}  ({match['hf_id']})")
            return match

        # Completely unknown — use defaults
        print(f"\n  Custom model: {raw}")
        print(f"  Using default hyperparameters (override with CLI flags).")
        return {
            "id": raw.split("/")[-1],
            "hf_id": raw,
            "hyperparameters": DEFAULT_HYPERPARAMETERS.copy(),
        }


# ---------------------------------------------------------------------------
# 3. Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune any HF model on Pie tactic prediction data.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Model — accepts either short id (from models.json) or full HF id
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Short id from models.json (e.g. deepseek-coder-1.3b)\n"
            "or full HuggingFace id (e.g. deepseek-ai/deepseek-coder-1.3b-base).\n"
            "If omitted, the interactive picker is shown."
        ),
    )

    # Data
    parser.add_argument("--data", type=str, default="training-data.jsonl")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.1)

    # Quantisation (rarely needed with A100 80GB but kept for flexibility)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")

    # Hyperparameter overrides — all optional, fall back to models.json values
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)

    # Multi-GPU parallel training
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Train every model in models.json simultaneously, one per GPU.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt (used internally by --parallel).",
    )

    return parser.parse_args()


def resolve_hyperparameters(model_entry: dict, cli_args) -> dict:
    """
    Merge hyperparameters: model defaults from models.json, overridden by
    any CLI flags the user explicitly passed (i.e. non-None values).
    """
    hp = model_entry["hyperparameters"].copy()

    overrides = {
        "lora_r": cli_args.lora_r,
        "lora_alpha": cli_args.lora_alpha,
        "lora_dropout": cli_args.lora_dropout,
        "epochs": cli_args.epochs,
        "batch_size": cli_args.batch_size,
        "grad_accum": cli_args.grad_accum,
        "lr": cli_args.lr,
        "max_seq_len": cli_args.max_seq_len,
        "warmup_ratio": cli_args.warmup_ratio,
    }
    for key, val in overrides.items():
        if val is not None:
            hp[key] = val

    # CLI quantisation flags always win
    if cli_args.load_in_4bit:
        hp["load_in_4bit"] = True
        hp["load_in_8bit"] = False
    if cli_args.load_in_8bit:
        hp["load_in_8bit"] = True
        hp["load_in_4bit"] = False

    return hp


# ---------------------------------------------------------------------------
# 4. Data loading and formatting
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """\
### Theorem
{theorem_name}: {theorem_type}

### Global context
{global_context}

### Local context
{local_context}

### Current goal
{goal}

### Next tactic
"""


def format_global_context(entries: list[dict]) -> str:
    if not entries:
        return "(empty)"
    return "  ".join(f"{e['name']} : {e['type'].strip()}" for e in entries)


def format_local_context(entries: list[dict]) -> str:
    if not entries:
        return "(empty)"
    return "  ".join(f"{e['name']} : {e['type'].strip()}" for e in entries)


def row_to_text(row: dict) -> str:
    prompt = PROMPT_TEMPLATE.format(
        theorem_name=row["theoremName"],
        theorem_type=row["theoremType"].strip(),
        global_context=format_global_context(row["globalContext"]),
        local_context=format_local_context(row["localContext"]),
        goal=row["goal"].strip(),
    )
    return prompt + row["tactic"].strip()


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_by_proof(
    rows: list[dict], val_frac: float, test_frac: float, seed: int
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split at proof level — no proof leaks across train / val / test."""
    proof_map: dict[str, list[dict]] = {}
    for row in rows:
        proof_map.setdefault(row["theoremName"], []).append(row)

    names = list(proof_map.keys())
    random.Random(seed).shuffle(names)

    n = len(names)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))

    test_names = set(names[:n_test])
    val_names = set(names[n_test : n_test + n_val])
    train_names = set(names[n_test + n_val :])

    train = [r for name in train_names for r in proof_map[name]]
    val = [r for name in val_names for r in proof_map[name]]
    test = [r for name in test_names for r in proof_map[name]]
    return train, val, test


def make_hf_dataset(rows: list[dict]) -> Dataset:
    return Dataset.from_dict({"text": [row_to_text(r) for r in rows]})


# ---------------------------------------------------------------------------
# 5. Model and tokeniser
# ---------------------------------------------------------------------------


def load_tokeniser(hf_id: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"
    return tok


def load_model(hf_id: str, load_in_4bit: bool, load_in_8bit: bool):
    bnb = None
    if load_in_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif load_in_8bit:
        bnb = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not (load_in_4bit or load_in_8bit) else None,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model


# ---------------------------------------------------------------------------
# 6. LoRA
# ---------------------------------------------------------------------------


def make_lora_config(hp: dict) -> LoraConfig:
    return LoraConfig(
        r=hp["lora_r"],
        lora_alpha=hp["lora_alpha"],
        lora_dropout=hp["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",  # LLaMA / Mistral / DeepSeek
            "gate_proj",
            "up_proj",
            "down_proj",  # MLP layers
            "query_key_value",  # Falcon
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",  # GPT-NeoX
        ],
    )


# ---------------------------------------------------------------------------
# 7. Training
# ---------------------------------------------------------------------------


def train(hf_id: str, hp: dict, args):
    output_dir = args.output_dir or f"./output/{args.model_id}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  hf_id    : {hf_id}")
    print(f"  lora_r   : {hp['lora_r']}  |  lora_alpha : {hp['lora_alpha']}")
    print(f"  epochs   : {hp['epochs']}  |  lr         : {hp['lr']}")
    print(f"  batch    : {hp['batch_size']}  |  grad_accum : {hp['grad_accum']}")
    print(
        f"  quant    : {'4-bit' if hp['load_in_4bit'] else '8-bit' if hp['load_in_8bit'] else 'none (bf16)'}"
    )
    print(f"  output   : {output_dir}")
    print(f"{'='*62}\n")

    # Data
    print("Loading data...")
    rows = load_jsonl(args.data)
    train_rows, val_rows, test_rows = split_by_proof(
        rows, args.val_split, args.test_split, args.seed
    )
    print(f"  train={len(train_rows)}  val={len(val_rows)}  test={len(test_rows)}")

    train_ds = make_hf_dataset(train_rows)
    val_ds = make_hf_dataset(val_rows)

    test_path = os.path.join(output_dir, "test_rows.jsonl")
    with open(test_path, "w") as f:
        for r in test_rows:
            f.write(json.dumps(r) + "\n")

    # Tokeniser + model
    print("\nLoading tokeniser...")
    tokenizer = load_tokeniser(hf_id)

    print("Loading model...")
    model = load_model(hf_id, hp["load_in_4bit"], hp["load_in_8bit"])

    print("Applying LoRA...")
    model = get_peft_model(model, make_lora_config(hp))
    model.print_trainable_parameters()

    # Loss masking — train only on the tactic completion
    collator = DataCollatorForCompletionOnlyLM(
        response_template="\n### Next tactic\n",
        tokenizer=tokenizer,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=hp["epochs"],
        per_device_train_batch_size=hp["batch_size"],
        per_device_eval_batch_size=hp["batch_size"],
        gradient_accumulation_steps=hp["grad_accum"],
        learning_rate=hp["lr"],
        lr_scheduler_type="cosine",
        warmup_ratio=hp["warmup_ratio"],
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=20,
        report_to="none",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        optim=(
            "paged_adamw_8bit"
            if (hp["load_in_4bit"] or hp["load_in_8bit"])
            else "adamw_torch"
        ),
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        dataset_text_field="text",
        max_seq_length=hp["max_seq_len"],
        args=training_args,
    )

    print("\nTraining...\n")
    trainer.train()

    final_path = os.path.join(output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nSaved → {final_path}")

    # Save resolved hyperparameters alongside the model
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump({"hf_id": hf_id, "hyperparameters": hp}, f, indent=2)

    return trainer, tokenizer, model, test_rows


# ---------------------------------------------------------------------------
# 8. Quick inference check
# ---------------------------------------------------------------------------


def test_inference(model, tokenizer, test_rows: list[dict], n: int = 5):
    print(f"\n{'='*62}")
    print("  Sample predictions on held-out test set")
    print(f"{'='*62}\n")

    model.eval()
    for i, row in enumerate(random.sample(test_rows, min(n, len(test_rows)))):
        prompt = PROMPT_TEMPLATE.format(
            theorem_name=row["theoremName"],
            theorem_type=row["theoremType"].strip(),
            global_context=format_global_context(row["globalContext"]),
            local_context=format_local_context(row["localContext"]),
            goal=row["goal"].strip(),
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        prediction = prediction.split("\n")[0].strip()

        print(f"[{i+1}] {row['theoremName']} / step {row['stepIndex']}")
        print(f"  goal   : {row['goal'].strip()[:80]}")
        print(f"  truth  : {row['tactic']}")
        print(f"  pred   : {prediction}")
        print(f"  match  : {'✓' if prediction == row['tactic'] else '✗'}\n")


# ---------------------------------------------------------------------------
# 9. Entry point
# ---------------------------------------------------------------------------


def _run_parallel(args):
    """Spawn one subprocess per model in models.json, capped to n_gpus concurrent jobs."""
    import queue
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor, as_completed

    models = load_models_list()
    if not models:
        print("[parallel] No models found in models.json — nothing to do.")
        return

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        print("[parallel] No CUDA GPUs detected — cannot run parallel training.")
        exit(1)

    print(f"\n[parallel] {len(models)} model(s) × {n_gpus} GPU(s) available")
    print("[parallel] Models will be queued and dispatched as GPUs free up.\n")

    gpu_pool: queue.Queue[int] = queue.Queue()
    for i in range(n_gpus):
        gpu_pool.put(i)

    def launch(model_entry: dict):
        gpu_idx = gpu_pool.get()
        try:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

            cmd = [
                sys.executable,
                __file__,
                "--model",
                model_entry["id"],
                "--yes",
                "--data",
                args.data,
                "--val-split",
                str(args.val_split),
                "--test-split",
                str(args.test_split),
                "--seed",
                str(args.seed),
            ]
            if args.output_dir:
                cmd += ["--output-dir", args.output_dir]
            if args.load_in_4bit:
                cmd.append("--load-in-4bit")
            if args.load_in_8bit:
                cmd.append("--load-in-8bit")

            print(f"  [GPU {gpu_idx}] starting  {model_entry['id']}")
            proc = subprocess.Popen(cmd, env=env)
            proc.wait()
            rc = proc.returncode
            status = "done" if rc == 0 else f"FAILED (exit {rc})"
            print(f"  [GPU {gpu_idx}] finished  {model_entry['id']} — {status}")
            return model_entry["id"], rc
        finally:
            gpu_pool.put(gpu_idx)

    with ThreadPoolExecutor(max_workers=n_gpus) as pool:
        futures = [pool.submit(launch, m) for m in models]
        for fut in as_completed(futures):
            fut.result()  # re-raise any unexpected exception

    print("\n[parallel] All training jobs complete.")


if __name__ == "__main__":
    args = parse_args()
    models = load_models_list()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.parallel:
        _run_parallel(args)
        exit(0)

    # Resolve which model to use
    if args.model is None:
        model_entry = pick_model_interactively(models)
    else:
        model_entry = find_model(models, args.model)
        if model_entry is None:
            # Completely unknown id — treat as raw HF id with defaults
            print(
                f"  '{args.model}' not found in models.json — using default hyperparameters."
            )
            model_entry = {
                "id": args.model.split("/")[-1],
                "hf_id": args.model,
                "hyperparameters": DEFAULT_HYPERPARAMETERS.copy(),
            }

    # Stash the short id for output directory naming
    args.model_id = model_entry["id"]

    # Merge model defaults with any CLI overrides
    hp = resolve_hyperparameters(model_entry, args)

    # Final confirmation
    effective_batch = hp["batch_size"] * hp["grad_accum"]
    print(f"\n{'='*62}")
    print(f"  model            : {model_entry['hf_id']}")
    print(f"  lora_r / alpha   : {hp['lora_r']} / {hp['lora_alpha']}")
    print(f"  epochs           : {hp['epochs']}")
    print(f"  lr               : {hp['lr']}")
    print(
        f"  batch / accum    : {hp['batch_size']} / {hp['grad_accum']}  (effective: {effective_batch})"
    )
    print(
        f"  quantisation     : {'4-bit' if hp['load_in_4bit'] else '8-bit' if hp['load_in_8bit'] else 'none (bf16)'}"
    )
    print(f"  output dir       : {args.output_dir or './output/' + args.model_id}")
    print(f"{'='*62}")

    if not args.yes:
        confirm = input("\n  Start training? [Y/n]: ").strip().lower()
        if confirm not in ("", "y", "yes"):
            print("  Aborted.")
            exit(0)

    trainer, tokenizer, model, test_rows = train(model_entry["hf_id"], hp, args)
    test_inference(model, tokenizer, test_rows, n=5)
    print("\nDone.")
