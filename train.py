import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

# Use fast local storage for model downloads instead of the default ~/.cache
_ssh_host = os.environ.get("SSH_CONNECTION", "") + os.environ.get("HOSTNAME", "")
if any(h in _ssh_host for h in ("twinkle2", "twinkle3")):
    os.environ.setdefault("HF_HOME", "/localhome/timbao/.cache/huggingface")
else:
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


class _CompletionOnlyCollator:
    """Masks prompt tokens so the model only trains on the tactic completion.

    Each feature must contain ``input_ids`` (list[int]), ``prompt_length``
    (int), and ``seq_length`` (int -- the *un-padded* token count).  Tokens at
    positions < prompt_length are masked with -100 so they do not contribute
    to the loss, and positions >= seq_length (i.e. padding) are also masked.

    Using an explicit ``seq_length`` instead of scanning for ``pad_token_id``
    avoids silently masking real tokens when ``pad_token_id == eos_token_id``
    (which is the common fallback for models that lack a native pad token).
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict:
        input_id_lists = [f["input_ids"] for f in features]
        prompt_lengths = [f["prompt_length"] for f in features]
        seq_lengths = [f["seq_length"] for f in features]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in input_id_lists],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = torch.zeros_like(input_ids)
        for i, slen in enumerate(seq_lengths):
            attention_mask[i, :slen] = 1

        labels = input_ids.clone()

        for i, (plen, slen) in enumerate(zip(prompt_lengths, seq_lengths)):
            labels[i, :plen] = -100  # mask prompt
            labels[i, slen:] = -100  # mask padding

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


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
        print(f"[warning] {MODELS_FILE} not found -- model list unavailable.")
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
    print("  Pie Tactic Fine-tuning -- Model Selection")
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
                print(f"  Enter 1-{len(models) + 1}.")
                continue

        # Typed id -- could be short id, full hf_id, or completely custom
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

        # Completely unknown -- use defaults
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

    # Model -- accepts either short id (from models.json) or full HF id
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

    # Hyperparameter overrides -- all optional, fall back to models.json values
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
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        metavar="CHECKPOINT_DIR",
        help=(
            "Resume training from a checkpoint.\n"
            "Pass a path to resume from a specific checkpoint, or pass the flag\n"
            "without a value to auto-detect the latest checkpoint in the output dir."
        ),
    )
    parser.add_argument(
        "--smoke-test",
        nargs="?",
        const=10,
        default=None,
        type=int,
        metavar="N",
        help=(
            "Quick pipeline sanity check: train on N samples (default: 10) for\n"
            "1 epoch to verify the full train/eval/save path works.\n"
            "Output goes to output/{model_id}-smoke-test/ unless --output-dir is set."
        ),
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


def row_to_prompt(row: dict) -> str:
    """Return just the prompt (everything up to and including the separator)."""
    return PROMPT_TEMPLATE.format(
        theorem_name=row["theoremName"],
        theorem_type=row["theoremType"].strip(),
        global_context=format_global_context(row["globalContext"]),
        local_context=format_local_context(row["localContext"]),
        goal=row["goal"].strip(),
    )


def row_to_text(row: dict) -> str:
    return row_to_prompt(row) + row["tactic"].strip()


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        content = f.read()
    decoder = json.JSONDecoder()
    rows = []
    idx = 0
    while idx < len(content):
        while idx < len(content) and content[idx].isspace():
            idx += 1
        if idx >= len(content):
            break
        obj, end = decoder.raw_decode(content, idx)
        rows.append(obj)
        idx = end
    return rows


def split_by_proof(
    rows: list[dict], val_frac: float, test_frac: float, seed: int
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split at proof level -- no proof leaks across train / val / test."""
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


def make_hf_dataset(rows: list[dict], tokenizer, max_seq_len: int) -> Dataset:
    """Pre-tokenise rows and record how many tokens belong to the prompt.

    Because the prompt template ends with a newline and BPE pre-tokenisers
    split on newlines, ``tokenize(prompt + tactic)[:n]`` equals
    ``tokenize(prompt)`` -- so the prompt token count is an accurate mask
    boundary.  This is immune to tokeniser quirks that broke the previous
    token-ID-scanning approach.

    Each sample stores:
    - ``input_ids``: the full tokenised sequence (prompt + tactic)
    - ``prompt_length``: number of tokens belonging to the prompt
    - ``seq_length``: total un-padded token count (used by the collator to
      distinguish real tokens from padding without relying on pad_token_id,
      which may equal eos_token_id)
    """
    all_input_ids: list[list[int]] = []
    all_prompt_lengths: list[int] = []
    all_seq_lengths: list[int] = []
    n_fully_masked = 0
    for r in rows:
        prompt = row_to_prompt(r)
        text = prompt + r["tactic"].strip()
        full_enc = tokenizer(
            text, add_special_tokens=True, truncation=True, max_length=max_seq_len
        )
        prompt_enc = tokenizer(
            prompt, add_special_tokens=True, truncation=True, max_length=max_seq_len
        )
        full_ids = full_enc["input_ids"]
        plen = len(prompt_enc["input_ids"])
        if plen >= len(full_ids):
            n_fully_masked += 1
        all_input_ids.append(full_ids)
        all_prompt_lengths.append(plen)
        all_seq_lengths.append(len(full_ids))

    if n_fully_masked:
        print(
            f"  WARNING: {n_fully_masked}/{len(rows)} samples have "
            f"prompt_length >= seq_length (completion tokens truncated away). "
            f"Consider increasing --max-seq-len."
        )

    return Dataset.from_dict(
        {
            "input_ids": all_input_ids,
            "prompt_length": all_prompt_lengths,
            "seq_length": all_seq_lengths,
        }
    )


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
    _smoke = args.smoke_test is not None
    output_dir = args.output_dir or (
        f"./output/{args.model_id}-smoke-test"
        if _smoke
        else f"./output/{args.model_id}"
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    if _smoke:
        print(f"  *** SMOKE TEST ({args.smoke_test} samples) ***")
    print(f"  hf_id    : {hf_id}")
    print(f"  lora_r   : {hp['lora_r']}  |  lora_alpha : {hp['lora_alpha']}")
    print(
        f"  epochs   : {'1 (smoke)' if _smoke else hp['epochs']}  |  lr         : {hp['lr']}"
    )
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
    if _smoke:
        n = args.smoke_test
        rng = random.Random(args.seed)
        train_rows = rng.sample(train_rows, min(n, len(train_rows)))
        val_rows = rng.sample(val_rows, min(n, len(val_rows)))
    print(f"  train={len(train_rows)}  val={len(val_rows)}  test={len(test_rows)}")

    # Tokeniser (needed before dataset creation for pre-tokenisation)
    print("\nLoading tokeniser...")
    tokenizer = load_tokeniser(hf_id)

    train_ds = make_hf_dataset(train_rows, tokenizer, hp["max_seq_len"])
    val_ds = make_hf_dataset(val_rows, tokenizer, hp["max_seq_len"])

    test_path = os.path.join(output_dir, "test_rows.jsonl")
    with open(test_path, "w") as f:
        for r in test_rows:
            f.write(json.dumps(r) + "\n")

    # Model
    print("Loading model...")
    model = load_model(hf_id, hp["load_in_4bit"], hp["load_in_8bit"])

    print("Applying LoRA...")
    model = get_peft_model(model, make_lora_config(hp))
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # Loss masking -- train only on the tactic completion
    collator = _CompletionOnlyCollator(tokenizer=tokenizer)

    # For smoke tests, collapse everything into 1 epoch with tiny step counts
    # so the entire train->eval->save path is exercised quickly.
    if _smoke:
        _num_epochs = 1
        _steps_per_epoch = max(1, len(train_rows) // hp["batch_size"])
        _eval_steps = _steps_per_epoch
        _save_steps = _steps_per_epoch
        _log_steps = 1
        _save_limit = 1
        _warmup = 0.0
    else:
        _num_epochs = hp["epochs"]
        _eval_steps = 100
        _save_steps = 100
        _log_steps = 20
        _save_limit = 3
        _warmup = hp["warmup_ratio"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        remove_unused_columns=False,
        num_train_epochs=_num_epochs,
        per_device_train_batch_size=hp["batch_size"],
        per_device_eval_batch_size=hp["batch_size"],
        gradient_accumulation_steps=hp["grad_accum"],
        learning_rate=hp["lr"],
        lr_scheduler_type="cosine",
        warmup_ratio=_warmup,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=_eval_steps,
        save_strategy="steps",
        save_steps=_save_steps,
        save_total_limit=_save_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=_log_steps,
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

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        args=training_args,
    )

    print("\nTraining...\n")
    t0 = time.time()

    resume_from = None
    if args.resume:
        if isinstance(args.resume, str):
            # Explicit checkpoint path supplied
            resume_from = args.resume
        else:
            # Auto-detect latest checkpoint in output dir
            resume_from = find_latest_checkpoint(output_dir)
        if resume_from:
            print(f"  Resuming from checkpoint: {resume_from}\n")
        else:
            print("  --resume set but no checkpoint found -- starting from scratch.\n")

    trainer.train(resume_from_checkpoint=resume_from)
    total_training_seconds = time.time() - t0

    final_path = os.path.join(output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nSaved -> {final_path}")

    # Save resolved hyperparameters alongside the model
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump({"hf_id": hf_id, "hyperparameters": hp}, f, indent=2)

    # Save per-step training log as CSV
    _save_training_csv(
        trainer,
        hf_id=hf_id,
        hp=hp,
        total_training_seconds=total_training_seconds,
        train_size=len(train_rows),
        val_size=len(val_rows),
        output_dir=output_dir,
    )

    return trainer, tokenizer, model, test_rows


# ---------------------------------------------------------------------------
# 8. Training CSV export
# ---------------------------------------------------------------------------


def _save_training_csv(
    trainer,
    *,
    hf_id: str,
    hp: dict,
    total_training_seconds: float,
    train_size: int,
    val_size: int,
    output_dir: str,
):
    """Write a detailed per-step CSV from trainer.state.log_history."""
    log_history = trainer.state.log_history
    if not log_history:
        print("[csv] No log history available -- skipping CSV export.")
        return

    # Collect all unique column names across all log entries, preserving a
    # sensible order: step/epoch first, then losses, then accuracies, rest.
    priority = [
        "step",
        "epoch",
        "loss",
        "eval_loss",
        "mean_token_accuracy",
        "eval_mean_token_accuracy",
        "entropy",
        "eval_entropy",
        "grad_norm",
        "learning_rate",
        "eval_runtime",
        "eval_samples_per_second",
        "eval_steps_per_second",
        "train_runtime",
        "train_samples_per_second",
        "train_steps_per_second",
        "total_flos",
    ]
    seen = set()
    ordered_cols: list[str] = []
    for col in priority:
        if any(col in entry for entry in log_history):
            ordered_cols.append(col)
            seen.add(col)
    for entry in log_history:
        for col in entry:
            if col not in seen:
                ordered_cols.append(col)
                seen.add(col)

    csv_path = os.path.join(output_dir, "training_log.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_cols, extrasaction="ignore")
        writer.writeheader()
        for entry in log_history:
            writer.writerow(entry)

    # Append a summary row for quick scanning
    summary_path = os.path.join(output_dir, "training_summary.csv")
    state = trainer.state

    # Pull best eval loss and best step from log history
    eval_entries = [e for e in log_history if "eval_loss" in e]
    best_eval_loss = min((e["eval_loss"] for e in eval_entries), default="")
    best_eval_acc = (
        max((e.get("eval_mean_token_accuracy", 0) for e in eval_entries), default="")
        if eval_entries
        else ""
    )
    train_entries = [e for e in log_history if "loss" in e and "eval_loss" not in e]
    final_train_loss = train_entries[-1]["loss"] if train_entries else ""
    final_train_acc = (
        train_entries[-1].get("mean_token_accuracy", "") if train_entries else ""
    )

    summary_fields = [
        "hf_id",
        "total_training_seconds",
        "total_training_minutes",
        "total_steps",
        "total_epochs",
        "train_samples",
        "val_samples",
        "final_train_loss",
        "final_train_accuracy",
        "best_eval_loss",
        "best_eval_accuracy",
        "lora_r",
        "lora_alpha",
        "batch_size",
        "grad_accum",
        "effective_batch_size",
        "lr",
        "epochs_config",
        "max_seq_len",
    ]
    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "hf_id": hf_id,
                "total_training_seconds": round(total_training_seconds, 1),
                "total_training_minutes": round(total_training_seconds / 60, 2),
                "total_steps": state.global_step,
                "total_epochs": round(state.epoch, 4) if state.epoch else "",
                "train_samples": train_size,
                "val_samples": val_size,
                "final_train_loss": final_train_loss,
                "final_train_accuracy": final_train_acc,
                "best_eval_loss": best_eval_loss,
                "best_eval_accuracy": best_eval_acc,
                "lora_r": hp["lora_r"],
                "lora_alpha": hp["lora_alpha"],
                "batch_size": hp["batch_size"],
                "grad_accum": hp["grad_accum"],
                "effective_batch_size": hp["batch_size"] * hp["grad_accum"],
                "lr": hp["lr"],
                "epochs_config": hp["epochs"],
                "max_seq_len": hp["max_seq_len"],
            }
        )

    print(f"  Training log  -> {csv_path}")
    print(f"  Summary row   -> {summary_path}")


# ---------------------------------------------------------------------------
# 9. Quick inference check
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
        print(f"  match  : {'[ok]' if prediction == row['tactic'] else '[x]'}\n")


# ---------------------------------------------------------------------------
# 9. Entry point
# ---------------------------------------------------------------------------


def is_trained(model_id: str, base_output_dir: str | None = None) -> bool:
    """
    Return True only if training completed fully (final/adapter_model.safetensors
    exists). A crashed run with checkpoints but no final/ returns False so that
    --parallel will pick it back up and resume it.
    """
    out_dir = Path(base_output_dir or f"./output/{model_id}")
    return (out_dir / "final" / "adapter_model.safetensors").exists()


def find_latest_checkpoint(output_dir: str) -> str | None:
    """
    Return the path of the highest-numbered checkpoint-N directory that
    contains a valid adapter, or None if no such checkpoint exists.
    """
    out = Path(output_dir)
    checkpoints = []
    if not out.is_dir():
        return None
    for child in out.iterdir():
        if child.is_dir() and child.name.startswith("checkpoint-"):
            if (child / "adapter_model.safetensors").exists():
                try:
                    step = int(child.name.split("-")[1])
                    checkpoints.append((step, child))
                except (IndexError, ValueError):
                    pass
    if not checkpoints:
        return None
    _, latest = max(checkpoints, key=lambda x: x[0])
    return str(latest)


def _run_parallel(args):
    """Spawn one subprocess per model in models.json, capped to n_gpus concurrent jobs."""
    import queue
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor, as_completed

    models = load_models_list()
    if not models:
        print("[parallel] No models found in models.json -- nothing to do.")
        return

    # Filter to only untrained models
    pending, skipped = [], []
    for m in models:
        if is_trained(m["id"], args.output_dir):
            skipped.append(m["id"])
        else:
            pending.append(m)

    if skipped:
        print(f"[parallel] Skipping already-trained model(s): {', '.join(skipped)}")
    if not pending:
        print("[parallel] All models are already trained -- nothing to do.")
        return

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        print("[parallel] No CUDA GPUs detected -- cannot run parallel training.")
        exit(1)

    print(f"\n[parallel] {len(pending)} model(s) to train x {n_gpus} GPU(s) available")
    print("[parallel] Models will be queued and dispatched as GPUs free up.\n")

    gpu_pool: queue.Queue[int] = queue.Queue()
    for i in range(n_gpus):
        gpu_pool.put(i)

    def launch(model_entry: dict):
        gpu_idx = gpu_pool.get()
        try:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

            out_dir = args.output_dir or f"./output/{model_entry['id']}"
            latest_ckpt = find_latest_checkpoint(out_dir)

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
            if args.smoke_test is not None:
                cmd += ["--smoke-test", str(args.smoke_test)]
            if args.load_in_4bit:
                cmd.append("--load-in-4bit")
            if args.load_in_8bit:
                cmd.append("--load-in-8bit")
            if latest_ckpt:
                cmd += ["--resume", latest_ckpt]
            elif args.resume:
                # propagate explicit --resume if set at the parallel level
                if isinstance(args.resume, str):
                    cmd += ["--resume", args.resume]
                else:
                    cmd.append("--resume")

            action = f"resuming from {latest_ckpt}" if latest_ckpt else "starting fresh"
            print(f"  [GPU {gpu_idx}] {model_entry['id']} -- {action}")
            proc = subprocess.Popen(cmd, env=env)
            proc.wait()
            rc = proc.returncode
            status = "done" if rc == 0 else f"FAILED (exit {rc})"
            print(f"  [GPU {gpu_idx}] finished  {model_entry['id']} -- {status}")
            return model_entry["id"], rc
        finally:
            gpu_pool.put(gpu_idx)

    with ThreadPoolExecutor(max_workers=n_gpus) as pool:
        futures = [pool.submit(launch, m) for m in pending]
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
        import sys

        if not sys.stdin.isatty():
            print(
                "[error] --model must be specified when running non-interactively "
                "(e.g. under nohup). Use --model <id> or --parallel."
            )
            exit(1)
        model_entry = pick_model_interactively(models)
    else:
        model_entry = find_model(models, args.model)
        if model_entry is None:
            # Completely unknown id -- treat as raw HF id with defaults
            print(
                f"  '{args.model}' not found in models.json -- using default hyperparameters."
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
    if args.smoke_test is not None:
        _smoke_dir = args.output_dir or f"./output/{args.model_id}-smoke-test"
        print(
            f"  smoke test       : {args.smoke_test} samples -- writes to {_smoke_dir}"
        )
    if args.resume:
        _out = args.output_dir or f"./output/{args.model_id}"
        _ckpt = (
            args.resume
            if isinstance(args.resume, str)
            else find_latest_checkpoint(_out)
        )
        print(f"  resume from      : {_ckpt or '(auto -- no checkpoint found yet)'}")
    print(f"{'='*62}")

    if not args.yes:
        import sys

        if not sys.stdin.isatty():
            # Running under nohup / backgrounded -- treat as confirmed
            print("  (non-interactive stdin detected -- auto-confirming)")
        else:
            confirm = input("\n  Start training? [Y/n]: ").strip().lower()
            if confirm not in ("", "y", "yes"):
                print("  Aborted.")
                exit(0)

    trainer, tokenizer, model, test_rows = train(model_entry["hf_id"], hp, args)
    test_inference(model, tokenizer, test_rows, n=5)
    print("\nDone.")
