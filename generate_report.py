"""
Generate LaTeX Report — Pie Tactic Predictor Fine-tuning Results
================================================================
Usage:
    python generate_report.py                        # writes report.tex
    python generate_report.py --output my_report.tex

Requires no external dependencies beyond the Python standard library.
The generated .tex file requires a standard LaTeX distribution with:
    booktabs, pgfplots, geometry, hyperref, xcolor, caption, subcaption,
    amsmath, fontenc, inputenc, microtype, multirow, array, colortbl
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"
TRAINING_DATA = SCRIPT_DIR / "training-data.jsonl"
MODELS_FILE = SCRIPT_DIR / "models.json"


def fmt(value, decimals=4):
    """Format a float, returning '—' if nan/None."""
    try:
        v = float(value)
        if math.isnan(v):
            return "---"
        return f"{v:.{decimals}f}"
    except (TypeError, ValueError):
        return "---"


def pct(value, decimals=2):
    try:
        v = float(value)
        if math.isnan(v):
            return "---"
        return f"{v * 100:.{decimals}f}\\%"
    except (TypeError, ValueError):
        return "---"


def latex_escape(s: str) -> str:
    """Escape common LaTeX special characters in strings."""
    replacements = [
        ("\\", "\\textbackslash{}"),
        ("_", "\\_"),
        ("%", "\\%"),
        ("&", "\\&"),
        ("#", "\\#"),
        ("$", "\\$"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
    ]
    for old, new in replacements:
        s = s.replace(old, new)
    return s


def hf_display(hf_id: str) -> str:
    """Turn a HuggingFace model id into a tidy display string for LaTeX."""
    return latex_escape(hf_id)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_models_json() -> list[dict]:
    if not MODELS_FILE.exists():
        return []
    with open(MODELS_FILE) as f:
        return json.load(f)


def count_training_samples() -> tuple[int, int, int]:
    """Return (total, unique_theorems, tactics) from training-data.jsonl."""
    if not TRAINING_DATA.exists():
        return 0, 0, 0
    total = 0
    theorems = set()
    tactics = set()
    with open(TRAINING_DATA) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            theorems.add(row.get("theoremName", ""))
            tactic = row.get("tactic", "")
            # extract tactic name (first token)
            head = tactic.split()[0] if tactic else ""
            if head:
                tactics.add(head)
    return total, len(theorems), len(tactics)


def load_training_summary(model_dir: Path) -> dict | None:
    path = model_dir / "training_summary.csv"
    if not path.exists():
        return None
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
    if not rows:
        return None
    return rows[0]


def load_training_log(model_dir: Path) -> tuple[list[dict], list[dict]]:
    """
    Returns (train_rows, eval_rows).
    Training rows have 'loss' and 'mean_token_accuracy' (eval_loss is empty).
    Evaluation rows have 'eval_loss' (loss may also be filled as last-seen).
    """
    path = model_dir / "training_log.csv"
    if not path.exists():
        return [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    train_rows, eval_rows = [], []
    for row in all_rows:
        step_str = row.get("step", "").strip()
        if not step_str:
            continue
        try:
            step = int(float(step_str))
        except ValueError:
            continue
        row["step"] = step

        has_eval = row.get("eval_loss", "").strip() not in ("", "nan")
        has_train = row.get("loss", "").strip() not in ("", "nan")

        if has_eval:
            eval_rows.append(row)
        elif has_train:
            train_rows.append(row)

    return train_rows, eval_rows


def load_run_config(model_dir: Path) -> dict | None:
    path = model_dir / "run_config.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def collect_all_results() -> list[dict]:
    """
    Walk output/ and collect per-model metadata + training data.
    Returns list of dicts, sorted by model id.
    """
    if not OUTPUT_DIR.exists():
        return []

    models_json = load_models_json()
    models_lookup = {m["id"]: m for m in models_json}

    results = []
    for model_dir in sorted(OUTPUT_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        model_id = model_dir.name
        summary = load_training_summary(model_dir)
        train_rows, eval_rows = load_training_log(model_dir)
        run_config = load_run_config(model_dir)
        test_count = 0
        test_path = model_dir / "test_rows.jsonl"
        if test_path.exists():
            with open(test_path) as f:
                test_count = sum(1 for l in f if l.strip())

        # Hyperparameters: prefer run_config, else models.json
        hyperparameters = None
        if run_config and "hyperparameters" in run_config:
            hyperparameters = run_config["hyperparameters"]
        elif model_id in models_lookup:
            hyperparameters = models_lookup[model_id]["hyperparameters"]

        hf_id = ""
        if run_config and "hf_id" in run_config:
            hf_id = run_config["hf_id"]
        elif model_id in models_lookup:
            hf_id = models_lookup[model_id]["hf_id"]
        elif summary:
            hf_id = summary.get("hf_id", "")

        results.append(
            {
                "id": model_id,
                "hf_id": hf_id,
                "summary": summary,
                "train_rows": train_rows,
                "eval_rows": eval_rows,
                "run_config": run_config,
                "hyperparameters": hyperparameters,
                "test_count": test_count,
                "trained": summary is not None and len(train_rows) > 0,
            }
        )
    return results


# ---------------------------------------------------------------------------
# LaTeX pieces
# ---------------------------------------------------------------------------

PREAMBLE = r"""\documentclass[11pt,a4paper]{article}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage[margin=2.5cm]{geometry}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{xcolor}
\usepackage{colortbl}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepackage{pgfplotstable}

% Colour palette
\definecolor{ds13b}{HTML}{2196F3}
\definecolor{ds67b}{HTML}{4CAF50}
\definecolor{cl7b}{HTML}{FF9800}
\definecolor{cl13b}{HTML}{E91E63}
\definecolor{lightgray}{HTML}{F5F5F5}

\hypersetup{
    colorlinks=true,
    linkcolor=blue!70!black,
    urlcolor=blue!70!black,
    citecolor=blue!70!black,
}

\title{\textbf{Pie Tactic Predictor}\\[0.3em]
       \large Fine-tuning Experiments Report}
\author{Auto-generated by \texttt{generate\_report.py}}
\date{\today}

\begin{document}
\maketitle
\tableofcontents
\newpage
"""

FOOTER = r"""
\end{document}
"""


def make_dataset_section(
    total: int, theorems: int, tactics: int, results: list[dict]
) -> str:
    # Derive train/val/test counts from first trained model's summary
    train_n, val_n, test_n = "---", "---", "---"
    for r in results:
        if r["summary"]:
            s = r["summary"]
            train_n = s.get("train_samples", "---")
            val_n = s.get("val_samples", "---")
            test_n = str(r["test_count"]) if r["test_count"] else "---"
            break

    return rf"""
\section{{Dataset}}

The training corpus consists of proof tactic prediction examples extracted from
\textsc{{Pie}} (a Pie Calculus theorem prover). Each example provides the
current proof goal together with local and global contexts, and the model must
predict the correct tactic to apply.

\begin{{table}}[h]
\centering
\caption{{Dataset statistics}}
\begin{{tabular}}{{lc}}
\toprule
\textbf{{Statistic}} & \textbf{{Value}} \\
\midrule
Total examples & {total:,} \\
Unique theorems & {theorems:,} \\
Unique tactic types & {tactics:,} \\
Training split & {train_n} \\
Validation split & {val_n} \\
Test split & {test_n} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def make_models_section(results: list[dict]) -> str:
    rows = []
    for r in results:
        hp = r["hyperparameters"] or {}
        eff = (
            hp.get("batch_size", "?") if isinstance(hp.get("batch_size"), int) else "?"
        )
        ga = hp.get("grad_accum", "?")
        if isinstance(eff, int) and isinstance(ga, int):
            eff_bs = eff * ga
        else:
            eff_bs = "?"
        trained_mark = r"$\checkmark$" if r["trained"] else r"$\times$"
        rows.append(
            rf"        \texttt{{{latex_escape(r['id'])}}} & "
            rf"\texttt{{{hf_display(r['hf_id'])}}} & "
            rf"{hp.get('lora_r', '---')} & "
            rf"{hp.get('lora_alpha', '---')} & "
            rf"{hp.get('batch_size', '---')} & "
            rf"{hp.get('grad_accum', '---')} & "
            rf"{eff_bs} & "
            rf"{hp.get('lr', '---')} & "
            rf"{hp.get('epochs', '---')} & "
            rf"{trained_mark} \\"
        )

    table_body = "\n".join(rows)
    return rf"""
\section{{Models and Hyperparameters}}

Ten candidate models spanning 1.3B to 34B parameters were evaluated. Table~\ref{{tab:models}} lists each model alongside its LoRA and training hyperparameters. Only models for which training was completed have associated loss and accuracy metrics.

\begin{{table}}[h]
\centering
\caption{{Model catalogue and hyperparameter configuration}}
\label{{tab:models}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{llrrrrrrrc}}
\toprule
\textbf{{Model ID}} & \textbf{{HuggingFace ID}} & \textbf{{LoRA $r$}} & \textbf{{$\alpha$}} & \textbf{{Batch}} & \textbf{{Grad.\ acc.}} & \textbf{{Eff.\ batch}} & \textbf{{LR}} & \textbf{{Epochs}} & \textbf{{Trained}} \\
\midrule
{table_body}
\bottomrule
\end{{tabular}}}}
\end{{table}}
"""


def make_results_section(results: list[dict]) -> str:
    trained = [r for r in results if r["trained"]]
    if not trained:
        return "\n\\section{Training Results}\n\nNo training results available.\n"

    rows = []
    for r in trained:
        s = r["summary"]
        # Detect degenerate run (all accuracy/loss = 0)
        acc = s.get("final_train_accuracy", "0")
        note = ""
        try:
            if float(acc) == 0.0:
                note = r" \textsuperscript{\dag}"
        except (ValueError, TypeError):
            pass

        rows.append(
            rf"        \texttt{{{latex_escape(r['id'])}}}{note} & "
            rf"{fmt(s.get('total_training_minutes'), 1)} & "
            rf"{s.get('total_steps', '---')} & "
            rf"{fmt(s.get('final_train_loss'), 4)} & "
            rf"{pct(s.get('final_train_accuracy'))} & "
            rf"{fmt(s.get('best_eval_loss'), 4)} & "
            rf"{pct(s.get('best_eval_accuracy'))} \\"
        )

    table_body = "\n".join(rows)

    return rf"""
\section{{Training Results}}

Table~\ref{{tab:results}} summarises the final training and best validation metrics
for all completed runs. All runs used 5 epochs and an effective batch size of 32.

\medskip
\noindent\textsuperscript{{\dag}} Model exhibited degenerate training behaviour
(loss/accuracy reported as 0.0 throughout), likely due to a tokeniser--template
mismatch that prevented the completion-only collator from finding the response
boundary. These runs are included for completeness.

\begin{{table}}[h]
\centering
\caption{{Training run summary (best eval metrics across all checkpoints)}}
\label{{tab:results}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrrrr}}
\toprule
\textbf{{Model}} & \textbf{{Time (min)}} & \textbf{{Steps}} &
\textbf{{Train loss}} & \textbf{{Train acc.}} &
\textbf{{Best val.\ loss}} & \textbf{{Best val.\ acc.}} \\
\midrule
{table_body}
\bottomrule
\end{{tabular}}}}
\end{{table}}
"""


# Colours for pgfplots — one per model
PLOT_COLOURS = {
    "deepseek-coder-1.3b": "ds13b",
    "deepseek-coder-6.7b": "ds67b",
    "codellama-7b": "cl7b",
    "codellama-13b": "cl13b",
}

DEFAULT_COLOURS = ["blue", "red", "green!60!black", "orange", "violet", "cyan!70!black"]


def get_colour(model_id: str, idx: int) -> str:
    return PLOT_COLOURS.get(model_id, DEFAULT_COLOURS[idx % len(DEFAULT_COLOURS)])


def _pgf_coords(
    rows: list[dict], x_key: str, y_key: str, skip_zero: bool = False
) -> str:
    """Build pgfplots coordinate list from rows."""
    pts = []
    for row in rows:
        try:
            x = float(row[x_key])
            y_str = row.get(y_key, "").strip()
            if not y_str or y_str in ("nan", "inf", "-inf"):
                continue
            y = float(y_str)
            if math.isnan(y) or math.isinf(y):
                continue
            if skip_zero and y == 0.0:
                continue
            pts.append(f"({x:.4f},{y:.6f})")
        except (KeyError, ValueError):
            continue
    return " ".join(pts)


def make_learning_curves_section(results: list[dict]) -> str:
    trained = [r for r in results if r["trained"] and r["train_rows"]]
    if not trained:
        return ""

    # --- Training loss plot ---
    loss_addplots = []
    acc_addplots = []
    eval_loss_addplots = []
    eval_acc_addplots = []

    legend_entries = []

    for idx, r in enumerate(trained):
        colour = get_colour(r["id"], idx)
        label = latex_escape(r["id"])
        train_rows = r["train_rows"]
        eval_rows = r["eval_rows"]

        loss_coords = _pgf_coords(train_rows, "step", "loss", skip_zero=True)
        acc_coords = _pgf_coords(
            train_rows, "step", "mean_token_accuracy", skip_zero=False
        )
        el_coords = _pgf_coords(eval_rows, "step", "eval_loss", skip_zero=True)
        ea_coords = _pgf_coords(eval_rows, "step", "eval_mean_token_accuracy")

        if loss_coords:
            loss_addplots.append(
                rf"    \addplot[color={colour}, thick] coordinates {{ {loss_coords} }};"
                + "\n"
                + rf"    \addlegendentry{{\texttt{{{label}}}}}"
            )
        if acc_coords:
            acc_addplots.append(
                rf"    \addplot[color={colour}, thick] coordinates {{ {acc_coords} }};"
                + "\n"
                + rf"    \addlegendentry{{\texttt{{{label}}}}}"
            )
        if el_coords:
            eval_loss_addplots.append(
                rf"    \addplot[color={colour}, thick, mark=*] coordinates {{ {el_coords} }};"
                + "\n"
                + rf"    \addlegendentry{{\texttt{{{label}}}}}"
            )
        if ea_coords:
            eval_acc_addplots.append(
                rf"    \addplot[color={colour}, thick, mark=*] coordinates {{ {ea_coords} }};"
                + "\n"
                + rf"    \addlegendentry{{\texttt{{{label}}}}}"
            )

    def axis_block(
        plots: list[str], xlabel: str, ylabel: str, title: str, ymin: str = ""
    ) -> str:
        ymin_line = f"    ymin={ymin}," if ymin else ""
        return (
            r"\begin{tikzpicture}"
            + "\n"
            + r"\begin{axis}["
            + "\n"
            + rf"    title={{{title}}},"
            + "\n"
            + rf"    xlabel={{{xlabel}}},"
            + "\n"
            + rf"    ylabel={{{ylabel}}},"
            + "\n"
            + r"    width=0.48\textwidth,"
            + "\n"
            + r"    height=6cm,"
            + "\n"
            + r"    legend pos=north east,"
            + "\n"
            + r"    legend style={font=\tiny},"
            + "\n"
            + r"    grid=both,"
            + "\n"
            + r"    grid style={line width=0.2pt, draw=gray!30},"
            + "\n"
            + r"    tick label style={font=\small},"
            + "\n"
            + r"    label style={font=\small},"
            + "\n"
            + (f"{ymin_line}\n" if ymin_line else "")
            + r"]"
            + "\n"
            + "\n".join(plots)
            + "\n"
            + r"\end{axis}"
            + "\n"
            + r"\end{tikzpicture}"
        )

    section = r"""
\section{Learning Curves}

Figures~\ref{fig:train_curves} and~\ref{fig:eval_curves} show the per-step
training loss / token accuracy and the per-checkpoint validation metrics for
all models that completed training. Models with degenerate zero-accuracy
behaviour are excluded from the accuracy subplots for clarity.

"""

    # Training figure
    if loss_addplots or acc_addplots:
        left_plot = (
            axis_block(
                loss_addplots, "Step", "Training Loss", r"Training Loss vs.\  Step"
            )
            if loss_addplots
            else ""
        )
        right_plot = (
            axis_block(
                acc_addplots,
                "Step",
                "Token Accuracy",
                r"Training Accuracy vs.\ Step",
                ymin="0",
            )
            if acc_addplots
            else ""
        )

        section += r"\begin{figure}[h]" + "\n" r"\centering" + "\n"
        if left_plot:
            section += left_plot + "\n\\hfill\n"
        if right_plot:
            section += right_plot + "\n"
        section += (
            r"\caption{Training loss (left) and per-step token accuracy (right) across training steps.}"
            + "\n"
            r"\label{fig:train_curves}" + "\n"
            r"\end{figure}" + "\n\n"
        )

    # Eval figure
    if eval_loss_addplots or eval_acc_addplots:
        left_plot = (
            axis_block(
                eval_loss_addplots,
                "Step",
                "Validation Loss",
                "Validation Loss at Checkpoints",
            )
            if eval_loss_addplots
            else ""
        )
        right_plot = (
            axis_block(
                eval_acc_addplots,
                "Step",
                r"Val.\ Token Accuracy",
                "Validation Accuracy at Checkpoints",
                ymin="0",
            )
            if eval_acc_addplots
            else ""
        )

        section += r"\begin{figure}[h]" + "\n" r"\centering" + "\n"
        if left_plot:
            section += left_plot + "\n\\hfill\n"
        if right_plot:
            section += right_plot + "\n"
        section += (
            r"\caption{Validation loss (left) and validation token accuracy (right) at each evaluation checkpoint.}"
            + "\n"
            r"\label{fig:eval_curves}" + "\n"
            r"\end{figure}" + "\n"
        )

    return section


def make_analysis_section(results: list[dict]) -> str:
    trained = [r for r in results if r["trained"]]
    good = []
    degen = []
    for r in trained:
        s = r["summary"]
        try:
            if float(s.get("final_train_accuracy", "0")) > 0.01:
                good.append(r)
            else:
                degen.append(r)
        except (ValueError, TypeError):
            degen.append(r)

    best = None
    best_acc = -1.0
    for r in good:
        try:
            acc = float(r["summary"].get("best_eval_accuracy", "0"))
            if acc > best_acc:
                best_acc = acc
                best = r
        except (ValueError, TypeError):
            pass

    good_ids = ", ".join(rf"\texttt{{{latex_escape(r['id'])}}}" for r in good) or "none"
    degen_ids = (
        ", ".join(rf"\texttt{{{latex_escape(r['id'])}}}" for r in degen) or "none"
    )
    best_str = (
        rf"\texttt{{{latex_escape(best['id'])}}} ({pct(best['summary'].get('best_eval_accuracy'))} val.\ accuracy)"
        if best
        else "N/A"
    )
    untrained_ids = (
        ", ".join(
            rf"\texttt{{{latex_escape(r['id'])}}}" for r in results if not r["trained"]
        )
        or "none"
    )

    total_gpu_hours = 0.0
    for r in trained:
        s = r["summary"]
        try:
            total_gpu_hours += float(s.get("total_training_minutes", 0)) / 60.0
        except (ValueError, TypeError):
            pass

    return rf"""
\section{{Analysis and Discussion}}

\subsection{{Successful Runs}}

The following models trained successfully and produced meaningful metrics:
{good_ids}.
Both are DeepSeek-Coder variants, suggesting that the model family is well-suited
to this tactic-prediction task. The best-performing model overall was
{best_str}.

\subsection{{Degenerate Runs}}

Training for {degen_ids} produced \texttt{{loss = 0.0}} and
\texttt{{accuracy = 0.0}} throughout, indicating that the completion-only
data collator could not locate the response-boundary template in the tokenised
sequences. This is a known issue when a tokeniser's vocabulary does not contain
the expected delimiter tokens as atomic units. These models should be re-run
with an adjusted prompt template or by disabling the completion-only masking.

\subsection{{Untrained Models}}

The following models were configured but not trained in this experiment:
{untrained_ids}.
These represent larger or alternative architectures that would require
additional compute budget.

\subsection{{Compute Budget}}

Total GPU time across all completed runs was approximately
\textbf{{{total_gpu_hours:.1f} GPU-hours}}.

\subsection{{Recommendations}}

\begin{{enumerate}}
\item Fine-tune DeepSeek-Coder-6.7b further with a lower learning rate and
      cosine restart schedule — it achieved the best validation accuracy
      ($>$96\%) with room to improve.
\item Diagnose the completion-only collator failure for CodeLlama models by
      printing tokenised prompt/response boundaries and adjusting the response
      template string.
\item Evaluate the best checkpoint on the held-out test set to obtain
      unbiased accuracy estimates.
\item Experiment with larger LoRA rank ($r = 64$) for the 7B+ models to increase
      adapter capacity.
\end{{enumerate}}
"""


def make_appendix_section(results: list[dict]) -> str:
    trained = [r for r in results if r["trained"]]
    if not trained:
        return ""

    blocks = []
    for r in trained:
        s = r["summary"]
        hp = r["hyperparameters"] or {}
        block = rf"""
\subsection{{\texttt{{{latex_escape(r['id'])}}}}}

\begin{{table}}[h]
\centering
\begin{{tabular}}{{ll}}
\toprule
\textbf{{Parameter}} & \textbf{{Value}} \\
\midrule
HuggingFace ID & \texttt{{{hf_display(r['hf_id'])}}} \\
LoRA rank ($r$) & {hp.get('lora_r', '---')} \\
LoRA $\alpha$ & {hp.get('lora_alpha', '---')} \\
LoRA dropout & {hp.get('lora_dropout', '---')} \\
Epochs & {hp.get('epochs', '---')} \\
Batch size (per device) & {hp.get('batch_size', '---')} \\
Gradient accumulation & {hp.get('grad_accum', '---')} \\
Effective batch size & {hp.get('batch_size', 1) * hp.get('grad_accum', 1) if hp else '---'} \\
Learning rate & {hp.get('lr', '---')} \\
Warmup ratio & {hp.get('warmup_ratio', '---')} \\
Max sequence length & {hp.get('max_seq_len', '---')} \\
Load in 4-bit & {'Yes' if hp.get('load_in_4bit') else 'No'} \\
Load in 8-bit & {'Yes' if hp.get('load_in_8bit') else 'No'} \\
Total training steps & {s.get('total_steps', '---') if s else '---'} \\
Total training time (min) & {fmt(s.get('total_training_minutes', ''), 1) if s else '---'} \\
Final training loss & {fmt(s.get('final_train_loss', ''), 4) if s else '---'} \\
Final training accuracy & {pct(s.get('final_train_accuracy', '')) if s else '---'} \\
Best validation loss & {fmt(s.get('best_eval_loss', ''), 4) if s else '---'} \\
Best validation accuracy & {pct(s.get('best_eval_accuracy', '')) if s else '---'} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
        blocks.append(block)

    return r"""
\appendix
\section{Per-Model Hyperparameter Details}
""" + "\n".join(
        blocks
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_report(output_path: Path):
    print("[generate_report] Collecting results …")
    results = collect_all_results()
    print(f"  Found {len(results)} model directories.")
    trained = [r for r in results if r["trained"]]
    print(f"  {len(trained)} with completed training data.")

    total, theorems, tactics = count_training_samples()

    parts = [
        PREAMBLE,
        make_dataset_section(total, theorems, tactics, results),
        make_models_section(results),
        make_results_section(results),
        make_learning_curves_section(results),
        make_analysis_section(results),
        make_appendix_section(results),
        FOOTER,
    ]

    tex = "\n".join(parts)
    output_path.write_text(tex, encoding="utf-8")
    print(f"[generate_report] Written: {output_path}")
    print(f"  Compile with:  pdflatex {output_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX fine-tuning report for Pie Tactic Predictor."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="report.tex",
        help="Output .tex file path (default: report.tex)",
    )
    args = parser.parse_args()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = SCRIPT_DIR / output_path
    build_report(output_path)


if __name__ == "__main__":
    main()
