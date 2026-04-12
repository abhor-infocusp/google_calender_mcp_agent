#!/usr/bin/env python3
"""SFT training for 100 epochs with loss masking, per-epoch checkpoints.

Key improvements over base sft_train.py:
  1. Loss masking — only trains on assistant tokens (tool calls + final answers),
     not system prompt, user queries, or tool results. 10.6% → 100% useful signal.
  2. LR schedule — cosine_with_restarts (10 cycles) so LR doesn't decay to 0.
  3. LoRA rank 64 — more capacity for multi-step tool-use patterns.

Usage:
    PYTHONPATH=src python scripts/training/sft_train_100ep.py
"""

import csv
import json
import glob
import os
import random

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from transformers import TrainerCallback

from calendar_agent.core import format_tool_result
from calendar_agent.tools import get_openai_tools_minimal
from calendar_agent.paths import SFT_DATA_DIR as _SFT_DATA_DIR, SFT_OUTPUT_DIR as _SFT_OUTPUT_DIR

random.seed(42)

# ── Paths ──────────────────────────────────────────────────
SFT_DATA_DIR = str(_SFT_DATA_DIR)
TRAJ_DIR = str(_SFT_DATA_DIR / "trajectories_augmented")
OUTPUT_DIR = str(_SFT_OUTPUT_DIR)
LOSS_CSV = os.path.join(OUTPUT_DIR, "epoch_losses.csv")

# ── Model Config ───────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LENGTH = 3076
LORA_RANK = 64
NUM_EPOCHS = 10

TOOLS = get_openai_tools_minimal()


# ── Trajectory to Chat Conversion ─────────────────────────

def trajectory_to_messages(traj: dict) -> list[dict]:
    messages = []
    steps = traj["trajectory"]
    i = 0
    while i < len(steps):
        step = steps[i]
        if step["role"] == "user":
            messages.append({"role": "user", "content": step["content"]})
            i += 1
        elif step["role"] == "tool_call":
            # Check if previous step was assistant narration — merge it as content
            narration = ""
            if messages and messages[-1]["role"] == "assistant" and "tool_calls" not in messages[-1]:
                narration = messages.pop()["content"]

            call_index = 0
            while i < len(steps) and steps[i]["role"] == "tool_call":
                tc = steps[i]
                tool_call_id = f"call_{call_index}"
                messages.append({
                    "role": "assistant",
                    "content": narration if call_index == 0 else "",
                    "tool_calls": [{
                        "type": "function", "id": tool_call_id,
                        "function": {"name": tc["name"], "arguments": tc["args"]},
                    }],
                })
                compacted = format_tool_result(tc["result"])
                messages.append({
                    "role": "tool",
                    "content": compacted if isinstance(compacted, str) else json.dumps(compacted, default=str),
                    "tool_call_id": tool_call_id,
                })
                narration = ""  # only first tool call gets the narration
                call_index += 1
                i += 1
        elif step["role"] == "assistant":
            messages.append({"role": "assistant", "content": step["content"]})
            i += 1
        else:
            i += 1
    return messages


def compute_assistant_labels(input_ids: list[int], assistant_header_ids: list[int], im_end_id: int) -> list[int]:
    """Build labels that train only on assistant content (not user/system/tool).

    Scans for <|im_start|>assistant\\n ... <|im_end|> sections and unmasks
    the content + <|im_end|> token. Everything else is masked (-100).
    """
    labels = [-100] * len(input_ids)
    marker_len = len(assistant_header_ids)
    i = 0
    while i <= len(input_ids) - marker_len:
        if input_ids[i : i + marker_len] == assistant_header_ids:
            content_start = i + marker_len
            j = content_start
            while j < len(input_ids):
                if input_ids[j] == im_end_id:
                    # Unmask assistant content + <|im_end|> token
                    for k in range(content_start, j + 1):
                        labels[k] = input_ids[k]
                    i = j + 1
                    break
                j += 1
            else:
                # No <|im_end|> found — unmask to end
                for k in range(content_start, len(input_ids)):
                    labels[k] = input_ids[k]
                break
        else:
            i += 1
    return labels


class AssistantOnlyCollator:
    """Pads input_ids, attention_mask, and pre-computed labels."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        max_len = max(len(e["input_ids"]) for e in examples)
        input_ids, attention_mask, labels = [], [], []
        for e in examples:
            pad_len = max_len - len(e["input_ids"])
            input_ids.append(e["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(e["attention_mask"] + [0] * pad_len)
            labels.append(e["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }


def load_trajectories() -> list[dict]:
    all_trajs = []
    for f in sorted(glob.glob(os.path.join(TRAJ_DIR, "*.json"))):
        if os.path.basename(f).startswith("_"):
            continue
        data = json.load(open(f))
        all_trajs.extend(data)
    return all_trajs


def build_dataset(tokenizer) -> Dataset:
    trajs = load_trajectories()
    # Filter out error-ending trajectories (teacher model failures)
    before = len(trajs)
    trajs = [t for t in trajs if not any(
        s.get("role") == "error" for s in t.get("trajectory", [])
    )]
    if len(trajs) < before:
        print(f"Filtered {before - len(trajs)} error-ending trajectories")
    random.shuffle(trajs)
    print(f"Loaded {len(trajs)} solved trajectories")

    # Build marker token sequences for assistant-only label masking
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assistant_nl_ids = tokenizer.encode("assistant\n", add_special_tokens=False)
    assistant_header_ids = [im_start_id] + assistant_nl_ids
    print(f"  Assistant header IDs: {assistant_header_ids} (len={len(assistant_header_ids)})")
    print(f"  <|im_end|> ID: {im_end_id}")

    all_input_ids = []
    all_attention_mask = []
    all_labels = []
    skipped = 0
    long_skipped = 0
    for traj in trajs:
        messages = trajectory_to_messages(traj)
        try:
            text = tokenizer.apply_chat_template(
                messages, tools=TOOLS, tokenize=False, add_generation_prompt=False,
            )
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  Skipped trajectory: {e}")
            continue

        tokens = tokenizer(text, truncation=False, padding=False, return_tensors=None)
        if len(tokens["input_ids"]) > MAX_SEQ_LENGTH:
            long_skipped += 1
            continue

        input_ids = tokens["input_ids"]
        labels = compute_assistant_labels(input_ids, assistant_header_ids, im_end_id)

        all_input_ids.append(input_ids)
        all_attention_mask.append(tokens["attention_mask"])
        all_labels.append(labels)

    total = len(all_input_ids)
    print(f"Converted & tokenized {total} trajectories ({skipped} skipped, {long_skipped} too long)")

    # Verify label masking on first example
    if total > 0:
        sample_labels = all_labels[0]
        sample_ids = all_input_ids[0]
        trained = sum(1 for l in sample_labels if l != -100)
        masked = sum(1 for l in sample_labels if l == -100)
        pct = trained / (trained + masked) * 100
        print(f"\n  Loss masking verification (sample 0):")
        print(f"    Trained tokens:  {trained} ({pct:.1f}%)")
        print(f"    Masked tokens:   {masked} ({100-pct:.1f}%)")
        trained_ids = [sample_ids[i] for i, l in enumerate(sample_labels) if l != -100]
        if trained_ids:
            snippet = tokenizer.decode(trained_ids[:50])
            print(f"    First trained tokens: '{snippet[:120]}...'")

    ds = Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    })
    return ds



# ── Epoch Loss Logger Callback ─────────────────────────────

class EpochLossLogger(TrainerCallback):
    """Log training and eval loss per epoch to a CSV file."""

    def __init__(self, csv_path, resume=False):
        self.csv_path = csv_path
        self.epoch_train_losses = []
        self.pending_train_loss = None
        if not resume or not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "eval_loss"])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.epoch_train_losses.append(logs["loss"])

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_train_losses:
            self.pending_train_loss = sum(self.epoch_train_losses) / len(self.epoch_train_losses)
        else:
            self.pending_train_loss = float("nan")
        self.epoch_train_losses = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        epoch = int(state.epoch)
        train_loss = self.pending_train_loss if self.pending_train_loss is not None else float("nan")
        eval_loss = metrics.get("eval_loss", float("nan")) if metrics else float("nan")

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{eval_loss:.6f}"])

        print(f"\n>>> Epoch {epoch}: train_loss={train_loss:.4f}, eval_loss={eval_loss:.4f}")
        self.pending_train_loss = None


# ── Main ───────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check for existing checkpoints to resume from
    from transformers.trainer_utils import get_last_checkpoint
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
    resume = last_checkpoint is not None
    if resume:
        print(f"Resuming from checkpoint: {last_checkpoint}")

    print("=" * 60)
    print(f"SFT Training: {MODEL_NAME} — {NUM_EPOCHS} epochs")
    print(f"  Loss masking: assistant-only (pre-computed labels)")
    print(f"  LR schedule:  cosine_with_restarts (10 cycles)")
    print(f"  LoRA rank:    {LORA_RANK}")
    print(f"  Output:       {OUTPUT_DIR}")
    if resume:
        print(f"  RESUMING from: {last_checkpoint}")
    print("=" * 60)

    # Load model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    # Restore chat template if Unsloth stripped it
    if not getattr(tokenizer, "chat_template", None):
        from transformers import AutoTokenizer as _AT
        _ref_tok = _AT.from_pretrained(MODEL_NAME)
        tokenizer.chat_template = _ref_tok.chat_template
        print("Restored chat template from base tokenizer")

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_RANK,
        lora_dropout=0,
        use_gradient_checkpointing="unsloth",
    )

    # Build dataset
    dataset = build_dataset(tokenizer)

    # Train/val split (same seed as original)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    val_dataset = split["test"]
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # ── Collator (labels pre-computed in dataset) ──
    collator = AssistantOnlyCollator(pad_token_id=tokenizer.pad_token_id)

    steps_per_epoch = max(1, len(train_dataset) // 4)  # batch=1, grad_accum=4
    print(f"\nSteps per epoch: ~{steps_per_epoch}")
    print(f"Total steps: ~{steps_per_epoch * NUM_EPOCHS}")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=2e-4,
        lr_scheduler_type="cosine_with_restarts",
        lr_scheduler_kwargs={"num_cycles": 5},
        warmup_ratio=0.03,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=NUM_EPOCHS,
        fp16=True,
        bf16=False,
        fp16_full_eval=True,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        optim="paged_adamw_8bit",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        seed=42,
        report_to="none",
        load_best_model_at_end=False,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    # Loss logger callback
    loss_logger = EpochLossLogger(LOSS_CSV, resume=resume)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        args=training_args,
        callbacks=[loss_logger],
    )

    print(f"\nStarting training: {NUM_EPOCHS} epochs, ~{steps_per_epoch} steps/epoch")
    print(f"Loss CSV: {LOSS_CSV}")
    print()

    trainer.train(resume_from_checkpoint=last_checkpoint)

    # Save final model
    final_dir = os.path.join(OUTPUT_DIR, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nModel saved to {final_dir}")

    # Also save the full training history
    history_path = os.path.join(OUTPUT_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"Training history saved to {history_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
