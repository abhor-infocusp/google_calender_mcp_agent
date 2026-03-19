#!/usr/bin/env python3
"""SFT training script for Qwen3-4B on solved calendar agent trajectories.

Uses Unsloth for efficient 4-bit LoRA fine-tuning with TRL's SFTTrainer.
Loads solved trajectories from sft_data/trajectories/ and converts them
to Qwen3 tool-calling chat format for supervised fine-tuning.

Usage:
    python sft_training.py
"""

import json
import glob
import os
import random

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

from calendar_agent.core import SYSTEM_PROMPT
from calendar_agent.tools import get_openai_tools, compact_tool_result
from calendar_agent.paths import SFT_DATA_DIR as _SFT_DATA_DIR, SFT_OUTPUT_DIR as _SFT_OUTPUT_DIR

random.seed(42)

# ── Paths ──────────────────────────────────────────────────

SFT_DATA_DIR = str(_SFT_DATA_DIR)
TRAJ_DIR = str(_SFT_DATA_DIR / "trajectories")
OUTPUT_DIR = str(_SFT_OUTPUT_DIR)

# ── Model Config ───────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LENGTH = 4096  # 160/161 trajectories fit after result compaction; 1 outlier dropped
LORA_RANK = 32

TOOLS = get_openai_tools()


# ── Trajectory to Chat Conversion ─────────────────────────


def trajectory_to_messages(traj: dict) -> list[dict]:
    """Convert a solved trajectory to Qwen3 chat messages with tool calling.

    The raw trajectory format has steps like:
        {"role": "user", "content": "..."}
        {"role": "tool_call", "name": "...", "args": {...}, "result": {...}}
        {"role": "tool_call", "name": "...", "args": {...}, "result": {...}}
        {"role": "assistant", "content": "..."}

    Each tool_call is serialized into its own assistant turn (single tool_call)
    followed by the tool response. This teaches the model to read each result
    before deciding the next call, rather than batching calls in parallel.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    steps = traj["trajectory"]

    i = 0
    while i < len(steps):
        step = steps[i]

        if step["role"] == "user":
            messages.append({"role": "user", "content": step["content"]})
            i += 1

        elif step["role"] == "tool_call":
            # Serialize: each tool_call gets its own assistant turn so the
            # model learns to read each result before deciding the next call.
            call_index = 0
            while i < len(steps) and steps[i]["role"] == "tool_call":
                tc = steps[i]
                tool_call_id = f"call_{call_index}"
                tool_call_obj = {
                    "type": "function",
                    "id": tool_call_id,
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["args"],
                    },
                }

                # One assistant message with a single tool_call
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [tool_call_obj],
                })

                # Corresponding tool response
                compacted = compact_tool_result(tc["name"], tc["result"])
                result_str = json.dumps(compacted, default=str)
                messages.append({
                    "role": "tool",
                    "content": result_str,
                    "tool_call_id": tool_call_id,
                })

                call_index += 1
                i += 1

        elif step["role"] == "assistant":
            messages.append({"role": "assistant", "content": step["content"]})
            i += 1

        elif step["role"] == "error":
            # Skip error steps
            i += 1

        else:
            i += 1

    return messages


def load_trajectories() -> list[dict]:
    """Load all solved trajectories from sft_data/trajectories/."""
    all_trajs = []
    for f in sorted(glob.glob(os.path.join(TRAJ_DIR, "*.json"))):
        data = json.load(open(f))
        all_trajs.extend(data)
    return all_trajs


def build_dataset(tokenizer) -> Dataset:
    """Load trajectories, convert to chat format, and tokenize."""
    trajs = load_trajectories()
    random.shuffle(trajs)
    print(f"Loaded {len(trajs)} solved trajectories")

    texts = []
    skipped = 0
    for traj in trajs:
        messages = trajectory_to_messages(traj)
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tools=TOOLS,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  Skipped trajectory: {e}")

    print(f"Converted {len(texts)} trajectories ({skipped} skipped)")

    # Filter by token length
    before = len(texts)
    filtered_texts = []
    for text in texts:
        n_tokens = len(tokenizer.encode(text))
        if n_tokens <= MAX_SEQ_LENGTH:
            filtered_texts.append(text)
    print(f"After length filter (<={MAX_SEQ_LENGTH} tokens): {len(filtered_texts)}/{before}")

    return Dataset.from_dict({"text": filtered_texts})


# ── Main ───────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("SFT Training: Qwen2.5-1.5B-Instruct on Calendar Agent Trajectories")
    print("=" * 60)

    # Load model with Unsloth
    print(f"\nLoading model: {MODEL_NAME}")
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

    # Train/val split
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    val_dataset = split["test"]
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Training config
    num_epochs = 5
    # ~36 steps/epoch with 145 train samples, batch=1, grad_accum=4
    steps_per_epoch = max(1, len(train_dataset) // 4)
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=num_epochs,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=num_epochs,
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,
        seed=42,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
    )

    print(f"\nStarting training...")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Steps/epoch: ~{steps_per_epoch}")
    print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Save strategy: per-epoch")
    print()

    trainer.train()

    # Save final model
    final_dir = os.path.join(OUTPUT_DIR, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nModel saved to {final_dir}")
    print("Training complete.")


if __name__ == "__main__":
    main()
