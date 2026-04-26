#!/usr/bin/env python3
"""1-epoch SFT on top of the RL Modifier & Correction checkpoint.

Loads sft_output/merged_tmp + RL LoRA, merges in memory, adds fresh LoRA,
trains 1 epoch on the full SFT dataset. Goal: recover from catastrophic
forgetting on categories that regressed during RL.

Usage:
    PYTHONPATH=src python scripts/training/sft/sft_on_rl.py
"""

import json
import glob
import os
import random

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from datasets import Dataset

from calendar_agent.core import format_tool_result
from calendar_agent.tools import get_openai_tools_minimal
from calendar_agent.paths import SFT_DATA_DIR as _SFT_DATA_DIR

random.seed(42)

# ── Paths ──────────────────────────────────────────────────
SFT_DATA_DIR = str(_SFT_DATA_DIR)
TRAJ_DIR = str(_SFT_DATA_DIR / "trajectories_augmented")
OUTPUT_DIR = "sft_on_rl_output"
RL_CHECKPOINT = "rl_runs/single_category_modifier_correction/checkpoint"
BASE_MODEL = "sft_output/merged_tmp"

# ── Model Config ───────────────────────────────────────────
MAX_SEQ_LENGTH = 3076
LORA_RANK = 64
NUM_EPOCHS = 1

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
            call_index = 0
            while i < len(steps) and steps[i]["role"] == "tool_call":
                tc = steps[i]
                tool_call_id = f"call_{call_index}"
                messages.append({
                    "role": "assistant", "content": "",
                    "tool_calls": [{
                        "type": "function", "id": tool_call_id,
                        "function": {"name": tc["name"], "arguments": tc["args"]},
                    }],
                })
                compacted = format_tool_result(tc["result"])
                messages.append({
                    "role": "tool",
                    "content": json.dumps(compacted, default=str),
                    "tool_call_id": tool_call_id,
                })
                call_index += 1
                i += 1
        elif step["role"] == "assistant":
            messages.append({"role": "assistant", "content": step["content"]})
            i += 1
        else:
            i += 1
    return messages


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
    random.shuffle(trajs)
    print(f"Loaded {len(trajs)} solved trajectories")

    all_input_ids = []
    all_attention_mask = []
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

        tokens = tokenizer(text, truncation=True, max_length=MAX_SEQ_LENGTH,
                           padding=False, return_tensors=None)
        if len(tokens["input_ids"]) > MAX_SEQ_LENGTH:
            long_skipped += 1
            continue

        all_input_ids.append(tokens["input_ids"])
        all_attention_mask.append(tokens["attention_mask"])

    total = len(all_input_ids)
    print(f"Converted & tokenized {total} trajectories ({skipped} skipped, {long_skipped} too long)")

    return Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
    })


# ── Main ───────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("SFT Recovery: 1 epoch on top of RL Modifier checkpoint")
    print(f"  Base model:    {BASE_MODEL}")
    print(f"  RL checkpoint: {RL_CHECKPOINT}")
    print(f"  LoRA rank:     {LORA_RANK}")
    print(f"  Output:        {OUTPUT_DIR}")
    print("=" * 60)

    # Step 1: Merge RL LoRA into base and save as fp16
    merged_tmp = os.path.join(OUTPUT_DIR, "_merged_rl_tmp")
    if not os.path.exists(os.path.join(merged_tmp, "config.json")):
        print("\n[1/5] Merging RL LoRA into base model (fp16)...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=RL_CHECKPOINT,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,
        )
        model.save_pretrained_merged(merged_tmp, tokenizer, save_method="merged_16bit")
        print(f"  Saved merged model to {merged_tmp}")
        # Free memory before reloading
        del model, tokenizer
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("\n[1/5] Merged model already exists, skipping...")

    # Step 2: Load merged model in 4-bit for training
    print("\n[2/5] Loading merged RL model (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=merged_tmp,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    # Restore chat template if Unsloth stripped it
    if not getattr(tokenizer, "chat_template", None):
        from transformers import AutoTokenizer as _AT
        _ref_tok = _AT.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        tokenizer.chat_template = _ref_tok.chat_template
        print("  Restored chat template from base tokenizer")

    # Step 3: Add fresh LoRA adapters for SFT
    print("\n[3/5] Adding fresh LoRA adapters...")
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

    # Step 4: Build dataset
    print("\n[4/5] Building dataset...")
    dataset = build_dataset(tokenizer)

    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    val_dataset = split["test"]
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Loss masking collator
    response_template = "<|im_start|>assistant\n"
    instruction_template = "<|im_start|>user\n"

    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template=response_template,
        instruction_template=instruction_template,
    )

    # Verify masking
    import torch
    sample = train_dataset[0]
    input_ids = torch.tensor(sample["input_ids"])
    attention_mask = torch.tensor(sample["attention_mask"])
    batch = collator([{"input_ids": input_ids, "attention_mask": attention_mask}])
    labels = batch["labels"][0]
    total_trained = (labels != -100).sum().item()
    total_masked = (labels == -100).sum().item()
    pct = total_trained / (total_trained + total_masked) * 100
    print(f"  Loss masking: {pct:.1f}% tokens trained, {100-pct:.1f}% masked")
    if pct < 1:
        print("  ERROR: No tokens being trained!")
        return

    steps_per_epoch = max(1, len(train_dataset) // 4)
    print(f"  Steps per epoch: ~{steps_per_epoch}")

    # Step 5: Train
    print("\n[5/5] Training 1 epoch...")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=False,
        bf16=True,
        bf16_full_eval=True,
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

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        args=training_args,
    )

    trainer.train()

    # Save final
    final_dir = os.path.join(OUTPUT_DIR, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nModel saved to {final_dir}")

    history_path = os.path.join(OUTPUT_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"Training history saved to {history_path}")
    print("Done!")


if __name__ == "__main__":
    main()
