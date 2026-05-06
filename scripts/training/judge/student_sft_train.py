#!/usr/bin/env python3
"""SFT training for the student judge (Qwen3-8B).

Forked from scripts/training/judge/judge_sft_train.py with the only change
being the data source: reads
  data/judge/v2_20260502/student_sft_{train,dev}.jsonl
produced by scripts/data_generation/student_judge_assemble.py.

Each row has:
  - messages: [system (router-qwen-v2 + /no_think), user (judge prompt),
               assistant (Qwen-v2 teacher CoT + verdict)]
  - meta: sid, rollout_hash, cat, label, label_source, ...

Loss is masked to assistant tokens only (the CoT + verdict). Output
schema is intentionally identical to judge_sft_train.py so downstream
serving (vLLM hermes parser) works unchanged.

Usage:
    PYTHONPATH=src python scripts/training/judge/student_sft_train.py
"""
from __future__ import annotations
import csv
import json
import os
import random

from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from transformers import TrainerCallback

from calendar_agent.paths import PROJECT_ROOT

random.seed(42)

# ── Paths ──────────────────────────────────────────────────
TRAIN_JSONL = os.environ.get(
    "STUDENT_TRAIN_JSONL",
    str(PROJECT_ROOT / "data/judge/v2_20260502/student_sft_train.jsonl"),
)
DEV_JSONL = os.environ.get(
    "STUDENT_DEV_JSONL",
    str(PROJECT_ROOT / "data/judge/v2_20260502/student_sft_dev.jsonl"),
)
OUTPUT_DIR = os.environ.get(
    "STUDENT_RUN_DIR",
    str(PROJECT_ROOT / "runs/student_judge_qwen3_8b_20260504"),
)
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOSS_CSV = os.path.join(OUTPUT_DIR, "diagnostics", "epoch_losses.csv")

# ── Model Config ───────────────────────────────────────────
MODEL_NAME = os.environ.get("STUDENT_MODEL_NAME", "Qwen/Qwen3-8B")
MAX_SEQ_LENGTH = 4096
LORA_RANK = 64
NUM_EPOCHS = int(os.environ.get("STUDENT_EPOCHS", "3"))


def load_jsonl_messages(path: str) -> list[list[dict]]:
    msgs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "messages" not in d:
                raise ValueError(f"row missing `messages` field in {path}")
            msgs.append(d["messages"])
    return msgs


def compute_assistant_labels(input_ids: list[int], assistant_header_ids: list[int], im_end_id: int) -> list[int]:
    labels = [-100] * len(input_ids)
    marker_len = len(assistant_header_ids)
    i = 0
    while i <= len(input_ids) - marker_len:
        if input_ids[i : i + marker_len] == assistant_header_ids:
            content_start = i + marker_len
            j = content_start
            while j < len(input_ids):
                if input_ids[j] == im_end_id:
                    for k in range(content_start, j + 1):
                        labels[k] = input_ids[k]
                    i = j + 1
                    break
                j += 1
            else:
                for k in range(content_start, len(input_ids)):
                    labels[k] = input_ids[k]
                break
        else:
            i += 1
    return labels


class AssistantOnlyCollator:
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


def build_split_dataset(tokenizer, jsonl_path: str, label: str) -> Dataset:
    msgs_list = load_jsonl_messages(jsonl_path)
    print(f"  [{label}] loaded {len(msgs_list)} examples from {jsonl_path}")

    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assistant_nl_ids = tokenizer.encode("assistant\n", add_special_tokens=False)
    assistant_header_ids = [im_start_id] + assistant_nl_ids

    all_input_ids, all_attention_mask, all_labels = [], [], []
    skipped = long_skipped = 0
    for messages in msgs_list:
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  [{label}] skipped: {e}")
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

    print(f"  [{label}] tokenized {len(all_input_ids)} (skipped {skipped}, too-long {long_skipped})")
    if all_input_ids:
        s_labels = all_labels[0]
        trained = sum(1 for l in s_labels if l != -100)
        masked = sum(1 for l in s_labels if l == -100)
        pct = trained / max(1, trained + masked) * 100
        print(f"  [{label}] sample 0: trained tokens {trained} ({pct:.1f}%), masked {masked}")

    return Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    })


class EpochLossLogger(TrainerCallback):
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
            csv.writer(f).writerow([epoch, f"{train_loss:.6f}", f"{eval_loss:.6f}"])
        print(f"\n>>> Epoch {epoch}: train_loss={train_loss:.4f}, eval_loss={eval_loss:.4f}")
        self.pending_train_loss = None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOSS_CSV), exist_ok=True)

    from transformers.trainer_utils import get_last_checkpoint
    last_checkpoint = get_last_checkpoint(CHECKPOINT_DIR)
    resume = last_checkpoint is not None
    if resume:
        print(f"Resuming from checkpoint: {last_checkpoint}")

    print("=" * 60)
    print(f"Student Judge SFT: {MODEL_NAME} — {NUM_EPOCHS} epochs")
    print(f"  LoRA rank:   {LORA_RANK}")
    print(f"  Train:       {TRAIN_JSONL}")
    print(f"  Dev:         {DEV_JSONL}")
    print(f"  Output:      {OUTPUT_DIR}")
    print("=" * 60)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    if not getattr(tokenizer, "chat_template", None):
        from transformers import AutoTokenizer as _AT
        _ref_tok = _AT.from_pretrained(MODEL_NAME)
        tokenizer.chat_template = _ref_tok.chat_template
        print("Restored chat template from base tokenizer")

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK,
        lora_dropout=0,
        use_gradient_checkpointing="unsloth",
    )

    train_dataset = build_split_dataset(tokenizer, TRAIN_JSONL, "train")
    dev_dataset = build_split_dataset(tokenizer, DEV_JSONL, "dev")
    print(f"Train: {len(train_dataset)}, Dev: {len(dev_dataset)}")

    collator = AssistantOnlyCollator(pad_token_id=tokenizer.pad_token_id)

    steps_per_epoch = max(1, len(train_dataset) // 4)
    print(f"Steps per epoch: ~{steps_per_epoch}; total: ~{steps_per_epoch * NUM_EPOCHS}")

    training_args = SFTConfig(
        output_dir=CHECKPOINT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=NUM_EPOCHS,
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

    loss_logger = EpochLossLogger(LOSS_CSV, resume=resume)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collator,
        args=training_args,
        callbacks=[loss_logger],
    )

    trainer.train(resume_from_checkpoint=last_checkpoint)

    final_dir = os.path.join(CHECKPOINT_DIR, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nModel saved to {final_dir}")

    history_path = os.path.join(OUTPUT_DIR, "diagnostics", "training_history.json")
    with open(history_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"Training history saved to {history_path}")
    print("Training complete.")


if __name__ == "__main__":
    main()
