#!/usr/bin/env python3
"""Packing throughput benchmark — fork of judge_sft_train.py.

Goal: measure step time / tokens-per-second with TRL's built-in sequence packing
to see whether it beats the unpacked-bs=4 wall (~3.6 s/step on 1g.24gb MIG slice).

Differences vs judge_sft_train.py:
  - packing=True (TRL packs short examples into PACK_LEN-token sequences)
  - drops the custom AssistantOnlyCollator + skip_prepare_dataset
  - lets TRL handle templating from the {messages: [...]} dataset
  - max_steps=50 (benchmark only — not a full training run)
  - no eval, no LoRA-rank changes, same model/quant as the main run

Loss-mask caveat: TRL's default packed training does NOT do assistant-only loss
masking with the Qwen3 chat template (no {% generation %} markers). This is a
THROUGHPUT TEST only — quality is not comparable. If packing wins on speed,
follow up with a proper packed+masked training run.
"""
import csv
import json
import os
import random

from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from transformers import TrainerCallback

random.seed(42)

from calendar_agent.paths import PROJECT_ROOT  # noqa

TRAIN_JSONL = os.environ.get("JUDGE_TRAIN_JSONL", str(PROJECT_ROOT / "data/judge/v3_20260507/train.jsonl"))
VAL_JSONL = os.environ.get("JUDGE_VAL_JSONL", str(PROJECT_ROOT / "data/judge/v3_20260507/val.jsonl"))
OUTPUT_DIR = os.environ.get("JUDGE_RUN_DIR", str(PROJECT_ROOT / "runs/judge_v3_packed_bench_20260507"))
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_PATH = os.path.join(OUTPUT_DIR, "diagnostics", "step_times.csv")

MODEL_NAME = os.environ.get("JUDGE_MODEL_NAME", "Qwen/Qwen3-14B")
MAX_SEQ_LENGTH = int(os.environ.get("JUDGE_MAX_SEQ_LENGTH", "2048"))
LORA_RANK = int(os.environ.get("JUDGE_LORA_RANK", "64"))
PER_DEV_BS = int(os.environ.get("JUDGE_PER_DEV_BS", "4"))
GRAD_ACCUM = int(os.environ.get("JUDGE_GRAD_ACCUM", "1"))
MAX_STEPS = int(os.environ.get("JUDGE_MAX_STEPS", "50"))


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class StepTimeLogger(TrainerCallback):
    def __init__(self, csv_path):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "loss", "step_time_s", "samples_per_s"])
        self._t_last = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        import time
        now = time.time()
        dt = (now - self._t_last) if self._t_last else float("nan")
        self._t_last = now
        if logs and "loss" in logs:
            sps = logs.get("train_samples_per_second", float("nan"))
            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow([state.global_step, logs["loss"], f"{dt:.3f}", f"{sps}"])


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("=" * 60)
    print(f"Packing throughput benchmark: {MODEL_NAME}")
    print(f"  packing=True  max_seq_length={MAX_SEQ_LENGTH}  bs={PER_DEV_BS} accum={GRAD_ACCUM}")
    print(f"  max_steps={MAX_STEPS}")
    print(f"  output={OUTPUT_DIR}")
    print("=" * 60)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    if not getattr(tokenizer, "chat_template", None):
        from transformers import AutoTokenizer as _AT
        _ref = _AT.from_pretrained(MODEL_NAME)
        tokenizer.chat_template = _ref.chat_template

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

    # Dataset is just {messages: [...]} — let TRL handle templating + packing.
    train_rows = load_jsonl(TRAIN_JSONL)
    print(f"loaded {len(train_rows)} train rows from {TRAIN_JSONL}")
    # Pre-render to plain text via chat template so packing can splice freely.
    train_texts = [
        {"text": tokenizer.apply_chat_template(r["messages"], tokenize=False,
                                               add_generation_prompt=False)}
        for r in train_rows
    ]
    train_ds = Dataset.from_list(train_texts)

    cfg = SFTConfig(
        output_dir=CHECKPOINT_DIR,
        per_device_train_batch_size=PER_DEV_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_steps=MAX_STEPS,
        learning_rate=2e-4,
        lr_scheduler_type="constant",
        warmup_ratio=0.0,
        logging_steps=5,
        save_strategy="no",
        eval_strategy="no",
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=True,
        seed=42,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        args=cfg,
        callbacks=[StepTimeLogger(LOG_PATH)],
    )

    import time
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\nElapsed for {MAX_STEPS} steps: {elapsed:.1f}s ({elapsed/MAX_STEPS:.3f} s/step)")
    print(f"Step-time log: {LOG_PATH}")


if __name__ == "__main__":
    main()
