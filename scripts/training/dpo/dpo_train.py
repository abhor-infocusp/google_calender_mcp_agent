"""DPO training on Qwen3-14B calendar-agent pairs.

Configurable via env vars so the same script runs for both experiments:
    DPO_BASE_MODEL   — path or HF name of the starting model
    DPO_PAIRS        — JSONL produced by mine_dpo_pairs.py
    DPO_OUTPUT_DIR   — where to save checkpoints + logs
    DPO_RUN_NAME     — tag for logs / wandb / config.json

Matches SFT v6 LoRA setup (rank 64, same target modules) so the two
starting points differ only in weights.
"""

import json
import os
import sys
from datetime import datetime

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

from calendar_agent.tools import get_openai_tools


# ── Config (env-var-driven so sbatch can override) ──

BASE_MODEL = os.environ.get("DPO_BASE_MODEL", "Qwen/Qwen3-14B")
PAIRS_PATH = os.environ.get("DPO_PAIRS", "runs/dpo/pairs_from_14b_rl.jsonl")
OUTPUT_DIR = os.environ.get("DPO_OUTPUT_DIR", f"runs/dpo_qwen3_14b_{datetime.now().strftime('%Y%m%d')}")
RUN_NAME = os.environ.get("DPO_RUN_NAME", os.path.basename(OUTPUT_DIR))

MAX_LENGTH = int(os.environ.get("DPO_MAX_LENGTH", 2048))
MAX_PROMPT_LENGTH = int(os.environ.get("DPO_MAX_PROMPT_LENGTH", 1024))
BETA = float(os.environ.get("DPO_BETA", 0.1))
LR = float(os.environ.get("DPO_LR", 5e-7))
EPOCHS = float(os.environ.get("DPO_EPOCHS", 3))
GRAD_ACCUM = int(os.environ.get("DPO_GRAD_ACCUM", 4))
LORA_RANK = int(os.environ.get("DPO_LORA_RANK", 64))
LORA_ALPHA = int(os.environ.get("DPO_LORA_ALPHA", 64))
OPTIM = os.environ.get("DPO_OPTIM", "paged_adamw_8bit")

CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ── Telemetry: phase tracker, heartbeat, metadata, stuck-alerts ─────
# Shared across trainers; lives in calendar_agent.run_telemetry. DPO doesn't
# have multi-stage phases like RL, but consistent metadata + heartbeats keep
# stop_run.sh / list_runs.sh / lnav skills uniform across all training kinds.
from calendar_agent.run_telemetry import init_telemetry, set_phase  # noqa: E402

init_telemetry(run_dir=OUTPUT_DIR, script_path=__file__)
set_phase("startup")


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts} dpo_train] {msg}", flush=True)


def main():
    log(f"BASE_MODEL={BASE_MODEL}")
    log(f"PAIRS_PATH={PAIRS_PATH}")
    log(f"OUTPUT_DIR={OUTPUT_DIR}")
    log(f"LR={LR} BETA={BETA} EPOCHS={EPOCHS} GRAD_ACCUM={GRAD_ACCUM}")
    log(f"LORA r={LORA_RANK} alpha={LORA_ALPHA}")
    log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    log(f"GPU count: {torch.cuda.device_count()}")

    # ── Save config.json ──
    config_out = {
        "kind": "dpo",
        "base_model": BASE_MODEL,
        "pairs": PAIRS_PATH,
        "hparams": {
            "beta": BETA,
            "learning_rate": LR,
            "num_epochs": EPOCHS,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": GRAD_ACCUM,
            "max_length": MAX_LENGTH,
            "max_prompt_length": MAX_PROMPT_LENGTH,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "quantization": "4bit-bnb",
            "bf16": True,
        },
        "started_at": datetime.now().isoformat(),
        "run_name": RUN_NAME,
    }
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(config_out, f, indent=2)

    # ── Tokenizer ──
    log("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model (QLoRA 4-bit) ──
    log("Loading base model in 4-bit bnb")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # ── LoRA config (PEFT handles ref model = frozen base w/ adapter disabled) ──
    peft_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # ── Dataset ──
    log(f"Loading pairs from {PAIRS_PATH}")
    ds = load_dataset("json", data_files=PAIRS_PATH, split="train")
    log(f"Loaded {len(ds)} pairs")
    # DPOTrainer expects columns `prompt`, `chosen`, `rejected`; we already have those.
    # Drop `metadata` to avoid confusing TRL.
    if "metadata" in ds.column_names:
        ds = ds.remove_columns(["metadata"])

    # ── DPO config ──
    # `tools=` is passed through to apply_chat_template — critical so DPO sees
    # the same ~925-token prompt (system + user + tool schemas) that SFT trained
    # on and inference serves. Without it the model learns to emit <tool_call>
    # tokens in a 61-token context that never occurs at eval.
    config = DPOConfig(
        output_dir=CKPT_DIR,
        run_name=RUN_NAME,
        beta=BETA,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        tools=get_openai_tools(),
        bf16=True,
        fp16=False,
        optim=OPTIM,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=int(EPOCHS) + 1,
        remove_unused_columns=False,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        precompute_ref_log_probs=True,
    )

    # ── Train ──
    log("Initializing DPOTrainer")
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT: ref_model inferred by disabling adapters
        args=config,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    log(f"Starting training: {len(ds)} pairs × {EPOCHS} epochs, grad_accum={GRAD_ACCUM}")
    trainer.train()

    log("Saving final adapter")
    final_path = os.path.join(OUTPUT_DIR, "adapter_final")
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    log(f"Saved to {final_path}")


if __name__ == "__main__":
    main()
