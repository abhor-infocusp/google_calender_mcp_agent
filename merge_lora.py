#!/usr/bin/env python3
"""Merge SFT LoRA adapter into base model and save as fp16.

The merged model can then be:
  1. Served by vLLM for evaluation (eval_qwen.py)
  2. Used as base_model in ART for RL fine-tuning (fine_tuning.py)

Usage:
    python merge_lora.py
"""
import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from unsloth import FastLanguageModel

LORA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sft_output", "final")
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sft_output", "merged")

print(f"Loading base model + LoRA adapter from: {LORA_PATH}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=LORA_PATH,
    max_seq_length=4096,
    load_in_4bit=True,
)

print(f"Merging LoRA and saving as 16-bit to: {OUTPUT_PATH}")
model.save_pretrained_merged(OUTPUT_PATH, tokenizer, save_method="merged_16bit")
print("Done!")
