# RL Run: Single-Category — Modifier & Correction

## Run Config

| Parameter | Value |
|---|---|
| **Category** | Modifier & Correction (Rescheduling/Updates) |
| **Scenarios** | 88 (79 train / 9 val) |
| **Epochs** | 5 (395 steps) |
| **Base model** | `sft_output/merged_tmp` (SFT ckpt 933, Qwen2.5-1.5B-Instruct) |
| **Algorithm** | GRPO via ART 0.5.4 |
| **Learning rate** | 5e-6 |
| **Beta (KL penalty)** | 0.0 |
| **Rollouts per group** | 8 |
| **Num generations** | 2 |
| **Max grad norm** | 0.1 |
| **Optimizer** | paged_adamw_8bit |
| **GPU memory utilization** | 0.85 |
| **Max model len** | 3076 |
| **Max LoRA rank** | 32 |
| **LoRA rank** | 8 (ART default peft_args.r), vLLM max_lora_rank=32 |

## Training Results (Epoch-by-Epoch)

| Epoch | Steps | Avg Accuracy | Training Rate | vs SFT Baseline (32.5%) |
|---|---|---|---|---|
| 0 | 79/79 | 25.5% | 43% | -7.0% |
| 1 | 79/79 | 32.8% | 51% | At baseline |
| 2 | 79/79 | 39.7% | 59% | **+7.2%** |
| 3 | 79/79 | 40.5% | 62% | **+8.0%** |
| 4 | 79/79 | 41.3% | 64% | **+8.8%** |

Training accuracy (diagnostic): 27.5% → 43.8% over 395 steps.

## Eval Results (Full RL Data — 280 Queries, 7 Categories)

| Category | SFT Baseline | After RL | Delta |
|---|---|---|---|
| Relative Time References | 29/40 (72.5%) | 29/40 (72.5%) | 0 |
| Information Retrieval | 25/40 (62.5%) | 29/40 (72.5%) | **+10.0%** |
| Schedule a Single Event | 21/40 (52.5%) | 27/40 (67.5%) | **+15.0%** |
| Vague & Contextual | 20/40 (50.0%) | 17/40 (42.5%) | -7.5% |
| Modifier & Correction | 13/40 (32.5%) | 18/40 (45.0%) | **+12.5%** |
| Human Chaos (Edge Cases) | 6/40 (15.0%) | 5/40 (12.5%) | -2.5% |
| Complex Logic & Conflict | 5/40 (12.5%) | 9/40 (22.5%) | **+10.0%** |
| **Overall** | **119/280 (42.5%)** | **134/280 (47.9%)** | **+5.4%** |

**Key findings:**
- Target category (Modifier & Correction) improved 32.5% → 45.0% (+12.5pp)
- Positive transfer to Information Retrieval (+10pp), Schedule (+15pp), Complex Logic (+10pp)
- Small regression on Vague & Contextual (-7.5pp) and Human Chaos (-2.5pp)
- Overall: 42.5% → 47.9% (+5.4pp)

## How to Serve

**CRITICAL: Do NOT merge this LoRA to fp16.** vLLM LoRA serving must use the adapter directly.

```bash
# Start vLLM with LoRA support
VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve sft_output/merged_tmp \
  --served-model-name calendar-agent-rl \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 3076 \
  --gpu-memory-utilization 0.80 \
  --enable-lora \
  --lora-modules rl-modifier=rl_runs/single_category_modifier_correction/checkpoint \
  --max-lora-rank 32

# Then in eval, use: --model rl-modifier
```

## Files

| File | Description |
|---|---|
| `checkpoint/` | LoRA adapter (adapter_config.json + safetensors + tokenizer) |
| `eval/eval_rl395_rl.json` | RL eval on full RL data (47.9%, 280 queries) |
| `eval/eval_sft_baseline_rl.json` | SFT baseline eval for comparison (42.5%, 280 queries) |
| `diagnostics/rl_diagnostic.json` | Per-step training diagnostics (395 steps) |

## Dependencies

- Base model: `sft_output/merged_tmp` (SFT checkpoint 933 merged to fp16)
- RL data: `rl_data/` (622 scenarios across 50 calendars)
- ART patches: `src/calendar_agent/art_patches.py`
