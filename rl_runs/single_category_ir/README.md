# RL Run 2: Single-Category — Information Retrieval

## Run Config

| Parameter | Value |
|---|---|
| **Category** | Information Retrieval (Querying) |
| **Scenarios** | ~90 (all IR from 50 calendars, no val split) |
| **Steps** | 234 |
| **Epochs** | ~5 (incomplete — training stopped at step 234) |
| **Base model** | `sft_output/merged_tmp` (SFT ckpt 933, Qwen2.5-1.5B-Instruct) |
| **Seeded from** | RL1 Modifier & Correction checkpoint (injected via `INJECT_LORA_CHECKPOINT`) |
| **Algorithm** | GRPO via ART 0.5.4 |
| **Learning rate** | 5e-6 |
| **Beta (KL penalty)** | 0.0 |
| **Rollouts per group** | 8 |
| **Num generations** | 2 |
| **Max grad norm** | 0.1 |
| **Optimizer** | paged_adamw_8bit |
| **GPU memory utilization** | 0.85 |
| **Max model len** | 3076 |
| **LoRA rank** | 8 (ART default peft_args.r), vLLM max_lora_rank=32 |

## Eval Results: 3-Way Comparison (Full RL Data — 280 Queries, 7 Categories)

| Category | SFT Baseline | RL1 Modifier (395) | RL2 IR (234) | RL1→RL2 Delta |
|---|---|---|---|---|
| Complex Logic & Conflict | 5/40 (12%) | 9/40 (22%) | 7/40 (18%) | -5% |
| Human Chaos (Edge Cases) | 6/40 (15%) | 5/40 (12%) | 6/40 (15%) | +2% |
| Information Retrieval | 25/40 (62%) | 29/40 (72%) | 30/40 (75%) | **+2%** |
| Modifier & Correction | 13/40 (32%) | 18/40 (45%) | 20/40 (50%) | **+5%** |
| Relative Time References | 29/40 (72%) | 29/40 (72%) | 32/40 (80%) | **+8%** |
| Schedule a Single Event | 21/40 (52%) | 27/40 (68%) | 24/40 (60%) | -8% |
| Vague & Contextual | 20/40 (50%) | 17/40 (42%) | 14/40 (35%) | -8% |
| **Overall** | **119/280 (42.5%)** | **134/280 (47.9%)** | **133/280 (47.5%)** | **-0.4%** |

## Key Findings

- **Overall flat**: 47.5% vs 47.9% from RL1 — no net improvement
- **Target category (IR) improved slightly**: 72% → 75% (+2pp)
- **Modifier & Correction retained gains**: 45% → 50% (+5pp), suggesting continued learning on that category
- **Relative Time References improved**: 72% → 80% (+8pp) — unexpected positive transfer
- **Regressions**: Schedule a Single Event (68% → 60%, -8pp), Vague & Contextual (42% → 35%, -8pp)
- **Vague & Contextual is steadily regressing**: 50% → 42% → 35% across all three checkpoints
- **Gains offset by regressions** — classic catastrophic forgetting / reward hacking pattern

## How to Serve

**CRITICAL: Do NOT merge this LoRA to fp16.** vLLM LoRA serving must use the adapter directly.

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve sft_output/merged_tmp \
  --served-model-name calendar-agent-rl \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 3076 \
  --gpu-memory-utilization 0.80 \
  --enable-lora \
  --lora-modules rl-ir=rl_runs/single_category_ir/checkpoint \
  --max-lora-rank 32

# Then in eval, use: --model rl-ir
```

## Files

| File | Description |
|---|---|
| `checkpoint/` | LoRA adapter (adapter_config.json + safetensors + tokenizer) |
| `eval/eval_rl234_ir_full.json` | RL2 eval on full RL data (47.5%, 280 queries) |
| `eval/eval_rl395_rl.json` | RL1 eval for comparison (47.9%, 280 queries) |
| `eval/eval_sft_baseline_rl.json` | SFT baseline eval for comparison (42.5%, 280 queries) |
| `diagnostics/history.jsonl` | ART training history (per-step metrics, 234 steps) |
| `logs/rl_training.log` | Full training stdout/stderr |
| `logs/eval_ir_234.log` | Evaluation log |

## Dependencies

- Base model: `sft_output/merged_tmp` (SFT checkpoint 933 merged to fp16)
- RL1 checkpoint: `rl_runs/single_category_modifier_correction/checkpoint` (injected at init)
- RL data: `rl_data/` (622 scenarios across 50 calendars)
- ART patches: `src/calendar_agent/art_patches.py`
