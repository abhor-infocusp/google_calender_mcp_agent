# Training Pipeline Progress

> **Last updated:** 2026-03-19
> **Current phase:** SFT v3 training stopped (epoch 7/10) — NEEDS INVESTIGATION

---

## Pipeline Overview

```
Compact Data → Re-Augment → SFT Training v3 → Merge LoRA → Evaluate → RL Training
   DONE         DONE         NEEDS INVESTIGATION                        PAUSED

Code Cleanup → DONE (fdf0ea6, 5df60cf, + polish 2026-03-19)
```

## Phase Status

### 1. Data Preparation — NEEDS RE-AUGMENTATION

**Original SFT data (compact):** `sft_data/trajectories/` — 161 trajectories
- Compacted tool results: avg 1469 → 859 tokens/trajectory (41% reduction)
- 159/161 fit within 3076 tokens (up from 4096 needed for verbose)
- Script: `scripts/data_generation/compact_tool_results.py`

**Augmented data:** `sft_data/trajectories_augmented/` — 1,039 trajectories (DONE 2026-03-16)
- 161 original + 556 paraphrased + 322 entity-substituted
- 18 JSON files, compact format preserved throughout

**RL data:** `rl_data/` — 622 scenarios across 50 calendars (unchanged)

### 2. Compact Tool Results — DONE

Reduced token usage by 41% across all training data:
- Flattened attendee objects to plain email strings
- Removed double-serialized JSON in tool results
- Dropped verbose fields (creator, organizer, htmlLink, etc.)
- Environment returns compact format directly via `_compact_event()`
- All 18 trajectory files rewritten + verified

### 3. SFT Training v1 — DONE (insufficient)
- 5 epochs, 175 steps, loss 1.19 → 0.22
- Model: Qwen/Qwen2.5-1.5B-Instruct, 4-bit, LoRA rank 32
- Config: lr=2e-4, cosine, paged_adamw_8bit, MAX_SEQ_LENGTH=4096
- **Output deleted** — retraining with improvements

### 4. SFT Evaluation v1 — DONE

**On RL data (held-out, 10 calendars): 30% correct (43/142)**

| Calendar | Correct | Total | Rate |
|----------|---------|-------|------|
| 0 | 2 | 14 | 14% |
| 1 | 4 | 14 | 29% |
| 2 | 5 | 14 | 36% |
| 5 | 2 | 14 | 14% |
| 10 | 8 | 14 | 57% |
| 15 | 5 | 14 | 36% |
| 20 | 4 | 14 | 29% |
| 25 | 6 | 14 | 43% |
| 30 | 6 | 14 | 43% |
| 40 | 1 | 16 | 6% |
| **Total** | **43** | **142** | **30%** |

**On SFT training data (8 calendars): 15% overall**

| | Trained queries | Untrained queries |
|----------|-----------------|-------------------|
| Correct | 17/72 (24%) | 0/40 (0%) |

### 5. SFT Training v2 (verbose format) — STALE (historical)

- Trained on verbose format, now superseded by compact format
- Epoch 2 eval: SFT 46.6%, RL 28.6% (old checkpoints deleted)

### 6. SFT Training v3 (compact format) — NEEDS INVESTIGATION

- Started: 2026-03-16, stopped at epoch 7/10 (last checkpoint: 2026-03-18 09:04)
- 935 train / 104 val trajectories, all 1,039 fit (0 skipped)
- Config: 10 epochs, LoRA rank 64, loss masking (10.6% assistant tokens), cosine_with_restarts
- MAX_SEQ_LENGTH=3076, batch=1, grad_accum=4, 2,330 total steps
- Output: `sft_output/`, loss CSV: `sft_output/epoch_losses.csv`
- **7 checkpoints saved:** 234, 468, 699, 933, 1167, 1401, 1635
- **Loss:** train 0.18→0.02, eval 0.10→0.10 (eval plateaued/rising after epoch 2)
- **Eval results very poor:** checkpoint-234 1.2% SFT / 0% RL, checkpoint-468 1.2% / 0%, checkpoint-699 0.6% / 0%
- Results far worse than v1's 30% — needs investigation before continuing

### 7. Eval Pipeline — DONE

New evaluation scripts with Gemini judge:
- `scripts/eval/eval_batch.py` — batch eval on SFT and/or RL data
- `scripts/eval/eval_all_checkpoints.py` — automated merge → serve → eval loop
- `src/calendar_agent/evaluation.py` — SIGALRM 30s timeout per eval, 60s per query
- vLLM must use `--max-model-len 3076` (not 2048) for reliable eval

### 8. RL Training — PAUSED (focus on SFT first)

#### Run 1 (shaped rewards → reward hacking)
- Shaped rewards caused model to optimize for 0.3 tier
- Validation correct: 20% → 0% by step 25

#### Run 2 (NameError bug → silent failure)
- All 202 steps ran with zero training (rollouts crashed silently)

#### Run 3-4 (binary, high skip rate)
- Binary rewards, ~48-60% skip rate due to same-reward rollouts

#### Run 5-6 (speed experiments)
- KV cache (0.20 GiB) is the binding constraint

---

## Key Bugs Found & Fixed

### Double-encoded tool call arguments (CRITICAL)
- `sft_train.py`: `json.dumps(tc["args"])` → `tc["args"]`

### NameError in RL rollout (CRITICAL)
- `rl_train.py`: debug print referenced deleted `shaped_reward` variable

### Other fixes
- max_model_len=1536 too tight → 2048 → 3076 for eval
- No gradient clipping → grad_norm=nan → max_grad_norm=0.1
- Unsloth strips chat template → restore from base tokenizer

---

## Next Steps

1. ~~Re-augment trajectories~~ — DONE (1,039 trajectories)
2. ~~Retrain SFT v3~~ — STOPPED at epoch 7/10
3. ~~Evaluate v3 checkpoints~~ — DONE (results very poor: 0-1.2%)
4. ~~Code cleanup~~ — DONE (centralized ~1300 LOC, fixed bugs, removed dead code)
5. **Investigate SFT v3 regression** — why 0-1.2% vs v1's 30%? (loss masking? augmentation quality? tokenization?)
6. **RL training** — blocked on SFT baseline

## Hardware Constraints
- GPU: NVIDIA TITAN X Pascal, 12 GiB VRAM, compute 6.1
- NO bfloat16, NO FlashAttention2, fp16 only
- vLLM 0.7.3 rebuilt for sm_61, Punica patched to CPU
- KV cache only 0.20 GiB → max ~3.65 concurrent requests at 2048 tokens
