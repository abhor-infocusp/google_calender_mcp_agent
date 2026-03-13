# Training Pipeline Progress

> **Last updated:** 2026-03-12 ~23:15 UTC
> **Current phase:** RL Training (run 2 — binary rewards, 420 steps)

---

## Pipeline Overview

```
SFT Training → Merge LoRA → Evaluate SFT → RL Training → Evaluate RL
     DONE         DONE         DONE        IN PROGRESS     [6]
```

## Phase Status

### 1. Data Preparation — DONE
- SFT data: `sft_data/trajectories/` — 161 trajectories, 159 fit in 4096 tokens
- RL data: `rl_data/` — 622 scenarios across 50 calendars
- Train/val split: 90/10 for both

### 2. SFT Training — DONE
- **Run 1 (BROKEN):** Double-encoded tool args bug. Output in `sft_output_broken_args/`.
- **Run 3 (FINAL):** 5 epochs, 175 steps, loss 1.19 → 0.22
  - Model: Qwen/Qwen2.5-1.5B-Instruct, 4-bit, LoRA rank 32
  - Config: lr=2e-4, cosine, paged_adamw_8bit, MAX_SEQ_LENGTH=4096
  - Runtime: ~2.5 hours
  - Checkpoints: `sft_output/checkpoint-{36,72,108,144,175}`
  - Final adapter: `sft_output/final`

### 3. LoRA Merge — DONE
- Input: `sft_output/final`
- Output: `sft_output/merged_instruct` (fp16, ~2.88 GiB)

### 4. SFT Evaluation — DONE (30% correct)
- 5-epoch model on 10 RL-data calendars (4096 context, temp=0.7, 6 turns):

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

- Epoch 1 baseline was 12/70 (17%) → full training improved to 30%
- Strongest on Info Retrieval and Relative Time queries
- Weakest on Complex Logic and Human Chaos categories

### 5. RL Training — IN PROGRESS (Run 2)

#### Run 1 (STOPPED at step 88/420 — shaped rewards caused reward hacking)
- Started: ~16:30 UTC, stopped ~23:10 UTC on 2026-03-12
- Log: `/tmp/rl_train_full.log`
- **Problem:** Shaped rewards (0.0/0.1/0.2/0.3/1.0) caused model to optimize for 0.3 (use tools + any answer) instead of 1.0 (correct answer)
- Validation correct rate: 20% at step 0 → **0% by step 25** and never recovered
- Training correct rate: 64/676 = 9.5% overall, but declining

#### Run 2 (CURRENT — binary rewards)
- Started: ~23:12 UTC on 2026-03-12
- Log: `/tmp/rl_train_binary.log`
- **Change:** Binary rewards only — 0.0 (wrong/no answer) or 1.0 (Gemini judge says Correct)
- Config: groups_per_step=4, epochs=3, rollouts_per_group=2, lr=5e-6, beta=0.0
- Engine: max_model_len=2048, gpu_mem=0.55, max_num_seqs=4
- Total steps: 420, est. ~5-7 min/step → ~35-49 hours

### 6. RL Evaluation — NOT STARTED

---

## Key Bugs Found & Fixed

### Bug: Double-encoded tool call arguments (CRITICAL)
- **File:** `scripts/training/sft_train.py` line 184
- **Was:** `"arguments": json.dumps(tc["args"])` — produces `"arguments": "{}"` in training data
- **Fix:** `"arguments": tc["args"]` — produces `"arguments": {}` (correct JSON object)
- **Verification:** Confirmed with tokenizer + eval + RL test

### Bug: max_model_len=1536 too tight → fixed to 2048
### Bug: No gradient clipping → grad_norm=nan → added max_grad_norm=0.1

---

## Hardware Constraints
- GPU: NVIDIA TITAN X Pascal, 12 GiB VRAM, compute 6.1
- NO bfloat16, NO FlashAttention2, fp16 only
- vLLM 0.7.3 rebuilt for sm_61, Punica patched to CPU

## Files Modified (uncommitted)
- `scripts/training/sft_train.py` — double-encoding fix + Qwen2.5 model name
- `scripts/training/rl_train.py` — engine/trainer args, training config, API fixes
- `scripts/training/merge_lora.py` — output path fix
