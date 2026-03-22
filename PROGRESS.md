# Training Pipeline Progress

> **Last updated:** 2026-03-22
> **Current phase:** RL Training — Modifier & Correction run COMPLETE (5 epochs, 25.5% → 41.3%), next: evaluation

---

## Pipeline Overview

```
Compact Data → Re-Augment → SFT Training v3 → Merge LoRA → Evaluate → RL Training
   DONE         DONE         DONE (ckpt 933)    DONE         DONE       IN PROGRESS
```

## Phase Status

### 1. Data Preparation — DONE

**Original SFT data (compact):** `sft_data/trajectories/` — 161 trajectories
- Compacted tool results: avg 1469 → 859 tokens/trajectory (41% reduction)
- 159/161 fit within 3076 tokens (up from 4096 needed for verbose)

**Augmented data:** `sft_data/trajectories_augmented/` — 1,039 trajectories
- 161 original + 556 paraphrased + 322 entity-substituted

**RL data:** `rl_data/` — 622 scenarios across 50 calendars, 7 categories

### 2. SFT Training v3 (compact format) — DONE

- 935 train / 104 val trajectories, all 1,039 fit
- Config: 10 epochs, LoRA rank 64, loss masking, cosine_with_restarts
- 7 checkpoints saved: 234, 468, 699, 933, 1167, 1401, 1635
- Best checkpoint for RL: **933** (merged to `sft_output/merged_tmp`)

### 3. SFT Evaluation (checkpoint 933) — DONE

**Per-category accuracy on RL data (unseen calendars, 280 queries):**

| Category | SFT data | RL data (generalization) |
|---|---|---|
| Relative Time References | 92.6% | 72.5% |
| Information Retrieval | 79.3% | 62.5% |
| Schedule a Single Event | 81.5% | 52.5% |
| Vague & Contextual | 77.8% | 50.0% |
| Modifier & Correction | 85.2% | 32.5% |
| Human Chaos (Edge Cases) | 60.0% | 15.0% |
| Complex Logic & Conflict | 73.7% | 12.5% |
| **Overall** | **131/161 (81.4%)** | **119/280 (42.5%)** |

Source: `/tmp/eval_ckpt933_rl.json`, `/tmp/eval_ckpt933_sft.json`

### 4. RL Training — IN PROGRESS

#### Infrastructure Fixes (2026-03-19/20)

Three critical bugs fixed to get RL training working:

1. **Training hang (deadlock):** `gradient_accumulation_steps=4` + `logging_steps=500` caused deadlock — ART's `service.py` puts 1 item on `inputs_queue` and waits for `results_queue`, but HF Trainer needs 2000 `_prepare_inputs()` calls before calling `log()`. Fix: `gradient_accumulation_steps=1`, `logging_steps=1`.

2. **OOM on second model load:** ART's `_monitor_openai_server` health check times out during validation → `done_callback` removes service from cache → next `_get_service()` creates new `ModelState` → tries to load model again while first is still in GPU → OOM. Fix: patched `done_callback` in `art/local/backend.py` to not remove service on error/cancel.

3. **ART asyncio patches (from prior session):** `asyncio.Queue` → `queue.Queue` for `inputs_queue`; `trainer.train()` via `run_in_executor`; thread-safe `results_queue` via `call_soon_threadsafe`.

#### Run: Mixed categories (151 steps, ~1 epoch partial)

- All 7 categories, 559 training scenarios
- Results: 38% avg accuracy, 35% training rate (59% skip), no improvement trend
- Skip dominated by bimodal difficulty: easy scenarios (100%) and hard (0%)
- Dashboard: `rl_dashboard.png`

#### Run: Modifier & Correction focused — COMPLETE

- **88 scenarios** (79 train / 9 val), 5 epochs, 395 steps
- Config: `gpu_memory_utilization=0.85`, `max_lora_rank=32`, `rollouts_per_group=8`, `lr=5e-6`, `beta=0.0`
- KV cache: 2.06 GiB (was 0.87), concurrency 25x (was 10.5x)
- Inference: ~110-165 tps

**Final results (5 epochs):**

| Epoch | Steps | Avg Accuracy | Training Rate | vs SFT Baseline (32.5%) |
|---|---|---|---|---|
| 0 | 79/79 | 25.5% | 43% | -7.0% |
| 1 | 79/79 | 32.8% | 51% | At baseline |
| 2 | 79/79 | 39.7% | 59% | **+7.2%** |
| 3 | 79/79 | 40.5% | 62% | **+8.0%** |
| 4 | 79/79 | 41.3% | 64% | **+8.8%** |

Validation (n=5, noisy): oscillates 0-40%, no clear trend due to small sample.

**Observations:**
- Training accuracy improved: 25.5% → 41.3% (+15.8pp over 5 epochs)
- Training rate improved: 43% → 64% (fewer all-wrong skips)
- Improvement plateauing in later epochs (~1pp/epoch vs ~7pp in early epochs)
- Entropy low (~0.02-0.05) — model is very deterministic, limited exploration
- Loss pattern: mix of positive (0.5-1.0) and negative (-0.5 to -2.0), healthy for GRPO
- Next step: formal eval on full RL data to measure generalization

---

## Key Technical Details

### ART/Unsloth Patches (reproducible via `src/calendar_agent/art_patches.py`)

All patches are applied as runtime monkey-patches at import time. No manual
site-packages edits needed — just `import calendar_agent.art_patches` before `import art`.

| Patch | Target | Fix |
|---|---|---|
| A | `ModelState.__init__` | `asyncio.Queue` → `queue.Queue`; sync `_prepare_inputs` |
| B | `vLLMState.train_mode` | gc/empty_cache between inference and training |
| C | `train()` / `get_log_fn()` | `run_in_executor`; `call_soon_threadsafe` |
| D | `_calculate_logprobs()` | entropy detach from autograd; remove chunk_size assertion |
| E | `_prepare_backend_for_training` | guard `done_callback` against error/cancel |

### RL Config (current)
```python
gpu_memory_utilization=0.85, max_model_len=3076, max_num_seqs=4
max_lora_rank=32, enforce_eager=True, swap_space=2, enable_sleep_mode=True
gradient_accumulation_steps=1, logging_steps=1, num_generations=2
per_device_train_batch_size=1, max_grad_norm=0.1, optim="paged_adamw_8bit"
rollouts_per_group=8, learning_rate=5e-6, beta=0.0
```

### Memory Profile (0.85 utilization)
- Model weights: 2.88 GiB (shared Unsloth/vLLM)
- Activation peak: 5.13 GiB
- KV cache: 2.06 GiB (4818 blocks, 25x concurrency)
- After gc+empty_cache: ~5.1 GiB alloc, ~6.4 GiB free
- Stable across steps, no memory leak

### Constraints
- `max_lora_rank=64` OOMs during vLLM profiling (6.57 GiB peak vs 5.1 GiB free)
- `gpu_memory_utilization=0.95` OOMs during profiling
- Context overflow on calendars with many events (e.g., cal_47)

## Hardware
- GPU: NVIDIA TITAN X Pascal, 12 GiB VRAM, compute 6.1
- NO bfloat16, NO FlashAttention2, fp16 only
- vLLM 0.7.3 rebuilt for sm_61, Punica patched to CPU

## Key Files
- ART patches: `src/calendar_agent/art_patches.py` (import before `art`)
- RL training: `scripts/training/rl_train.py`
- Dashboard plot: `scripts/utils/plot_rl_dashboard.py` → `rl_dashboard.png`
- Reward plot: `scripts/utils/plot_rewards.py` → `reward_curve.png`
- SFT eval results: `/tmp/eval_ckpt933_rl.json`, `/tmp/eval_ckpt933_sft.json`
- Training log: task `bjuyq5ida`
