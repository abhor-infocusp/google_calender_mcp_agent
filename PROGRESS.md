# Training Pipeline Progress

> **Last updated:** 2026-04-12
> **Current phase:** SFT v5 eval complete. Best checkpoint: epoch 4 (74.6%). Ready for RL training.

---

## Pipeline Overview

```
Compact Data → Re-Augment → SFT Training v3 → Merge LoRA → Evaluate → RL Training
   DONE         DONE         DONE (ckpt 933)    DONE         DONE       MILESTONE
```

## Phase Status

### 1. Data Preparation — DONE

**Original SFT data (compact):** `sft_data/trajectories/` — 1,164 trajectories (114 calendars, gemini-2.5-pro teacher)
- Compact string tool results (not JSON dicts), median 596 tokens/trajectory

**Augmented data:** `sft_data/trajectories_augmented/` — 6,947 trajectories
- 1,164 original + entity substitution (2x) + category-weighted paraphrasing (2-5x)
- 7 categories balanced at 917-1,074 each; 19 over 3076 token budget (0.27%)

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

Source: `rl_runs/single_category_modifier_correction/eval/eval_sft_baseline_rl.json`

### 4. RL Training — IN PROGRESS

#### Curriculum Sampler — ABANDONED

- Curriculum learning approach did not produce improvement (+0.006 overall after 266 steps)
- Single-category rerun without proper per-scenario multi-rollout grouping also failed (22% avg after 40 steps, below 32.5% SFT baseline)
- Root cause: GRPO needs multiple rollouts of the *same* query per group to learn from reward contrast. Without this, comparing different queries provides weak/noisy signal.
- Files removed: `curriculum_learning.md`, `src/calendar_agent/curriculum.py`
- **Next:** fix rl_train.py to use proper per-scenario grouping (1 TrajectoryGroup per scenario, N rollouts each) before attempting any further training

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
- Config: `gpu_memory_utilization=0.85`, LoRA r=8 (ART default), `rollouts_per_group=8`, `lr=5e-6`, `beta=0.0`
- LoRA rank: 8, alpha 16 (ART peft_args defaults — rank 64 OOMs during vLLM profiling)
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

#### Eval: RL checkpoint 395 vs SFT baseline (full RL data, 280 queries)

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
- Target category improved 32.5% → 45.0% (+12.5pp)
- Positive transfer to 3 other categories (Info Retrieval, Schedule, Complex Logic)
- Small regression on Vague & Contextual (-7.5pp) and Human Chaos (-2.5pp)
- **LoRA merge issue:** merging LoRA to fp16 produces garbage output; must serve via `--enable-lora`

**Archive:** `rl_runs/single_category_modifier_correction/` (checkpoint, eval JSONs, diagnostics, README)

#### Run: Information Retrieval focused — COMPLETE

- **~90 scenarios** (all IR from 50 calendars, no val split), seeded from RL1 Modifier checkpoint
- **234 steps**, same hyperparams as RL1
- Config: `INJECT_LORA_CHECKPOINT = "rl_runs/single_category_modifier_correction/checkpoint"`

**3-Way Per-Category Comparison (280 queries):**

| Category | SFT Baseline | RL1 Modifier (395) | RL2 IR (234) | RL1→RL2 |
|---|---|---|---|---|
| Complex Logic & Conflict | 5/40 (12%) | 9/40 (22%) | 7/40 (18%) | -5% |
| Human Chaos (Edge Cases) | 6/40 (15%) | 5/40 (12%) | 6/40 (15%) | +2% |
| Information Retrieval | 25/40 (62%) | 29/40 (72%) | 30/40 (75%) | +2% |
| Modifier & Correction | 13/40 (32%) | 18/40 (45%) | 20/40 (50%) | +5% |
| Relative Time References | 29/40 (72%) | 29/40 (72%) | 32/40 (80%) | +8% |
| Schedule a Single Event | 21/40 (52%) | 27/40 (68%) | 24/40 (60%) | -8% |
| Vague & Contextual | 20/40 (50%) | 17/40 (42%) | 14/40 (35%) | -8% |
| **Overall** | **119/280 (42.5%)** | **134/280 (47.9%)** | **133/280 (47.5%)** | **-0.4%** |

**Key findings:**
- Overall flat (47.5% vs 47.9%) — no net gain from second RL run
- IR target category: 72% → 75% (+2pp marginal improvement)
- Modifier & Correction retained/extended: 45% → 50% (+5pp)
- Relative Time improved unexpectedly: 72% → 80% (+8pp)
- **Vague & Contextual regressing steadily**: 50% → 42% → 35% across checkpoints
- Schedule a Single Event regressed: 68% → 60% (-8pp)
- Pattern: gains in trained categories offset by regressions elsewhere (catastrophic forgetting)

**Archive:** `rl_runs/single_category_ir/` (checkpoint, eval JSONs, diagnostics, logs, README)

#### Experiment: SFT Recovery on RL1 — FAILED

- 1 epoch SFT on top of RL1 Modifier checkpoint (merged to fp16, then 4-bit + fresh rank-64 LoRA)
- Used full 1,039 augmented SFT trajectories, same config as SFT v3
- Train loss: 0.113, eval loss: 0.102

**Result: 39.3% (110/280) — worse than SFT baseline (42.5%)**

| Category | SFT Baseline | RL1 (47.9%) | SFT-on-RL (39.3%) |
|---|---|---|---|
| Complex Logic & Conflict | 5/40 (12%) | 9/40 (22%) | 7/40 (18%) |
| Human Chaos | 6/40 (15%) | 5/40 (12%) | 5/40 (12%) |
| Information Retrieval | 25/40 (62%) | 29/40 (72%) | 23/40 (57%) |
| Modifier & Correction | 13/40 (32%) | 18/40 (45%) | 10/40 (25%) |
| Relative Time References | 29/40 (72%) | 29/40 (72%) | 30/40 (75%) |
| Schedule a Single Event | 21/40 (52%) | 27/40 (68%) | 21/40 (52%) |
| Vague & Contextual | 20/40 (50%) | 17/40 (42%) | 14/40 (35%) |

**Conclusion:** SFT overwrites RL-learned behaviors. Modifier dropped 45% → 25%, IR dropped 72% → 57%. SFT recovery is not a viable approach for combating catastrophic forgetting in this setup.

**Archive:** `sft_on_rl_output/` (merged model, LoRA, eval JSON, training history)

---

### 5. SFT Enhancement — Phases 1-6 COMPLETE

Full details in `sft_enhancement.md`. Summary of each phase:

**Phase 1 — Teacher model tuning:** gemini-2.5-pro + v11 prompt + 3-attempt retry = 84.3% (59/70 queries)
**Phase 2 — Calendar generation:** 100 new calendars (indices 20-119), 115 total with existing 0-19
**Phase 3 — Trajectory generation:** 1,164 trajectories from 1,616 queries (72% solve rate), 114 calendars
**Phase 4 — Augmentation:** 6,947 balanced trajectories (6.0x expansion), 917-1,074 per category
**Phase 5 — SFT Training v5:** 10 epochs, narration merge fix, 6,901 trajectories
**Phase 6 — Full eval (all 10 checkpoints):** Best = epoch 4 at 74.6%

#### Phase 4 Bug Fix (2026-04-01)

Entity substitution augmentation had a critical bug: `extract_entities()` incorrectly treated `get_current_time` results (timestamps like `"2024-08-04 10:00:00 | Sunday"`) as event summaries and replaced them with random event names (e.g. `"Onboarding"`). **29.4% of training data (2,044/6,947) was corrupted.**

First training run on corrupted data scored **20.7% (epoch 1) and 18.6% (epoch 2)** on RL data vs 42.5% SFT v3 baseline. Model learned to narrate intent ("I will first get the current time...") instead of calling tools.

Fix: added `get_current_time` name check in `extract_entities()` to skip timestamp results. Re-generated all 2,328 entity_substitution variants. Also added error-ending trajectory filter in training script (removes 46 trajectories with teacher model failures).

#### Phase 5 — SFT Training v5

- **6,901 trajectories** (46 error-ending filtered), 6,832 tokenized (69 too long)
- 6,151 train / 684 val, ~1,538 steps/epoch, 10 epochs
- Config: LoRA rank 64, loss masking, cosine_with_restarts (5 cycles), lr=2e-4
- **Narration merge fix:** `trajectory_to_messages()` merges assistant narration into tool_call messages. Previously separate messages taught model to stop after narrating instead of calling tools (70% of data affected).
- Training completed 2026-04-11, all 10 checkpoints + final model saved

#### Phase 6 — SFT v5 Eval (all 10 checkpoints, RL data, 280 queries)

| Epoch | Ckpt | Eval Loss | **Overall** | Complex | Chaos | IR | Modifier | RelTime | Schedule | Vague |
|-------|------|-----------|-------------|---------|-------|----|----------|---------|----------|-------|
| 1 | 1538 | 0.172 | 63.9% | 10/40 | 28/40 | 32/40 | 22/40 | 37/40 | 29/40 | 21/40 |
| 2 | 3076 | 0.097 | 68.9% | 14/40 | 26/40 | 35/40 | 25/40 | 39/40 | 29/40 | 25/40 |
| 3 | 4614 | 0.117 | 67.1% | 10/40 | 28/40 | 37/40 | 27/40 | 39/40 | 27/40 | 20/40 |
| **4** | **6152** | **0.086** | **74.6%** | 13/40 | 31/40 | 38/40 | 30/40 | 37/40 | 35/40 | 25/40 |
| 5 | 7690 | 0.106 | 64.6% | 9/40 | 25/40 | 34/40 | 21/40 | 38/40 | 32/40 | 22/40 |
| 6 | 9228 | 0.088 | 73.2% | 13/40 | 32/40 | 36/40 | 29/40 | 38/40 | 32/40 | 25/40 |
| 7 | 10766 | 0.095 | 68.2% | 11/40 | 30/40 | 32/40 | 26/40 | 39/40 | 30/40 | 23/40 |
| 8 | 12304 | 0.095 | 70.4% | 13/40 | 30/40 | 35/40 | 28/40 | 37/40 | 32/40 | 22/40 |
| 9 | 13842 | 0.101 | 67.1% | 11/40 | 29/40 | 35/40 | 23/40 | 37/40 | 31/40 | 22/40 |
| 10 | 15370 | 0.101 | 69.6% | 11/40 | 30/40 | 34/40 | 29/40 | 38/40 | 26/40 | 27/40 |

**Key findings:**
- **Best checkpoint: epoch 4 (ckpt 6152) at 74.6%** — up from v3's 42.5% (+32pp)
- Even/odd cosine pattern: epochs 2,4,6,8 consistently beat 3,5,7,9 (LR restart effect)
- Performance peaks at epoch 4, overfits thereafter despite cosine restarts
- Strongest categories: IR (95%), RelTime (92.5%), Schedule (87.5%) at epoch 4
- Weakest: Complex (32.5%), Vague (62.5%) — candidates for RL improvement

**Best checkpoint for RL:** epoch 4 (checkpoint-6152), merged to `sft_output/merged_tmp`

Source: `sft_output/eval/summary.csv`, per-checkpoint details in `sft_output/eval/checkpoint-*.json`

#### Eval Infrastructure Fixes (2026-04-01)

- **vLLM pipe buffer deadlock:** `eval_all_checkpoints.py` piped vLLM stdout to `subprocess.PIPE` without reading it. After ~7-8 queries the 64KB buffer filled, vLLM blocked on write, all subsequent requests timed out. Fix: redirect to log file.
- **max_model_len 3076→4096:** Tool definitions use ~509 tokens, leaving insufficient room for multi-turn conversations at 3076. Increased to 4096 for eval serving.
- **HF_HUB_OFFLINE=1:** Added to merge subprocess to avoid HuggingFace connectivity issues on shared systems.
- **Agent error logging:** eval_batch now shows actual error messages instead of generic "agent error".
- **Unified tool result formatting:** All pipelines (SFT, eval, RL) use single `format_tool_result()` from `core.py`.

**Augmented data:** `sft_data/trajectories_augmented/` — 6,947 trajectories (fixed), 114 calendars, 7 balanced categories

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
LoRA r=8 alpha=16 (ART default), enforce_eager=True, swap_space=2, enable_sleep_mode=True
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
- LoRA rank 64 OOMs during vLLM profiling (6.57 GiB peak vs 5.1 GiB free); RL uses rank 8 (ART default)
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
- SFT eval results: `rl_runs/single_category_modifier_correction/eval/eval_sft_baseline_rl.json`
- Training log: task `bjuyq5ida`
