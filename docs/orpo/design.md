# ORPO trainer — design

> **Status:** active design + implementation. Living doc — update as decisions evolve.
> **Last updated:** 2026-05-01

## TL;DR

`rl_orpo.py` is a single-slice on-policy ORPO trainer for the calendar agent.
It composes three ideas:

1. **AR3PO-style adaptive sampling** of training prompts (not a separate algorithm —
   just a difficulty-weighted sampler with adaptive rollout count per prompt).
2. **AR3PO-style response-reuse buffer** for all-fail rescue.
3. **ORPO loss** (Hong et al., arxiv 2403.07691) instead of GRPO. Off-policy-tolerant,
   reference-free, no importance sampling, naturally handles both positive (chosen)
   and negative (rejected) signal in a single supervised-style update.

Built on top of `openpipe-art` 0.5.17's vLLM sleep/wake plumbing — single MIG slice
per training run. Initial LoRA is SFT v6 ckpt-4659 (held-out test best at 80.1%).

## Why ORPO and not the alternatives

Detailed conversation history in the chat where this design was finalized.
Short version of why we landed here:

| Algorithm | Considered? | Why not |
|---|---|---|
| GRPO (current) | Baseline | Saturates on this dataset; no learning on all-fail groups. |
| AR3PO (arxiv 2509.25808) | Implemented + scrapped | Withdrawn ICLR 2026, no code, ~1pp gains over GRPO in single-run benchmarks. Marginal. |
| ARPO (arxiv 2507.19849) | Considered | Multi-turn-tool-agent fit is excellent, ICLR 2026 accepted, open code. But it's an *exploration* algorithm — doesn't address the "learn from off-policy correct rollouts" problem. |
| DOTS (arxiv 2506.05316) | Considered | NeurIPS 2025 poster, open code. But: difficulty-targeted selection *avoids* all-fail prompts; rollout-replay buffer rejects all-0 groups. Opposite of our goal. |
| AWR/AWAC | Considered | With binary rewards, collapses to rejection-sampling SFT. Too conservative at LLM scale; abandons GRPO framework entirely. |
| RFT (rejection-sampling fine-tune) | Considered | Doesn't teach what NOT to do. |
| DPO (prior attempt, 2026-04-25) | Failed | DPO-from-Instruct hit textbook likelihood displacement (Pal et al. 2024). DPO-from-SFT trailed SFT v6 by 0.8pp. Implementation was buggy: off-policy pairs, missing `tools=` in chat template, mid-flight patching. Auto-memory: `feedback_dpo_skipped`. |
| **ORPO** | **Chosen** | Reference-free (simpler than DPO), SFT loss term prevents likelihood displacement, accepts off-policy data well, accepted ICLR 2024 + battle-tested in TRL. |

## Algorithm

For each training step:

1. **Sample N=20 scenarios without replacement** from the difficulty-weighted
   distribution `P(q) ∝ 1 − p_q^G − (1 − p_q)^G` where `p_q` is the per-scenario
   pass-rate EMA and `G` is the rollout budget for that scenario. Cold-start
   scenarios (n_observations < 8) get uniform weight 1.0.
2. **Adaptive rollout per scenario.** `k = 8` if `p_q < 0.3` (hard) or cold-start.
   `k = 4` otherwise. Run all rollouts concurrently via vLLM.
3. **Score** each rollout via the local Qwen3-14B-fp8 judge service
   (95.44% on manual oracle).
4. **Build pairs.** For each scenario:
   - Let C = correct rollouts, I = incorrect rollouts in this step.
   - If `|C| > 0` and `|I| > 0`: emit all `|C| × |I|` pairs (no cap).
   - If `|C| == 0` and reuse-buffer has prior correct for this scenario:
     splice in one randomly-sampled buffered correct as `chosen`, pair with each
     on-policy fail as `rejected`. Tag pairs with `from_reuse_buffer=True`.
   - If `|C| == 0` and buffer empty: skip scenario (no chosen available).
   - If `|I| == 0`: skip scenario (no rejected available).
5. **Update difficulty tracker** (EMA) with on-policy pass rates only.
6. **Update reuse buffer:** push every newly-generated correct rollout into its
   scenario's FIFO (cap 4 per scenario).
7. **Sleep vLLM** (level 2 if no unfinished requests, else level 1).
8. **Reload training model to GPU.**
9. **Run ORPO gradient updates over all pairs.** Pairs are split into minibatches
   (per_device_train_batch_size=4, gradient_accumulation_steps=4 → effective
   batch 16). Multiple optimizer steps per rollout phase. ORPO is off-policy
   tolerant so this is OK; freshness drift across the inner loop is bounded by
   the fact that rollouts came from a single generation pass.
10. **Save checkpoint, offload training model to CPU, wake vLLM.**
11. **Add new LoRA adapter to vLLM** so subsequent rollouts use the updated weights.
12. Repeat.

### Hyperparameters

| Param | Value | Notes |
|---|---|---|
| `N_QUERIES_PER_STEP` | 20 | Without replacement. |
| `K_HARD` | 8 | Pass-rate < 0.3 or cold-start. |
| `K_EASY` | 4 | Pass-rate ≥ 0.3. |
| `EMA_α` | 0.3 | Pass-rate smoothing. |
| `COLD_START_OBS` | 8 | Rollouts before bucket assigned. |
| `HARD_THRESHOLD` | 0.3 | Adaptive-k threshold (also informs P(produces_pair)). |
| `BUFFER_PER_SCENARIO` | 4 | FIFO. |
| `MAX_PAIRS_PER_SCENARIO` | None | Per-scenario cap dropped — full all-pairs. |
| `LORA_RANK` | 8 | Match prior RL convention for benchmarking. |
| `ORPO_β` | 0.1 | TRL default. |
| `LR` | 5e-6 | Match prior RL runs. |
| `per_device_train_batch_size` | 4 | |
| `gradient_accumulation_steps` | 4 | Effective batch 16. |
| `max_grad_norm` | 1.0 | |
| `MAX_TURNS` | 8 | Agent rollout cap. |

## Architecture: single-slice via ART sleep/wake

We re-use `openpipe-art`'s in-process vLLM + Unsloth integration for everything
*except* the loss function. ART loads the model once (Qwen3-14B 4-bit + LoRA);
vLLM serves generations from the same model object; sleep/wake frees the KV
cache for training. We bypass `model.train()` (which is hard-wired to GRPO) and
write our own `train_orpo` driver that mirrors ART's `UnslothService.train_sft`
pattern.

### Lifecycle (one training step)

```
[gen phase, vLLM awake]
    sample N scenarios → adaptive-k rollouts via OpenAI client → judge → pairs

[transition: vLLM → sleep]
    llm.pause_generation()
    determine sleep level (1 if unfinished requests, 2 if clean)
    run_on_workers(llm, do_sleep, level=sleep_level)
    gc_and_empty_cuda_cache()
    state.reload_to_gpu()

[train phase, vLLM asleep]
    peft_model.train()
    for minibatch in pairs.split(per_device_train_batch_size * grad_accum):
        loss = orpo_loss(peft_model, minibatch)   # see § ORPO loss below
        loss.backward()
        if step_boundary:
            clip_grad_norm
            optimizer.step()
            optimizer.zero_grad()
    save_checkpoint(trainer, output_dir)

[transition: vLLM → awake]
    state.offload_to_cpu()
    gc_and_empty_cuda_cache()
    run_on_workers(llm, do_wake_up)
    llm.add_lora(LoRARequest(...))   # register new adapter
    llm.resume_generation()

[next step, with updated LoRA]
```

### ART surface area we depend on

```python
from art.unsloth.service import (
    do_sleep, do_wake_up, save_checkpoint, _get_trainer_optimizer,
)
from art.unsloth.train import gc_and_empty_cuda_cache
from art.vllm import run_on_workers, get_llm
from art.local.backend import LocalBackend
# private but stable in 0.5.17:
backend._services[model_name]  # → UnslothService
service._state.peft_model
service._state.trainer
service._state.reload_to_gpu()
service._state.offload_to_cpu()
service.llm  # → cached_property, AsyncLLM
```

If ART bumps versions and these names move, `art_patches.py` is the place to
add a compatibility shim.

## ORPO loss

Standard formulation from Hong et al. 2024:

```
L_ORPO = L_SFT(chosen) − λ · log σ(log_odds_ratio)

where log_odds(y|x) = log P(y|x) − log(1 − P(y|x))
      log_odds_ratio = β · (log_odds(chosen) − log_odds(rejected))
      L_SFT = standard cross-entropy on response tokens of `chosen` only
```

We compute log P(y|x) by summing per-token log-probs over response tokens
(prompt tokens masked to -100). Both `chosen` and `rejected` go through one
concatenated forward pass for memory efficiency, then split.

We can either invoke TRL's `ORPOTrainer.concatenated_forward` directly, or
inline the loss (~30 lines). We'll inline it for clarity and full control —
no need to reach into TRL's Trainer state for a single-step gradient call.

## Logging (detailed for debuggability)

All to `runs/<run>/orpo_diagnostic.jsonl`, one record per step.

```jsonc
{
  "step": int,
  "epoch": int,
  "timestamp": iso,
  "phase_timing_s": {
    "sample": float, "rollouts": float, "judge": float,
    "pair_build": float, "sleep_vllm": float, "reload_gpu": float,
    "train": float, "save_ckpt": float, "wake_vllm": float, "step_total": float
  },
  "sampling": {
    "scenario_ids": [str], "weights": [float],
    "k_per_scenario": [int],
    "bucket_counts_all": {"cold": n, "hard": n, "mid": n, "easy": n},
    "buckets_sampled": {...}
  },
  "rollouts": {
    "total": int, "correct": int, "incorrect": int,
    "tokens_prompt": int, "tokens_completion": int,
    "tokens_per_sec": float, "judge_latency_s": float,
    "had_error": int, "no_final_answer": int
  },
  "pairs": {
    "total": int,
    "from_reuse_buffer": int,
    "per_scenario": [{
      "scenario_id": str, "category": str, "bucket": str,
      "k_intended": int, "k_actual": int,
      "n_correct": int, "n_incorrect": int, "n_pairs": int,
      "skip_reason": null|"all_correct"|"all_fail_no_buffer"
    }],
    "skipped": {"all_correct": n, "all_fail_no_buffer": n}
  },
  "buffer": {
    "size_total": int, "scenarios_covered": int,
    "added_this_step": int, "rescue_attempts": int, "rescue_hits": int
  },
  "tracker": {
    "migrations": {"hard->mid": n, ...},
    "visit_stats": {"min": n, "p50": n, "max": n, "ratio": float},
    "frozen_alert": bool
  },
  "training": {
    "n_pairs": int, "minibatches": int, "optimizer_steps": int,
    "loss_orpo_mean": float, "loss_sft_mean": float, "loss_or_mean": float,
    "logp_chosen_mean": float, "logp_rejected_mean": float,
    "rewards_chosen_mean": float, "rewards_rejected_mean": float,
    "rewards_accuracy": float,  # fraction where chosen > rejected
    "rewards_margin_mean": float,
    "grad_norm_max": float, "grad_norm_mean": float
  },
  "gpu": {
    "before_rollouts": {...}, "after_rollouts": {...},
    "before_train": {...}, "after_train": {...},
    "peak_allocated_gb": float
  }
}
```

The `training.rewards_accuracy` metric is the most informative single signal
for whether ORPO is learning — it's the fraction of pairs where the model now
prefers the chosen over the rejected. Should rise from ~0.5 (random) toward
1.0 over training.

## Bugs explicitly guarded against

From `feedback_dpo_skipped` (the prior DPO attempt):

1. **On-policy pair source.** Pairs come from the LoRA serving in vLLM right
   now. ✓ by construction (we generate fresh rollouts per step).
2. **Tool schema in chat template.** `src/calendar_agent/orpo/tokenize.py`
   applies `tokenizer.apply_chat_template(..., tools=OPENAI_TOOLS)`.
   `tests/test_orpo_tokenize.py` asserts the rendered prompt contains tool
   definitions. **First-class concern.**
3. **Likelihood displacement.** ORPO's SFT loss term prevents the failure mode
   that broke DPO-from-Instruct.
4. **Mid-flight patching.** Operational bugs trigger restart, not in-place fix.
   This doc gets updated, code is rebuilt cleanly, run ID is incremented.

## File layout

```
docs/orpo/
  design.md                 ← this file
  run_log.md                ← per-run results (created on first run)

src/calendar_agent/orpo/
  __init__.py
  difficulty_tracker.py     Per-scenario pass-rate EMA + weighted sampler
  reuse_buffer.py           Per-scenario FIFO of correct trajectories
  pair_builder.py           All-pairs C×I + buffer rescue + skip logic
  tokenize.py               Chat template w/ tools= → token IDs for ORPO loss
  orpo_loss.py              Inline ORPO loss (concatenated forward + odds-ratio)

scripts/training/rl/
  rl_orpo.py                Main trainer
  rl_orpo.sbatch            Slurm submitter (single slice + auto_restart.sh)

tests/
  test_orpo_difficulty_tracker.py
  test_orpo_reuse_buffer.py
  test_orpo_pair_builder.py
  test_orpo_tokenize.py     ← guards the prior `tools=` bug
  test_orpo_loss.py
```

## Run history & gotchas

### 2026-05-01: post-launch fixes round 2 (length norm + OOM + optimizer state)
After the rank-mismatch fix below, run 84 successfully entered the train
phase but a code review caught three real bugs and a CUDA OOM hit on the
first ORPO forward:

1. **Length-bias in OR term.** `_token_logps` returned sum-of-logps;
   `log_odds(y|x) = logp − log(1 − exp(logp))` then collapsed to
   `≈ logp` for long sequences (since `exp(−500) ≈ 0`), making
   `log_odds_ratio` driven by length, not preference. Fixed by
   length-normalizing in the loss: divide by `n_toks` before computing
   `log_odds`. TRL's ORPOTrainer uses the same pattern.
   `tests/test_orpo_loss.py::test_length_mismatch_does_not_dominate_rewards`
   guards against regression.
2. **CUDA OOM.** `F.log_softmax(shift_logits.float(), ...)` materialized
   a `(B*2, T-1, V) = (8, 4095, 152064)` fp32 tensor → 19.9 GiB on a
   24 GiB slice. Replaced with chunked `gather − logsumexp` along T
   (chunk=256), which keeps peak at ~1.2 GiB. Plus dropped
   `per_device_train_batch_size` 4→2, raised `grad_accum` 4→8 to keep
   effective batch ≈16. Forward+backward at bf16 (B=4 chosen+rejected
   stacked, T=4096) now fits with comfortable headroom.
3. **AdamW state not persisted across restarts.** The custom AdamW
   built on `peft_model.parameters()` was reinitialized on every
   auto_restart, costing ~10–20 steps of warmup each time. Added
   `runs/<run>/orpo_optimizer.pt` sidecar saved on every flush + final.

### 2026-05-01: rank-mismatch on first launch (resolved)
First launch tried `RL_BASE_MODEL=Qwen/Qwen3-14B` + `ART_INJECT_LORA_CHECKPOINT=
runs/sft_v6_qwen3_14b_20260420/checkpoints/checkpoint-4659` (rank-64 SFT LoRA
on top of base). PEFT raised "size mismatch ... shape [64, 5120] from
checkpoint, shape [8, 5120] in current model" — rank 64 SFT can't load into
rank 8 ORPO. Fix: switch `RL_BASE_MODEL` to the existing merged checkpoint
at `runs/sft_v6_qwen3_14b_20260420/eval_test/merged_tmp_4659/` (28 GB fp16
Qwen3-14B with SFT updates baked in) and drop the inject hook entirely. Per
CLAUDE.md / rl_train.py comment: "For runs starting from an SFT LoRA, merge
the SFT into base first (post-SFT fp16 model) and pass that as RL_BASE_MODEL."

The `merged_tmp_4659` directory was created by an eval run; the name `tmp`
is misleading — it's a stable artifact, not GC'd. If it ever gets cleaned
up, regenerate with `scripts/training/common/merge_lora.py`.

## Open questions / future work

- **Rank ablation.** We start at LoRA r=8 to match prior RL benchmarks. After
  the first run produces a usable baseline, ablate r=16/32 separately.
- **`ORPO_β` ablation.** Default 0.1 (TRL default). May want to sweep [0.05, 0.5].
- **Drift between minibatches.** If `rewards_accuracy` collapses or oscillates
  within a single rollout phase, regenerate after each optimizer step instead
  of running all minibatches on the same batch. Easy to add a flag if telemetry
  shows this is needed.
- **Sampler with different convex shape.** P(produces_pair) currently
  `1 − p^G − (1−p)^G`. Could swap for a triangular weighting peaked at 0.5
  if entropy-style weighting underperforms.

## Postmortem (run-1) / v2 plan

Run-1 results live in `docs/orpo/run_log.md`. Summary: +4.1pp held-out at
ckpt-600 (84.25%) vs SFT v6 baseline 80.1%. Wall-clock parity with prior
GRPO run; ~6× more signal-dense per rollout but per-step compute ~20×
heavier so net is a wash.

**What worked:**
- ORPO loss converged smoothly (no oscillation, no likelihood displacement).
- Buffer rescue hit 88% peak, kept gradient signal flowing on all-fail groups.
- Adaptive-k cut step time 66% as cold-start drained.
- Margin grew 8× over the run (+0.14 → +1.19); rew_acc broke through the
  0.61 plateau to 0.71 in the late phase.

**What failed:**
- **Difficulty sampler couldn't escape saturated easies.** `1 − p^G − (1−p)^G`
  weights easy at 0.48 vs mid at 0.88 — only 1.8× ratio. With 574 easies
  vs 1 mid in the pool by step 621, sampling shifted to 88% easy / 7% mid /
  5% hard regardless of weights. Late-run skip_easy hit 16/20 per step →
  80% wasted rollouts.
- **Pair-producing scenario diversity collapsed** 272 → 34. Top-50 share of
  pairs: 38% → 100%. The late gradient pile-drove on the same ~30
  scenarios — closer to memorization than learning.
- **Modifier regressed −2pp** from forgetting. The pile-driving on residual
  hard scenarios pulled the model away from Modifier patterns.
- **Buffer wasn't persisted across the first restart.** Cost ~150 steps to
  re-warm. Patched, but the run-1 ckpts paid the price.

**v2 plan — replace the difficulty machinery with DAPO Dynamic Sampling.**

The canonical fix from the literature (DAPO, ByteDance Seed + Tsinghua,
arxiv 2503.14476) is conceptually one rule: **filter prompt groups with
`std == 0` and accumulate across over-sampled batches until N
mixed-reward groups collected.** This subsumes our entire
difficulty-tracker + weighted-sample + adaptive-k + skip-handling logic
into a simpler more aggressive rule.

```python
# v2 step (replaces sections § Algorithm 1-2 above)
N_TARGET = 10           # mixed-reward groups per step
K = 8                   # rollouts per prompt
BATCH_MULT = 3          # oversample factor
MAX_OVERSAMPLE = 6      # bail if pool is genuinely exhausted

producing = []
batches_done = 0
while len(producing) < N_TARGET:
    if batches_done >= MAX_OVERSAMPLE:
        raise StalledError("training pool exhausted — generate new scenarios")
    prompts = uniform_sample(BATCH_MULT * N_TARGET)
    batches_done += 1
    for p in prompts:
        c, i = rollout_and_judge(p, K)
        if c > 0 and i > 0:        # std > 0
            producing.append((p, c, i))
            if len(producing) >= N_TARGET: break
# Build C×I pairs from each producing group; run ORPO update.
# Buffer rescue stays — orthogonal to dynamic sampling, helps all-fail groups.
```

**Other v2 changes (planned but ablate one at a time):**

- **β = 0.3 (from 0.1).** Run-1 rew_acc only hit 0.71 — preference signal
  is too soft. β=0.3 steepens the gradient at the cost of more
  divergence; literature supports β=0.1-0.5 for ORPO.
- **LoRA rank 16 (from 8).** GRPO baseline used r=16 and matched ORPO on
  held-out at 1/12 the steps. r=8 is the cheaper hypothesis; r=16 is the
  hedge against the capacity-ceiling reading. Doubles checkpoint size.
- **Trigger scenario generation when stalled.** When dynamic sampling
  raises `StalledError` (pool fully saturated), trigger augmentation via
  the existing `generate_data.py` path with the failing scenario as a
  seed. Rewires the curriculum end-to-end instead of pile-driving on the
  same residual.

**Buffer & persistence are already fixed in run-1's code** (see
`ReuseBuffer.save/load`; `optimizer.state` device-fix in `orpo_train_step`).

Not for v2 (separate experiment):
- **Clip-Higher** (DAPO's other lever) — PPO-specific; doesn't transfer
  directly to ORPO. The intent (preserve exploration) could be served by
  an entropy bonus on `logp_chosen`, but that's a research direction.
- **Self-Evolving Curriculum (SEC, arxiv 2505.14970)** — multi-armed
  bandit over categories. Promising but adds bookkeeping; revisit after
  v2 baseline is in.
