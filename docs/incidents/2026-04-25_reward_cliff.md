# 2026-04-25 — Real-RL reward cliff under unisolated GPU contention

**Severity**: high. ~600 training steps trained on degraded gradients before
discovery; no rollback possible due to a separate checkpoint-retention bug.

**Run affected**: `runs/rl_qwen3_14b_20260420/` (Qwen3-14B GRPO).

## Timeline

| When (UTC) | What |
|---|---|
| 2026-04-23 to 04-25 morning | Real-RL running cleanly on slice 0. Mean reward MA-300 ≈ 0.85, tps ~800. |
| **04-25 06:01** | **Cliff begins**. tps drops 800 → 400. `errors`/`no_answer` per step go 0 → 4–7. Reward falls to ~0.4. Same python process, no restart, no code change. |
| 04-25 06:01–18:00 | Cliff persists for ~600 steps (8200 → ~8800). Bad gradients are absorbed into the LoRA. |
| 04-25 ~18:00 | Discovered during a routine reward-curve check. `nvidia-smi` shows another `python scripts/training/rl_train_adaptive.py` running on slice 1, started ~12h earlier. |
| 04-25 ~18:30 | Investigation: TPS plot confirmed external interference, not model collapse. Outputs were still coherent — vLLM was just generating slowly enough that rollouts truncated before producing final answers. |
| 04-25 ~19:30 | Attempted rollback. Found ART had pruned every prior checkpoint — only the poisoned step-9220 LoRA remained. **No rollback possible.** |
| 04-25 evening | Stopped both trainings. Designed isolation + checkpoint-retention plan. |
| 04-26 | Shipped fixes (commits below). Verified Patch K v2 keeps milestones across 55-step delete cycles. |

## Root cause

**MIG GPU partitioning isolates SMs and VRAM, but not the host CPU / PCIe / BLAS thread pools.**
When `rl_train_adaptive.py` started on slice 1, both PyTorch+vLLM processes
oversubscribed the 128-core host. PyTorch's BLAS defaults to `nproc` threads
per process; with two trainings, that's 256 BLAS threads fighting for 128
cores plus PCIe bandwidth. Result: vLLM generation slowed ~50%, rollouts
hit token/time budgets before producing final answers, and the GRPO reward
signal degraded into noise.

The slowdown is gradual enough that no single check fires (all monitoring
hooks looked at "is it making progress?", not "is it making *good* progress?").

## Why we couldn't recover

ART's `model.delete_checkpoints(best_checkpoint_metric=...)` keeps only
`[latest, best]` and prunes everything else. Our Patch K v1 attempted to
work around this with a conditional skip on milestone steps:

```python
is_milestone = batch.step > 0 and batch.step % CHECKPOINT_KEEP_EVERY == 0
if not is_milestone:
    await model.delete_checkpoints(best_checkpoint_metric="train/reward")
```

This is broken: even though we "skipped" delete on step 8500 (milestone),
the very next step's call wiped the just-saved 8500 checkpoint. So in
practice no milestones survived past one step.

When the cliff was discovered at step ~9220, every checkpoint between
step 4517 (the previous run's start) and 9220 had been pruned.

## Contributing factors

- No `taskset` / `OMP_NUM_THREADS` cap on either training. Each process
  was free to claim all 128 cores.
- No preflight check at launch warning that another training was active on
  the same physical GPU.
- No automatic detection of reward collapse (we explicitly opted out of
  detection during the post-incident design — the user wanted prevention,
  not detection).
- Per-script `*_loop.sh` wrappers had drifted; the new training that started
  the cliff didn't have all the same isolation hardening.

## What we shipped

| Change | Commit | What it does |
|---|---|---|
| Patch G v2 (smart-retry on ART deadlock timeout) | `ba09da5` | Distinguishes spurious queue waits from real hangs; retries the spurious 93% instead of exiting |
| Patch K v1 (broken — see above) | (rolled into Patch K v2) | Attempted milestone retention via conditional skip — didn't work |
| **Centralized launch wrapper** with isolation built in | `5afd582` | `scripts/training/common/auto_restart.sh` — one wrapper, OMP/MKL caps default 8, taskset CPU pinning required |
| **Slice map** for MIG ↔ CUDA UUID ↔ CPU range | `5afd582` | `scripts/training/common/slice_map.sh` — host-specific (azkaban: 32 cores per slice) |
| **`docs/multi_tenant_training.md`** | `5afd582` | Operational protocol: launch recipe, recovery steps, anti-patterns |
| `scripts/training/` subdir reorg into `rl/`, `sft/`, `dpo/`, `judge/`, `common/` | `e0a59f0` | Reduces flat-dir cognitive load; legacy wrappers archived |
| **Patch K v2 (the real fix)** | `bc0185f` | Bypasses ART's `model.delete_checkpoints`; uses lower-level `art.local.checkpoints.delete_checkpoints(output_dir, excluding)` with explicit keep-list including all milestones + latest + best. **Verified end-to-end on small-RL: 6 milestones survived 55 step-cycles.** |
| `data/{rl,sft,test,judge}/` consolidation + env-var path overrides | `3b06a0e` | Single source of truth for input data; `CALENDAR_AGENT_*_DATA_DIR` env vars cascade |
| `/rl-stop` skill + `scripts/utils/stop_run.sh` | `3b06a0e` | Reads `metadata.jsonl` for the exact pid, refuses kill if recorded script doesn't match — prevents the cross-run pgrep-filter mistake |
| `/rl-status` skill + lnav 0.12.4 + 3 format files | `3b06a0e` | SQL queries on `heartbeat.jsonl` / `deadlock_detected.jsonl` / `metadata.jsonl` instead of multi-step grep rituals |
| MAX_HOURS mid-run enforcement | this commit | Watcher subshell that signals python at the wall-clock deadline |
| `src/calendar_agent/run_telemetry.py` shared module + wired into all 4 trainers | this commit | One call sets up metadata + heartbeat + stuck-alert + Patch G phase-getter; eliminates drift |

## Verification

The dual-tenant test on 2026-04-25 (real-RL on slice 0 + small-RL on slice 1,
both with isolation) showed:
- Solo baseline tps: mean **703**, median 731.
- Dual-tenant tps: mean **706**, median 627 (-14% on median, ~0% on mean).

Versus yesterday's unisolated contention: **50% tps drop**. The isolation
delivers most of the win.

We did NOT test:
- Two same-size trainings (14B + 14B) with isolation, ≥2 hour soak.
- Negative control (isolation off on the new wrapper) to A/B confirm.

## Open follow-ups

- Two-real-tenants ≥2h soak test before assuming we can run multiple
  experiments concurrently as a steady state.
- Decide whether to bring `runs/archive/sft_v5_qwen1.5b/` historical eval
  JSONs out of git (~130k lines of mostly stable artifacts).
- The reward-collapse detector was explicitly out of scope per user — but
  if multiple trainings become routine, revisit.

## Lessons

- "It's running" ≠ "it's training well." The discovery was 12 hours late
  because every monitoring hook only looked at progress/liveness, not
  reward dynamics.
- A safety net you've shipped but never tested is not a safety net.
  Patch K v1 was in the repo for ~12 hours before we needed it; we found out
  it was broken at exactly the wrong time. (Mitigation now: any future
  patches that exist for recovery purposes get a smoke test on small-RL
  before being trusted.)
- MIG slice isolation is necessary but not sufficient. Always set
  `taskset` + thread caps. The new `auto_restart.sh` enforces this; resist
  the urge to "just for one quick run, skip it."
