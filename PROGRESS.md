# Training Pipeline Progress

> **Last updated:** 2026-04-26
> **Best model:** SFT v6 ckpt-4659 (ep 3) — **80.1%** on `test_data/` (held-out, canonical).
> Older RL-data benchmark winner is ckpt-6212 (ep 4) at 82.5%; different ckpts win on different sets.
> **Active work:** RL adaptive run on Qwen3-14B (see `runs/rl_adaptive_qwen3_14b_20260424/`).

For per-category breakdowns, failure modes, and improvement targets see
[`docs/categories/`](docs/categories/). For 1.5B-era history (SFT v3/v5, RL1-3 on
TITAN X) see [`docs/archive/progress_1.5b_era.md`](docs/archive/progress_1.5b_era.md).

---

## Active runs

See `runs/<run_dir>/metadata.jsonl` for live status (process start, git sha, MIG slice).
Don't maintain a status table here — it goes stale within hours.

```
runs/sft_v6_qwen3_14b_20260420/        SFT v6 — DONE (2026-04-22)
runs/dpo_qwen3_14b_sft_20260423/       DPO from SFT v6 — DONE, paused
runs/dpo_qwen3_14b_instruct_20260423/  DPO from Instruct — DONE, paused
runs/rl_qwen3_14b_20260420/            RL GRPO — paused after queue deadlock at step 2325
runs/rl_adaptive_qwen3_14b_20260424/   RL adaptive (current focus)
runs/judge_v1_qwen3_7b_20260425/       Local judge SFT — see local_judge.md
```

---

## Timeline

### 2026-04-26 — RL adaptive in progress
Restarted from sft_v6 ckpt-6212. See `runs/rl_adaptive_qwen3_14b_20260424/`.

### 2026-04-25 — DPO experiments paused as uninformative
DPO-from-SFT: 79.3% (−0.8 pp vs SFT). DPO-from-Instruct: 61.3% (−1.7 pp vs Instruct
baseline 63.0%). Per-category trade is wash (Schedule +9, Complex −7). Likely root
causes: off-policy pair source (mined from 5,500-step ART rollouts), pair saturation
near SFT ceiling, likelihood displacement on un-SFT'd Instruct. Full results:
[`runs/analysis/test_eval_summary.md`](runs/analysis/test_eval_summary.md).
Memory note: `feedback_dpo_skipped.md`.

### 2026-04-24 — Held-out `test_data/` benchmark created
49 calendars × 692 queries, 7 categories balanced. No overlap with any training
source. Created because `rl_data/` was contaminated for DPO eval (74% of RL
scenarios appeared in DPO training pairs). Generation: `scripts/data_generation/generate_test_data.py`,
gemini-2.0-flash, seed `20260424`. **Canonical benchmark going forward.**

### 2026-04-22 — SFT v6 (Qwen3-14B) DONE
- Base: `Qwen/Qwen3-14B`, 4-bit bnb, LoRA r=64, bf16, `/no_think` system prompt
- 5 epochs, cosine LR 2e-4, MAX_SEQ_LENGTH=4096, 6,947 augmented trajectories
- Best on RL data: **ckpt-6212 (ep 4) at 82.5%** (+7.9 pp vs SFT v5 1.5B)
- Best on test_data: **ckpt-4659 (ep 3) at 80.1%**
- Eval loss is misleading for ckpt selection (`feedback_eval_loss_misleading.md`)
- Full per-checkpoint summary: `runs/sft_v6_qwen3_14b_20260420/eval/summary.csv`

### 2026-04-20 — Directory reorg
All run output consolidated under `runs/<kind>_<model>_<date>/`. Replaced
scattered `sft_output/`, `rl_runs/`, top-level `logs/`, etc.
See `runs/README.md` and `project_runs_dir_convention.md` (memory).

### 2026-04-15 — Stack upgrade for Blackwell
torch 2.10.0+cu128, vLLM 0.19.0, unsloth 2026.4.4, transformers 4.57.6,
trl 0.24.0, openpipe-art 0.5.17, peft 0.19.0. bf16 + FA2 enabled.

### Pre-2026-04-15 — 1.5B era
SFT v3/v5 on Qwen2.5-1.5B, RL1-3 (Modifier, IR, Vague). GRPO hit a 1.5B
comprehension ceiling (79% of failures were reasoning, not procedure).
Decision: switch to Qwen3-14B QLoRA. Full record:
[`docs/archive/progress_1.5b_era.md`](docs/archive/progress_1.5b_era.md).

---

## Per-category targets (from test_data, SFT v6 ckpt-4659)

| Category | Acc | Status | Doc |
|---|---|---|---|
| RelTime | 95% | saturated | [reltime.md](docs/categories/reltime.md) |
| IR | 90% | saturated | [ir.md](docs/categories/ir.md) |
| Modifier | 89% | saturated | [modifier.md](docs/categories/modifier.md) |
| Chaos | 83% | strong | [chaos.md](docs/categories/chaos.md) |
| Vague | 80% | room | [vague.md](docs/categories/vague.md) |
| Schedule | 70% | room | [schedule.md](docs/categories/schedule.md) |
| Complex | 59% | weakest | [complex.md](docs/categories/complex.md) |

Numbers from `runs/analysis/test_eval_summary.md`. Update when a new ckpt wins.

---

## Open threads

- **Local judge** (Qwen3-7B): replaces ~99k Gemini API calls per RL run.
  Status + design: [`local_judge.md`](local_judge.md).
- **ART asyncio deadlock**: workaround deployed (Patch G/I) but upstream not filed.
  Analysis: [`docs/art_asyncio_deadlock_analysis.md`](docs/art_asyncio_deadlock_analysis.md).
- **RL beyond GRPO+binary rewards**: RFT/expert iteration, Dr. GRPO, non-zero β.
  Notes scattered in archive — synthesize before next attempt.

---

## Key references

- Architecture / paths / imports: `CLAUDE.md`
- Launch protocol: `docs/multi_tenant_training.md`
- Run layout: `runs/README.md`
- Held-out eval results: `runs/analysis/test_eval_summary.md`
- ART runtime patches: `src/calendar_agent/art_patches.py`
