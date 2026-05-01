# Training Pipeline Progress

> **Last updated:** 2026-05-01
> **Best agent model:** SFT v6 ckpt-4659 (ep 3) — **80.1%** on `test_data/` (held-out, canonical).
> Older RL-data benchmark winner is ckpt-6212 (ep 4) at 82.5%; different ckpts win on different sets.
> **Best local judge:** Qwen3-14B fp8 + router prompt — **95.44%** on the manual oracle.
> Gemini-2.0-flash on the same oracle: 86.67%. See [`docs/judge/`](docs/judge/).
> **Canonical judge truth:** `runs/judge_baseline_20260430/eval/manual_verdicts.jsonl`
> (285 hand-labeled ART trajectories, 185 Correct / 100 Incorrect).
> **Judge-as-a-service:** FastAPI sidecar wrapping vLLM at `:8765`
> (`src/calendar_agent/judge/`, `scripts/serving/judge_service.sbatch`).
> RL trainers swapped from Gemini to local judge (no Gemini fallback).
> **Active work:** Phase 1.5 judge distillation. RL paused — saturated under
> binary rewards on this dataset (see 2026-05-01 entry).

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
runs/rl_qwen3_14b_20260420/            RL GRPO — paused at step 9220 after 2026-04-25 cliff
runs/rl_adaptive_qwen3_14b_20260424/   RL adaptive — paused (interfered with main RL run)
runs/rl_grpo_qwen3_14b_base_20260426/      RL GRPO from base — paused at step 5077 / 12440 (40%)
runs/rl_grpo_qwen3_14b_sft4659_20260426/   RL GRPO from SFT v6 ckpt-4659 — paused at step 4952 / 12440 (39%)
runs/rl_adaptive_qwen3_14b_base_20260426/  RL adaptive (AR3PO) — paused at step 3496 / 12440 (28%)
runs/judge_v1_qwen3_7b_20260425/       Local judge SFT — see local_judge.md
```

---

## Timeline

### 2026-05-01 — Judge-as-a-service deployed
Stood up the local judge as an HTTP service so RL can stop calling Gemini and
also so the verdict prompt logic stops being scattered across eval scripts.

- `src/calendar_agent/judge/` — new package. `prompts.py` lifts
  `CHECKLIST_V2_SYS`, `FEWSHOT_EXAMPLES_{V1..V4}`, the three winning builders
  (`build_fewshot`, `build_fewshot_v3`, `build_fewshot_v4_dayfocus`), and
  `ROUTER_MAP` verbatim from `scripts/eval/judge_prompt_tune.py`. Stamped
  with `PROMPT_VERSION = "router-v1-20260501"` so reward-signal drift from
  mid-run prompt edits is detectable.
- `server.py` — FastAPI sidecar on `127.0.0.1:8765`. POST `/verdict` takes
  `{cat, query, final, expected, before, after, scenario_id}`, builds the
  router prompt server-side, calls a local vLLM (`/v1/chat/completions`),
  generates full reasoning to keep accuracy at 95.44%, and returns only the
  verdict. Every call is appended to `runs/judge_service_<date>/calls.jsonl`
  with the full prompt + raw response — that JSONL is also Phase 1.5
  distillation training data.
- `client.py` — async client with explicit `JudgeUnavailable`. **No Gemini
  fallback** — RL trainers `sys.exit(43)` on judge errors so a downed
  service halts auto-restart instead of silently zeroing the reward signal.
- `scripts/serving/judge_service.sbatch` — 1× MIG slice, vLLM (Qwen3-14B
  fp8 + hermes parser, `:8000`) + FastAPI sidecar (`:8765`), taskset/OMP
  isolation, metadata.jsonl start/stop stamping.
- Swapped Gemini-based `evaluate_trajectory` in `rl_train.py` and
  `rl_train_adaptive.py` to call the judge client. `rl_train_small.py`
  not migrated yet.

**Tested:** end-to-end `/verdict` with 10 manual-oracle samples → 10/10
agreement. Trainer-side integration test (importing the swapped
`evaluate_trajectory` and calling it) passed both the happy path (Correct
verdict) and the hard-fail path (`JUDGE_URL=:9` → `SystemExit(43)`).
Smoke scripts: `scripts/eval/judge_service_smoke.py`,
`scripts/eval/judge_rl_integration_smoke.py`.

### 2026-05-01 — RL paused: dataset saturated under binary rewards
After ~3 days of training across 3 concurrent runs, halting RL for now and
moving to other phases (judge distillation, harder data, shaped rewards).

**What was running:**
| run | starting LoRA | last step / 12440 | mean reward (recent 500) |
|---|---|---|---|
| `rl_grpo_qwen3_14b_base_20260426` | Qwen3-14B base | 5077 (40%) | ~0.85 |
| `rl_grpo_qwen3_14b_sft4659_20260426` | SFT v6 ckpt-4659 | 4952 (39%) | ~0.84 |
| `rl_adaptive_qwen3_14b_base_20260426` | Qwen3-14B base | 3496 (28%) | ~0.65 |

**Skip-rate fix on adaptive (2026-04-28).** The original adaptive sampler
was producing 70% skipped steps because the per-scenario weight had a 0.65
floor and a 3× retest boost on *easy* scenarios. Replaced with the analytic
non-skip probability `1 − pᴳ − (1−p)ᴳ`, plus AR3PO's two recovery
mechanisms (multi-stage rollout + per-scenario response-reuse buffer).
Skip rate dropped to ~28% within 50 steps. See
`src/calendar_agent/scenario_tracker.py` and `tests/test_scenario_tracker.py`.

**Why we're stopping anyway.** The fix worked, but reward stopped climbing.
Inspecting the live tracker after ~3500 post-fix steps:

| pass-rate band | n scenarios |
|---|---|
| p > 0.95 (saturated solved) | 441 |
| 0.30 ≤ p ≤ 0.70 (productive) | ~35 |
| p < 0.05 (model has memorized failure) | 49 |

So **91% of the 622-scenario pool is at one of the two saturated extremes**
under the current Qwen3-14B + binary-reward regime. The "real" training set
is the ~35 scenarios in the productive middle, which by themselves can't
drive further headline gains. Multi-stage rollout dutifully extends groups
on the 49 dead-hard cases to 24 rollouts each — gradient signal is weak
because the policy has memorized the failure mode (no exploration pressure
since `temperature=1.0`, `beta=0.0`).

**Implications going forward** (none scheduled for this round):
1. Shaped/partial rewards would re-populate the productive band.
2. KL anchor (`beta>0`) against base/SFT to prevent collapse on saturated easy.
3. Higher rollout temperature on hard bucket to break memorized failures.
4. Fresh harder training scenarios — the existing 622 are chewed through.
5. Evaluate the latest checkpoints on `test_data/` to see if the off-policy
   gains are real even when headline reward is flat.

**What ships in this commit batch:**
- `scripts/training/rl/rl_grpo_base.sbatch`, `rl_grpo_sft4659.sbatch`,
  `rl_adaptive_base.sbatch` — three slurm submitters wired to
  `auto_restart.sh`.
- `src/calendar_agent/scenario_tracker.py` — new sample_weight semantics
  (deletes RETEST_BOOST + WEIGHT_FLOOR-0.65 ad-hoc heuristics).
- `scripts/training/rl/rl_train_adaptive.py` — multi-stage rollout +
  response-reuse buffer; on-policy filtering for headline metrics.
- `tests/test_scenario_tracker.py` — 12 unit tests including a
  production-shape simulation.

### 2026-05-01 — Judge prompt tuning + latency budget
- 19 prompt variants on Qwen3-14B fp8. Best is `router` (per-category dispatch
  to specialised few-shot prompts): **95.44%** on the manual oracle, **+8.8 pp
  over Gemini**. Full leaderboard: [`docs/judge/prompt_tuning.md`](docs/judge/prompt_tuning.md).
- 4 manual labels relabeled (cases #8, #39, #65, #280) — empty-diff +
  claimed-success cases that were originally lenient. New oracle: 185/100
  Correct/Incorrect. Originals saved to `manual_verdicts_v1.jsonl`.
- Latency benchmark vs Gemini-2.0-flash (extracted from RL production logs,
  6,213 calls): Gemini p50 = 0.46s, our 14B fp8 router p50 at concurrency=16
  = 0.83s, p50 at concurrency=1 = 9.77s. Cudagraphs and AWQ gave ~0 speedup
  on Blackwell+fp8 (decode is memory-bandwidth bound). Speculative decoding
  hurts at concurrency≥4. Verdict-only output without training drops accuracy
  to 71.93%. See [`docs/judge/latency.md`](docs/judge/latency.md).
- Conclusion: **prompt engineering is at its frontier** for this stack; the
  remaining latency gap to Gemini requires distillation (verdict-only target)
  rather than prompt tricks. See [`docs/judge/plan.md`](docs/judge/plan.md).

### 2026-04-30 — Judge baseline + canonical manual labels
Sampled 286 trajectories from the RL adaptive ART hold-out and ran three
base-model judges (Qwen3-8B, 14B, 32B) plus the existing Gemini gt label.
Then hand-labeled all 285 (one trajectory had no scenario record) to create
a canonical truth set:

`runs/judge_baseline_20260430/eval/manual_verdicts.jsonl` — **the verdict
oracle going forward**. Distribution: 187 Correct / 98 Incorrect.

Accuracy vs manual: **Gemini 86.7%**, 32B base 82.1%, 14B base 79.0%, 8B base
71.9%. Gemini is still the best judge but only by ~4.5 pp over 32B base, and
14B base actually beats Gemini on Human Chaos (82.6% vs 76.1%) — Gemini is
too lenient when the agent hedges/asks for clarification on fragmentary
queries. 32B beats Gemini on Modifier (90.2% vs 85.4%).

**Implication for RL:** the existing GRPO runs train against a 86.7%-accurate
reward signal — ~13% of trajectories are mis-rewarded, capping how well any
RL run can do. A trained local judge has real headroom to match or beat
Gemini on the hard categories.

Full breakdown: [`runs/judge_baseline_20260430/eval/SUMMARY.md`](runs/judge_baseline_20260430/eval/SUMMARY.md).

### 2026-04-26 — Multi-tenant hardening + Patch K v2 verified
Pipeline-level cleanup and safeguards after the 2026-04-25 reward cliff
(see [`docs/incidents/2026-04-25_reward_cliff.md`](docs/incidents/2026-04-25_reward_cliff.md)):

- **Patch K v2 (real fix)**: ART's `model.delete_checkpoints` was pruning
  even our just-saved milestone. New code uses
  `art.local.checkpoints.delete_checkpoints(output_dir, excluding)` directly
  with explicit `[latest, best, ...all milestones]` keep-list. Verified
  end-to-end on small-RL: 6 milestones survived 55 step-cycles.
- **Multi-tenant isolation**: centralized `scripts/training/common/auto_restart.sh`
  enforces `OMP_NUM_THREADS=8`, taskset CPU pinning per MIG slice, mid-run
  `MAX_HOURS` watcher. `slice_map.sh` is the single source of truth for
  MIG ↔ CUDA UUID ↔ CPU range.
- **Telemetry shared module** (`src/calendar_agent/run_telemetry.py`):
  `init_telemetry(run_dir, script_path)` writes metadata, starts heartbeat
  + stuck-alert daemons, registers Patch G phase-getter. Wired into all 4
  trainers (rl_train, rl_train_small, rl_train_adaptive, dpo_train).
- **Repo reorg**: `scripts/training/{rl,sft,dpo,judge,common,legacy}/`,
  `data/{rl,sft,test,judge}/` with env-var path overrides in
  `calendar_agent.paths`. Restored `pyproject.toml` + `README.md`.
- **Skills**: `/rl-stop` (safe single-run kill via `metadata.jsonl`) and
  `/rl-status` (lnav SQL queries on the 3 JSONL streams). Replaces
  ~80% of the old grep-based status ritual.

Real-RL is paused at step 9220 (LoRA at `.art/calendar-agent/.../checkpoints/9220/`).
Resume with the new isolation protocol when ready.

### 2026-04-26 — RL adaptive in progress (earlier today)
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

- **Local judge** — see [`docs/judge/`](docs/judge/) for everything.
  - Today: 14B fp8 + router prompt at 95.44% (Phase 0.5 done).
  - Next: Phase 1.5 distillation (router → small model with verdict-only
    output) for the latency win. The previous 7B-on-Gemini-labels run is
    superseded.
  - Phase 3 (RL integration) plan in [`docs/judge/rl_integration.md`](docs/judge/rl_integration.md).
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
