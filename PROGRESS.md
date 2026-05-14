# Training Pipeline Progress

> **Last updated:** 2026-05-14
> **Best agent model:** ORPO ckpt-600 — **84.25%** on `test_data/` (held-out, canonical), +4.1 pp over SFT v6 ckpt-4659 (80.1%). ckpt-427 ties at 84.2%. See `runs/rl_orpo_qwen3_14b_20260508_0625/`.
> Older RL-data benchmark winner is SFT v6 ckpt-6212 (ep 4) at 82.5%; different ckpts win on different sets.
> **Production judge (2026-05-08):** **Gemini-2.0-flash + `router_structured`** at **92.98%** on the tool-audited 285-trajectory oracle. Served via `scripts/serving/judge_service_gemini.sbatch` (no GPU; calls Vertex AI; same `/verdict` + `/health` API on `:8765`). Smoke p50 ~2.2–2.5s. Local v3 / rl-sft-4952 service is parked but preserved.
> **Previous local-judge deployment (2026-05-06, parked):** `rl_grpo_qwen3_14b_sft4659 @ step 4952` (LoRA on SFT v6 ckpt-4659 merged base) with `JUDGE_ROUTER=router JUDGE_NO_THINK=1`. Tier-1 ~92%, Tier-2 ~60%, p50 670ms. Superseded by Gemini-structured.
> **Re-tuned per-judge maps (v2, retired):** `ROUTER_MAP_QWEN_V2` / `ROUTER_MAP_GEMINI_V2` in `src/calendar_agent/judge/prompts.py` — kept around for offline relabeling but no longer the served path.
> Gemini-2.0-flash + EVAL_SYSTEM_PROMPT (incumbent): 86.67%. See [`docs/judge/`](docs/judge/).
> **Canonical judge truth:** `runs/judge_baseline_20260430/eval/manual_verdicts.jsonl` (285 hand-labeled ART trajectories, 185 Correct / 100 Incorrect). 30% stratified holdout locked at `data/judge/v2_20260502/holdout_sids.json`.
> **Curated v2 dataset:** `data/judge/v2_20260502/{train.jsonl (744), eval.jsonl (110), disagreements.jsonl (85)}`. Supersedes v1 `data/judge/{train,val}.jsonl` (built on 86.7% Gemini labels).
> **Judge-as-a-service:** FastAPI sidecar wrapping vLLM at `:8765` (`src/calendar_agent/judge/`, `scripts/serving/judge_service.sbatch`).
> RL trainers swapped from Gemini to local judge (no Gemini fallback).
> **Active work:** Phase 1.5 judge distillation **CANCELLED 2026-05-05**. RL ckpt `rl_grpo_qwen3_14b_sft4659 @ step 4952` (LoRA on top of SFT v6 ckpt-4659) shipped as the judge: tier-1 ~93%, median 670ms, 26-char output for Correct verdicts, 3.6% noise vs Gemini on the candidate pool (vs 10.2% for the deployed Qwen-v2 router). Sweep across all 11 grpo-sft4659 ckpts shows flat judge accuracy — judge ability comes from the SFT stage; RL on top does not compound on the judge side. Updated `scripts/serving/judge_service.sbatch` to support LoRA serving. See `docs/judge/student_sft.md` "Direction shift".

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
runs/sft_qwen3_8b_20260502/            SFT Qwen3-8B — RUNNING (slurm 89)
runs/sft_qwen3_4b_20260502/            SFT Qwen3-4B — RUNNING (slurm 90)
```

---

## Timeline

### 2026-05-14 — ORPO trainer ran end-to-end; new RL baseline at +4.1 pp held-out

First successful preference-RL run on Qwen3-14B. **ORPO ckpt-600 hits 84.25%
on `test_data/` (+4.1 pp over SFT v6 ckpt-4659 baseline of 80.1%)**;
ckpt-427 ties at 84.2%. New best agent ckpt.

**Run.** `runs/rl_orpo_qwen3_14b_20260508_0625/`. Trainer
`scripts/training/rl/rl_orpo.py` (design in `docs/orpo/design.md`).
Initial weights: SFT v6 ckpt-4659 *merged into base*
(`runs/sft_v6_qwen3_14b_20260420/eval_test/merged_tmp_4659`); ORPO LoRA
fresh r=8. Config: β=0.1, λ=1.0, LR=5e-6, K_HARD=8 / K_EASY=4,
N_QUERIES_PER_STEP=20, BUFFER_PER_SCENARIO=4, per_device_bs=1,
grad_accum=16. 622/622 steps (20 epochs) completed. Wall-clock ~3 days
+ ~36 h resume = ~4.5 days total.

**Resume bugs (now patched).** First resume after 72 h slurm timeout
crashed: optimizer Adam moments saved on CPU but params re-loaded on GPU
— fixed in `orpo_train_step` (move opt state to param device after
`reload_to_gpu`). Buffer was not persisted across the first restart and
re-warmed cold (cost ~150 steps); added `ReuseBuffer.save/load`
(`src/calendar_agent/orpo/reuse_buffer.py`) and persistence hooks in
`rl_orpo.py` for future resumes.

**Held-out trajectory (Gemini-structured judge, `test_data/`):**

| ckpt | overall | notes |
|---|---|---|
| SFT v6 ckpt-4659 (baseline) | 80.1% | |
| 50 | 79.9% | ≈ SFT |
| 100 | 80.6% | |
| 150 | 80.5% | |
| 200 | 83.1% | first breakthrough |
| 250–400 | 82.1–82.8% | |
| **427** | **84.2%** | tied peak |
| 450 | 82.2% | post-resume regress |
| 475–525 | 82.95–83.8% | |
| 550 | skipped (eval retry produced 0%) | |
| 575 | 82.8% | |
| **600** | **84.25%** | tied peak, new best |
| 620 | 83.1% | |
| 621 | in flight | |

**Per-category at ckpt-600 vs ckpt-50 (≈ SFT):** Vague 74.5 → 85.7
(+11.2), Schedule 79.6 → 87.8 (+8.2), Complex 59.2 → 66.3 (+7.1, still
the bottleneck), Chaos 81.6 → 84.7 (+3.1), RelTime 83.7 → 86.5 (+2.8),
IR 93.9 → 93.9 (0), Modifier 86.7 → 84.7 (−2.0, small forgetting).

**vs prior GRPO run (`rl_grpo_qwen3_14b_sft4659 @ 4952`).** GRPO ~3.5 d
active for +5 pp / ~40k rollouts; ORPO ~3 d active for +4.1 pp / ~51k
rollouts. Per wall-clock roughly tied. ORPO trained on ~30k preference
pairs vs GRPO's 4,952 advantage updates → **~6× more signal-dense per
rollout**.

**Algorithm notes worth keeping.**
- Difficulty sampler `1 − pᴳ − (1−p)ᴳ` worked at margins (mid bucket
  29× over-sampled vs uniform) but couldn't escape easies as pool
  composition skewed to 574 easy / 1 mid by step 621.
- Adaptive-k cut rollout cost: k=8 share 67% → 5%; step time 819s →
  275s (−66%).
- Buffer rescue hit 88% peak hit-rate; reset on first resume cost
  ~150 steps re-warm.
- Late-training waste: skip_easy 8.7 → 16/20 per step; pair-producing
  scenarios collapsed 272 → 34.
- Margin grew 0.14 → 1.19 peak, rew_acc 0.59 → 0.71; no likelihood
  displacement (logp_chosen rose monotonically toward 0).

**Followups.** Promote ckpt-600; update
`runs/analysis/test_eval_summary.md` and the per-category table below
when re-eval lands. Next iteration (separate experiment, see
`docs/orpo/design.md` "Postmortem / v2 plan"): replace static
difficulty weighting with DAPO-style dynamic sampling
(oversample → filter std=0 → accumulate); raise β to 0.3; LoRA rank → 16.

### 2026-05-08 — Judge v3 SFT, structured prompts, oracle re-audit, Gemini-structured shipped

One session, four connected milestones. Net result: **production judge moves
from local rl-sft-4952 to Gemini-flash + `router_structured` at 92.98%** on
the tool-audited oracle.

**1. Judge v3 SFT (Qwen3-14B base + LoRA r=64, 1 epoch).** Combined corpus
from existing reasoning-bearing sources (v1 eval-JSON 7,482 + v2 Gemini
relabel 13,520), deduped to 10,481 → 9,915 train / 566 val. Working config
after speed iterations (slurm 152→155): bs=4, seq=1024, grad_ckpt=off,
~2h21m on one MIG slice. Train loss 0.389, eval loss 0.375. Output:
`runs/judge_v3_qwen3_14b_20260507/checkpoints/final` (996 MB LoRA).
Builders: `scripts/data_generation/build_judge_v3.py`,
`data/judge/v3_20260507/{train.jsonl,val.jsonl,metadata.json}`.

v3 on the (then) 285 manual oracle: **87.02%**. Per-cat: Modifier 95.12,
Schedule 94.74, IR 90.48, RelTime 89.47, Vague 86.84, Chaos 84.78,
**Complex 71.43**. Generation analysis: 25.3% of v3 outputs spontaneously
emitted an (A)–(E) checklist (inherited from corpus); that subset hit 94.4%.
Forcing checklist via prompt didn't transfer — selection bias confirmed.

**2. Structured prompts (per-cat).** `src/calendar_agent/judge/{features.py,
structured_prompts.py}` replaces raw before/after dumps with pre-computed
DIFF (with explicit MOVED detection for cross-day relocations),
RESPONSE_CITATIONS, AGENT_ACTION, EXPECTED_ANSWER_TYPE, RESPONSE_WELL_FORMED.
Tailored prompts for Modifier, Chaos, Complex, IR, RelTime; Schedule and
Vague fall back to plain `router`. Variant `router_structured` lives in
`scripts/eval/judge_gemini_router.py`. With Gemini-flash on the 285:
plain router 89.47% → structured 90.88% → after fixes (MOVED diff,
EXPECTED_ANSWER_TYPE, hard-fail RESPONSE_WELL_FORMED, IR + RelTime added)
**93.33%** on the relabeled oracle.

**3. Oracle re-audit — 2 of 5 prior flips were wrong.** The 2026-05-01
relabel (5 flips from "4-judge unanimous disagreement with gt") had errors.
Tool-using Gemini (`scripts/eval/judge_tool_sim.py`,
`scripts/eval/judge_tool_eval.py`) re-audited each flip with access to ask
for source state instead of being fed it. Result: **3 confirmed**
(cal_8_q_8, cal_19_q_8, cal_22_q_2), **2 reverted** (cal_32_q_7,
cal_19_q_1). The 2 bad flips were cases where 4 context-fed judges all
hallucinated the same wrong fact (e.g. assumed "Dinner with Family" had
attendees when source data had none); the tool-using judge had to look it
up and so didn't fall in. Canonical file:
`runs/judge_baseline_20260430/eval/manual_verdicts_relabeled.jsonl`
(3 flips, audit log in `..._meta.json`).

**4. Final scores on tool-audited relabeled gt:**

| judge | overall | notes |
|---|---|---|
| Local v3 | 88.07% | Complex 71.43% — capacity-limited |
| Gemini router (plain) | 91.93% | |
| **Gemini structured** | **92.98%** | shipped |
| Gemini tool-using | 85.96% | 4× cost, +44% latency; only wins Complex (95.24 vs 92.86) |

Tool-using judge underperformed overall — useful as an audit tool, not as
a production judge.

**5. Production switch.** Built `src/calendar_agent/judge/server_gemini.py`
(drop-in for `server.py`) + `scripts/serving/judge_service_gemini.sbatch`
(no GPU, Vertex AI). Same `/verdict` and `/health` on `:8765`. Smoke p50
2.2–2.5s. RL trainers unchanged (still hard-fail rc=43 on judge errors).
Local v3 / rl-sft-4952 service preserved for offline relabeling.

### 2026-05-02 — Smaller-model SFT sweep launched (8B + 4B)
Mirroring SFT v6 hyperparams (5 epochs, LoRA r=64 on q/k/v/o + gate/up/down,
LR 2e-4 cosine, bs=1×grad_accum=4, bf16, 4-bit, max_seq_len=4096,
paged_adamw_8bit, same `sft_data/trajectories_augmented/`) on smaller bases
to see whether 14B is over- or under-sized for this task before more RL spend.

- `Qwen/Qwen3-8B` → `runs/sft_qwen3_8b_20260502/` (slurm 89)
- `Qwen/Qwen3-4B` → `runs/sft_qwen3_4b_20260502/` (slurm 90)
- 1.7B deferred — only 2 free MIG slices (judge service holds the third).
- Parameterized launcher: `scripts/training/sft/sft_train.sbatch`
  driven by `SFT_MODEL_NAME`/`SFT_RUN_DIR`/`SLICE`. `sft_train.py`
  now reads `SFT_MODEL_NAME` from env (default unchanged: Qwen3-14B).

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

- **Judge** — see [`docs/judge/`](docs/judge/) for everything.
  - Production: Gemini-flash + `router_structured` at 92.98% on tool-audited
    oracle. Served via `judge_service_gemini.sbatch`.
  - Local v3 (Qwen3-14B SFT, 87.02% / 88.07%) parked; Complex remains the
    weakest category (71.43%) and is the natural next target if a local
    judge is revisited.
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
