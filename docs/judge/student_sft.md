# Student Judge SFT — design log

> Started 2026-05-04. Tracks the data-pipeline + eval decisions for distilling
> the Qwen3-14B v2 router into a smaller (Qwen3-8B) student judge.

## Goal

Match the v2-router judge's accuracy on tier-1 holdout (≥92% overall, per-cat
gates) while shrinking inference cost from ~11s/call to ≤0.3s/call. Latency,
not accuracy upgrade, is the win.

Two-phase plan:
- **Phase 1 (this doc):** SFT distillation. Student trained on (prompt → CoT
  + verdict) pairs from the v2 router teacher. CoT preserved at training time.
- **Phase 2 (deferred):** GRPO with reward = `1{correct} * (1 − λ * max(0, len − target)/target)`.
  Length penalty applies only to correct verdicts so the student can't game by
  being short-and-wrong. Anchor KL more strongly on Complex/Vague.

Match-teacher target on Complex/Vague (~78%). Don't try to exceed teacher
on the hardest categories — distillation approaches teacher, doesn't surpass
without a stronger label source.

## Data sources

Three sources mined into `data/judge/v2_20260502/student_candidates_combined.jsonl`
(13,517 rows after holdout exclusion + per-sid cap of 15 + 12-char hash dedup):

| Source | Path | Raw rows | Notes |
|---|---|---:|---|
| Live RL judge calls | `runs/judge_service_20260501/calls.jsonl` | 5,396 | Best match to deployment dist; verdicts re-judged with v2 |
| SFT eval rl-results | `runs/**/eval/checkpoint-*.json:.rl.results` | 3,774 | 14 SFT ckpts × 280 sids |
| RL training parquets | `runs/rl_adaptive_qwen3_14b_*/.art/.../trajectories/train/*.parquet` | 27,750 | Replayed via `CalendarEnvironment` (`scripts/data_generation/student_judge_mine_parquets.py`) to materialize `before`/`after`. 76% replay-success rate. |

Replay script reuses `dispatch_tool_call` / `snapshot_events` /
`format_day_state_text` from production. Agent's recorded tool calls are
re-executed against the env exactly as in RL training, so `before`/`after`
text is byte-identical to what the judge would have seen at RL time.

## Re-judging

Every candidate gets two fresh verdicts (router-v1 verdicts in calls.jsonl
and Gemini verdicts in eval JSONs are not trusted):

- **Qwen-v2** (`build_router_qwen_v2` + `/no_think`, max_tokens=512) → verdict
  + raw CoT (the SFT target). 13,517 / 13,517.
- **Gemini-v2** (`build_router_gemini_v2`, gemini-2.0-flash-001) → verdict
  only. All 13,517 rows re-judged for the silver agreement filter.

Concurrency: vLLM judge service is OOM-sensitive on 24 GiB MIG slices.
Stable at 12 workers/process. Two parallel judge services on separate
slices double throughput; idempotent shard flag (`--shard X/N` on
`scripts/data_generation/student_judge_relabel.py`) splits work by hash.

## Filter pipeline (`scripts/data_generation/student_judge_assemble.py`)

For each candidate, in order:

1. **Holdout filter:** drop if sid in `data/judge/v2_20260502/holdout_sids.json` (82 sids).
2. **No Qwen relabel:** drop (vLLM crash recovery left a few rows unjudged).
3. **Gold lookup:** match `(sid, hash8)` against
   `data/judge/v2_20260502/{train,disagreements}.jsonl`. If matched →
   label = gold label, label_source = gold's source.
4. **Silver path** (if no gold match):
   - Qwen-v2 ∧ Gemini-v2 must agree on verdict
   - label = the agreed verdict, label_source = `two_way_agree`
5. **Qwen-CoT-matches-label gate:** drop if Qwen-v2's verdict (and therefore
   its CoT's conclusion) doesn't match the assigned label. Prevents teaching
   the student wrong reasoning.
6. **Min CoT length gate:** drop rows where `target_cot < 200 chars`. Catches
   the ~7% of teacher outputs where Qwen skipped reasoning entirely (just
   emitted `Correct`/`Incorrect`). Training on those would teach the student
   to skip reasoning.
7. **Per-cat class balance:** trim majority class to natural live ratio
   (target 0.73 ± 0.05 Correct, currently imperfect — see "Open issues").
8. **Per-cat silver cap:** silver per cat ≤ `SILVER_CAP=30 ×` gold count.

## Decisions

### Direction shift: SFT-14B as judge (2026-05-05)

While running validation experiments to choose a silver-label filter, we
evaluated `sft_v6 ckpt-4659` (the calendar-agent SFT model, never trained as
a judge) on tier-1 and the candidate pool. **It hit 91.8–93.6% on tier-1
with a 26-char median output** (literally `<think></think>Correct`).

| Judge | Tier-1 acc (run1/run2) | Median output |
|---|---|---|
| Base Qwen3-14B + router-v2 | 90.9% / 91.8% | 1011 chars |
| **SFT-14B (no judge prompt-tune)** | **91.8% / 93.6%** | **26 chars** |
| Gemini-2.0-flash + router-v2 | 94.5% / 93.6% | n/a |

Implication: the long CoT in `target_cot` was a *crutch the base model needed*
to be accurate, not load-bearing reasoning. A model that's accurate without
it just doesn't write it. This calls into question the whole premise of
distilling Qwen-v2's CoT into a smaller student.

CoT length on SFT-14B is bimodal:
- pred=Correct → exactly 26 chars (just the verdict)
- pred=Incorrect → 200–500 chars of justification

Live distribution is ~73% Correct, so most production calls would be 26-char
sub-second responses on a 14B model.

**Phase-1 (student SFT) is cancelled.** Validation showed:

1. Sweep across all 11 grpo-sft4659 checkpoints (steps 500–4953) on tier-1
   shows **flat accuracy 91.8–93.6%** — RL on top of SFT did not improve
   judge ability. Step 500 is as good as step 4952. The judge ability comes
   entirely from the SFT stage.
2. Sweep across all 12 grpo-base checkpoints shows the same flat 90.9–93.6%
   range, but with 12-second median latency (long CoT, no terse-correct).
3. On tier-2 hard cases, rl-sft-4952 hits 60.0% vs sft14b 52.9% — small edge
   inside CIs but consistent across the latency-free regime.
4. On a 500-row pool sample, `rl_sft ∧ gem` gives 3.56% noise-proxy vs
   `qwen_v2 ∧ gem` 10.18% — the cleanest silver-filter candidate.

**Decision (2026-05-05):** ship `rl-sft-4952` as the judge. Updated
`scripts/serving/judge_service.sbatch` to support `LORA_PATH` env var (now
required — no fallback to base-only).

**Prompt switch (2026-05-06):** sweep on rl-sft-4952 with 12 prompt variants
× 3 repeated runs showed `router_v1` (the original 2026-04-30 per-cat map)
beats `router_qwen_v2` by **+5.5pp tier-2 (60.4 vs 54.9, 4σ)** and **+8.9pp
on Complex (68.9 vs 60.0, 1.7σ)** with tier-1 unchanged at 91.5%. Reason:
qwen_v2 was CV-tuned for Qwen3-14B base in /no_think mode; rl-sft has
different priors (terse-correct from SFT) and a *less aggressive* per-cat
dispatch fits better. Switched `JUDGE_ROUTER=qwen_v2 → JUDGE_ROUTER=router`.

**Live launch:**
`sbatch --export=ALL,MODEL_PATH=runs/sft_v6_qwen3_14b_20260420/eval/merged_tmp_4659,LORA_PATH=.art/grpo-sft4659-20260426/.../checkpoints/4952,JUDGE_ROUTER=router,JUDGE_NO_THINK=1 scripts/serving/judge_service.sbatch`

**Self-improving loop is real but bounded:** SFT'ing the agent gives a free
fast judge. RL on top does NOT compound on the judge side. So refresh the
judge after each SFT round, not after each RL step.

Open follow-ups (not blocking):
- Re-eval all SFT/RL agent checkpoints with the new judge — current rankings
  use 86.7% Gemini judge; rl-sft is ~93%, so old numbers are noisier than
  needed.
- Mixed-SFT experiment: train a fresh Qwen3-14B on agent + judge data
  jointly. Could push 93% → 95%+ at the same latency.

Validation artifacts: `runs/judge_filter_validation_20260505/`.

### Silver-block relaxation for Complex/Vague (2026-05-04)

The v2 dataset rule was: silver disallowed in Complex (P(both wrong | agree) ≈ 12%)
and Vague (≈ 8%). For SFT this rule was too restrictive — those cats only
got ~50 gold-matched rows each.

**Decision:** silver-block disabled in `student_sft_train.jsonl` only.
`eval.jsonl` remains gold-only. Rationale: SFT distillation is robust to
~10% label noise; Complex/Vague growing from 50 to ~1000 rows is worth more
than guaranteed-clean labels at small scale. The locked tier-1 holdout is
unchanged so eval metrics still reflect true accuracy.

### Train/dev split: held-out calendars (2026-05-04)

Originally proposed held-out sids (cal_X_q_Y). Switched to **held-out
calendars** (cal_X) — if cal_X is in dev, no rollout of any cal_X_q_Y is
in train. Stronger generalization test: the student has never seen the
calendar's persona / event corpus / style.

This subsumes the "same calendar, different question" leakage risk.

### Class-balance philosophy (2026-05-04)

Original target was 0.73 pos_frac (live distribution). User pushed for
**~0.5 per cat** to avoid biasing the student. Direction is to grow
under-represented classes (more Incorrect rollouts, especially RelTime
which is 91% Correct), not down-trim majority class.

Mitigation if this is hard to fully achieve: class weights at SFT time.

### Min CoT length filter (2026-05-04)

Spot check found 7.4% of teacher CoTs are <200 chars (just verdict-only
outputs from rare teacher behavior). Filter set at 200 chars. Drops ~350
rows. Documented in spot-check notes — these are not "duplicates" but
genuine teacher misbehavior we don't want to distill.

### Re-judging discipline (2026-05-04)

Stored verdicts in source files (router-v1 in calls.jsonl, Gemini-2.0-flash
in eval JSONs from older judge versions) are bookkeeping only. Every row
gets fresh Qwen-v2 + Gemini-v2 labels at relabel time. Otherwise we'd be
distilling router-v1's mistakes.

## Eval design

Tiered, with each tier catching a different failure mode:

| Tier | Set | Path | Catches |
|---|---|---|---|
| 0 | train-dev (~10% of cals held out) | `student_sft_dev.jsonl` | ckpt selection during SFT — held-out *calendars* |
| 1 | anchor (locked) | `eval.jsonl` (110) | absolute correctness, ship gate |
| 2 | hard cases | `disagreements.jsonl` (85) | catches student that only learned easy signal |
| 3 (planned) | live distribution | TBD | distribution shift to current RL trajectories |
| 4 (planned) | independent arbiter | GPT-OSS-20B / Claude-agent triangulation | correlated-error catch |

Per-cat ship gates on tier 1 (with margin for distillation gap):

| Cat | Qwen-v2 (teacher) | Student target |
|---|---:|---:|
| Modifier | 95% | ≥90% |
| IR | 95% | ≥90% |
| Schedule | 93% | ≥88% |
| Chaos | 93% | ≥88% |
| RelTime | 92% | ≥87% |
| Vague | 92% | ≥85% |
| Complex | 78% | ≥73% |

Plus overall ≥88% on tier 1.

Eval-correctness diagnostics in the harness:
- 95% bootstrap CIs per cat (n is small, point estimates lie)
- **Pos-stratified accuracy:** acc on Correct rows vs acc on Incorrect rows.
  Catches prior-domination (e.g. high overall acc but only because student
  predicts "Correct" everywhere).
- Confusion matrix (label → pred).

Planned but not yet implemented:
- **Effective sample size per cat:** cluster within-sid rollouts by content
  similarity, count distinct clusters. Train has ~470 unique cals but
  ~5,000 rows; effective N is closer to 1,000.
- **Per-sid prediction correlation:** intra-sid accuracy variance vs
  inter-sid. Catches memorization.
- **Tier 3 (live):** sample 500 fresh `calls.jsonl` rows, label via 3-way
  Qwen-v2 ∧ Gemini-v2 ∧ Claude-agent agreement, retain only where all
  three agree.

## Files

| What | Path |
|---|---|
| Trajectory miner (calls.jsonl + eval JSONs) | `scripts/data_generation/student_judge_mine.py` |
| Trajectory miner (RL parquets, with replay) | `scripts/data_generation/student_judge_mine_parquets.py` |
| Re-judger (Qwen-v2 + Gemini-v2) | `scripts/data_generation/student_judge_relabel.py` |
| Filter / assembler | `scripts/data_generation/student_judge_assemble.py` |
| SFT trainer | `scripts/training/judge/student_sft_train.py` |
| Tiered eval harness | `scripts/eval/student_judge_eval.py` |
| Combined candidate pool | `data/judge/v2_20260502/student_candidates_combined.jsonl` |
| Qwen-v2 relabels | `data/judge/v2_20260502/relabel_qwen.jsonl` |
| Gemini-v2 relabels | `data/judge/v2_20260502/relabel_gemini.jsonl` |
| SFT train | `data/judge/v2_20260502/student_sft_train.jsonl` |
| SFT dev | `data/judge/v2_20260502/student_sft_dev.jsonl` |
| Assembly metadata | `data/judge/v2_20260502/student_sft_metadata.json` |

## Open issues

1. **Class balance not at 0.5.** Modifier is 0.50 ✓ but Chaos 0.72, RelTime 0.91.
   Path forward: weak-model (4B) Incorrect-generator pass to grow the
   under-represented Incorrect class without trimming the majority.
2. **Complex/Vague still small after silver-block relaxation.** Pending
   Gemini-v2 pass on those cats. Estimated ~900-1,200 silver rows in each
   after agreement filter.
3. **Tier 3 live eval not yet built.** Distribution-shift signal is the
   biggest unknown.
4. **No effective-sample-size diagnostic.** Currently reporting raw row
   counts which overstate signal.
