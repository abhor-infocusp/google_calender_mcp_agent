# Local Judge — Current State

> **Deployed 2026-05-05 (model), 2026-05-06 (prompt):** `rl_grpo_qwen3_14b_sft4659 @ step 4952` (LoRA on top of SFT v6 ckpt-4659 merged base) with `JUDGE_ROUTER=router JUDGE_NO_THINK=1 JUDGE_MAX_TOKENS=512`. Tier-1 ~92%, Tier-2 ~60%, median latency 670ms (26-char output for "Correct"; ~280 chars for "Incorrect"). Beats prior Qwen3-14B + router-v1 deployment (91.93% / 11s p50) on both accuracy and latency.
>
> **Prompt = router_v1.** The earlier qwen_v2 router was CV-tuned for the Qwen3-14B base in /no_think mode; on rl-sft (terse-correct natively from SFT) it over-prescribes. router_v1 adds **+5.5pp tier-2 / +8.9pp Complex** at 4σ across 3 runs, with tier-1 unchanged. Validation in `student_sft.md` "Direction shift".
>
> **Launch:** `sbatch --export=ALL,MODEL_PATH=runs/sft_v6_qwen3_14b_20260420/eval/merged_tmp_4659,LORA_PATH=.art/grpo-sft4659-20260426/models/qwen3-14b-sft4659/checkpoints/4952,JUDGE_ROUTER=router,JUDGE_NO_THINK=1 scripts/serving/judge_service.sbatch`. `LORA_PATH` is required — script refuses to launch without it (no silent fallback to base-only).
>
> Historical context (Phase 0–2 prompt tuning, ROUTER_MAP_v2, etc.) below; left intact for traceability but no longer the deployed path.

## Where the judge stands today

| Judge | Acc vs manual oracle (live, 2026-05-02) | Held-out CV est. | Notes |
|---|---:|---:|---|
| **rl-sft-4952 (DEPLOYED)** + router_v1 + /no_think | tier-1 91.5% ± 0.5% (3 runs) / tier-2 60.4% ± 1.4% | – | LoRA on SFT v6 ckpt-4659 merged. p50 670ms. router_v1 chosen over qwen_v2 after 4σ tier-2 win. |
| Gemini-2.0-flash + EVAL_SYSTEM_PROMPT (incumbent) | 86.67% | – | API. p50 ~0.5s. Used at training-data-gen time, not serving. |
| Qwen3-14B fp8 + `router` (v1), thinking on (retired) | 93.33% | 92.4% ± 1.8% | Was deployed thru 2026-05-02. p50 21s, p99 51s due to 6.2% truncation. |
| Qwen3-14B fp8 + `router` (v1), /no_think + max_tokens=512 (retired) | 91.93% | – | Phase-0 fix. p50 11s, no truncation. -1.4pp vs thinking-on overall (Vague +10.5, Chaos -15.2). |
| **Qwen3-14B fp8 + `ROUTER_MAP_QWEN_V2`**, /no_think | – | see Phase-2 table | Phase 2 v2 re-tune; ship-gate per-cat below. |
| **Gemini-2.0-flash + `ROUTER_MAP_GEMINI_V2`**, temp=0 | – | see Phase-2 table | Per-cat divergent from Qwen v2. |
| Qwen3-8B fp8, router | 88.07% | – | Faster, real accuracy hit. |
| gpt-oss-20B MXFP4, router, high | 86.67% | – | Slower + less accurate. See [`gpt_oss_20b.md`](gpt_oss_20b.md). |

**Canonical truth:** `runs/judge_baseline_20260430/eval/manual_verdicts.jsonl` (285 hand-labeled trajectories, 185 Correct / 100 Incorrect after the 2026-05-01 relabel). For v2, a 30% stratified holdout is locked at `data/judge/v2_20260502/holdout_sids.json` and never used during tuning.

## Where the historical 95.44% came from

`docs/judge/prompt_tuning.md` summarized the 2026-04-30 sweep where one `router` run scored 178+94 = 272/285 = **95.44%** — that arithmetic is reproducible from `runs/judge_prompt_tune_20260430/results/summary.csv`. The corresponding `router.jsonl` was overwritten by a later run (now 88.07%) so the original raw outputs are lost. Three later router runs (cudagraphs, specdec, today's live) all score in a tight 93.3–93.7% band on the same labels — there is a real ~2pp drift toward over-strictness between the 95.44% run and current production (FN went from 6 to 14, FP shrank from 7 to 5).

## v2 ship-gate per-category accuracy (locked holdout, n=110)

| Cat | Qwen v2 | Gemini v2 |
|---|---:|---:|
| Schedule | 100% | 100% |
| Modifier | 100% | 100% |
| RelTime | 100% | 100% |
| IR | 94% | 94% |
| Chaos | **83% FAIL** | 94% |
| Vague | 100% | 93% |
| Complex | **84% FAIL** | **79% FAIL** |

Complex fails on both judges — this is real category difficulty, not a tuning miss. Chaos fails on Qwen only (Gemini's lenient `fewshot_v4_dayfocus+L` clears it). For the v2 dataset, the silver-label gate (P(both wrong | agree) ≤ 5%) blocks Complex (14.29%) and Vague (14.29%); other cats keep silver privileges.

## Map

| Topic | Doc |
|---|---|
| Strategy + phases (replaces old `local_judge.md`) | [`plan.md`](plan.md) |
| Manual oracle + baseline judge agreement | [`baseline.md`](baseline.md) |
| Prompt-tuning experiments (19 variants) | [`prompt_tuning.md`](prompt_tuning.md) |
| vLLM optimization sweep + Gemini latency comparison | [`latency.md`](latency.md) |
| gpt-oss-20B as judge (negative result, 2026-05-01) | [`gpt_oss_20b.md`](gpt_oss_20b.md) |
| How RL calls the judge | [`rl_integration.md`](rl_integration.md) |

## Per-category accuracy on the LIVE production judge, post-Phase-0 fix (n=285)

After enabling `/no_think` and capping `max_tokens=512` (truncation noise eliminated):

| Category | Acc | Right/Total | Notes |
|---|---:|---|---|
| Modifier | 97.56% | 40/41 | strong |
| IR | 95.24% | 40/42 | strong |
| Schedule | 94.74% | 36/38 | strong |
| RelTime | 94.74% | 36/38 | regressed -5pp from thinking-on (needs reasoning) |
| Vague | 92.11% | 35/38 | improved +10.5pp by removing truncation noise |
| Complex | 88.10% | 37/42 | **multi-step partial execution** is the dominant failure |
| Chaos | 82.61% | 38/46 | regressed -15pp from thinking-on (fragmentary queries need reasoning) |
| **Overall** | **91.93%** | 262/285 | -1.4pp net vs thinking-on, but no truncation cliff. |

For v2 ship-gate (locked holdout, n=110) per-judge per-cat numbers, see top-of-file table.

Detailed failure-mode analysis in [`prompt_tuning.md`](prompt_tuning.md#failure-modes).

## Open questions

- **Should we train?** Yes — distill `ROUTER_MAP_QWEN_V2` and `ROUTER_MAP_GEMINI_V2` into a small student. Use `data/judge/v2_20260502/train.jsonl` (744 records, 50/50 gold/silver, P(both wrong) gate enforced).
- **Phase 1 (7B SFT on 86.7% Gemini labels)**: *superseded*. Re-do on `data/judge/v2_20260502/`.
- **Complex stays at ~80% on either judge** in CV and on holdout. No prompt variant we tested clears 90% on Complex. Distillation will inherit that ceiling unless we change the supervision signal (per-step trace labels rather than per-trajectory binary).
- **RL integration**: already in production via `JUDGE_BACKEND=local` — see [`rl_integration.md`](rl_integration.md).
- **Service health & crash diagnosis**: watchdog + postmortem layout, what each field means, and the 2026-05-04 silent-death incident — see [`rl_integration.md#service-health--crash-diagnosis`](rl_integration.md#service-health--crash-diagnosis).

## Artifacts

- Run dirs: `runs/judge_baseline_20260430/` (oracle + 4 baselines), `runs/judge_prompt_tune_20260430/` (19 variants + latency)
- Memory: `memory/reference_judge_manual_truth.md`
- Harness: `scripts/eval/judge_prompt_tune.py`, `scripts/eval/judge_latency_bench.py`
- Servers: `scripts/eval/judge_prompt_serve*.sbatch`, `scripts/eval/judge_serve_specdec.sbatch`, `scripts/eval/judge_serve_awq.sbatch`
