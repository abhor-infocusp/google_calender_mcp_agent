# gpt-oss-20B as judge — comprehensive report

> **Historical (2026-05-01).** Comparison was done against the v1 router (95.44%
> headline at the time). For the current Qwen3-14B accuracy in production see
> [`README.md`](README.md). Conclusion (gpt-oss-20B is slower + less accurate)
> stands; the absolute Qwen3 number it was compared against has been corrected.

> Run date: 2026-05-01. Compares OpenAI's `openai/gpt-oss-20b` (MXFP4, ~13 GiB)
> against the production Qwen3-14B fp8 judge on the canonical 285-trajectory
> manual oracle (`runs/judge_baseline_20260430/eval/manual_verdicts.jsonl`).

## TL;DR

**gpt-oss-20B is strictly worse than Qwen3-14B on this judge task** —
~9 pp lower accuracy *and* ~6.5× slower per call. Latency loss is structural
(hidden chain-of-thought from the harmony reasoning model), not prompt-fixable.

| Judge                              | Acc vs manual | Per-call p50 (serial) | Per-call p50 (c=16) | Notes |
|------------------------------------|--------------:|----------------------:|--------------------:|-------|
| Qwen3-14B fp8, **router** ★         | **95.44%**    | **0.83 s**            | 0.83 s              | current production |
| gpt-oss-20B MXFP4, router, high     | 86.67%        | n/a                   | 8.31 s              | matches Gemini      |
| gpt-oss-20B MXFP4, router, medium   | 85.96%        | n/a                   | 8.10 s              |                     |
| gpt-oss-20B MXFP4, router, low      | 83.51%        | 5.40 s                | 8.01 s              | over-conservative   |
| gpt-oss-20B MXFP4, **short** prompt | (not tested for accuracy) | 5.75 s | n/a            | prompt size doesn't help |
| Gemini-2.0-flash (incumbent)        | 86.67%        | 0.46 s                | 0.46 s              | API call            |

★ ship-gate baseline.

## Setup

- vLLM 0.19.0, MXFP4 auto-detected (`MARLIN` Mxfp4 MoE backend), `enforce_eager=True`
- 1× MIG 1g.24gb slice (slice 1; slice 0 is the production judge service)
- `max_model_len=8192`, `max_num_seqs=32`, port 8020
- Reasoning level controlled via `chat_template_kwargs={"reasoning_effort": <low|medium|high>}`
- Same router prompt as Qwen3-14B (`scripts/eval/judge_prompt_tune.py:build_router`)
- Sbatch: [`scripts/eval/judge_serve_gpt_oss_20b.sbatch`](../../scripts/eval/judge_serve_gpt_oss_20b.sbatch)
- Runner: [`scripts/eval/judge_eval_gpt_oss.py`](../../scripts/eval/judge_eval_gpt_oss.py)
- Latency bench: [`scripts/eval/judge_latency_bench_gptoss.py`](../../scripts/eval/judge_latency_bench_gptoss.py)

## Per-category accuracy (n=285)

| Category  | low    | medium | high   | Qwen3-14B router | Δ (high−Qwen) |
|-----------|-------:|-------:|-------:|-----------------:|--------------:|
| Chaos     | 89.13% | 91.30% | 93.48% | 95.65%           | −2.2 pp        |
| IR        | 92.86% | 88.10% | 92.86% | 97.62%           | −4.8 pp        |
| Complex   | 85.71% | 83.33% | 85.71% | 90.48%           | −4.8 pp        |
| Vague     | 68.42% | 68.42% | 68.42% | 94.74%           | **−26.3 pp**   |
| Schedule  | 71.05% | 84.21% | 84.21% | 97.37%           | −13.2 pp       |
| Modifier  | 85.37% | 90.24% | 90.24% | 97.56%           | −7.3 pp        |
| RelTime   | 89.47% | 94.74% | 89.47% | 94.74%           | 0 pp           |

**Vague is the killer**: 68% across all reasoning levels — gpt-oss never improves
on this category, suggesting the failure isn't compute-budget, it's prompt
calibration. `Vague` cases require interpreting "agent should have asked for
clarification" rules; gpt-oss interprets them strictly.

## Confusion matrices

| Effort | C→C | C→I (false-Inc) | I→C (false-Cor) | I→I |
|--------|----:|---------------:|----------------:|----:|
| low    | 140 | 45             | 2               | 98  |
| medium | 148 | 37             | 3               | 97  |
| high   | 150 | 35             | 3               | 97  |

The model is **systematically over-conservative** — almost no false positives,
many false negatives. Even at high effort it under-credits 35/185 trajectories
the manual oracle marks Correct.

## Latency (serial, n=30, warm server)

```
config                                  n   min   p50  mean   p90   p99   max
gptoss20b_router_low                   30  2.07  5.40  7.26 16.11 21.50 21.50
gptoss20b_short_low                    30  2.69  5.75  5.58  9.86 10.76 10.76
qwen3_14b_router  (from latency.md)    —   —    0.83   —     —     —     —
```

The short-prompt p50 (5.75 s) is essentially the same as the router p50 (5.40 s).
**This rules out prefill as the bottleneck** — the 4 k-token router prompt
adds <1 s. The 5–8 s wall is the model generating hidden CoT in the analysis
channel even when `reasoning_effort="low"` and the visible `content` is just
one word ("Correct"/"Incorrect"). The vLLM `openai_gptoss` reasoning parser
strips analysis channel from the response payload, so we pay for tokens we
never see.

## Disagreement analysis vs Qwen3-14B router

Where the two judges differ on the same 285 trajectories (using gpt-oss `high`):

|                          | count |
|--------------------------|------:|
| both right               |   222 |
| Qwen right, gpt-oss wrong |   29 |
| gpt-oss right, Qwen wrong |   25 |
| both wrong               |     9 |

Even though gpt-oss is worse overall, it's right on 25 cases Qwen misses —
suggesting an ensemble (Qwen + gpt-oss agree → Correct) could push above
95.4%. Not pursued here because the latency cost is prohibitive.

## Why "much faster" didn't pan out

Original hypothesis: 20B MoE with ~3.6 B active params on MXFP4 should beat
14B dense fp8 on tokens/sec. Reality:

1. **Reasoning-model overhead.** gpt-oss is post-trained on harmony format with
   long analysis traces. Even `reasoning_effort="low"` keeps generating ~hundreds
   of analysis tokens per call, which vLLM hides from `content` but still
   decodes.
2. **`enforce_eager=True`.** Required on Blackwell to avoid the cudagraph
   memory blow-up that bit us on Qwen — but gpt-oss is more sensitive (decode
   throughput drops further without graph capture). MARLIN MoE backend can't
   use cudagraphs anyway in this vLLM build.
3. **MIG slice contention.** A 1g.24gb slice has limited SM throughput; a 20B
   MoE pays a worse latency floor than 14B dense at low concurrency.

p50 latency does *not* meaningfully change between `low` and `high` effort
(8.0 → 8.3 s wall at c=16), reinforcing that the cost is intrinsic to the
model and serving config rather than the reasoning budget.

## Verdict

**Do not adopt gpt-oss-20B as the judge.** It loses to the existing Qwen3-14B
router on every dimension we care about:

- 95.44% → 86.67% accuracy (−9 pp)
- 0.83 s → 8.31 s p50 latency (10× worse at c=16, 6.5× serial)
- Same VRAM footprint, same MIG slice cost
- No improvement on the highest-risk category (Complex 85.7% vs 90.5%)
- Catastrophic loss on Vague (−26 pp) that doesn't respond to more reasoning

If we want a faster judge, the path forward is **distilling the router into
a small verdict-only model** (already in `docs/judge/plan.md` as the trained-
judge phase), not swapping the base model.

## Follow-ups worth ~30 minutes each

1. **Ensemble check** — combine Qwen3-14B router + gpt-oss-20B router with
   AND-on-Correct. Cheap accuracy upper bound; won't ship (latency) but tells
   us whether the trained judge target should be ≥97%.
2. **gpt-oss with no system prompt + only diff** — the router prompt is
   5 distinct fewshot bundles tuned for Qwen. Might recover some Vague accuracy
   if gpt-oss prefers a stricter, Qwen-agnostic spec. Low expected value, but
   trivial to try.
3. **gpt-oss-120B** — if a future ART/vLLM bump enables tensor-parallel across
   2 MIG slices, the 120B variant might cross the Qwen accuracy line. Same
   latency story would apply.

## Artifacts

- `runs/judge_prompt_tune_20260430/results/gptoss20b_router_low.jsonl`
- `runs/judge_prompt_tune_20260430/results/gptoss20b_router_medium.jsonl`
- `runs/judge_prompt_tune_20260430/results/gptoss20b_router_high.jsonl`
- `runs/judge_prompt_tune_20260430/results/gptoss20b_router_low_dryrun.jsonl`
- `runs/judge_prompt_tune_20260430/results/latency_bench_gptoss.json`
- Serve log: `runs/judge_prompt_tune_20260430/logs/serve_gptoss_76.log`
- Server still running on `azkaban:8020` as Slurm job 76 (kill with
  `scancel 76` when done).
