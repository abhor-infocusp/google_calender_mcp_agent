# Judge Prompt Tuning — From 86.67% to 95.44%

> **Historical (2026-04-30).** This is the original sweep that produced the
> v1 router. The 95.44% headline was real on the day it was measured; later
> router runs settled at 93.3-93.7% and the live deployment is **91.93%** with
> `/no_think` (see [`README.md`](README.md)). The v2 work (2026-05-02) re-tuned
> per-judge per-cat — see [`../../data/judge/v2_20260502/README.md`](../../data/judge/v2_20260502/README.md)
> and `runs/judge_tune_per_judge_20260502_1628/`.

> Iteration log of pushing Qwen3-14B fp8 from baseline matched-Gemini accuracy
> to **+8.8 pp ahead of Gemini** through prompt engineering alone. No model
> training. Source: `runs/judge_prompt_tune_20260430/`.

## Headline

| Variant | Acc | Per-category strength |
|---|---:|---|
| Gemini-2.0-flash | 86.67% | RelTime 94.7, IR/Complex 90.5 |
| 14B fp8 baseline (existing EVAL_SYSTEM_PROMPT) | 86.67% | matches Gemini |
| 14B fp8 + `fewshot_v3` (unified prompt, 13 worked examples) | 92.98% | Chaos 95.7, IR/Mod 97.6, Vague 94.7 |
| **14B fp8 + `router` (per-category dispatch)** ★ | **95.44%** | Chaos 95.7, IR 97.6, Mod 97.6, Schedule 97.4, Vague 94.7, RelTime 94.7, Complex 90.5 |

★ Production-ready judge. See [`README.md`](README.md) for top-line numbers.

## Three highest-impact wins (cumulative)

1. **fp8 vs 4-bit nf4 quantization** at model load: silent **+7.7 pp**
   (the original 14B-base eval used 4-bit and reported 79% — fp8 is 86.67%
   without changing the prompt at all). See [`baseline.md`](baseline.md).
2. **Few-shot with worked examples**: +6.31 pp (86.67 → 92.98%). Concrete
   Correct/Incorrect examples anchor the model far better than abstract rules.
3. **Per-category routing** to the best whole-prompt for each category:
   +0.70 pp (92.98 → 93.68% with v1 labels, 95.44% with v2 labels).

## Variant ranking

19 prompt variants tested. Full data: `runs/judge_prompt_tune_20260430/results/summary.csv`.

| Variant | Acc | Notes |
|---|---:|---|
| **router** | **93.68% / 95.44%\*** | per-category dispatch. \*after 4 label flips on 2026-05-01 |
| fewshot_v3 | 92.98% | unified prompt + 13 examples (Schedule, Modifier, IR, Vague, Chaos, Complex, RelTime, info-retrieval clarifier) |
| fewshot_v4_dayfocus | 92.63% | v4 examples + state filtered to query-relevant days |
| fewshot_v3_dayfocus | 91.93% | dayfocus + v3 examples — slight regression |
| fewshot_v2 | 91.23% | + 4 Chaos-specific examples on top of fewshot |
| fewshot_self_consistency | 91.23% | 5×@T=0.7 vote — no improvement, 2.5× slower |
| fewshot_v4 | 90.88% | added 4 stricter examples — over-corrected |
| fewshot | 90.18% | first few-shot variant, 7 worked examples |
| cot_checklist_v2 | 88.77% | unified rules + leniency, no examples |
| router (8B) | 88.07% | same prompt, 8B model — 7.4pp drop |
| fewshot_no_expected | 87.72% | hide ground-truth `expected` hint — costs 3.5pp |
| cot_checklist | 87.37% | unified checklist, no leniency |
| per_category | 87.37% | first per-cat attempt — short prompts, lost rules |
| baseline | 86.67% | EVAL_SYSTEM_PROMPT (matches existing rl_train.py) |
| cot_checklist_v3 | 86.67% | strict rules without examples — over-strict |
| per_category_v3 | 86.32% | strict rules + per-cat hint — Chaos cratered |
| fewshot_per_category | 85.96% | examples + per-cat hint — examples got drowned |
| diff_plus_full | 85.61% | calendar diff + full state — diff confused model |
| cat_bespoke | 84.91% | full bespoke per-cat prompts — too strict everywhere |
| diff | 83.86% | calendar diff alone — lost context for IR queries |
| cot_checklist_think | 82.11% | enabled Qwen3 `/think` — judge over-reasoned |
| router_verdict_only | 71.93% | no CoT, just verdict — model can't skip reasoning |
| router (max_tokens=50) | 63.51% | output truncated mid-CoT |

## What the router actually routes

`scripts/eval/judge_prompt_tune.py:build_router` dispatches each trajectory
to the prompt that scored best on its category in prior runs:

| Category | Builder | Rationale |
|---|---|---|
| Complex | `fewshot` (7 examples) | v3's extra examples distract on multi-step |
| Chaos | `fewshot_v3` (13 examples) | needs the v3 vague/IR examples |
| IR | `fewshot_v3` | v3 example 12 ("event in BEFORE = not hallucination") matters |
| Modifier | `fewshot_v3` | diff-as-truth examples are dispositive |
| RelTime | `fewshot` | v3's denial example causes spurious flips |
| Schedule | `fewshot_v3` | strongest baseline |
| Vague | `fewshot_v4_dayfocus` | state-filter to relevant days helps |

## What hurt

| Bad idea | Δ vs winner | Why |
|---|---:|---|
| Enable Qwen3 `/think` for the judge | −10.9 pp | Judge over-reasons itself into wrong territory. Schedule fell to 65.8%. |
| Calendar diff alone (no full state) | −9.1 pp | Loses context for IR/Vague queries where the answer comes from the state itself. |
| Per-category prompts replacing the unified rules | −5.6 pp | Stripped out useful general rules; left the model under-instructed on hallucination/leniency. |
| v3 strict-rule rewrite (no examples) | −6.3 pp | Concrete worked examples >> abstract rules. |
| Self-consistency 5×@T=0.7 vote | −1.75 pp | No precision gain, 2.5× slower. |
| Verdict-only output (no CoT) | −22.5 pp | Model needs CoT to reason through each case. |

## Failure modes (router's remaining 13 errors with v2 labels)

The 4.56% error rate clusters into 4 patterns:

### A. Multi-step partial execution (Complex, Modifier)

User requests N actions; agent claims all N done; diff shows only K<N executed.
Most dangerous failure mode — RL would learn "claiming success is enough".

Example: `cal_3_q_9` — user says "decline movie night AND move weekly meeting
to that slot". Diff shows the meeting moved but movie night still in
BEFORE/AFTER unchanged. Half-done.

### B. Duplicate-instead-of-update (Complex, Human Chaos)

User wants to *modify* an existing event; agent *creates a new* event at
adjacent times. Original unchanged.

Example: `cal_47_q_9` — "Add 30 minutes to Ventilation Problem" (existing
8:00-10:00). Diff: `+ 10:00-10:30 Ventilation Problem` (new). Original
8-10 unchanged.

### C. Self-conflict / event-identity confusion (Complex)

Conflict-check queries where the agent confuses the queried event with
itself.

Example: `cal_26_q_9` — "If I move basketball pickup to 7 PM, what conflicts?"
The event itself is "Pick up son from basketball" 6:30-7:30. Agent reports
that as the conflict.

### D. Subjective interpretation (Complex, Vague)

Genuinely ambiguous cases where both verdicts are defensible. Example: "Am I
working during dinner with my family?" — does Study Group count as "work"?

### Root causes from reading judge reasoning

Reading the actual judge reasoning on these failures shows three
underlying weaknesses:

1. **Confabulation of state changes.** The judge invents diff lines that
   would satisfy the agent's narrative. (See `cal_47_q_9` — the judge wrote
   "Service Call - Ventilation Problem was extended" when the diff shows no
   such modification.)
2. **Skim-reading of agent response.** The judge assigns the *expected*
   interpretation of the response rather than what the agent literally said.
   (See `cal_26_q_9` — the judge wrote "no conflicts" when the agent
   actually claimed a self-conflict.)
3. **Failed lookups in BEFORE.** The judge declares hallucination on events
   that exist in the BEFORE state. (See `cal_19_q_8` — "Study Group" is in
   BEFORE on Tuesday but the judge said "not in calendar".)

All three are forms of "pattern-match a plausible scenario instead of
audit the diff token-by-token". Hard to fix in prompting; cleaner via
training.

## Verifier experiments (negative result)

Tried a stage-2 verifier pass: feed stage-1's reasoning + verdict to a
second judge call asking "is this verdict right? KEEP or FLIP?".

| Verifier | Flips | Helped | Hurt | Final acc |
|---|---:|---:|---:|---:|
| (none — router alone) | — | — | — | 95.44% |
| Default verifier | 104 | 11 | 93 | 66.32% |
| Conservative ("default keep, only flip on hard violations") | 36 | 4 | 32 | 85.26% |

The 14B has *some* signal (catches 78% of stage-1 errors) but **terrible
precision** (~11%). Every error it fixes, it breaks 8. Treats "audit this"
as license to find fault.

Possible workable variant (untested): triage filter — only run the verifier
on cases matching specific suspicious patterns (e.g. "stage-1 said Correct
but diff is empty on action query"). Surgical use of the signal.

## Latency findings (preview, see `latency.md`)

- **Router at concurrency=16 (RL-relevant): 0.83 s/call.** ~2× slower than Gemini.
- **Router at concurrency=1: 9.77 s/call.** ~21× slower than Gemini.
- Cudagraphs/AWQ on Blackwell+fp8 give ~zero speedup (decode is memory-bound).
- Speculative decoding hurts at concurrency≥4.
- Verdict-only output without training fails (router_verdict_only: 71.93%).

The path to fast + accurate is training, not more prompt engineering.

## Files in `runs/judge_prompt_tune_20260430/`

- `results/summary.csv` — every run's overall + per-category accuracy + latency
- `results/<variant>.jsonl` — per-trajectory predictions for each variant
- `results/latency_bench.json` — separate latency-only benchmark
- `logs/serve_*.log` — vLLM server logs across all variants
- `SUMMARY.md` — run-dir summary (kept for traceability; *this* doc is canonical)
