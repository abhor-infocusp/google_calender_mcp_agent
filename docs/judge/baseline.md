# Judge Baseline — Manual Oracle + Cold-Model Judges

> **Historical (2026-04-30).** Numbers below describe the baseline state *before*
> prompt tuning and the v2 re-tune. For current production accuracy see
> [`README.md`](README.md); for v2 dataset + per-judge per-cat maps see
> [`../../data/judge/v2_20260502/README.md`](../../data/judge/v2_20260502/README.md).

> Baseline measurement of how well **untrained, off-the-shelf** models judge
> calendar-agent trajectories, vs Gemini-2.0-flash and a hand-labeled oracle.
> Source: `runs/judge_baseline_20260430/`.

## The manual oracle

`runs/judge_baseline_20260430/eval/manual_verdicts.jsonl` is the canonical
truth used to evaluate every judge from this point forward.

- 285 trajectories sampled from
  `runs/rl_adaptive_qwen3_14b_20260424/.art/.../trajectories/train/*.parquet`
- One trajectory dropped (no scenario record) → 285 scored
- Hand-labeled by Claude (the present author), 2026-04-30
- Relabeled 4 cases on 2026-05-01 (cases #8, #39, #65, #280):
  - `manual_verdicts_v1.jsonl` is the original snapshot
  - `manual_verdicts.jsonl` is the canonical truth going forward
- Distribution: **185 Correct / 100 Incorrect** (after relabel)

For ambiguous queries, labeled the interpretation that makes sense for a
calendar MCP agent given the context. Specifically lenient on:
- Cosmetic date naming (e.g. agent says "Saturday April 13" while actual
  Saturday is April 20 — trust the calendar diff)
- Attendee names vs full emails
- Listing extra events alongside the asked-for one for vague queries
- Asking for clarification when the query truly needs it

Strict on:
- Empty diff + claimed success (the 4 relabels in v2)
- Hallucinated events not in BEFORE
- Wrong action (move vs delete vs create)
- Multi-step requests with skipped steps

## Cold-model judges vs manual

| Judge | Acc | Agree/Total |
|---|---:|---:|
| Gemini-2.0-flash (existing gt label) | **86.67%** | 247/285 |
| Qwen3-32B base, 4-bit nf4 | 82.11% | 234/285 |
| Qwen3-14B base, 4-bit nf4 | 78.95% | 225/285 |
| Qwen3-14B base, fp8 (revised baseline 2026-04-30) | **86.67%** | 247/285 |
| Qwen3-8B base, 4-bit nf4 | 71.93% | 205/285 |

Notable: **fp8 quantization vs 4-bit nf4 is a silent +7.7 pp gain** at the
model layer. The "14B base = 79%" number that previously appeared in
`local_judge.md` was the 4-bit measurement. fp8 unlocks the model's real
ability before any prompt engineering.

## Per-category baseline (vs manual)

| Category | Gemini | 32B (nf4) | 14B (nf4) | 8B (nf4) |
|---|---:|---:|---:|---:|
| Complex Logic & Conflict | 90.5% | 76.2% | 73.8% | 73.8% |
| Human Chaos (Fragments) | 76.1% | 80.4% | 82.6% | 50.0% |
| Information Retrieval | 90.5% | 90.5% | 78.6% | 66.7% |
| Modifier & Correction | 85.4% | 90.2% | 75.6% | 68.3% |
| Relative Time References | 94.7% | 94.7% | 92.1% | 92.1% |
| Schedule a Single Event | 78.9% | 60.5% | 73.7% | 76.3% |
| Vague & Contextual | 92.1% | 81.6% | 76.3% | 81.6% |

Surprises:
- **14B base beats Gemini on Human Chaos (82.6% vs 76.1%).** Gemini is too
  lenient when the agent hedges/asks for clarification on fragmentary queries.
- **32B beats Gemini on Modifier (90.2% vs 85.4%).** Catches actual update
  failures Gemini missed.
- **32B is uniquely bad on Schedule (60.5%).** Nitpicks date-naming
  inconsistencies that don't matter for a calendar MCP agent.

## Confusion matrix — error correlation between judges (vs manual, n=285)

| Model | I→C (lets bad pass) | C→I (rejects good) | I→I (catches bad) |
|---|---:|---:|---:|
| 8B  | **54** | 19 | 62 |
| 14B | 11 | 35 | 105 |
| 32B | 15 | 32 | 101 |

8B is a soft pass-through; 14B/32B are stringent and similar. Adding 8B to
an ensemble *hurts* — its 28% error rate dilutes the signal.

## Ensembles tried (all lost to a single 14B fp8)

| Ensemble | Acc |
|---|---:|
| Gemini alone (gt label) | 86.67% |
| majority(Gemini, 32B, 14B) | 85.26% |
| majority(all 4, tiebreak=Correct) | 86.32% |
| Weighted by accuracy | 85.26% |
| Weighted Gemini=2 | 85.61% |

Errors are too correlated (Qwen variants share ~40% of errors). Voting
doesn't beat the strongest single judge.

## Implication for training

The **base 14B fp8 already matches Gemini at 86.67%** with the existing
EVAL_SYSTEM_PROMPT. After prompt engineering it goes to 95.44% — see
[`prompt_tuning.md`](prompt_tuning.md). That's the labeler we use to train
a smaller, faster judge in Phase 1.5 (see [`plan.md`](plan.md)).

## Files in `runs/judge_baseline_20260430/`

- `eval/manual_verdicts.jsonl` — canonical truth (185 C / 100 I)
- `eval/manual_verdicts_v1.jsonl` — pre-2026-05-01 snapshot
- `eval/manual_review_input.jsonl` — compact view of the 285 trajectories
- `eval/art_holdout_qwen3_{8b,14b,32b}_base.json` — per-trajectory predictions
- `eval/manual_judge_comparison.json` — full per-cat agreement numbers
- `eval/SUMMARY.md` — original run-summary (kept for traceability;
  *this* doc is the up-to-date version)
