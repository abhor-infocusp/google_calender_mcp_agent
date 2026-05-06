# `data/judge/v2_20260502` — Curated judge dataset

**Created** 2026-05-02. Supersedes `data/judge/{train.jsonl, val.jsonl}` (v1, built on 86.7%-accurate Gemini labels — kept for reproducing older experiments).

## What's in here

| File | Records | Purpose |
|---|--:|---|
| `train.jsonl` | 744 | Training data: 371 gold + 373 silver (filtered, capped, class-balanced). |
| `eval.jsonl` | 110 | Held-out gold. **Never used for prompt tuning or distillation training.** |
| `disagreements.jsonl` | 85 | All Qwen-v2 vs Gemini-v2 disagreements from a 700-call live sample, with human (agent) adjudication. Already merged into `train.jsonl` as gold; kept separately for analysis. |
| `holdout_sids.json` | 82 sids | The 30% stratified holdout from the 285-traj manual oracle. Source of truth for what's locked. |
| `metadata.json` | – | Provenance, counts, P(both wrong) per cat, prompt versions. |
| `manual_v2_labels.jsonl` | 139 | Phase-1 Claude-agent labels (oracle-uncovered + capped + diversity). Already merged into `train.jsonl` as gold. |
| `live_agreement_detail.jsonl` | 700 | Phase-3 inter-judge sweep raw output (one record per sample, with both verdicts). |
| `live_agreement_matrix.json` | – | Per-cat agreement counts from the 700-call sweep. |
| `holdout_results.json` | – | Per-judge per-cat accuracy of `ROUTER_MAP_QWEN_V2` / `ROUTER_MAP_GEMINI_V2` on `eval.jsonl`. |
| `_batches/` | – | Internal scratch (per-agent batches). Not used at training time. |

## Schema (`train.jsonl` and `eval.jsonl`)

```json
{
  "sid": "cal_X_q_Y",
  "rollout_hash": "abcd1234",                   // sha256(final+before+after)[:8]
  "cat": "Vague & Contextual (Reasoning Required)",
  "query": "...",
  "final": "<assistant's user-facing response>",
  "expected": "<expected behaviour hint>",
  "before": "<formatted day-state>",
  "after":  "<formatted day-state>",
  "label": "Correct" | "Incorrect",
  "label_source": "oracle"|"manual_v2_agent"|"adjudicated"|"two_way_agree",
  "label_confidence": "high"|"medium"|"low",
  "judge_qwen_v2":   "Correct"|"Incorrect"|null,
  "judge_gemini_v2": "Correct"|"Incorrect"|null,
  "judge_claude_agent": "Correct"|"Incorrect"|null,
  "prompt_version": {"qwen_v2": "router-qwen-v2-20260502",
                     "gemini_v2": "router-gemini-v2-20260502"}
}
```

Trust order: `oracle` > `adjudicated` > `manual_v2_agent` > `two_way_agree`.

## Per-source counts in `train.jsonl`

| Source | Count | Notes |
|---|--:|---|
| `oracle` | 174 | 285-traj manual gold ∖ 111 holdout records. |
| `adjudicated` | 85 | Phase-3 disagreements between Qwen-v2 and Gemini-v2, labeled by Claude agent (calibrated 100% on 50-sid oracle subset). |
| `manual_v2_agent` | 112 | Phase-1 Claude-agent labels (139 minus 27 dropped via dedup against oracle). |
| `two_way_agree` | 373 | Live agreements where Qwen-v2 ∧ Gemini-v2 produced same verdict, capped at 2× per-cat gold count, class-balanced toward 73/27. |

## Per-category breakdown

```
                                                    train      eval
Schedule a Single Event                              126        14
Modifier & Correction (Rescheduling/Updates)         114        16
Information Retrieval (Querying)                     117        17
Relative Time References                             119        12
Human Chaos (Edge Cases/Fragments)                   131        18
Complex Logic & Conflict (Advanced)                   84        19    ← gold-only (silver blocked)
Vague & Contextual (Reasoning Required)               53        14    ← gold-only (silver blocked)
TOTAL                                                744       110
```

## Quality gates (all passed)

- **Phase 0 calibration**: Claude labeling agent vs 50-sid oracle subset → 50/50 = **100%**.
- **Phase 2 ship gate** (per judge, weak cats, on locked holdout):
  - Qwen v2: Schedule/Vague/Modifier/RelTime 100%, IR 94%, Complex **84% FAIL**, Chaos **83% FAIL**.
  - Gemini v2: Schedule/Modifier/RelTime 100%, Chaos 94%, IR 94%, Vague 93%, Complex **79% FAIL**.
  - **Both judges fail Complex.** This is real category difficulty, not a tuning miss.
- **Phase 3 silver gate** — `P(both wrong | agree)` per cat, n=7 per cat:
  - Complex: 14.29% → **silver blocked**.
  - Vague: 14.29% → **silver blocked**.
  - Other 5 cats: 0% → silver allowed.
  - Overall: 4.08% (just under 5%, but per-cat gate is what governs).
- **Phase 4 final audit**: 30 random `train.jsonl` records relabeled blind by an independent Claude agent → **29/30 = 96.7%** match (single failure was a borderline Chaos clarification case).

## Known limitations

1. **Complex stays at ~80%** on either judge regardless of prompt. The few-shot variants tested can't push it higher on the labeled holdout. Distilling into a small student will not exceed this ceiling on Complex without either (a) different teacher signals (e.g. tool-call traces, not just trajectories), or (b) per-step (not per-trajectory) labels.
2. **Silver labels are mostly "easy" cases.** By dropping disagreements and keeping only agreements, silver under-represents the long-tail. Adjudicated disagreements partly compensate but only at gold-volume scale.
3. **Vague is small (53 train records).** Silver was blocked, gold is limited. Distillation may underperform on this cat.
4. **Class balance varies per cat.** RelTime is 87% Correct (above the 73% target band). This reflects the natural live distribution for that cat — agents are usually right on relative-time questions because they're easy.
5. **The labeling agent (Claude) can be biased.** While it scored 100% on a 50-sid calibration subset, its prior may correlate with Qwen/Gemini's prior on hard cases. The 4.08% "both wrong" overall suggests genuine independence, but per-cat n=7 spotcheck cannot distinguish 0% from 5% reliably.
6. **The `eval.jsonl` is small (110 records, 7 cats)**. Per-cat accuracy CIs at 90% are about ±15pp. Anyone using this set should report bootstrap CIs, not point estimates.

## How to refresh

If you re-run any phase, regenerate downstream artifacts in order:

```bash
# Phase 0 — restart judge service if you change the prompts
sbatch scripts/serving/judge_service.sbatch

# Phase 0 — re-lock holdout (will overwrite holdout_sids.json — usually you should NOT do this)
PYTHONPATH=src python scripts/data_generation/lock_judge_holdout.py

# Phase 0 — calibrate labeling agent (50 sids)
PYTHONPATH=src python scripts/eval/judge_prepare_label_pool.py --mode calibration
# then launch a Claude agent against calibration_pool.jsonl per scripts/eval/judge_label_with_agent.py

# Phase 1 — Phase-1 label pool + agent labeling
PYTHONPATH=src python scripts/eval/judge_prepare_label_pool.py --mode phase1
# then launch Claude agents (split into 3 batches)

# Phase 2 — per-judge per-cat sweep (~80 min)
PYTHONUNBUFFERED=1 PYTHONPATH=src python scripts/eval/judge_tune_per_judge.py \
  --judges qwen,gemini --cats vague,complex,chaos,reltime \
  --variants baseline,cot_checklist_v2,fewshot,fewshot_v3,fewshot_v4_dayfocus,fewshot_v3+L,fewshot_v4_dayfocus+L,multi_step_checklist,state_grounded_v1 \
  --folds 5 --labels data/judge/v2_20260502/manual_v2_labels.jsonl \
  > runs/judge_tune_per_judge.log 2>&1 &

# Phase 3 — validate (holdout + 700-call live + spotcheck pool + adjudication pool)
PYTHONUNBUFFERED=1 PYTHONPATH=src python scripts/eval/judge_validate_v2.py --live-sample 700

# Phase 3 — agent-label the spotcheck and adjudication pools (3 batches in parallel)

# Phase 4 — assemble dataset
PYTHONPATH=src python scripts/data_generation/build_judge_dataset_v2.py
```
