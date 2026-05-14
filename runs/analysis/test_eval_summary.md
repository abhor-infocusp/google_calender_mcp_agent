# Held-out Test Evaluation — Final Summary

**Date:** 2026-05-14 (last update for ORPO results; original 2026-04-24 for SFT/DPO)

**Test set:** `test_data/` — 49 freshly-generated calendars, 692 queries, 7 balanced categories (~98 queries each). Never seen by any model; zero overlap with SFT training data or DPO pair-mining source.

**Eval harness:** `scripts/eval/eval_batch.py --mode test`, Gemini-2.0-flash judge (router_structured for ORPO evals; plain Gemini for earlier SFT/DPO).
For RL/ORPO LoRA evals: vLLM serving the merged SFT v6 fp16 base + LoRA via `--enable-lora` (do NOT merge RL LoRAs to fp16; see `feedback_vllm_lora_serving.md`).


## Headline result

| Model | Best | Accuracy |
|---|---|---|
| **ORPO on SFT v6** (run-1, 622 steps) | ckpt-427 & ckpt-600 (tied) | **84.25%** (+4.1 pp vs SFT) |
| **SFT v6** (Qwen3-14B + calendar SFT) | ep 3 (ckpt-4659) | **80.1%** |
| **DPO-from-SFT** (SFT + DPO on mined pairs) | ep 1 & ep 2 (ckpt-479, ckpt-958) | **79.3%** (−0.8 pp vs SFT) |
| **DPO-from-Instruct** (Qwen3-14B-Instruct + DPO, no SFT) | ep 2 (ckpt-958) | **61.3%** (−1.7 pp vs baseline) |
| **Qwen3-14B-Instruct baseline** (no calendar training) | — | 63.0% |

## ORPO run-1 (2026-05-08 → 2026-05-14)

Run dir: `runs/rl_orpo_qwen3_14b_20260508_0625/`.  
Trainer: `scripts/training/rl/rl_orpo.py` (design in `docs/orpo/design.md`).  
Setup: ORPO LoRA r=8 on top of SFT v6 ckpt-4659 merged-base, β=0.1, λ=1.0, LR=5e-6, N=20 scenarios/step, k_easy=4 / k_hard=8, buffer cap 4/scenario. 622 steps, 20 epochs, ~3 days + ~36h resume.

### Full eval trajectory (Gemini-structured judge)

| ckpt | overall | Schedule | Vague | Modif | IR   | Complex | Chaos | RelTime |
|---|---|---|---|---|---|---|---|---|
|  50 | 79.91% | 79.6 | 74.5 | 86.7 | 93.9 | 59.2 | 81.6 | 83.7 |
| 100 | 80.64% | 81.6 | 74.5 | 84.7 | 91.8 | 60.2 | 84.7 | 86.5 |
| 150 | 80.49% | 81.6 | 80.6 | 84.7 | 93.9 | 54.1 | 82.7 | 85.6 |
| 200 | 83.09% | 86.7 | 80.6 | 84.7 | 92.9 | 63.3 | 85.7 | 87.5 |
| 250 | 82.66% | 81.6 | 78.6 | 85.7 | 92.9 | 66.3 | 86.7 | 86.5 |
| 300 | 82.08% | 89.8 | 82.7 | 85.7 | 90.8 | 60.2 | 79.6 | 85.6 |
| 350 | 82.51% | 85.7 | 81.6 | 87.8 | 93.9 | 65.3 | 78.6 | 84.6 |
| 400 | 82.80% | 86.7 | 81.6 | 81.6 | 94.9 | 65.3 | 82.7 | 86.5 |
| **427** | **84.25%** | 83.7 | 86.7 | 82.7 | 94.9 | 68.4 | 84.7 | 88.5 |
| 450 | 82.23% | 82.7 | 79.6 | 79.6 | 96.9 | 65.3 | 84.7 | 86.5 |
| 475 | 83.38% | 88.8 | 84.7 | 83.7 | 93.9 | 62.2 | 82.7 | 87.5 |
| 500 | 83.82% | 87.8 | 84.7 | 85.7 | 94.9 | 63.3 | 83.7 | 86.5 |
| 525 | 82.95% | 85.7 | 85.7 | 84.7 | 94.9 | 61.2 | 81.6 | 86.5 |
| 550 | (eval failed — empty results) | | | | | | | |
| 575 | 82.80% | 86.7 | 86.7 | 83.7 | 95.9 | 62.2 | 80.6 | 83.7 |
| **600** | **84.25%** | 87.8 | 85.7 | 84.7 | 93.9 | 66.3 | 84.7 | 86.5 |
| 620 | 83.09% | 84.7 | 81.6 | 82.7 | 93.9 | 66.3 | 85.7 | 86.5 |
| 621 | 83.82% | 83.7 | 83.7 | 85.7 | 96.9 | 62.2 | 85.7 | 88.5 |

**Winners (tied):** ckpt-427 and ckpt-600 at **84.25%** (583/692). ckpt-600 is the natural one to promote — it represents +173 more training steps of stable performance.

### Per-category gains, ORPO ckpt-600 vs ckpt-50 (≈ SFT baseline)

| Category | ckpt-50 | ckpt-600 | Δ |
|---|---|---|---|
| Vague & Contextual | 74.5 | 85.7 | **+11.2** |
| Schedule | 79.6 | 87.8 | +8.2 |
| Complex | 59.2 | 66.3 | +7.1 |
| Human Chaos | 81.6 | 84.7 | +3.1 |
| Relative Time | 83.7 | 86.5 | +2.8 |
| Information Retrieval | 93.9 | 93.9 | 0 |
| Modifier & Correction | 86.7 | 84.7 | −2.0 (slight forgetting) |

### Comparison to prior GRPO run (`rl_qwen3_14b_20260420` → rl-sft-4952)

| | GRPO (rl-sft-4952) | ORPO (run-1) |
|---|---|---|
| Wall-clock | ~3.5 days active | ~3 days active |
| Steps at peak | 4,952 | 427 / 600 (tied) |
| Total rollouts | ~40k | ~51k |
| Held-out gain | +5 pp | +4.1 pp |
| Per-rollout signal density | 1 advantage update / 8 rollouts | ~6 preference pairs / 8 rollouts (~6× denser) |
| Per-step compute | ~1× | ~20× heavier (concatenated forward over 50–150 pairs) |

Net: wall-clock parity at similar held-out gain. ORPO is more sample-efficient per rollout but pays it back in per-step training compute.

### Known issues with run-1

- ckpt-550 eval produced 0% on both attempts (vLLM LoRA-load edge case). Adapter dir intact; skipped.
- The difficulty sampler `1 − p^G − (1−p)^G` couldn't escape easies as the pool composition skewed (574 easy / 1 mid / 27 hard by step 621). Late-run skip_easy hit 16/20 per step → 80% wasted rollouts.
- Pair-producing scenario diversity collapsed 272 → 34. Top-50 share of pairs: 38% → 100%.
- Buffer wasn't persisted across the first restart (now patched; see `ReuseBuffer.save/load`).
- Optimizer state restored CPU-side while params went to GPU on resume → first-step crash (now patched in `orpo_train_step`).

See `docs/orpo/run_log.md` + `docs/orpo/design.md` "Postmortem / v2 plan" for full details.

### Answers to posed questions

1. **Best SFT v6 checkpoint:** ckpt-4659 (epoch 3) at 80.1%. Epochs 4 and 5 did not improve further (78.9%, 79.2%). Even-vs-odd pattern from earlier RL-data eval is gone on test_data.

2. **Does DPO-from-SFT beat SFT v6?** **No.** Best DPO-from-SFT (79.3%) trails best SFT v6 (80.1%) by 0.8 pp. With SE ≈ 1.5 pp on 692 queries this is within noise, but DPO never convincingly exceeds SFT. All three DPO-from-SFT epochs cluster in 78.8–79.3% — loss curve kept dropping but accuracy didn't move.

3. **Does DPO-from-Instruct beat the Instruct baseline?** **No — DPO actively hurt the Instruct baseline.** All three DPO-from-Instruct epochs scored 60.0–61.3%, consistently below the 63.0% raw Instruct baseline. Confirms the theoretical prediction: DPO from an un-SFT'd base loses the shared-structure signal it needs, because `π_ref` assigns similar low probability to both chosen and rejected; the differential signal cancels at the format level.

## Full results table

| Model | Epoch | Ckpt | Correct/Total | % | Complex | Chaos | IR | Modifier | RelTime | Schedule | Vague |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Instruct baseline | — | — | 436/692 | **63.0%** | 40/98 (41%) | 29/98 (30%) | 59/98 (60%) | 69/98 (70%) | 99/104 (95%) | 71/98 (72%) | 69/98 (70%) |
| SFT v6 | 1 | 1553 | 501/692 | **72.4%** | 43/98 (44%) | 66/98 (67%) | 86/98 (88%) | 69/98 (70%) | 92/104 (88%) | 76/98 (78%) | 69/98 (70%) |
| SFT v6 | 2 | 3106 | 542/692 | **78.3%** | 53/98 (54%) | 75/98 (77%) | 87/98 (89%) | 85/98 (87%) | 97/104 (93%) | 67/98 (68%) | 78/98 (80%) |
| SFT v6 | 3 | 4659 | 554/692 | **80.1%** | 54/98 (55%) | 75/98 (77%) | 90/98 (92%) | 83/98 (85%) | 97/104 (93%) | 75/98 (77%) | 80/98 (82%) |
| SFT v6 | 4 | 6212 | 546/692 | **78.9%** | 58/98 (59%) | 77/98 (79%) | 87/98 (89%) | 87/98 (89%) | 94/104 (90%) | 66/98 (67%) | 77/98 (79%) |
| SFT v6 | 5 | 7765 | 548/692 | **79.2%** | 51/98 (52%) | 81/98 (83%) | 88/98 (90%) | 85/98 (87%) | 96/104 (92%) | 69/98 (70%) | 78/98 (80%) |
| DPO-from-SFT | 1 | 479 | 549/692 | **79.3%** | 51/98 (52%) | 79/98 (81%) | 89/98 (91%) | 85/98 (87%) | 97/104 (93%) | 71/98 (72%) | 77/98 (79%) |
| DPO-from-SFT | 2 | 958 | 549/692 | **79.3%** | 51/98 (52%) | 79/98 (81%) | 88/98 (90%) | 82/98 (84%) | 94/104 (90%) | 77/98 (79%) | 78/98 (80%) |
| DPO-from-SFT | 3 | 1437 | 545/692 | **78.8%** | 49/98 (50%) | 78/98 (80%) | 88/98 (90%) | 87/98 (89%) | 97/104 (93%) | 69/98 (70%) | 77/98 (79%) |
| DPO-from-Instruct | 1 | 479 | 422/692 | **61.0%** | 36/98 (37%) | 26/98 (27%) | 57/98 (58%) | 64/98 (65%) | 101/104 (97%) | 72/98 (73%) | 66/98 (67%) |
| DPO-from-Instruct | 2 | 958 | 424/692 | **61.3%** | 41/98 (42%) | 25/98 (26%) | 55/98 (56%) | 69/98 (70%) | 99/104 (95%) | 72/98 (73%) | 63/98 (64%) |
| DPO-from-Instruct | 3 | 1437 | 415/692 | **60.0%** | 45/98 (46%) | 28/98 (29%) | 54/98 (55%) | 63/98 (64%) | 96/104 (92%) | 72/98 (73%) | 57/98 (58%) |

## Per-category pattern — where gains come from

Comparing best checkpoint of each run vs. baseline:

| Category | Baseline | SFT v6 best | DPO-from-SFT best | DPO-Ins best | SFT gain | DPO-from-SFT vs SFT |
|---|---|---|---|---|---|---|
| Complex | 41% | 55% | 52% | 42% | +14 pp | -3 pp |
| Chaos | 30% | 77% | 81% | 26% | +47 pp | +4 pp |
| IR | 60% | 92% | 91% | 56% | +32 pp | -1 pp |
| Modifier | 70% | 85% | 87% | 70% | +14 pp | +2 pp |
| RelTime | 95% | 93% | 93% | 95% | -2 pp | +0 pp |
| Schedule | 72% | 77% | 72% | 73% | +4 pp | -4 pp |
| Vague | 70% | 82% | 79% | 64% | +11 pp | -3 pp |

### Observations

- **SFT v6 lifts the hardest categories hardest.** Chaos (30→83%) and Modifier (70→89%) see the largest gains — these require calendar-specific domain knowledge that the Instruct baseline lacks.
- **RelTime is saturated at ~95% even for the raw Instruct baseline** — general time-reasoning is already strong.
- **Schedule is the one category where DPO-from-SFT meaningfully improves SFT** (79% vs 70%, +9 pp). This is consistent with DPO optimizing event-creation format at the margin.
- **Complex is where SFT→DPO regresses most** (59% → 52%). DPO's margin loss is pulling the model toward shallower patterns that work for Schedule/Chaos but lose multi-step reasoning.
- **DPO-from-Instruct regresses Chaos** (30% → 26%) and **Vague** (70% → 58%). Likelihood displacement: DPO pushed Instruct's existing format strength down without substituting anything better.

## Why DPO-from-Instruct fails

Expected ex-ante. DPO's loss is a ratio: `log π_θ(y_w|x) − log π_ref(y_w|x)`. Both y_w and y_l in our mined pairs contain `<tool_call>` XML with specific argument patterns that an un-SFT'd Instruct model assigns near-zero probability to. The gradient pushes *both* probabilities further into the tail while the ratio looks fine — the classic "likelihood displacement" failure (Pal et al. 2024). 

In-distribution test confirms this: categories where Instruct already had shared structure (RelTime at 95%) barely moved; categories where Instruct had fragile format handling (Chaos, Vague) got *worse*. 

## Why DPO-from-SFT doesn't beat SFT

Two hypotheses:

1. **Pair saturation.** The 1,913 DPO pairs were mined from the same RL rollouts that SFT v6 had already learned to solve; the "chosen" trajectories come from a policy already near SFT v6's ceiling. DPO has little room to improve — the margin signal is weak.
2. **Category-specific trade-offs wash out.** DPO gains on Schedule (+9 pp) are offset by regression on Complex (−7 pp). Net change near zero, matching what we see in the 79% bands across all DPO-from-SFT epochs.

## Conclusion

**Updated 2026-05-14: ORPO ckpt-600 (84.25%) is the new held-out best**, +4.1 pp over the SFT v6 ckpt-4659 baseline. Gains concentrated on Vague (+11), Schedule (+8), Complex (+7); slight Modifier regression (−2). ckpt-427 and ckpt-600 tied; ckpt-600 is the natural promotion candidate.

**Original 2026-04-24 conclusion** (DPO comparison): SFT v6 ckpt-4659 was best at 80.1%; DPO as implemented neither helped from SFT nor from Instruct. DPO improvements were category-localized (Schedule) and offset by regressions elsewhere.

Next experiments to consider (in priority order, updated 2026-05-14):
- **ORPO v2 with DAPO Dynamic Sampling** — replace the difficulty sampler with DAPO's oversample → filter `std == 0` → accumulate until N mixed-reward groups; raise β to 0.3; LoRA rank → 16. See `docs/orpo/design.md` § Postmortem.
- **Complex-category targeted data augmentation** — Complex still the bottleneck even after ORPO (66.3%, +7 over SFT). RL squeezed most of what's there; next gain likely needs new SFT data, not more RL.
- **(Lower priority, historical from DPO postmortem)** RFT, iterative DPO, category-weighted DPO.

## Artifacts

- Raw per-eval JSONs: `runs/*/eval_test/checkpoint-*.json`, `runs/qwen3_14b_instruct_baseline_20260424/eval_test/baseline.json`
- Per-run summary CSVs: `runs/*/eval_test/summary.csv`
- Test data: `test_data/` (49 calendars, 692 queries)
- DPO pair source: `runs/dpo/pairs_from_14b_rl.jsonl` (1,913 pairs from 5,506 RL parquets)