# Local Judge Plan — 2026-05-01

> **Goal:** Replace Gemini-as-judge in RL training with a local judge that is
> (a) more accurate than Gemini, (b) latency-comparable or better, (c) free
> from API quota.

## Current status (updated 2026-05-02)

- **Phase 0 — Baseline measurement.** ✅ Done (2026-04-30).
- **Phase 0.5 — Prompt engineering (router v1).** ✅ Done (2026-04-30 → 2026-05-01).
  - 95.44% number measured on 2026-04-30 18:24. Re-measured against the SAME labels on 2026-05-01: 93.68% (cudagraphs/specdec); on 2026-05-02 live: **93.33%**. Confusion-matrix shape shifted — deployed prompt drifted toward over-strictness. The 95.44% number is real-but-not-current.
- **Phase 0.6 — Truncation fix + per-judge re-tune (v2).** ✅ Done (2026-05-02).
  - Disabled `<think>` on the judge service (`/no_think` + `max_tokens=512`) — eliminates the 6.2% truncation bias and drops p50 latency 21s → 11s. Net accuracy on 285 oracle: 91.93% (-1.4pp vs thinking-on, but no truncation cliff and Vague +10.5pp).
  - Built `ROUTER_MAP_QWEN_V2` and `ROUTER_MAP_GEMINI_V2` via 5-fold stratified CV on (oracle ∖ holdout) ∪ (139 new agent labels). Divergent variants picked to reduce correlated-error risk.
  - Held-out (n=110) ship gate: passes on Modifier/Schedule/IR/RelTime; fails on Complex (~80% both judges) and Chaos-Qwen (83%).
  - See `data/judge/v2_20260502/README.md` for full provenance and quality gates.
- **Phase 1 — SFT distillation from Gemini (7B).** ❌ Superseded.
  - Trained 7B at `runs/judge_v1_qwen3_7b_20260425/` distilled from 86.7%-accurate labels. Don't ship.
- **Phase 1.5 — SFT distillation from v2 dataset** (NEW). 🟡 Planned.
  - Train a small (8B or 14B) student on `data/judge/v2_20260502/train.jsonl` (744 records, 50/50 gold/silver, P(both wrong) gate enforced). Verdict-only target. Eval on `data/judge/v2_20260502/eval.jsonl` (110 held-out gold).
- **Phase 2 — Persistent vLLM serving.** ✅ Done. The judge service supports `JUDGE_ROUTER=router|qwen_v2|gemini_v2` env var.
- **Phase 3 — RL integration.** 🟡 Not started.
- **Phase 4 — Production rollout.** 🔮 Future.

## Decision tree for the next milestone

The latency benchmarks (see [`latency.md`](latency.md)) showed:

1. **Cudagraphs and AWQ don't help on Blackwell+fp8** — decode is
   memory-bandwidth bound, not kernel-launch bound.
2. **Speculative decoding hurts at concurrency≥4** — adds compute that doesn't
   pay off when batching is already saturating the device.
3. **Verdict-only output without training fails** (router_verdict_only:
   71.93% acc). The model genuinely needs the CoT to reason correctly.
4. **Trained judge with verdict-only output is the only path to fast +
   accurate** — distillation moves the rules into weights so inference can
   skip the CoT.

## Phase 1.5 plan (Router → trained judge)

### 1.5.1 — Re-label training corpus with the router

The Phase-1 corpus is `judge_data/{train,val}.jsonl`, derived from
`runs/**/eval/checkpoint-*.json` (3,774 pairs, Gemini-labeled).

Run the `router` variant on each pair and replace the verdict + reasoning.
Save as `judge_data/{train,val}_v2.jsonl`. The user prompt structure is
already byte-identical to `rl_train.py`'s prompt, so the data flows in
without rework.

Cost: 3,774 pairs × 0.83s/call (c=16) ≈ 53 minutes of vLLM time.

### 1.5.2 — Train Qwen3-8B-judge-v2 with verdict-only target

Reuse `scripts/training/judge/judge_sft_train.py` from the Phase-1 run.
Changes:
- `MODEL_NAME = "Qwen/Qwen3-8B"` (was 7B in Phase 1)
- Training data: `judge_data/{train,val}_v2.jsonl`
- **Assistant target = "Correct" or "Incorrect" only** (drop the reasoning).
- Three epochs, LoRA r=64, bf16, batch 1 × accum 4, LR 2e-4 cosine.

### 1.5.3 — Validate against the manual oracle

Run the trained judge on `manual_review_input.jsonl` (285 cases). Ship gate:
- Overall ≥ 93% (within 2.5pp of router)
- Per-category: ≥ 88% on every category
- Per-call latency ≤ 0.2s p50 at concurrency=16

If gates fail, look at: per-category gap (which categories distill poorly?),
calibration (are confidences in the wrong direction?), data quality (are the
router's reasoning steps confusing the student?).

### 1.5.4 — If 8B falls short, try 14B with verdict-only

Same recipe but with Qwen3-14B base. Should match router accuracy (~95%) at
~0.4s p50. Falls between 8B-fast and router-accurate.

## Phase 2 — Serving

For Phase 1.5, just reuse `scripts/eval/judge_prompt_serve.sbatch` with the
trained model path. fp8 quantization stays. `enforce_eager=True` stays
(cudagraphs don't help, see latency.md). `max_model_len=4096` is enough now
since prompts are short.

For Phase 3 (RL), an auto-restart wrapper is needed (mirror the existing
`scripts/training/rl/rl_train_loop.sh`). Health-check via
`curl /v1/models`.

## Phase 3 — RL integration

See [`rl_integration.md`](rl_integration.md) for code-level changes.

Headline:
- New module `src/calendar_agent/local_judge_client.py` with an
  `evaluate_trajectory_local(...)` async function.
- Module-level `JUDGE_BACKEND = os.environ.get("JUDGE_BACKEND", "gemini")`
  toggle in `rl_train.py` and `rl_train_adaptive.py`.
- On `local` ConnectionError → fall back to Gemini path.

## Risks

- **Distillation gap.** Trained 8B might land at 91–93%, below router. Acceptable.
- **Latency variance.** vLLM serving over a long RL run can stall on KV cache
  exhaustion or similar. Auto-restart wrapper mitigates.
- **Reward signal noise.** Even at 95% the judge is wrong on 1 in 20
  trajectories. RL reward shaping may want to weight by judge confidence
  (future work — needs token-level logprobs from the trained judge).

## Why this plan, not the alternatives

| Alternative | Why not |
|---|---|
| Keep prompt-engineering further | Hit a wall at 95.44%; remaining ~5% is label noise + hard cases. Latency floor is ~0.8s/call at c=16, ~9.7s at c=1. Can't beat Gemini on per-call latency without reducing output length, which requires training. |
| Train a different size (e.g. 32B) | Bigger == slower. The pareto frontier for our setup (one MIG slice, RL with c=8–16) wants the smallest model that matches the router. |
| Use Gemini Pro instead | Cost. The Pro models burned 20× monthly budget in a day; a hard rule is to never use Pro. |
| Don't replace Gemini | Quota: ~99k judge calls per RL run is the dominant cost driver. Local removes the quota and lets us run more RL experiments. |
