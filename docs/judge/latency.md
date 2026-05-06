# Judge Latency — Optimization Sweep + Gemini Comparison

> **Historical (2026-05-01).** Numbers in this file are from before the
> Phase-0 fix. The deployed judge now runs with `/no_think` + `max_tokens=512`
> (live p50 = 11s on the 285 oracle, no truncation cliff). The "0.83s p50"
> figure below was achieved without thinking but in production thinking-on
> was used until 2026-05-02; the historical context here documents what
> optimizations did and didn't work and remains accurate for that purpose.

> Where the judge's wall-clock time goes, what optimizations we tried, and
> what didn't help. Source: `runs/judge_prompt_tune_20260430/results/`.

## Top-line

| Config | Per-call (c=16 effective) | Per-call (c=1 serial) | p50 from latency bench | Acc |
|---|---:|---:|---:|---:|
| Gemini-2.0-flash (RL production, n=6213) | — | — | **0.46s** | 86.67% |
| 14B fp8, **router prompt** (winner) | 0.83s | 9.77s | 9.66s | **95.44%** |
| 14B fp8, baseline EVAL prompt | 0.83s | 9.7s | — | 86.67% |
| 14B fp8, short prompt (no examples) | — | — | 9.48s | — |
| 8B fp8, short prompt | — | — | 4.81s | — |
| 8B fp8, router prompt | 0.46s | — | — | 88.07% |

**Gemini is ~21× faster than our local 14B at concurrency=1**, ~2× faster at
concurrency=16 effective throughput. The gap closes under parallelism but
never reverses without training.

## Where the 9.7s p50 actually goes

Decomposition for 14B fp8 at concurrency=1:
```
prefill (4k prompt tokens @ ~5,000 tok/s)     ~0.8s
decode  (256 output tokens @ ~26 tok/s)       ~9.0s
─────────────────────────────────────────────
total                                         ~9.7s
```

Decode dominates 93% of the time. Output length is the bottleneck — the
prompt asks for full chain-of-thought reasoning (~256 tokens median), and
each token takes ~0.038s on Blackwell+fp8.

## Why Gemini is so much faster

| Factor | Our 14B fp8 | Gemini-2.0-flash | Ratio |
|---|---|---|---:|
| Output tokens generated | ~256 (CoT reasoning) | ~30–50 (terse verdict) | ~5× |
| Per-token decode | ~26 tok/s (single-batch, fp8 on Blackwell, `enforce_eager`) | ~150–300 tok/s (TPUv5+, batched, cudagraphs, speculative decoding) | ~5–10× |
| Prompt prefill | ~1s on 4k tokens | ~0.05s on 4k tokens | ~20× but small absolute |
| Network round-trip | 0 (local) | ~0.05s | (favors local) |
| **Combined** | **~9.7s** | **~0.5s** | **~21×** |

Two compounding factors: shorter output × faster per-token decode = ~25–50×
ceiling, observed ~21× because some prefill+overhead is fixed.

## Optimization sweep on 14B fp8

| Config | c=16 wall (n=285) | c=16 per-call | c=1 per-call (n=60) | Acc | Net |
|---|---:|---:|---:|---:|---|
| Baseline (`--enforce-eager` on, fp8) | 235.9s | 0.83s | 9.77s | 95.44% | reference |
| Cudagraphs ON (`--enforce-eager` off) | 234.9s | 0.82s | **9.78s** | 93.68%* | **0 gain** |
| Speculative decoding (ngram, c=16) | 330.3s | 1.16s | — | 93.68%* | **40% slower** |
| Speculative decoding (ngram, c=1) | — | — | **8.35s** | 86.67%* | 15% faster |
| AWQ 4-bit (vs fp8) | (cancelled at 18min) | — | — | — | **~3× slower** at c=16 |
| max_tokens=200 (cap output) | 203.8s | 0.71s | — | 89.12% | -6.3pp acc, 13% faster |
| max_tokens=50 | 71.5s | 0.25s | — | 63.51% | catastrophic |
| Verdict-only (no CoT, mt=10) | 27.7s | 0.10s | — | 71.93% | model can't skip CoT |

\* Accuracy variance vs baseline is vLLM nondeterminism between server
restarts (fp8 + concurrent batching has some non-determinism even at
temperature=0).

## What we learned

### Cudagraphs gave nothing on Blackwell+fp8
Decode is **memory-bandwidth bound**, not kernel-launch bound. Cudagraphs
save kernel-launch overhead, which isn't where the time is going. Confirms
the existing project memory note: `enforce_eager=True` on Blackwell costs
nothing in performance and avoids cudagraph memory overhead. Keep it on.

### Speculative decoding is concurrency-conditional
- At **c=1**: ngram-based spec-dec saves 15% (9.77→8.35s). The model's
  output has predictable structure (`(A)... (B)...`, diff lines, summaries
  echoed from prompt) so n-gram matching finds wins.
- At **c=16**: spec-dec **slows things down by 40%**. The main model is
  already busy decoding 16 streams in parallel — adding speculation =
  more compute per accepted token without saving wall time.

For RL (which always parallelizes judge calls within a step), spec-dec is
not useful.

### AWQ is worse than fp8 on Blackwell
The 14B-AWQ bench was still running at 18 minutes when fp8 finished in 4.
Generation throughput was 47 tok/s vs fp8's effective ~50 tok/s under load,
but more importantly AWQ has higher per-iteration overhead. fp8 has native
Blackwell tensor-core support; AWQ has to dequant-and-multiply. **Use fp8.**

### Concurrency is the biggest free lever
- c=1: 9.77s/call
- c=16: 0.83s/call (12× speedup)

vLLM batches concurrent requests automatically. RL training already issues
judge calls in parallel within a step — we get this for free.

### Output length is the bottleneck for any further optimization
- Cap output at 200 tokens → -6.3pp accuracy.
- Cap at 50 → catastrophic (model gets cut off mid-reasoning).
- Tell model "no reasoning, output Correct/Incorrect" → 71.93% (model can't
  skip CoT and still reason correctly).

The model genuinely needs the chain-of-thought to reach the right verdict.
The only way to reduce output length without losing accuracy is **training
the rules into the weights** (distillation), then asking the trained model
for verdict-only output at inference.

## What a trained judge can do

Trained 14B with verdict-only output (~5 tokens), short prompt:
```
prefill (~500 tok prompt @ optimized)         ~0.1s
decode  (~5 tokens @ optimized)               ~0.05s
─────────────────────────────────────────────
total                                         ~0.15s/call serial
```

With concurrency=16 effective: ~0.01s per call — 30–50× faster than Gemini.

A trained 8B at the same recipe: ~0.05–0.1s serial. Even faster.

This is why Phase 1.5 of [`plan.md`](plan.md) targets distilling the router
into a smaller model with verdict-only training targets.

## Implication for RL inference cost

A typical RL run is ~99,200 judge calls (12,400 steps × 8 rollouts).

| Judge | Per-call (RL c=16) | Total judge time per RL run |
|---|---:|---:|
| Gemini-2.0-flash | 0.46s | ~12.7 hours |
| 14B fp8 router (current) | 0.83s | ~22.9 hours |
| Trained 14B verdict-only | ~0.15s | ~4.1 hours |
| Trained 8B verdict-only | ~0.07s | ~1.9 hours |

A trained judge brings judge inference from a near-blocker (12+ hours) down
to a non-issue (~2 hours). Plus no API quota.

## Files

- `runs/judge_prompt_tune_20260430/results/summary.csv` — every variant +
  per-call latency
- `runs/judge_prompt_tune_20260430/results/latency_bench.json` — separate
  serial-latency benchmark (n=30 per config)
- `runs/judge_prompt_tune_20260430/logs/serve_*.log` — vLLM server logs for
  each config (cudagraphs, spec-dec, AWQ, baseline)
- `scripts/eval/judge_latency_bench.py` — the benchmark harness
- `scripts/eval/judge_serve_*.sbatch` — server launch scripts for each config
