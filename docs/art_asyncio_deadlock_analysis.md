# ART asyncio Deadlock — Analysis & Upstream Issue Draft

> **Status:** Workaround deployed locally (Patches G/I in `src/calendar_agent/art_patches.py`).
> Upstream issue **not yet filed** on https://github.com/OpenPipe/ART/issues. The body
> below is the draft to post when ready.
> **Last reviewed:** 2026-04-26

---

## Title
`UnslothService`: `_async_prepare_inputs` spuriously times out on healthy long rollouts + suggested fix

## Body

**ART version:** 0.5.17
**Env:** torch 2.10.0+cu128, vLLM 0.19.0, unsloth 2026.4.4, transformers 4.57.6, trl 0.24.0, peft 0.19.0, Python 3.11
**Hardware:** NVIDIA RTX PRO 6000 Blackwell, 1x 1g.24gb MIG slice
**Workload:** GRPO on Qwen3-14B (4-bit), multi-turn tool-calling rollouts with external judge, 622 scenarios × 20 epochs

### Summary

`art/unsloth/service.py`'s training bridge (`_async_prepare_inputs` at line 988-996) uses `asyncio.Queue` + a nested `asyncio.run(...)` under `nest_asyncio`. When we add a timeout to catch the known nest_asyncio race (deadlock reported in issues and PR #429), we observe the timeout firing **~93% of the time on perfectly healthy processes** — not on real deadlocks.

### Root cause

The `_train_task` created on `service.py:688-694` is long-lived and reused across every subsequent `model.train()` call (else branch sets `warmup=False`). HF `trainer.train()` runs forever (10M-row dummy dataset, no `max_steps`) and repeatedly calls `_prepare_inputs`, which does `await inputs_queue.get()`.

**Between `model.train()` calls**, there is no producer. `process_train_batch` (lines 109-146) has exited and main() is doing rollouts / judge / GC / checkpoint_delete. The long-lived train_task is immediately back in `_prepare_inputs`, waiting for the **next** `model.train()` to produce. During that inter-call gap, `asyncio.wait_for(queue.get(), timeout=T)` is running — and the clock is ticking.

For workloads where inter-call gaps exceed T, every timer fires spuriously.

### Data from our pipeline (2 days, Qwen3-14B)

- Historical "deadlock" events at T=300s: **59**.
- Phase at time of timeout (heartbeat-based): **93% `rollouts`, 6% `gc_empty_cache`/`checkpoint_delete`, 2% `model_train`**. If the process were really hung in `_prepare_inputs`, the outer phase would stay stuck — but it's advancing normally (steps complete, phase transitions happen). The main coroutine is running via `nest_asyncio`'s re-entrant `_run_once` even while the nested `asyncio.run` is stuck.
- Max phase age observed at timeout: **115s** for rollouts, **58s** for model_train — nothing approaches 300s. Confirms the process is healthy.
- Rollouts duration distribution: **p50=108s, p95=291s, p99=431s, max=560s**. At T=300s we expect ~5% of rollouts phases to trigger spurious timeouts.
- Baseline rate: ~1 spurious "deadlock" per 10-20 min at T=300s. Restart + model reload takes ~90-120s. Effective throughput loss: 10-20%.

### Genuine deadlock exists but is rare

We *did* see one real hang on 2026-04-17: the process was stuck inside `_prepare_inputs` for **57 hours** with no progress (phase stuck at `model_train`, phase_age reached 3669s in heartbeat log). That's the nest_asyncio lost-wakeup race that PR #429 targets. It has happened exactly once in weeks of running.

### Proposed fix

Two independent improvements, in order of impact:

1. **Short-term (timeout semantics)**: Either
   - (a) Make the inter-call wait explicit: after `process_train_batch` exhausts, put a sentinel on `inputs_queue` and teach `_prepare_inputs` to `await asyncio.Event.wait()` (event set by the next call to `_train_shared`). No spurious timer.
   - (b) Differentiate spurious vs real in the timeout handler: retry if the training-step is not actually in progress (e.g. `trainer.state.is_in_train == False` or inspect a caller-supplied phase callback).

2. **Long-term (architecture)**: Replace the `asyncio.Queue` + `nest_asyncio` bridge with a `queue.Queue` + running `trainer.train()` on a real OS thread via `loop.run_in_executor`. Eliminates the whole class of reentrancy bugs and the need for `nest_asyncio.apply()` in `art/unsloth/train.py:20`. Results queue also needs a thread-safe bridge.

### Our temporary workaround

We've implemented (b) as a monkey-patch: on timeout, check a caller-registered phase snapshot. If phase != `model_train` or time-since-last-put < timeout → retry. Only exit on a true hang (phase == `model_train` AND phase_age > timeout AND no puts in > timeout) or after hitting a 30-min hard ceiling. Works, but it would be better as a first-class ART feature since every user hits this.

Happy to share the patch code and more of our diagnostic logs (heartbeat.jsonl, py-spy stacks from the real 57h hang) if useful.

### Repro

Minimal reproducer confirming the nest_asyncio + asyncio.Queue race itself is timing-dependent: https://gist.github.com/... *(TODO: upload tests/repro_art_deadlock.py)*. Harder to deterministically repro the spurious timeouts — they depend on rollout-duration tail.
