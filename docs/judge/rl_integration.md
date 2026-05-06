# RL Integration — Phase 3

> **Updated 2026-05-02.** Judge service is live (`scripts/serving/judge_service.sbatch`)
> with `/no_think` + `max_tokens=512` and supports per-judge map selection via
> `JUDGE_ROUTER` env var (`router` v1 / `qwen_v2` / `gemini_v2`). RL trainers
> already POST to `:8765/verdict` (no Gemini fallback). Live measured accuracy
> on the 285 oracle: 91.93%; held-out CV ~92.4%.

> How the local judge replaces the Gemini call inside `rl_train.py` /
> `rl_train_adaptive.py`. Code-level plan; not yet shipped.

## What changes (and what doesn't)

| Component | Today (Gemini) | After (local judge) |
|---|---|---|
| Prompt builder (`build_user_prompt`) | unchanged | unchanged |
| Verdict parser (`_extract_verdict`) | unchanged | unchanged |
| `evaluate_trajectory(...)` body | `vertexai.GenerativeModel.generate_content` | HTTP POST to `JUDGE_API_URL/v1/chat/completions` |
| `EVAL_SYSTEM_PROMPT` | the existing one | replaced with the **router** system prompt |
| Cost / quota | Gemini API | Local vLLM, no quota |

**Do NOT modify** `src/calendar_agent/evaluation.py` — its sync
`evaluate_trajectory` still serves eval / data-gen flows that keep using
Gemini. Only RL training swaps.

## The new module

Create `src/calendar_agent/local_judge_client.py`:

```python
"""HTTP client for the local judge running on vLLM.

Used only by RL training (rl_train.py, rl_train_adaptive.py) when
JUDGE_BACKEND=local. Other code paths continue to use Gemini via
calendar_agent.evaluation.
"""
import asyncio
import os
from openai import AsyncOpenAI

# These come from scripts/eval/judge_prompt_tune.py at the time the local
# judge was qualified. Keep in sync if the judge prompt changes.
from scripts.eval.judge_prompt_tune import build_router  # type: ignore

_client = AsyncOpenAI(
    base_url=os.environ.get("JUDGE_API_URL", "http://localhost:8011/v1"),
    api_key="dummy",
)
_MODEL = os.environ.get("JUDGE_MODEL", "judge")
_TIMEOUT_S = float(os.environ.get("JUDGE_TIMEOUT_S", "30"))


def _extract_verdict(text: str) -> str:
    if not text:
        return "Incorrect"
    lines = [l.strip().strip(".,!?:; \"'*`#") for l in text.splitlines() if l.strip()]
    for line in reversed(lines):
        ll = line.lower()
        if ll == "correct":
            return "Correct"
        if ll == "incorrect":
            return "Incorrect"
    for line in reversed(lines):
        ll = line.lower()
        if "incorrect" in ll: return "Incorrect"
        if "correct" in ll:   return "Correct"
    return "Incorrect"


async def evaluate_trajectory_local(
    query: str, final_output: str, expected: str,
    before_text: str, after_text: str,
) -> str:
    rec = {
        "query": query, "final": final_output, "expected": expected,
        "before": before_text, "after": after_text,
        "cat": "",  # router falls back to fewshot_v3 when cat unknown
    }
    system, user, opts = build_router(rec)
    for attempt in range(3):
        try:
            r = await asyncio.wait_for(
                _client.chat.completions.create(
                    model=_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                    max_tokens=opts.get("max_tokens", 1024),
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                ),
                timeout=_TIMEOUT_S,
            )
            return _extract_verdict(r.choices[0].message.content)
        except asyncio.TimeoutError:
            continue
        except Exception:
            return "Incorrect"  # falls back to Incorrect — same as Gemini path
    return "Incorrect"
```

Note: this depends on `scripts/eval/judge_prompt_tune.py:build_router` being
importable. Cleaner is to lift `build_router` (and the prompt strings it
references) into a stable location like
`src/calendar_agent/judge_prompts.py`. Do that as part of the integration
PR — don't hot-link to the experiment script.

## Modifying `rl_train.py`

At module import (around line 51):

```python
JUDGE_BACKEND = os.environ.get("JUDGE_BACKEND", "gemini")
if JUDGE_BACKEND == "local":
    from calendar_agent.local_judge_client import evaluate_trajectory_local
```

In `evaluate_trajectory(...)` (line 316), at the top of the function:

```python
async def evaluate_trajectory(query, final_output, expected, before_days, after_days):
    if JUDGE_BACKEND == "local":
        before_text = format_day_state_text(before_days)
        after_text  = format_day_state_text(after_days)
        try:
            return await evaluate_trajectory_local(
                query, final_output, expected, before_text, after_text,
            )
        except Exception as e:
            print(f"[LOCAL JUDGE FAIL] {type(e).__name__}: {e} — falling back to Gemini")
            # fall through to existing Gemini path
    # existing Gemini code unchanged below this line
    ...
```

Same edit in `rl_train_adaptive.py` (line ~439).

## Phase 3 ship gates

Spec: 10-step RL smoke run with `JUDGE_BACKEND=local`. For every trajectory,
also call Gemini in parallel and log both verdicts.

| Gate | Spec |
|---|---|
| Verdict agreement with Gemini | ≤ 5% disagreement |
| Verdict agreement with **manual labels** (where available) | ≤ 5% disagreement (the harder bar — Gemini is 86.7% accurate, the trained judge is 95.44%) |
| `[EVAL ERROR]` log spam | none |
| `judge_latency_s` p50 | ≤ 0.5s at concurrency=16 |
| No fallback-to-Gemini events for transient errors | < 1% |

## Production rollout (Phase 4 — future)

- Flip default `JUDGE_BACKEND=local` in
  `scripts/training/rl/rl_train_adaptive_loop.sh`.
- Add 1% Gemini spot-check inside RL: re-judge 1% of rollouts with Gemini,
  log disagreement rate. Abort RL if >10% over a 100-step window.
- Auto-restart wrapper for the judge server (mirror
  `rl_train_loop.sh`). Health-check via `curl
  http://localhost:8011/v1/models`.

## MIG slice plan during RL

| Slice | Role |
|---|---|
| MIG 0 | Judge vLLM server (persistent, port 8011) |
| MIG 1 | RL training (vLLM serves the agent) |
| MIG 2 | RL adaptive run (concurrent) |
| MIG 3 | Free / dev |

Two RL runs + judge fits comfortably. The judge server has the lowest
priority — if other users need slices, judge moves to port 8014 and the
config follows.

## Risks specific to local-judge RL

- **Judge stalls during long run.** Auto-restart wrapper + Gemini fallback
  cover transient stalls. Hard hangs need monitoring.
- **Judge accuracy drift over training.** The judge prompt was tuned on
  trajectories from the *current* RL run. As the agent improves, its
  trajectories may shift distribution. Phase 4 spot-check catches this.
- **Verdict bias toward Correct.** The router has 6 false positives
  (vs 7 false negatives) — slightly biased to lenient. Could systematically
  over-reward. Watch the running mean reward — sustained climbing without
  acc improvements is a red flag.
- **Reward shaping with judge confidence.** Future work — a trained judge
  could expose token-level logprobs, letting RL weight reward by judge
  certainty.

## Service health & crash diagnosis

The sbatch (`scripts/serving/judge_service.sbatch`) launches three things and a
watchdog:

```
vLLM (EngineCore subprocess) ── fp8 Qwen3-14B on one MIG slice
FastAPI sidecar (port :JUDGE_PORT) ── prompt build + verdict extraction
Watchdog (background subshell) ── 10s telemetry → logs/watchdog.jsonl
```

If a child dies, `wait -n` returns and the script writes
`logs/postmortem_<jobid>_<epoch>.log` before exiting (Slurm requeues).

### What the watchdog samples (each 10s)

| Field | Why it matters |
|---|---|
| `gpu_mem_used_free_util` | Per-MIG GPU memory + util. Sudden drop = engine died. |
| `ecc_unc_vol_agg` | Volatile + Aggregate uncorrectable ECC counts. Best non-privileged proxy for hardware Xids on this driver — true Xid history lives only in `dmesg` (sudo-gated, postmortem also tries `journalctl -k`). Won't catch Xid 13/74 (illegal access / SM fault) but does catch ECC-related Xid 48/63. |
| `cg_mem_current` / `cg_mem_max` | Slurm cgroup memory (the `--mem=80G` enforcement). If `current` approaches `max`, the kernel is about to SIGKILL the largest victim in the cgroup — usually the EngineCore subprocess. |
| `cg_oom_kill` | Counter from `memory.events`. Increments **only** when this cgroup OOM-killed something. The smoking gun for silent EngineCore death. |
| `shm_used_kb` / `shm_avail_kb` | vLLM v1 uses `/dev/shm` for IPC between APIServer and EngineCore. Exhaustion can crash the engine without a Python traceback. |
| `mem_avail_kb` | Host-wide RAM. Useful only for system-wide OOM (rare on this box). |
| `vllm_alive` / `sidecar_alive` / `engine_pids` | PID liveness. The EngineCore is `vLLM_PID`'s child. |

> ⚠️ **Watchdog ordering matters.** The subshell snapshots `$VLLM_PID` /
> `$SIDECAR_PID` at fork time, so it must launch *after* both PIDs are set.
> An earlier version forked before `VLLM_PID=$!` and silently logged
> `vllm_alive=0` for the entire run — useless for diagnosis. Fixed
> 2026-05-04 after both judges crashed at the same instant.

### The 2026-05-04 10:38 incident

Two judge jobs (slices 0 + 2) died simultaneously while a parallel eval
on a third slice kept running.

- **Symptom:** `EngineDeadError: EngineCore died unexpectedly` — no Python
  traceback in `vllm.log`. APIServer exited rc=0 cleanly.
- **Ruled out:** Xid (would have hit the eval slice on the same physical
  GPU); ECC (counters zero); host-wide OOM (242 GiB available); Slurm
  signal (rc=0).
- **Most likely:** per-cgroup OOM-kill (SIGKILL is uncatchable, leaves no
  traceback). Both judges grow on parallel curves under identical eval
  fan-out and would cross `--mem=80G` within the same kernel scan.
- **Other live possibilities:** poison-request bug in vLLM 0.19 + fp8 on
  Blackwell; `/dev/shm` exhaustion.
- **Why we couldn't tell:** old watchdog reported `vllm_alive=0` always;
  no Xid sampling; no cgroup memory sampling; no `/dev/shm` sampling.
  Postmortem `dmesg` requires sudo on this box.

The watchdog upgrade above lands all four signals so the next incident
self-diagnoses.

### Reading a crash

```bash
# Did the cgroup OOM-kill anything?
jq 'select(.cg_oom_kill > 0)' runs/judge_service_20260501/logs/watchdog.jsonl | head

# Memory pressure curve in the 5min before crash
jq -r '[.ts, .cg_mem_current, .cg_mem_max] | @tsv' \
    runs/judge_service_20260501/logs/watchdog.jsonl | tail -30

# Any uncorrectable ECC events ever? (proxy for hardware Xids on this driver)
jq -r 'select(.ecc_unc_vol_agg != "0,0" and .ecc_unc_vol_agg != "")' \
    runs/judge_service_20260501/logs/watchdog.jsonl

# /dev/shm trend
jq -r '[.ts, .shm_used_kb, .shm_avail_kb] | @tsv' \
    runs/judge_service_20260501/logs/watchdog.jsonl | tail -30
```

If `cg_oom_kill` ever increments → bump `--mem` in the sbatch (try 120 G)
and look at what's growing in `vllm_rss_kb` over time. If
`ecc_unc_vol_agg` ever becomes non-zero → likely hardware ECC fault;
capture `dmesg` (sudo) for the exact Xid code. If `shm_avail_kb` falls
toward zero → mount a larger `/dev/shm` or switch vLLM IPC to a
file-backed transport.

**Xid blind spot:** this driver's `nvidia-smi` doesn't expose a `-d XID`
display type, so the watchdog can't sample Xid counters directly. The
postmortem tries `journalctl -k` as a best-effort fallback. If kernel
logs are also locked down on this machine, the only path forward is to
get someone with sudo to enable `setcap cap_syslog+ep /usr/bin/dmesg` (or
add the operator to the `systemd-journal` group).

## Open: per-category routing in production

`build_router` dispatches based on `rec["cat"]`. In RL, do we have the
category? Yes — it's available in the scenario metadata (each scenario file
records its category). Need to thread `category` through to the
`evaluate_trajectory_local` call. If not threaded, `build_router` falls
back to `fewshot_v3` (a slight per-category accuracy loss but not fatal).
