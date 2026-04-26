---
name: rl-status
description: Get a fast status snapshot of an RL/SFT training run via lnav SQL queries on its JSONL streams. Replaces the multi-step ad-hoc grep ritual (nvidia-smi → tail train.log → grep STEP SUMMARY → wc -l deadlock_detected.jsonl → ...) with structured queries. Use when the user asks "is it running / how is it doing / how many deadlocks / what phase".
---

# rl-status — structured status via lnav

## When to use

When the user asks any flavor of "is the training running / how is it doing /
where are we / how many deadlocks / what phase". Use this skill instead of a
manual chain of `nvidia-smi`, `tail`, `grep`, `wc -l`.

## Setup (one-time per machine)

If `lnav` is missing or our format files aren't installed, run:

```bash
scripts/utils/lnav_install_formats.sh
```

This installs three lnav format files into `~/.lnav/formats/installed/`:
- `_calendar_agent_heartbeat` — phase / step / phase_age_s / pid (from heartbeat.jsonl)
- `_calendar_agent_deadlock` — Patch G/I retry/exit events (from deadlock_detected.jsonl)
- `_calendar_agent_run_metadata` — git sha, env, taskset, isolation knobs (from metadata.jsonl)

(Format keys are `_`-prefixed to win lex-priority over lnav's built-in `caddy_log` JSON detector, which matches anything with a `ts` field.)

## How to use it

`scripts/utils/lnav_rl.sh <RUN_DIR> --exec '<sql>'` runs a query non-interactively against all 3 streams of one run.

Or open the run interactively (TUI):

```bash
scripts/utils/lnav_rl.sh runs/rl_qwen3_14b_20260420
```

## Canonical queries

Always start by running `list_runs.sh` to find which RUN_DIR is alive:

```bash
scripts/utils/list_runs.sh
```

Then for the alive run:

### Latest phase + step
```bash
scripts/utils/lnav_rl.sh runs/rl_qwen3_14b_20260420 --exec ';SELECT log_time, phase, step, phase_age_s FROM _calendar_agent_heartbeat ORDER BY log_time DESC LIMIT 5'
```

### Deadlock event breakdown (real vs spurious)
```bash
scripts/utils/lnav_rl.sh runs/rl_qwen3_14b_20260420 --exec ';SELECT phase, reason, count(*) FROM _calendar_agent_deadlock GROUP BY phase, reason ORDER BY 3 DESC'
```

### Most recent process start (any restarts? what isolation?)
```bash
scripts/utils/lnav_rl.sh runs/rl_qwen3_14b_20260420 --exec ';SELECT pid, taskset_cpus, omp_num_threads, git_commit FROM _calendar_agent_run_metadata ORDER BY log_time DESC LIMIT 3'
```

### How long has the current phase been stuck?
```bash
scripts/utils/lnav_rl.sh runs/rl_qwen3_14b_20260420 --exec ';SELECT phase, max(phase_age_s) FROM _calendar_agent_heartbeat GROUP BY phase'
```

### Deadlock rate (last hour)
```bash
scripts/utils/lnav_rl.sh runs/rl_qwen3_14b_20260420 --exec ';SELECT event, count(*) FROM _calendar_agent_deadlock WHERE log_time > datetime("now", "-1 hour") GROUP BY event'
```

## Things lnav formats don't cover

- **STEP SUMMARY rewards** are inside `train_*.log` (free text), not in any
  JSONL. Use `grep '\\[STEP [0-9]\\+ SUMMARY\\]' runs/.../train_*.log | tail -10`
  for those — or extend the format with a regex format file later.
- **Reward curve** — use `scripts/utils/plot_rewards.py` (not lnav).

## Why this beats the old ritual

The old "is it healthy?" ritual used 10–15 bash calls per check (nvidia-smi,
tail several logs, grep multiple patterns, wc -l, py-spy parse). Three lnav
SQL queries replace ~80% of that. The remaining: `nvidia-smi` for GPU + the
STEP SUMMARY tail.
