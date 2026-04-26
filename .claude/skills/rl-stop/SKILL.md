---
name: rl-stop
description: Safely stop a single RL/SFT training run by reading its metadata.jsonl. Refuses to kill anything not matching the recorded script. Use this instead of pgrep+kill to avoid accidentally killing the wrong run.
---

# rl-stop — safe single-run shutdown

## When to use

When the user asks to stop, kill, terminate, or shut down a specific training
run. **Do not invent pgrep filters by hand** — they have already produced one
accidental cross-run kill in this repo. Use this skill.

## How it works

`scripts/utils/stop_run.sh <RUN_DIR>` reads the most recent entry of
`<RUN_DIR>/metadata.jsonl`, extracts the python pid + script, verifies the live
process is still running THAT script (refusing the kill if not), then takes
down the wrapper + the python's process group cleanly.

## Usage

```bash
scripts/utils/stop_run.sh runs/rl_qwen3_14b_20260420
```

Exit codes:
- `0` — clean stop (or already stopped)
- `1` — usage error / no metadata
- `2` — pid does not match the recorded script — refused to kill (good safety net)
- `3` — kill failed (shouldn't happen)

## Sibling helper

`scripts/utils/list_runs.sh` — table of all run dirs with their last-seen pids
and whether each is currently ALIVE. Run this before stop_run if unsure which
run to stop.

```bash
scripts/utils/list_runs.sh
```

## Anti-patterns (do NOT do these)

- ❌ `pgrep -f rl_train.py | xargs kill` — kills ALL rl_train.py runs across
  the host. Two trainings on different slices? Goodbye both.
- ❌ Filtering with awk against `pgrep` output of "auto_restart.sh" — every
  run uses the same wrapper, so the filter has to be by python pid which
  pgrep alone can't disambiguate cleanly.
- ❌ `kill -9 <pid>` without first killing the wrapper — the wrapper's
  auto-restart loop will immediately spawn a fresh python.
