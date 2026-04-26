# runs/

Per-training-run output. Each run is self-contained.

## Layout convention

```
runs/<run_name>/
  config.json      # model, hparams, git sha, start time, MIG slice — write at launch
  logs/            # stdout/stderr and any heartbeat/debug dumps
  checkpoints/     # HF Trainer checkpoints (SFT) or symlink into .art/ (RL)
  eval/            # per-checkpoint eval JSONs, summary.csv
  diagnostics/     # loss curves, plots, training_history.json
```

Run name: `<kind>_<model_short>_<YYYYMMDD>` — kind ∈ `{sft, rl}`.

## What's tracked in git

`config.json`, `eval/*.json`, `eval/summary.csv`, `diagnostics/*.csv`, `README.md`.
Checkpoints and logs are ignored (too large / noisy). See repo `.gitignore`.

## Live status

Don't maintain a current-runs table here — it goes stale fast.
Each run writes `metadata.jsonl` (process start, git sha, MIG slice, sibling PIDs).
For pipeline state see `PROGRESS.md`.

## How scripts use this

- `scripts/training/sft/sft_train.py` writes to `$SFT_RUN_DIR` (default `runs/sft_v6_qwen3_14b_20260420`).
- `scripts/training/common/auto_restart.sh` writes to `$RUN_DIR/logs/` (loop log + train logs).
- `scripts/training/rl/rl_train.py` writes debug dumps to `$RL_RUN_DIR/logs/debug`.

## Starting a new run

See `docs/multi_tenant_training.md` for the full launch protocol. Short version:

1. Pick an unused MIG slice; check `nvidia-smi`.
2. `source scripts/training/common/slice_map.sh`.
3. Set `CUDA_VISIBLE_DEVICES`, `TASKSET_CPUS`, `SCRIPT_PATH`, `RUN_DIR`.
4. `nohup scripts/training/common/auto_restart.sh > $RUN_DIR/logs/loop.log 2>&1 &`.
