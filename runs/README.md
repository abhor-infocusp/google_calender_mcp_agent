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

## Current runs

| Run | Kind | Model | Status |
|---|---|---|---|
| `rl_qwen3_14b_20260420` | RL (GRPO) | Qwen3-14B | relaunched after reorg, MIG 0 |
| `sft_v6_qwen3_14b_20260420` | SFT | Qwen3-14B | training, MIG 1 |
| `archive/sft_v5_qwen1.5b` | SFT (archived) | Qwen2.5-1.5B | 74.6% best (ckpt 6152) |

## How scripts use this

- `scripts/training/sft_train.py` writes to `$SFT_RUN_DIR` (default `runs/sft_v6_qwen3_14b_20260420`).
- `scripts/training/rl_train_loop.sh` writes to `runs/rl_qwen3_14b_20260420/logs/`.
- `scripts/training/rl_train.py` writes debug dumps to `$RL_RUN_DIR/logs/debug`.

## Starting a new run

1. Create `runs/<kind>_<model>_<date>/`.
2. Write `config.json` capturing hparams + git sha.
3. Launch training with env var pointing at the run dir.
4. Logs/checkpoints land under that dir; nothing at project root.
