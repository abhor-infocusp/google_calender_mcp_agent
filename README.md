# Google Calendar MCP Agent

Tool-calling agent for Google Calendar with a full SFT + RL training pipeline
on Qwen3-14B (4-bit) using ART/GRPO, vLLM, and Gemini as the reward judge.

## Quick links

| You want to... | See |
|---|---|
| Understand what's in the repo | this file (below) |
| Launch RL training (correctly) | [`docs/multi_tenant_training.md`](docs/multi_tenant_training.md) |
| Recover a corrupted run | [`docs/multi_tenant_training.md#recovery-after-interference`](docs/multi_tenant_training.md#recovery-after-interference) |
| File an upstream ART bug | [`docs/art_asyncio_deadlock_analysis.md`](docs/art_asyncio_deadlock_analysis.md) |
| See current pipeline state | `PROGRESS.md` |

## Repo layout

```
src/calendar_agent/         # Installable package — import as calendar_agent.*
  core.py                     Tool declarations, dispatch, system prompt, snapshots
  tools.py                    OpenAI tool conversion + final-answer tool
  evaluation.py               Gemini-judge eval helpers
  environment/                Calendar environment (Pydantic models, CRUD methods)
  paths.py                    PROJECT_ROOT, DATA_DIR, RL_DATA_DIR, etc.
  art_patches.py              Runtime monkey-patches for ART 0.5.17 (D, E, G, H, I)

scripts/
  data_generation/            Generate calendars + queries + trajectories via Gemini
  training/
    common/                   Cross-cutting: launch wrapper, slice map, lora merge
      auto_restart.sh           Centralized launch wrapper — see multi_tenant_training.md
      slice_map.sh              Host MIG-slice ↔ CUDA UUID ↔ CPU range
      merge_lora.py             Merge a LoRA adapter into fp16 base for serving
    rl/                       RL training family
      rl_train.py               Main RL trainer (Qwen3-14B + ART/GRPO + Gemini judge)
      rl_train_small.py         Same pipeline, Qwen2.5-0.5B — fast iteration / debug
      rl_train_adaptive.py      Alternate trainer with best-checkpoint retention
    sft/                      SFT family
      sft_train.py              SFT on tool-call trajectories
    dpo/                      DPO family
      dpo_train.py              DPO trainer
      mine_dpo_pairs.py         Pair mining from existing trajectories
    judge/                    Local-judge SFT family (replacing Gemini API)
      judge_data_prep.py        Build train/val jsonl from Gemini rollouts
      judge_sft_train.py        SFT a smaller judge model
      judge_train_launch.sh     Launch wrapper specific to judge SFT
    legacy/                   Archived per-script wrappers (don't use)
  eval/
    eval_qwen.py                Single-calendar eval via vLLM
    eval_batch.py               Batch eval with Gemini judge
    eval_all_checkpoints.py     Multi-checkpoint orchestrator
  utils/
    plot_rewards.py             Reward curve plot
    view_results.py             ART results viewer

tests/                       pytest suite (env, serialization, repro)
runs/                        per-experiment output dirs (gitignored except metadata)
.art/                        ART checkpoints + history (gitignored)
docs/                        Operational docs (multi-tenant launch, upstream issues)
```

## Standard launch (single-line copy paste)

```bash
source scripts/training/common/slice_map.sh
SLICE=0  # pick an unused slice; check `nvidia-smi`
CUDA_VISIBLE_DEVICES=$(slice_cuda_uuid $SLICE) \
TASKSET_CPUS=$(slice_cpu_range $SLICE) \
SCRIPT_PATH=scripts/training/rl/rl_train.py \
RUN_DIR=runs/rl_qwen3_14b_$(date +%Y%m%d) \
nohup scripts/training/common/auto_restart.sh \
    > runs/rl_qwen3_14b_$(date +%Y%m%d)/logs/loop.log 2>&1 &
disown
```

`auto_restart.sh` handles setsid, process-group cleanup, deadlock retry,
thread caps, CPU pinning, and milestone checkpoint retention. Always set
`TASKSET_CPUS` matching the slice — see `docs/multi_tenant_training.md`.

## Pipeline status snapshot

- **SFT**: Qwen3-14B v6 best checkpoint at 82.5% on RL data (epoch 4, ckpt 6212).
- **RL**: GRPO training, 7 categories joint, 622 scenarios × 20 epochs target. Mean reward plateau ~0.85 pre-collapse.
- **Judge**: `gemini-2.0-flash-001` (never use `pro` — burned the budget once).
- **Eval**: vLLM + `--tool-call-parser hermes` (NOT `qwen3_xml`).
- **Checkpoints**: ART keeps milestone every 500 steps + best-by-reward (since the 2026-04-25 incident).

## Environment

- Conda env: `agentic` at `/home/abhor/miniconda3/envs/agentic`
- Pinned versions in `pyproject.toml` — bump deliberately, the stack has
  version-specific monkey-patches in `art_patches.py`.
- Hardware: NVIDIA RTX PRO 6000 Blackwell, MIG-partitioned into 4× 1g.24gb
  slices. 128 logical CPUs.

## Telemetry / debugging

Every run writes structured logs under `runs/<run>/logs/`:

| File | What's in it |
|---|---|
| `train_*.log` | Full stdout — STEP SUMMARYs, debug traces |
| `loop.log` | `auto_restart.sh` lifecycle events |
| `debug/heartbeat.jsonl` | Phase + step every 30s (daemon thread) |
| `debug/deadlock_detected.jsonl` | Patch G/I timeout events (retry / exit) |
| `debug/pyspy_*.txt` | Stack dumps from `run_with_hang_watchdog` |
| `runs/<run>/metadata.jsonl` | One line per process start: git sha, env, sibling PIDs |

## Tests

```bash
PYTHONPATH=src /home/abhor/miniconda3/envs/agentic/bin/pytest tests/
```
