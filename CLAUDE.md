# Google Calendar MCP Agent

Tool-calling agent for Google Calendar with SFT + RL pipelines on Qwen3-14B.

This file holds **stable facts**: paths, imports, conventions. Anything dated —
status, results, current best ckpt — lives in `PROGRESS.md`. Per-category
analysis lives in `docs/categories/`. Cross-session warnings live in auto-memory.

## Where things are

| You want... | Look at |
|---|---|
| Pipeline state, recent milestones | `PROGRESS.md` |
| Per-category performance & targets | `docs/categories/README.md` |
| Launch protocol (MIG slices, auto_restart) | `docs/multi_tenant_training.md` |
| Held-out eval results | `runs/analysis/test_eval_summary.md` |
| Run output layout | `runs/README.md` |
| ART asyncio deadlock context | `docs/art_asyncio_deadlock_analysis.md` |
| Local judge — current state, plan, integration | [`docs/judge/`](docs/judge/) |
| Repo overview for newcomers | `README.md` |

## Environment
- Python: `/home/abhor/miniconda3/envs/agentic/bin/python` (3.11)
- Run scripts as: `PYTHONPATH=src /home/abhor/miniconda3/envs/agentic/bin/python scripts/<path>.py`
- GPU: NVIDIA RTX PRO 6000 Blackwell, MIG-partitioned into 4× ~24 GiB slices, compute 12.0, bf16 + FA2
- Stack pinned in `pyproject.toml`. Bump deliberately — `art_patches.py` is version-specific.

## Package layout (`src/calendar_agent/`)

```
core.py            Tool declarations, dispatch, system prompt, snapshots, format_tool_result, DAY_NAMES, C
tools.py           OpenAI tool conversion, RETURN_FINAL_ANSWER_TOOL
evaluation.py      EVAL_SYSTEM_PROMPT, evaluate_trajectory, format_day_state_text
paths.py           PROJECT_ROOT, DATA_DIR, SFT_DATA_DIR, RL_DATA_DIR, CREDENTIALS_PATH
prompts.py         Data-generation prompt templates
art_patches.py     Runtime monkey-patches for ART 0.5.17 (D, E, G, H, I) — import before `art`
environment/       CalendarEnvironment + Pydantic models (User, Attendee, Event, Calendar)
```

## Scripts (`scripts/`)

```
run_agent.py                    Core agent loop (Gemini + calendar tools)
data_generation/                generate_data.py, generate_trajectories.py, generate_test_data.py
training/
  common/                       auto_restart.sh, slice_map.sh, merge_lora.py
  sft/sft_train.py              SFT (Unsloth + TRL, loss masking, CSV logging)
  rl/rl_train.py                GRPO (ART 0.5.17 + vLLM)
  rl/rl_train_adaptive.py       Variant w/ best-checkpoint retention
  rl/rl_train_small.py          Qwen2.5-0.5B fast iteration
  dpo/                          dpo_train.py, mine_dpo_pairs.py (paused — see feedback_dpo_skipped)
  judge/                        Local judge SFT (Qwen3-7B, in progress)
  legacy/                       Don't use
eval/
  eval_qwen.py                  Single-calendar eval via vLLM
  eval_batch.py                 Batch eval with Gemini judge
  eval_all_checkpoints.py       Multi-checkpoint orchestrator
utils/                          plot_rewards.py, view_results.py
```

## Key imports
```python
from calendar_agent.core import (
    SYSTEM_PROMPT, TOOL_DECLARATIONS, CALENDAR_TOOL,
    dispatch_tool_call, format_tool_result, snapshot_events,
    filter_by_days, compute_fallback_now, DAY_NAMES, C,
)
from calendar_agent.tools import get_openai_tools, RETURN_FINAL_ANSWER_TOOL
from calendar_agent.evaluation import EVAL_SYSTEM_PROMPT, evaluate_trajectory, format_day_state_text
from calendar_agent.environment import CalendarEnvironment
from calendar_agent.paths import PROJECT_ROOT, DATA_DIR, SFT_DATA_DIR, RL_DATA_DIR, CREDENTIALS_PATH
```

## Domain conventions

**Tools (7):** `get_current_time`, `list_events`, `get_event`, `create_event`,
`update_event`, `delete_event`, `respond_to_event`.

**Trajectory step format:**
- `{"role": "user", "content": "..."}`
- `{"role": "tool_call", "name": "...", "args": {...}, "result": {...}}`
- `{"role": "assistant", "content": "..."}`

Consecutive `tool_call` steps = parallel calls from one model turn.

**Tool result formatting:** all pipelines (SFT, eval, RL) go through
`format_tool_result()` in `core.py`. Environment methods return human-readable
strings; `format_tool_result` passes them through. The model always sees the
same format it was trained on. Don't bypass this.

**Evaluation:** Gemini-as-judge compares calendar state before/after against
expected behavior. `evaluate_trajectory(...)` → "Correct" / "Incorrect".

## Model & training

- **Base:** Qwen3-14B, 4-bit bnb, LoRA r=64 on q/k/v/o + gate/up/down, bf16,
  `/no_think` system prompt prepended at training time.
- **SFT:** `scripts/training/sft/sft_train.py`. Output under `runs/<run>/checkpoints/`.
  Merge with `scripts/training/common/merge_lora.py` (CPU peft merge — *not* unsloth).
- **RL (GRPO):** `scripts/training/rl/rl_train.py`. `gpu_memory_utilization=0.85`
  (0.90 OOMs during vLLM profile), `max_model_len=4096`, `enforce_eager=True`,
  `bf16=True`. Always launch via `scripts/training/common/auto_restart.sh`
  (handles Patch G deadlock retry) — never bare `python`.
- **Eval:** `eval_qwen.py` (single calendar), `eval_batch.py` (batch + Gemini
  judge), `eval_all_checkpoints.py` (orchestrator).

### vLLM serving
Tool-call parser must be `hermes` (NOT `qwen3_xml`).

```
vllm serve <model_path> --served-model-name <name> \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --max-model-len 4096 --gpu-memory-utilization 0.85 \
    --enforce-eager --quantization fp8
```

For RL adapters: serve via `--enable-lora` — do **NOT** merge an RL LoRA to
fp16 (produces 0% accuracy; see `feedback_vllm_lora_serving.md`).

## Tests

```bash
PYTHONPATH=src /home/abhor/miniconda3/envs/agentic/bin/pytest tests/
```

Files: `test_core.py`, `test_tools.py`, `test_paths.py`, `test_prompts.py`,
`test_evaluation_offline.py`, `test_evaluation_external.py`, `test_calendar_env.py`,
`test_data_integration.py`, `test_serialization.py`. Repro: `repro_art_deadlock.py`.

## Cross-session warnings (auto-memory has the details)

- **Never use Gemini Pro models** — burned 20× monthly budget. Use `gemini-2.0-flash-001`.
- **Never merge RL LoRA to fp16** — serve via `--enable-lora`.
- **Always use Slurm/sbatch for GPU work** — never bare processes.
- **Always read `PROGRESS.md` first** — it's the cross-session pipeline state.
