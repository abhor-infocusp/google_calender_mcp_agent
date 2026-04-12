# Google Calendar MCP Agent

## Project Overview
An agentic calendar assistant that uses tool-calling to manage Google Calendar events. Includes data generation, benchmarking, SFT training, and RL training pipelines.

## Environment
- Python env: `/home/abhor/miniconda3/envs/agentic/bin/python` (Python 3.11)
- Run scripts with: `PYTHONPATH=src /home/abhor/miniconda3/envs/agentic/bin/python scripts/<path>.py`
- GPU: NVIDIA TITAN X Pascal (12 GiB VRAM, compute 6.1, no bfloat16, no FlashAttention2, fp16 only)
- Pinned stack: torch 2.5.1+cu124, vLLM 0.7.3 (rebuilt for sm_61), unsloth 2025.2.15, transformers 4.49.0, trl 0.15.2, openpipe-art 0.5.4
- Set `VLLM_WORKER_MULTIPROC_METHOD=spawn` before importing vLLM (CUDA fork issue)
- vLLM Punica patched to use PunicaWrapperCPU (Triton LLVM fails on Pascal)
- vLLM source build backup: `vllm-build/` (for rebuilding `_C.abi3.so` for Pascal sm_61)

## Project Structure

```
src/calendar_agent/           # Installable package (PYTHONPATH=src)
  __init__.py                 # Re-exports from core + evaluation + tools
  core.py                     # Tool declarations, dispatch, snapshots, system prompt, colors
  tools.py                    # OpenAI tool conversion, compact tool results, RETURN_FINAL_ANSWER_TOOL
  evaluation.py               # EVAL_SYSTEM_PROMPT, evaluate_trajectory, format_day_state_text
  paths.py                    # PROJECT_ROOT, DATA_DIR, SFT_DATA_DIR, RL_DATA_DIR, etc.
  prompts.py                  # Prompt templates for data generation
  environment/
    __init__.py               # Re-exports CalendarEnvironment, models
    environment.py            # CalendarEnvironment class with CRUD methods
    models.py                 # Pydantic data models (User, Attendee, Event, Calendar)

scripts/
  run_agent.py                # Core agent loop (Gemini + calendar tools)
  data_generation/
    generate_data.py           # Generate calendar/persona/query data using Gemini
    generate_trajectories.py   # Run queries through agent, save correct trajectories
  training/
    sft_train.py               # SFT training (Qwen2.5-1.5B, Unsloth + TRL)
    rl_train.py                # GRPO RL training (ART framework + vLLM)
    merge_lora.py              # Merge LoRA adapter into fp16 model
  eval/
    eval_qwen.py               # Evaluate Qwen via vLLM OpenAI-compatible API
    eval_batch.py              # Batch eval on SFT/RL data with Gemini judge
    eval_all_checkpoints.py    # Multi-checkpoint merge → serve → eval orchestrator
    benchmark_gemini.py        # Benchmark Gemini models on high-complexity queries
    test_judge.py              # Quick test for eval judge prompt
  utils/
    plot_rewards.py            # Plot training/validation reward curves
    view_results.py            # ART training results viewer

tests/                        # pytest tests
  test_calendar_env.py         # Unit tests for CalendarEnvironment
  test_data_integration.py     # Integration tests for data pipeline
  test_serialization.py        # SFT trajectory serialization tests

data/                         # Raw generation output (not tracked). Not needed if using sft_data/ or rl_data/.
sft_data/                     # SFT training data (tracked)
rl_data/                      # RL training data (tracked)
```

## Key Imports
```python
from calendar_agent.core import SYSTEM_PROMPT, TOOL_DECLARATIONS, CALENDAR_TOOL, dispatch_tool_call, format_tool_result, snapshot_events, filter_by_days, compute_fallback_now, DAY_NAMES, C
from calendar_agent.tools import get_openai_tools, RETURN_FINAL_ANSWER_TOOL
from calendar_agent.evaluation import EVAL_SYSTEM_PROMPT, evaluate_trajectory, format_day_state_text
from calendar_agent.environment import CalendarEnvironment
from calendar_agent.paths import PROJECT_ROOT, DATA_DIR, SFT_DATA_DIR, RL_DATA_DIR, CREDENTIALS_PATH
```

## Tools (7 calendar tools)
`get_current_time`, `list_events`, `get_event`, `create_event`, `update_event`, `delete_event`, `respond_to_event`

## Trajectory Format
Each trajectory has steps:
- `{"role": "user", "content": "..."}` — User query
- `{"role": "tool_call", "name": "...", "args": {...}, "result": {...}}` — Tool call + result
- `{"role": "assistant", "content": "..."}` — Final response
Consecutive `tool_call` steps = parallel calls from one model turn.

### Tool Result Formatting
All pipelines (SFT, eval, RL) use `format_tool_result()` from `core.py` to format tool results.
Environment methods return human-readable strings; `format_tool_result` passes them through.
This ensures the model always sees the same format it was trained on.

## Evaluation
Uses Gemini as judge model. Compares calendar state before/after against expected behavior.
`evaluate_trajectory(eval_model, query, final_output, expected, before_days, after_days)` -> "Correct"/"Incorrect"

## Model Training
- Base model: Qwen/Qwen2.5-1.5B-Instruct (Qwen3-1.7B incompatible with vLLM 0.7.3)
- LoRA rank 64, targets: q/k/v/o/gate/up/down projections
- SFT: `scripts/training/sft_train.py` (base config), `sft_train_100ep.py` (100-epoch with loss CSV)
- SFT output: written to `sft_output/` (gitignored). Merge with `merge_lora.py`.
- RL (ART/GRPO): `scripts/training/rl_train.py`. gpu_memory_utilization=0.55, max_model_len=2048, load_in_4bit=True, use_bitsandbytes=False
- RL critical: must override max_num_batched_tokens and max_num_seqs (Unsloth defaults cause OOM on 12 GiB)
- Eval: `scripts/eval/eval_qwen.py` (single calendar), `eval_batch.py` (batch eval with Gemini judge), `eval_all_checkpoints.py` (multi-checkpoint orchestrator)

## Qwen Evaluation via vLLM
- **Tool-call parser**: Must use `hermes` (not `qwen3_xml`).
- Requires a running vLLM server:
```
vllm serve <model_path> --served-model-name <name> --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 3076 --gpu-memory-utilization 0.80
```
- Run eval: `PYTHONPATH=src python scripts/eval/eval_qwen.py <calendar_index> [--query-index N] [--model MODEL] [--sft-data] [--with-final-answer]`
