# Google Calendar MCP Agent

## Project Overview
An agentic calendar assistant that uses tool-calling to manage Google Calendar events. Includes data generation, benchmarking, SFT training, and RL training pipelines.

## Environment
- Python env: `/home/abhor_gupta/miniconda3/envs/agentic/bin/python` (Python 3.11)
- Run scripts with: `PATH="/home/abhor_gupta/miniconda3/envs/agentic/bin:$PATH" /home/abhor_gupta/miniconda3/envs/agentic/bin/python <script>.py`
- GPU: Tesla T4 (14.57 GiB VRAM, compute 7.5, no bfloat16, no FlashAttention2, fp16 only)
- Set `VLLM_WORKER_MULTIPROC_METHOD=spawn` before importing vLLM (CUDA fork issue)

## Key Files

### Core
- `run_trajectory.py` — Core agent loop, tool declarations, evaluation logic, system prompts
  - Exports: `SYSTEM_PROMPT`, `EVAL_SYSTEM_PROMPT`, `TOOL_DECLARATIONS`, `CALENDAR_TOOL`, `dispatch_tool_call`, `snapshot_events`, `filter_by_days`, `evaluate_trajectory`, `get_query_now`, `DAY_NAMES`, `C`, `diff_snapshots`
- `environment/environment.py` — `CalendarEnvironment` class with CRUD methods, `load_json_calendar()`
- `environment/models.py` — Pydantic data models (`User`, `Attendee`, calendar event models)
- `prompt.py` — Prompt templates for data generation (persona, calendar, query prompts)

### Data Generation & Benchmarking
- `data_generation.py` — Generates calendar/persona/query data using Gemini
- `generate_sft_trajectories.py` — Runs queries through agent loop with Gemini 2.0 Flash, saves correct trajectories
- `benchmark_models.py` — Benchmarks Gemini models on high-complexity queries

### Training
- `sft_training.py` — SFT training for Qwen3-4B using Unsloth + TRL SFTTrainer
- `fine_tuning.py` — GRPO RL training with ART framework (Unsloth + vLLM)

### Evaluation & Utilities
- `eval_qwen.py` — Evaluate Qwen models via vLLM OpenAI-compatible API on calendar tasks
- `eval_checkpoints.py` — Automated per-checkpoint evaluation (merge LoRA → start vLLM → eval → kill)
- `merge_lora.py` — Merge LoRA adapter into fp16 model using Unsloth
- `test_judge.py` — Quick test for the eval judge prompt (uses Gemini via Vertex AI)
- `view_results.py` — ART training results viewer (summary, metrics table, trajectory analysis)
- `plot_rewards.py` — Plot training/validation reward curves from ART trajectory data
- `kill_gpu.sh` — Kill orphaned GPU processes (safety: only kills if exactly 1 process)
- `main.py` — Placeholder entry point

### Tests
- `tests/test_calendar_env.py` — Unit tests for CalendarEnvironment
- `tests/test_data_integration.py` — Integration tests for data pipeline

## Data Layout
- `data/` — Raw generated data
  - `calender/` — Natural language calendar descriptions
  - `json_calender/` — Structured JSON calendar data
  - `persona/` — Generated persona descriptions
  - `queries/` — Generated query sets with complexity/expected behavior
- `sft_data/` — SFT training & benchmarking data
  - `trajectories/` — 18 solved trajectories as JSON (from Gemini 2.5 Flash)
  - `json_calender/` (20 files), `queries/` (18 files), `calender/` (20 files)
  - `benchmark_results.json` — Gemini model benchmark results
- `rl_data/` — RL training data (calender/, json_calender/, persona/, queries/)
- `sft_output/` — SFT training outputs (gitignored)
  - `checkpoint-36/` through `checkpoint-180/` — Per-epoch LoRA checkpoints (5 epochs)
  - `final/` — Final epoch LoRA adapters
  - `merged_eval/` — Temporary fp16 merged model for evaluation
- `checkpoint_eval_results/` — Per-checkpoint eval results (gitignored)
- `.art/` — ART (RL) framework artifacts (gitignored)
  - `calendar-agent/models/calendar-agent-001/` — checkpoints, history, trajectories, logs

## Tools (7 calendar tools)
`get_current_time`, `list_events`, `get_event`, `create_event`, `update_event`, `delete_event`, `respond_to_event`

## Trajectory Format
Each trajectory has steps:
- `{"role": "user", "content": "..."}` — User query
- `{"role": "tool_call", "name": "...", "args": {...}, "result": {...}}` — Tool call + result
- `{"role": "assistant", "content": "..."}` — Final response
Consecutive `tool_call` steps = parallel calls from one model turn.

## Evaluation
Uses Gemini as judge model. Compares calendar state before/after against expected behavior.
`evaluate_trajectory(eval_model, query, final_output, expected, before_days, after_days)` -> "Correct"/"Incorrect"

## Model Training
- Base model: Qwen/Qwen3-4B, 4-bit quantization via Unsloth
- LoRA rank 32, targets: q/k/v/o/gate/up/down projections
- SFT: 5 epochs, lr=2e-4, cosine schedule, paged_adamw_8bit, save per-epoch
- Max sequence length: 4096 tokens
- Eval disabled during training (OOM on T4 due to fp32 conversion)
- 161 trajectories total, ~145 train / ~16 val after length filter and 90/10 split

## SFT Experiment Results (5-epoch)
- Accuracy flat at ~14% across all 5 epochs (4/28 correct on calendars 0+5)
- Training loss: 1.19 → 0.35 (epoch 1) → ~0.22 (epochs 3-5)
- Model learns tool-call format by epoch 1 but doesn't learn to reason with tool results
- Known issues: ignores `get_current_time` output, hallucinates dates, calls tools in parallel when sequential is needed
- Only Low-complexity "Schedule a Single Event" queries pass consistently

## Qwen Evaluation via vLLM
- **Tool-call parser**: Must use `hermes` (not `qwen3_xml`). Qwen3 produces `<tool_call>{"name":...}</tool_call>` JSON-in-XML format which hermes parses correctly.
- Requires a running vLLM server:
```
vllm serve <model_path> --served-model-name <name> --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 3072 --gpu-memory-utilization 0.85
```
- Run eval: `python eval_qwen.py <calendar_index> [--query-index N] [--model MODEL] [--sft-data] [--with-final-answer]`
- Automated checkpoint eval: `python eval_checkpoints.py` (merges LoRA, starts/stops vLLM per checkpoint)
