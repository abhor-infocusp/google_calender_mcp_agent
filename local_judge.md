# Local Qwen3-7B Judge — Train + Serve Plan (Phased)

## Context
RL training calls Gemini-2.0-flash ~99,200 times per run (12,400 steps × 8 rollouts) at `scripts/training/rl/rl_train.py:411-483` and `rl_train_adaptive.py:439-512` to score trajectories. This is the dominant per-run cost and a frequent timeout source. Goal: train a local Qwen3-7B judge from existing Gemini labels and serve it on one MIG slice (always-on) so RL on the other slices can call it over HTTP at near-zero marginal cost.

User decisions: **Qwen3-7B base**, **reasoning + verdict output** (preserves the existing `_extract_verdict` last-line scan), **RL-only swap** (eval_batch / run_agent / generate_trajectories keep using Gemini), **distill from existing Gemini-labeled eval JSONs**, **launch bare python (Slurm broken)**, **validate via ART GRPO trajectories** (reconstruct judge inputs from scenario, compare local-judge verdict to ART-saved verdict).

## Architecture (target end state)

```
┌─────────────────────────┐       ┌────────────────────────────┐
│ MIG slice 0 (persistent)│       │ MIG slice 1 (RL training)  │
│ vllm serve judge:fp8    │◀──────│ rl_train_adaptive.py       │
│ port 8001  /v1/...      │ HTTP  │   JUDGE_BACKEND=local      │
└─────────────────────────┘       └────────────────────────────┘
                                  ┌────────────────────────────┐
                                  │ MIG slice 2 (spare/eval)   │
                                  └────────────────────────────┘
```

Prompt-build and verdict-parse stay **client-side** in RL. The judge is a "no-tools" Qwen3-7B fine-tuned to emit Gemini-style reasoning ending with `Correct` / `Incorrect`.

---

## Phase 1 — Data prep + training (DO NOW, long-running)

**Goal**: produce a Qwen3-7B LoRA judge checkpoint that hits ≥ 95 % verdict agreement on a held-out set of ART GRPO trajectories. Everything in this phase is local file work + one bare-python training launch on a chosen MIG slice — no RL or serving touched.

### 1.1 Data prep — `scripts/training/judge/judge_data_prep.py`

Source: `runs/**/eval/checkpoint-*.json` — each result row has `query`, `expected`, `final_output`, `before` (already a formatted day-state string), `after` (string), `verdict`, `judge_reasoning`. 15 files, 3,774 pairs total (2,690 Correct / 1,084 Incorrect).

Pipeline:
1. Glob `runs/**/eval/checkpoint-*.json`; iterate `d["sft"]["results"]`, `d["rl"]["results"]`, `d["test"]["results"]`.
2. Dedupe key: SHA1 of `(query + "|" + final_output)`. Same trajectory across checkpoints → keep first.
3. Split **by query/scenario hash** 95/5 train/val (no test split — ART trajectories are the test set, see §1.4).
4. Class balance: duplicate Incorrect rows ×2 (≈ 5.4k examples, 2,690 Correct vs 2,168 Incorrect).
5. Emit JSONL: `{"messages":[{"role":"system","content":EVAL_SYSTEM_PROMPT},{"role":"user","content":<Q/R/E/B/A block built from row>},{"role":"assistant","content":judge_reasoning}]}` to `judge_data/{train,val}.jsonl`.
6. **Use Gemini's saved `judge_reasoning` verbatim** — recomputing defeats the distillation goal.
7. The user-prompt block must be **byte-identical** to the prompt template at `rl_train.py:425-441`. Rows store `before`/`after` as strings already — pass through, do NOT re-call `format_day_state_text`.

Reuses: `EVAL_SYSTEM_PROMPT` from `src/calendar_agent/evaluation.py:6-25`.

### 1.2 Training script — `scripts/training/judge/judge_sft_train.py`

Fork `scripts/training/sft/sft_train.py`. Reuse unchanged: tokenizer setup, `compute_assistant_labels` (line 105), `AssistantOnlyCollator`, `EpochLossLogger`, LoRA target-modules list, `SFTConfig` skeleton.

Changes:
- `MODEL_NAME = "Qwen/Qwen3-7B"`, `MAX_SEQ_LENGTH = 4096`.
- Drop `tools=` from `apply_chat_template` (judge does no tool-calling).
- Replace `load_trajectories` / `trajectory_to_messages` with a JSONL reader returning the `messages` list directly.
- LoRA r=64 / α=64 / dropout=0; targets `q,k,v,o,gate,up,down` (same as agent).
- `SFTConfig`: bf16, batch 1, accum 4, LR 2e-4 cosine, warmup 3 %, **3 epochs** (smaller dataset than agent SFT), save per-epoch.
- Output dir: `runs/judge_v1_qwen3_7b_20260425/{checkpoints,diagnostics,logs}/`.

### 1.3 Bare-python launcher — `scripts/training/judge/judge_train_launch.sh`

Slurm is broken; launch directly on a chosen MIG slice. Pick an idle slice from `scripts/eval/run_test_evals.sh` (the three pinned MIG UUIDs). Pattern:

```bash
#!/usr/bin/env bash
set -euo pipefail
RUN_DIR=runs/judge_v1_qwen3_7b_20260425
mkdir -p "$RUN_DIR/logs"
LOG="$RUN_DIR/logs/train_$(date +%Y%m%d_%H%M%S).log"

# Pick a free MIG slice (UUIDs from scripts/eval/run_test_evals.sh)
export CUDA_VISIBLE_DEVICES=MIG-abbb3894-4f8c-5e33-b602-6a485436950d  # or another idle slice
export PYTHONPATH=src

nohup /home/abhor/miniconda3/envs/agentic/bin/python \
    scripts/training/judge/judge_sft_train.py \
    > "$LOG" 2>&1 &
echo "PID $! → $LOG"
```

Notes:
- Run from the repo root.
- Use `nohup … &` so the job survives the shell. Tail `$LOG` to monitor.
- Confirm slice is free first with `nvidia-smi -L | grep MIG` and `ps -ef | grep python` so we don't collide with the running RL job.
- Expected wall time: ~3-6 h for 3 epochs over ~5.4k examples on Qwen3-7B + LoRA r=64 (smaller than the SFT-v6 14B run).

### 1.4 ART-trajectory validation — `scripts/eval/eval_judge_on_art.py`

**Replaces a Gemini-labeled holdout.** Use ART GRPO trajectories as the test set since user prefers it: same Gemini-labelled verdicts, vastly larger (~65k), already on disk.

ART parquet rows have:
- `messages` (full convo: system, user query, intermediate assistant + tool messages, final assistant)
- `metrics` JSON → `verdict` ∈ {0, 1} (the Gemini label at training time)
- `metadata` JSON → `scenario_id` like `cal_37_q_8`, `category`

To call the judge we still need `expected`, `before_days`, `after_days`. Reconstruction:

1. **`query`** = first `user` role message in `messages`.
2. **`final_output`** = last `assistant` role message content.
3. **`expected`** = look up by `scenario_id` in `rl_data/queries/` (the source query files; structure is `cal_<N>_q_<I>` → query record with `expected_behavior` field). Confirm exact file layout when implementing — `rl_data/{calender,queries,persona,json_calender}/` exist; the query-loading helpers from `rl_train.py` (look for `load_rl_scenarios` / similar) are the canonical path.
4. **`before_days`** = build a `CalendarEnvironment` from the scenario's initial state (same way RL rollout does), then call `snapshot_events` filtered to the relevant days (use `filter_by_days` from `core.py`).
5. **`after_days`** = replay the trajectory's tool calls through that same `CalendarEnvironment` (iterate `messages`, dispatch each `tool_calls` entry via `dispatch_tool_call` from `core.py`), then snapshot again.
6. Send `(query, final_output, expected, before_days, after_days)` through the trained judge (load LoRA into vLLM or transformers in-process), compare its verdict to `metrics["verdict"]`.

Sample size: 1-2k randomly sampled trajectories is enough for Phase-1 validation; full 65k can wait. Sample stratified by `category` and `verdict` so both classes and all 7 categories are well represented.

**Reuses**: `dispatch_tool_call`, `snapshot_events`, `filter_by_days` from `src/calendar_agent/core.py`; `CalendarEnvironment` from `src/calendar_agent/environment/`; scenario-loading helpers from `rl_train.py`. Read those before writing the script.

Output: `runs/judge_v1_qwen3_7b_20260425/eval/art_holdout.json` with per-trajectory predicted vs ground-truth verdict, overall agreement %, per-category agreement %, confusion matrix.

### 1.5 Phase 1 ship gates

- **Gate A**: ART-holdout verdict agreement **≥ 95 %**.
- **Gate B**: Per-category agreement **≥ 90 %** for every category (no silent skew on Modifier/Vague).

If either fails: investigate failure modes from `art_holdout.json` (which class? which category? prompt mismatch?), then either (a) fix the data-prep / training-prompt mismatch, or (b) augment training data by re-judging more ART trajectories with Gemini (Source B fallback). **Do not start Phase 2 until Phase 1 ship gates pass.**

### Files created in Phase 1
- `scripts/training/judge/judge_data_prep.py`
- `scripts/training/judge/judge_sft_train.py`
- `scripts/training/judge/judge_train_launch.sh`
- `scripts/eval/eval_judge_on_art.py`
- `judge_data/{train,val}.jsonl`
- `runs/judge_v1_qwen3_7b_20260425/...`

### Phase 1 verification

1. `PYTHONPATH=src python scripts/training/judge/judge_data_prep.py` → confirm `judge_data/train.jsonl` exists, ~5.4k rows, no scenario leak across train/val (assert in script).
2. `bash scripts/training/judge/judge_train_launch.sh` → tail log, confirm bf16 + LoRA r=64 logged; wait ~3-6 h for completion.
3. `PYTHONPATH=src python scripts/eval/eval_judge_on_art.py --checkpoint runs/judge_v1_.../checkpoints/checkpoint-final --num-samples 2000` → prints overall + per-category agreement; assert against gates.

---

## Phase 2 — Serving infrastructure (start AFTER Phase 1 gates pass)

**Goal**: stand up a persistent vLLM server on one MIG slice that exposes the judge over OpenAI-compatible HTTP.

### 2.1 Merge LoRA → fp16
Reuse `scripts/training/common/merge_lora.py` (or generate a one-off `_merge.py` mirroring `eval_all_checkpoints.py:142-183`). Output: `runs/judge_v1_qwen3_7b_20260425/checkpoints/checkpoint-final-merged/`.

### 2.2 Bare-python serving wrapper — `scripts/serving/judge_serve.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
MODEL=runs/judge_v1_qwen3_7b_20260425/checkpoints/checkpoint-final-merged
LOG=runs/judge_v1_qwen3_7b_20260425/logs/serve_$(date +%Y%m%d_%H%M%S).log
export CUDA_VISIBLE_DEVICES=MIG-<judge-slice-uuid>

nohup /home/abhor/miniconda3/envs/agentic/bin/vllm serve "$MODEL" \
    --served-model-name judge \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --quantization fp8 \
    --port 8001 \
    > "$LOG" 2>&1 &
echo "PID $! → $LOG"
```

No `--enable-auto-tool-choice`, no `--tool-call-parser`. Memory: 7B fp8 weights ≈ 7 GiB + KV @ 4096 ≈ 4 GiB → fits one 24 GiB MIG slice comfortably.

### 2.3 Auto-restart wrapper — `scripts/serving/judge_serve_loop.sh`

While-true around `judge_serve.sh`, log-rotate, exit code 42 → restart (mirrors `rl_train_loop.sh`). Health-check: poll `curl http://localhost:8001/v1/models` until 200 before declaring ready.

### Phase 2 ship gates
- `curl http://localhost:8001/v1/models` returns 200 with `id: "judge"`.
- Manual chat-completion request returns reasoning + `Correct` or `Incorrect` last line.
- Server stays up for 1 h continuous (no OOM, no Blackwell driver crash).

---

## Phase 3 — RL integration (after Phase 2 ship gates pass)

**Goal**: route RL judge calls to the local server behind an env-var toggle, with Gemini fallback on connection error.

### 3.1 New module — `src/calendar_agent/local_judge_client.py`

```python
_client = OpenAI(base_url=os.environ["JUDGE_API_URL"], api_key="dummy")

async def judge_local(query, final_output, expected, before_days, after_days):
    user_prompt = build_user_prompt(query, final_output, expected, before_days, after_days)
    for attempt in range(3):
        try:
            r = await asyncio.wait_for(asyncio.to_thread(
                _client.chat.completions.create,
                model="judge", temperature=0, max_tokens=512,
                messages=[{"role": "system", "content": EVAL_SYSTEM_PROMPT},
                          {"role": "user",   "content": user_prompt}]),
                timeout=30)
            return _extract_verdict(r.choices[0].message.content)
        except asyncio.TimeoutError:
            continue
        except Exception:
            return "Incorrect"
    return "Incorrect"
```

`build_user_prompt` and `_extract_verdict` lifted verbatim from `rl_train.py:425-475`. `EVAL_SYSTEM_PROMPT` re-imported from `src/calendar_agent/evaluation.py`.

### 3.2 Modify `scripts/training/rl/rl_train.py:411-483` and `rl_train_adaptive.py:439-512`

- At module import, `JUDGE_BACKEND = os.environ.get("JUDGE_BACKEND", "gemini")`.
- Keep `format_day_state_text` + prompt-build code untouched.
- In the body of `async def evaluate_trajectory(...)`: if `JUDGE_BACKEND=="local"`, `return await judge_local(...)`. Else fall through to existing Gemini code.
- On `local` `ConnectionError` / 5xx, fall back to existing Gemini path; reuse `gemini_*_count` counters (rename to `judge_*_count` in same edit if cheap).

**Do NOT modify** `src/calendar_agent/evaluation.py` (sync `evaluate_trajectory` keeps Gemini for eval/data-gen, per scope decision).

### Phase 3 ship gates
- 10-step RL smoke run with `JUDGE_BACKEND=local`: capture every `(query, final_output, expected, before, after)` and re-judge those same trajectories with Gemini in parallel. **≤ 5 % verdict disagreement**.
- No `[EVAL ERROR]` spam in logs; `judge_latency_s` p50 < 2 s (vs Gemini 3-5 s).

---

## Phase 4 — Production rollout

- Flip default `JUDGE_BACKEND=local` in the RL launcher (`scripts/training/rl/rl_train_adaptive_loop.sh`).
- Add ongoing 1 % Gemini spot-check inside RL (re-judge 1 % of rollouts, log disagreement %); abort RL if > 10 % over a 100-step window.
- Monitor judge server uptime; restart wrapper handles transient failures.

---

## MIG slice plan (target end state)

| Slice | Role |
|---|---|
| MIG 0 | Judge vLLM server (persistent, port 8001) — added in Phase 2 |
| MIG 1 | `rl_train_adaptive` |
| MIG 2 | Spare — eval / merging / dev (Phase 1 training runs here) |

Confirm fit in Phase 2 with: `vllm serve Qwen/Qwen3-7B --quantization fp8 --max-model-len 4096 --gpu-memory-utilization 0.90` on the target slice.

## Critical files (anchors used across phases)

- `src/calendar_agent/evaluation.py:6-25` — `EVAL_SYSTEM_PROMPT` (reused verbatim).
- `src/calendar_agent/evaluation.py:28-44` — `format_day_state_text`.
- `src/calendar_agent/core.py` — `dispatch_tool_call`, `snapshot_events`, `filter_by_days` (used by Phase-1 ART replay).
- `src/calendar_agent/environment/environment.py` — `CalendarEnvironment` (Phase-1 replay).
- `scripts/training/rl/rl_train.py:394-397` — `eval_model` singleton (parallel for `_client`).
- `scripts/training/rl/rl_train.py:411-483` — async judge body to swap (Phase 3); also the source of `build_user_prompt` / `_extract_verdict`.
- `scripts/training/rl/rl_train_adaptive.py:439-512` — same swap (Phase 3).
- `scripts/training/sft/sft_train.py:61-105` — `trajectory_to_messages` / `compute_assistant_labels` (template for Phase 1).
- `scripts/eval/eval_all_checkpoints.py:142-242` — vLLM merge + serve lifecycle (template for Phase 2).
- `runs/rl_adaptive_qwen3_14b_20260424/.art/calendar-agent/models/calendar-agent-001/trajectories/train/*.parquet` — ART validation source for Phase 1.

## Risks (cross-phase)

- **Slurm broken / processes collide** — confirm chosen MIG slice is idle before launching (`nvidia-smi`, `ps -ef | grep python`) so judge training does not OOM the active RL run on another slice.
- **Reward bias from judge drift** — Phase 4 spot-check + abort threshold.
- **Server crash stalls RL** — Phase 3 client falls back to Gemini on `ConnectionError`; Phase 2 `judge_serve_loop.sh` auto-restart.
- **3.7k pairs too small** — Phase 1 ship gate (ART-holdout) catches this; if it fails, mine more ART trajectories and re-judge with Gemini to grow training data, retrain.
