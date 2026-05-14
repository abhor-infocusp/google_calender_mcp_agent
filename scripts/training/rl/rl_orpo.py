"""ORPO trainer for the calendar agent.

Architecture: single-MIG-slice, in-process via ART 0.5.17's LocalBackend +
sleep/wake. Bypasses ART's GRPO `model.train()` and runs our own ORPO loss
on the same PEFT model that vLLM is serving from.

See `docs/orpo/design.md` for the full algorithm description, hyperparameter
choices, and rationale (including why we landed on ORPO over AR3PO/DOTS/
AWR/RFT/DPO). See `feedback_dpo_skipped` for the prior DPO failure modes
that this implementation explicitly guards against.

Per-step lifecycle:
  1. Wake vLLM (already awake on first step after register).
  2. Sample N=20 scenarios w/o replacement from difficulty tracker.
  3. Per-scenario adaptive rollout (k=4 easy, k=8 hard/cold), all concurrent.
  4. Score each via local Qwen3-14B-fp8 judge service.
  5. Update difficulty tracker; push correct rollouts into reuse buffer.
  6. Build ORPO pairs (no per-scenario cap; all-fail rescued via buffer).
  7. Sleep vLLM workers; reload training model to GPU.
  8. Run ORPO loss + optimizer step over pair minibatches.
  9. Save checkpoint; offload training model; wake vLLM workers; register new LoRA.
  10. Append per-step JSONL diagnostic; flush every 25 steps.
"""

import asyncio
import gc
import json
import logging
import os
import random
import subprocess
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict
from datetime import datetime

logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Allow py-spy / gdb attach without sudo (PR_SET_PTRACER_ANY).
import ctypes as _ctypes
try:
    _ctypes.CDLL("libc.so.6").prctl(0x59616d61, -1, 0, 0, 0)
except Exception:
    pass

import calendar_agent.art_patches  # noqa: F401 — must precede `import art`

import art
import torch
from art import dev
from art.local import LocalBackend
from art.utils import iterate_dataset
from openai import AsyncOpenAI
from pydantic import BaseModel

# ART internals we use to drive sleep/wake + checkpointing without going through
# `model.train()` (which is hard-wired to GRPO). All importable in 0.5.17;
# `calendar_agent.art_patches` is the place to add a shim if any of these move.
from art.unsloth.service import (
    do_sleep,
    do_wake_up,
    save_checkpoint as art_save_checkpoint,
    _get_trainer_optimizer,
)
from art.unsloth.train import gc_and_empty_cuda_cache
from art.vllm import run_on_workers
from art.local.checkpoints import delete_checkpoints as _backend_delete_checkpoints
from vllm.lora.request import LoRARequest

from calendar_agent.environment import CalendarEnvironment
from calendar_agent.core import (
    compute_fallback_now,
    dispatch_tool_call,
    filter_by_days,
    format_tool_result,
    snapshot_events,
)
from calendar_agent.evaluation import format_day_state_text
from calendar_agent.paths import RL_JSON_CALENDAR_DIR, RL_QUERY_DIR
from calendar_agent.tools import get_openai_tools
from calendar_agent.judge.client import verdict as _judge_verdict, JudgeUnavailable

from calendar_agent.orpo.difficulty_tracker import DifficultyTracker
from calendar_agent.orpo.reuse_buffer import ReuseBuffer
from calendar_agent.orpo.pair_builder import build_pairs_for_step
from calendar_agent.orpo.tokenize import tokenize_pair
from calendar_agent.orpo.orpo_loss import orpo_loss


random.seed(42)


# ── Configuration (env-overridable) ───────────────────────────────────

RUN_DIR = os.environ.get("RL_RUN_DIR", "runs/rl_orpo_qwen3_14b_default")
DEBUG_DIR = os.path.join(RUN_DIR, "logs", "debug")
os.makedirs(DEBUG_DIR, exist_ok=True)

# AR3PO-style hyperparameters (see docs/orpo/design.md)
N_QUERIES_PER_STEP = int(os.environ.get("ORPO_N_QUERIES", "20"))
K_HARD = int(os.environ.get("ORPO_K_HARD", "8"))
K_EASY = int(os.environ.get("ORPO_K_EASY", "4"))
EMA_ALPHA = float(os.environ.get("ORPO_EMA_ALPHA", "0.3"))
COLD_START_OBS = int(os.environ.get("ORPO_COLD_START_OBS", "8"))
HARD_THRESHOLD = float(os.environ.get("ORPO_HARD_THRESHOLD", "0.3"))
EASY_THRESHOLD = float(os.environ.get("ORPO_EASY_THRESHOLD", "0.7"))
BUFFER_PER_SCENARIO = int(os.environ.get("ORPO_BUFFER_PER_SCENARIO", "4"))

# ORPO loss hyperparameters
ORPO_BETA = float(os.environ.get("ORPO_BETA", "0.1"))
ORPO_LAMBDA = float(os.environ.get("ORPO_LAMBDA", "1.0"))

# Training knobs
LR = float(os.environ.get("ORPO_LR", "5e-6"))
PER_DEVICE_BATCH_SIZE = int(os.environ.get("ORPO_PER_DEVICE_BATCH", "4"))
GRADIENT_ACCUM_STEPS = int(os.environ.get("ORPO_GRAD_ACCUM", "4"))
MAX_GRAD_NORM = float(os.environ.get("ORPO_MAX_GRAD_NORM", "1.0"))
NUM_EPOCHS = int(os.environ.get("ORPO_NUM_EPOCHS", "20"))
MAX_STEPS = int(os.environ.get("ORPO_MAX_STEPS", "0"))  # 0 = no limit
MAX_TOKEN_LEN = int(os.environ.get("ORPO_MAX_TOKEN_LEN", "4096"))

# Checkpoint retention — keep every Nth + the latest K.
CHECKPOINT_KEEP_EVERY = int(os.environ.get("CHECKPOINT_KEEP_EVERY", "50"))
CHECKPOINT_KEEP_LATEST = int(os.environ.get("CHECKPOINT_KEEP_LATEST", "2"))
FLUSH_EVERY = int(os.environ.get("ORPO_FLUSH_EVERY", "25"))


# ── Telemetry plumbing ────────────────────────────────────────────────
from calendar_agent.run_telemetry import init_telemetry, set_phase  # noqa: E402

init_telemetry(run_dir=RUN_DIR, script_path=__file__)


def _write_run_metadata() -> None:
    import socket as _sock
    meta_path = os.path.join(RUN_DIR, "metadata.jsonl")

    def _sh(cmd, default=""):
        try:
            return subprocess.check_output(cmd, stderr=subprocess.DEVNULL,
                                            timeout=5).decode().strip()
        except Exception:
            return default

    def _pkg(name):
        try:
            from importlib.metadata import version
            return version(name)
        except Exception:
            return "?"

    entry = {
        "ts": datetime.now().isoformat(),
        "pid": os.getpid(),
        "host": _sock.gethostname(),
        "script": __file__,
        "git_commit": _sh(["git", "rev-parse", "HEAD"], "?"),
        "git_dirty": bool(_sh(["git", "status", "--porcelain"])),
        "run_dir": RUN_DIR,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "orpo_hparams": {
            "n_queries_per_step": N_QUERIES_PER_STEP,
            "k_hard": K_HARD, "k_easy": K_EASY,
            "ema_alpha": EMA_ALPHA, "cold_start_obs": COLD_START_OBS,
            "hard_threshold": HARD_THRESHOLD, "easy_threshold": EASY_THRESHOLD,
            "buffer_per_scenario": BUFFER_PER_SCENARIO,
            "beta": ORPO_BETA, "lambda": ORPO_LAMBDA,
            "lr": LR,
            "per_device_batch_size": PER_DEVICE_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUM_STEPS,
            "max_token_len": MAX_TOKEN_LEN,
        },
        "art_deadlock_timeout_s": os.environ.get("ART_DEADLOCK_TIMEOUT_S", "default"),
        "python_version": sys.version.split()[0],
        "packages": {
            pkg: _pkg(pkg)
            for pkg in ["openpipe-art", "unsloth", "trl", "transformers",
                        "vllm", "torch", "peft"]
        },
    }
    try:
        os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
        with open(meta_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[metadata] wrote run metadata → {meta_path}")
    except Exception as e:
        print(f"[metadata] failed: {e}")


_write_run_metadata()


# ── Hang watchdog + py-spy ────────────────────────────────────────────


def dump_pyspy(reason: str) -> str | None:
    pyspy = "/home/abhor/miniconda3/envs/agentic/bin/py-spy"
    if not os.path.exists(pyspy):
        return None
    out_path = os.path.join(
        DEBUG_DIR,
        f"pyspy_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{reason}.txt",
    )
    try:
        with open(out_path, "w") as f:
            f.write(
                f"=== py-spy dump reason={reason} pid={os.getpid()} "
                f"ts={datetime.now().isoformat()} ===\n\n"
            )
            subprocess.run(
                [pyspy, "dump", "--pid", str(os.getpid())],
                stdout=f, stderr=subprocess.STDOUT, timeout=20,
            )
        print(f"[PYSPY] wrote {out_path}")
        return out_path
    except Exception as e:
        print(f"[PYSPY ERROR] {e}")
        return None


async def run_with_hang_watchdog(coro, *, label, warn_after_s, kill_after_s):
    task = asyncio.create_task(coro)
    warned = False
    start = time.time()
    while not task.done():
        elapsed = time.time() - start
        remaining = warn_after_s - elapsed if not warned else kill_after_s - elapsed
        sleep_for = max(1.0, min(30.0, remaining))
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=sleep_for)
        except asyncio.TimeoutError:
            pass
        elapsed = time.time() - start
        if not warned and elapsed >= warn_after_s:
            warned = True
            print(f"[WATCHDOG] {label} >{warn_after_s}s — py-spy")
            dump_pyspy(f"warn_{label}")
        if elapsed >= kill_after_s:
            print(f"[WATCHDOG] {label} >{kill_after_s}s — KILL")
            dump_pyspy(f"kill_{label}")
            task.cancel()
            raise TimeoutError(f"{label} exceeded {kill_after_s}s")
    return task.result()


# ── Data ──────────────────────────────────────────────────────────────

JSON_CALENDAR_DIR = str(RL_JSON_CALENDAR_DIR)
QUERY_DIR = str(RL_QUERY_DIR)

MAX_TURNS = 8
ROLLOUT_SYSTEM_PROMPT = (
    "/no_think\nYou are a calendar assistant. Use the provided tools to "
    "manage events. Call get_current_time first to know the current date."
)


class CalendarScenario(BaseModel):
    id: str
    calendar_index: int
    query_index: int
    query: str
    expected_behavior: str
    category: str
    complexity: str
    addressed_days: list[str]
    current_time: str
    calendar_file_path: str


class ProjectTrajectory(art.Trajectory):
    final_answer_text: str | None = None


def load_all_scenarios() -> list[CalendarScenario]:
    scenarios = []
    for cal_index in range(50):
        cal_path = os.path.join(JSON_CALENDAR_DIR, f"{cal_index}.txt")
        query_path = os.path.join(QUERY_DIR, f"{cal_index}.txt")
        if not os.path.exists(cal_path) or not os.path.exists(query_path):
            continue
        fallback_now = compute_fallback_now(cal_path)
        with open(query_path) as f:
            queries = json.load(f)
        for q_index, q in enumerate(queries):
            current_time = q.get("current_time", "")
            if current_time:
                current_time = current_time.replace("T", " ")
            else:
                current_time = fallback_now
            scenarios.append(CalendarScenario(
                id=f"cal_{cal_index}_q_{q_index}",
                calendar_index=cal_index,
                query_index=q_index,
                query=q["query"],
                expected_behavior=q.get("expected_behavior", ""),
                category=q.get("category", "Unknown"),
                complexity=q.get("complexity", "Unknown"),
                addressed_days=q.get("addressed_days", []),
                current_time=current_time,
                calendar_file_path=os.path.abspath(cal_path),
            ))
    return scenarios


OPENAI_TOOLS = get_openai_tools()


# ── Judge ─────────────────────────────────────────────────────────────


judge_error_count = 0


async def evaluate_trajectory(query, final_output, expected, before_days,
                              after_days, *, category, scenario_id=None):
    global judge_error_count
    before_text = format_day_state_text(before_days)
    after_text = format_day_state_text(after_days)
    try:
        resp = await _judge_verdict(
            cat=category, query=query,
            final=final_output or "", expected=expected or "",
            before=before_text, after=after_text,
            scenario_id=scenario_id,
        )
    except JudgeUnavailable as e:
        judge_error_count += 1
        print(f"[JUDGE DOWN] {e} — exiting rc=43.")
        sys.stdout.flush()
        sys.exit(43)
    return resp["verdict"]


# ── Rollout (one trajectory) ──────────────────────────────────────────


async def rollout(model: art.Model, scenario: CalendarScenario,
                  step: int) -> ProjectTrajectory:
    """One on-policy rollout. Uses vLLM via OpenAI-compat API on the model's
    inference URL. Same as rl_train.py — only the surrounding training step
    differs."""
    env = CalendarEnvironment()
    events = CalendarEnvironment.load_json_calendar(scenario.calendar_file_path)
    env.initialize(events=events, now=scenario.current_time)
    before_snap = snapshot_events(env)

    traj = ProjectTrajectory(
        reward=0.0,
        messages_and_choices=[],
        metadata={
            "scenario_id": scenario.id, "step": step,
            "category": scenario.category, "complexity": scenario.complexity,
        },
    )
    traj.messages_and_choices = [
        {"role": "system", "content": ROLLOUT_SYSTEM_PROMPT},
        {"role": "user", "content": scenario.query},
    ]
    traj.tools = OPENAI_TOOLS

    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )

    final_answer_text = None
    num_turns = 0
    num_tool_calls = 0
    had_error = False
    tool_names_list = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for turn_idx in range(MAX_TURNS):
        num_turns += 1
        try:
            response = await client.chat.completions.create(
                model=model.get_inference_name(),
                temperature=1, messages=traj.messages(), tools=traj.tools,
            )
        except Exception as e:
            err_str = str(e)
            if "maximum context length" in err_str:
                traj.metrics["context_overflow"] = 1.0
            else:
                print(f"[ROLLOUT ERROR] {e}")
                traceback.print_exc()
            had_error = True
            break

        if response.usage:
            total_prompt_tokens += response.usage.prompt_tokens or 0
            total_completion_tokens += response.usage.completion_tokens or 0

        msg = response.choices[0].message
        traj.messages_and_choices.append(response.choices[0])

        if not msg.tool_calls:
            if msg.content:
                content = msg.content
                if "</think>" in content:
                    content = content.split("</think>")[-1].strip()
                final_answer_text = content
            break

        hit_final = False
        try:
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments)
                if tool_name == "return_final_answer":
                    final_answer_text = tool_args.get("answer", "")
                    traj.messages_and_choices.append({
                        "role": "tool", "tool_call_id": tc.id,
                        "name": tool_name, "content": final_answer_text,
                    })
                    hit_final = True
                    break
                num_tool_calls += 1
                tool_names_list.append(tool_name)
                result = dispatch_tool_call(env, tool_name, tool_args)
                result_str = format_tool_result(result)
                traj.messages_and_choices.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "name": tool_name, "content": result_str,
                })
        except Exception as e:
            print(f"[TOOL ERROR] {e}")
            had_error = True
            break

        if hit_final:
            break

    # ── Score ──
    reward = 0.0
    verdict = "NoAnswer"
    judge_latency_s = 0.0
    if final_answer_text is not None:
        after_snap = snapshot_events(env)
        before_days = filter_by_days(before_snap, scenario.addressed_days)
        after_days = filter_by_days(after_snap, scenario.addressed_days)
        judge_start = time.monotonic()
        verdict = await evaluate_trajectory(
            query=scenario.query, final_output=final_answer_text,
            expected=scenario.expected_behavior,
            before_days=before_days, after_days=after_days,
            category=scenario.category, scenario_id=scenario.id,
        )
        judge_latency_s = round(time.monotonic() - judge_start, 2)
        if verdict == "Correct":
            reward = 1.0

    traj.reward = reward
    traj.metrics["correct"] = 1.0 if verdict == "Correct" else 0.0
    traj.metrics["verdict"] = {"Correct": 1, "Incorrect": 0}.get(verdict, -1)
    traj.metrics["num_turns"] = float(num_turns)
    traj.metrics["num_tool_calls"] = float(num_tool_calls)
    traj.metrics["had_error"] = 1.0 if had_error else 0.0
    traj.metrics["no_final_answer"] = 1.0 if final_answer_text is None else 0.0
    traj.metadata["tool_names"] = ",".join(tool_names_list) if tool_names_list else ""
    traj.metrics["judge_latency_s"] = judge_latency_s
    traj.metrics["prompt_tokens"] = float(total_prompt_tokens)
    traj.metrics["completion_tokens"] = float(total_completion_tokens)
    traj.final_answer_text = final_answer_text
    return traj


# ── GPU snapshot helper ───────────────────────────────────────────────


def gpu_snapshot(label="") -> dict:
    if not torch.cuda.is_available():
        return {}
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    free, _ = torch.cuda.mem_get_info()
    snap = {
        "allocated_gb": round(allocated, 3),
        "reserved_gb": round(reserved, 3),
        "free_gb": round(free / 1024**3, 3),
    }
    if label:
        print(f"[GPU {label}] alloc={allocated:.2f} res={reserved:.2f} free={free/1024**3:.2f}")
    return snap


class StepTimer:
    def __init__(self):
        self.records: dict[str, float] = {}
        self._cur = None
        self._t0 = 0.0

    def start(self, phase):
        self.stop()
        self._cur = phase
        self._t0 = time.monotonic()

    def stop(self):
        if self._cur is not None:
            self.records[self._cur] = round(time.monotonic() - self._t0, 2)
            self._cur = None

    def get(self):
        self.stop()
        return dict(self.records)


# ── Checkpoint pruning ────────────────────────────────────────────────


def _prune_checkpoints(output_dir: str) -> None:
    """Keep every CHECKPOINT_KEEP_EVERY-th checkpoint plus the latest
    CHECKPOINT_KEEP_LATEST. Deletes everything else under
    `{output_dir}/checkpoints/`. No-op if either knob is ≤ 0 (treats as
    "keep all" for that axis)."""
    ckpt_base = os.path.join(output_dir, "checkpoints")
    if not os.path.isdir(ckpt_base):
        return
    steps = sorted(
        int(d) for d in os.listdir(ckpt_base)
        if d.isdigit() and os.path.isdir(os.path.join(ckpt_base, d))
    )
    if not steps:
        return
    keep: set[int] = set()
    if CHECKPOINT_KEEP_EVERY > 0:
        keep.update(s for s in steps if s % CHECKPOINT_KEEP_EVERY == 0)
    if CHECKPOINT_KEEP_LATEST > 0:
        keep.update(steps[-CHECKPOINT_KEEP_LATEST:])
    else:
        keep.update(steps)  # don't prune if KEEP_LATEST disabled
    _backend_delete_checkpoints(output_dir, sorted(keep))


# ── ORPO training step (mirrors ART's train_sft lifecycle) ────────────


async def orpo_train_step(
    service,
    pairs,                          # list[TokenizedPair]
    *,
    optimizer,
    lr: float,
    beta: float,
    lambda_or: float,
    per_device_batch_size: int,
    gradient_accum_steps: int,
    max_grad_norm: float,
    output_dir: str,
    verbose: bool = True,
) -> dict:
    """Run ORPO loss + optimizer step over `pairs`, then save checkpoint.

    Mirrors `art.unsloth.service.UnslothService.train_sft` lifecycle:
      pause vLLM → sleep workers → reload training to GPU → forward+loss+
      backward+optimizer steps → save → offload training to CPU →
      gc → wake workers → add new LoRA adapter → resume vLLM.

    Returns aggregate metrics for diagnostic logging.
    """
    if not pairs:
        # No pairs this step (all scenarios skipped) — skip the entire
        # train phase. Return zeros so the diagnostic log still has a record.
        return {
            "n_pairs": 0, "minibatches": 0, "optimizer_steps": 0,
            "loss_orpo_mean": 0.0, "loss_sft_mean": 0.0, "loss_or_mean": 0.0,
            "rewards_chosen_mean": 0.0, "rewards_rejected_mean": 0.0,
            "rewards_accuracy": 0.0, "rewards_margin": 0.0,
            "grad_norm_max": 0.0, "grad_norm_mean": 0.0,
            "logp_chosen_mean": 0.0, "logp_rejected_mean": 0.0,
        }

    llm = await service.llm

    # ── Pause + sleep vLLM ──
    await llm.pause_generation()
    has_unfinished = llm.output_processor.has_unfinished_requests()
    sleep_level = 1 if has_unfinished else 2
    if not has_unfinished:
        await llm.reset_prefix_cache()
    await run_on_workers(llm, do_sleep, level=sleep_level)
    service._is_sleeping = True
    gc_and_empty_cuda_cache()

    # ── Reload training model to GPU ──
    service._state.reload_to_gpu()
    peft_model = service._state.peft_model

    # Move any CPU optimizer state tensors to the param's device. Resuming
    # from disk loads state to CPU; without this, optimizer.step() crashes
    # with "tensors on different devices" once params reach GPU.
    if optimizer.state:
        for p, st in optimizer.state.items():
            tgt = p.device
            for k, v in st.items():
                if isinstance(v, torch.Tensor) and v.device != tgt:
                    st[k] = v.to(tgt)

    # Capture peak GPU state right after reload — this is the closest we
    # get to a "pre-forward" baseline, useful for diagnosing OOMs further
    # along in the train loop. The `before_train` snapshot in main() was
    # taken pre-sleep, which is misleading.
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1024**3
        f, _ = torch.cuda.mem_get_info()
        print(f"[ORPO TRAIN] post-reload-to-GPU: alloc={a:.2f} GiB free={f/1024**3:.2f} GiB")

    # Reset env var that GRPO mode sets — we need standard LM forward (with
    # logits, not hidden states) for ORPO loss.
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"

    # Update LR on optimizer (allows env-driven LR change between restarts)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    peft_model.train()
    device = next(peft_model.parameters()).device

    # ── Build minibatches ──
    # `pairs` is a list of TokenizedPair. We chunk into per_device_batch_size
    # microbatches, each processed in one forward pass. `gradient_accum_steps`
    # microbatches contribute to one optimizer step.
    n_pairs = len(pairs)
    if verbose:
        print(f"[ORPO TRAIN] {n_pairs} pairs, "
              f"per_device_batch={per_device_batch_size}, "
              f"grad_accum={gradient_accum_steps}, β={beta}, λ={lambda_or}")

    # Aggregates for diagnostic log
    losses_orpo: list[float] = []
    losses_sft: list[float] = []
    losses_or: list[float] = []
    rewards_chosen_all: list[float] = []
    rewards_rejected_all: list[float] = []
    rewards_acc_all: list[float] = []
    grad_norms: list[float] = []
    logp_chosen_all: list[float] = []
    logp_rejected_all: list[float] = []
    n_minibatches = 0
    n_optimizer_steps = 0

    optimizer.zero_grad()
    accum_count = 0

    for batch_start in range(0, n_pairs, per_device_batch_size):
        batch_pairs = pairs[batch_start:batch_start + per_device_batch_size]
        B = len(batch_pairs)
        n_minibatches += 1

        # Pad to common T per side, then stack
        T_c = max(p.chosen.input_ids.shape[0] for p in batch_pairs)
        T_r = max(p.rejected.input_ids.shape[0] for p in batch_pairs)

        def _stack(side: str, T: int):
            ids = torch.zeros(B, T, dtype=torch.long)
            attn = torch.zeros(B, T, dtype=torch.long)
            lbl = torch.full((B, T), -100, dtype=torch.long)
            for i, p in enumerate(batch_pairs):
                t = getattr(p, side)
                L = t.input_ids.shape[0]
                ids[i, :L] = t.input_ids
                attn[i, :L] = t.attention_mask
                lbl[i, :L] = t.labels
            return ids.to(device), attn.to(device), lbl.to(device)

        c_ids, c_attn, c_lbl = _stack("chosen", T_c)
        r_ids, r_attn, r_lbl = _stack("rejected", T_r)

        # Forward + loss
        loss_out = orpo_loss(
            peft_model,
            c_ids, c_attn, c_lbl,
            r_ids, r_attn, r_lbl,
            beta=beta, lambda_or=lambda_or,
        )

        # Scale loss for gradient accumulation (so the eventual optimizer
        # step sees the average loss across `gradient_accum_steps` microbatches).
        scaled_loss = loss_out.loss / gradient_accum_steps
        scaled_loss.backward()

        # Track metrics (use full-scale loss value, not the scaled one)
        losses_orpo.append(loss_out.loss.item())
        losses_sft.append(loss_out.sft_loss.item())
        losses_or.append(loss_out.or_loss.item())
        rewards_chosen_all.extend(loss_out.rewards_chosen.cpu().tolist())
        rewards_rejected_all.extend(loss_out.rewards_rejected.cpu().tolist())
        rewards_acc_all.append(loss_out.rewards_accuracy.item())
        logp_chosen_all.append(loss_out.logp_chosen_mean.item())
        logp_rejected_all.append(loss_out.logp_rejected_mean.item())

        accum_count += 1
        # Optimizer step at gradient_accum_steps boundary OR end of epoch
        if accum_count >= gradient_accum_steps or batch_start + per_device_batch_size >= n_pairs:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                peft_model.parameters(), max_grad_norm
            ).item()
            grad_norms.append(grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            accum_count = 0
            n_optimizer_steps += 1

            if verbose and (n_optimizer_steps == 1 or n_optimizer_steps % 4 == 0):
                print(
                    f"[ORPO TRAIN] step {n_optimizer_steps}: "
                    f"loss={loss_out.loss.item():.4f} "
                    f"sft={loss_out.sft_loss.item():.4f} "
                    f"or={loss_out.or_loss.item():.4f} "
                    f"rew_acc={loss_out.rewards_accuracy.item():.3f} "
                    f"margin={loss_out.rewards_margin.item():.3f} "
                    f"grad_norm={grad_norm:.3f}"
                )

    # ── Save checkpoint ──
    checkpoint_dir = art_save_checkpoint(
        trainer=service._state.trainer,
        output_dir=output_dir,
        verbose=verbose,
    )

    # ── Prune old checkpoints: keep every Nth + the latest K ──
    _prune_checkpoints(output_dir)

    # ── Offload + wake vLLM ──
    service._state.offload_to_cpu()
    gc_and_empty_cuda_cache()
    await asyncio.sleep(0.5)
    await run_on_workers(llm, do_wake_up)
    service._is_sleeping = False

    # Register new LoRA adapter
    new_step = int(os.path.basename(checkpoint_dir))
    added = await llm.add_lora(LoRARequest(
        lora_name=f"{service.model_name}@{new_step}",
        lora_int_id=service._next_lora_id(),
        lora_path=checkpoint_dir,
    ))
    if not added:
        raise RuntimeError(
            f"Failed to add LoRA adapter for step {new_step} at {checkpoint_dir}"
        )
    service._latest_step = new_step
    await llm.resume_generation()

    return {
        "n_pairs": n_pairs,
        "minibatches": n_minibatches,
        "optimizer_steps": n_optimizer_steps,
        "loss_orpo_mean": float(sum(losses_orpo) / len(losses_orpo)),
        "loss_sft_mean": float(sum(losses_sft) / len(losses_sft)),
        "loss_or_mean": float(sum(losses_or) / len(losses_or)),
        "rewards_chosen_mean": float(sum(rewards_chosen_all) / max(len(rewards_chosen_all), 1)),
        "rewards_rejected_mean": float(sum(rewards_rejected_all) / max(len(rewards_rejected_all), 1)),
        "rewards_accuracy": float(sum(rewards_acc_all) / len(rewards_acc_all)),
        "rewards_margin": float(
            (sum(rewards_chosen_all) - sum(rewards_rejected_all))
            / max(len(rewards_chosen_all), 1)
        ),
        "grad_norm_max": max(grad_norms) if grad_norms else 0.0,
        "grad_norm_mean": float(sum(grad_norms) / max(len(grad_norms), 1)),
        "logp_chosen_mean": float(sum(logp_chosen_all) / len(logp_chosen_all)),
        "logp_rejected_mean": float(sum(logp_rejected_all) / len(logp_rejected_all)),
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_step": new_step,
    }


# ── Main ──────────────────────────────────────────────────────────────


async def main():
    rl_base_model = os.environ.get("RL_BASE_MODEL", "Qwen/Qwen3-14B")
    rl_project = os.environ.get("RL_PROJECT", "calendar-agent-orpo")
    rl_model_name = os.environ.get("RL_MODEL_NAME", "calendar-agent-orpo-001")
    rl_vllm_port = int(os.environ.get("RL_VLLM_PORT", "8009"))

    print(
        f"[rl_orpo] base_model={rl_base_model} project={rl_project} "
        f"name={rl_model_name} port={rl_vllm_port} "
        f"N={N_QUERIES_PER_STEP} k_easy={K_EASY} k_hard={K_HARD} "
        f"β={ORPO_BETA} λ={ORPO_LAMBDA} lr={LR}"
    )

    model = art.TrainableModel(
        name=rl_model_name,
        project=rl_project,
        base_model=rl_base_model,
        _internal_config=dev.InternalModelConfig(
            init_args=dev.InitArgs(load_in_4bit=True, max_lora_rank=64),
            peft_args=dev.PeftArgs(
                r=8, lora_alpha=16,  # rank 8 to match prior RL benchmarks
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                lora_dropout=0, bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            ),
            engine_args=dev.EngineArgs(
                max_model_len=MAX_TOKEN_LEN,
                max_num_batched_tokens=MAX_TOKEN_LEN,
                max_num_seqs=16,
                gpu_memory_utilization=0.85,
                enforce_eager=True,
                enable_sleep_mode=True,  # critical for single-slice training
                quantization="bitsandbytes",
                load_format="bitsandbytes",
            ),
            trainer_args=dev.TrainerArgs(
                per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
                gradient_accumulation_steps=GRADIENT_ACCUM_STEPS,
                logging_steps=1,
                num_generations=4,  # not used by us, but ART requires it
                max_completion_length=512,
                max_grad_norm=MAX_GRAD_NORM,
                optim="adamw_torch",
                bf16=True, fp16=False,
            ),
        ),
    )

    backend = LocalBackend(
        in_process=True,
        path=os.path.join(RUN_DIR, ".art"),
    )
    from art.dev.openai_server import OpenAIServerConfig, ServerArgs
    await model.register(backend, _openai_client_config=OpenAIServerConfig(
        server_args=ServerArgs(port=rl_vllm_port),
    ))

    # Reach into ART to get the inner service for the ORPO training driver
    service = backend._services[rl_model_name]

    # ── Tokenizer (for pair tokenization) ──
    # Pull from peft_model — same model the trainer + vLLM are using.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(rl_base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Optimizer (custom AdamW; persisted across restarts) ──
    # ART's GRPOTrainer optimizer wraps things we don't need (GRPO-specific
    # paged adam etc.). Build our own on peft_model.parameters(); persist
    # state_dict() to a sidecar so auto_restart preserves Adam moments
    # rather than warming up cold for ~20 steps each time.
    peft_model = service._state.peft_model
    trainable_params = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR)
    optimizer_state_path = os.path.join(RUN_DIR, "orpo_optimizer.pt")
    if os.path.exists(optimizer_state_path):
        try:
            sd = torch.load(optimizer_state_path, map_location="cpu",
                             weights_only=False)
            optimizer.load_state_dict(sd)
            print(f"[rl_orpo] resumed optimizer from {optimizer_state_path}")
        except Exception as e:
            print(f"[rl_orpo] failed to load optimizer state ({e}); starting fresh")
    print(f"[rl_orpo] AdamW over {len(trainable_params)} trainable tensors "
          f"(persisted at {optimizer_state_path})")

    # ── Data ──
    all_scenarios = load_all_scenarios()
    CATEGORY_FILTER = os.environ.get("ORPO_CATEGORY_FILTER", "")
    if CATEGORY_FILTER:
        all_scenarios = [s for s in all_scenarios if CATEGORY_FILTER in s.category]
    random.shuffle(all_scenarios)
    print(f"Total scenarios: {len(all_scenarios)}")

    # ── State: tracker + buffer ──
    tracker = DifficultyTracker(
        alpha=EMA_ALPHA, cold_start_n=COLD_START_OBS,
        hard_threshold=HARD_THRESHOLD, easy_threshold=EASY_THRESHOLD,
    )
    tracker.register_many(all_scenarios)
    buffer = ReuseBuffer(per_scenario_cap=BUFFER_PER_SCENARIO)
    scenarios_by_id = {s.id: s for s in all_scenarios}

    tracker_path = os.path.join(RUN_DIR, "difficulty_tracker.json")
    if os.path.exists(tracker_path):
        try:
            tracker.load(tracker_path)
            print(f"[TRACKER] resumed from {tracker_path}: {len(tracker.stats)} scenarios")
        except Exception as e:
            print(f"[TRACKER] failed to load ({e}); starting fresh")

    buffer_path = os.path.join(RUN_DIR, "reuse_buffer.pkl")
    if os.path.exists(buffer_path):
        try:
            buffer.load(buffer_path)
            print(f"[BUFFER] resumed from {buffer_path}: "
                  f"{buffer.total_size()} traj over {buffer.scenarios_covered()} scenarios")
        except Exception as e:
            print(f"[BUFFER] failed to load ({e}); starting fresh")

    diag_jsonl_path = os.path.join(RUN_DIR, "orpo_diagnostic.jsonl")
    diagnostic_log: list[dict] = []

    # ── Step counter (mock the iterate_dataset stepping pattern) ──
    initial_step = await model.get_step()
    n_total_steps = (len(all_scenarios) * NUM_EPOCHS) // max(N_QUERIES_PER_STEP, 1)
    print(f"[rl_orpo] initial_step={initial_step} planned_steps={n_total_steps} "
          f"(epochs={NUM_EPOCHS}, scenarios/epoch={len(all_scenarios)}, "
          f"N/step={N_QUERIES_PER_STEP})")

    step = initial_step
    epoch = step // (len(all_scenarios) // max(N_QUERIES_PER_STEP, 1) or 1)
    rng = random.Random(42 + step)

    # ── Training loop ──
    while True:
        if MAX_STEPS > 0 and step >= MAX_STEPS:
            print(f"[rl_orpo] reached MAX_STEPS={MAX_STEPS}")
            break
        if step >= n_total_steps:
            print(f"[rl_orpo] reached planned step budget {n_total_steps}")
            break

        step_timer = StepTimer()
        torch.cuda.reset_peak_memory_stats()
        print(f"\n{'='*60}\nORPO step {step} (epoch {epoch})\n{'='*60}")

        gpu_before = gpu_snapshot("before_rollouts")

        # ── 1. Sample N scenarios w/o replacement ──
        set_phase("sample", step=step)
        step_timer.start("sample")
        sampled_ids = tracker.sample_without_replacement(
            N_QUERIES_PER_STEP, k_easy=K_EASY, k_hard=K_HARD, rng=rng,
        )
        sampled_scenarios = [scenarios_by_id[i] for i in sampled_ids]
        sampled_buckets = [tracker.bucket(i) for i in sampled_ids]
        sampled_ks = [tracker.k_for(i, k_easy=K_EASY, k_hard=K_HARD) for i in sampled_ids]
        sampled_weights = [
            tracker.weight(i, k) for i, k in zip(sampled_ids, sampled_ks)
        ]
        step_timer.stop()
        print(
            f"  sampled {len(sampled_ids)} scenarios; "
            f"bucket counts: {Counter(sampled_buckets)}; "
            f"k distribution: {Counter(sampled_ks)}"
        )

        # ── 2. Adaptive rollouts (concurrent across all scenarios * k each) ──
        set_phase("rollouts", step=step)
        step_timer.start("rollouts")
        rollout_coros = []
        rollout_owner: list[str] = []  # parallel list mapping each coro → scenario_id
        for sc, k in zip(sampled_scenarios, sampled_ks):
            for _ in range(k):
                rollout_coros.append(rollout(model, sc, step))
                rollout_owner.append(sc.id)
        rollouts_flat = await asyncio.gather(*rollout_coros, return_exceptions=False)
        step_timer.stop()

        # Group rollouts by scenario
        rollouts_by_scenario: dict[str, list] = {sid: [] for sid in sampled_ids}
        for sid, traj in zip(rollout_owner, rollouts_flat):
            rollouts_by_scenario[sid].append(traj)

        # ── 3. Tracker update + buffer push ──
        step_timer.start("tracker_update")
        for sid, trajs in rollouts_by_scenario.items():
            on_policy_correct = [
                t.metrics.get("correct", 0.0) == 1.0
                for t in trajs
            ]
            tracker.update(sid, on_policy_correct, step=step)
        added_to_buffer = buffer.add_correct(
            t for trajs in rollouts_by_scenario.values() for t in trajs
        )
        step_timer.stop()

        # ── 4. Build pairs ──
        step_timer.start("pair_build")
        per_scenario_pairs, flat_pairs = build_pairs_for_step(
            rollouts_by_scenario, reuse_buffer=buffer, rng=rng,
        )
        step_timer.stop()

        n_pairs_total = len(flat_pairs)
        skip_counter = Counter(
            sp.skip_reason for sp in per_scenario_pairs if sp.skip_reason
        )
        rescue_count = sum(1 for sp in per_scenario_pairs if sp.used_reuse_buffer)
        print(
            f"  rollouts={len(rollouts_flat)} pairs={n_pairs_total} "
            f"skips={dict(skip_counter)} rescued={rescue_count} "
            f"buf_size={buffer.total_size()} buf_scen={buffer.scenarios_covered()}"
        )

        # ── 5. Tokenize pairs ──
        step_timer.start("tokenize")
        tokenized_pairs = []
        for sp in per_scenario_pairs:
            for chosen, rejected in sp.pairs:
                meta = {
                    "scenario_id": sp.scenario_id,
                    "from_reuse_buffer": sp.used_reuse_buffer,
                    "category": chosen.metadata.get("category"),
                }
                tokenized_pairs.append(tokenize_pair(
                    chosen, rejected, tokenizer, OPENAI_TOOLS,
                    max_length=MAX_TOKEN_LEN, metadata=meta,
                ))
        step_timer.stop()
        print(f"  tokenized {len(tokenized_pairs)} pairs in {step_timer.records['tokenize']}s")

        # ── 6. ORPO training step ──
        gpu_before_train = gpu_snapshot("before_train")
        set_phase("orpo_train", step=step)
        step_timer.start("orpo_train")
        train_metrics = await run_with_hang_watchdog(
            orpo_train_step(
                service,
                tokenized_pairs,
                optimizer=optimizer,
                lr=LR,
                beta=ORPO_BETA, lambda_or=ORPO_LAMBDA,
                per_device_batch_size=PER_DEVICE_BATCH_SIZE,
                gradient_accum_steps=GRADIENT_ACCUM_STEPS,
                max_grad_norm=MAX_GRAD_NORM,
                output_dir=str(model._get_output_dir()),
                verbose=True,
            ),
            label=f"orpo_train_step_{step}",
            warn_after_s=300, kill_after_s=900,
        )
        step_timer.stop()
        gpu_after_train = gpu_snapshot("after_train")
        peak_alloc = round(torch.cuda.max_memory_allocated() / 1024**3, 3) if torch.cuda.is_available() else 0

        # ── 7. Diagnostic record ──
        # Per-scenario summary (skip the big trajectory objects — only stats)
        per_scenario_summary = []
        for sp in per_scenario_pairs:
            sid = sp.scenario_id
            per_scenario_summary.append({
                "scenario_id": sid,
                "category": scenarios_by_id[sid].category,
                "bucket": tracker.bucket(sid),
                "k_intended": tracker.k_for(sid, k_easy=K_EASY, k_hard=K_HARD),
                "k_actual": len(rollouts_by_scenario[sid]),
                "n_correct": sp.n_correct,
                "n_incorrect": sp.n_incorrect,
                "n_pairs": len(sp.pairs),
                "skip_reason": sp.skip_reason,
                "used_reuse_buffer": sp.used_reuse_buffer,
            })

        all_traj = [t for trajs in rollouts_by_scenario.values() for t in trajs]
        n_correct = sum(1 for t in all_traj if t.reward == 1.0)
        n_incorrect = len(all_traj) - n_correct

        record = {
            "step": step, "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "phase_timing_s": step_timer.get(),
            "gpu": {
                "before_rollouts": gpu_before, "before_train": gpu_before_train,
                "after_train": gpu_after_train, "peak_allocated_gb": peak_alloc,
            },
            "sampling": {
                "scenario_ids": sampled_ids,
                "weights": [round(w, 4) for w in sampled_weights],
                "k_per_scenario": sampled_ks,
                "bucket_counts_all": tracker.bucket_counts(),
                "buckets_sampled": dict(Counter(sampled_buckets)),
                "visit_stats": tracker.visit_stats(),
            },
            "rollouts": {
                "total": len(all_traj),
                "correct": n_correct, "incorrect": n_incorrect,
                "tokens_prompt": int(sum(t.metrics.get("prompt_tokens", 0) for t in all_traj)),
                "tokens_completion": int(sum(t.metrics.get("completion_tokens", 0) for t in all_traj)),
                "judge_latency_s_mean": round(sum(t.metrics.get("judge_latency_s", 0) for t in all_traj) / max(len(all_traj), 1), 2),
                "had_error": sum(1 for t in all_traj if t.metrics.get("had_error", 0) > 0),
                "no_final_answer": sum(1 for t in all_traj if t.metrics.get("no_final_answer", 0) > 0),
                "judge_errors_total": judge_error_count,
            },
            "pairs": {
                "total": n_pairs_total,
                "from_reuse_buffer": sum(1 for tp in tokenized_pairs if tp.metadata.get("from_reuse_buffer")),
                "per_scenario": per_scenario_summary,
                "skipped": dict(skip_counter),
            },
            "buffer": {
                "size_total": buffer.total_size(),
                "scenarios_covered": buffer.scenarios_covered(),
                "added_this_step": added_to_buffer,
                "rescue_attempts": sum(
                    1 for sp in per_scenario_pairs
                    if sp.n_correct == 0 and sp.n_incorrect > 0
                ),
                "rescue_hits": rescue_count,
            },
            "training": train_metrics,
        }
        diagnostic_log.append(record)

        # ── 8. Headline summary line ──
        print(
            f"\n  [STEP {step} SUMMARY] "
            f"acc={n_correct}/{len(all_traj)} ({n_correct/max(len(all_traj),1)*100:.1f}%) "
            f"pairs={n_pairs_total} "
            f"loss={train_metrics['loss_orpo_mean']:.4f} "
            f"sft={train_metrics['loss_sft_mean']:.4f} "
            f"or={train_metrics['loss_or_mean']:.4f} "
            f"rew_acc={train_metrics['rewards_accuracy']:.3f} "
            f"margin={train_metrics['rewards_margin']:.3f} "
            f"step_total={sum(step_timer.records.values()):.1f}s"
        )

        # ── 9. Periodic flush ──
        if step % FLUSH_EVERY == 0 and step > 0:
            try:
                with open(diag_jsonl_path, "a") as f:
                    for rec in diagnostic_log:
                        f.write(json.dumps(rec, default=str) + "\n")
                diagnostic_log.clear()
                tracker.save(tracker_path)
                # Atomic-ish optimizer-state save via tmp+replace.
                tmp = optimizer_state_path + ".tmp"
                torch.save(optimizer.state_dict(), tmp)
                os.replace(tmp, optimizer_state_path)
                buffer.save(buffer_path)
                print(f"  [FLUSH step={step}] diag tracker optimizer buffer → {RUN_DIR}")
            except Exception as e:
                print(f"  [FLUSH ERROR step={step}] {e}")

        step += 1
        epoch = step * N_QUERIES_PER_STEP // max(len(all_scenarios), 1)

        gc.collect()

    # Final flush
    try:
        with open(diag_jsonl_path, "a") as f:
            for rec in diagnostic_log:
                f.write(json.dumps(rec, default=str) + "\n")
        tracker.save(tracker_path)
        torch.save(optimizer.state_dict(), optimizer_state_path)
        buffer.save(buffer_path)
    except Exception as e:
        print(f"[FINAL FLUSH ERROR] {e}")
    print(f"\nDiagnostic log: {diag_jsonl_path}")
    print(f"Tracker: {tracker_path}")
    print("ORPO training complete.")


if __name__ == "__main__":
    asyncio.run(main())
    os._exit(0)
