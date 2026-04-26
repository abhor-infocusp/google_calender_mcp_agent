import asyncio
import gc
import json
import logging
import os
import random
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime

logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Allow py-spy / gdb attach from the same user without sudo, even after
# re-parenting to init. PR_SET_PTRACER = 0x59616d61, PR_SET_PTRACER_ANY = -1.
import ctypes as _ctypes
try:
    _ctypes.CDLL("libc.so.6").prctl(0x59616d61, -1, 0, 0, 0)
except Exception:
    pass

import calendar_agent.art_patches  # noqa: F401 — must be before art imports

# No LoRA injection — starting fresh from SFT v5 baseline (74.6%)
# calendar_agent.art_patches.INJECT_LORA_CHECKPOINT = "rl_runs/single_category_modifier_correction/checkpoint"

import art
import torch
import vertexai
from art import dev
from art.local import LocalBackend
from art.utils import iterate_dataset
from openai import AsyncOpenAI
from pydantic import BaseModel
from vertexai.generative_models import GenerativeModel

from calendar_agent.environment import CalendarEnvironment
from calendar_agent.core import (
    compute_fallback_now,
    dispatch_tool_call,
    filter_by_days,
    format_tool_result,
    snapshot_events,
)
from calendar_agent.evaluation import EVAL_SYSTEM_PROMPT, format_day_state_text
from calendar_agent.paths import RL_DATA_DIR as _RL_DATA_DIR, RL_JSON_CALENDAR_DIR, RL_QUERY_DIR, CREDENTIALS_PATH
from calendar_agent.tools import get_openai_tools_minimal, get_openai_tools

random.seed(42)


# ── Debug / Hang Instrumentation ───────────────────────────
#
# Goal: when training hangs, we want (a) a timestamped record of the last
# phase the code was in, (b) a py-spy stack dump taken automatically at the
# moment we detect the stall, and (c) a hard timeout so the hang surfaces as
# an exception with context instead of indefinite silence.
#
# Everything writes to disk under ./logs/debug/ so we still have evidence
# even if the process is killed.

DEBUG_DIR = os.environ.get("RL_RUN_DIR", "runs/rl_qwen3_14b_20260420") + "/logs/debug"
# Keep every Nth ART checkpoint to allow rollback from interference / collapse.
# Plus the latest checkpoint and the best-by-reward checkpoint.
CHECKPOINT_KEEP_EVERY = int(os.environ.get("CHECKPOINT_KEEP_EVERY", "500"))


async def _smart_delete_checkpoints(model, current_step: int) -> None:
    """Like ART's model.delete_checkpoints, but also retains every Nth
    milestone checkpoint (CHECKPOINT_KEEP_EVERY).

    ART's stock `model.delete_checkpoints(best_checkpoint_metric=...)` keeps
    only `[latest, best]` and prunes everything else. So even if we skipped
    the call on milestone steps, the very next non-milestone step's call
    would wipe the milestone we just preserved (observed 2026-04-26).

    Bypass that by computing the keep-list ourselves and using ART's
    lower-level art.local.checkpoints.delete_checkpoints(output_dir, excluding)
    which accepts an explicit list of step numbers to retain.
    """
    from art.local.checkpoints import delete_checkpoints as _backend_delete

    output_dir = str(model._get_output_dir())
    keep: set[int] = {current_step}  # always keep latest

    # Also keep best-by-reward (from history.jsonl, same logic ART uses).
    try:
        import polars as pl
        best_step = (
            pl.read_ndjson(f"{output_dir}/history.jsonl")
            .drop_nulls(subset=["train/reward"])
            .group_by("step")
            .mean()
            .sort("train/reward")
            .select(pl.col("step").last())
            .item()
        )
        if best_step is not None:
            keep.add(int(best_step))
    except Exception:
        pass  # no history yet / no train/reward column — no big deal

    # Keep every Nth milestone we've ever passed.
    if CHECKPOINT_KEEP_EVERY > 0:
        for milestone in range(
            CHECKPOINT_KEEP_EVERY, current_step + 1, CHECKPOINT_KEEP_EVERY
        ):
            keep.add(milestone)

    _backend_delete(output_dir, list(keep))
os.makedirs(DEBUG_DIR, exist_ok=True)
HEARTBEAT_PATH = os.path.join(DEBUG_DIR, "heartbeat.jsonl")
PHASE_LOCK = threading.Lock()
_current_phase = {"phase": "startup", "step": None, "phase_start": time.time()}


def set_phase(phase: str, step: int | None = None) -> None:
    """Mark the current training phase. Called at key transition points so
    the heartbeat log and any hang-time dump tell us exactly where we were."""
    with PHASE_LOCK:
        _current_phase["phase"] = phase
        _current_phase["phase_start"] = time.time()
        if step is not None:
            _current_phase["step"] = step
    # Also print so it shows up in the main log inline
    print(f"[PHASE] {phase} step={_current_phase.get('step')} t={datetime.now().isoformat()}")


def _phase_snapshot() -> dict:
    """Phase-getter for art_patches.Patch G — lets the timeout handler
    distinguish healthy inter-call waits from real hangs."""
    with PHASE_LOCK:
        snap = dict(_current_phase)
    return {
        "phase": snap.get("phase", "?"),
        "phase_age_s": time.time() - snap.get("phase_start", time.time()),
        "step": snap.get("step"),
    }


# Register with art_patches if it's been imported (it is — rl_train.py
# imports calendar_agent.art_patches at module top via set_phase stack).
try:
    from calendar_agent import art_patches as _art_patches
    _art_patches.register_phase_getter(_phase_snapshot)
except Exception:
    pass


def _heartbeat_loop(interval: int = 30) -> None:
    """Append one JSONL record every `interval` seconds with the current
    phase and how long we've been in it. If training stalls, the last few
    heartbeats pinpoint where."""
    while True:
        try:
            with PHASE_LOCK:
                snap = dict(_current_phase)
            now = time.time()
            record = {
                "schema_version": 1,
                "ts": datetime.now().isoformat(),
                "phase": snap["phase"],
                "step": snap["step"],
                "phase_age_s": round(now - snap["phase_start"], 1),
                "pid": os.getpid(),
            }
            with open(HEARTBEAT_PATH, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            # Never let heartbeat kill training
            print(f"[HEARTBEAT ERROR] {e}")
        time.sleep(interval)


# Start heartbeat thread at import time (daemon = dies with process)
threading.Thread(target=_heartbeat_loop, args=(30,), daemon=True).start()


# ── Run metadata snapshot ──────────────────────────────────────────────
def _write_run_metadata() -> None:
    """Append one entry to runs/<run>/metadata.jsonl at every process start.
    Each entry captures what this run looked like at launch: git sha, env,
    deps, pid. Never overwrites earlier entries — the jsonl grows with
    each restart so we can trace the full history of a long experiment."""
    import json as _json
    import subprocess as _sub
    import socket as _sock

    RUN_DIR = os.environ.get("RL_RUN_DIR", "runs/rl_qwen3_14b_20260420")
    meta_path = os.path.join(RUN_DIR, "metadata.jsonl")

    def _sh(cmd: list[str], default: str = "") -> str:
        try:
            return _sub.check_output(cmd, stderr=_sub.DEVNULL, timeout=5).decode().strip()
        except Exception:
            return default

    def _pkg_version(name: str) -> str:
        try:
            from importlib.metadata import version
            return version(name)
        except Exception:
            return "?"

    # Snapshot of GPU compute apps at our launch — gives us the full sibling list
    # for post-mortem audit ("what else was running on this host when we started?").
    nv_apps = _sh([
        "nvidia-smi",
        "--query-compute-apps=pid,gpu_uuid,used_memory",
        "--format=csv,noheader",
    ])

    entry = {
        "schema_version": 2,  # v2: added isolation knobs + nvidia_smi snapshot
        "ts": datetime.now().isoformat(),
        "pid": os.getpid(),
        "host": _sock.gethostname(),
        "script": __file__,
        "git_commit": _sh(["git", "rev-parse", "HEAD"], "?"),
        "git_dirty": bool(_sh(["git", "status", "--porcelain"])),
        "run_dir": RUN_DIR,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        # Isolation knobs (set by auto_restart.sh / slice_map.sh)
        "taskset_cpus": os.environ.get("TASKSET_CPUS", ""),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", ""),
        # Patches
        "art_deadlock_timeout_s": os.environ.get("ART_DEADLOCK_TIMEOUT_S", "default"),
        "art_deadlock_hard_ceiling_s": os.environ.get("ART_DEADLOCK_HARD_CEILING_S", "default"),
        "art_use_threading_bridge": os.environ.get("ART_USE_THREADING_BRIDGE", "0"),
        "checkpoint_keep_every": os.environ.get("CHECKPOINT_KEEP_EVERY", str(CHECKPOINT_KEEP_EVERY)),
        # Sibling GPU processes seen at our launch — empty list if we're alone
        "nvidia_smi_compute_apps": [
            line.strip() for line in nv_apps.splitlines() if line.strip()
        ],
        "python_version": sys.version.split()[0],
        "packages": {
            pkg: _pkg_version(pkg)
            for pkg in ["openpipe-art", "unsloth", "trl", "transformers", "vllm", "torch", "peft"]
        },
    }
    try:
        os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
        with open(meta_path, "a") as f:
            f.write(_json.dumps(entry) + "\n")
        print(f"[metadata] wrote run metadata → {meta_path}")
    except Exception as e:
        print(f"[metadata] failed to write: {e}")


_write_run_metadata()


# ── Heartbeat-stuck alert thread ───────────────────────────────────────
# Watches _current_phase and logs a LOUD warning if the phase hasn't
# changed for too long. Cheap supplement to Patch G — catches "soft"
# stalls (e.g. gather hung on slow external API) that aren't reflected
# by the inputs_queue.
def _stuck_alert_loop(check_interval: int = 60, alert_after: int = 600) -> None:
    last_alerted_phase = None
    while True:
        try:
            with PHASE_LOCK:
                snap = dict(_current_phase)
            age = time.time() - snap.get("phase_start", time.time())
            phase = snap.get("phase", "?")
            if age >= alert_after:
                # alert once per stall-event (resets when phase changes)
                if (phase, snap.get("phase_start")) != last_alerted_phase:
                    print(
                        f"[STUCK-ALERT] phase={phase} has been running for "
                        f"{age:.0f}s (threshold={alert_after}s). step={snap.get('step')}",
                        flush=True,
                    )
                    last_alerted_phase = (phase, snap.get("phase_start"))
            else:
                last_alerted_phase = None
        except Exception:
            pass
        time.sleep(check_interval)


threading.Thread(target=_stuck_alert_loop, args=(60, 600), daemon=True).start()


def dump_pyspy(reason: str) -> str | None:
    """Shell out to py-spy to capture a stack dump of this process.
    Returns the path to the dump file, or None if py-spy unavailable."""
    pyspy = "/home/abhor/miniconda3/envs/agentic/bin/py-spy"
    if not os.path.exists(pyspy):
        print(f"[PYSPY] not found at {pyspy}")
        return None
    out_path = os.path.join(
        DEBUG_DIR,
        f"pyspy_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{reason}.txt",
    )
    try:
        with open(out_path, "w") as f:
            f.write(f"=== py-spy dump reason={reason} pid={os.getpid()} ts={datetime.now().isoformat()} ===\n\n")
            proc = subprocess.run(
                [pyspy, "dump", "--pid", str(os.getpid())],
                stdout=f, stderr=subprocess.STDOUT,
                timeout=20,
            )
        print(f"[PYSPY] wrote {out_path} (rc={proc.returncode})")
        return out_path
    except Exception as e:
        print(f"[PYSPY ERROR] {e}")
        return None


async def run_with_hang_watchdog(coro, *, label: str, warn_after_s: int, kill_after_s: int):
    """Run `coro`. If it takes longer than warn_after_s, trigger a py-spy
    dump to disk (the coroutine keeps running). If it takes longer than
    kill_after_s, take a second dump and raise TimeoutError.

    The dumps capture the stack at the exact moment we suspect a hang, so
    the next time this happens we can identify the exact deadlock site.
    """
    task = asyncio.create_task(coro)
    warned = False
    start = time.time()
    while not task.done():
        elapsed = time.time() - start
        remaining_to_warn = warn_after_s - elapsed if not warned else kill_after_s - elapsed
        sleep_for = max(1.0, min(30.0, remaining_to_warn))
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=sleep_for)
        except asyncio.TimeoutError:
            pass
        elapsed = time.time() - start
        if not warned and elapsed >= warn_after_s:
            warned = True
            print(f"[WATCHDOG] {label} still running after {elapsed:.0f}s — taking py-spy dump")
            dump_pyspy(f"warn_{label}")
        if elapsed >= kill_after_s:
            print(f"[WATCHDOG] {label} exceeded {kill_after_s}s — taking final py-spy dump and raising")
            dump_pyspy(f"kill_{label}")
            task.cancel()
            raise TimeoutError(f"{label} exceeded {kill_after_s}s")
    return task.result()


# ── Paths ──────────────────────────────────────────────────

JSON_CALENDAR_DIR = str(RL_JSON_CALENDAR_DIR)
QUERY_DIR = str(RL_QUERY_DIR)

# ── Configuration ──────────────────────────────────────────

MAX_TURNS = 8  # cap agent turns to bound trajectory length

# System prompt for rollouts — /no_think disables Qwen3 thinking mode
ROLLOUT_SYSTEM_PROMPT = "/no_think\nYou are a calendar assistant. Use the provided tools to manage events. Call get_current_time first to know the current date."


# ── Data Models ────────────────────────────────────────────


class CalendarScenario(BaseModel):
    """A flattened training scenario: one calendar + one query."""

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


class CalendarRolloutInput(BaseModel):
    step: int
    scenario: CalendarScenario


# ── Data Loading ───────────────────────────────────────────


def load_all_scenarios() -> list[CalendarScenario]:
    """Load all 50 calendar/query pairs and flatten into individual scenarios."""
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

            scenarios.append(
                CalendarScenario(
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
                )
            )

    return scenarios


OPENAI_TOOLS = get_openai_tools()

# ── Evaluation (uses Gemini judge via Vertex AI) ──────────

# Load Google credentials from file
CREDENTIALS_PATH = str(CREDENTIALS_PATH)

_gcp_credentials = None
if os.path.exists(CREDENTIALS_PATH):
    from google.oauth2.credentials import Credentials as OAuth2Credentials

    with open(CREDENTIALS_PATH) as _f:
        _cred_data = json.load(_f)
    _gcp_credentials = OAuth2Credentials(
        token=None,
        refresh_token=_cred_data["refresh_token"],
        client_id=_cred_data["client_id"],
        client_secret=_cred_data["client_secret"],
        token_uri="https://oauth2.googleapis.com/token",
    )

# Initialize Vertex AI for the Gemini evaluator
vertexai.init(
    project=os.environ.get("GCP_PROJECT", "internal-ml-exp"),
    location=os.environ.get("GCP_LOCATION", "us-central1"),
    credentials=_gcp_credentials,
)

eval_model = GenerativeModel(
    "gemini-2.0-flash-001",
    system_instruction=[EVAL_SYSTEM_PROMPT],
)


# Counters for observability — printed in rollout summary / step logs.
# Vertex AI's gRPC generate_content has no built-in timeout and will block
# indefinitely when Gemini stops responding, so we wrap each call in
# asyncio.wait_for(timeout=30) and retry up to 3 times.
GEMINI_TIMEOUT_SECS = 30
GEMINI_MAX_ATTEMPTS = 3
gemini_timeout_count = 0   # single 30s timeout (across all attempts)
gemini_giveup_count = 0    # all attempts timed out — fell back to Incorrect
gemini_error_count = 0     # non-timeout exceptions — fell back to Incorrect


async def evaluate_trajectory(
    query: str,
    final_output: str,
    expected: str,
    before_days: dict,
    after_days: dict,
) -> str:
    """Ask Gemini to evaluate whether the trajectory was correct.

    Returns one of: 'Correct', 'Incorrect'. Falls back to 'Incorrect' on
    exhausted timeouts or non-timeout errors.
    """
    global gemini_timeout_count, gemini_giveup_count, gemini_error_count

    before_text = format_day_state_text(before_days)
    after_text = format_day_state_text(after_days)

    prompt = f"""\
Query: {query}

Response: {final_output if final_output else '(no response)'}

Expected: {expected if expected else '(not specified)'}

Before:
{before_text}

After:
{after_text}

Was the task completed correctly? End with one word: Correct or Incorrect."""

    for attempt in range(1, GEMINI_MAX_ATTEMPTS + 1):
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(eval_model.generate_content, prompt),
                timeout=GEMINI_TIMEOUT_SECS,
            )
        except asyncio.TimeoutError:
            gemini_timeout_count += 1
            print(
                f"[EVAL TIMEOUT] attempt {attempt}/{GEMINI_MAX_ATTEMPTS} "
                f"({GEMINI_TIMEOUT_SECS}s) — total timeouts so far: {gemini_timeout_count}"
            )
            continue
        except Exception as e:
            gemini_error_count += 1
            print(f"[EVAL ERROR] {type(e).__name__}: {e} — total errors: {gemini_error_count}")
            return "Incorrect"

        verdict_text = response.text.strip()
        lines = [l.strip() for l in verdict_text.splitlines() if l.strip()]
        # Scan from last line for exact verdict
        for line in reversed(lines):
            line_lower = line.lower()
            for token in ("Incorrect", "Correct"):
                if line_lower == token.lower():
                    return token
        # Fallback: substring scan
        for line in reversed(lines):
            line_lower = line.lower()
            for token in ("Incorrect", "Correct"):
                if token.lower() in line_lower:
                    return token
        return "Incorrect"

    # All attempts timed out
    gemini_giveup_count += 1
    print(
        f"[EVAL TIMEOUT GIVEUP] all {GEMINI_MAX_ATTEMPTS} attempts timed out — "
        f"total give-ups: {gemini_giveup_count}"
    )
    return "Incorrect"


# ── Rollout ────────────────────────────────────────────────

async def rollout(
    model: art.Model, calendar_input: CalendarRolloutInput
) -> ProjectTrajectory:
    scenario = calendar_input.scenario

    # Fresh environment per rollout
    env = CalendarEnvironment()
    events = CalendarEnvironment.load_json_calendar(scenario.calendar_file_path)
    env.initialize(events=events, now=scenario.current_time)

    # Snapshot BEFORE state
    before_snap = snapshot_events(env)

    traj = ProjectTrajectory(
        reward=0.0,
        messages_and_choices=[],
        metadata={
            "scenario_id": scenario.id,
            "step": calendar_input.step,
            "category": scenario.category,
            "complexity": scenario.complexity,
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

    # ── Agent loop ──
    for _ in range(MAX_TURNS):
        num_turns += 1
        try:
            response = await client.chat.completions.create(
                model=model.get_inference_name(),
                temperature=1,
                messages=traj.messages(),
                tools=traj.tools,
            )
        except Exception as e:
            err_str = str(e)
            if "maximum context length" in err_str:
                print(f"  [ROLLOUT {scenario.id}] Context overflow at turn {_}, ending trajectory")
                traj.metrics["context_overflow"] = 1.0
            else:
                print(f"[ROLLOUT ERROR] LLM call failed: {e}")
                traceback.print_exc()
            had_error = True
            break

        # Track token usage
        if response.usage:
            total_prompt_tokens += response.usage.prompt_tokens or 0
            total_completion_tokens += response.usage.completion_tokens or 0

        response_message = response.choices[0].message
        traj.messages_and_choices.append(response.choices[0])

        # Debug: log what model generates
        tc_names = [tc.function.name for tc in (response_message.tool_calls or [])]
        content_preview = (response_message.content or "")[:100]
        n_tools = len(traj.tools) if traj.tools else 0
        n_msgs = len(traj.messages())
        model_name = model.get_inference_name()
        print(f"  [DEBUG {scenario.id}] turn={_} tool_calls={tc_names} content='{content_preview}' finish={response.choices[0].finish_reason} n_tools={n_tools} n_msgs={n_msgs} model={model_name}")

        if not response_message.tool_calls:
            if response_message.content:
                content = response_message.content
                # Strip <think> blocks from final answer
                if "</think>" in content:
                    content = content.split("</think>")[-1].strip()
                final_answer_text = content
            break

        hit_final_answer = False
        try:
            for tool_call in response_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                if tool_name == "return_final_answer":
                    final_answer_text = tool_args.get("answer", "")
                    traj.messages_and_choices.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": final_answer_text,
                        }
                    )
                    hit_final_answer = True
                    break

                num_tool_calls += 1
                tool_names_list.append(tool_name)

                # Dispatch to CalendarEnvironment via run_trajectory helper
                result = dispatch_tool_call(env, tool_name, tool_args)
                result_str = format_tool_result(result)
                traj.messages_and_choices.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": result_str,
                    }
                )
        except Exception as e:
            print(f"Error executing tool call: {e}")
            had_error = True
            break

        if hit_final_answer:
            break

    # ── Compute shaped reward ──
    # Collect tool names used during the trajectory
    tool_names_used = []
    for msg in traj.messages_and_choices:
        if isinstance(msg, dict) and msg.get("role") == "tool":
            name = msg.get("name", "")
            if name and name != "return_final_answer":
                tool_names_used.append(name)

    # Binary reward: 0.0 or 1.0 only. No shaped intermediates — forces the
    # model to optimise for correctness rather than gaming tool-use patterns.
    reward = 0.0
    verdict = "NoAnswer"

    judge_latency_s = 0.0
    if final_answer_text is not None:
        # Run Gemini judge for full correctness
        after_snap = snapshot_events(env)
        before_days = filter_by_days(before_snap, scenario.addressed_days)
        after_days = filter_by_days(after_snap, scenario.addressed_days)

        judge_start = time.monotonic()
        verdict = await evaluate_trajectory(
            query=scenario.query,
            final_output=final_answer_text,
            expected=scenario.expected_behavior,
            before_days=before_days,
            after_days=after_days,
        )
        judge_latency_s = round(time.monotonic() - judge_start, 2)

        if verdict == "Correct":
            reward = 1.0

    traj.reward = reward
    traj.metrics["correct"] = 1.0 if verdict == "Correct" else 0.0
    traj.metrics["verdict"] = {"Correct": 1, "Incorrect": 0}.get(verdict, -1)
    traj.metrics["shaped_reward"] = reward
    traj.metrics["num_turns"] = float(num_turns)
    traj.metrics["num_tool_calls"] = float(num_tool_calls)
    traj.metrics["had_error"] = 1.0 if had_error else 0.0
    traj.metrics["no_final_answer"] = 1.0 if final_answer_text is None else 0.0
    traj.metadata["tool_names"] = ",".join(tool_names_list) if tool_names_list else ""
    traj.metrics["judge_latency_s"] = judge_latency_s
    traj.metrics["prompt_tokens"] = float(total_prompt_tokens)
    traj.metrics["completion_tokens"] = float(total_completion_tokens)
    traj.final_answer_text = final_answer_text
    print(
        f"  [DEBUG {scenario.id}] verdict={verdict} reward={reward} "
        f"tools={tool_names_used} final_answer='{(final_answer_text or '')[:100]}'"
    )

    return traj


# ── Training Utilities ─────────────────────────────────────


def print_gpu_usage(label: str = ""):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(
            f"  [GPU {label}] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | Total: {total:.2f} GB"
        )


def gpu_snapshot(label: str = "") -> dict:
    """Return GPU memory stats as a dict and print them."""
    if not torch.cuda.is_available():
        return {}
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free, total_mem = torch.cuda.mem_get_info()
    free_gib = free / 1024**3
    snap = {
        "allocated_gb": round(allocated, 3),
        "reserved_gb": round(reserved, 3),
        "max_allocated_gb": round(max_allocated, 3),
        "total_gb": round(total, 3),
        "free_gb": round(free_gib, 3),
    }
    if label:
        print(f"  [GPU {label}] alloc={allocated:.2f} res={reserved:.2f} free={free_gib:.2f} max_alloc={max_allocated:.2f} total={total:.2f}")
    return snap


class StepTimer:
    """Simple wall-clock timer for named phases within a training step."""

    def __init__(self):
        self.records: dict[str, float] = {}
        self._current_phase: str | None = None
        self._phase_start: float = 0.0

    def start(self, phase: str):
        self.stop()
        self._current_phase = phase
        self._phase_start = time.monotonic()

    def stop(self):
        if self._current_phase is not None:
            elapsed = time.monotonic() - self._phase_start
            self.records[self._current_phase] = round(elapsed, 2)
            self._current_phase = None

    def get(self) -> dict[str, float]:
        self.stop()
        return dict(self.records)


# ── Main ───────────────────────────────────────────────────


async def main():
    # ── Model & Backend ──
    model = art.TrainableModel(
        name="calendar-agent-001",
        project="calendar-agent",
        base_model="Qwen/Qwen3-14B",
        _internal_config=dev.InternalModelConfig(
            init_args=dev.InitArgs(
                load_in_4bit=True,
                max_lora_rank=64,
            ),
            engine_args=dev.EngineArgs(
                max_model_len=4096,
                max_num_batched_tokens=4096,
                max_num_seqs=16,
                gpu_memory_utilization=0.85,
                enforce_eager=True,
                enable_sleep_mode=True,
                quantization="bitsandbytes",
                load_format="bitsandbytes",
            ),
            trainer_args=dev.TrainerArgs(
                per_device_train_batch_size=4,
                gradient_accumulation_steps=1,
                logging_steps=1,
                num_generations=4,
                max_completion_length=512,
                max_grad_norm=0.1,
                optim="adamw_torch",
                bf16=True,
                fp16=False,
            ),
        ),
    )

    # in_process=True runs vLLM in the main process so errors are visible
    backend = LocalBackend(in_process=True)
    from art.dev.openai_server import OpenAIServerConfig, ServerArgs
    await model.register(backend, _openai_client_config=OpenAIServerConfig(
        server_args=ServerArgs(port=8005),
    ))

    # ── Load Data ──
    all_scenarios = load_all_scenarios()

    # Filter to single category for focused training (empty = all categories)
    CATEGORY_FILTER = ""
    if CATEGORY_FILTER:
        all_scenarios = [s for s in all_scenarios if CATEGORY_FILTER in s.category]
        print(f"Filtered to '{CATEGORY_FILTER}': {len(all_scenarios)} scenarios")

    random.shuffle(all_scenarios)
    training_scenarios = all_scenarios

    print(f"Total scenarios: {len(all_scenarios)} (all used for training, no val split)")

    # ── Training Config ──
    training_config = {
        "groups_per_step": 1,
        "num_epochs": 20,
        "rollouts_per_group": 8,  # More rollouts = lower skip rate with binary rewards
        "learning_rate": 5e-6,
        "max_steps": 0,  # 0 = no limit (run all epochs)
    }

    training_iterator = iterate_dataset(
        training_scenarios,
        groups_per_step=training_config["groups_per_step"],
        num_epochs=training_config["num_epochs"],
        initial_step=await model.get_step(),
    )

    # ── Diagnostic Log ──
    diagnostic_log: list[dict] = []
    from collections import Counter

    # ── Training Loop ──
    for batch in training_iterator:
        step_timer = StepTimer()
        step_timer.start("step_total")

        torch.cuda.reset_peak_memory_stats()

        print(
            f"\n{'='*60}\n"
            f"Training step {batch.step}, epoch {batch.epoch}, epoch step {batch.epoch_step}\n"
            f"Batch contains {len(batch.items)} scenarios\n"
            f"{'='*60}"
        )

        gpu_before_rollouts = gpu_snapshot("before rollouts")

        # ── Rollouts ──
        set_phase("rollouts", step=batch.step)
        step_timer.start("rollout_generation_s")

        train_groups = []
        for scenario in batch.items:
            train_groups.append(
                art.TrajectoryGroup(
                    (
                        rollout(
                            model,
                            CalendarRolloutInput(step=batch.step, scenario=scenario),
                        )
                        for _ in range(training_config["rollouts_per_group"])
                    )
                )
            )

        finished_train_groups = await art.gather_trajectory_groups(
            train_groups,
            pbar_desc="gather",
            max_exceptions=training_config["rollouts_per_group"] * len(batch.items),
        )
        step_timer.stop()

        gpu_after_rollouts = gpu_snapshot("after rollouts")

        # ── Collect per-group and per-trajectory stats ──
        all_rewards = []
        group_details = []
        all_traj_metrics = []
        skip_count = 0

        for group in finished_train_groups:
            group_rewards = [t.reward for t in group.trajectories]
            group_verdicts = [
                t.metrics.get("verdict", -1) for t in group.trajectories
            ]
            scenario_id = group.trajectories[0].metadata.get("scenario_id", "?") if group.trajectories else "?"
            category = group.trajectories[0].metadata.get("category", "?") if group.trajectories else "?"

            # GRPO skip: all rewards identical → zero gradient
            skipped = len(set(group_rewards)) <= 1
            if skipped:
                skip_count += 1

            group_details.append({
                "scenario_id": scenario_id,
                "category": category,
                "rewards": group_rewards,
                "skipped": skipped,
                "verdicts": group_verdicts,
            })

            all_rewards.extend(group_rewards)
            for t in group.trajectories:
                all_traj_metrics.append(t.metrics)

        correct_count = sum(1 for r in all_rewards if r == 1.0)
        total_count = len(all_rewards)
        mean_reward = sum(all_rewards) / total_count if total_count else 0.0
        std_reward = (sum((r - mean_reward) ** 2 for r in all_rewards) / total_count) ** 0.5 if total_count else 0.0

        # Trajectory-level stats
        mean_turns = sum(m.get("num_turns", 0) for m in all_traj_metrics) / len(all_traj_metrics) if all_traj_metrics else 0
        no_answer_count = sum(1 for m in all_traj_metrics if m.get("no_final_answer", 0) > 0)
        tool_call_errors = sum(1 for m in all_traj_metrics if m.get("had_error", 0) > 0)
        mean_tool_calls = sum(m.get("num_tool_calls", 0) for m in all_traj_metrics) / len(all_traj_metrics) if all_traj_metrics else 0
        mean_judge_latency = sum(m.get("judge_latency_s", 0) for m in all_traj_metrics) / len(all_traj_metrics) if all_traj_metrics else 0
        total_prompt_tok = sum(m.get("prompt_tokens", 0) for m in all_traj_metrics)
        total_completion_tok = sum(m.get("completion_tokens", 0) for m in all_traj_metrics)

        # Tool name distribution (stored in metadata, not metrics)
        tool_counter: Counter = Counter()
        for group in finished_train_groups:
            for t in group.trajectories:
                names_str = t.metadata.get("tool_names", "")
                if isinstance(names_str, str) and names_str:
                    for n in names_str.split(","):
                        tool_counter[n.strip()] += 1

        # Compute inference tokens/sec from rollout time
        rollout_time = step_timer.records.get("rollout_generation_s", 1.0)
        inference_tps = round((total_prompt_tok + total_completion_tok) / rollout_time, 1) if rollout_time > 0 else 0

        print(f"\n  [STEP {batch.step} SUMMARY] reward={mean_reward:.3f}±{std_reward:.3f} "
              f"acc={correct_count}/{total_count} ({correct_count/total_count*100:.1f}%) "
              f"skip={skip_count}/{len(finished_train_groups)} ({skip_count/len(finished_train_groups)*100:.0f}%) "
              f"errors={tool_call_errors} no_answer={no_answer_count} "
              f"tokens={total_prompt_tok+total_completion_tok} tps={inference_tps} "
              f"judge_timeouts={gemini_timeout_count} judge_giveups={gemini_giveup_count} "
              f"judge_errors={gemini_error_count}")

        # ── Checkpoint delete + Train ──
        # Keep every Nth checkpoint as a rollback safety net (default every 500).
        # Between milestones, also pass best_checkpoint_metric so ART preserves
        # the highest-reward checkpoint we've seen, not just the most recent.
        set_phase("checkpoint_delete", step=batch.step)
        step_timer.start("checkpoint_delete_s")
        await _smart_delete_checkpoints(model, batch.step)
        step_timer.stop()

        set_phase("gc_empty_cache", step=batch.step)
        gpu_before_train = gpu_snapshot("before train (pre-gc)")
        gc.collect()
        torch.cuda.empty_cache()
        gpu_snapshot("before train (post-gc+empty_cache)")

        set_phase("model_train", step=batch.step)
        step_timer.start("train_step_s")
        # Watchdog: warn (py-spy dump) at 5 min, kill at 15 min.
        # Pre-hang step pace is ~50-80s; anything over 5 min is stuck.
        await run_with_hang_watchdog(
            model.train(
                finished_train_groups,
                config=art.TrainConfig(
                    learning_rate=training_config["learning_rate"],
                    beta=0.0,
                ),
                _config=dev.TrainConfig(logprob_calculation_chunk_size=128),
            ),
            label=f"train_step_{batch.step}",
            warn_after_s=300,
            kill_after_s=900,
        )
        step_timer.stop()
        set_phase("post_train", step=batch.step)

        gpu_after_train = gpu_snapshot("after train")
        peak_allocated = round(torch.cuda.max_memory_allocated() / 1024**3, 3) if torch.cuda.is_available() else 0

        # Stop total timer
        step_timer.start("_dummy")
        step_timer.stop()
        timing = step_timer.get()
        # Compute step_total from all phases
        timing["step_total_s"] = round(sum(v for k, v in timing.items() if k != "_dummy"), 2)
        timing.pop("_dummy", None)
        timing.pop("step_total", None)

        # ── Build step record ──
        step_record = {
            "step": batch.step,
            "epoch": batch.epoch,
            "timestamp": datetime.now().isoformat(),
            "timing": timing,
            "gpu": {
                "before_rollouts": gpu_before_rollouts,
                "after_rollouts": gpu_after_rollouts,
                "before_train": gpu_before_train,
                "after_train": gpu_after_train,
                "peak_allocated_gb": peak_allocated,
            },
            "rewards": {
                "mean": round(mean_reward, 4),
                "std": round(std_reward, 4),
                "correct_count": correct_count,
                "total_count": total_count,
                "accuracy": round(correct_count / total_count, 4) if total_count else 0,
                "skip_count": skip_count,
                "skip_rate": round(skip_count / len(finished_train_groups), 4) if finished_train_groups else 0,
            },
            "groups": group_details,
            "trajectory_stats": {
                "mean_turns": round(mean_turns, 2),
                "no_answer_count": no_answer_count,
                "tool_call_errors": tool_call_errors,
                "mean_tool_calls": round(mean_tool_calls, 2),
                "tool_name_distribution": dict(tool_counter),
                "mean_judge_latency_s": round(mean_judge_latency, 2),
                "total_prompt_tokens": int(total_prompt_tok),
                "total_completion_tokens": int(total_completion_tok),
                "inference_tokens_per_sec": inference_tps,
            },
        }
        diagnostic_log.append(step_record)

        print(f"\n  Completed training step {batch.step} in {timing.get('step_total_s', '?')}s")

        if training_config["max_steps"] > 0 and batch.step >= training_config["max_steps"]:
            break

    # ── Save diagnostic log ──
    diag_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "rl_diagnostic.json")
    with open(diag_path, "w") as f:
        json.dump(diagnostic_log, f, indent=2, default=str)
    print(f"\nDiagnostic log saved to {diag_path}")
    print("Training complete.")


if __name__ == "__main__":
    asyncio.run(main())
    # ART/vLLM daemon threads don't shut down cleanly — force exit to free GPU
    os._exit(0)
